"""The generic training loop, :class:`Trainer`.

A task supplies a ``train_step`` (and optionally an ``eval_step``) callable
turning ``(model, batch, device)`` into a :class:`StepOutput`; ``Trainer``
supplies the epoch/batch loop, device placement, ``backward()``, optimizer
and scheduler stepping, gradient clipping and accumulation, metric averaging
and logging, checkpointing, early stopping and resuming. Batches are moved
to the device generically: anything with a ``.to(device)`` method is moved,
recursing through dicts, lists and tuples. Single device, full precision.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor, nn

from gm_attention.training.checkpointing import CheckpointManager, load_checkpoint
from gm_attention.training.logging import Logger


@dataclass
class StepOutput:
    """What a ``train_step``/``eval_step`` callable returns.

    Parameters
    ----------
    loss
        Scalar tensor; ``Trainer`` calls ``.backward()`` on this directly
        during training, and never during evaluation.
    metrics
        Scalar metrics to log and average over the epoch. If it has no
        ``"loss"`` key, ``Trainer`` fills one in from ``loss``.
    """

    loss: Tensor
    metrics: dict[str, float] = field(default_factory=dict)


StepFn = Callable[[nn.Module, Any, torch.device], StepOutput]

# ReduceLROnPlateau does not share LRScheduler's step() signature (its step
# takes the monitored metric); Trainer supports both.
Scheduler = torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau


def _move_to_device(value: Any, device: torch.device) -> Any:
    """Recursively move a batch to ``device``.

    Anything exposing a ``.to(device)`` method is moved; dicts, lists and
    tuples are recursed into; anything else is passed through.
    """
    if hasattr(value, "to") and callable(value.to):
        return value.to(device)

    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}

    if isinstance(value, tuple) and hasattr(value, "_fields"):
        # A namedtuple's constructor takes the fields as separate positional
        # arguments, unlike a plain tuple's single iterable.
        return type(value)(*(_move_to_device(item, device) for item in value))

    if isinstance(value, (list, tuple)):
        return type(value)(_move_to_device(item, device) for item in value)

    return value


def _total_gradient_norm(model: nn.Module) -> Tensor:
    """The 2-norm of every gradient concatenated, as ``clip_grad_norm_`` defines it."""
    norms = [
        torch.linalg.vector_norm(parameter.grad.detach())
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not norms:
        return torch.zeros(())
    return torch.linalg.vector_norm(torch.stack(norms))


class _RunningAverages:
    """Unweighted mean of every metric logged across an epoch's steps."""

    def __init__(self) -> None:
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def update(self, metrics: dict[str, float]) -> None:
        for name, value in metrics.items():
            self._sums[name] = self._sums.get(name, 0.0) + float(value)
            self._counts[name] = self._counts.get(name, 0) + 1

    def averages(self) -> dict[str, float]:
        return {name: total / self._counts[name] for name, total in self._sums.items()}


class Trainer:
    """Drives a task-agnostic train/validate/checkpoint loop.

    Parameters
    ----------
    model
        Moved to ``device`` once, here.
    optimizer
        Stepped once per training batch.
    train_step
        ``(model, batch, device) -> StepOutput``. Must not call
        ``backward()``, ``optimizer.step()``, or ``optimizer.zero_grad()``
        itself -- ``Trainer`` owns all three.
    eval_step
        Same signature, used under ``torch.no_grad()`` during validation.
        If ``None``, validation is skipped even if a ``val_loader`` is given
        to :meth:`fit`.
    scheduler
        Stepped once per ``scheduler_step_on`` unit, if given.
    scheduler_step_on
        Whether ``scheduler.step()`` is called once per epoch or once per
        training batch.
    scheduler_metric
        If given, ``scheduler.step(value)`` is called with that metric's
        value (from ``val_metrics`` if a validation pass ran this epoch,
        else ``train_metrics``) instead of ``scheduler.step()``, as a
        metric-driven scheduler such as ``ReduceLROnPlateau`` requires.
        Requires ``scheduler_step_on="epoch"``.
    accumulation_steps
        Number of consecutive training batches to accumulate gradients over
        before one ``optimizer.step()``. Each micro-batch's loss is scaled
        by ``1 / (batches accumulated in this step)`` before its
        ``backward()``, so the accumulated gradient equals the average a
        batch of that size would produce. ``global_step`` and
        ``scheduler_step_on="step"`` advance once per accumulated group.
        Does not apply during validation. Defaults to ``1``.
    device
        Where the model and every batch are moved to.
    grad_clip_norm
        If given, ``torch.nn.utils.clip_grad_norm_`` is applied after
        ``backward()`` and before the optimizer step.
    logger
        Receives averaged metrics once per epoch, per split. ``None``
        disables logging.
    checkpoint_manager
        Driven once per epoch, after validation (or after training, if
        there is no validation this epoch). ``None`` disables checkpointing.
    early_stopping_patience
        Stop once this many consecutive epochs pass without
        ``checkpoint_manager``'s monitored metric improving. Requires a
        ``checkpoint_manager``.
    validate_every
        Run validation only every this many epochs; epoch 0 always
        validates when a ``val_loader`` and ``eval_step`` are given. On
        epochs without validation the scheduler, the ``best.pt`` decision
        and early stopping are skipped rather than fed training metrics,
        so the monitored value always comes from one distribution;
        ``last.pt`` is still written every epoch.
    on_epoch_start
        Called with the epoch number at the start of each epoch, before
        training; the hook an epoch-indexed dataset needs.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_step: StepFn,
        *,
        eval_step: StepFn | None = None,
        scheduler: Scheduler | None = None,
        scheduler_step_on: Literal["epoch", "step"] = "epoch",
        scheduler_metric: str | None = None,
        accumulation_steps: int = 1,
        device: torch.device | str = "cpu",
        grad_clip_norm: float | None = None,
        logger: Logger | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        early_stopping_patience: int | None = None,
        validate_every: int = 1,
        on_epoch_start: Callable[[int], None] | None = None,
    ) -> None:
        if grad_clip_norm is not None and not (grad_clip_norm > 0.0):
            raise ValueError(f"grad_clip_norm must be positive if given, got {grad_clip_norm}.")

        if scheduler_metric is not None and scheduler_step_on != "epoch":
            raise ValueError(
                "scheduler_metric requires scheduler_step_on='epoch', got "
                f"{scheduler_step_on!r}: a metric-driven scheduler needs a full "
                "epoch's averaged value, not a single training step's."
            )

        if (
            isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
            and scheduler_metric is None
        ):
            raise ValueError(
                "scheduler is a ReduceLROnPlateau, which requires scheduler_metric "
                "to be set: its step() takes the monitored value as a required "
                "argument, unlike an ordinary scheduler's no-argument step()."
            )

        if accumulation_steps < 1:
            raise ValueError(f"accumulation_steps must be at least 1, got {accumulation_steps}.")

        if validate_every < 1:
            raise ValueError(f"validate_every must be at least 1, got {validate_every}.")

        if early_stopping_patience is not None:
            if early_stopping_patience < 1:
                raise ValueError(
                    "early_stopping_patience must be at least 1 if given, got "
                    f"{early_stopping_patience}."
                )
            if checkpoint_manager is None:
                raise ValueError(
                    "early_stopping_patience requires a checkpoint_manager: that is "
                    "what defines whether a given epoch 'improved'."
                )

        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.train_step = train_step
        self.eval_step = eval_step
        self.scheduler = scheduler
        self.scheduler_step_on = scheduler_step_on
        self.scheduler_metric = scheduler_metric
        self.accumulation_steps = accumulation_steps
        self.grad_clip_norm = grad_clip_norm
        # Reset at the start of every training epoch; see _run_epoch.
        self._epoch_grad_norms: list[float] = []
        self.logger = logger
        self.checkpoint_manager = checkpoint_manager
        self.early_stopping_patience = early_stopping_patience
        self.validate_every = validate_every
        self.on_epoch_start = on_epoch_start

        self.epoch = 0
        self.global_step = 0
        # Restored on resume, not reset.
        self.epochs_without_improvement = 0

    def fit(
        self,
        train_loader: Iterable[Any],
        *,
        max_epochs: int,
        val_loader: Iterable[Any] | None = None,
        resume_from: str | Path | None = None,
        restore_scheduler: bool = True,
    ) -> None:
        """Run the train/validate/checkpoint loop through ``max_epochs``.

        Parameters
        ----------
        train_loader
            Iterable of batches; iterated once per epoch.
        max_epochs
            Epochs are 0-indexed internally; the loop runs while
            ``epoch < max_epochs``.
        val_loader
            Iterable of batches for validation. Ignored if ``eval_step`` was
            not given at construction.
        resume_from
            Path to a checkpoint to restore model, optimizer, scheduler and
            RNG state and ``epoch``/``global_step`` from before the loop
            starts, continuing at ``epoch + 1``.
        restore_scheduler
            Whether the checkpoint's scheduler state is restored too.
            ``False`` is a warm restart: the scheduler given at construction
            starts fresh, the optimizer's state (Adam's moments) is restored
            from the checkpoint, and each param group's learning rate is set
            to the fresh scheduler's ``base_lrs`` (any ``LRScheduler``;
            ``ReduceLROnPlateau`` has none).
        """
        if max_epochs < 1:
            raise ValueError(f"max_epochs must be at least 1, got {max_epochs}.")

        if resume_from is not None:
            metadata = load_checkpoint(
                resume_from,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler if restore_scheduler else None,
                map_location=self.device,
            )
            if not restore_scheduler and self.scheduler is not None:
                base_lrs = getattr(self.scheduler, "base_lrs", None)
                if base_lrs is None:
                    raise ValueError(
                        "restore_scheduler=False needs a scheduler with base_lrs (an "
                        "LRScheduler, not ReduceLROnPlateau): the fresh cycle's "
                        "starting rates come from there."
                    )
                # optimizer.load_state_dict restored the checkpoint's final lr;
                # a recursive schedule would carry the decayed rate forward.
                for group, lr in zip(self.optimizer.param_groups, base_lrs, strict=True):
                    group["lr"] = lr
                    group["initial_lr"] = lr
            self.epoch = metadata.epoch + 1
            self.global_step = metadata.global_step
            self.epochs_without_improvement = metadata.epochs_without_improvement
            if self.checkpoint_manager is not None:
                # Otherwise the first resumed epoch beats a best-so-far of None.
                self.checkpoint_manager.restore_best_value(metadata)

        validating = val_loader is not None and self.eval_step is not None

        for epoch in range(self.epoch, max_epochs):
            self.epoch = epoch
            if self.on_epoch_start is not None:
                self.on_epoch_start(epoch)

            train_metrics = self._run_epoch(train_loader, training=True)
            if self.logger is not None:
                self.logger.log(train_metrics, step=self.global_step, epoch=epoch, split="train")

            val_metrics: dict[str, float] | None = None
            if validating and epoch % self.validate_every == 0:
                assert val_loader is not None  # implied by `validating`
                val_metrics = self._run_epoch(val_loader, training=False)
                if self.logger is not None:
                    self.logger.log(val_metrics, step=self.global_step, epoch=epoch, split="val")

            # A non-validation epoch of a validating run has no monitored value:
            # scheduler, best-model decision and early stopping do not advance.
            monitored = val_metrics if validating else train_metrics

            if self.scheduler is not None and self.scheduler_step_on == "epoch":
                if self.scheduler_metric is None:
                    self.scheduler.step()
                elif monitored is not None:
                    if self.scheduler_metric not in monitored:
                        raise KeyError(
                            f"scheduler_metric {self.scheduler_metric!r} not found in "
                            f"metrics {sorted(monitored)}."
                        )
                    # scheduler_metric pairs with a ReduceLROnPlateau-shaped
                    # step(value); a plain LRScheduler.step accepts int | None.
                    self.scheduler.step(monitored[self.scheduler_metric])  # type: ignore[arg-type]

            if self.checkpoint_manager is not None:
                is_best = self.checkpoint_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    global_step=self.global_step,
                    metrics=monitored if monitored is not None else train_metrics,
                    # Only a monitored epoch may claim `best`; an unmonitored
                    # one still refreshes last.pt.
                    update_best=monitored is not None,
                    epochs_without_improvement=self.epochs_without_improvement,
                )

                if self.early_stopping_patience is not None and monitored is not None:
                    self.epochs_without_improvement = (
                        0 if is_best else self.epochs_without_improvement + 1
                    )
                    if self.epochs_without_improvement >= self.early_stopping_patience:
                        break

    def _run_epoch(self, loader: Iterable[Any], *, training: bool) -> dict[str, float]:
        self.model.train(training)
        step_fn = self.train_step if training else self.eval_step
        assert step_fn is not None  # eval only reached when eval_step is set

        averages = _RunningAverages()
        # Per-step pre-clip gradient norms, training only.
        self._epoch_grad_norms = []
        context = torch.enable_grad() if training else torch.no_grad()
        # Accumulation applies to training only; each eval batch is its own group.
        group_size = self.accumulation_steps if training else 1

        with context:
            group: list[Any] = []
            for batch in loader:
                group.append(batch)
                if len(group) == group_size:
                    self._run_group(group, step_fn, training, averages)
                    group = []

            if group:  # trailing group shorter than group_size
                self._run_group(group, step_fn, training, averages)

        metrics = averages.averages()
        if training and self._epoch_grad_norms:
            metrics.update(self._gradient_norm_metrics())
        return metrics

    def _gradient_norm_metrics(self) -> dict[str, float]:
        """Summarize this epoch's pre-clip gradient norms.

        ``grad_clip_fraction`` is the share of optimizer steps whose norm
        exceeded ``grad_clip_norm`` and were rescaled.
        """
        norms = sorted(self._epoch_grad_norms)
        count = len(norms)
        middle = count // 2
        median = (
            norms[middle]
            if count % 2
            else 0.5 * (norms[middle - 1] + norms[middle])
        )
        metrics = {
            "grad_norm_mean": sum(norms) / count,
            "grad_norm_median": median,
            "grad_norm_max": norms[-1],
        }
        if self.grad_clip_norm is not None:
            clipped = sum(1 for norm in norms if norm > self.grad_clip_norm)
            metrics["grad_clip_fraction"] = clipped / count
        return metrics

    def _run_group(
        self,
        group: list[Any],
        step_fn: StepFn,
        training: bool,
        averages: _RunningAverages,
    ) -> None:
        """Run one optimizer step's worth of micro-batches.

        During training ``group`` holds ``accumulation_steps`` micro-batches
        (or fewer, for a trailing group); each loss is scaled by
        ``1 / len(group)`` and backwarded immediately, so a short trailing
        group is still averaged correctly and only one micro-batch's
        activations are live at once. During evaluation ``group`` holds
        exactly one batch.
        """
        if training:
            self.optimizer.zero_grad(set_to_none=True)

        for batch in group:
            batch = _move_to_device(batch, self.device)
            output = step_fn(self.model, batch, self.device)
            output.metrics.setdefault("loss", float(output.loss.detach()))

            if training:
                (output.loss / len(group)).backward()

            averages.update(output.metrics)

        if training:
            # clip_grad_norm_ returns the total norm before clipping; recorded
            # whether or not clipping is enabled.
            total_norm = (
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_norm
                )
                if self.grad_clip_norm is not None
                else _total_gradient_norm(self.model)
            )
            self._epoch_grad_norms.append(float(total_norm))

            self.optimizer.step()

            if self.scheduler is not None and self.scheduler_step_on == "step":
                self.scheduler.step()

            self.global_step += 1
