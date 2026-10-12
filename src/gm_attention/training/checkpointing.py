"""Checkpoint save/load, and a manager that tracks a run's "last" and "best".

A checkpoint is a dict of ``state_dict()``s plus run metadata written with
``torch.save``. ``CheckpointManager``, driven once per epoch by
:class:`~gm_attention.training.engine.Trainer`, always overwrites
``last.pt``, writes ``best.pt`` when the monitored metric improves, and
optionally writes periodic numbered snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import nn

from gm_attention.training.reproducibility import get_rng_state, set_rng_state


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    """Run metadata recovered from a checkpoint by :func:`load_checkpoint`.

    ``best_value`` and ``epochs_without_improvement`` are the decision state
    a resume restores: without them the first resumed epoch always counts
    as an improvement and the early-stopping counter restarts at zero. Both
    default to "nothing known yet" so a checkpoint without them loads.
    """

    epoch: int
    global_step: int
    metrics: dict[str, float]
    best_value: float | None = None
    epochs_without_improvement: int = 0


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    epoch: int,
    global_step: int,
    metrics: dict[str, float] | None = None,
    best_value: float | None = None,
    epochs_without_improvement: int = 0,
    include_rng_state: bool = True,
) -> None:
    """Write a checkpoint to ``path``.

    The file is written to a sibling ``.tmp`` path and renamed into place,
    so an interrupted write never leaves a corrupt file at ``path``.

    Parameters
    ----------
    path
        Destination file. Parent directories are created if needed.
    model
        Its ``state_dict()`` is saved.
    optimizer, scheduler
        Their ``state_dict()``s are saved if given, else omitted.
    epoch, global_step
        The run's current position; returned by :func:`load_checkpoint`.
    metrics
        Scalar metrics describing this point in training; returned by
        :func:`load_checkpoint`.
    best_value, epochs_without_improvement
        The run's best monitored value so far and its early-stopping
        counter; see :class:`CheckpointMetadata`.
    include_rng_state
        Whether to capture :func:`~gm_attention.training.reproducibility.get_rng_state`
        alongside the model, for a bit-exact resume.
    """
    payload: dict[str, object] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "metrics": dict(metrics) if metrics is not None else {},
        "best_value": best_value,
        "epochs_without_improvement": epochs_without_improvement,
        "rng_state": get_rng_state() if include_rng_state else None,
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    map_location: torch.device | str | None = "cpu",
    restore_rng_state: bool = True,
) -> CheckpointMetadata:
    """Load a checkpoint written by :func:`save_checkpoint`, restoring in place.

    Parameters
    ----------
    path
        Checkpoint file to load.
    model
        Has ``load_state_dict`` called on it.
    optimizer, scheduler
        Have ``load_state_dict`` called on them if given and the checkpoint
        holds that state; otherwise left untouched.
    map_location
        Passed to ``torch.load``.
    restore_rng_state
        Whether to restore the RNG state the checkpoint captured, if any.

    Returns
    -------
    CheckpointMetadata
        The run's ``epoch``, ``global_step``, and ``metrics`` at save time.

    Notes
    -----
    Loaded with ``weights_only=False``: the checkpoint stores an
    :class:`~gm_attention.training.reproducibility.RngState` and optimizer
    and scheduler state dicts, which are outside the allowlist of
    ``weights_only=True``. Checkpoints are therefore trusted files.
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(payload["model"])

    if optimizer is not None and payload["optimizer"] is not None:
        optimizer.load_state_dict(payload["optimizer"])

    if scheduler is not None and payload["scheduler"] is not None:
        scheduler.load_state_dict(payload["scheduler"])

    if restore_rng_state and payload["rng_state"] is not None:
        set_rng_state(payload["rng_state"])

    return CheckpointMetadata(
        epoch=payload["epoch"],
        global_step=payload["global_step"],
        metrics=payload["metrics"],
        # .get: a checkpoint without these fields must still load.
        best_value=payload.get("best_value"),
        epochs_without_improvement=payload.get("epochs_without_improvement", 0),
    )


class CheckpointManager:
    """Drives ``save_checkpoint`` once per epoch: always ``last.pt``, and
    ``best.pt`` whenever ``monitor`` improves.

    Parameters
    ----------
    directory
        Directory to write checkpoints into; created if it does not exist.
    monitor
        Key into the ``metrics`` dict passed to :meth:`save` that decides
        ``best.pt``.
    mode
        ``"min"`` if smaller ``monitor`` values are better (e.g. a loss),
        ``"max"`` if larger is better (e.g. an accuracy).
    keep_periodic_every
        If given, also write a numbered ``epoch_{epoch:04d}.pt`` snapshot
        every this many epochs, in addition to ``last.pt``/``best.pt``.
        ``None`` (the default) keeps only the two.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        monitor: str = "loss",
        mode: Literal["min", "max"] = "min",
        keep_periodic_every: int | None = None,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}.")

        if keep_periodic_every is not None and keep_periodic_every < 1:
            raise ValueError(
                f"keep_periodic_every must be at least 1 if given, got {keep_periodic_every}."
            )

        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode: Literal["min", "max"] = mode
        self.keep_periodic_every = keep_periodic_every
        self._best_value: float | None = None

    @property
    def best_value(self) -> float | None:
        """The best monitored value seen so far, or ``None`` if none yet."""
        return self._best_value

    def restore_best_value(self, metadata: CheckpointMetadata) -> None:
        """Re-arm the best-so-far comparison from a loaded checkpoint.

        Called on resume; otherwise the first resumed epoch overwrites
        ``best.pt``. Falls back to the checkpoint's recorded ``metrics``
        when it holds no ``best_value``.
        """
        if metadata.best_value is not None:
            self._best_value = metadata.best_value
        elif self.monitor in metadata.metrics:
            self._best_value = metadata.metrics[self.monitor]

    def save(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        epoch: int,
        global_step: int,
        metrics: dict[str, float] | None = None,
        update_best: bool = True,
        epochs_without_improvement: int = 0,
    ) -> bool:
        """Write ``last.pt`` (and maybe a periodic snapshot and ``best.pt``).

        Parameters
        ----------
        update_best
            Whether this call may claim ``best.pt``. ``False`` for an epoch
            without a monitored value (a non-validation epoch under
            ``validate_every > 1``): ``last.pt`` still refreshes, but no
            best-model decision is made.
        epochs_without_improvement
            The caller's early-stopping counter, stored for resume.

        Returns
        -------
        bool
            Whether ``monitor`` improved this call, i.e. whether ``best.pt``
            was written. ``False`` if ``metrics`` is empty or
            ``update_best`` is ``False``.
        """
        metrics = metrics or {}

        # Decided before anything is written, so last.pt records the bar this
        # epoch set and a resume restores it.
        is_best = False
        if metrics and update_best:
            if self.monitor not in metrics:
                raise KeyError(
                    f"Monitored metric {self.monitor!r} not found in metrics "
                    f"{sorted(metrics)}."
                )
            value = metrics[self.monitor]
            is_best = self._best_value is None or (
                value < self._best_value if self.mode == "min" else value > self._best_value
            )
            if is_best:
                self._best_value = value

        def write(name: str) -> None:
            save_checkpoint(
                self.directory / name,
                model=model, optimizer=optimizer, scheduler=scheduler,
                epoch=epoch, global_step=global_step, metrics=metrics,
                best_value=self._best_value,
                epochs_without_improvement=epochs_without_improvement,
            )

        write("last.pt")

        if (
            self.keep_periodic_every is not None
            and (epoch + 1) % self.keep_periodic_every == 0
        ):
            write(f"epoch_{epoch:04d}.pt")

        if is_best:
            write("best.pt")

        return is_best
