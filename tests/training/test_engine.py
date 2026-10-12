"""Tests for gm_attention.training.engine.

The step functions used throughout are a trivial synthetic linear
regression -- the point of these tests is the loop's own behavior (device
placement, backward/step/clip ordering, scheduler cadence, checkpointing,
early stopping, resuming), not model convergence in any real sense.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from gm_attention.representations.field_type import SO2FeatureField, SO2FieldType
from gm_attention.training.checkpointing import CheckpointManager
from gm_attention.training.engine import (
    StepOutput,
    Trainer,
    _move_to_device,
    _RunningAverages,
)
from gm_attention.training.logging import MetricLogger


def _regression_batches(n_batches: int = 4, batch_size: int = 8) -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(0)
    true_weight = torch.tensor([[1.0], [-2.0], [0.5], [3.0]])
    batches = []
    for _ in range(n_batches):
        x = torch.randn(batch_size, 4, generator=generator)
        y = x @ true_weight
        batches.append((x, y))
    return batches


def _train_step(model: nn.Module, batch, device: torch.device) -> StepOutput:
    x, y = batch
    prediction = model(x)
    loss = F.mse_loss(prediction, y)
    return StepOutput(loss=loss, metrics={"mse": float(loss.detach())})


# ==========================================================================
# _move_to_device
# ==========================================================================


def test_move_to_device_moves_a_tensor() -> None:
    tensor = torch.zeros(2)
    moved = _move_to_device(tensor, torch.device("cpu"))
    assert moved.device == torch.device("cpu")


def test_move_to_device_moves_an_so2_feature_field() -> None:
    field_type = SO2FieldType({0: 2})
    field = SO2FeatureField.from_flat(field_type, torch.zeros(2, dtype=torch.float64))

    moved = _move_to_device(field, torch.device("cpu"))

    assert isinstance(moved, SO2FeatureField)
    assert moved.device == torch.device("cpu")


def test_move_to_device_recurses_into_dicts_lists_and_tuples() -> None:
    batch = {"x": torch.zeros(1), "extra": [torch.zeros(2), (torch.zeros(3), "label")]}
    moved = _move_to_device(batch, torch.device("cpu"))

    assert moved["x"].device == torch.device("cpu")
    assert moved["extra"][0].device == torch.device("cpu")
    assert moved["extra"][1][0].device == torch.device("cpu")
    assert moved["extra"][1][1] == "label"  # passed through untouched


def test_move_to_device_passes_through_plain_values() -> None:
    assert _move_to_device(5, torch.device("cpu")) == 5
    assert _move_to_device("hello", torch.device("cpu")) == "hello"


def test_move_to_device_reconstructs_a_namedtuple_by_field_not_a_single_iterable() -> None:
    """A namedtuple's constructor takes its fields as separate positional
    arguments -- unlike a plain tuple's, which accepts one iterable -- so
    reconstructing one the same way a plain tuple is reconstructed raises a
    missing-arguments TypeError past the first field. Regression test for
    exactly that bug."""

    class Batch(NamedTuple):
        x: torch.Tensor
        y: torch.Tensor

    batch = Batch(torch.zeros(2), torch.ones(3))
    moved = _move_to_device(batch, torch.device("cpu"))

    assert isinstance(moved, Batch)
    assert moved.x.device == torch.device("cpu")
    assert moved.y.device == torch.device("cpu")
    assert torch.equal(moved.x, batch.x)
    assert torch.equal(moved.y, batch.y)


# ==========================================================================
# _RunningAverages
# ==========================================================================


def test_running_averages_computes_the_unweighted_mean() -> None:
    averages = _RunningAverages()
    averages.update({"loss": 2.0, "acc": 1.0})
    averages.update({"loss": 4.0, "acc": 0.0})

    result = averages.averages()
    assert result["loss"] == pytest.approx(3.0)
    assert result["acc"] == pytest.approx(0.5)


def test_running_averages_coerces_tensor_values_to_float() -> None:
    averages = _RunningAverages()
    averages.update({"loss": torch.tensor(2.0)})  # type: ignore[dict-item]

    assert isinstance(averages.averages()["loss"], float)


# ==========================================================================
# Trainer construction validation
# ==========================================================================


def test_rejects_a_nonpositive_grad_clip_norm() -> None:
    with pytest.raises(ValueError):
        Trainer(
            nn.Linear(4, 1), torch.optim.SGD(nn.Linear(4, 1).parameters(), lr=0.1),
            _train_step, grad_clip_norm=0.0,
        )


# ==========================================================================
# Gradient-norm diagnostics
# ==========================================================================


def test_training_metrics_carry_gradient_norm_statistics() -> None:
    """A learning-rate comparison is only interpretable alongside these.

    If clipping binds on most steps, the configured rate is not the step
    size the model took, and the comparison measures the clipping threshold
    rather than the rate.
    """
    model = nn.Linear(4, 1)
    trainer = Trainer(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        _train_step,
        grad_clip_norm=1.0,
    )

    metrics = trainer._run_epoch(_regression_batches(), training=True)

    for name in (
        "grad_norm_mean",
        "grad_norm_median",
        "grad_norm_max",
        "grad_clip_fraction",
    ):
        assert name in metrics, f"{name} missing from training metrics"

    assert metrics["grad_norm_max"] >= metrics["grad_norm_median"]
    assert metrics["grad_norm_max"] >= metrics["grad_norm_mean"]
    assert 0.0 <= metrics["grad_clip_fraction"] <= 1.0


def test_the_recorded_norm_is_measured_before_clipping() -> None:
    """The useful quantity is how large the step *would* have been.

    Recording the post-clip norm would report the threshold back to itself
    and never show clipping as active.
    """
    model = nn.Linear(4, 1)
    with torch.no_grad():
        # A large initial weight makes the first gradients far exceed a tiny
        # clipping threshold, so a pre-clip norm is unmistakably above it.
        model.weight.mul_(100.0)
    trainer = Trainer(
        model,
        torch.optim.SGD(model.parameters(), lr=1e-6),
        _train_step,
        grad_clip_norm=1e-3,
    )

    metrics = trainer._run_epoch(_regression_batches(), training=True)

    assert metrics["grad_norm_max"] > 1e-3
    assert metrics["grad_clip_fraction"] == 1.0


def test_gradient_norms_are_recorded_without_clipping_configured() -> None:
    """The diagnostic must not depend on the clipping configuration, or a
    run with clipping off would report nothing about its step sizes."""
    model = nn.Linear(4, 1)
    trainer = Trainer(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        _train_step,
        grad_clip_norm=None,
    )

    metrics = trainer._run_epoch(_regression_batches(), training=True)

    assert metrics["grad_norm_max"] > 0.0
    # No threshold means no fraction to report, rather than a misleading 0.
    assert "grad_clip_fraction" not in metrics


def test_validation_metrics_carry_no_gradient_norms() -> None:
    """Evaluation never backwards, so a norm there would be meaningless."""
    model = nn.Linear(4, 1)
    trainer = Trainer(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        _train_step,
        eval_step=_train_step,
        grad_clip_norm=1.0,
    )

    metrics = trainer._run_epoch(_regression_batches(), training=False)

    assert not any(name.startswith("grad_") for name in metrics)


def test_each_epoch_reports_its_own_gradient_norms() -> None:
    """Carrying norms across epochs would blur exactly the early-training
    transient the learning-rate bracket is looking at."""
    model = nn.Linear(4, 1)
    trainer = Trainer(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        _train_step,
        grad_clip_norm=1.0,
    )
    batches = _regression_batches()

    trainer._run_epoch(batches, training=True)
    assert len(trainer._epoch_grad_norms) == len(batches)

    trainer._run_epoch(batches, training=True)
    assert len(trainer._epoch_grad_norms) == len(batches)


def test_rejects_a_nonpositive_validate_every() -> None:
    with pytest.raises(ValueError):
        Trainer(
            nn.Linear(4, 1), torch.optim.SGD(nn.Linear(4, 1).parameters(), lr=0.1),
            _train_step, validate_every=0,
        )


def test_rejects_a_nonpositive_accumulation_steps() -> None:
    with pytest.raises(ValueError):
        Trainer(
            nn.Linear(4, 1), torch.optim.SGD(nn.Linear(4, 1).parameters(), lr=0.1),
            _train_step, accumulation_steps=0,
        )


def test_rejects_a_nonpositive_early_stopping_patience(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        Trainer(
            nn.Linear(4, 1), torch.optim.SGD(nn.Linear(4, 1).parameters(), lr=0.1),
            _train_step, early_stopping_patience=0,
            checkpoint_manager=CheckpointManager(tmp_path),
        )


def test_early_stopping_requires_a_checkpoint_manager() -> None:
    with pytest.raises(ValueError):
        Trainer(
            nn.Linear(4, 1), torch.optim.SGD(nn.Linear(4, 1).parameters(), lr=0.1),
            _train_step, early_stopping_patience=2,
        )


def test_fit_rejects_a_nonpositive_max_epochs() -> None:
    model = nn.Linear(4, 1)
    trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.1), _train_step)

    with pytest.raises(ValueError):
        trainer.fit(_regression_batches(), max_epochs=0)


# ==========================================================================
# Basic fit()
# ==========================================================================


def test_fit_reduces_the_training_loss() -> None:
    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    trainer = Trainer(model, torch.optim.Adam(model.parameters(), lr=0.05), _train_step)

    batches = _regression_batches()
    first_epoch_loss = trainer._run_epoch(batches, training=True)["mse"]
    for _ in range(20):
        trainer._run_epoch(batches, training=True)
    last_epoch_loss = trainer._run_epoch(batches, training=True)["mse"]

    assert last_epoch_loss < first_epoch_loss


def test_step_output_backfills_loss_into_metrics_when_absent() -> None:
    def step(model, batch, device):
        x, y = batch
        loss = F.mse_loss(model(x), y)
        return StepOutput(loss=loss, metrics={})

    model = nn.Linear(4, 1)
    trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.01), step)
    result = trainer._run_epoch(_regression_batches(n_batches=1), training=True)

    assert "loss" in result


def test_step_output_does_not_overwrite_an_explicit_loss_metric() -> None:
    def step(model, batch, device):
        x, y = batch
        loss = F.mse_loss(model(x), y)
        return StepOutput(loss=loss, metrics={"loss": -1.0})

    model = nn.Linear(4, 1)
    trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.01), step)
    result = trainer._run_epoch(_regression_batches(n_batches=1), training=True)

    assert result["loss"] == -1.0


def test_train_and_eval_mode_are_set_correctly_during_each_phase() -> None:
    observed_training_flags = []

    def step(model, batch, device):
        observed_training_flags.append(model.training)
        x, y = batch
        return StepOutput(loss=F.mse_loss(model(x), y))

    model = nn.Linear(4, 1)
    trainer = Trainer(
        model, torch.optim.SGD(model.parameters(), lr=0.01), step, eval_step=step
    )
    trainer.fit(_regression_batches(n_batches=1), max_epochs=1, val_loader=_regression_batches(n_batches=1))

    assert observed_training_flags == [True, False]


def test_eval_step_runs_under_no_grad() -> None:
    def eval_step(model, batch, device):
        x, y = batch
        prediction = model(x)
        assert prediction.requires_grad is False
        return StepOutput(loss=F.mse_loss(prediction, y))

    model = nn.Linear(4, 1)
    trainer = Trainer(
        model, torch.optim.SGD(model.parameters(), lr=0.01), _train_step, eval_step=eval_step
    )
    trainer.fit(
        _regression_batches(n_batches=1), max_epochs=1, val_loader=_regression_batches(n_batches=1)
    )


def test_validation_is_skipped_without_an_eval_step(tmp_path: Path) -> None:
    model = nn.Linear(4, 1)
    logger = MetricLogger(tmp_path)
    trainer = Trainer(
        model, torch.optim.SGD(model.parameters(), lr=0.01), _train_step, logger=logger
    )
    trainer.fit(
        _regression_batches(n_batches=1), max_epochs=1, val_loader=_regression_batches(n_batches=1)
    )

    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert all(record["split"] != "val" for record in records)


# ==========================================================================
# Gradient clipping
# ==========================================================================


def _total_grad_norm(model: nn.Module) -> torch.Tensor:
    squared_norms = torch.stack(
        [p.grad.pow(2).sum() for p in model.parameters() if p.grad is not None]
    )
    return torch.sqrt(squared_norms.sum())


def test_grad_clip_norm_bounds_the_gradient_norm_after_backward() -> None:
    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    with torch.no_grad():
        model.weight.fill_(1000.0)  # force a huge gradient

    trainer = Trainer(
        model, torch.optim.SGD(model.parameters(), lr=1e-8), _train_step, grad_clip_norm=1.0
    )
    trainer._run_epoch(_regression_batches(n_batches=1), training=True)

    assert _total_grad_norm(model) <= 1.0 + 1e-4


def test_without_grad_clip_norm_the_gradient_is_left_unbounded() -> None:
    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    with torch.no_grad():
        model.weight.fill_(1000.0)

    trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=1e-8), _train_step)
    trainer._run_epoch(_regression_batches(n_batches=1), training=True)

    assert _total_grad_norm(model) > 1.0


# ==========================================================================
# Gradient accumulation
# ==========================================================================


def test_accumulation_steps_groups_batches_into_one_optimizer_step() -> None:
    model = nn.Linear(4, 1)
    trainer = Trainer(
        model, torch.optim.SGD(model.parameters(), lr=0.01), _train_step, accumulation_steps=3,
    )
    trainer._run_epoch(_regression_batches(n_batches=6, batch_size=2), training=True)

    assert trainer.global_step == 2  # 6 batches / 3 per group = 2 optimizer steps


def test_accumulation_produces_the_same_gradient_as_a_real_batch() -> None:
    """The whole point of accumulation: splitting one batch into several
    micro-batches, each backwarded separately and scaled by 1/(group size),
    must land on exactly the gradient a single real batch would have
    produced -- not merely something in the right ballpark."""
    torch.manual_seed(0)
    x = torch.randn(4, 4)
    true_weight = torch.tensor([[1.0], [-2.0], [0.5], [3.0]])
    y = x @ true_weight

    model = nn.Linear(4, 1)
    initial_state = {name: value.clone() for name, value in model.state_dict().items()}

    reference_prediction = model(x)
    F.mse_loss(reference_prediction, y).backward()
    assert model.weight.grad is not None
    reference_grad = model.weight.grad.clone()

    model.load_state_dict(initial_state)
    model.zero_grad()

    micro_batches = [(x[i : i + 1], y[i : i + 1]) for i in range(4)]
    # lr=0.0: optimizer.step() is still called, but never moves the
    # weights, so the check below is isolated to the accumulated .grad.
    trainer = Trainer(
        model, torch.optim.SGD(model.parameters(), lr=0.0), _train_step, accumulation_steps=4,
    )
    trainer._run_epoch(micro_batches, training=True)

    assert model.weight.grad is not None
    assert torch.allclose(model.weight.grad, reference_grad, atol=1e-6)


def test_a_short_trailing_group_is_still_correctly_averaged() -> None:
    """2 micro-batches with accumulation_steps=3 forms one short trailing
    group. Its scaling must use its own actual size (2), not the configured
    accumulation_steps (3) -- otherwise its gradient would come out 2/3 the
    size a real batch of 2 would have produced."""
    torch.manual_seed(1)
    x = torch.randn(2, 4)
    true_weight = torch.tensor([[1.0], [-2.0], [0.5], [3.0]])
    y = x @ true_weight

    model = nn.Linear(4, 1)
    initial_state = {name: value.clone() for name, value in model.state_dict().items()}

    reference_prediction = model(x)
    F.mse_loss(reference_prediction, y).backward()
    assert model.weight.grad is not None
    reference_grad = model.weight.grad.clone()

    model.load_state_dict(initial_state)
    model.zero_grad()

    micro_batches = [(x[i : i + 1], y[i : i + 1]) for i in range(2)]
    trainer = Trainer(
        model, torch.optim.SGD(model.parameters(), lr=0.0), _train_step, accumulation_steps=3,
    )
    trainer._run_epoch(micro_batches, training=True)  # shorter than one full group

    assert model.weight.grad is not None
    assert torch.allclose(model.weight.grad, reference_grad, atol=1e-6)
    assert trainer.global_step == 1


def test_grad_clip_norm_applies_once_to_the_fully_accumulated_gradient() -> None:
    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    with torch.no_grad():
        model.weight.fill_(1000.0)  # force a huge gradient

    trainer = Trainer(
        model, torch.optim.SGD(model.parameters(), lr=1e-8), _train_step,
        accumulation_steps=4, grad_clip_norm=1.0,
    )
    trainer._run_epoch(_regression_batches(n_batches=4, batch_size=2), training=True)

    assert _total_grad_norm(model) <= 1.0 + 1e-4


# ==========================================================================
# Scheduler stepping cadence
# ==========================================================================


def test_scheduler_steps_once_per_epoch_by_default() -> None:
    model = nn.Linear(4, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 0.5**step)

    trainer = Trainer(model, optimizer, _train_step, scheduler=scheduler)
    trainer.fit(_regression_batches(n_batches=3), max_epochs=3)

    # 3 epochs -> 3 scheduler.step() calls -> multiplier 0.5**3.
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0 * 0.5**3)


def test_scheduler_steps_once_per_batch_when_configured() -> None:
    model = nn.Linear(4, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 0.5**step)

    trainer = Trainer(
        model, optimizer, _train_step, scheduler=scheduler, scheduler_step_on="step"
    )
    trainer.fit(_regression_batches(n_batches=3), max_epochs=2)

    # 2 epochs * 3 batches = 6 scheduler.step() calls.
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0 * 0.5**6)


# ==========================================================================
# Metric-driven schedulers (ReduceLROnPlateau)
# ==========================================================================


def test_scheduler_metric_requires_scheduler_step_on_epoch() -> None:
    model = nn.Linear(4, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    with pytest.raises(ValueError, match="scheduler_metric"):
        Trainer(
            model,
            optimizer,
            _train_step,
            scheduler=scheduler,
            scheduler_metric="loss",
            scheduler_step_on="step",
        )


def test_reduce_lr_on_plateau_requires_scheduler_metric() -> None:
    model = nn.Linear(4, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    with pytest.raises(ValueError, match="ReduceLROnPlateau"):
        Trainer(model, optimizer, _train_step, scheduler=scheduler)


def test_scheduler_metric_steps_reduce_lr_on_plateau_using_validation_metrics() -> None:
    model = nn.Linear(4, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=0)

    def eval_step(model, batch, device):
        # Constant, non-improving validation loss every epoch: with
        # patience=0, ReduceLROnPlateau reduces starting the epoch after
        # the first (which just sets the initial "best").
        return StepOutput(loss=torch.tensor(0.0), metrics={"val_loss": 1.0})

    trainer = Trainer(
        model,
        optimizer,
        _train_step,
        eval_step=eval_step,
        scheduler=scheduler,
        scheduler_metric="val_loss",
    )
    trainer.fit(
        _regression_batches(n_batches=1), max_epochs=3, val_loader=_regression_batches(n_batches=1)
    )

    assert optimizer.param_groups[0]["lr"] < 1.0


def test_scheduler_metric_falls_back_to_train_metrics_without_validation() -> None:
    model = nn.Linear(4, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=0)

    def train_step(model, batch, device):
        x, y = batch
        loss = F.mse_loss(model(x), y)
        return StepOutput(loss=loss, metrics={"mse": 1.0})  # constant: never "improves"

    trainer = Trainer(
        model, optimizer, train_step, scheduler=scheduler, scheduler_metric="mse"
    )
    trainer.fit(_regression_batches(n_batches=1), max_epochs=3)  # no val_loader at all

    assert optimizer.param_groups[0]["lr"] < 1.0


def test_scheduler_metric_raises_a_clear_error_when_the_metric_is_missing() -> None:
    model = nn.Linear(4, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    trainer = Trainer(
        model, optimizer, _train_step, scheduler=scheduler, scheduler_metric="nonexistent_metric"
    )

    with pytest.raises(KeyError, match="nonexistent_metric"):
        trainer.fit(_regression_batches(n_batches=1), max_epochs=1)


# ==========================================================================
# Checkpointing integration
# ==========================================================================


def test_fit_with_a_checkpoint_manager_writes_last_and_best(tmp_path: Path) -> None:
    model = nn.Linear(4, 1)
    manager = CheckpointManager(tmp_path, monitor="mse", mode="min")
    trainer = Trainer(
        model, torch.optim.Adam(model.parameters(), lr=0.05), _train_step,
        eval_step=_train_step, checkpoint_manager=manager,
    )
    trainer.fit(
        _regression_batches(), max_epochs=3, val_loader=_regression_batches(n_batches=1)
    )

    assert (tmp_path / "last.pt").exists()
    assert (tmp_path / "best.pt").exists()


def test_resuming_continues_from_the_saved_epoch_not_from_scratch(tmp_path: Path) -> None:
    model = nn.Linear(4, 1)
    manager = CheckpointManager(tmp_path)
    trainer = Trainer(
        model, torch.optim.SGD(model.parameters(), lr=0.01), _train_step,
        checkpoint_manager=manager,
    )
    trainer.fit(_regression_batches(n_batches=1), max_epochs=2)
    assert trainer.epoch == 1  # 0-indexed: epochs 0 and 1 ran

    resumed_model = nn.Linear(4, 1)
    resumed_trainer = Trainer(
        resumed_model, torch.optim.SGD(resumed_model.parameters(), lr=0.01), _train_step,
    )
    resumed_trainer.fit(
        _regression_batches(n_batches=1), max_epochs=4, resume_from=tmp_path / "last.pt"
    )

    assert resumed_trainer.epoch == 3  # continued through epochs 2 and 3, not restarted


def test_restore_scheduler_false_starts_the_fresh_schedule_from_its_own_rate(
    tmp_path: Path,
) -> None:
    """A warm restart: the optimizer's state comes back from the checkpoint,
    the learning rate does not -- the new cycle starts at the fresh
    scheduler's base rate, not the decayed rate the checkpoint's param
    groups carry, and decays from there."""
    batches = _regression_batches(n_batches=1)
    model = nn.Linear(4, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    Trainer(
        model, optimizer, _train_step, scheduler=scheduler,
        checkpoint_manager=CheckpointManager(tmp_path),
    ).fit(batches, max_epochs=2)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0)  # its cycle ran out

    resumed = nn.Linear(4, 1)
    fresh_optimizer = torch.optim.Adam(resumed.parameters(), lr=0.05)
    fresh = torch.optim.lr_scheduler.CosineAnnealingLR(fresh_optimizer, T_max=2)
    seen: list[float] = []
    trainer = Trainer(
        resumed, fresh_optimizer, _train_step, scheduler=fresh,
        on_epoch_start=lambda epoch: seen.append(fresh_optimizer.param_groups[0]["lr"]),
    )
    trainer.fit(
        batches, max_epochs=4, resume_from=tmp_path / "last.pt", restore_scheduler=False
    )

    assert seen == pytest.approx([0.05, 0.025])  # the new peak, then cos(pi/2)'s midpoint
    assert trainer.epoch == 3
    assert any("exp_avg" in state for state in fresh_optimizer.state.values()), (
        "Adam's moments were restored alongside the reset rate"
    )


def test_restore_scheduler_false_needs_a_scheduler_with_base_rates(tmp_path: Path) -> None:
    batches = _regression_batches(n_batches=1)
    model = nn.Linear(4, 1)
    Trainer(
        model, torch.optim.SGD(model.parameters(), lr=0.01), _train_step,
        checkpoint_manager=CheckpointManager(tmp_path),
    ).fit(batches, max_epochs=1)

    resumed = nn.Linear(4, 1)
    optimizer = torch.optim.SGD(resumed.parameters(), lr=0.01)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    trainer = Trainer(
        resumed, optimizer, _train_step, scheduler=plateau, scheduler_metric="mse"
    )
    with pytest.raises(ValueError, match="base_lrs"):
        trainer.fit(
            batches, max_epochs=2, resume_from=tmp_path / "last.pt", restore_scheduler=False
        )


# ==========================================================================
# Early stopping
# ==========================================================================


def test_early_stopping_halts_once_the_monitored_metric_plateaus(tmp_path: Path) -> None:
    scripted_losses = iter([5.0, 4.0, 3.0, 3.0, 3.0])  # plateaus after epoch 2

    def eval_step(model, batch, device):
        x, y = batch
        real_loss = F.mse_loss(model(x), y)
        return StepOutput(loss=real_loss, metrics={"loss": next(scripted_losses)})

    model = nn.Linear(4, 1)
    manager = CheckpointManager(tmp_path, monitor="loss", mode="min")
    trainer = Trainer(
        model, torch.optim.SGD(model.parameters(), lr=0.01), _train_step,
        eval_step=eval_step, checkpoint_manager=manager, early_stopping_patience=2,
    )

    trainer.fit(
        _regression_batches(n_batches=1), max_epochs=10, val_loader=_regression_batches(n_batches=1)
    )

    # Best at epoch 2 (loss=3.0); epochs 3 and 4 don't improve -> patience
    # of 2 exhausted at epoch 4, well short of max_epochs=10.
    assert trainer.epoch == 4


# ==========================================================================
# Validation-only monitoring, epoch hook, early-stopping persistence
# ==========================================================================


class _RecordingScheduler:
    """A ReduceLROnPlateau-shaped scheduler that just records what it is fed."""

    def __init__(self) -> None:
        self.values: list[float] = []

    def step(self, value: float) -> None:
        self.values.append(float(value))

    def state_dict(self) -> dict[str, object]:
        return {}

    def load_state_dict(self, state: dict[str, object]) -> None:
        pass


def _constant_step(loss: float):
    def step(model: nn.Module, batch, device: torch.device) -> StepOutput:
        parameter = next(model.parameters())
        return StepOutput(loss=parameter.sum() * 0.0 + loss, metrics={"monitored": loss})

    return step


def test_scheduler_is_not_fed_training_metrics_on_a_non_validation_epoch(
    tmp_path: Path,
) -> None:
    """The substitution bug: with validate_every=2, odd epochs previously
    handed the scheduler a *training* value, silently switching which
    distribution the monitored number came from between epochs."""
    scheduler = _RecordingScheduler()
    trainer = Trainer(
        nn.Linear(2, 1),
        torch.optim.SGD(nn.Linear(2, 1).parameters(), lr=0.1),
        _constant_step(1.0),  # training reports 1.0
        eval_step=_constant_step(0.5),  # validation reports 0.5
        scheduler=scheduler,  # type: ignore[arg-type]
        scheduler_step_on="epoch",
        scheduler_metric="monitored",
        validate_every=2,
    )
    batches = [(torch.randn(2, 2), torch.randn(2, 1))]
    trainer.fit(batches, max_epochs=4, val_loader=batches)

    # Epochs 0 and 2 validate; 1 and 3 do not and must contribute nothing.
    assert scheduler.values == [0.5, 0.5]


def test_scheduler_still_steps_every_epoch_without_a_scheduler_metric() -> None:
    """A plain no-argument scheduler is unaffected by the validation rule."""
    calls = []

    class _Plain:
        def step(self) -> None:
            calls.append(1)

        def state_dict(self) -> dict[str, object]:
            return {}

        def load_state_dict(self, state: dict[str, object]) -> None:
            pass

    trainer = Trainer(
        nn.Linear(2, 1),
        torch.optim.SGD(nn.Linear(2, 1).parameters(), lr=0.1),
        _constant_step(1.0),
        eval_step=_constant_step(0.5),
        scheduler=_Plain(),  # type: ignore[arg-type]
        scheduler_step_on="epoch",
        validate_every=2,
    )
    batches = [(torch.randn(2, 2), torch.randn(2, 1))]
    trainer.fit(batches, max_epochs=4, val_loader=batches)

    assert len(calls) == 4


def test_best_checkpoint_is_only_judged_on_validation_epochs(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path, monitor="monitored", mode="min")
    trainer = Trainer(
        nn.Linear(2, 1),
        torch.optim.SGD(nn.Linear(2, 1).parameters(), lr=0.1),
        # Training looks spectacular; validation does not. Only validation
        # may set best.pt.
        _constant_step(0.001),
        eval_step=_constant_step(0.5),
        checkpoint_manager=manager,
        validate_every=2,
    )
    batches = [(torch.randn(2, 2), torch.randn(2, 1))]
    trainer.fit(batches, max_epochs=4, val_loader=batches)

    assert manager.best_value == pytest.approx(0.5)


def test_training_metrics_still_drive_monitoring_when_there_is_no_validation(
    tmp_path: Path,
) -> None:
    """A run with no val_loader at all is unaffected: training metrics are
    the only thing it has, and that is a consistent choice, not a switch."""
    manager = CheckpointManager(tmp_path, monitor="monitored", mode="min")
    trainer = Trainer(
        nn.Linear(2, 1),
        torch.optim.SGD(nn.Linear(2, 1).parameters(), lr=0.1),
        _constant_step(0.25),
        checkpoint_manager=manager,
    )
    trainer.fit([(torch.randn(2, 2), torch.randn(2, 1))], max_epochs=2)

    assert manager.best_value == pytest.approx(0.25)


def test_on_epoch_start_receives_every_epoch_number_before_training() -> None:
    seen: list[int] = []
    trainer = Trainer(
        nn.Linear(2, 1),
        torch.optim.SGD(nn.Linear(2, 1).parameters(), lr=0.1),
        _constant_step(1.0),
        on_epoch_start=seen.append,
    )
    trainer.fit([(torch.randn(2, 2), torch.randn(2, 1))], max_epochs=3)

    assert seen == [0, 1, 2]


def test_early_stopping_counter_survives_resume(tmp_path: Path) -> None:
    """Reset to zero on resume, a run gets a fresh patience budget every
    restart and can never actually stop."""
    model = nn.Linear(2, 1)
    manager = CheckpointManager(tmp_path, monitor="monitored", mode="min")
    batches = [(torch.randn(2, 2), torch.randn(2, 1))]

    first = Trainer(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        _constant_step(1.0),
        eval_step=_constant_step(1.0),  # never improves after epoch 0
        checkpoint_manager=manager,
        early_stopping_patience=3,
    )
    first.fit(batches, max_epochs=3, val_loader=batches)
    assert first.epochs_without_improvement == 2  # epoch 0 set the bar; 1 and 2 did not beat it

    resumed = Trainer(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        _constant_step(1.0),
        eval_step=_constant_step(1.0),
        checkpoint_manager=CheckpointManager(tmp_path, monitor="monitored", mode="min"),
        early_stopping_patience=3,
    )
    resumed.fit(
        batches, max_epochs=10, val_loader=batches, resume_from=tmp_path / "last.pt"
    )

    # It must stop almost immediately (one more bad epoch reaches patience),
    # not run all the way to epoch 10 with a reset counter.
    assert resumed.epochs_without_improvement == 3
    assert resumed.epoch < 9
