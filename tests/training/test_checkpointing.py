"""Tests for gm_attention.training.checkpointing."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from gm_attention.training.checkpointing import (
    CheckpointManager,
    CheckpointMetadata,
    load_checkpoint,
    save_checkpoint,
)


def _tiny_model() -> nn.Linear:
    return nn.Linear(4, 2)


# ==========================================================================
# save_checkpoint / load_checkpoint
# ==========================================================================


def test_round_trip_restores_model_and_optimizer_state(tmp_path: Path) -> None:
    model = _tiny_model()
    # Adam (rather than plain SGD) so the optimizer's state_dict actually
    # has non-trivial per-parameter state to round-trip.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    loss = model(torch.randn(3, 4)).sum()
    loss.backward()
    optimizer.step()

    saved_model_state = {k: v.clone() for k, v in model.state_dict().items()}
    saved_optimizer_state = optimizer.state_dict()

    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path, model=model, optimizer=optimizer, epoch=3, global_step=100,
        metrics={"loss": 0.5},
    )

    # Mutate both in place, so loading is the only way to get the saved
    # values back.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    optimizer.param_groups[0]["lr"] = 999.0

    fresh_model = _tiny_model()
    fresh_optimizer = torch.optim.Adam(fresh_model.parameters(), lr=0.1)
    metadata = load_checkpoint(path, model=fresh_model, optimizer=fresh_optimizer)

    for key, value in fresh_model.state_dict().items():
        assert torch.equal(value, saved_model_state[key])

    assert fresh_optimizer.state_dict()["param_groups"] == saved_optimizer_state["param_groups"]
    assert metadata.epoch == 3
    assert metadata.global_step == 100
    assert metadata.metrics == {"loss": 0.5}


def test_load_without_optimizer_or_scheduler_in_the_checkpoint_leaves_them_untouched(
    tmp_path: Path,
) -> None:
    model = _tiny_model()
    path = tmp_path / "model_only.pt"
    save_checkpoint(path, model=model, epoch=0, global_step=0)

    fresh_model = _tiny_model()
    fresh_optimizer = torch.optim.SGD(fresh_model.parameters(), lr=0.5)
    original_lr = fresh_optimizer.param_groups[0]["lr"]

    load_checkpoint(path, model=fresh_model, optimizer=fresh_optimizer)

    assert fresh_optimizer.param_groups[0]["lr"] == original_lr


def test_save_checkpoint_leaves_no_tmp_file_behind(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model=_tiny_model(), epoch=0, global_step=0)

    assert path.exists()
    assert not path.with_suffix(path.suffix + ".tmp").exists()


def test_save_checkpoint_creates_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "run1" / "checkpoint.pt"
    save_checkpoint(path, model=_tiny_model(), epoch=0, global_step=0)

    assert path.exists()


def test_include_rng_state_false_omits_it_and_restore_is_a_no_op(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path, model=_tiny_model(), epoch=0, global_step=0, include_rng_state=False
    )

    fresh_model = _tiny_model()  # construction itself consumes RNG draws (weight init)

    torch.manual_seed(123)
    before = torch.get_rng_state().clone()

    load_checkpoint(path, model=fresh_model, restore_rng_state=True)  # must not raise

    assert torch.equal(torch.get_rng_state(), before)


# ==========================================================================
# CheckpointManager
# ==========================================================================


def test_manager_rejects_an_invalid_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        CheckpointManager(tmp_path, mode="lower")  # type: ignore[arg-type]


def test_manager_rejects_a_nonpositive_keep_periodic_every(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        CheckpointManager(tmp_path, keep_periodic_every=0)


def test_manager_always_writes_last(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path)
    manager.save(model=_tiny_model(), epoch=0, global_step=0)

    assert (tmp_path / "last.pt").exists()


def test_manager_with_empty_metrics_never_writes_best(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path, monitor="loss")
    is_best = manager.save(model=_tiny_model(), epoch=0, global_step=0)

    assert is_best is False
    assert not (tmp_path / "best.pt").exists()


def test_manager_raises_when_monitor_is_missing_from_nonempty_metrics(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path, monitor="loss")

    with pytest.raises(KeyError):
        manager.save(model=_tiny_model(), epoch=0, global_step=0, metrics={"accuracy": 0.9})


def test_manager_min_mode_tracks_the_lowest_value(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path, monitor="loss", mode="min")

    assert manager.save(model=_tiny_model(), epoch=0, global_step=0, metrics={"loss": 5.0})
    assert manager.save(model=_tiny_model(), epoch=1, global_step=1, metrics={"loss": 6.0}) is False
    assert manager.save(model=_tiny_model(), epoch=2, global_step=2, metrics={"loss": 4.0})

    best_metadata = load_checkpoint(tmp_path / "best.pt", model=_tiny_model())
    assert best_metadata.epoch == 2  # the epoch-1 write must not have overwritten best.pt


def test_manager_max_mode_tracks_the_highest_value(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path, monitor="accuracy", mode="max")

    assert manager.save(model=_tiny_model(), epoch=0, global_step=0, metrics={"accuracy": 0.5})
    assert manager.save(
        model=_tiny_model(), epoch=1, global_step=1, metrics={"accuracy": 0.4}
    ) is False
    assert manager.save(model=_tiny_model(), epoch=2, global_step=2, metrics={"accuracy": 0.9})

    best_metadata = load_checkpoint(tmp_path / "best.pt", model=_tiny_model())
    assert best_metadata.epoch == 2


def test_manager_keep_periodic_every_writes_only_on_the_configured_epochs(
    tmp_path: Path,
) -> None:
    manager = CheckpointManager(tmp_path, keep_periodic_every=2)

    for epoch in range(4):  # epochs 0, 1, 2, 3
        manager.save(model=_tiny_model(), epoch=epoch, global_step=epoch)

    # (epoch + 1) % 2 == 0 fires at epoch 1 and epoch 3.
    assert (tmp_path / "epoch_0001.pt").exists()
    assert (tmp_path / "epoch_0003.pt").exists()
    assert not (tmp_path / "epoch_0000.pt").exists()
    assert not (tmp_path / "epoch_0002.pt").exists()


def test_manager_without_keep_periodic_every_writes_no_numbered_snapshots(
    tmp_path: Path,
) -> None:
    manager = CheckpointManager(tmp_path)
    for epoch in range(4):
        manager.save(model=_tiny_model(), epoch=epoch, global_step=epoch)

    assert not any(tmp_path.glob("epoch_*.pt"))


# ==========================================================================
# Best-value and early-stopping state across resume (P1 correctness repair)
# ==========================================================================


def test_checkpoint_carries_best_value_and_early_stopping_count(tmp_path: Path) -> None:
    save_checkpoint(
        tmp_path / "c.pt",
        model=_tiny_model(),
        epoch=3,
        global_step=30,
        metrics={"field_mse": 0.25},
        best_value=0.11,
        epochs_without_improvement=4,
    )

    metadata = load_checkpoint(tmp_path / "c.pt", model=_tiny_model())

    assert metadata.best_value == pytest.approx(0.11)
    assert metadata.epochs_without_improvement == 4


def test_checkpoint_without_the_new_fields_still_loads(tmp_path: Path) -> None:
    """A checkpoint written before this state existed must not fail to load."""
    payload = torch.load(  # build one, then strip the new keys back out
        _write_and_read_payload(tmp_path), map_location="cpu", weights_only=False
    )
    del payload["best_value"], payload["epochs_without_improvement"]
    torch.save(payload, tmp_path / "legacy.pt")

    metadata = load_checkpoint(tmp_path / "legacy.pt", model=_tiny_model())

    assert metadata.best_value is None
    assert metadata.epochs_without_improvement == 0


def _write_and_read_payload(tmp_path: Path) -> Path:
    save_checkpoint(
        tmp_path / "seed.pt", model=_tiny_model(), epoch=0, global_step=0, metrics={"loss": 1.0}
    )
    return tmp_path / "seed.pt"


def test_resumed_manager_does_not_overwrite_best_with_a_worse_model(tmp_path: Path) -> None:
    """The actual bug: a fresh manager's _best_value is None, so the first
    post-resume epoch beats nothing and claims best.pt regardless."""
    model = _tiny_model()
    original = CheckpointManager(tmp_path, monitor="loss", mode="min")
    assert original.save(model=model, epoch=0, global_step=0, metrics={"loss": 0.1}) is True

    resumed = CheckpointManager(tmp_path, monitor="loss", mode="min")
    metadata = load_checkpoint(tmp_path / "last.pt", model=model)
    resumed.restore_best_value(metadata)

    # A clearly worse epoch must not be treated as an improvement.
    assert resumed.save(model=model, epoch=1, global_step=10, metrics={"loss": 5.0}) is False
    assert resumed.best_value == pytest.approx(0.1)

    # ... and a genuinely better one still is.
    assert resumed.save(model=model, epoch=2, global_step=20, metrics={"loss": 0.05}) is True


def test_restore_best_value_falls_back_to_recorded_metrics(tmp_path: Path) -> None:
    """An older checkpoint has no best_value, but its own metrics are a
    better starting bar than none at all."""
    manager = CheckpointManager(tmp_path, monitor="loss", mode="min")
    manager.restore_best_value(
        CheckpointMetadata(epoch=1, global_step=1, metrics={"loss": 0.42})
    )

    assert manager.best_value == pytest.approx(0.42)


def test_restore_best_value_leaves_it_unset_when_nothing_is_known(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path, monitor="loss", mode="min")
    manager.restore_best_value(CheckpointMetadata(epoch=1, global_step=1, metrics={}))

    assert manager.best_value is None


def test_update_best_false_refreshes_last_but_never_best(tmp_path: Path) -> None:
    model = _tiny_model()
    manager = CheckpointManager(tmp_path, monitor="loss", mode="min")

    assert manager.save(model=model, epoch=0, global_step=0, metrics={"loss": 1.0}) is True
    best_mtime = (tmp_path / "best.pt").stat().st_mtime_ns

    # An unmonitored epoch with a spectacular metric still must not claim best.
    assert (
        manager.save(
            model=model, epoch=1, global_step=10, metrics={"loss": 0.0}, update_best=False
        )
        is False
    )

    assert (tmp_path / "last.pt").exists()
    assert (tmp_path / "best.pt").stat().st_mtime_ns == best_mtime
    assert manager.best_value == pytest.approx(1.0)


def test_saved_best_value_reflects_the_bar_that_call_set(tmp_path: Path) -> None:
    model = _tiny_model()
    manager = CheckpointManager(tmp_path, monitor="loss", mode="min")
    manager.save(model=model, epoch=0, global_step=0, metrics={"loss": 0.7})

    metadata = load_checkpoint(tmp_path / "best.pt", model=model)
    assert metadata.best_value == pytest.approx(0.7)


def test_a_resume_cannot_replace_the_true_best_with_a_worse_epoch(tmp_path: Path) -> None:
    """last.pt must record the bar the epoch itself set: a resume restores
    _best_value from it, and a stale bar let 0.007 overwrite a 0.005 best."""
    manager = CheckpointManager(tmp_path, monitor="loss")
    model = _tiny_model()
    manager.save(model=model, epoch=0, global_step=1, metrics={"loss": 0.010})
    assert manager.save(model=model, epoch=1, global_step=2, metrics={"loss": 0.005})

    resumed = CheckpointManager(tmp_path, monitor="loss")
    resumed.restore_best_value(load_checkpoint(tmp_path / "last.pt", model=_tiny_model()))
    assert resumed.save(model=model, epoch=2, global_step=3, metrics={"loss": 0.007}) is False
    assert load_checkpoint(tmp_path / "best.pt", model=_tiny_model()).metrics["loss"] == pytest.approx(0.005)
