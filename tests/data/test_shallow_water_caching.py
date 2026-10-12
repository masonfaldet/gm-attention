"""Tests for ShallowWaterDataset's on-disk spectral cache.

It is an optimization, so the only thing that matters is that it cannot
change what a run trains on. Each test below is a way that could happen.
"""

from __future__ import annotations

import torch

from gm_attention.data.shallow_water import ShallowWaterDataset

SOLVER = {
    "nlat": 32, "nlon": 64, "lead_time_seconds": 600.0,
    "solver_substeps": 1, "num_examples": 3, "dtype": torch.float32,
}


def _dataset(tmp_path=None, split="val", policy="fixed_index", **kwargs):
    return ShallowWaterDataset(
        base_seed=7, split=split,
        sample_policy=policy,
        spectral_cache_dir=tmp_path, **SOLVER, **kwargs,
    )


# ==========================================================================
# On-disk spectral cache
# ==========================================================================


def test_the_spectral_cache_reproduces_the_uncached_sample(tmp_path) -> None:
    """The claim the whole optimization rests on. A cached trajectory must
    be bit-identical to a solved one, or runs sharing a cache are training
    on different data than runs without it."""
    uncached = _dataset().spectral_pair(0)

    writer = _dataset(tmp_path=tmp_path)
    written = writer.spectral_pair(0)
    reader = _dataset(tmp_path=tmp_path)
    read_back = reader.spectral_pair(0)

    for a, b in zip(uncached, written, strict=True):
        assert torch.equal(a, b)
    for a, b in zip(uncached, read_back, strict=True):
        assert torch.equal(a, b)


def test_the_cache_is_actually_populated(tmp_path) -> None:
    dataset = _dataset(tmp_path=tmp_path)
    for index in range(len(dataset)):
        dataset.spectral_pair(index)

    entries = list(tmp_path.rglob("*.pt"))

    assert len(entries) == len(dataset)
    assert not list(tmp_path.rglob("*.tmp")), "temporaries must be renamed away"


def test_the_key_separates_different_solver_settings(tmp_path) -> None:
    """A collision here would train a model on another physics' trajectories
    while looking entirely normal."""
    base = _dataset(tmp_path=tmp_path)
    other = ShallowWaterDataset(
        base_seed=7, split="val",
        sample_policy="fixed_index", spectral_cache_dir=tmp_path,
        **{**SOLVER, "solver_substeps": 2},
    )

    assert base._spectral_cache_path(0) != other._spectral_cache_path(0)


def test_the_key_separates_precisions(tmp_path) -> None:
    """Reading a float64 entry into a float32 run, or the reverse, would
    change the trajectory's precision without changing anything visible."""
    single = _dataset(tmp_path=tmp_path)
    double = ShallowWaterDataset(
        base_seed=7, split="val",
        sample_policy="fixed_index", spectral_cache_dir=tmp_path,
        **{**SOLVER, "dtype": torch.float64},
    )

    assert single._spectral_cache_path(0) != double._spectral_cache_path(0)


def test_the_key_separates_splits_and_epochs(tmp_path) -> None:
    """sample_seed folds in split and epoch, so entries cannot cross."""
    train = _dataset(tmp_path=tmp_path, split="train", policy="epoch_index")
    val = _dataset(tmp_path=tmp_path)

    assert train._spectral_cache_path(0) != val._spectral_cache_path(0)

    epoch0 = train._spectral_cache_path(0)
    train.set_epoch(1)

    assert train._spectral_cache_path(0) != epoch0


def test_a_corrupt_entry_falls_back_to_solving(tmp_path) -> None:
    """A partial file from a concurrent writer must not end a week-long
    run; regenerating is always correct."""
    dataset = _dataset(tmp_path=tmp_path)
    expected = dataset.spectral_pair(0)

    entry = dataset._spectral_cache_path(0)
    entry.write_bytes(b"not a torch file")

    fresh = _dataset(tmp_path=tmp_path)
    recovered = fresh.spectral_pair(0)

    for a, b in zip(expected, recovered, strict=True):
        assert torch.equal(a, b)


def test_an_unwritable_cache_directory_does_not_stop_the_run(tmp_path) -> None:
    """A full or read-only filesystem is not a reason to lose a run."""
    blocked = tmp_path / "blocked"
    blocked.write_text("this is a file, not a directory")

    dataset = _dataset(tmp_path=blocked)
    sample = dataset.spectral_pair(0)

    assert torch.isfinite(sample[0]).all()


def test_no_cache_directory_means_no_files_anywhere(tmp_path) -> None:
    """The default must stay entirely off."""
    dataset = _dataset()
    dataset.spectral_pair(0)

    assert dataset.spectral_cache_dir is None
    assert not list(tmp_path.rglob("*"))


def test_changing_lead_time_cannot_reuse_a_cached_target(tmp_path) -> None:
    writer = _dataset(tmp_path=tmp_path)
    writer.spectral_pair(0)
    common = dict(base_seed=7, split="val",
                  **{**SOLVER, "lead_time_seconds": 300.0})
    reader = ShallowWaterDataset(spectral_cache_dir=tmp_path, **common)
    uncached = ShallowWaterDataset(**common)
    assert writer._spectral_cache_path(0) != reader._spectral_cache_path(0)
    for cached, fresh in zip(reader.spectral_pair(0), uncached.spectral_pair(0), strict=True):
        torch.testing.assert_close(cached, fresh, rtol=0, atol=0)
