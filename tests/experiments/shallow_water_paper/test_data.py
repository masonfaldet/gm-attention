"""Tests for the paper-state view of the shallow-water data."""

from __future__ import annotations

from typing import Any

import torch
from torch_harmonics.examples.losses import get_quadrature_weights

from experiments.shallow_water_paper.data import (
    CHANNELS,
    ScalarNormalization,
    ScalarShallowWaterDataset,
    calibrate_scalar_normalization,
)
from gm_attention.discretization.grids.torch_harmonics import build_torch_harmonics_grid

NLAT, NLON = 16, 32
SOLVER: dict[str, Any] = {"nlat": NLAT, "nlon": NLON, "lead_time_seconds": 300.0, "solver_substeps": 2}


def _dataset(split: str = "val", seed: int = 7, **overrides):
    nodes = build_torch_harmonics_grid("equiangular", NLAT, NLON, dtype=torch.float64).nodes
    kwargs: dict[str, Any] = {"base_seed": seed, "split": split, "num_examples": 3, "dtype": torch.float64, **SOLVER}
    kwargs.update(overrides)
    return ScalarShallowWaterDataset(nodes, **kwargs)


def test_the_three_channels_are_the_solvers_own_spec2grid_of_the_whole_state() -> None:
    """What PdeDataset returns and the paper trains on: the scalar inverse
    transform applied to (geopotential, vorticity, divergence) alike --
    checked at the solver's own grid points, where spec2grid is exact."""
    dataset = _dataset()
    input_spectral, target_spectral = dataset.source.spectral_pair(1)
    state = dataset[1]

    expected_input = dataset.source.solver.spec2grid(input_spectral).reshape(3, -1).T
    expected_target = dataset.source.solver.spec2grid(target_spectral).reshape(3, -1).T

    assert state.input.shape == (NLAT * NLON, 3)
    torch.testing.assert_close(state.input, expected_input, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(state.target, expected_target, rtol=1e-6, atol=1e-6)
    assert torch.isfinite(state.input).all() and torch.isfinite(state.target).all()


def test_the_view_shares_the_main_benchmarks_trajectories() -> None:
    """Same seeds, same solve: the source's own (h, u, v) sample and this
    view are two syntheses of one spectral pair, so height agrees with the
    geopotential channel up to the solver's own convention."""
    dataset = _dataset()
    spectral, _ = dataset.source.spectral_pair(0)
    height = dataset.source.solver.spec2grid(spectral[:1]).reshape(-1)

    torch.testing.assert_close(dataset[0].input[:, 0], height, rtol=1e-6, atol=1e-6)


def test_fixed_index_is_deterministic_and_epoch_index_moves_with_the_epoch() -> None:
    fixed = _dataset()
    torch.testing.assert_close(fixed[0].input, fixed[0].input)

    moving = _dataset(split="train", sample_policy="epoch_index")
    first = moving[0].input.clone()
    moving.set_epoch(1)
    assert not torch.allclose(moving[0].input, first)
    assert moving.sample_seed(0) != fixed.sample_seed(0)


def test_calibration_standardizes_every_channel_separately() -> None:
    """Area-weighted per-channel z-scores over the calibration set itself:
    mean zero and unit variance for each of the three, evaluated on the
    same equiangular points the calibration integrated over."""
    normalization = calibrate_scalar_normalization(
        seed=11, num_examples=3, dtype=torch.float64, **SOLVER
    )
    assert normalization.mean.shape == normalization.std.shape == (3,)
    assert (normalization.std > 0).all()
    # Geopotential is O(1e5) with a small spread; vorticity and divergence
    # are O(1e-5): three genuinely different scales, which is the point.
    assert normalization.mean[0] > 1e3 * normalization.mean[1:].abs().max()

    dataset = _dataset(split="normalization", seed=11, normalization=normalization)
    weights = get_quadrature_weights(NLAT, NLON, grid="equiangular", tile=True, normalized=True)
    weights = weights.reshape(-1).to(torch.float64)
    stacked = torch.stack([dataset[i].input for i in range(3)])  # (3, n, 3)
    mean = (weights[:, None] * stacked).sum(dim=1).mean(dim=0)
    variance = (weights[:, None] * stacked.square()).sum(dim=1).mean(dim=0) - mean.square()

    torch.testing.assert_close(mean, torch.zeros(3, dtype=torch.float64), atol=1e-6, rtol=0)
    torch.testing.assert_close(variance, torch.ones(3, dtype=torch.float64), atol=1e-6, rtol=0)


def test_normalization_round_trips_through_its_state_dict() -> None:
    normalization = ScalarNormalization(
        mean=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
        std=torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64),
        num_examples=2,
        calibration_seed=9,
    )
    rebuilt = ScalarNormalization.from_state_dict(normalization.state_dict())
    field = torch.randn(5, 3, dtype=torch.float64)

    assert normalization.state_dict()["channels"] == list(CHANNELS)
    torch.testing.assert_close(rebuilt.normalize(field), normalization.normalize(field))
    torch.testing.assert_close(normalization.denormalize(normalization.normalize(field)), field)
