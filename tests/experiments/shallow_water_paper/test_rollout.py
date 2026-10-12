"""Tests for the solver's own continuation of a held-out trajectory, the truth an
autoregressive rollout is scored against."""

from __future__ import annotations

import pytest
import torch
from hydra import compose, initialize
from omegaconf import DictConfig

from experiments.shallow_water_paper.data import ScalarShallowWaterDataset
from experiments.shallow_water_paper.rollout import true_trajectory

TINY = [
    "grid.subdivisions=2",
    "device=cpu",
    "seed=0",
    "data.nlat=32",
    "data.nlon=64",
    "data.solver_substeps=1",
]


def _cfg(*overrides: str) -> DictConfig:
    # The single-level L5 arm's bundle primary; only its data group matters here.
    with initialize(version_base=None, config_path="../../../SWE_256x512_ZeroShot/configs"):
        return compose(config_name="gm_l5_100", overrides=[*TINY, *overrides])


def test_the_true_trajectory_extends_the_dataset_pair_hour_by_hour() -> None:
    cfg = _cfg()
    from gm_attention.discretization.grids.icosphere import build_icosphere_grid

    nodes = build_icosphere_grid(2, dtype=torch.float32).nodes
    dataset = ScalarShallowWaterDataset(
        nodes, base_seed=2000, split="val", nlat=32, nlon=64,
        lead_time_seconds=float(cfg.data.lead_time_seconds), solver_substeps=1, num_examples=1,
    )
    states = true_trajectory(dataset, 0, steps=3)
    pair = dataset[0]

    assert len(states) == 4 and all(s.shape == (162, 3) for s in states)
    torch.testing.assert_close(states[0], pair.input)
    torch.testing.assert_close(states[1], pair.target)
    assert not torch.allclose(states[2], states[1]), "hour 2 is a further solver step"
    assert all(torch.isfinite(s).all() for s in states)


def test_truth_composes_the_training_map_with_restarted_solver_history() -> None:
    dataset = ScalarShallowWaterDataset(
        view=(32, 64), base_seed=3000, split="test", nlat=32, nlon=64,
        lead_time_seconds=600.0, solver_substeps=4, num_examples=1,
    )
    _, states = true_trajectory(dataset, 0, steps=3, return_spectral=True)
    initial, first = dataset.source.spectral_pair(0)
    torch.testing.assert_close(states[0], initial, rtol=0, atol=0)
    torch.testing.assert_close(states[1], first, rtol=0, atol=0)
    expected = first
    for k in (2, 3):
        expected = dataset.source.solver.timestep(expected, 4)
        torch.testing.assert_close(states[k], expected, rtol=0, atol=0)
    continuous = dataset.source.solver.timestep(initial, 12)
    assert not torch.allclose(states[3], continuous, rtol=1e-6, atol=0)
    torch.testing.assert_close(states[0], initial, rtol=0, atol=0)
    with pytest.raises(ValueError, match="steps"):
        true_trajectory(dataset, 0, steps=-1)
