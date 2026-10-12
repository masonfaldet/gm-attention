"""Tests for the cross-model spectral comparison.

The arms are the supplement's bundle primaries composed with the tiny
overrides below: the depth-1 GM arm is the ablation's ``gm_attention_full``,
SFNO the zero-shot ``sfno`` row, and the Legendre-Gauss view arm the
zero-shot ``s2_transformer_sl`` row (the shipped arm that reads a
Legendre-Gauss view; the 128x256 benchmark's ``s2_transformer`` reads an
equiangular one).
"""

from __future__ import annotations

import math

import pytest
import torch
from hydra import compose, initialize
from omegaconf import DictConfig

from experiments.shallow_water_paper.compare_spectral import (
    arm_from_experiment,
    check_arms_share_data,
    evaluate_arms,
)
from experiments.shallow_water_paper.spectral import full_bandwidth
from experiments.shallow_water_paper.train import build_scalar_experiment

TINY = [
    "grid.subdivisions=3",
    "device=cpu",
    "seed=0",
    "data.nlat=32",
    "data.nlon=64",
    "data.solver_substeps=1",
    "data.lead_time_seconds=300.0",
    "data.normalize=false",
]

#: Each arm's bundle (relative to the repository root) and primary config.
ARMS = {
    "gm": ("SWE_256x512_Ablation", "gm_attention_full"),
    "sfno": ("SWE_256x512_ZeroShot", "sfno"),
    "lg": ("SWE_256x512_ZeroShot", "s2_transformer_sl"),
}


def _cfg(arm: str, *overrides: str) -> DictConfig:
    bundle, primary = ARMS[arm]
    with initialize(version_base=None, config_path=f"../../../{bundle}/configs"):
        return compose(config_name=primary, overrides=[*TINY, *overrides])


def _arm(label: str, arm: str, *overrides: str):
    cfg = _cfg(arm, *overrides)
    torch.manual_seed(0)
    experiment = build_scalar_experiment(cfg, torch.device("cpu"))
    return arm_from_experiment(label, cfg, experiment, None, torch.device("cpu"))


def test_arms_on_different_grids_are_scored_on_one_band_with_one_coverage() -> None:
    gm = _arm("gm", "gm")
    sfno = _arm("sfno", "sfno", "model.model_view.nlat=32", "model.model_view.nlon=64",
                "model.architecture.embed_dim=8")
    bandwidth = full_bandwidth(32)  # 10: the tiny solver's whole state
    assert bandwidth == 10

    results, coverage = evaluate_arms(
        [gm, sfno], bandwidth=bandwidth, num_examples=2, split="val", device=torch.device("cpu"), steps=2
    )

    assert coverage == pytest.approx(1.0), "the full band holds the whole (band-limited) target"
    assert [r.grid for r in results] == ["icosphere:3", "equiangular:32x64"]
    for r in results:
        assert r.condition < 10
        assert len(r.D) == bandwidth + 1
        assert sum(r.D) == pytest.approx(r.metrics[f"paper_squared_l2_L{bandwidth}"], rel=1e-9)
        assert all(math.isfinite(v) for v in r.metrics.values())
        assert r.parameters > 0
        # Two rollout hours on the same trajectories; hour one IS the
        # single-step evaluation, so its pooled objective must match it.
        assert [step.step for step in r.rollout] == [1, 2]
        band = f"paper_squared_l2_L{bandwidth}"
        assert r.rollout[0].rollout["paper_squared_l2"] == pytest.approx(r.metrics["paper_squared_l2"], rel=1e-6)
        assert r.rollout[0].rollout[band] == pytest.approx(r.metrics[band], rel=1e-6)
        assert r.rollout[0].rollout == r.rollout[0].one_step
        assert sum(r.rollout[1].D) == pytest.approx(r.rollout[1].rollout[band], rel=1e-9)
        assert all(math.isfinite(v) for step in r.rollout for v in step.persistence.values())
        assert all(step.coverage == pytest.approx(1.0) for step in r.rollout)


def test_arms_that_trained_on_different_data_are_refused() -> None:
    gm = _arm("gm", "gm")
    other = _arm("other", "gm", "data.lead_time_seconds=600.0")
    with pytest.raises(ValueError, match="lead_time_seconds"):
        check_arms_share_data([gm, other])


def test_the_trajectory_beyond_hour_one_is_shared_and_layout_aware() -> None:
    """Both layouts of the dataset produce the same normalized truth at
    every hour (the reference view is channel-first), from the same
    spectral states."""
    from experiments.shallow_water_paper.compare_spectral import arm_dataset
    from experiments.shallow_water_paper.data import ScalarNormalization
    from experiments.shallow_water_paper.rollout import true_trajectory

    normalization = ScalarNormalization(
        mean=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64), std=torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64),
        num_examples=1, calibration_seed=0,
    )
    gm = _arm("gm", "gm"); gm.normalization = normalization
    sfno = _arm("sfno", "sfno", "model.model_view.nlat=32", "model.model_view.nlon=64",
                "model.architecture.embed_dim=8"); sfno.normalization = normalization
    cpu = torch.device("cpu")
    by_nodes = arm_dataset(gm, num_examples=1, split="val", device=cpu)
    by_view = arm_dataset(sfno, num_examples=1, split="val", device=cpu)
    fields_n, spectral_n = true_trajectory(by_nodes, 0, 3, return_spectral=True)
    fields_v, spectral_v = true_trajectory(by_view, 0, 3, return_spectral=True)
    assert len(fields_n) == len(fields_v) == 4
    for a, b in zip(spectral_n, spectral_v, strict=True):
        torch.testing.assert_close(a, b)
    assert not torch.allclose(spectral_n[3], spectral_n[2]), "hour 3 is a further solver step"
    # The view's channel-first truth, normalized channel-first, matches the
    # icosphere's point-major truth in scale: both are z-scores of one state.
    assert fields_v[3].shape == (3, 32, 64) and fields_n[3].shape[-1] == 3
    a, b = float(fields_v[3][0].mean()), float(fields_n[3][:, 0].mean())
    # Unweighted means over different point sets differ at the 1e-4 level;
    # an un-normalized view would be off by the whole z-score offset.
    assert abs(a - b) < 1e-3 * abs(b)


@pytest.mark.parametrize("arm, overrides", [
    ("gm", ()),
    ("sfno", ("model.model_view.nlat=32", "model.model_view.nlon=64", "model.architecture.embed_dim=8")),
])
def test_the_zero_mean_update_removes_exactly_the_degree_zero_error(arm: str, overrides: tuple[str, ...]) -> None:
    """The constrained twin of an arm keeps the input's global mean per
    channel, so at one step its band-limited error is the unconstrained
    arm's minus its D(0), and nothing above degree 0 moves."""
    from experiments.shallow_water_paper.compare_spectral import step_native

    free = _arm("free", arm, *overrides)
    twin = _arm("twin", arm, *overrides)
    twin.zero_mean_update = True
    assert all(torch.equal(a, b) for a, b in zip(free.experiment.model.parameters(), twin.experiment.model.parameters()))
    bandwidth = full_bandwidth(32)

    results, _ = evaluate_arms(
        [free, twin], bandwidth=bandwidth, num_examples=2, split="val", device=torch.device("cpu"), steps=0
    )
    band = f"paper_squared_l2_L{bandwidth}"
    free_r, twin_r = results
    assert twin_r.zero_mean_update and not free_r.zero_mean_update
    assert free_r.D[0] > 0
    # what survives at degree 0 is the least-squares fit leaking out-of-band
    # content (none on a quadrature grid, ~1e-3 of that energy on the icosphere)
    assert twin_r.D[0] < 1e-4 * free_r.D[0]
    out_of_band = twin_r.metrics["paper_squared_l2"] - twin_r.metrics[band]
    assert twin_r.D[0] < 1e-2 * abs(out_of_band) + 1e-9 * twin_r.metrics[band]
    assert twin_r.D[1:] == pytest.approx(free_r.D[1:], rel=1e-6, abs=1e-9 * free_r.metrics[band])
    assert twin_r.metrics[band] == pytest.approx(free_r.metrics[band] - free_r.D[0] + twin_r.D[0], rel=1e-6)
    assert twin_r.metrics["paper_squared_l2"] < free_r.metrics["paper_squared_l2"]

    # the constraint itself: the update's quadrature mean vanishes per channel
    from experiments.shallow_water_paper.compare_spectral import (
        arm_dataset,
        point_major,
    )

    state = arm_dataset(twin, num_examples=1, split="val", device=torch.device("cpu"))[0]
    update = point_major(twin, step_native(twin, state.input) - state.input)
    weights = twin.grid.quadrature_weights.reshape(-1)
    mean_update = (update * weights.unsqueeze(-1)).sum(dim=0) / weights.sum()
    # float32 roundoff of prediction - input, relative to the (raw, ~1e5) state
    scale = max(float(update.abs().max()), float(state.input.abs().max()))
    assert torch.allclose(mean_update, torch.zeros(3), atol=1e-6 * scale)


@pytest.mark.parametrize("enabled", [False, True])
def test_zero_mean_option_survives_arm_construction(enabled: bool) -> None:
    cfg = _cfg("gm")
    experiment = build_scalar_experiment(cfg, torch.device("cpu"))
    arm = arm_from_experiment("gm", cfg, experiment, None, torch.device("cpu"), zero_mean_update=enabled)
    assert arm.zero_mean_update is enabled


def test_normalization_comparison_uses_the_error_scale() -> None:
    from experiments.shallow_water_paper.data import ScalarNormalization

    a = _arm("a", "gm")
    b = _arm("b", "gm")
    mean = torch.tensor([98000.0, 0.0, 0.0], dtype=torch.float64)
    std = torch.tensor([100.0, 1e-6, 1e-6], dtype=torch.float64)
    a.normalization = ScalarNormalization(mean, std, 128, 4000)
    for other_mean, other_std in ((mean, std * torch.tensor([1., 1.005, 1.])),
                                  (mean + torch.tensor([0.05, 0., 0.]), std)):
        b.normalization = ScalarNormalization(other_mean, other_std, 128, 4000)
        with pytest.raises(ValueError, match="normalization differs"):
            check_arms_share_data([a, b])


def test_without_teacher_forcing_the_rollout_is_unchanged_and_the_forced_rows_are_empty() -> None:
    arm = _arm("a", "gm")
    with_forcing, _ = evaluate_arms([arm], bandwidth=4, num_examples=2, split="val", device=torch.device("cpu"), steps=2)
    without, _ = evaluate_arms([arm], bandwidth=4, num_examples=2, split="val", device=torch.device("cpu"), steps=2,
                               teacher_forcing=False)
    for a, b in zip(with_forcing[0].rollout, without[0].rollout, strict=True):
        assert b.one_step == {} and a.one_step
        assert b.rollout == pytest.approx(a.rollout) and b.persistence == pytest.approx(a.persistence)
        assert b.D == pytest.approx(a.D)


def test_a_legendre_gauss_view_arm_is_labelled_and_scored_on_that_grid() -> None:
    """On the zero-shot single-level S2 Transformer, the shipped arm that
    reads a Legendre-Gauss view (the released model at scale factor 1; the
    scale factor is beside the point here)."""
    from experiments.shallow_water_paper.compare_spectral import training_grid_of

    cfg = _cfg("lg", "model.model_view.nlat=32", "model.model_view.nlon=64")
    assert training_grid_of(cfg) == "legendre-gauss:32x64"
    experiment = build_scalar_experiment(cfg, torch.device("cpu"))
    arm = arm_from_experiment("lg", cfg, experiment, None, torch.device("cpu"))
    assert arm.grid.n_nodes == 32 * 64
    results, _ = evaluate_arms([arm], bandwidth=6, num_examples=1, split="val", device=torch.device("cpu"))
    assert results[0].grid == "legendre-gauss:32x64" and results[0].training_grid == "legendre-gauss:32x64"
    # the residual is the prediction's own out-of-band content, small for an untrained model's smooth output
    assert 0.0 <= results[0].metrics["fit_residual_fraction_L6"] < 0.05
