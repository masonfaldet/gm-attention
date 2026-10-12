"""Tests for the reference models on the paper's three-scalar state.

Configs compose from the bundle primaries: ``SWE_128x256_Benchmark/configs``
for the released S^2 Transformer on its equiangular view and
``SWE_256x512_ZeroShot/configs`` for SFNO and the single-level S^2
Transformer on its Legendre-Gauss view.
"""

from __future__ import annotations

import math

import pytest
import torch
from hydra import compose, initialize
from omegaconf import DictConfig

from experiments.shallow_water_paper.data import (
    ScalarNormalization,
    ScalarShallowWaterDataset,
)
from experiments.shallow_water_paper.metrics import ScalarStateMetrics
from experiments.shallow_water_paper.train import (
    build_datasets,
    build_loaders,
    build_normalization,
    build_protocol,
    build_scalar_experiment,
    real_parameter_count,
)

BENCHMARK = "../../../SWE_128x256_Benchmark/configs"
ZERO_SHOT = "../../../SWE_256x512_ZeroShot/configs"

TINY = [
    "grid.subdivisions=2",
    "device=cpu",
    "seed=0",
    "training.batch_size=2",
    "data.nlat=32",
    "data.nlon=64",
    "data.solver_substeps=1",
    "data.num_train_examples=4",
    "data.num_val_examples=2",
    "data.normalization_examples=1",
]


def _cfg(config_path: str, config_name: str, *overrides: str) -> DictConfig:
    with initialize(version_base=None, config_path=config_path):
        return compose(config_name=config_name, overrides=[*TINY, *overrides])


def test_the_reference_arm_builds_batches_and_steps_on_the_paper_state() -> None:
    """A tensor-grid arm through the paper entry point: the equiangular
    view, channel-first data, batched loaders with no accumulation, and a
    finite train step with the same metric names. The released S^2
    Transformer (``s2_transformer``, the nearest shipped equiangular-view
    arm) stands in for the development repository's two-head arm."""
    cfg = _cfg(BENCHMARK, "s2_transformer", "model.model_view.nlat=32", "model.model_view.nlon=64")
    device = torch.device("cpu")
    experiment = build_scalar_experiment(cfg, device)

    assert experiment.batched and experiment.view == (32, 64) and experiment.nodes is None
    assert experiment.n_nodes == 32 * 64

    # Normalized as every real run is: raw geopotential is ~1e5 and the
    # reference's attention overflows on it, which is the normalization's job.
    normalization = build_normalization(cfg, device)
    train, val = build_datasets(cfg, experiment, device, normalization)
    sample = val[0]
    assert sample.input.shape == (3, 32, 64) and sample.target.shape == (3, 32, 64)

    train_loader, val_loader, accumulation = build_loaders(cfg, train, val, batched=experiment.batched)
    batch = next(iter(train_loader))
    assert accumulation == 1
    assert batch.input.shape == (2, 3, 32, 64)

    experiment.model.train()
    output = experiment.step_fn(experiment.model, batch, device)
    output.loss.backward()
    assert math.isfinite(float(output.loss))
    assert {"paper_squared_l2", "vorticity_rmse", "divergence_rmse", "geopotential_rmse"} <= output.metrics.keys()
    assert all(math.isfinite(v) for v in output.metrics.values())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in experiment.model.parameters())

    protocol = build_protocol(cfg, normalization, experiment)
    assert protocol["model_view"] == [32, 64] and protocol["grid_subdivisions"] is None
    assert protocol["grid_n_nodes"] == 32 * 64


def test_the_view_synthesis_is_the_node_synthesis_on_the_same_points() -> None:
    """The channel-first view and the point-major synthesis are two layouts
    of one spectral state: at the solver's own grid they agree exactly."""
    from gm_attention.discretization.grids.torch_harmonics import (
        build_torch_harmonics_grid,
    )

    grid = build_torch_harmonics_grid("equiangular", 32, 64, dtype=torch.float32)
    common = dict(base_seed=2000, split="val", nlat=32, nlon=64, lead_time_seconds=300.0,
                  solver_substeps=1, num_examples=1)
    by_view = ScalarShallowWaterDataset(view=(32, 64), **common)[0]
    by_nodes = ScalarShallowWaterDataset(grid.nodes, **common)[0]

    torch.testing.assert_close(
        by_view.input.movedim(0, -1).reshape(-1, 3), by_nodes.input, rtol=1e-4, atol=1e-4
    )


def test_channel_first_normalization_matches_point_major() -> None:
    normalization = ScalarNormalization(
        mean=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
        std=torch.tensor([2.0, 4.0, 8.0], dtype=torch.float64),
        num_examples=1, calibration_seed=0,
    )
    field = torch.randn(3, 4, 6)
    torch.testing.assert_close(
        normalization.normalize_channel_first(field),
        normalization.normalize(field.movedim(0, -1)).movedim(-1, 0),
    )


def test_metrics_report_scalars_for_a_batch() -> None:
    metrics = ScalarStateMetrics(torch.ones(10))
    out = metrics(torch.randn(2, 10, 3), torch.randn(2, 10, 3))
    assert all(value.ndim == 0 for value in out.values())


def test_the_sfno_arm_runs_on_the_paper_state_through_the_same_step() -> None:
    """kind=sfno takes the tensor-grid path: same view, same batched
    loaders, same Coriolis concatenation and residual, same metrics."""
    cfg = _cfg(ZERO_SHOT, "sfno", "model.model_view.nlat=32", "model.model_view.nlon=64")
    device = torch.device("cpu")
    experiment = build_scalar_experiment(cfg, device)
    assert experiment.batched and experiment.view == (32, 64)

    normalization = build_normalization(cfg, device)
    train, val = build_datasets(cfg, experiment, device, normalization)
    loader, _, accumulation = build_loaders(cfg, train, val, batched=True)
    batch = next(iter(loader))
    experiment.model.train()
    output = experiment.step_fn(experiment.model, batch, device)
    output.loss.backward()
    assert accumulation == 1 and math.isfinite(float(output.loss))
    assert "divergence_rmse" in output.metrics


def test_the_paper_sfno_is_parameter_matched_at_the_full_resolution_view() -> None:
    import hydra

    cfg = _cfg(ZERO_SHOT, "sfno")
    a = cfg.model.architecture
    assert (a.nlat, a.nlon, a.embed_dim, a.num_layers, a.scale_factor) == (256, 512, 22, 4, 2)
    assert a.normalization_layer == "instance_norm"
    model = hydra.utils.instantiate(a)
    # Matched in real scalars: the spectral weights are complex, and numel
    # would count each of them once.
    assert any(p.is_complex() for p in model.parameters())
    assert real_parameter_count(model) == 503_866


def test_a_legendre_gauss_view_is_the_node_synthesis_on_the_legendre_gauss_points() -> None:
    from gm_attention.discretization.grids.torch_harmonics import (
        build_torch_harmonics_grid,
    )

    grid = build_torch_harmonics_grid("legendre-gauss", 32, 64, dtype=torch.float32)
    common = dict(base_seed=2000, split="val", nlat=32, nlon=64, lead_time_seconds=300.0, solver_substeps=1, num_examples=1)
    by_view = ScalarShallowWaterDataset(view=(32, 64), view_grid="legendre-gauss", **common)[0]
    by_nodes = ScalarShallowWaterDataset(grid.nodes, **common)[0]
    torch.testing.assert_close(by_view.input.movedim(0, -1).reshape(-1, 3), by_nodes.input, rtol=1e-4, atol=1e-4)
    equiangular = ScalarShallowWaterDataset(view=(32, 64), **common)[0]
    assert not torch.allclose(equiangular.input, by_view.input)  # different points


def test_the_released_s2_transformer_on_a_legendre_gauss_view_builds_steps_and_records_its_grid(tmp_path) -> None:
    """On a Legendre-Gauss view the quadrature, Coriolis map, synthesis and
    the model's own grid all follow the view's family. The single-level
    S^2 Transformer (``s2_transformer_sl``) is the shipped arm on that
    view; the development repository also had the scale-factor-2 variant,
    which this property does not depend on."""

    from experiments.shallow_water_paper.train import (
        build_datasets,
        build_protocol,
        resolve_view_grid,
    )

    cfg = _cfg(ZERO_SHOT, "s2_transformer_sl", "model.model_view.nlat=32", "model.model_view.nlon=64")
    assert resolve_view_grid(cfg) == "legendre-gauss" and cfg.model.architecture.grid == "legendre-gauss"
    experiment = build_scalar_experiment(cfg, torch.device("cpu"))
    assert experiment.view == (32, 64) and experiment.view_grid == "legendre-gauss"
    from experiments.shallow_water_paper.data import ScalarState

    train, val = build_datasets(cfg, experiment, torch.device("cpu"), None)
    assert train.view_grid == "legendre-gauss" and val.channel_first
    batch = val[0]
    output = experiment.step_fn(experiment.model, ScalarState(batch.input.unsqueeze(0), batch.target.unsqueeze(0)), torch.device("cpu"))
    assert torch.isfinite(output.loss)
    protocol = build_protocol(cfg, None, experiment)
    assert protocol["model_view"] == [32, 64] and protocol["model_view_grid"] == "legendre-gauss"


def test_micro_batches_accumulate_to_the_batch_for_a_tensor_kind() -> None:
    from experiments.shallow_water_paper.train import build_datasets, build_loaders

    cfg = _cfg(ZERO_SHOT, "s2_transformer_sl", "model.model_view.nlat=32", "model.model_view.nlon=64",
               "training.batch_size=4", "training.micro_batch_size=2")
    assert cfg.model.architecture.scale_factor == 1
    experiment = build_scalar_experiment(cfg, torch.device("cpu"))
    train, val = build_datasets(cfg, experiment, torch.device("cpu"), None)
    train_loader, val_loader, accumulation = build_loaders(cfg, train, val, batched=True)
    assert (train_loader.batch_size, val_loader.batch_size, accumulation) == (2, 2, 2)
    # The bundle primary trains this arm in micro-batches of 2; null is the default's.
    plain = _cfg(ZERO_SHOT, "s2_transformer_sl", "model.model_view.nlat=32", "model.model_view.nlon=64",
                 "training.micro_batch_size=null")
    assert build_loaders(plain, train, val, batched=True)[2] == 1  # null: batched normally
    cfg.training.micro_batch_size = 3
    with pytest.raises(ValueError, match="must divide"):
        build_loaders(cfg, train, val, batched=True)


def test_the_single_level_s2_transformer_keeps_the_released_parameter_count() -> None:
    from hydra import compose, initialize

    from experiments.shallow_water_paper.train import (
        build_scalar_experiment,
        real_parameter_count,
    )

    with initialize(version_base=None, config_path=ZERO_SHOT):
        cfg = compose(config_name="s2_transformer_sl", overrides=["device=cpu"])
    experiment = build_scalar_experiment(cfg, torch.device("cpu"))
    assert experiment.view == (128, 256) and experiment.view_grid == "legendre-gauss"
    assert experiment.model.scale_factor == 1 and (experiment.model.h, experiment.model.w) == (128, 256)
    assert real_parameter_count(experiment.model) == 535_424


def test_the_released_sfno_width_counts_as_the_benchmark_readme_says() -> None:
    """embed dim 32, the registry's own width (the 128x256 benchmark's
    ``sfno`` arm), on the 256x512 view: 541,408 by numel (complex spectral
    weights counted once) and 1,065,696 real scalars."""
    from experiments.shallow_water_paper.train import (
        build_scalar_experiment,
        real_parameter_count,
    )

    cfg = _cfg(ZERO_SHOT, "sfno", "model.architecture.embed_dim=32")
    model = build_scalar_experiment(cfg, torch.device("cpu")).model
    assert sum(p.numel() for p in model.parameters()) == 541_408
    assert real_parameter_count(model) == 1_065_696
