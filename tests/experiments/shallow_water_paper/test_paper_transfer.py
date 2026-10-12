"""Tests for the paper-state transfer probe: rebuilding a fitted run's model and
its geometry on a grid it was not trained on, and predicting there.

The supplement ships the probe's building blocks (``build_model``,
``geometry_for_config``, ``predict``) rather than its row scorer, so the
checks here are on those."""

from __future__ import annotations

import pytest
import torch
from hydra import compose, initialize
from omegaconf import DictConfig

from experiments.shallow_water.grid_transfer import GridSpec
from experiments.shallow_water_paper.data import CHANNELS, ScalarShallowWaterDataset
from experiments.shallow_water_paper.train import (
    CONDITIONED_FIELD_TYPE,
    STATE_FIELD_TYPE,
    resolve_level,
)
from experiments.shallow_water_paper.transfer import (
    FittedRun,
    build_model,
    geometry_for_config,
    predict,
)
from gm_attention.data.shallow_water import coriolis_channel

TINY = [
    "grid.subdivisions=2",
    "device=cpu",
    "seed=0",
    "data.nlat=32",
    "data.nlon=64",
    "data.solver_substeps=1",
    "data.num_val_examples=2",
]

# The bundle primaries of the two GM arms the probe transfers: the single-level
# L5 arm (SWE_256x512_ZeroShot/gm_l5_100) and the depth-1 arm
# (SWE_256x512_Ablation/gm_attention_full).
SINGLE_LEVEL = ("../../../SWE_256x512_ZeroShot/configs", "gm_l5_100")
DEPTH1 = ("../../../SWE_256x512_Ablation/configs", "gm_attention_full")


def _cfg(bundle: tuple[str, str], *overrides: str) -> DictConfig:
    config_path, primary = bundle
    with initialize(version_base=None, config_path=config_path):
        return compose(config_name=primary, overrides=[*TINY, *overrides])


def _run(bundle: tuple[str, str], *overrides: str) -> FittedRun:
    cfg = _cfg(bundle, *overrides)
    torch.manual_seed(0)
    model = build_model(cfg, chunk_size=17, device=torch.device("cpu"))
    assert model.input_field_type == CONDITIONED_FIELD_TYPE
    assert model.output_field_type == STATE_FIELD_TYPE
    return FittedRun(
        cfg=cfg, kind=str(cfg.model.kind), native_level=resolve_level(cfg),
        normalization=None, model=model,
    )


def test_a_depth1_arm_transfers_within_the_icosphere_family_only() -> None:
    """The depth-1 chain rebuilt on a finer icosphere predicts there; a tensor
    grid is refused. The row scorer's mesh ratio and metrics are not shipped,
    so the finer grid's prediction stands in for its scored row."""
    run = _run(DEPTH1)
    assert not run.single_level
    cpu = torch.device("cpu")

    grid, _, geometry, _ = geometry_for_config(run.cfg, run.kind, run.native_level, GridSpec.parse("icosphere:3"), cpu)
    assert grid.n_nodes == 642
    dataset = ScalarShallowWaterDataset(
        grid.nodes, base_seed=int(run.cfg.data.val_seed), split="val", normalization=run.normalization,
        nlat=int(run.cfg.data.nlat), nlon=int(run.cfg.data.nlon),
        lead_time_seconds=float(run.cfg.data.lead_time_seconds), solver_substeps=int(run.cfg.data.solver_substeps),
        mach=float(run.cfg.data.mach), omega=float(run.cfg.data.omega), num_examples=1, device=cpu,
    )
    prediction = predict(run, geometry, coriolis_channel(grid.nodes).to(cpu), dataset[0])
    assert prediction.shape == (642, len(CHANNELS)) and torch.isfinite(prediction).all()

    with pytest.raises(TypeError, match="icosphere family"):
        geometry_for_config(run.cfg, run.kind, run.native_level, GridSpec.parse("legendre-gauss:8x16"), cpu)


def test_an_omitted_chunk_size_keeps_the_runs_own() -> None:
    from hydra import compose, initialize

    from experiments.shallow_water_paper.transfer import build_model

    # gm_l5_100 trains the single-level arm at chunk_size 2048, its launch setting.
    with initialize(version_base=None, config_path="../../../SWE_256x512_ZeroShot/configs"):
        cfg = compose(config_name="gm_l5_100", overrides=["device=cpu"])
    assert int(cfg.model.architecture.chunk_size) == 2048
    assert build_model(cfg, None, torch.device("cpu")).chunk_size == 2048
    assert build_model(cfg, 64, torch.device("cpu")).chunk_size == 64
