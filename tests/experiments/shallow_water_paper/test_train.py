"""Tests for the paper-state experiment's config, model wiring, step, and loop.

The configs are the bundle primaries: the ablation's ``gm_attention_full``
(model group ``d1_attention_full``, the depth-1 arm) and the zero-shot
bundle's ``gm_l5_100`` (model group ``sl_attention``, the single-level L5
arm), composed with the tiny overrides below.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize
from omegaconf import DictConfig

from experiments.shallow_water_paper.data import CHANNELS, ScalarState
from experiments.shallow_water_paper.metrics import MONITOR_METRIC, ScalarStateMetrics
from experiments.shallow_water_paper.train import (
    CONDITIONED_FIELD_TYPE,
    PROTOCOL_FILENAME,
    STATE_FIELD_TYPE,
    build_scalar_experiment,
)
from experiments.shallow_water_paper.train import main as train_main
from gm_attention.representations.field_type import SO2FeatureField, scale_field_type

#: Small enough for CPU seconds, while exercising the real path.
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


#: The depth-1 arm: the ablation bundle's primary (model group d1_attention_full).
DEPTH1 = ("../../../SWE_256x512_Ablation/configs", "gm_attention_full")
#: The single-level L5 arm: the zero-shot bundle's primary (model group sl_attention).
SINGLE_LEVEL = ("../../../SWE_256x512_ZeroShot/configs", "gm_l5_100")


def _cfg(*overrides: str, bundle: tuple[str, str] = DEPTH1) -> DictConfig:
    # The bundle primaries draw their groups through ``hydra.searchpath:
    # [file://configs]``, resolved against the working directory: compose
    # from the repository root.
    config_path, config_name = bundle
    with initialize(version_base=None, config_path=config_path):
        return compose(config_name=config_name, overrides=[*TINY, *overrides])


def test_the_paper_config_composes_the_main_benchmarks_si_ds_arm_unchanged() -> None:
    cfg = _cfg()
    architecture = cfg.model.architecture

    assert architecture._target_.endswith("GmAttentionDepth1")
    assert architecture.window == "hard"
    assert architecture.nonlinearity == "scalar_gate_double_sigmoid"
    assert architecture.decode_self_interaction == "linear"
    assert "input_field_type" not in architecture, "the types are injected by train.py"
    assert cfg.objective.residual is True
    assert cfg.training.scheduler.kind == "plateau"
    assert cfg.data.train_seed == 1000 and cfg.data.val_seed == 2000


def test_the_model_takes_four_scalars_in_and_three_out_with_the_arms_hidden_type() -> None:
    cfg = _cfg()
    experiment = build_scalar_experiment(cfg, torch.device("cpu"))
    model = experiment.model

    assert model.input_field_type == CONDITIONED_FIELD_TYPE
    assert model.output_field_type == STATE_FIELD_TYPE
    assert model.hidden_field_type == scale_field_type(
        cfg.model.architecture.base_field_type, cfg.model.architecture.multiplicity
    )


def test_the_step_scores_the_papers_loss_on_the_residual_reconstruction() -> None:
    cfg = _cfg()
    experiment = build_scalar_experiment(cfg, torch.device("cpu"))
    model, geometry, step = experiment.model, experiment.geometry, experiment.step_fn
    n = experiment.grid.n_nodes
    generator = torch.Generator().manual_seed(3)
    sample = ScalarState(
        torch.randn(n, 3, generator=generator), torch.randn(n, 3, generator=generator)
    )

    model.eval()
    with torch.no_grad():
        output = step(model, sample, torch.device("cpu"))

        # The same computation by hand: Coriolis appended at frequency 0,
        # the three-channel update added to the three-channel input.
        from gm_attention.data.shallow_water import coriolis_channel

        coriolis = coriolis_channel(experiment.grid.nodes)
        conditioned = SO2FeatureField(
            CONDITIONED_FIELD_TYPE, {0: torch.cat([sample.input, coriolis[:, None]], dim=-1)}
        )
        update = model(conditioned, geometry)[0]
        expected = ScalarStateMetrics(experiment.grid.quadrature_weights)(
            sample.input + update, sample.target
        )

    assert torch.isfinite(output.loss)
    torch.testing.assert_close(output.loss, expected[MONITOR_METRIC])
    assert {
        "paper_squared_l2", "paper_l1", "paper_l2",
        "geopotential_rmse", "vorticity_rmse", "divergence_rmse",
    } <= output.metrics.keys()
    # The paper's squared L2 is the channel mean of the per-channel MSEs.
    channel_mse = sum(output.metrics[f"{name}_rmse"] ** 2 for name in CHANNELS) / 3
    assert output.metrics["paper_squared_l2"] == pytest.approx(channel_mse, rel=1e-5)

def test_a_tiny_run_trains_writes_its_protocol_and_resumes(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    # Composed before the chdir (see _cfg); the runs happen in the run directory.
    first = _cfg("training.max_epochs=1")
    resumed = _cfg("training.max_epochs=2", "training.resume=true")
    previous = Path.cwd()
    os.chdir(run)
    try:
        train_main(first)
        assert (run / PROTOCOL_FILENAME).exists()
        assert (run / "checkpoints" / "last.pt").exists()
        protocol = json.loads((run / PROTOCOL_FILENAME).read_text())
        assert protocol["task"] == "(geopotential, vorticity, divergence)"
        assert protocol["normalization"]["channels"] == ["geopotential", "vorticity", "divergence"]

        train_main(resumed)
    finally:
        os.chdir(previous)

    rows = [json.loads(line) for line in (run / "logs" / "metrics.jsonl").read_text().splitlines() if line]
    val = [row for row in rows if row["split"] == "val"]
    assert [row["epoch"] for row in val] == [0, 1]
    assert all(MONITOR_METRIC in row and "vorticity_rmse" in row for row in val)


def test_the_single_level_arm_runs_on_the_paper_state_too() -> None:
    """The zero-shot bundle's gm_l5_100 primary (the single-level arm)
    through the same entry point: one discretization with both radii
    prepared, the same two field types injected."""
    cfg = _cfg("model.architecture.chunk_size=17", bundle=SINGLE_LEVEL)
    experiment = build_scalar_experiment(cfg, torch.device("cpu"))

    assert cfg.model.kind == "gm_attention_single"
    assert experiment.model.input_field_type == CONDITIONED_FIELD_TYPE
    assert experiment.model.output_field_type == STATE_FIELD_TYPE
    assert experiment.model.chunk_size == 17
    assert experiment.grid.n_nodes == 162

    n = experiment.grid.n_nodes
    sample = ScalarState(torch.randn(n, 3), torch.randn(n, 3))
    experiment.model.eval()
    with torch.no_grad():
        output = experiment.step_fn(experiment.model, sample, torch.device("cpu"))
    assert torch.isfinite(output.loss)
    assert "vorticity_rmse" in output.metrics
