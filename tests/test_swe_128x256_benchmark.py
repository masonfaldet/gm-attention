"""The reproducibility bundle composes to the settings that produced the runs."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

CONFIGS = Path(__file__).resolve().parents[1] / "SWE_128x256_Benchmark" / "configs"


def _compose(arm: str, seed: int = 1, *, with_hydra: bool = False):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        return compose(config_name=arm, overrides=[f"seed={seed}"], return_hydra_config=with_hydra)


@pytest.mark.parametrize("arm", ["gm_attention_depth1", "sfno", "s2_transformer"])
def test_every_arm_shares_the_released_benchmarks_data(arm: str) -> None:
    cfg = _compose(arm)
    d = cfg.data
    assert (d.nlat, d.nlon, d.lead_time_seconds, d.solver_substeps) == (128, 256, 1800.0, 12)
    assert (d.train_seed, d.val_seed, d.normalization_seed, d.normalization_examples) == (1000, 2000, 4000, 128)
    assert (d.train_sample_policy, d.num_train_examples, d.num_val_examples) == ("epoch_index", 256, 128)
    assert cfg.objective.residual is True and cfg.training.batch_size == 4
    assert cfg.training.max_epochs == 100 and cfg.training.early_stopping_patience is None
    assert cfg.seed == 1


def test_the_gm_arm_is_the_depth1_attention_model_at_its_rate() -> None:
    cfg = _compose("gm_attention_depth1")
    assert cfg.model.kind == "gm_attention_depth1"
    assert cfg.model.architecture.nonlinearity == "scalar_gate_double_sigmoid"
    assert cfg.model.architecture.decode_self_interaction == "linear"
    assert cfg.training.lr == 1.2e-2 and cfg.training.grad_clip_norm == 1.0
    assert (cfg.training.scheduler.kind, cfg.training.scheduler.factor, cfg.training.scheduler.patience) == ("plateau", 0.5, 5)


def test_the_sfno_arm_is_the_registry_model_on_the_128x256_view() -> None:
    cfg = _compose("sfno")
    a = cfg.model.architecture
    assert cfg.model.kind == "sfno"
    assert (cfg.model.model_view.nlat, cfg.model.model_view.nlon) == (128, 256)
    assert (a.embed_dim, a.num_layers, a.scale_factor, a.normalization_layer) == (32, 4, 2, "instance_norm")
    assert cfg.training.lr == 3.2e-2 and cfg.training.grad_clip_norm == 1.0


def test_the_s2_arm_is_the_registry_model_under_the_scripts_protocol() -> None:
    cfg = _compose("s2_transformer")
    a = cfg.model.architecture
    assert cfg.model.kind == "attention_on_sphere"
    assert (a.embed_dim, a.num_heads, a.scale_factor, a.normalization_layer) == (128, 1, 2, "instance_norm")
    assert list(a.encoder_kernel_shape) == [5, 4] and a.filter_basis_type == "piecewise linear"
    assert a.grid_internal == "legendre-gauss"
    assert cfg.training.lr == 1e-4 and cfg.training.grad_clip_norm is None
    assert (cfg.training.scheduler.factor, cfg.training.scheduler.patience) == (0.1, 10)


def test_run_directories_are_keyed_by_arm_and_seed() -> None:
    for arm in ("gm_attention_depth1", "sfno", "s2_transformer"):
        cfg = _compose(arm, seed=2, with_hydra=True)
        assert cfg.hydra.run.dir == f"outputs/SWE_128x256_Benchmark/{arm}/seed2"
