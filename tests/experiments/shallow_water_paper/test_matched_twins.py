"""The ablation twins parameter-matched to the full arm: the convolution
twin at the full arm's own hidden type (3.3% below), the rho_0 (+) rho_1
twin at equal copies of rho_0 and rho_1, the closest admissible count above.

The arms are the SWE_256x512_Ablation bundle's primaries: gm_attention_full
(the full arm), gm_convolution_full (the convolution twin) and
gm_attention_rho01 (the rho_0 (+) rho_1 twin)."""

from __future__ import annotations

import pytest
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf

from experiments.shallow_water_paper.train import build_scalar_experiment

FULL = 491_404


def _experiment(primary: str):
    with initialize(version_base=None, config_path="../../../SWE_256x512_Ablation/configs"):
        cfg = compose(config_name=primary, overrides=["device=cpu"])
    return cfg, build_scalar_experiment(cfg, torch.device("cpu"))


def _count(experiment) -> int:
    return sum(p.numel() for p in experiment.model.parameters())


@pytest.mark.parametrize(
    ("primary", "expected", "hidden"),
    [
        ("gm_convolution_full", 475_276, {0: 16, 1: 16, 2: 16}),
        ("gm_attention_rho01", 493_580, {0: 26, 1: 26}),
    ],
)
def test_the_matched_twin_lands_within_a_few_percent_of_the_full_arm(primary: str, expected: int, hidden: dict[int, int]) -> None:
    cfg, experiment = _experiment(primary)
    assert _count(experiment) == expected
    assert 0.96 * FULL < expected < 1.02 * FULL
    a = cfg.model.architecture
    base = OmegaConf.to_container(a.base_field_type)
    assert {k: v * int(a.multiplicity) for k, v in base.items()} == hidden
    assert (a.nonlinearity, a.decode_self_interaction, a.num_heads) == ("scalar_gate_double_sigmoid", "linear", 2)


def test_the_conv_twin_keeps_its_mixer_and_the_rho01_twin_has_no_rho2() -> None:
    conv, _ = _experiment("gm_convolution_full")
    assert conv.model.architecture.deep_mixer == "convolution"
    full, _ = _experiment("gm_attention_full")
    # the same hidden type as the full arm: only the mixer differs
    assert OmegaConf.to_container(conv.model.architecture.base_field_type) == OmegaConf.to_container(full.model.architecture.base_field_type)
    assert conv.model.architecture.multiplicity == full.model.architecture.multiplicity == 16
    rho01, _ = _experiment("gm_attention_rho01")
    assert "deep_mixer" not in rho01.model.architecture
    assert set(OmegaConf.to_container(rho01.model.architecture.base_field_type)) == {0, 1}


def test_the_full_arm_is_the_reference_count() -> None:
    _, experiment = _experiment("gm_attention_full")
    assert _count(experiment) == FULL
