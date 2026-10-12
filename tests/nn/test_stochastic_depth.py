"""Tests for gm_attention.nn.stochastic_depth."""

from __future__ import annotations

import math

import pytest
import torch

from gm_attention.nn.stochastic_depth import SO2DropPath, copies, current_copies
from gm_attention.representations.field_type import SO2FeatureField, SO2FieldType

FIELD_TYPE = SO2FieldType({0: 2, 1: 1, 2: 1})


def _field(seed: int = 0) -> SO2FeatureField:
    generator = torch.Generator().manual_seed(seed)
    return SO2FeatureField.from_flat(
        FIELD_TYPE, torch.randn(7, FIELD_TYPE.dimension, dtype=torch.float64, generator=generator)
    )


def test_rejects_an_invalid_probability() -> None:
    with pytest.raises(ValueError):
        SO2DropPath(1.0)
    with pytest.raises(ValueError):
        SO2DropPath(-0.1)


def test_identity_in_eval_mode_and_at_zero_rate() -> None:
    x = _field()
    for module in (SO2DropPath(0.0).train(), SO2DropPath(0.5).eval()):
        y = module(x)
        for frequency in FIELD_TYPE.frequencies:
            assert torch.equal(y[frequency], x[frequency])


def test_training_drops_the_whole_branch_or_rescales_it() -> None:
    x = _field()
    module = SO2DropPath(0.5).train()
    outcomes = set()
    torch.manual_seed(0)
    for _ in range(64):
        y = module(x)
        zero = all(torch.equal(y[f], torch.zeros_like(x[f])) for f in FIELD_TYPE.frequencies)
        doubled = all(torch.allclose(y[f], 2.0 * x[f]) for f in FIELD_TYPE.frequencies)
        assert zero or doubled
        outcomes.add("zero" if zero else "doubled")
    assert outcomes == {"zero", "doubled"}


@pytest.mark.parametrize("count", [1, 7])
def test_commutes_with_a_gauge_transformation(count: int) -> None:
    """A scalar multiple of the identity on every node commutes with every
    rho_n, whether the scalar is one draw or one per sample of a union."""
    x = _field()
    angles = torch.rand(7, dtype=torch.float64) * 2 * math.pi
    module = SO2DropPath(0.5).train()
    with copies(count):
        torch.manual_seed(1)
        a = module(x.gauge_transform(angles))
        torch.manual_seed(1)
        b = module(x).gauge_transform(angles)
    for frequency in FIELD_TYPE.frequencies:
        torch.testing.assert_close(a[frequency], b[frequency])


def test_a_union_draws_once_per_sample() -> None:
    """Under copies(3) a field of 3 x 4 nodes is three blocks, each dropped or
    rescaled as a unit, with draws that differ across the blocks."""
    generator = torch.Generator().manual_seed(2)
    x = SO2FeatureField.from_flat(
        FIELD_TYPE, torch.randn(12, FIELD_TYPE.dimension, dtype=torch.float64, generator=generator)
    )
    module = SO2DropPath(0.5).train()
    patterns = set()
    torch.manual_seed(0)
    with copies(3):
        assert current_copies() == 3
        for _ in range(64):
            y = module(x)
            pattern = []
            for b in range(3):
                block = slice(4 * b, 4 * (b + 1))
                zero = all(torch.equal(y[f][block], torch.zeros_like(x[f][block])) for f in FIELD_TYPE.frequencies)
                doubled = all(torch.allclose(y[f][block], 2.0 * x[f][block]) for f in FIELD_TYPE.frequencies)
                assert zero or doubled
                pattern.append(zero)
            patterns.add(tuple(pattern))
    assert current_copies() == 1
    assert len(patterns) > 2  # one shared decision would allow only all-zero and all-doubled
    with pytest.raises(ValueError), copies(5):
        module(x)  # 12 nodes are not 5 blocks
    with pytest.raises(ValueError), copies(0):
        pass
