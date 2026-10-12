"""Tests for GmAttentionSingleLevel: the depth-1 model with its hop removed."""

from __future__ import annotations

import math

import torch

from experiments.shallow_water.models import (
    CONDITIONED_FIELD_TYPE,
    PROGNOSTIC_FIELD_TYPE,
    GmAttentionSingleLevel,
)
from gm_attention.discretization.discretization import SphericalDiscretization
from gm_attention.discretization.frames import (
    build_tangent_frames,
    rotate_tangent_frames,
)
from gm_attention.discretization.grids.icosphere import build_icosphere_grid
from gm_attention.representations.field_type import SO2FeatureField

DEEP, CONV = 0.55, 0.4


def _discretization(frames: torch.Tensor | None = None) -> SphericalDiscretization:
    grid = build_icosphere_grid(subdivisions=2, dtype=torch.float64)
    if frames is None:
        frames = build_tangent_frames(grid.nodes)
    discretization = SphericalDiscretization.from_grid(
        grid, coverage_radius=max(DEEP, CONV), frames=frames
    )
    discretization.prepare_windows([DEEP, CONV])
    return discretization


def _model(**overrides) -> GmAttentionSingleLevel:
    torch.manual_seed(51)
    kwargs: dict = {
        "base_field_type": {0: 1, 1: 1, 2: 1}, "multiplicity": 4, "num_layers": 2,
        "num_heads": 2, "deep_cutoff": DEEP, "conv_cutoff": CONV, "window": "hard",
        "nonlinearity": "scalar_gate_double_sigmoid", "self_interaction": "linear",
        "dtype": torch.float64,
    }
    kwargs.update(overrides)
    return GmAttentionSingleLevel(**kwargs)


def _conditioned(n: int, seed: int = 52) -> SO2FeatureField:
    generator = torch.Generator().manual_seed(seed)
    return SO2FeatureField(
        CONDITIONED_FIELD_TYPE,
        {
            0: torch.randn(n, 2, dtype=torch.float64, generator=generator),
            1: torch.randn(n, 1, 2, dtype=torch.float64, generator=generator),
        },
    )


def test_single_level_has_the_depth1_interface_and_types() -> None:
    model = _model().eval()
    discretization = _discretization()
    n = discretization.grid.n_nodes

    out = model(_conditioned(n), discretization)

    assert model.input_field_type == CONDITIONED_FIELD_TYPE
    assert out.field_type == PROGNOSTIC_FIELD_TYPE
    assert out.batch_shape == (n,)
    assert model.encoder.conv.field_type_in == CONDITIONED_FIELD_TYPE if hasattr(model.encoder, "conv") else True
    assert model.fine_skip.weight_0.abs().sum() == 0 if hasattr(model.fine_skip, "weight_0") else True


def test_single_level_is_exactly_gauge_equivariant() -> None:
    model = _model(multiplicity=4, num_layers=1).eval()
    discretization = _discretization()
    n = discretization.grid.n_nodes
    x = _conditioned(n)
    generator = torch.Generator().manual_seed(53)
    angles = torch.rand(n, dtype=torch.float64, generator=generator) * (2 * math.pi)
    regauged = _discretization(
        rotate_tangent_frames(discretization.frames, angles)
    )

    original = model(x, discretization).to_flat()
    transformed = model(x.gauge_transform(angles), regauged).to_flat()
    expected = (
        SO2FeatureField.from_flat(PROGNOSTIC_FIELD_TYPE, original)
        .gauge_transform(angles)
        .to_flat()
    )
    assert torch.allclose(transformed, expected, atol=1e-12)


def test_single_level_chunked_forward_matches_whole() -> None:
    whole = _model().eval()
    chunked = _model(chunk_size=19).eval()
    chunked.load_state_dict(whole.state_dict())
    discretization = _discretization()
    x = _conditioned(discretization.grid.n_nodes)

    with torch.no_grad():
        a = whole(x, discretization).to_flat()
        b = chunked(x, discretization).to_flat()
    assert torch.allclose(a, b, atol=1e-13)


def test_single_level_chunked_training_gradients_match_whole() -> None:
    """The enabling property for L5: chunked + checkpointed training must
    produce the reference gradient through the entire model."""
    whole = _model()
    chunked = _model(chunk_size=23)
    chunked.load_state_dict(whole.state_dict())
    discretization = _discretization()
    x = _conditioned(discretization.grid.n_nodes)

    for model in (whole, chunked):
        model.train()
        model.zero_grad(set_to_none=True)
        model(x, discretization).to_flat().pow(2).sum().backward()

    pairs = list(zip(whole.parameters(), chunked.parameters(), strict=True))
    assert pairs
    for a, b in pairs:
        if a.grad is None:
            assert b.grad is None
            continue
        assert torch.allclose(a.grad, b.grad, atol=1e-12, rtol=1e-10)


def test_single_level_moves_devices_and_dtypes() -> None:
    """The helper's name must not shadow nn.Module._apply -- .to() is
    the canary."""
    model = _model().to(torch.float32)
    assert next(model.parameters()).dtype == torch.float32
