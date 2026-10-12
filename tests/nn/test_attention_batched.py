"""The batched multi-head path is the sequential one: same parameters, same
function, same gradients, same training step, on every configuration the
sequential path admits."""

from __future__ import annotations

import copy
import math

import pytest
import torch

from gm_attention.discretization.discretization import SphericalDiscretization
from gm_attention.discretization.frames import (
    build_tangent_frames,
    rotate_tangent_frames,
)
from gm_attention.discretization.grids.icosphere import build_icosphere_grid
from gm_attention.nn.attention import SO2MHSA
from gm_attention.nn.blocks import SO2TransformerBlock
from gm_attention.representations.field_type import SO2FeatureField, SO2FieldType

CUTOFF = 0.6

CASES = [
    # (field type, heads, radial hidden features, radial basis, bias, window)
    ({0: 4, 1: 4, 2: 4}, 2, (16,), "gaussian", True, "hard"),
    ({0: 4, 1: 4, 2: 4}, 4, (), "gaussian", False, "quartic"),
    ({0: 4, 1: 4, 2: 4}, 1, (16,), "gaussian", True, "hard"),
    ({0: 4, 1: 2}, 2, (8, 6), "gaussian", True, "hard"),
    ({0: 6}, 3, (16,), "gaussian", True, "hard"),
    ({1: 4, 2: 2}, 2, (16,), "gaussian", True, "quartic"),
]
IDS = ["uniform-2h", "uniform-4h-nobias-quartic", "uniform-1h", "heterogeneous", "scalar-only", "no-scalar"]


def _discretization(frames: torch.Tensor | None = None) -> tuple[SphericalDiscretization, torch.Tensor]:
    grid = build_icosphere_grid(subdivisions=2, dtype=torch.float64)
    frames = build_tangent_frames(grid.nodes) if frames is None else frames
    return SphericalDiscretization.from_grid(grid, coverage_radius=CUTOFF, frames=frames), frames


def _pair(case, seed: int = 0) -> tuple[SO2MHSA, SO2MHSA, SO2FieldType]:
    """A sequential module and its batched copy, sharing every parameter value."""
    field_type, heads, hidden, basis, bias, window = case
    field_type = SO2FieldType(field_type)
    torch.manual_seed(seed)
    sequential = SO2MHSA(
        field_type, heads, cutoff=CUTOFF, window=window, radial_hidden_features=hidden,
        radial_basis=basis, bias=bias, dtype=torch.float64,
    )
    batched = copy.deepcopy(sequential)
    batched.batched = True
    return sequential, batched, field_type


def _leaf(field_type: SO2FieldType, n: int, seed: int = 1) -> tuple[torch.Tensor, SO2FeatureField]:
    """A leaf tensor and the field viewing it, so input gradients land on the leaf."""
    generator = torch.Generator().manual_seed(seed)
    flat = torch.randn(n, field_type.dimension, dtype=torch.float64, generator=generator).requires_grad_(True)
    return flat, SO2FeatureField.from_flat(field_type, flat)


def _field(field_type: SO2FieldType, n: int, seed: int = 1) -> SO2FeatureField:
    return _leaf(field_type, n, seed)[1]


def _max_diff(a: SO2FeatureField, b: SO2FeatureField) -> float:
    return max(float((a[f] - b[f]).abs().max()) for f in a.field_type.frequencies)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_outputs_input_gradients_and_parameter_gradients_agree(case) -> None:
    sequential, batched, field_type = _pair(case)
    discretization, _ = _discretization()
    leaf_seq, x_seq = _leaf(field_type, discretization.n_nodes)
    leaf_bat, x_bat = _leaf(field_type, discretization.n_nodes)

    y_seq, y_bat = sequential(x_seq, discretization), batched(x_bat, discretization)
    assert _max_diff(y_seq, y_bat) < 1e-12
    (y_seq.to_flat() ** 2).sum().backward()
    (y_bat.to_flat() ** 2).sum().backward()
    for (name, p_seq), (_, p_bat) in zip(sequential.named_parameters(), batched.named_parameters(), strict=True):
        assert p_seq.grad is not None and p_bat.grad is not None, name
        assert torch.allclose(p_seq.grad, p_bat.grad, atol=1e-11, rtol=0.0), name
    assert leaf_seq.grad is not None and leaf_bat.grad is not None
    assert torch.allclose(leaf_seq.grad, leaf_bat.grad, atol=1e-11, rtol=0.0)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_chunked_checkpointed_training_agrees_with_the_sequential_whole(case) -> None:
    sequential, batched, field_type = _pair(case)
    discretization, _ = _discretization()
    x_seq, x_bat = _field(field_type, discretization.n_nodes), _field(field_type, discretization.n_nodes)
    batched.train()
    y_seq = sequential(x_seq, discretization)
    y_bat = batched.forward_chunked(x_bat, discretization, chunk_size=7)
    assert _max_diff(y_seq, y_bat) < 1e-12
    (y_seq.to_flat() ** 2).sum().backward()
    (y_bat.to_flat() ** 2).sum().backward()
    for (name, p_seq), (_, p_bat) in zip(sequential.named_parameters(), batched.named_parameters(), strict=True):
        assert torch.allclose(p_seq.grad, p_bat.grad, atol=1e-11, rtol=0.0), name


def test_the_batched_path_is_exactly_gauge_equivariant() -> None:
    _, batched, field_type = _pair(CASES[0])
    discretization, frames = _discretization()
    generator = torch.Generator().manual_seed(4)
    angles = torch.rand(discretization.n_nodes, dtype=torch.float64, generator=generator) * 2 * math.pi
    regauged, _ = _discretization(rotate_tangent_frames(frames, angles))
    x = _field(field_type, discretization.n_nodes)
    expected = batched(x, discretization).gauge_transform(angles)
    actual = batched(x.gauge_transform(angles), regauged)
    assert _max_diff(expected, actual) < 1e-11


def test_one_optimizer_step_and_a_state_dict_round_trip_agree() -> None:
    """Same parameters in, same parameters out: the path owns nothing of its
    own, so checkpoints and optimizer state are the sequential path's."""
    sequential, batched, field_type = _pair(CASES[1])
    assert sequential.state_dict().keys() == batched.state_dict().keys()
    discretization, _ = _discretization()
    x = _field(field_type, discretization.n_nodes)
    optimizers = [torch.optim.Adam(m.parameters(), lr=1e-2) for m in (sequential, batched)]
    for module, optimizer in zip((sequential, batched), optimizers, strict=True):
        module.train()
        optimizer.zero_grad()
        (module(SO2FeatureField.from_flat(field_type, x.to_flat().detach()), discretization).to_flat() ** 2).sum().backward()
        optimizer.step()
    for (name, p_seq), (_, p_bat) in zip(sequential.named_parameters(), batched.named_parameters(), strict=True):
        assert torch.allclose(p_seq, p_bat, atol=1e-10, rtol=0.0), name
    # The batched module's checkpoint loads into a sequential one unchanged.
    fresh = SO2MHSA(field_type, 4, cutoff=CUTOFF, window="quartic", radial_hidden_features=(), radial_basis="gaussian", bias=False, dtype=torch.float64)
    fresh.load_state_dict(batched.state_dict())
    assert _max_diff(fresh(x, discretization), batched(x, discretization)) < 1e-12


def test_the_transformer_block_passes_the_flag_through() -> None:
    field_type = SO2FieldType({0: 4, 1: 4, 2: 4})
    torch.manual_seed(0)
    block = SO2TransformerBlock(field_type, 2, cutoff=CUTOFF, dtype=torch.float64)
    twin = copy.deepcopy(block)
    twin.attention.batched = True
    assert SO2TransformerBlock(field_type, 2, cutoff=CUTOFF, attention_batched=True).attention.batched is True
    discretization, _ = _discretization()
    x = _field(field_type, discretization.n_nodes)
    assert _max_diff(block(x, discretization), twin(x, discretization)) < 1e-12
    assert "batched=True" in repr(twin.attention)
