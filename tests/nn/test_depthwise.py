"""Tests for gm_attention.nn.depthwise: the per-copy steerable kernel.

The kernel is a sub-family of the general steerable kernel, so the primary
check is exact agreement with an ``SO2SteerableKernel`` whose off-diagonal
and cross-frequency blocks have been zeroed and whose diagonal blocks carry
the depthwise kernel's own radial heads. Equivariance is then inherited,
and checked directly once more on a real discretization.
"""

from __future__ import annotations

import math

import pytest
import torch

from gm_attention.discretization.discretization import SphericalDiscretization
from gm_attention.discretization.frames import (
    build_tangent_frames,
    rotate_tangent_frames,
)
from gm_attention.discretization.grids.icosphere import build_icosphere_grid
from gm_attention.nn._scatter import gather_to_edges
from gm_attention.nn.convolution import SO2Convolution
from gm_attention.nn.depthwise import SO2DepthwiseKernel
from gm_attention.nn.kernels import SO2SteerableKernel
from gm_attention.representations.field_type import SO2FeatureField, SO2FieldType

FIELD_TYPE = SO2FieldType({0: 3, 1: 2, 2: 2})
CUTOFF = 0.6


def _discretization(seed: int = 0) -> tuple[SphericalDiscretization, torch.Tensor]:
    grid = build_icosphere_grid(subdivisions=1, dtype=torch.float64)
    frames = build_tangent_frames(grid.nodes)
    return SphericalDiscretization.from_grid(grid, coverage_radius=CUTOFF, frames=frames), frames


def _field(n_nodes: int, seed: int = 0, field_type: SO2FieldType = FIELD_TYPE) -> SO2FeatureField:
    generator = torch.Generator().manual_seed(seed)
    return SO2FeatureField.from_flat(
        field_type, torch.randn(n_nodes, field_type.dimension, dtype=torch.float64, generator=generator)
    )


def _embed_into_general(depthwise: SO2DepthwiseKernel) -> SO2SteerableKernel:
    """A general kernel equal to ``depthwise``: same trunk, diagonal blocks only."""
    field_type = depthwise.field_type
    general = SO2SteerableKernel(
        field_type, field_type, cutoff=depthwise.cutoff,
        n_radial_basis=depthwise.radial_basis.n_basis,
        radial_hidden_features=tuple(
            layer.out_features for layer in depthwise.radial_trunk if isinstance(layer, torch.nn.Linear)
        ),
        dtype=torch.float64,
    )
    general.radial_trunk.load_state_dict(depthwise.radial_trunk.state_dict())
    with torch.no_grad():
        assert general.real_head is not None and depthwise.real_head is not None
        general.real_head.weight.zero_()
        general.real_head.bias.zero_()
        a0 = field_type.multiplicity(0)
        for i in range(a0):
            general.real_head.weight[i * a0 + i] = depthwise.real_head.weight[i]
            general.real_head.bias[i * a0 + i] = depthwise.real_head.bias[i]

        assert general.complex_head is not None and depthwise.complex_head is not None
        general.complex_head.weight.zero_()
        general.complex_head.bias.zero_()
        offset = 0
        for frequency in field_type.frequencies:
            if frequency == 0:
                continue
            a = field_type.multiplicity(frequency)
            for role_index, role in enumerate(("linear", "antilinear")):
                target = general._pair_ranges[(frequency, frequency, role)]
                for i in range(a):
                    for c in range(2):
                        source_column = offset + role_index * 2 * a + i * 2 + c
                        target_column = target.start + (i * a + i) * 2 + c
                        general.complex_head.weight[target_column] = depthwise.complex_head.weight[source_column]
                        general.complex_head.bias[target_column] = depthwise.complex_head.bias[source_column]
            offset += 4 * a
    return general


def test_depthwise_equals_the_general_kernel_with_only_diagonal_blocks() -> None:
    torch.manual_seed(3)
    depthwise = SO2DepthwiseKernel(FIELD_TYPE, cutoff=CUTOFF, dtype=torch.float64)
    general = _embed_into_general(depthwise)
    discretization, _ = _discretization()
    edges = discretization.edges_within(CUTOFF)
    source = gather_to_edges(_field(discretization.n_nodes), edges.source_index)

    ours = depthwise(source, edges)
    reference = general(source, edges)

    for frequency in FIELD_TYPE.frequencies:
        torch.testing.assert_close(ours[frequency], reference[frequency], atol=1e-12, rtol=0.0)


def test_depthwise_kernel_is_exactly_gauge_equivariant() -> None:
    torch.manual_seed(4)
    kernel = SO2DepthwiseKernel(FIELD_TYPE, cutoff=CUTOFF, dtype=torch.float64)
    grid = build_icosphere_grid(subdivisions=1, dtype=torch.float64)
    frames = build_tangent_frames(grid.nodes)
    generator = torch.Generator().manual_seed(9)
    angles = torch.rand(grid.n_nodes, dtype=torch.float64, generator=generator) * 2 * math.pi
    original = SphericalDiscretization.from_grid(grid, coverage_radius=CUTOFF, frames=frames)
    regauged = SphericalDiscretization.from_grid(
        grid, coverage_radius=CUTOFF, frames=rotate_tangent_frames(frames, angles)
    )
    x = _field(grid.n_nodes)
    edges, edges_regauged = original.edges_within(CUTOFF), regauged.edges_within(CUTOFF)

    expected = kernel(gather_to_edges(x, edges.source_index), edges).gauge_transform(
        angles[edges.target_index]
    )
    actual = kernel(
        gather_to_edges(x.gauge_transform(angles), edges_regauged.source_index), edges_regauged
    )

    for frequency in FIELD_TYPE.frequencies:
        torch.testing.assert_close(actual[frequency], expected[frequency], atol=1e-10, rtol=0.0)


def test_the_origin_value_is_a_per_copy_intertwiner() -> None:
    """At r = 0 the antilinear term vanishes, leaving z -> A(0) z per copy:
    the self-loop message of a rotated input is the rotated message."""
    torch.manual_seed(5)
    kernel = SO2DepthwiseKernel(FIELD_TYPE, cutoff=CUTOFF, dtype=torch.float64)
    discretization, _ = _discretization()
    edges = discretization.edges_within(CUTOFF)
    loops = edges.select(torch.nonzero(edges.distances == 0.0).flatten())
    x = _field(loops.n_edges, seed=2)
    theta = torch.full((loops.n_edges,), 0.7, dtype=torch.float64)

    rotate_then_apply = kernel(x.apply_rotation(theta), loops)
    apply_then_rotate = kernel(x, loops).apply_rotation(theta)

    for frequency in FIELD_TYPE.frequencies:
        torch.testing.assert_close(rotate_then_apply[frequency], apply_then_rotate[frequency], atol=1e-12, rtol=0.0)


def test_parameter_count_is_linear_in_the_multiplicity() -> None:
    """One real function per scalar copy, four per non-scalar copy, over a
    shared trunk -- so doubling every multiplicity doubles only the heads."""
    narrow = SO2DepthwiseKernel(SO2FieldType({0: 2, 1: 2}), cutoff=1.0, n_radial_basis=4, radial_hidden_features=(6,))
    wide = SO2DepthwiseKernel(SO2FieldType({0: 4, 1: 4}), cutoff=1.0, n_radial_basis=4, radial_hidden_features=(6,))
    trunk = 4 * 6 + 6
    heads_narrow = (6 + 1) * (2 + 4 * 2)
    heads_wide = (6 + 1) * (4 + 4 * 4)
    assert sum(p.numel() for p in narrow.parameters()) == trunk + heads_narrow
    assert sum(p.numel() for p in wide.parameters()) == trunk + heads_wide


def test_convolution_depthwise_flag_builds_the_depthwise_kernel_and_rejects_a_type_change() -> None:
    conv = SO2Convolution(FIELD_TYPE, FIELD_TYPE, cutoff=CUTOFF, depthwise=True)
    assert isinstance(conv.kernel, SO2DepthwiseKernel)
    assert "depthwise=True" in repr(conv)
    with pytest.raises(ValueError, match="depthwise"):
        SO2Convolution(FIELD_TYPE, SO2FieldType({0: 3, 1: 2}), cutoff=CUTOFF, depthwise=True)


def test_depthwise_convolution_forward_and_chunked_agree_and_gradients_flow() -> None:
    torch.manual_seed(6)
    conv = SO2Convolution(FIELD_TYPE, FIELD_TYPE, cutoff=CUTOFF, depthwise=True, dtype=torch.float64)
    discretization, _ = _discretization()
    x = _field(discretization.n_nodes)
    whole = conv(x, discretization)
    chunked = conv.forward_chunked(x, discretization, chunk_size=5)
    for frequency in FIELD_TYPE.frequencies:
        torch.testing.assert_close(whole[frequency], chunked[frequency], atol=1e-12, rtol=0.0)
    loss = sum((whole[f] ** 2).sum() for f in FIELD_TYPE.frequencies)
    loss.backward()
    for name, parameter in conv.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name


def test_rejects_mismatched_inputs() -> None:
    kernel = SO2DepthwiseKernel(FIELD_TYPE, cutoff=CUTOFF, dtype=torch.float64)
    discretization, _ = _discretization()
    edges = discretization.edges_within(CUTOFF)
    with pytest.raises(ValueError, match="type"):
        kernel(gather_to_edges(_field(discretization.n_nodes, field_type=SO2FieldType({0: 3})), edges.source_index), edges)
    with pytest.raises(ValueError, match="batch_shape"):
        kernel(_field(discretization.n_nodes), edges)
    far = SO2DepthwiseKernel(FIELD_TYPE, cutoff=0.5 * CUTOFF, dtype=torch.float64)
    with pytest.raises(ValueError, match="cutoff"):
        far(gather_to_edges(_field(discretization.n_nodes), edges.source_index), edges)
