"""Tests for gm_attention.nn.blocks: SO2TransformerBlock, IcoEncode, IcoDecode.

SO2TransformerBlock composes already-tested layers (SO2LayerNorm, SO2MHSA,
SO2Linear, SO2ScalarGatedNonlinearity, SO2Dropout), so the bar here isn't
re-proving each piece's own math -- it's proving the *wiring*: that the
residual connections are actually additive (checked by stubbing out a
branch with an exact zero and confirming the block degenerates to the
identity), that the whole assembly is still exactly gauge equivariant end
to end on a real discretization, and that gradients/dropout/edges-dispatch
all pass through correctly.

IcoEncode/IcoDecode get the same equivariance bar (the centerpiece: this
codebase's whole reason to exist), plus targeted checks of the two things
genuinely new about them -- the bipartite node-count change IcoEncode's
output undergoes, and IcoDecode's two-region (copy vs. transport-and-
average) upsample step, isolated the same way SO2TransformerBlock's own
tests isolate a branch: by stubbing out the *other* piece (the internal
conv, for IcoDecode) with something trivial, per test.
"""

from __future__ import annotations

import math

import pytest
import torch

from gm_attention.discretization.discretization import SphericalDiscretization
from gm_attention.discretization.edge_geometry import compute_edge_geometry
from gm_attention.discretization.frames import (
    build_tangent_frames,
    rotate_tangent_frames,
)
from gm_attention.discretization.grid import median_nearest_neighbor_distance
from gm_attention.discretization.grids.icosphere import (
    build_icosphere_downsample_edges,
    build_icosphere_grid,
    build_icosphere_multiresolution,
    build_icosphere_refinement,
    build_icosphere_upsample_edges,
)
from gm_attention.nn._scatter import gather_to_edges, scatter_add_to_nodes
from gm_attention.nn.blocks import (
    IcoDecode,
    IcoEncode,
    IcoLowPassRestriction,
    SO2TransformerBlock,
)
from gm_attention.nn.kernels import transport_field
from gm_attention.representations.field_type import SO2FeatureField, SO2FieldType


def _random_field(
    field_type: SO2FieldType,
    batch_shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    dtype: torch.dtype = torch.float64,
) -> SO2FeatureField:
    return SO2FeatureField.from_flat(
        field_type,
        torch.randn((*batch_shape, field_type.dimension), dtype=dtype, generator=generator),
    )


def _build_grid_and_edges(field_type: SO2FieldType):
    grid = build_icosphere_grid(subdivisions=1, dtype=torch.float64)
    frames = build_tangent_frames(grid.nodes)
    coverage_radius = 0.6 * math.pi
    edges = compute_edge_geometry(
        grid, frames, coverage_radius, max_frequency=field_type.max_frequency * 2
    )
    return grid, frames, coverage_radius, edges


def _zero_field(field_type: SO2FieldType, like: SO2FeatureField) -> SO2FeatureField:
    return SO2FeatureField(
        field_type,
        {
            frequency: torch.zeros_like(like[frequency])
            for frequency in field_type.frequencies
        },
    )


# ==========================================================================
# Construction validation
# ==========================================================================


def test_rejects_a_non_field_type() -> None:
    with pytest.raises(TypeError):
        SO2TransformerBlock({0: 4}, 2, ffn_multiplier=2, cutoff=1.0)  # type: ignore[arg-type]


def test_rejects_a_nonpositive_ffn_multiplier() -> None:
    with pytest.raises(ValueError):
        SO2TransformerBlock(
            SO2FieldType({0: 4, 1: 4}), 2, ffn_multiplier=0, cutoff=1.0
        )


def test_rejects_both_ffn_multiplier_and_hidden_field_type() -> None:
    """The two would usually disagree, so a silent precedence rule would
    quietly build a different network than the config states."""
    with pytest.raises(ValueError, match="not both"):
        SO2TransformerBlock(
            SO2FieldType({0: 4, 1: 4}),
            2,
            ffn_multiplier=3,
            hidden_field_type=SO2FieldType({0: 8, 1: 8}),
            cutoff=1.0,
        )


def test_accepts_a_heterogeneous_field_type() -> None:
    """Heterogeneous multiplicities are supported: only per-frequency head
    divisibility is required, not one shared multiplicity."""
    block = SO2TransformerBlock(
        SO2FieldType({0: 4, 1: 8}), 2, ffn_multiplier=2, cutoff=1.0
    )

    assert block.field_type == SO2FieldType({0: 4, 1: 8})
    assert block.hidden_field_type == SO2FieldType({0: 8, 1: 16})


def test_ffn_multiplier_preserves_heterogeneous_ratios() -> None:
    """Spec 11.3: residual {0: 8, 1: 4} with ffn_multiplier 2 -> {0: 16, 1: 8}."""
    block = SO2TransformerBlock(
        SO2FieldType({0: 8, 1: 4}), 4, ffn_multiplier=2, cutoff=1.0
    )

    assert block.hidden_field_type == SO2FieldType({0: 16, 1: 8})
    assert block.attention.field_type_attn == SO2FieldType({0: 2, 1: 1})


def test_rejects_num_heads_not_dividing_the_multiplicity() -> None:
    """Propagated from SO2MHSA."""
    with pytest.raises(ValueError):
        SO2TransformerBlock(
            SO2FieldType({0: 4, 1: 4}), 3, ffn_multiplier=2, cutoff=1.0
        )


def test_hidden_field_type_shares_the_frequency_spectrum() -> None:
    field_type = SO2FieldType({0: 4, 1: 4, 2: 4})
    block = SO2TransformerBlock(
        field_type, 2, ffn_multiplier=2, cutoff=1.0, dtype=torch.float64
    )

    assert block.hidden_field_type.frequencies == field_type.frequencies
    for frequency in field_type.frequencies:
        assert block.hidden_field_type.multiplicity(frequency) == 8


# ==========================================================================
# Forward: shape / type
# ==========================================================================


def test_forward_preserves_field_type_and_batch_shape() -> None:
    field_type = SO2FieldType({0: 4, 1: 4})
    torch.manual_seed(0)
    block = SO2TransformerBlock(
        field_type, 2, ffn_multiplier=2, cutoff=0.6 * math.pi, dtype=torch.float64
    )

    _, _, _, edges = _build_grid_and_edges(field_type)
    generator = torch.Generator().manual_seed(1)
    n_nodes = edges.row_ptr.shape[0] - 1
    x = _random_field(field_type, (n_nodes,), generator=generator)

    output = block(x, edges)

    assert output.field_type == field_type
    assert output.batch_shape == x.batch_shape


def test_forward_accepts_a_spherical_discretization_directly() -> None:
    """edges is passed straight through to the internal SO2MHSA, which
    resolves a SphericalDiscretization itself -- confirm that passthrough
    still works, not just the pre-sliced EdgeGeometry path."""
    field_type = SO2FieldType({0: 4, 1: 4})
    cutoff = 0.6 * math.pi
    torch.manual_seed(2)
    block = SO2TransformerBlock(
        field_type, 2, ffn_multiplier=2, cutoff=cutoff, dtype=torch.float64
    )

    grid = build_icosphere_grid(subdivisions=1, dtype=torch.float64)
    frames = build_tangent_frames(grid.nodes)
    edges = compute_edge_geometry(
        grid, frames, cutoff, max_frequency=field_type.max_frequency * 2
    )
    discretization = SphericalDiscretization(grid, frames, edges, cutoff)

    generator = torch.Generator().manual_seed(3)
    x = _random_field(field_type, (grid.n_nodes,), generator=generator)

    output = block(x, discretization)

    assert output.field_type == field_type


# ==========================================================================
# Residual wiring: stub a branch to exact zero and confirm the identity
# ==========================================================================


def test_zeroing_the_attention_branch_reduces_the_block_to_the_ffn_residual() -> None:
    field_type = SO2FieldType({0: 4, 1: 4})
    torch.manual_seed(4)
    block = SO2TransformerBlock(
        field_type, 2, ffn_multiplier=2, cutoff=0.6 * math.pi, dtype=torch.float64
    )

    _, _, _, edges = _build_grid_and_edges(field_type)
    n_nodes = edges.row_ptr.shape[0] - 1
    generator = torch.Generator().manual_seed(5)
    x = _random_field(field_type, (n_nodes,), generator=generator)

    # Override the instance's bound forward, not the nn.Module attribute
    # itself -- nn.Module.__setattr__ rejects reassigning a registered
    # submodule to a non-Module value.
    block.attention.forward = lambda x, edges: _zero_field(field_type, x)  # type: ignore[method-assign]

    output = block(x, edges)

    # With the attention branch zeroed, x' == x exactly, so the block
    # reduces to x + FFN(Norm2(x)) -- recompute that directly and compare.
    hidden = block.linear2(block.activation(block.linear1(block.norm2(x))))
    expected = SO2FeatureField(
        field_type,
        {
            frequency: x[frequency] + block.mlp_dropout(hidden)[frequency]
            for frequency in field_type.frequencies
        },
    )

    assert torch.allclose(output.to_flat(), expected.to_flat(), atol=1e-10)


def test_zeroing_both_branches_reduces_the_block_to_the_identity() -> None:
    field_type = SO2FieldType({0: 4, 1: 4})
    torch.manual_seed(6)
    block = SO2TransformerBlock(
        field_type, 2, ffn_multiplier=2, cutoff=0.6 * math.pi, dtype=torch.float64
    )

    _, _, _, edges = _build_grid_and_edges(field_type)
    n_nodes = edges.row_ptr.shape[0] - 1
    generator = torch.Generator().manual_seed(7)
    x = _random_field(field_type, (n_nodes,), generator=generator)

    original_x = x
    block.attention.forward = lambda x, edges: _zero_field(field_type, x)  # type: ignore[method-assign]
    # linear2's own parameter is hidden_field_type-shaped, so build the zero
    # from original_x (field_type-shaped) instead -- the parameter is still
    # named x to match SO2Linear.forward's signature.
    block.linear2.forward = lambda x: _zero_field(field_type, original_x)  # type: ignore[method-assign]

    output = block(x, edges)

    assert torch.equal(output.to_flat(), x.to_flat())


# ==========================================================================
# Gauge equivariance: the definitive correctness claim, end to end
# ==========================================================================


def test_forward_is_exactly_gauge_equivariant() -> None:
    field_type = SO2FieldType({0: 2, 1: 2, 2: 2})
    torch.manual_seed(8)
    block = SO2TransformerBlock(
        field_type, 2, ffn_multiplier=2, cutoff=0.6 * math.pi, dtype=torch.float64
    )

    grid, frames, coverage_radius, edges = _build_grid_and_edges(field_type)

    generator = torch.Generator().manual_seed(9)
    node_features = _random_field(field_type, (grid.n_nodes,), generator=generator)

    original_output = block(node_features, edges).to_flat()

    gauge_angles = torch.rand(grid.n_nodes, dtype=torch.float64, generator=generator) * (
        2 * math.pi
    )
    regauged_edges = compute_edge_geometry(
        grid,
        rotate_tangent_frames(frames, gauge_angles),
        coverage_radius,
        max_frequency=field_type.max_frequency * 2,
    )
    regauged_node_features = node_features.gauge_transform(gauge_angles)

    regauged_output = block(regauged_node_features, regauged_edges).to_flat()
    expected = (
        SO2FeatureField.from_flat(field_type, original_output)
        .gauge_transform(gauge_angles)
        .to_flat()
    )

    assert torch.allclose(regauged_output, expected, atol=1e-7)


# ==========================================================================
# Dropout: off by default, stochastic only when enabled and training
# ==========================================================================


def test_dropout_zero_is_deterministic_across_calls() -> None:
    field_type = SO2FieldType({0: 4, 1: 4})
    torch.manual_seed(10)
    block = SO2TransformerBlock(
        field_type, 2, ffn_multiplier=2, cutoff=0.6 * math.pi, dtype=torch.float64
    )
    block.train()

    _, _, _, edges = _build_grid_and_edges(field_type)
    n_nodes = edges.row_ptr.shape[0] - 1
    generator = torch.Generator().manual_seed(11)
    x = _random_field(field_type, (n_nodes,), generator=generator)

    first = block(x, edges).to_flat()
    second = block(x, edges).to_flat()

    assert torch.equal(first, second)


def test_dropout_nonzero_is_stochastic_in_training_mode() -> None:
    field_type = SO2FieldType({0: 4, 1: 4})
    torch.manual_seed(12)
    block = SO2TransformerBlock(
        field_type,
        2,
        ffn_multiplier=2,
        cutoff=0.6 * math.pi,
        dropout=0.5,
        dtype=torch.float64,
    )
    block.train()

    _, _, _, edges = _build_grid_and_edges(field_type)
    n_nodes = edges.row_ptr.shape[0] - 1
    generator = torch.Generator().manual_seed(13)
    x = _random_field(field_type, (n_nodes,), generator=generator)

    torch.manual_seed(100)
    first = block(x, edges).to_flat()
    torch.manual_seed(200)
    second = block(x, edges).to_flat()

    assert not torch.equal(first, second)


def test_dropout_nonzero_is_deterministic_in_eval_mode() -> None:
    field_type = SO2FieldType({0: 4, 1: 4})
    torch.manual_seed(14)
    block = SO2TransformerBlock(
        field_type,
        2,
        ffn_multiplier=2,
        cutoff=0.6 * math.pi,
        dropout=0.5,
        dtype=torch.float64,
    )
    block.eval()

    _, _, _, edges = _build_grid_and_edges(field_type)
    n_nodes = edges.row_ptr.shape[0] - 1
    generator = torch.Generator().manual_seed(15)
    x = _random_field(field_type, (n_nodes,), generator=generator)

    first = block(x, edges).to_flat()
    second = block(x, edges).to_flat()

    assert torch.equal(first, second)


# ==========================================================================
# Autograd
# ==========================================================================


def test_gradients_flow_through_every_sub_layer_and_to_the_input() -> None:
    field_type = SO2FieldType({0: 4, 1: 4})
    torch.manual_seed(16)
    block = SO2TransformerBlock(
        field_type, 2, ffn_multiplier=2, cutoff=0.6 * math.pi, dtype=torch.float64
    )

    _, _, _, edges = _build_grid_and_edges(field_type)
    n_nodes = edges.row_ptr.shape[0] - 1
    flat = torch.randn(
        n_nodes, field_type.dimension, dtype=torch.float64, requires_grad=True
    )
    x = SO2FeatureField.from_flat(field_type, flat)

    block(x, edges).to_flat().sum().backward()

    assert flat.grad is not None
    assert torch.isfinite(flat.grad).all()

    for name, parameter in block.named_parameters():
        assert parameter.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(parameter.grad).all()


# ==========================================================================
# The convolution mixer
# ==========================================================================


def test_rejects_an_unknown_mixer() -> None:
    with pytest.raises(ValueError, match="mixer"):
        SO2TransformerBlock(
            SO2FieldType({0: 2}), 2, ffn_multiplier=2, cutoff=1.0, mixer="pooling"  # type: ignore[arg-type]
        )


def test_the_convolution_mixer_builds_a_convolution_not_attention() -> None:
    """The attribute keeps the name `attention` so the residual, dropout,
    and chunked-evaluation wiring is mixer-agnostic -- but what lives there
    must actually be the plain steerable convolution."""
    from gm_attention.nn.attention import SO2MHSA
    from gm_attention.nn.convolution import SO2Convolution

    field_type = SO2FieldType({0: 2, 1: 2})
    conv_block = SO2TransformerBlock(
        field_type, 2, ffn_multiplier=2, cutoff=1.0, mixer="convolution"
    )
    attn_block = SO2TransformerBlock(field_type, 2, ffn_multiplier=2, cutoff=1.0)

    assert conv_block.mixer == "convolution"
    assert isinstance(conv_block.attention, SO2Convolution)
    assert conv_block.attention.self_interaction == "none"
    assert isinstance(attn_block.attention, SO2MHSA)


def test_the_convolution_mixer_accepts_an_odd_multiplicity() -> None:
    """num_heads never constructs under the convolution mixer, so the
    head-divisibility constraint that pins the attention arms to even
    multiplicities does not apply -- this is what lets the conv ablation
    land at m=17 (+0.4%) where attention steps ~128k parameters at a
    time."""
    odd = SO2FieldType({0: 3, 1: 3})

    with pytest.raises(ValueError):
        SO2TransformerBlock(odd, 2, ffn_multiplier=2, cutoff=1.0)

    block = SO2TransformerBlock(
        odd, 2, ffn_multiplier=2, cutoff=1.0, mixer="convolution"
    )
    assert block.mixer == "convolution"


def test_the_convolution_mixer_forward_preserves_field_type_and_batch_shape() -> None:
    field_type = SO2FieldType({0: 2, 1: 2, 2: 2})
    torch.manual_seed(21)
    block = SO2TransformerBlock(
        field_type,
        2,
        ffn_multiplier=2,
        cutoff=0.6 * math.pi,
        mixer="convolution",
        dtype=torch.float64,
    )

    grid, _, _, edges = _build_grid_and_edges(field_type)
    generator = torch.Generator().manual_seed(22)
    node_features = _random_field(field_type, (grid.n_nodes,), generator=generator)

    output = block(node_features, edges)

    assert output.field_type == field_type
    assert output.to_flat().shape == node_features.to_flat().shape


def test_the_convolution_mixer_is_exactly_gauge_equivariant() -> None:
    field_type = SO2FieldType({0: 2, 1: 2, 2: 2})
    torch.manual_seed(23)
    block = SO2TransformerBlock(
        field_type,
        2,
        ffn_multiplier=2,
        cutoff=0.6 * math.pi,
        mixer="convolution",
        dtype=torch.float64,
    )

    grid, frames, coverage_radius, edges = _build_grid_and_edges(field_type)

    generator = torch.Generator().manual_seed(24)
    node_features = _random_field(field_type, (grid.n_nodes,), generator=generator)

    original_output = block(node_features, edges).to_flat()

    gauge_angles = torch.rand(grid.n_nodes, dtype=torch.float64, generator=generator) * (
        2 * math.pi
    )
    regauged_edges = compute_edge_geometry(
        grid,
        rotate_tangent_frames(frames, gauge_angles),
        coverage_radius,
        max_frequency=field_type.max_frequency * 2,
    )
    regauged_node_features = node_features.gauge_transform(gauge_angles)

    regauged_output = block(regauged_node_features, regauged_edges).to_flat()
    expected = (
        SO2FeatureField.from_flat(field_type, original_output)
        .gauge_transform(gauge_angles)
        .to_flat()
    )

    assert torch.allclose(regauged_output, expected, atol=1e-12)


def test_the_convolution_mixer_chunked_forward_matches_the_ordinary_one() -> None:
    """The block's chunked-evaluation path addresses
    `self.attention.forward_chunked` without knowing which mixer lives
    there; the convolution must honor that contract too."""
    field_type = SO2FieldType({0: 2, 1: 2})
    torch.manual_seed(25)
    block = SO2TransformerBlock(
        field_type,
        2,
        ffn_multiplier=2,
        cutoff=0.6 * math.pi,
        mixer="convolution",
        dtype=torch.float64,
    )
    block.eval()

    grid, _, _, edges = _build_grid_and_edges(field_type)
    generator = torch.Generator().manual_seed(26)
    node_features = _random_field(field_type, (grid.n_nodes,), generator=generator)

    with torch.no_grad():
        ordinary = block(node_features, edges).to_flat()
        chunked = block.forward_chunked(node_features, edges, chunk_size=5).to_flat()

    assert torch.allclose(chunked, ordinary, atol=1e-13)


# ==========================================================================
# IcoEncode / IcoDecode
# ==========================================================================


def _hierarchy(coarse_subdivisions: int = 2):
    coarse, fine, refinement = build_icosphere_refinement(coarse_subdivisions, dtype=torch.float64)
    fine_frames = build_tangent_frames(fine.nodes)
    down_cutoff = median_nearest_neighbor_distance(coarse)
    return coarse, fine, refinement, fine_frames, down_cutoff


# --------------------------------------------------------------------------
# IcoEncode: construction / forward shape
# --------------------------------------------------------------------------


def test_ico_encode_rejects_an_invalid_window() -> None:
    with pytest.raises(ValueError):
        IcoEncode(
            SO2FieldType({0: 2}), SO2FieldType({0: 2}), cutoff=0.3, window="triangular"  # type: ignore[arg-type]
        )


def test_ico_encode_output_batch_shape_is_the_coarse_node_count() -> None:
    coarse, fine, refinement, fine_frames, down_cutoff = _hierarchy()
    field_type_in = SO2FieldType({0: 2, 1: 2})
    field_type_out = SO2FieldType({0: 3, 1: 3, 2: 3})

    torch.manual_seed(20)
    encode = IcoEncode(field_type_in, field_type_out, cutoff=down_cutoff, dtype=torch.float64)
    down_edges = build_icosphere_downsample_edges(
        fine, fine_frames, refinement.n_coarse_nodes, cutoff=down_cutoff
    )

    generator = torch.Generator().manual_seed(21)
    x = _random_field(field_type_in, (fine.n_nodes,), generator=generator)

    output = encode(x, down_edges)

    assert output.field_type == field_type_out
    assert output.batch_shape == (coarse.n_nodes,)
    assert output.batch_shape != x.batch_shape  # the whole point: a real node-count change


def test_ico_encode_rejects_a_mismatched_input_field_type() -> None:
    _, fine, refinement, fine_frames, down_cutoff = _hierarchy()
    encode = IcoEncode(
        SO2FieldType({0: 2}), SO2FieldType({0: 3}), cutoff=down_cutoff, dtype=torch.float64
    )
    down_edges = build_icosphere_downsample_edges(
        fine, fine_frames, refinement.n_coarse_nodes, cutoff=down_cutoff
    )
    wrong_type_x = _random_field(
        SO2FieldType({0: 5}), (fine.n_nodes,), generator=torch.Generator().manual_seed(22)
    )

    with pytest.raises(ValueError):
        encode(wrong_type_x, down_edges)


# --------------------------------------------------------------------------
# IcoEncode: gauge equivariance -- the centerpiece
# --------------------------------------------------------------------------


def test_ico_encode_is_exactly_gauge_equivariant() -> None:
    """Regauging every *fine* node must send IcoEncode's coarse output
    through the *same* regauging, restricted to the coarse nodes (the
    fine grid's own first n_coarse_nodes) -- IcoEncode's equivariance
    argument never depended on source and target being drawn from the same
    node set, only on SO2SteerableKernel's own per-edge transport law."""
    _, fine, refinement, fine_frames, down_cutoff = _hierarchy()
    n_coarse = refinement.n_coarse_nodes
    field_type_in = SO2FieldType({0: 2, 1: 2})
    field_type_out = SO2FieldType({0: 3, 1: 3, 2: 3})

    torch.manual_seed(23)
    encode = IcoEncode(
        field_type_in, field_type_out, cutoff=down_cutoff,
        n_radial_basis=6, radial_hidden_features=(8,), dtype=torch.float64,
    )

    generator = torch.Generator().manual_seed(24)
    x = _random_field(field_type_in, (fine.n_nodes,), generator=generator)

    down_edges = build_icosphere_downsample_edges(
        fine, fine_frames, n_coarse, cutoff=down_cutoff
    )
    original_output = encode(x, down_edges).to_flat()

    gauge_angles = torch.rand(fine.n_nodes, dtype=torch.float64, generator=generator) * (
        2 * math.pi
    )
    regauged_fine_frames = rotate_tangent_frames(fine_frames, gauge_angles)
    regauged_down_edges = build_icosphere_downsample_edges(
        fine, regauged_fine_frames, n_coarse, cutoff=down_cutoff
    )
    regauged_x = x.gauge_transform(gauge_angles)

    regauged_output = encode(regauged_x, regauged_down_edges).to_flat()
    expected = (
        SO2FeatureField.from_flat(field_type_out, original_output)
        .gauge_transform(gauge_angles[:n_coarse])
        .to_flat()
    )

    assert torch.allclose(regauged_output, expected, atol=1e-7)


# --------------------------------------------------------------------------
# IcoEncode: autograd
# --------------------------------------------------------------------------


def test_ico_encode_gradients_flow_to_input_and_every_parameter() -> None:
    _, fine, refinement, fine_frames, down_cutoff = _hierarchy()
    torch.manual_seed(25)
    encode = IcoEncode(
        SO2FieldType({0: 2, 1: 2}), SO2FieldType({0: 3, 1: 3}),
        cutoff=down_cutoff, dtype=torch.float64,
    )
    down_edges = build_icosphere_downsample_edges(
        fine, fine_frames, refinement.n_coarse_nodes, cutoff=down_cutoff
    )

    flat = torch.randn(fine.n_nodes, 6, dtype=torch.float64, requires_grad=True)
    x = SO2FeatureField.from_flat(SO2FieldType({0: 2, 1: 2}), flat)

    encode(x, down_edges).to_flat().sum().backward()

    assert flat.grad is not None and torch.isfinite(flat.grad).all()
    for name, parameter in encode.named_parameters():
        assert parameter.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(parameter.grad).all()


# --------------------------------------------------------------------------
# IcoDecode: forward shape
# --------------------------------------------------------------------------


def test_ico_decode_output_batch_shape_is_the_fine_node_count() -> None:
    coarse, fine, refinement, fine_frames, _ = _hierarchy()
    field_type_in = SO2FieldType({0: 3, 1: 3, 2: 3})
    field_type_out = SO2FieldType({0: 2, 1: 2})
    fine_cutoff = 0.5 * median_nearest_neighbor_distance(fine)

    torch.manual_seed(26)
    decode = IcoDecode(field_type_in, field_type_out, cutoff=fine_cutoff, dtype=torch.float64)

    up_edges = build_icosphere_upsample_edges(fine, fine_frames, refinement)
    fine_edges = compute_edge_geometry(
        fine, fine_frames, fine_cutoff, max_frequency=field_type_in.max_frequency + field_type_out.max_frequency
    )

    generator = torch.Generator().manual_seed(27)
    x = _random_field(field_type_in, (coarse.n_nodes,), generator=generator)

    output = decode(x, up_edges, fine_edges)

    assert output.field_type == field_type_out
    assert output.batch_shape == (fine.n_nodes,)


def test_ico_decode_rejects_a_mismatched_input_field_type() -> None:
    _, fine, refinement, fine_frames, _ = _hierarchy()
    fine_cutoff = 0.5 * median_nearest_neighbor_distance(fine)
    decode = IcoDecode(
        SO2FieldType({0: 3}), SO2FieldType({0: 2}), cutoff=fine_cutoff, dtype=torch.float64
    )
    up_edges = build_icosphere_upsample_edges(fine, fine_frames, refinement)
    fine_edges = compute_edge_geometry(fine, fine_frames, fine_cutoff, max_frequency=4)

    wrong_type_x = _random_field(
        SO2FieldType({0: 5}), (refinement.n_coarse_nodes,), generator=torch.Generator().manual_seed(28)
    )

    with pytest.raises(ValueError):
        decode(wrong_type_x, up_edges, fine_edges)


# --------------------------------------------------------------------------
# IcoDecode: the two-region upsample step, isolated from the final conv
# --------------------------------------------------------------------------


def test_ico_decode_inherited_region_is_an_exact_untransported_copy() -> None:
    """Stub the internal conv to the identity (valid since field_type_in ==
    field_type_out here) to isolate step 1 alone, then check its first
    n_coarse_nodes rows equal the coarse input exactly -- same node, same
    gauge, so a direct copy, not merely "close" or "correctly transported"."""
    coarse, fine, refinement, fine_frames, _ = _hierarchy()
    field_type = SO2FieldType({0: 3, 1: 3, 2: 3})
    fine_cutoff = 0.5 * median_nearest_neighbor_distance(fine)

    torch.manual_seed(29)
    decode = IcoDecode(field_type, field_type, cutoff=fine_cutoff, dtype=torch.float64)
    decode.conv.forward = lambda x, edges: x  # type: ignore[method-assign]

    up_edges = build_icosphere_upsample_edges(fine, fine_frames, refinement)
    fine_edges = compute_edge_geometry(fine, fine_frames, fine_cutoff, max_frequency=field_type.max_frequency * 2)

    generator = torch.Generator().manual_seed(30)
    x = _random_field(field_type, (coarse.n_nodes,), generator=generator)

    output = decode(x, up_edges, fine_edges)

    assert torch.equal(
        output.to_flat()[: refinement.n_coarse_nodes],
        x.to_flat(),
    )


def test_ico_decode_midpoint_region_is_the_transported_parent_average() -> None:
    """Same isolation as above, but checks the *other* region: a new fine
    node's value must equal the average of its two coarse parents, each
    transported into its own gauge -- computed independently here via
    transport_field (already tested on its own in test_kernels.py /
    test_edge_geometry.py), not by re-deriving IcoDecode's own math."""
    coarse, fine, refinement, fine_frames, _ = _hierarchy()
    field_type = SO2FieldType({0: 3, 1: 3, 2: 3})
    fine_cutoff = 0.5 * median_nearest_neighbor_distance(fine)

    torch.manual_seed(31)
    decode = IcoDecode(field_type, field_type, cutoff=fine_cutoff, dtype=torch.float64)
    decode.conv.forward = lambda x, edges: x  # type: ignore[method-assign]

    up_edges = build_icosphere_upsample_edges(fine, fine_frames, refinement)
    fine_edges = compute_edge_geometry(fine, fine_frames, fine_cutoff, max_frequency=field_type.max_frequency * 2)

    generator = torch.Generator().manual_seed(32)
    x = _random_field(field_type, (coarse.n_nodes,), generator=generator)

    output = decode(x, up_edges, fine_edges)

    independently_computed = SO2FeatureField(
        field_type,
        {
            frequency: 0.5
            * scatter_add_to_nodes(tensor, up_edges.row_ptr)
            for frequency, tensor in transport_field(
                gather_to_edges(x, up_edges.source_index), up_edges
            ).coefficients.items()
        },
    )

    assert torch.allclose(
        output.to_flat()[refinement.n_coarse_nodes :],
        independently_computed.to_flat()[refinement.n_coarse_nodes :],
        atol=1e-12,
    )


# --------------------------------------------------------------------------
# IcoDecode: gauge equivariance -- the centerpiece
# --------------------------------------------------------------------------


def test_ico_decode_is_exactly_gauge_equivariant() -> None:
    _, fine, refinement, fine_frames, _ = _hierarchy()
    n_coarse = refinement.n_coarse_nodes
    field_type_in = SO2FieldType({0: 3, 1: 3, 2: 3})
    field_type_out = SO2FieldType({0: 2, 1: 2})
    fine_cutoff = 0.5 * median_nearest_neighbor_distance(fine)
    max_frequency = field_type_in.max_frequency + field_type_out.max_frequency

    torch.manual_seed(33)
    decode = IcoDecode(
        field_type_in, field_type_out, cutoff=fine_cutoff,
        n_radial_basis=6, radial_hidden_features=(8,), dtype=torch.float64,
    )

    generator = torch.Generator().manual_seed(34)
    x = _random_field(field_type_in, (n_coarse,), generator=generator)

    up_edges = build_icosphere_upsample_edges(fine, fine_frames, refinement)
    fine_edges = compute_edge_geometry(fine, fine_frames, fine_cutoff, max_frequency=max_frequency)
    original_output = decode(x, up_edges, fine_edges).to_flat()

    gauge_angles = torch.rand(fine.n_nodes, dtype=torch.float64, generator=generator) * (
        2 * math.pi
    )
    regauged_fine_frames = rotate_tangent_frames(fine_frames, gauge_angles)
    regauged_up_edges = build_icosphere_upsample_edges(fine, regauged_fine_frames, refinement)
    regauged_fine_edges = compute_edge_geometry(
        fine, regauged_fine_frames, fine_cutoff, max_frequency=max_frequency
    )
    regauged_x = x.gauge_transform(gauge_angles[:n_coarse])

    regauged_output = decode(regauged_x, regauged_up_edges, regauged_fine_edges).to_flat()
    expected = (
        SO2FeatureField.from_flat(field_type_out, original_output)
        .gauge_transform(gauge_angles)
        .to_flat()
    )

    assert torch.allclose(regauged_output, expected, atol=1e-7)


# --------------------------------------------------------------------------
# IcoDecode: autograd
# --------------------------------------------------------------------------


def test_ico_decode_gradients_flow_to_input_and_every_parameter() -> None:
    coarse, fine, refinement, fine_frames, _ = _hierarchy()
    fine_cutoff = 0.5 * median_nearest_neighbor_distance(fine)
    field_type_in = SO2FieldType({0: 3, 1: 3})
    field_type_out = SO2FieldType({0: 2, 1: 2})

    torch.manual_seed(35)
    decode = IcoDecode(field_type_in, field_type_out, cutoff=fine_cutoff, dtype=torch.float64)
    up_edges = build_icosphere_upsample_edges(fine, fine_frames, refinement)
    fine_edges = compute_edge_geometry(
        fine, fine_frames, fine_cutoff, max_frequency=field_type_in.max_frequency + field_type_out.max_frequency
    )

    flat = torch.randn(coarse.n_nodes, field_type_in.dimension, dtype=torch.float64, requires_grad=True)
    x = SO2FeatureField.from_flat(field_type_in, flat)

    decode(x, up_edges, fine_edges).to_flat().sum().backward()

    assert flat.grad is not None and torch.isfinite(flat.grad).all()
    for name, parameter in decode.named_parameters():
        assert parameter.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(parameter.grad).all()


# --------------------------------------------------------------------------
# End to end: encode then decode composes without shape/type errors
# --------------------------------------------------------------------------


def test_encode_then_decode_round_trips_field_type_and_node_count() -> None:
    coarse, fine, refinement, fine_frames, down_cutoff = _hierarchy()
    fine_type = SO2FieldType({0: 2, 1: 2})
    latent_type = SO2FieldType({0: 4, 1: 4, 2: 4})
    fine_cutoff = 0.5 * median_nearest_neighbor_distance(fine)

    torch.manual_seed(36)
    encode = IcoEncode(fine_type, latent_type, cutoff=down_cutoff, dtype=torch.float64)
    decode = IcoDecode(latent_type, fine_type, cutoff=fine_cutoff, dtype=torch.float64)

    down_edges = build_icosphere_downsample_edges(
        fine, fine_frames, refinement.n_coarse_nodes, cutoff=down_cutoff
    )
    up_edges = build_icosphere_upsample_edges(fine, fine_frames, refinement)
    fine_edges = compute_edge_geometry(
        fine, fine_frames, fine_cutoff, max_frequency=latent_type.max_frequency + fine_type.max_frequency
    )

    generator = torch.Generator().manual_seed(37)
    x = _random_field(fine_type, (fine.n_nodes,), generator=generator)

    latent = encode(x, down_edges)
    assert latent.batch_shape == (coarse.n_nodes,)

    reconstructed = decode(latent, up_edges, fine_edges)
    assert reconstructed.field_type == fine_type
    assert reconstructed.batch_shape == (fine.n_nodes,)
    assert all(torch.isfinite(t).all() for t in reconstructed.coefficients.values())


# ==========================================================================
# IcosphereMultiresolution: encode/block(s)/decode composition
# ==========================================================================
#
# Pure-geometry facts about build_icosphere_multiresolution itself (node
# counts, transitive prefixes, the local_cutoffs override) live in
# test_grids.py alongside the pairwise primitives it composes. These tests
# cover the nn-level composition: encode/decode/transformer-block layers
# actually working together through that geometry, at depth 1 (the shipped
# depth-1 model's own pattern) and depth 2 (proving the generalization).


def test_multiresolution_geometry_node_counts_and_cutoffs() -> None:
    geometry = build_icosphere_multiresolution(3, depth=1, coarsest_cutoff=0.3, dtype=torch.float64)

    assert geometry.finest.n_nodes == build_icosphere_grid(3, dtype=torch.float64).n_nodes
    assert geometry.coarsest.n_nodes == build_icosphere_grid(2, dtype=torch.float64).n_nodes
    assert geometry.hops[0].down_edges.row_ptr.shape == (geometry.coarsest.n_nodes + 1,)
    assert geometry.hops[0].up_edges.row_ptr.shape == (geometry.finest.n_nodes + 1,)
    assert geometry.coarsest.coverage_radius == 0.3


def test_multiresolution_end_to_end_through_encode_block_decode() -> None:
    field_type = SO2FieldType({0: 2, 1: 2})
    hidden_type = SO2FieldType({0: 4, 1: 4, 2: 4})

    torch.manual_seed(38)
    geometry = build_icosphere_multiresolution(3, depth=1, coarsest_cutoff=0.3, dtype=torch.float64)
    encode_decode_cutoff = geometry.hops[0].refinement.coarse_edge_length

    encode = IcoEncode(field_type, hidden_type, cutoff=encode_decode_cutoff, dtype=torch.float64)
    block = SO2TransformerBlock(
        hidden_type, 2, ffn_multiplier=2, cutoff=geometry.coarsest.coverage_radius, dtype=torch.float64
    )
    decode = IcoDecode(hidden_type, field_type, cutoff=encode_decode_cutoff, dtype=torch.float64)

    generator = torch.Generator().manual_seed(39)
    x = _random_field(field_type, (geometry.finest.n_nodes,), generator=generator)

    latent = block(encode(x, geometry.hops[0].down_edges), geometry.coarsest)
    output = decode(latent, geometry.hops[0].up_edges, geometry.finest)

    assert output.field_type == field_type
    assert output.batch_shape == (geometry.finest.n_nodes,)
    assert all(torch.isfinite(t).all() for t in output.coefficients.values())


def test_multiresolution_bare_composition_is_exactly_gauge_equivariant_at_depth_one() -> None:
    """Closes a real pre-existing gap: test_encode_then_decode_round_trips_field_type_and_node_count
    above only checks shape/finiteness for the bare (no model wrapper)
    composition -- equivariance of this exact composition was previously
    proven only indirectly, at the packaged-model level
    (tests/experiments/shallow_water/test_models.py::test_depth1_is_exactly_gauge_equivariant)."""
    field_type = SO2FieldType({0: 2, 1: 2})
    hidden_type = SO2FieldType({0: 4, 1: 4, 2: 4})

    torch.manual_seed(50)
    finest_frames = build_tangent_frames(build_icosphere_grid(3, dtype=torch.float64).nodes)
    geometry = build_icosphere_multiresolution(
        3, depth=1, coarsest_cutoff=0.3, finest_frames=finest_frames, dtype=torch.float64
    )
    encode_decode_cutoff = geometry.hops[0].refinement.coarse_edge_length

    encode = IcoEncode(field_type, hidden_type, cutoff=encode_decode_cutoff, dtype=torch.float64)
    block = SO2TransformerBlock(
        hidden_type, 2, ffn_multiplier=2, cutoff=geometry.coarsest.coverage_radius, dtype=torch.float64
    )
    decode = IcoDecode(hidden_type, field_type, cutoff=encode_decode_cutoff, dtype=torch.float64)

    def run(g: object, x: SO2FeatureField) -> SO2FeatureField:
        latent = block(encode(x, g.hops[0].down_edges), g.coarsest)  # type: ignore[attr-defined]
        return decode(latent, g.hops[0].up_edges, g.finest)  # type: ignore[attr-defined]

    generator = torch.Generator().manual_seed(51)
    x = _random_field(field_type, (geometry.finest.n_nodes,), generator=generator)
    original_output = run(geometry, x).to_flat()

    gauge_angles = torch.rand(geometry.finest.n_nodes, dtype=torch.float64, generator=generator) * (
        2 * math.pi
    )
    regauged_frames = rotate_tangent_frames(finest_frames, gauge_angles)
    regauged_geometry = build_icosphere_multiresolution(
        3, depth=1, coarsest_cutoff=0.3, finest_frames=regauged_frames, dtype=torch.float64
    )
    regauged_x = x.gauge_transform(gauge_angles)

    regauged_output = run(regauged_geometry, regauged_x).to_flat()
    expected = (
        SO2FeatureField.from_flat(field_type, original_output).gauge_transform(gauge_angles).to_flat()
    )

    assert torch.allclose(regauged_output, expected, atol=1e-7)


def test_multiresolution_bare_composition_is_exactly_gauge_equivariant_at_depth_two() -> None:
    """The depth-2 generalization: two IcoEncode/IcoDecode steps, with an
    SO2TransformerBlock run at the intermediate level too (not just the
    coarsest) -- directly exercising levels[1]'s own discretization from
    both an independent SO2TransformerBlock call and IcoDecode's internal
    conv, the one plausible depth-2 architecture variant (per-level
    processing) beyond what the shipped depth-1 model exercises."""
    field_type = SO2FieldType({0: 2, 1: 2})
    hidden_type = SO2FieldType({0: 4, 1: 4, 2: 4})

    torch.manual_seed(52)
    finest_frames = build_tangent_frames(build_icosphere_grid(3, dtype=torch.float64).nodes)
    geometry = build_icosphere_multiresolution(
        3, depth=2, coarsest_cutoff=0.5, finest_frames=finest_frames, dtype=torch.float64
    )

    encode0 = IcoEncode(
        field_type, hidden_type, cutoff=geometry.hops[0].refinement.coarse_edge_length, dtype=torch.float64
    )
    encode1 = IcoEncode(
        hidden_type, hidden_type, cutoff=geometry.hops[1].refinement.coarse_edge_length, dtype=torch.float64
    )
    block_mid = SO2TransformerBlock(
        hidden_type, 2, ffn_multiplier=2, cutoff=geometry.levels[1].coverage_radius, dtype=torch.float64
    )
    block_coarse = SO2TransformerBlock(
        hidden_type, 2, ffn_multiplier=2, cutoff=geometry.levels[2].coverage_radius, dtype=torch.float64
    )
    decode1 = IcoDecode(
        hidden_type, hidden_type, cutoff=geometry.hops[1].refinement.coarse_edge_length, dtype=torch.float64
    )
    decode0 = IcoDecode(
        hidden_type, field_type, cutoff=geometry.hops[0].refinement.coarse_edge_length, dtype=torch.float64
    )

    def run(g: object, x: SO2FeatureField) -> SO2FeatureField:
        x1 = block_mid(encode0(x, g.hops[0].down_edges), g.levels[1])  # type: ignore[attr-defined]
        x2 = block_coarse(encode1(x1, g.hops[1].down_edges), g.levels[2])  # type: ignore[attr-defined]
        x1b = decode1(x2, g.hops[1].up_edges, g.levels[1])  # type: ignore[attr-defined]
        return decode0(x1b, g.hops[0].up_edges, g.levels[0])  # type: ignore[attr-defined]

    generator = torch.Generator().manual_seed(53)
    x = _random_field(field_type, (geometry.finest.n_nodes,), generator=generator)
    original_output = run(geometry, x).to_flat()

    gauge_angles = torch.rand(geometry.finest.n_nodes, dtype=torch.float64, generator=generator) * (
        2 * math.pi
    )
    regauged_frames = rotate_tangent_frames(finest_frames, gauge_angles)
    regauged_geometry = build_icosphere_multiresolution(
        3, depth=2, coarsest_cutoff=0.5, finest_frames=regauged_frames, dtype=torch.float64
    )
    regauged_x = x.gauge_transform(gauge_angles)

    regauged_output = run(regauged_geometry, regauged_x).to_flat()
    expected = (
        SO2FeatureField.from_flat(field_type, original_output).gauge_transform(gauge_angles).to_flat()
    )

    assert torch.allclose(regauged_output, expected, atol=1e-7)


# ==========================================================================
# IcoLowPassRestriction -- the alternative, normalized encode path
# ==========================================================================


def _restriction_geometry(subdivisions: int = 3):
    return build_icosphere_multiresolution(
        subdivisions, depth=1, coarsest_cutoff=0.6, dtype=torch.float64
    )


def _restriction(field_type, geometry, **overrides):
    return IcoLowPassRestriction(
        field_type,
        cutoff=geometry.hops[0].refinement.coarse_edge_length,
        **overrides,
    )


def test_low_pass_restriction_preserves_scalar_constants_exactly() -> None:
    """The defining property, on genuinely nonuniform quadrature weights --
    an unnormalized aggregation gets this wrong."""
    geometry = _restriction_geometry()
    field_type = SO2FieldType({0: 2})
    restriction = _restriction(field_type, geometry)

    constant = SO2FeatureField(
        field_type,
        {0: torch.full((geometry.levels[0].n_nodes, 2), 3.5, dtype=torch.float64)},
    )
    output = restriction(constant, geometry.hops[0].down_edges)

    assert torch.allclose(
        output[0], torch.full_like(output[0], 3.5), atol=1e-12
    )


def test_low_pass_restriction_output_lives_on_the_coarse_grid() -> None:
    geometry = _restriction_geometry()
    field_type = SO2FieldType({0: 1, 1: 1})
    restriction = _restriction(field_type, geometry)

    x = SO2FeatureField.from_flat(
        field_type,
        torch.randn(geometry.levels[0].n_nodes, field_type.dimension, dtype=torch.float64),
    )
    output = restriction(x, geometry.hops[0].down_edges)

    assert output.field_type == field_type  # type is unchanged, by construction
    assert output.batch_shape == (geometry.levels[1].n_nodes,)


def test_low_pass_restriction_commutes_with_a_random_gauge_change() -> None:
    geometry = _restriction_geometry()
    field_type = SO2FieldType({0: 1, 1: 1, 2: 1})
    restriction = _restriction(field_type, geometry)

    x = SO2FeatureField.from_flat(
        field_type,
        torch.randn(geometry.levels[0].n_nodes, field_type.dimension, dtype=torch.float64),
    )
    gauge_angles = (
        torch.rand(geometry.levels[0].n_nodes, dtype=torch.float64) * 2 * math.pi
    )

    regauged_geometry = build_icosphere_multiresolution(
        3,
        depth=1,
        coarsest_cutoff=0.6,
        finest_frames=rotate_tangent_frames(geometry.levels[0].frames, gauge_angles),
        dtype=torch.float64,
    )

    original = restriction(x, geometry.hops[0].down_edges)
    regauged = restriction(
        x.gauge_transform(gauge_angles), regauged_geometry.hops[0].down_edges
    )

    # The coarse level's own nodes are a prefix of the fine level's, so the
    # coarse gauge rotated by the same prefix of angles.
    expected = original.gauge_transform(gauge_angles[: original.batch_shape[0]])

    assert torch.allclose(regauged.to_flat(), expected.to_flat(), atol=1e-9)


def test_low_pass_restriction_preserves_a_transported_constant_vector_field() -> None:
    """The geometric analogue of constant preservation for a rho_1 field: a
    covariantly-constant field must survive the average unchanged."""
    geometry = _restriction_geometry()
    field_type = SO2FieldType({1: 1})
    restriction = _restriction(field_type, geometry)

    # A field whose ambient value is one fixed vector everywhere, expressed
    # in each node's own gauge -- i.e. constant up to transport.
    frames = geometry.levels[0].frames  # (n, 3, 2)
    ambient = torch.tensor([0.3, -0.7, 0.5], dtype=torch.float64)
    tangential = ambient - (frames.new_zeros(1) + geometry.levels[0].nodes) * (
        geometry.levels[0].nodes @ ambient
    ).unsqueeze(-1)
    gauge = torch.einsum("nij,ni->nj", frames, tangential)  # (n, 2)

    x = SO2FeatureField(field_type, {1: gauge.unsqueeze(-2)})
    output = restriction(x, geometry.hops[0].down_edges)

    # Recover the ambient vector at the coarse nodes and compare.
    n_coarse = output.batch_shape[0]
    coarse_frames = frames[:n_coarse]
    recovered = torch.einsum("nij,nj->ni", coarse_frames, output[1][:, 0, :])
    expected = ambient - geometry.levels[1].nodes * (
        geometry.levels[1].nodes @ ambient
    ).unsqueeze(-1)

    # Curvature means the average of transported vectors is not exact, only
    # close -- so this is a genuine tolerance, not machine precision.
    assert torch.allclose(recovered, expected, atol=0.05)


def test_low_pass_restriction_attenuates_high_frequencies_more_than_low() -> None:
    """The actual anti-aliasing check. Constant preservation alone is not
    evidence of a low-pass operator -- a plain identity preserves constants
    too. What matters is that a rapidly-oscillating signal loses more of its
    amplitude than a slowly-varying one.
    """
    geometry = _restriction_geometry(subdivisions=4)
    field_type = SO2FieldType({0: 1})
    restriction = _restriction(field_type, geometry)

    fine_nodes = geometry.levels[0].nodes
    coarse_nodes = geometry.levels[1].nodes

    def retained_amplitude(frequency: int) -> float:
        # A zonal oscillation in z: higher `frequency` oscillates faster.
        fine = torch.cos(frequency * torch.pi * fine_nodes[:, 2])
        coarse_truth = torch.cos(frequency * torch.pi * coarse_nodes[:, 2])

        output = restriction(
            SO2FeatureField(field_type, {0: fine.unsqueeze(-1)}),
            geometry.hops[0].down_edges,
        )
        return float(output[0][:, 0].std() / coarse_truth.std())

    low = retained_amplitude(1)
    high = retained_amplitude(8)

    assert high < low
    assert low > 0.9  # a slowly-varying signal survives essentially intact
    assert high < 0.7  # a rapidly-oscillating one is genuinely damped


def test_low_pass_restriction_and_ico_encode_are_both_available() -> None:
    """IcoEncode must not have been replaced -- they are alternatives."""
    geometry = _restriction_geometry()
    field_type = SO2FieldType({0: 1, 1: 1})

    encode = IcoEncode(
        field_type,
        SO2FieldType({0: 2, 1: 2, 2: 2}),
        cutoff=geometry.hops[0].refinement.coarse_edge_length,
        dtype=torch.float64,
    )
    restriction = _restriction(field_type, geometry)

    x = SO2FeatureField.from_flat(
        field_type,
        torch.randn(geometry.levels[0].n_nodes, field_type.dimension, dtype=torch.float64),
    )

    encoded = encode(x, geometry.hops[0].down_edges)
    restricted = restriction(x, geometry.hops[0].down_edges)

    # Same node count, different types: the learned path changes type, the
    # normalized path cannot.
    assert encoded.batch_shape == restricted.batch_shape
    assert encoded.field_type != restricted.field_type


def test_low_pass_restriction_has_no_learnable_parameters() -> None:
    restriction = _restriction(SO2FieldType({0: 1}), _restriction_geometry())
    assert list(restriction.parameters()) == []


def test_low_pass_restriction_rejects_an_unknown_window() -> None:
    with pytest.raises(ValueError, match="window"):
        IcoLowPassRestriction(SO2FieldType({0: 1}), cutoff=0.3, window="gaussian")  # type: ignore[arg-type]


def test_low_pass_restriction_rejects_a_mismatched_field_type() -> None:
    geometry = _restriction_geometry()
    restriction = _restriction(SO2FieldType({0: 1}), geometry)
    wrong = SO2FeatureField.from_flat(
        SO2FieldType({0: 2}),
        torch.randn(geometry.levels[0].n_nodes, 2, dtype=torch.float64),
    )

    with pytest.raises(ValueError):
        restriction(wrong, geometry.hops[0].down_edges)


# ==========================================================================
# SO2TransformerBlock nonlinearity selection
# ==========================================================================
#
# The supplement ships the two scalar-gated kinds only (the norm-based
# SO2NormSiLU / SO2NormReLU of the development repository were never used by
# the reported experiments), so "every kind" below means those two.


@pytest.mark.parametrize("nonlinearity", ["scalar_gate", "scalar_gate_double_sigmoid"])
def test_every_nonlinearity_kind_runs_and_stays_equivariant(nonlinearity: str) -> None:
    grid = build_icosphere_grid(subdivisions=1, dtype=torch.float64)
    frames = build_tangent_frames(grid.nodes)
    coverage_radius = 0.6
    field_type = SO2FieldType({0: 2, 1: 2})

    torch.manual_seed(70)
    block = SO2TransformerBlock(
        field_type,
        1,
        ffn_multiplier=2,
        cutoff=coverage_radius,
        nonlinearity=nonlinearity,  # type: ignore[arg-type]
        dtype=torch.float64,
    )

    edges = compute_edge_geometry(grid, frames, coverage_radius, max_frequency=2)
    x = SO2FeatureField.from_flat(
        field_type, torch.randn(grid.n_nodes, field_type.dimension, dtype=torch.float64)
    )
    gauge_angles = torch.rand(grid.n_nodes, dtype=torch.float64) * 2 * math.pi
    regauged_edges = compute_edge_geometry(
        grid, rotate_tangent_frames(frames, gauge_angles), coverage_radius, max_frequency=2
    )

    regauged = block(x.gauge_transform(gauge_angles), regauged_edges).to_flat()
    expected = block(x, edges).gauge_transform(gauge_angles).to_flat()

    assert torch.allclose(regauged, expected, atol=1e-8)


def test_transformer_block_rejects_an_unknown_nonlinearity() -> None:
    with pytest.raises(ValueError, match="nonlinearity"):
        SO2TransformerBlock(
            SO2FieldType({0: 1}),
            1,
            ffn_multiplier=2,
            cutoff=0.5,
            nonlinearity="gelu",  # type: ignore[arg-type]
        )


# ==========================================================================
# Heterogeneous field types (spec 7.5)
# ==========================================================================


@pytest.mark.parametrize("num_heads", [1, 2, 4])
def test_heterogeneous_block_constructs_and_runs_for_every_valid_head_count(
    num_heads: int,
) -> None:
    field_type = SO2FieldType({0: 8, 1: 4})
    block = SO2TransformerBlock(
        field_type, num_heads, ffn_multiplier=2, cutoff=0.6 * math.pi,
        dtype=torch.float64,
    )
    grid, _, _, edges = _build_grid_and_edges(field_type)

    output = block(
        _random_field(
            field_type, (grid.n_nodes,), generator=torch.Generator().manual_seed(3)
        ),
        edges,
    )

    assert output.field_type == field_type
    assert block.attention.field_type_attn == SO2FieldType(
        {0: 8 // num_heads, 1: 4 // num_heads}
    )


def test_head_concatenation_reconstructs_the_requested_multiplicities() -> None:
    """Each head carries a_n / h copies of frequency n; concatenating h of
    them frequency-wise must return exactly a_n, not a reshuffled total."""
    field_type = SO2FieldType({0: 8, 1: 4, 2: 12})
    block = SO2TransformerBlock(
        field_type, 4, ffn_multiplier=2, cutoff=0.6 * math.pi, dtype=torch.float64
    )
    grid, _, _, edges = _build_grid_and_edges(field_type)

    attended = block.attention(
        _random_field(
            field_type, (grid.n_nodes,), generator=torch.Generator().manual_seed(4)
        ),
        edges,
    )

    assert attended.field_type == field_type
    assert attended[0].shape == (grid.n_nodes, 8)
    assert attended[1].shape == (grid.n_nodes, 4, 2)
    assert attended[2].shape == (grid.n_nodes, 12, 2)


def test_heterogeneous_block_is_exactly_gauge_equivariant() -> None:
    """The claim that matters: generalizing the multiplicity bookkeeping
    must not cost the exactness the uniform path already has."""
    field_type = SO2FieldType({0: 8, 1: 4, 2: 12})
    torch.manual_seed(31)
    block = SO2TransformerBlock(
        field_type, 4, ffn_multiplier=2, cutoff=0.6 * math.pi, dtype=torch.float64
    ).eval()

    grid, frames, coverage_radius, edges = _build_grid_and_edges(field_type)
    generator = torch.Generator().manual_seed(32)
    node_features = _random_field(field_type, (grid.n_nodes,), generator=generator)

    original_output = block(node_features, edges).to_flat()

    gauge_angles = torch.rand(
        grid.n_nodes, dtype=torch.float64, generator=generator
    ) * (2 * math.pi)
    regauged_edges = compute_edge_geometry(
        grid,
        rotate_tangent_frames(frames, gauge_angles),
        coverage_radius,
        max_frequency=field_type.max_frequency * 2,
    )

    regauged_output = block(
        node_features.gauge_transform(gauge_angles), regauged_edges
    ).to_flat()
    expected = (
        SO2FeatureField.from_flat(field_type, original_output)
        .gauge_transform(gauge_angles)
        .to_flat()
    )

    assert torch.allclose(regauged_output, expected, atol=1e-7)


def test_heterogeneous_block_has_finite_gradients_for_every_frequency() -> None:
    field_type = SO2FieldType({0: 8, 1: 4, 2: 12})
    block = SO2TransformerBlock(
        field_type, 2, ffn_multiplier=2, cutoff=0.6 * math.pi, dtype=torch.float64
    )
    grid, _, _, edges = _build_grid_and_edges(field_type)

    features = _random_field(
        field_type, (grid.n_nodes,), generator=torch.Generator().manual_seed(5)
    )
    block(features, edges).to_flat().sum().backward()

    grads = {
        name: parameter.grad
        for name, parameter in block.named_parameters()
        if parameter.grad is not None
    }
    assert grads, "no parameter received a gradient"
    for name, grad in grads.items():
        assert torch.isfinite(grad).all(), f"non-finite gradient in {name}"

    # every frequency's layer-norm gain is reached, not just frequency 0
    for frequency in field_type.frequencies:
        assert f"norm1.gamma.{frequency}" in grads


# ==========================================================================
# IcoDecode: chunked evaluation on a multiresolution geometry
# ==========================================================================


def _decode_geometry():
    from gm_attention.discretization.grids.icosphere import (
        build_icosphere_multiresolution,
    )

    return build_icosphere_multiresolution(
        2, depth=1, coarsest_cutoff=0.55, dtype=torch.float64
    )


def test_ico_decode_chunked_matches_unchunked() -> None:
    """IcoDecode.forward_chunked must reproduce forward to summation order.
    The development repository checks this on its enriched-skip IcoDecode
    (skip_field_type / skip=); the supplement ships the plain decode only,
    so the same check, seeds, chunk size and tolerance run without a skip."""
    geometry = _decode_geometry()
    hidden = SO2FieldType({0: 2, 1: 2})
    torch.manual_seed(25)
    decode = IcoDecode(
        hidden,
        SO2FieldType({0: 1, 1: 1}),
        cutoff=0.4,
        dtype=torch.float64,
    )
    generator = torch.Generator().manual_seed(26)
    coarse = _random_field(hidden, (geometry.levels[1].grid.n_nodes,), generator=generator)

    full = decode(coarse, geometry.hops[0].up_edges, geometry.levels[0])
    chunked = decode.forward_chunked(
        coarse, geometry.hops[0].up_edges, geometry.levels[0],
        chunk_size=37,
    )
    assert torch.allclose(full.to_flat(), chunked.to_flat(), atol=1e-13)


@pytest.mark.parametrize("mode", ["0", "cpu"])
def test_ico_encode_takes_the_factored_path_to_the_same_values(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    """Under GM_FACTORED_KERNEL the merge's integral runs through the kernel's
    factored order (eager, or the fused kernel's CPU driver), the same
    function to summation order; gradients too."""
    from gm_attention.nn import fused_factored, kernels

    if mode == "cpu":
        try:
            fused_factored.extension(cuda=False)
        except Exception as error:  # noqa: BLE001
            pytest.skip(f"fused factored extension did not build here: {error}")
    coarse, fine, refinement, fine_frames, down_cutoff = _hierarchy()
    field_type_in = SO2FieldType({0: 2, 1: 2})
    field_type_out = SO2FieldType({0: 3, 1: 3, 2: 3})
    torch.manual_seed(20)
    encode = IcoEncode(field_type_in, field_type_out, cutoff=down_cutoff, dtype=torch.float64)
    down_edges = build_icosphere_downsample_edges(fine, fine_frames, refinement.n_coarse_nodes, cutoff=down_cutoff)
    generator = torch.Generator().manual_seed(21)
    x = _random_field(field_type_in, (fine.n_nodes,), generator=generator)
    for f in x.field_type.frequencies:
        x[f].requires_grad_(True)

    monkeypatch.setattr(kernels, "FACTORED", False)
    assert not encode._factored()
    general = encode(x, down_edges)
    loss = sum((general[f] ** 2).sum() for f in general.field_type.frequencies)
    reference = torch.autograd.grad(loss, [x[f] for f in x.field_type.frequencies] + list(encode.parameters()))

    monkeypatch.setattr(kernels, "FACTORED", True)
    monkeypatch.setattr(fused_factored, "MODE", mode)
    assert encode._factored()
    ops: list[str] = []
    original_apply = fused_factored.FactoredAggregate.apply
    monkeypatch.setattr(fused_factored.FactoredAggregate, "apply", lambda *a, **k: (ops.append("aggregate"), original_apply(*a, **k))[1])
    factored = encode(x, down_edges)
    assert ops == (["aggregate"] if mode == "cpu" else [])  # the bipartite sum through the driver's aggregated op
    assert factored.field_type == field_type_out and factored.batch_shape == (coarse.n_nodes,)
    for f in field_type_out.frequencies:
        torch.testing.assert_close(factored[f], general[f], rtol=1e-12, atol=1e-12)
    loss = sum((factored[f] ** 2).sum() for f in factored.field_type.frequencies)
    grads = torch.autograd.grad(loss, [x[f] for f in x.field_type.frequencies] + list(encode.parameters()))
    for a, b in zip(grads, reference, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)
