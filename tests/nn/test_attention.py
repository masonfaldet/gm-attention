"""Tests for gm_attention.nn.attention.

Two things carry the most risk here and get the most scrutiny:

1. The query/key evaluation order (project to rho_attn per node, transport
   second) is claimed to be *exactly* equal to the naive order (transport
   per edge, then project) -- not an approximation. This is checked directly
   against a from-scratch SO2Linear-based reference, and again in situ
   against SO2Attention's actual forward pass.
2. The softmax-style normalization: attention weights must sum to exactly 1
   per target node, and the max-subtraction stabilization must change
   nothing about the result (only what stays finite in float32).
"""

from __future__ import annotations

import math

import pytest
import torch

from gm_attention.discretization.discretization import SphericalDiscretization
from gm_attention.discretization.edge_geometry import (
    EdgeGeometry,
    compute_edge_geometry,
)
from gm_attention.discretization.frames import (
    build_tangent_frames,
    rotate_tangent_frames,
)
from gm_attention.discretization.grids.icosphere import build_icosphere_grid
from gm_attention.nn.attention import SO2MHSA, SO2Attention
from gm_attention.nn.linear import SO2Linear
from gm_attention.representations import so2
from gm_attention.representations.field_type import (
    SO2FeatureField,
    SO2FieldType,
)

ATOL = 1e-8


def _synthetic_edges(
    target_index: torch.Tensor,
    source_index: torch.Tensor,
    n_nodes: int,
    distances: torch.Tensor,
    polar_angles: torch.Tensor,
    transport_angles: torch.Tensor,
    tangent_quadrature_weights: torch.Tensor,
    *,
    max_frequency: int,
) -> EdgeGeometry:
    edge_index = torch.stack((target_index, source_index))
    counts = torch.bincount(target_index, minlength=n_nodes)
    row_ptr = torch.zeros(n_nodes + 1, dtype=torch.long)
    torch.cumsum(counts, dim=0, out=row_ptr[1:])

    log_coordinates = distances.unsqueeze(-1) * torch.stack(
        (torch.cos(polar_angles), torch.sin(polar_angles)), dim=-1
    )

    return EdgeGeometry(
        edge_index=edge_index,
        row_ptr=row_ptr,
        log_coordinates=log_coordinates,
        distances=distances,
        polar_angles=polar_angles,
        transport_angles=transport_angles,
        tangent_quadrature_weights=tangent_quadrature_weights,
        angular_phases=so2.phase_powers(polar_angles, max_frequency),
        transport_phases=so2.phase_powers(transport_angles, max_frequency),
    )


def _multi_neighbor_edges(max_frequency: int) -> EdgeGeometry:
    """Two target nodes (0, 1), each with a self-loop plus two distinct
    neighbors -- matches tests/nn/test_convolution.py's fixture."""
    target_index = torch.tensor([0, 0, 0, 1, 1, 1])
    source_index = torch.tensor([0, 2, 3, 1, 2, 4])
    distances = torch.tensor([0.0, 0.3, 0.6, 0.0, 0.4, 0.55], dtype=torch.float64)
    polar_angles = torch.tensor([0.0, 0.7, -1.2, 0.0, 2.5, -0.3], dtype=torch.float64)
    transport_angles = torch.tensor([0.0, 0.4, -0.9, 0.0, 1.1, 0.2], dtype=torch.float64)
    quadrature_weights = torch.tensor([0.9, 1.1, 0.8, 0.95, 1.05, 1.2], dtype=torch.float64)

    return _synthetic_edges(
        target_index,
        source_index,
        n_nodes=5,
        distances=distances,
        polar_angles=polar_angles,
        transport_angles=transport_angles,
        tangent_quadrature_weights=quadrature_weights,
        max_frequency=max_frequency,
    )


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


# --------------------------------------------------------------------------
# The identity the key-caching optimization depends on
# --------------------------------------------------------------------------


def test_intertwiner_commutes_with_transport() -> None:
    """W_K(rho_in(g) v) == rho_attn(g) (W_K v) for an SO2Linear W_K.

    This is the algebraic fact SO2Attention's query/key ordering relies on:
    projecting to rho_attn commutes with transport, so projecting first
    (once per node) and transporting second (per edge, but on the smaller
    rho_attn space) is exactly equal to the naive order, not an
    approximation of it.
    """
    field_type_in = SO2FieldType({0: 3, 1: 2, 2: 1})
    field_type_attn = SO2FieldType({0: 2, 1: 1, 2: 1})

    torch.manual_seed(0)
    key = SO2Linear(field_type_in, field_type_attn, bias=False, dtype=torch.float64)

    generator = torch.Generator().manual_seed(1)
    f = _random_field(field_type_in, (), generator=generator)
    gamma = torch.tensor(0.9137, dtype=torch.float64)

    transport_then_project = key(f.apply_rotation(gamma)).to_flat()
    project_then_transport = key(f).apply_rotation(gamma).to_flat()

    assert torch.allclose(transport_then_project, project_then_transport, atol=1e-12)


# --------------------------------------------------------------------------
# SO2Attention: construction
# --------------------------------------------------------------------------


def test_attention_rejects_an_unknown_window_kind() -> None:
    field_type = SO2FieldType({0: 1})
    with pytest.raises(ValueError):
        SO2Attention(field_type, field_type, field_type, cutoff=1.0, window="triangular")  # type: ignore[arg-type]


def test_attention_score_scale_uses_the_attn_type_dimension() -> None:
    field_type_in = SO2FieldType({0: 1, 1: 1})
    field_type_attn = SO2FieldType({0: 1, 1: 2})  # dimension = 1 + 2*2 = 5

    attn = SO2Attention(
        field_type_in, field_type_attn, field_type_in, cutoff=1.0, dtype=torch.float64
    )
    assert attn._score_scale == pytest.approx(1.0 / math.sqrt(5))


# --------------------------------------------------------------------------
# SO2Attention: forward() validation
# --------------------------------------------------------------------------


def test_attention_forward_rejects_a_mismatched_field_type() -> None:
    field_type = SO2FieldType({0: 1})
    attn = SO2Attention(field_type, field_type, field_type, cutoff=1.0, dtype=torch.float64)
    edges = _multi_neighbor_edges(max_frequency=0)
    wrong_field = SO2FeatureField.from_flat(
        SO2FieldType({0: 2}), torch.zeros(5, 2, dtype=torch.float64)
    )

    with pytest.raises(ValueError):
        attn(wrong_field, edges)


def test_attention_forward_rejects_a_bad_batch_shape() -> None:
    field_type = SO2FieldType({0: 1})
    attn = SO2Attention(field_type, field_type, field_type, cutoff=1.0, dtype=torch.float64)
    edges = _multi_neighbor_edges(max_frequency=0)  # 5 nodes
    x = SO2FeatureField.from_flat(field_type, torch.zeros(3, 1, dtype=torch.float64))

    with pytest.raises(ValueError):
        attn(x, edges)


# --------------------------------------------------------------------------
# SO2Attention: attention weights are a proper partition of unity
# --------------------------------------------------------------------------


def test_attention_weights_sum_to_one_per_target_node() -> None:
    field_type_in = SO2FieldType({0: 2, 1: 1})
    field_type_attn = SO2FieldType({0: 1, 1: 1})
    field_type_out = SO2FieldType({0: 1, 2: 1})

    torch.manual_seed(2)
    attn = SO2Attention(
        field_type_in, field_type_attn, field_type_out, cutoff=1.0, dtype=torch.float64
    )
    edges = _multi_neighbor_edges(
        max_frequency=max(field_type_in.max_frequency, field_type_attn.max_frequency)
        + field_type_out.max_frequency
    )

    generator = torch.Generator().manual_seed(3)
    x = _random_field(field_type_in, (5,), generator=generator)

    # Recompute the normalized attention weight per edge exactly as forward()
    # does, then check it sums to 1 within each target's group.
    q = attn.query(x)
    k = attn.key(x)
    from gm_attention.nn._scatter import gather_to_edges, scatter_add_to_nodes
    from gm_attention.nn.kernels import transport_field

    q_edges = gather_to_edges(q, edges.target_index)
    k_edges = transport_field(gather_to_edges(k, edges.source_index), edges)
    score = (q_edges.to_flat() * k_edges.to_flat()).sum(-1) * attn._score_scale

    from gm_attention.nn._scatter import scatter_max_to_nodes
    from gm_attention.nn.convolution import evaluate_aggregation_window

    node_max = scatter_max_to_nodes(score, edges.row_ptr)
    activated = torch.exp(score - node_max[edges.target_index])
    unnormalized = (
        evaluate_aggregation_window(edges.distances, 1.0, "hard")
        * edges.tangent_quadrature_weights
        * activated
    )
    normalization = scatter_add_to_nodes(unnormalized, edges.row_ptr)
    weight = unnormalized / normalization[edges.target_index]

    per_target_sum = scatter_add_to_nodes(weight, edges.row_ptr)

    # Only nodes 0 and 1 have any incident edges in this fixture (2, 3, 4
    # appear solely as sources) -- a real EdgeGeometry guarantees every node
    # at least its own self-loop, but this hand-built one deliberately
    # doesn't, so nodes without incident edges correctly sum to 0, not 1.
    assert torch.allclose(
        per_target_sum[:2], torch.ones(2, dtype=torch.float64), atol=1e-10
    )


# --------------------------------------------------------------------------
# SO2Attention: master correctness test, an independent re-derivation
# --------------------------------------------------------------------------


def test_forward_matches_a_from_scratch_python_reimplementation() -> None:
    """Independently recompute the full Def 4.9/4.10 formula with a plain
    Python loop over edges (grouping by target manually), using the naive
    transport-then-project key order rather than the cached one -- this
    checks both the aggregation wiring AND that the key optimization
    produces the same answer as the textbook order, in one test."""
    field_type_in = SO2FieldType({0: 2, 1: 1})
    field_type_attn = SO2FieldType({0: 1, 1: 1})
    field_type_out = SO2FieldType({0: 1, 1: 2})

    torch.manual_seed(4)
    attn = SO2Attention(
        field_type_in,
        field_type_attn,
        field_type_out,
        cutoff=1.0,
        window="quartic",
        bias=True,
        dtype=torch.float64,
    )
    edges = _multi_neighbor_edges(
        max_frequency=max(field_type_in.max_frequency, field_type_attn.max_frequency)
        + field_type_out.max_frequency
    )

    generator = torch.Generator().manual_seed(5)
    x = _random_field(field_type_in, (5,), generator=generator)

    output = attn(x, edges)

    from gm_attention.nn.convolution import evaluate_aggregation_window

    # Reference: for each target node, gather its incident edges by hand,
    # compute the score using the NAIVE key order (transport f_j first, then
    # project with W_K -- the mathematically-defining order, not the
    # optimized one), and manually normalize. Only nodes 0 and 1 have any
    # incident edges in this fixture (2, 3, 4 appear solely as sources).
    # This reference is a plain numerical recomputation, not a backward
    # pass, so it runs detached from the autograd graph.
    with torch.no_grad():
        for target in (0, 1):
            incident = (edges.target_index == target).nonzero().flatten().tolist()
            q_i = attn.query(SO2FeatureField(
                field_type_in, {f: x[f][target : target + 1] for f in field_type_in.frequencies}
            )).to_flat()[0]

            scores = []
            for edge in incident:
                source = int(edges.source_index[edge])
                f_j = SO2FeatureField(
                    field_type_in, {f: x[f][source : source + 1] for f in field_type_in.frequencies}
                )
                transported_f_j = f_j.apply_rotation(edges.transport_angles[edge : edge + 1])
                k_j = attn.key(transported_f_j).to_flat()[0]
                scores.append(float((q_i * k_j).sum()) * attn._score_scale)

            scores_t = torch.tensor(scores, dtype=torch.float64)
            activated = torch.exp(scores_t - scores_t.max())

            weights = []
            values = []
            for local_idx, edge in enumerate(incident):
                source = int(edges.source_index[edge])
                window = evaluate_aggregation_window(
                    edges.distances[edge : edge + 1], 1.0, "quartic"
                )[0]
                w = window * edges.tangent_quadrature_weights[edge] * activated[local_idx]
                weights.append(w)

                source_field = SO2FeatureField(
                    field_type_in, {f: x[f][source : source + 1] for f in field_type_in.frequencies}
                )
                edge_geometry_slice = edges.select(torch.tensor([edge]))
                value_message = attn.value(source_field, edge_geometry_slice)
                values.append(value_message.to_flat()[0])

            weights_t = torch.stack(weights)
            weights_t = weights_t / weights_t.sum()
            expected = sum(wt * v for wt, v in zip(weights_t, values, strict=True))

            assert torch.allclose(output.to_flat()[target], expected, atol=1e-8)


# --------------------------------------------------------------------------
# SO2Attention: numerical stability
# --------------------------------------------------------------------------


def test_stabilization_does_not_change_the_result_for_small_scores() -> None:
    """With small scores (no overflow risk either way), the stabilized
    computation forward() performs must agree with a naive, unstabilized one
    -- proving the max-subtraction is a no-op on the *result*."""
    field_type = SO2FieldType({0: 1})

    torch.manual_seed(6)
    attn = SO2Attention(field_type, field_type, field_type, cutoff=1.0, dtype=torch.float64)
    edges = _multi_neighbor_edges(max_frequency=0)

    generator = torch.Generator().manual_seed(7)
    x = _random_field(field_type, (5,), generator=generator)

    stabilized_output = attn(x, edges).to_flat()

    from gm_attention.nn._scatter import gather_to_edges, scatter_add_to_nodes
    from gm_attention.nn.convolution import evaluate_aggregation_window
    from gm_attention.nn.kernels import transport_field

    q_edges = gather_to_edges(attn.query(x), edges.target_index)
    k_edges = transport_field(gather_to_edges(attn.key(x), edges.source_index), edges)
    score = (q_edges.to_flat() * k_edges.to_flat()).sum(-1) * attn._score_scale

    naive_activated = torch.exp(score)  # no stabilization
    unnormalized = (
        evaluate_aggregation_window(edges.distances, 1.0, "hard")
        * edges.tangent_quadrature_weights
        * naive_activated
    )
    normalization = scatter_add_to_nodes(unnormalized, edges.row_ptr)
    weight = unnormalized / normalization[edges.target_index]

    source = gather_to_edges(x, edges.source_index)
    messages = attn.value(source, edges)
    flat_messages = messages.to_flat()
    weighted = weight.unsqueeze(-1) * flat_messages
    naive_flat = torch.zeros(5, field_type.dimension, dtype=torch.float64)
    naive_flat.index_add_(0, edges.target_index, weighted)

    assert torch.allclose(stabilized_output, naive_flat, atol=1e-9)


def test_stabilization_prevents_overflow_for_large_scores() -> None:
    """Scale query/key weights up until a naive exp() would overflow, and
    confirm SO2Attention's actual output stays finite."""
    field_type = SO2FieldType({0: 4})

    torch.manual_seed(8)
    attn = SO2Attention(field_type, field_type, field_type, cutoff=1.0, dtype=torch.float32)

    with torch.no_grad():
        attn.query.weight["0"] *= 1e4
        attn.key.weight["0"] *= 1e4

    edges = _multi_neighbor_edges(max_frequency=0)
    edges = EdgeGeometry(
        edge_index=edges.edge_index,
        row_ptr=edges.row_ptr,
        log_coordinates=edges.log_coordinates.float(),
        distances=edges.distances.float(),
        polar_angles=edges.polar_angles.float(),
        transport_angles=edges.transport_angles.float(),
        tangent_quadrature_weights=edges.tangent_quadrature_weights.float(),
        angular_phases=edges.angular_phases.to(torch.complex64),
        transport_phases=edges.transport_phases.to(torch.complex64),
    )

    generator = torch.Generator().manual_seed(9)
    x = _random_field(field_type, (5,), generator=generator, dtype=torch.float32)

    output = attn(x, edges)

    for tensor in output.coefficients.values():
        assert torch.isfinite(tensor).all()

    # Sanity: an unstabilized exp() really would have overflowed here,
    # confirming this test exercises the regime it claims to.
    q = attn.query(x)
    k = attn.key(x)
    from gm_attention.nn._scatter import gather_to_edges
    from gm_attention.nn.kernels import transport_field

    q_edges = gather_to_edges(q, edges.target_index)
    k_edges = transport_field(gather_to_edges(k, edges.source_index), edges)
    score = (q_edges.to_flat() * k_edges.to_flat()).sum(-1) * attn._score_scale
    assert bool(torch.isinf(torch.exp(score)).any())


# --------------------------------------------------------------------------
# SO2Attention: self-loop-only sanity check
# --------------------------------------------------------------------------


def test_pure_self_loop_graph_reduces_to_the_value_kernel_at_the_origin() -> None:
    """With only self-loops, attention normalizes to weight 1 on each node's
    own edge, so the output is exactly the value kernel evaluated at r = 0 --
    which is now the kernel's own genuine origin limit (an intertwiner), not
    a separately-learned map substituted at that edge."""
    field_type = SO2FieldType({0: 1, 1: 1})

    torch.manual_seed(10)
    attn = SO2Attention(field_type, field_type, field_type, cutoff=1.0, bias=True, dtype=torch.float64)

    n_nodes = 3
    edges = _synthetic_edges(
        torch.arange(n_nodes),
        torch.arange(n_nodes),
        n_nodes=n_nodes,
        distances=torch.zeros(n_nodes, dtype=torch.float64),
        polar_angles=torch.zeros(n_nodes, dtype=torch.float64),
        transport_angles=torch.zeros(n_nodes, dtype=torch.float64),
        tangent_quadrature_weights=torch.ones(n_nodes, dtype=torch.float64),
        max_frequency=field_type.max_frequency * 2,
    )

    generator = torch.Generator().manual_seed(11)
    x = _random_field(field_type, (n_nodes,), generator=generator)

    output = attn(x, edges)
    expected = attn.value(x, edges)

    assert torch.allclose(output.to_flat(), expected.to_flat(), atol=1e-9)


def test_attention_self_interaction_defaults_to_absent() -> None:
    """The transformer block's own outer residual already supplies the
    pointwise identity path, so this must not be on by default."""
    field_type = SO2FieldType({0: 1, 1: 1})
    attn = SO2Attention(field_type, field_type, field_type, cutoff=1.0, dtype=torch.float64)

    assert attn.self_interaction == "none"
    assert attn.self_linear is None


def test_attention_self_interaction_adds_a_pointwise_term_outside_normalization() -> None:
    """The whole point of moving it out: its magnitude must not depend on
    the attention weights, which sum to 1 regardless of neighbor count."""
    field_type = SO2FieldType({0: 1, 1: 1})

    torch.manual_seed(30)
    attn = SO2Attention(
        field_type,
        field_type,
        field_type,
        cutoff=1.0,
        self_interaction="linear",
        dtype=torch.float64,
    )
    edges = _multi_neighbor_edges(max_frequency=field_type.max_frequency * 2)

    generator = torch.Generator().manual_seed(31)
    n_nodes = edges.row_ptr.shape[0] - 1
    x = _random_field(field_type, (n_nodes,), generator=generator)

    with_self = attn(x, edges)
    attn.self_linear = None  # the same layer, integral term only
    without_self = attn(x, edges)

    attn.self_linear = SO2Linear(field_type, field_type, bias=False, dtype=torch.float64)
    difference = SO2FeatureField(
        field_type,
        {
            frequency: with_self[frequency] - without_self[frequency]
            for frequency in field_type.frequencies
        },
    )

    assert not torch.allclose(
        difference.to_flat(), torch.zeros_like(difference.to_flat())
    )


def test_attention_self_interaction_is_gauge_equivariant() -> None:
    """A pointwise intertwiner commutes with a nodewise regauging, so
    enabling it must not disturb the layer's equivariance.

    Regauges the *geometry* as well as the features, on a real
    discretization -- the actual equivariance statement. Rotating features
    alone while leaving the polar/transport angles in the old gauge is not
    a regauging at all, and would fail for a perfectly correct layer.
    """
    grid = build_icosphere_grid(subdivisions=1, dtype=torch.float64)
    frames = build_tangent_frames(grid.nodes)
    coverage_radius = 0.6 * math.pi
    field_type = SO2FieldType({0: 1, 1: 1})

    torch.manual_seed(32)
    attn = SO2Attention(
        field_type,
        field_type,
        field_type,
        cutoff=coverage_radius,
        self_interaction="linear",
        n_radial_basis=6,
        radial_hidden_features=(8,),
        dtype=torch.float64,
    )

    max_frequency = 2 * field_type.max_frequency
    edges = compute_edge_geometry(grid, frames, coverage_radius, max_frequency=max_frequency)

    generator = torch.Generator().manual_seed(33)
    x = _random_field(field_type, (grid.n_nodes,), generator=generator)
    gauge_angles = torch.rand(
        grid.n_nodes, dtype=torch.float64, generator=generator
    ) * (2 * math.pi)

    regauged_edges = compute_edge_geometry(
        grid,
        rotate_tangent_frames(frames, gauge_angles),
        coverage_radius,
        max_frequency=max_frequency,
    )

    regauged_output = attn(x.gauge_transform(gauge_angles), regauged_edges).to_flat()
    expected = attn(x, edges).gauge_transform(gauge_angles).to_flat()

    assert torch.allclose(regauged_output, expected, atol=1e-7)


def test_attention_rejects_an_unknown_self_interaction() -> None:
    field_type = SO2FieldType({0: 1})
    with pytest.raises(ValueError, match="self_interaction"):
        SO2Attention(
            field_type,
            field_type,
            field_type,
            cutoff=1.0,
            self_interaction="mlp",  # type: ignore[arg-type]
        )


def test_quartic_window_attention_is_finite_when_the_top_score_is_at_the_boundary() -> None:
    """The soft-window stability case: with a quartic window the largest raw
    score can sit on an edge whose window value is almost zero. Stabilizing
    on score alone pushes every other exponent far negative for no reason;
    stabilizing on the complete log-weight does not."""
    field_type = SO2FieldType({0: 1, 1: 1})

    torch.manual_seed(34)
    attn = SO2Attention(
        field_type, field_type, field_type, cutoff=1.0, window="quartic", dtype=torch.float64
    )

    # One self-loop plus a neighbor sitting essentially on the cutoff, where
    # the quartic window is ~0.
    n_nodes = 1
    edges = _synthetic_edges(
        torch.tensor([0, 0]),
        torch.tensor([0, 0]),
        n_nodes=n_nodes,
        distances=torch.tensor([0.0, 1.0 - 1e-12], dtype=torch.float64),
        polar_angles=torch.tensor([0.0, 0.4], dtype=torch.float64),
        transport_angles=torch.tensor([0.0, 0.0], dtype=torch.float64),
        tangent_quadrature_weights=torch.ones(2, dtype=torch.float64),
        max_frequency=field_type.max_frequency * 2,
    )

    generator = torch.Generator().manual_seed(35)
    x = _random_field(field_type, (n_nodes,), generator=generator)

    output = attn(x, edges).to_flat()

    assert torch.isfinite(output).all()

    output.square().sum().backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in attn.parameters()
    )


def test_edges_outside_the_window_contribute_exactly_zero() -> None:
    """Compact support must be exact, not merely small: an edge beyond the
    cutoff gets weight 0, not exp(large negative)."""
    field_type = SO2FieldType({0: 1})

    torch.manual_seed(36)
    attn = SO2Attention(
        field_type, field_type, field_type, cutoff=1.0, window="quartic", dtype=torch.float64
    )

    common = {
        "n_nodes": 1,
        "polar_angles": torch.tensor([0.0, 0.3], dtype=torch.float64),
        "transport_angles": torch.zeros(2, dtype=torch.float64),
        "tangent_quadrature_weights": torch.ones(2, dtype=torch.float64),
        "max_frequency": 2,
    }
    generator = torch.Generator().manual_seed(37)
    x = _random_field(field_type, (1,), generator=generator)

    # Second edge exactly at the cutoff -> window is exactly 0.
    at_cutoff = attn(
        x,
        _synthetic_edges(
            torch.tensor([0, 0]),
            torch.tensor([0, 0]),
            distances=torch.tensor([0.0, 1.0], dtype=torch.float64),
            **common,
        ),
    )
    # ... which must equal a graph that simply does not contain that edge.
    self_only = attn(
        x,
        _synthetic_edges(
            torch.tensor([0]),
            torch.tensor([0]),
            n_nodes=1,
            distances=torch.tensor([0.0], dtype=torch.float64),
            polar_angles=torch.tensor([0.0], dtype=torch.float64),
            transport_angles=torch.zeros(1, dtype=torch.float64),
            tangent_quadrature_weights=torch.ones(1, dtype=torch.float64),
            max_frequency=2,
        ),
    )

    assert torch.allclose(at_cutoff.to_flat(), self_only.to_flat(), atol=1e-12)


# --------------------------------------------------------------------------
# SO2Attention: autograd
# --------------------------------------------------------------------------


def test_attention_gradients_flow_to_every_parameter_and_input() -> None:
    field_type = SO2FieldType({0: 1, 1: 1})
    attn = SO2Attention(field_type, field_type, field_type, cutoff=1.0, dtype=torch.float64)
    edges = _multi_neighbor_edges(max_frequency=field_type.max_frequency * 2)

    flat = torch.randn(5, field_type.dimension, dtype=torch.float64, requires_grad=True)
    x = SO2FeatureField.from_flat(field_type, flat)

    output = attn(x, edges)
    loss = sum(t.pow(2).sum() for t in output.coefficients.values())
    loss.backward()

    assert flat.grad is not None
    assert torch.isfinite(flat.grad).all()

    for name, parameter in attn.named_parameters():
        assert parameter.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(parameter.grad).all()


# --------------------------------------------------------------------------
# SO2Attention: end-to-end gauge equivariance on a real discretization
# --------------------------------------------------------------------------


@pytest.mark.parametrize("window", ["hard", "quartic"])
def test_attention_end_to_end_gauge_equivariance(window: str) -> None:
    grid = build_icosphere_grid(subdivisions=1, dtype=torch.float64)
    frames = build_tangent_frames(grid.nodes)
    coverage_radius = 0.6 * math.pi

    field_type_in = SO2FieldType({0: 1, 1: 1})
    field_type_attn = SO2FieldType({0: 1, 1: 1})
    field_type_out = SO2FieldType({0: 1, 1: 1, 2: 1})

    torch.manual_seed(12)
    attn = SO2Attention(
        field_type_in,
        field_type_attn,
        field_type_out,
        cutoff=coverage_radius,
        window=window,  # type: ignore[arg-type]
        n_radial_basis=6,
        radial_hidden_features=(8,),
        dtype=torch.float64,
    )

    max_frequency = max(
        field_type_in.max_frequency, field_type_attn.max_frequency
    ) + field_type_out.max_frequency
    edges = compute_edge_geometry(grid, frames, coverage_radius, max_frequency=max_frequency)

    generator = torch.Generator().manual_seed(13)
    node_features = _random_field(field_type_in, (grid.n_nodes,), generator=generator)

    original_output = attn(node_features, edges).to_flat()

    gauge_angles = torch.rand(grid.n_nodes, dtype=torch.float64, generator=generator) * (
        2 * math.pi
    )
    regauged_edges = compute_edge_geometry(
        grid, rotate_tangent_frames(frames, gauge_angles), coverage_radius, max_frequency=max_frequency
    )
    regauged_node_features = node_features.gauge_transform(gauge_angles)

    regauged_output = attn(regauged_node_features, regauged_edges).to_flat()
    expected = SO2FeatureField.from_flat(field_type_out, original_output).gauge_transform(
        gauge_angles
    ).to_flat()

    assert torch.allclose(regauged_output, expected, atol=1e-7)


# --------------------------------------------------------------------------
# SO2MHSA: construction
# --------------------------------------------------------------------------


def test_mhsa_accepts_a_heterogeneous_field_type() -> None:
    """Multiplicities need not agree between frequencies -- only each one
    must be divisible by num_heads."""
    module = SO2MHSA(SO2FieldType({0: 2, 1: 3}), num_heads=1, cutoff=1.0)

    assert module.field_type_attn == SO2FieldType({0: 2, 1: 3})


def test_mhsa_splits_each_frequency_by_its_own_multiplicity() -> None:
    """Spec 7.5 case 3: {0: 8, 1: 4, 2: 12} with 4 heads -> {0: 2, 1: 1, 2: 3}."""
    module = SO2MHSA(SO2FieldType({0: 8, 1: 4, 2: 12}), num_heads=4, cutoff=1.0)

    assert module.field_type_attn == SO2FieldType({0: 2, 1: 1, 2: 3})
    assert len(module.heads) == 4


def test_mhsa_rejects_a_head_count_not_dividing_the_multiplicity() -> None:
    field_type = SO2FieldType({0: 4, 1: 4})
    with pytest.raises(ValueError):
        SO2MHSA(field_type, num_heads=3, cutoff=1.0)


def test_mhsa_head_check_is_per_frequency_and_names_the_offender() -> None:
    """A total-dimension check would pass {0: 8, 1: 4} with 8 heads
    (dimension 16), but that cannot split frequency 1's four copies."""
    with pytest.raises(ValueError, match=r"\{1: 4\}"):
        SO2MHSA(SO2FieldType({0: 8, 1: 4}), num_heads=8, cutoff=1.0)


def test_mhsa_rejects_fewer_than_one_head() -> None:
    field_type = SO2FieldType({0: 2})
    with pytest.raises(ValueError):
        SO2MHSA(field_type, num_heads=0, cutoff=1.0)


def test_mhsa_builds_the_expected_per_head_attn_type() -> None:
    field_type = SO2FieldType({0: 4, 1: 4, 2: 4})
    mhsa = SO2MHSA(field_type, num_heads=2, cutoff=1.0, dtype=torch.float64)

    assert mhsa.field_type_attn == SO2FieldType({0: 2, 1: 2, 2: 2})
    assert len(mhsa.heads) == 2


def test_mhsa_heads_are_independently_initialized() -> None:
    field_type = SO2FieldType({0: 4, 1: 4})
    torch.manual_seed(14)
    mhsa = SO2MHSA(field_type, num_heads=2, cutoff=1.0, dtype=torch.float64)

    assert not torch.equal(
        mhsa.heads[0].query.weight["0"], mhsa.heads[1].query.weight["0"]
    )


# --------------------------------------------------------------------------
# SO2MHSA: forward correctness
# --------------------------------------------------------------------------


def test_mhsa_output_field_type_equals_the_input_field_type() -> None:
    field_type = SO2FieldType({0: 2, 1: 2})
    torch.manual_seed(15)
    mhsa = SO2MHSA(field_type, num_heads=2, cutoff=1.0, dtype=torch.float64)
    edges = _multi_neighbor_edges(max_frequency=field_type.max_frequency * 2)

    generator = torch.Generator().manual_seed(16)
    x = _random_field(field_type, (5,), generator=generator)

    output = mhsa(x, edges)
    assert output.field_type == field_type


def test_mhsa_matches_manually_running_heads_and_fusing() -> None:
    field_type = SO2FieldType({0: 4, 1: 4})
    torch.manual_seed(17)
    mhsa = SO2MHSA(field_type, num_heads=2, cutoff=1.0, bias=True, dtype=torch.float64)
    edges = _multi_neighbor_edges(max_frequency=field_type.max_frequency * 2)

    generator = torch.Generator().manual_seed(18)
    x = _random_field(field_type, (5,), generator=generator)

    output = mhsa(x, edges)

    head_outputs = [head(x, edges) for head in mhsa.heads]
    fused_flat = torch.cat(
        [
            torch.cat([h[f] for h in head_outputs], dim=-1)
            if f == 0
            else torch.cat([h[f] for h in head_outputs], dim=-2).flatten(-2)
            for f in field_type.frequencies
        ],
        dim=-1,
    )
    fused = SO2FeatureField.from_flat(field_type, fused_flat)
    expected = mhsa.output_projection(fused)

    assert torch.allclose(output.to_flat(), expected.to_flat(), atol=1e-10)


def test_mhsa_with_one_head_still_mixes_via_the_output_projection() -> None:
    """num_heads=1 should NOT be a no-op: the output_projection still mixes
    within each frequency's multiplicity space."""
    field_type = SO2FieldType({0: 2})
    torch.manual_seed(19)
    mhsa = SO2MHSA(field_type, num_heads=1, cutoff=1.0, bias=False, dtype=torch.float64)
    edges = _multi_neighbor_edges(max_frequency=0)

    generator = torch.Generator().manual_seed(20)
    x = _random_field(field_type, (5,), generator=generator)

    output = mhsa(x, edges)
    single_head_output = mhsa.heads[0](x, edges)

    assert not torch.allclose(output.to_flat(), single_head_output.to_flat(), atol=1e-6)


# --------------------------------------------------------------------------
# SO2MHSA: autograd
# --------------------------------------------------------------------------


def test_mhsa_gradients_flow_to_every_parameter_and_input() -> None:
    field_type = SO2FieldType({0: 2, 1: 2})
    mhsa = SO2MHSA(field_type, num_heads=2, cutoff=1.0, dtype=torch.float64)
    edges = _multi_neighbor_edges(max_frequency=field_type.max_frequency * 2)

    flat = torch.randn(5, field_type.dimension, dtype=torch.float64, requires_grad=True)
    x = SO2FeatureField.from_flat(field_type, flat)

    output = mhsa(x, edges)
    loss = sum(t.pow(2).sum() for t in output.coefficients.values())
    loss.backward()

    assert flat.grad is not None
    assert torch.isfinite(flat.grad).all()

    for name, parameter in mhsa.named_parameters():
        assert parameter.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(parameter.grad).all()


# --------------------------------------------------------------------------
# SO2MHSA: discretization dispatch
# --------------------------------------------------------------------------


def test_mhsa_accepts_a_discretization_directly() -> None:
    field_type = SO2FieldType({0: 2, 1: 2})
    coverage_radius = 0.6 * math.pi
    layer_cutoff = 0.3 * math.pi

    torch.manual_seed(21)
    mhsa = SO2MHSA(field_type, num_heads=2, cutoff=layer_cutoff, dtype=torch.float64)

    grid = build_icosphere_grid(subdivisions=1, dtype=torch.float64)
    discretization = SphericalDiscretization.from_grid(grid, coverage_radius=coverage_radius)

    generator = torch.Generator().manual_seed(22)
    x = _random_field(field_type, (discretization.n_nodes,), generator=generator)

    via_discretization = mhsa(x, discretization)
    via_explicit_edges = mhsa(x, discretization.edges_within(layer_cutoff))

    assert torch.allclose(via_discretization.to_flat(), via_explicit_edges.to_flat(), atol=1e-10)


# --------------------------------------------------------------------------
# SO2MHSA: end-to-end gauge equivariance on a real discretization
# --------------------------------------------------------------------------


def test_mhsa_end_to_end_gauge_equivariance() -> None:
    grid = build_icosphere_grid(subdivisions=1, dtype=torch.float64)
    frames = build_tangent_frames(grid.nodes)
    coverage_radius = 0.6 * math.pi

    field_type = SO2FieldType({0: 2, 1: 2})

    torch.manual_seed(23)
    mhsa = SO2MHSA(
        field_type,
        num_heads=2,
        cutoff=coverage_radius,
        n_radial_basis=6,
        radial_hidden_features=(8,),
        dtype=torch.float64,
    )

    max_frequency = field_type.max_frequency * 2
    edges = compute_edge_geometry(grid, frames, coverage_radius, max_frequency=max_frequency)

    generator = torch.Generator().manual_seed(24)
    node_features = _random_field(field_type, (grid.n_nodes,), generator=generator)

    original_output = mhsa(node_features, edges).to_flat()

    gauge_angles = torch.rand(grid.n_nodes, dtype=torch.float64, generator=generator) * (
        2 * math.pi
    )
    regauged_edges = compute_edge_geometry(
        grid, rotate_tangent_frames(frames, gauge_angles), coverage_radius, max_frequency=max_frequency
    )
    regauged_node_features = node_features.gauge_transform(gauge_angles)

    regauged_output = mhsa(regauged_node_features, regauged_edges).to_flat()
    expected = SO2FeatureField.from_flat(field_type, original_output).gauge_transform(
        gauge_angles
    ).to_flat()

    assert torch.allclose(regauged_output, expected, atol=1e-7)
