"""Tests for gm_attention.nn.convolution.

SO2Convolution's own job -- gather, window, quadrature-weight, scatter-add --
is deliberately tested in isolation from the per-edge value transform, whose
correctness is already established in tests/nn/test_kernels.py. The primary
tool for that isolation is a hand-built multi-neighbor EdgeGeometry, cross
checked against a plain Python re-implementation of the aggregation sum.
"""

from __future__ import annotations

import dataclasses
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
from gm_attention.nn._scatter import gather_to_edges, scatter_add_to_nodes
from gm_attention.nn.convolution import (
    SO2Convolution,
    evaluate_aggregation_window,
)
from gm_attention.representations import so2
from gm_attention.representations.field_type import (
    SO2FeatureField,
    SO2FieldType,
)

ATOL = 1e-8


def _synthetic_edges(
    target_index: torch.Tensor,  # (E,) long
    source_index: torch.Tensor,  # (E,) long
    n_nodes: int,
    distances: torch.Tensor,  # (E,)
    polar_angles: torch.Tensor,  # (E,)
    transport_angles: torch.Tensor,  # (E,)
    tangent_quadrature_weights: torch.Tensor,  # (E,)
    *,
    max_frequency: int,
) -> EdgeGeometry:
    """Build an EdgeGeometry by hand with arbitrary connectivity.

    Edges must already be target-major sorted (EdgeGeometry's own contract);
    callers below construct them in that order directly rather than sorting
    here, so an ordering error surfaces in the test itself.
    """
    order = torch.argsort(target_index, stable=True)
    if not torch.equal(order, torch.arange(target_index.shape[0])):
        raise ValueError("target_index must already be sorted (test construction bug).")

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


def _multi_neighbor_edges(max_frequency: int) -> EdgeGeometry:
    """Two target nodes (0, 1), each with a self-loop plus two distinct
    neighbors, all distinct edges -- enough structure to catch a gather or
    scatter index bug that a single-neighbor graph could not."""
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


# --------------------------------------------------------------------------
# evaluate_aggregation_window
# --------------------------------------------------------------------------


def test_hard_window_is_the_indicator_function() -> None:
    distances = torch.tensor([0.0, 0.49, 0.5, 0.51, 1.0], dtype=torch.float64)
    weight = evaluate_aggregation_window(distances, radius=0.5, kind="hard")

    assert torch.equal(weight, torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0], dtype=torch.float64))


def test_quartic_window_is_one_at_the_origin() -> None:
    weight = evaluate_aggregation_window(
        torch.zeros(3, dtype=torch.float64), radius=1.0, kind="quartic"
    )
    assert torch.allclose(weight, torch.ones(3, dtype=torch.float64), atol=ATOL)


def test_quartic_window_vanishes_at_and_beyond_the_radius() -> None:
    distances = torch.tensor([1.0, 1.5, 10.0], dtype=torch.float64)
    weight = evaluate_aggregation_window(distances, radius=1.0, kind="quartic")
    assert torch.allclose(weight, torch.zeros(3, dtype=torch.float64), atol=ATOL)


def test_quartic_window_matches_its_closed_form() -> None:
    r = torch.tensor([0.1, 0.3, 0.7, 0.95], dtype=torch.float64)
    radius = 1.3
    expected = (1.0 - (r / radius) ** 2) ** 2
    assert torch.allclose(
        evaluate_aggregation_window(r, radius, "quartic"), expected, atol=ATOL
    )


def test_quartic_window_is_monotonically_decreasing_on_the_support() -> None:
    r = torch.linspace(0.0, 1.0, 50, dtype=torch.float64)
    weight = evaluate_aggregation_window(r, radius=1.0, kind="quartic")
    assert bool((weight[1:] <= weight[:-1] + 1e-12).all())


def test_quartic_window_derivative_vanishes_at_the_boundary() -> None:
    """c'(radius) = 0, unlike the hard window's jump discontinuity there."""
    radius = 1.0
    r = torch.tensor(radius, dtype=torch.float64, requires_grad=True)
    weight = evaluate_aggregation_window(r.reshape(1), radius, "quartic")
    (grad,) = torch.autograd.grad(weight.sum(), r)
    assert abs(float(grad)) < 1e-9


def test_window_rejects_an_unknown_kind() -> None:
    with pytest.raises(ValueError):
        evaluate_aggregation_window(torch.zeros(2, dtype=torch.float64), 1.0, "gaussian")  # type: ignore[arg-type]


def test_window_rejects_a_nonpositive_radius() -> None:
    with pytest.raises(ValueError):
        evaluate_aggregation_window(torch.zeros(2, dtype=torch.float64), 0.0, "hard")


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


def test_rejects_an_unknown_window_kind() -> None:
    with pytest.raises(ValueError):
        SO2Convolution(
            SO2FieldType({0: 1}),
            SO2FieldType({0: 1}),
            cutoff=1.0,
            window="triangular",  # type: ignore[arg-type]
        )


def test_delegates_field_types_to_the_internal_kernel() -> None:
    field_type_in = SO2FieldType({0: 1, 1: 2})
    field_type_out = SO2FieldType({0: 2, 2: 1})

    conv = SO2Convolution(field_type_in, field_type_out, cutoff=1.0, dtype=torch.float64)

    assert conv.kernel.field_type_in == field_type_in
    assert conv.kernel.field_type_out == field_type_out
    assert conv.kernel.radial_basis.cutoff == pytest.approx(1.0)


# --------------------------------------------------------------------------
# forward() validation
# --------------------------------------------------------------------------


def test_forward_rejects_a_mismatched_field_type() -> None:
    conv = SO2Convolution(
        SO2FieldType({0: 1}), SO2FieldType({0: 1}), cutoff=1.0, dtype=torch.float64
    )
    edges = _multi_neighbor_edges(max_frequency=0)
    wrong_field = SO2FeatureField.from_flat(
        SO2FieldType({0: 2}), torch.zeros(5, 2, dtype=torch.float64)
    )

    with pytest.raises(ValueError):
        conv(wrong_field, edges)


def test_forward_rejects_a_batch_shape_disagreeing_with_row_ptr() -> None:
    field_type = SO2FieldType({0: 1})
    conv = SO2Convolution(field_type, field_type, cutoff=1.0, dtype=torch.float64)
    edges = _multi_neighbor_edges(max_frequency=0)  # 5 nodes
    x = SO2FeatureField.from_flat(field_type, torch.zeros(3, 1, dtype=torch.float64))

    with pytest.raises(ValueError):
        conv(x, edges)


# --------------------------------------------------------------------------
# Master correctness test: the aggregation sum itself
# --------------------------------------------------------------------------


def test_forward_matches_a_manual_gather_weight_scatter() -> None:
    field_type_in = SO2FieldType({0: 2, 1: 1})
    field_type_out = SO2FieldType({0: 1, 1: 2})

    torch.manual_seed(0)
    conv = SO2Convolution(
        field_type_in,
        field_type_out,
        cutoff=1.0,
        window="hard",
        bias=False,
        dtype=torch.float64,
    )

    edges = _multi_neighbor_edges(max_frequency=field_type_in.max_frequency + field_type_out.max_frequency)

    generator = torch.Generator().manual_seed(1)
    x = _random_field(field_type_in, (5,), generator=generator)

    output = conv(x, edges)

    # Manual re-derivation: gather, run the SAME kernel instance, weight by
    # window * quadrature, and sum per target with a plain Python loop --
    # deliberately not using index_add_, so this exercises different code
    # from convolution.py's own implementation.
    source = SO2FeatureField(
        field_type_in,
        {f: t[edges.source_index] for f, t in x.coefficients.items()},
    )
    messages = conv.kernel(source, edges)
    weight = evaluate_aggregation_window(edges.distances, 1.0, "hard") * edges.tangent_quadrature_weights

    for frequency in field_type_out.frequencies:
        expected = torch.zeros_like(output[frequency])
        for edge in range(edges.n_edges):
            target = int(edges.target_index[edge])
            expected[target] += weight[edge] * messages[frequency][edge]
        assert torch.allclose(output[frequency], expected, atol=1e-9)


def test_forward_separates_aggregation_per_target_node() -> None:
    """Perturbing a neighbor of node 1 must not change node 0's output."""
    field_type = SO2FieldType({0: 1, 1: 1})

    torch.manual_seed(2)
    conv = SO2Convolution(field_type, field_type, cutoff=1.0, bias=False, dtype=torch.float64)
    edges = _multi_neighbor_edges(max_frequency=field_type.max_frequency * 2)

    generator = torch.Generator().manual_seed(3)
    x = _random_field(field_type, (5,), generator=generator)

    before = conv(x, edges)

    x_perturbed = SO2FeatureField(
        field_type,
        {
            0: x[0].clone(),
            1: x[1].clone(),
        },
    )
    x_perturbed[0][4] += 5.0  # node 4 only feeds node 1's aggregation

    after = conv(x_perturbed, edges)

    assert torch.allclose(before[0][0], after[0][0], atol=ATOL)
    assert torch.allclose(before[1][0], after[1][0], atol=ATOL)
    assert not torch.allclose(before[0][1], after[0][1], atol=1e-4)


# --------------------------------------------------------------------------
# The window and quadrature weight are actually used
# --------------------------------------------------------------------------


def test_hard_and_quartic_windows_agree_at_the_origin() -> None:
    """Self-loops (r=0) must be unaffected by the choice of window."""
    field_type = SO2FieldType({0: 1, 1: 1})

    torch.manual_seed(4)
    hard = SO2Convolution(field_type, field_type, cutoff=1.0, window="hard", bias=True, dtype=torch.float64)
    quartic = SO2Convolution(field_type, field_type, cutoff=1.0, window="quartic", bias=True, dtype=torch.float64)
    quartic.load_state_dict(hard.state_dict())

    # A graph of pure self-loops: r = 0 everywhere.
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

    generator = torch.Generator().manual_seed(5)
    x = _random_field(field_type, (n_nodes,), generator=generator)

    assert torch.allclose(hard(x, edges).to_flat(), quartic(x, edges).to_flat(), atol=1e-9)


def test_quartic_window_downweights_a_near_boundary_neighbor_relative_to_hard() -> None:
    field_type = SO2FieldType({0: 1})

    torch.manual_seed(6)
    hard = SO2Convolution(field_type, field_type, cutoff=1.0, window="hard", bias=False, dtype=torch.float64)
    quartic = SO2Convolution(field_type, field_type, cutoff=1.0, window="quartic", bias=False, dtype=torch.float64)
    quartic.load_state_dict(hard.state_dict())

    # Node 0 <- node 1, a single neighbor at 95% of the cutoff radius.
    edges = _synthetic_edges(
        torch.tensor([0, 1]),
        torch.tensor([1, 1]),
        n_nodes=2,
        distances=torch.tensor([0.95, 0.0], dtype=torch.float64),
        polar_angles=torch.tensor([0.3, 0.0], dtype=torch.float64),
        transport_angles=torch.tensor([0.1, 0.0], dtype=torch.float64),
        tangent_quadrature_weights=torch.tensor([1.0, 1.0], dtype=torch.float64),
        max_frequency=0,
    )

    generator = torch.Generator().manual_seed(7)
    x = _random_field(field_type, (2,), generator=generator)

    hard_out = hard(x, edges)[0][0]
    quartic_out = quartic(x, edges)[0][0]

    # hard weight = 1, quartic weight = (1 - 0.95^2)^2 ~= 0.0095 at r=0.95:
    # the quartic-windowed contribution from node 1 should be far smaller.
    assert (quartic_out - hard_out).abs().item() > 1e-3


def test_quadrature_weight_scales_the_contribution_linearly() -> None:
    field_type = SO2FieldType({0: 1})

    torch.manual_seed(8)
    conv = SO2Convolution(field_type, field_type, cutoff=1.0, bias=False, dtype=torch.float64)

    def edges_with_weight(weight: float) -> EdgeGeometry:
        return _synthetic_edges(
            torch.tensor([0]),
            torch.tensor([1]),
            n_nodes=2,
            distances=torch.tensor([0.3], dtype=torch.float64),
            polar_angles=torch.tensor([0.2], dtype=torch.float64),
            transport_angles=torch.tensor([0.0], dtype=torch.float64),
            tangent_quadrature_weights=torch.tensor([weight], dtype=torch.float64),
            max_frequency=0,
        )

    generator = torch.Generator().manual_seed(9)
    x = _random_field(field_type, (2,), generator=generator)

    baseline = conv(x, edges_with_weight(1.0))[0][0]
    doubled = conv(x, edges_with_weight(2.0))[0][0]

    assert torch.allclose(doubled, 2.0 * baseline, atol=1e-9)


# --------------------------------------------------------------------------
# Autograd
# --------------------------------------------------------------------------


def test_gradients_flow_to_kernel_parameters_and_input() -> None:
    field_type = SO2FieldType({0: 1, 1: 1})
    conv = SO2Convolution(field_type, field_type, cutoff=1.0, window="quartic", dtype=torch.float64)
    edges = _multi_neighbor_edges(max_frequency=field_type.max_frequency * 2)

    flat = torch.randn(5, field_type.dimension, dtype=torch.float64, requires_grad=True)
    x = SO2FeatureField.from_flat(field_type, flat)

    output = conv(x, edges)
    loss = sum(t.pow(2).sum() for t in output.coefficients.values())
    loss.backward()

    assert flat.grad is not None
    assert torch.isfinite(flat.grad).all()

    for name, parameter in conv.named_parameters():
        assert parameter.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(parameter.grad).all()


# --------------------------------------------------------------------------
# Dispatch on SphericalDiscretization: uses (and reuses) its radius cache
# --------------------------------------------------------------------------


def _small_discretization(coverage_radius: float) -> SphericalDiscretization:
    grid = build_icosphere_grid(subdivisions=1, dtype=torch.float64)
    return SphericalDiscretization.from_grid(grid, coverage_radius=coverage_radius)


def test_passing_a_discretization_matches_passing_its_edges_within_directly() -> None:
    field_type = SO2FieldType({0: 1, 1: 1})
    coverage_radius = 0.6 * math.pi
    layer_cutoff = 0.3 * math.pi

    torch.manual_seed(20)
    conv = SO2Convolution(field_type, field_type, cutoff=layer_cutoff, dtype=torch.float64)

    discretization = _small_discretization(coverage_radius)
    generator = torch.Generator().manual_seed(21)
    x = _random_field(field_type, (discretization.n_nodes,), generator=generator)

    via_discretization = conv(x, discretization)
    via_explicit_edges = conv(x, discretization.edges_within(layer_cutoff))

    assert torch.allclose(
        via_discretization.to_flat(), via_explicit_edges.to_flat(), atol=1e-10
    )


def test_passing_a_discretization_populates_and_reuses_its_radius_cache() -> None:
    field_type = SO2FieldType({0: 1})
    coverage_radius = 0.6 * math.pi
    layer_cutoff = 0.25 * math.pi

    conv = SO2Convolution(field_type, field_type, cutoff=layer_cutoff, dtype=torch.float64)
    discretization = _small_discretization(coverage_radius)

    # Only the coverage-radius entry is pre-populated at construction time.
    assert len(discretization.cached_radii) == 1

    generator = torch.Generator().manual_seed(22)
    x = _random_field(field_type, (discretization.n_nodes,), generator=generator)

    conv(x, discretization)
    assert len(discretization.cached_radii) == 2

    # A second call at the same cutoff must reuse the cached EdgeGeometry
    # object rather than rebuilding it -- verified by identity, not just by
    # equal values -- and must not grow the cache further.
    cached_edges = discretization.edges_within(layer_cutoff)
    conv(x, discretization)
    assert discretization.edges_within(layer_cutoff) is cached_edges
    assert len(discretization.cached_radii) == 2


def test_a_cutoff_beyond_coverage_radius_raises_the_discretizations_own_error() -> None:
    field_type = SO2FieldType({0: 1})
    discretization = _small_discretization(coverage_radius=0.2 * math.pi)
    conv = SO2Convolution(field_type, field_type, cutoff=0.5 * math.pi, dtype=torch.float64)

    x = SO2FeatureField.from_flat(
        field_type, torch.zeros(discretization.n_nodes, 1, dtype=torch.float64)
    )

    with pytest.raises(ValueError, match="coverage radius"):
        conv(x, discretization)


# --------------------------------------------------------------------------
# End-to-end gauge equivariance on a real discretization
# --------------------------------------------------------------------------


@pytest.mark.parametrize("window", ["hard", "quartic"])
def test_end_to_end_gauge_equivariance_on_a_real_discretization(window: str) -> None:
    grid = build_icosphere_grid(subdivisions=1, dtype=torch.float64)
    frames = build_tangent_frames(grid.nodes)
    coverage_radius = 0.6 * math.pi

    field_type_in = SO2FieldType({0: 1, 1: 1})
    field_type_out = SO2FieldType({0: 1, 1: 1, 2: 1})

    torch.manual_seed(10)
    conv = SO2Convolution(
        field_type_in,
        field_type_out,
        cutoff=coverage_radius,
        window=window,  # type: ignore[arg-type]
        n_radial_basis=6,
        radial_hidden_features=(8,),
        bias=True,
        dtype=torch.float64,
    )

    max_frequency = field_type_in.max_frequency + field_type_out.max_frequency
    edges = compute_edge_geometry(
        grid, frames, coverage_radius, max_frequency=max_frequency
    )

    generator = torch.Generator().manual_seed(11)
    node_features = _random_field(field_type_in, (grid.n_nodes,), generator=generator)

    original_output = conv(node_features, edges).to_flat()

    gauge_angles = torch.rand(grid.n_nodes, dtype=torch.float64, generator=generator) * (
        2 * math.pi
    )

    regauged_edges = compute_edge_geometry(
        grid,
        rotate_tangent_frames(frames, gauge_angles),
        coverage_radius,
        max_frequency=max_frequency,
    )
    regauged_node_features = node_features.gauge_transform(gauge_angles)

    regauged_output = conv(regauged_node_features, regauged_edges).to_flat()

    expected = SO2FeatureField.from_flat(field_type_out, original_output).gauge_transform(
        gauge_angles
    ).to_flat()

    assert torch.allclose(regauged_output, expected, atol=1e-7)


# ==========================================================================
# self_interaction: a pointwise map, outside the quadrature integral
# ==========================================================================


def _conv_pair(field_type, **overrides):
    """Two identical convolutions differing only in self_interaction."""
    torch.manual_seed(50)
    integral_only = SO2Convolution(
        field_type, field_type, cutoff=0.6, dtype=torch.float64, **overrides
    )
    torch.manual_seed(50)
    with_self = SO2Convolution(
        field_type,
        field_type,
        cutoff=0.6,
        self_interaction="linear",
        dtype=torch.float64,
        **overrides,
    )
    return integral_only, with_self


def test_self_interaction_defaults_to_none() -> None:
    """The default must leave this a pure discretized integral operator."""
    conv = SO2Convolution(
        SO2FieldType({0: 1}), SO2FieldType({0: 1}), cutoff=0.5, dtype=torch.float64
    )

    assert conv.self_interaction == "none"
    assert conv.self_linear is None


def test_self_interaction_none_matches_the_integral_only_behavior() -> None:
    field_type = SO2FieldType({0: 1, 1: 1})
    grid = build_icosphere_grid(subdivisions=1, dtype=torch.float64)
    frames = build_tangent_frames(grid.nodes)
    edges = compute_edge_geometry(grid, frames, 0.6, max_frequency=2)

    torch.manual_seed(51)
    conv = SO2Convolution(field_type, field_type, cutoff=0.6, dtype=torch.float64)

    generator = torch.Generator().manual_seed(52)
    x = _random_field(field_type, (grid.n_nodes,), generator=generator)

    # Reconstruct the integral by hand from the same pieces the layer uses.
    source = gather_to_edges(x, edges.source_index)
    messages = conv.kernel(source, edges)
    weight = evaluate_aggregation_window(edges.distances, 0.6, "hard") * (
        edges.tangent_quadrature_weights
    )
    expected = {
        frequency: scatter_add_to_nodes(
            weight.reshape(edges.n_edges, *([1] * (messages[frequency].ndim - 1)))
            * messages[frequency],
            edges.row_ptr,
        )
        for frequency in field_type.frequencies
    }

    output = conv(x, edges)
    for frequency in field_type.frequencies:
        assert torch.allclose(output[frequency], expected[frequency], atol=1e-12)


def test_self_interaction_adds_exactly_the_pointwise_linear_output() -> None:
    field_type = SO2FieldType({0: 1, 1: 1})
    grid = build_icosphere_grid(subdivisions=1, dtype=torch.float64)
    frames = build_tangent_frames(grid.nodes)
    edges = compute_edge_geometry(grid, frames, 0.6, max_frequency=2)

    integral_only, with_self = _conv_pair(field_type)
    generator = torch.Generator().manual_seed(53)
    x = _random_field(field_type, (grid.n_nodes,), generator=generator)

    difference = with_self(x, edges).to_flat() - integral_only(x, edges).to_flat()
    assert with_self.self_linear is not None
    assert torch.allclose(difference, with_self.self_linear(x).to_flat(), atol=1e-12)


def test_only_the_integral_term_responds_to_the_quadrature_weights() -> None:
    """The distinction the whole change exists for. Scaling every quadrature
    weight scales the integral term with it; the pointwise term must not
    move at all."""
    field_type = SO2FieldType({0: 1, 1: 1})
    grid = build_icosphere_grid(subdivisions=1, dtype=torch.float64)
    frames = build_tangent_frames(grid.nodes)
    edges = compute_edge_geometry(grid, frames, 0.6, max_frequency=2)

    scaled_edges = dataclasses.replace(
        edges, tangent_quadrature_weights=edges.tangent_quadrature_weights * 3.0
    )

    integral_only, with_self = _conv_pair(field_type)
    generator = torch.Generator().manual_seed(54)
    x = _random_field(field_type, (grid.n_nodes,), generator=generator)

    # The integral term genuinely tripled...
    assert torch.allclose(
        integral_only(x, scaled_edges).to_flat(),
        3.0 * integral_only(x, edges).to_flat(),
        atol=1e-12,
    )

    # ... while the added pointwise term is identical under both.
    pointwise_at_original = with_self(x, edges).to_flat() - integral_only(x, edges).to_flat()
    pointwise_at_scaled = (
        with_self(x, scaled_edges).to_flat() - integral_only(x, scaled_edges).to_flat()
    )
    assert torch.allclose(pointwise_at_original, pointwise_at_scaled, atol=1e-12)


def test_the_pointwise_term_does_not_shrink_under_mesh_refinement() -> None:
    """A Riemann-sum term's per-neighbor contribution shrinks as the mesh
    refines; a pointwise map's does not. Refining the grid must leave the
    pointwise term's magnitude essentially unchanged."""
    field_type = SO2FieldType({0: 1})
    _, with_self = _conv_pair(field_type)
    assert with_self.self_linear is not None

    magnitudes = []
    for subdivisions in (1, 2):
        grid = build_icosphere_grid(subdivisions=subdivisions, dtype=torch.float64)
        generator = torch.Generator().manual_seed(55)
        x = _random_field(field_type, (grid.n_nodes,), generator=generator)
        magnitudes.append(float(with_self.self_linear(x).to_flat().abs().mean()))

    assert magnitudes[1] == pytest.approx(magnitudes[0], rel=0.3)


@pytest.mark.parametrize("window", ["hard", "quartic"])
def test_self_interaction_preserves_gauge_equivariance(window: str) -> None:
    field_type = SO2FieldType({0: 1, 1: 1})
    grid = build_icosphere_grid(subdivisions=1, dtype=torch.float64)
    frames = build_tangent_frames(grid.nodes)
    coverage_radius = 0.6

    torch.manual_seed(56)
    conv = SO2Convolution(
        field_type,
        field_type,
        cutoff=coverage_radius,
        window=window,  # type: ignore[arg-type]
        self_interaction="linear",
        dtype=torch.float64,
    )

    edges = compute_edge_geometry(grid, frames, coverage_radius, max_frequency=2)
    generator = torch.Generator().manual_seed(57)
    x = _random_field(field_type, (grid.n_nodes,), generator=generator)
    gauge_angles = torch.rand(
        grid.n_nodes, dtype=torch.float64, generator=generator
    ) * (2 * math.pi)

    regauged_edges = compute_edge_geometry(
        grid, rotate_tangent_frames(frames, gauge_angles), coverage_radius, max_frequency=2
    )

    regauged_output = conv(x.gauge_transform(gauge_angles), regauged_edges).to_flat()
    expected = conv(x, edges).gauge_transform(gauge_angles).to_flat()

    assert torch.allclose(regauged_output, expected, atol=1e-8)


def test_rejects_an_unknown_self_interaction() -> None:
    with pytest.raises(ValueError, match="self_interaction"):
        SO2Convolution(
            SO2FieldType({0: 1}),
            SO2FieldType({0: 1}),
            cutoff=0.5,
            self_interaction="mlp",  # type: ignore[arg-type]
        )


# --------------------------------------------------------------------------
# normalize=True: the integral as a weighted mean
# --------------------------------------------------------------------------


def test_normalized_forward_divides_the_integral_by_each_targets_window_mass() -> None:
    """With ``normalize=True`` every target's aggregated message is the plain
    sum divided by ``sum_j c(v_ij) w_ij`` over its incident edges -- one
    scalar per target, the denominator IcoLowPassRestriction uses -- and
    the pointwise self-interaction is added undivided."""
    field_type_in = SO2FieldType({0: 2, 1: 1})
    field_type_out = SO2FieldType({0: 1, 1: 2})
    torch.manual_seed(0)
    plain = SO2Convolution(field_type_in, field_type_out, cutoff=1.0, window="quartic",
                           self_interaction="linear", bias=False, dtype=torch.float64)
    normalized = SO2Convolution(field_type_in, field_type_out, cutoff=1.0, window="quartic",
                                self_interaction="linear", normalize=True, bias=False, dtype=torch.float64)
    normalized.load_state_dict(plain.state_dict())
    assert normalized.normalize and not plain.normalize and "normalize=True" in normalized.extra_repr()

    edges = _multi_neighbor_edges(max_frequency=field_type_in.max_frequency + field_type_out.max_frequency)
    generator = torch.Generator().manual_seed(1)
    x = _random_field(field_type_in, (5,), generator=generator)

    weight = evaluate_aggregation_window(edges.distances, 1.0, "quartic") * edges.tangent_quadrature_weights
    mass = torch.zeros(5, dtype=torch.float64).index_add_(0, edges.target_index, weight)
    assert (mass[:2] > 0).all() and (mass[2:] == 0).all()  # only nodes 0 and 1 are targets

    pointwise = plain.self_linear(x)
    integral_plain = plain(x, edges)
    integral_normalized = normalized(x, edges)
    for frequency in field_type_out.frequencies:
        summed = integral_plain[frequency] - pointwise[frequency]
        expected = summed / mass.clamp_min(normalized.eps).reshape(-1, *([1] * (summed.ndim - 1))) + pointwise[frequency]
        assert torch.allclose(integral_normalized[frequency], expected, atol=1e-12)
        # a target with no edges keeps exactly its pointwise term, no division by zero
        assert torch.allclose(integral_normalized[frequency][2:], pointwise[frequency][2:], atol=1e-12)


def test_normalized_forward_is_the_weighted_mean_for_a_constant_scalar_kernel() -> None:
    """Scalars, a depthwise kernel forced to 1: the normalized output of a
    node is exactly the window-weighted mean of its neighbours' values."""
    field_type = SO2FieldType({0: 3})
    conv = SO2Convolution(field_type, field_type, cutoff=1.0, window="quartic", depthwise=True,
                          normalize=True, bias=False, dtype=torch.float64)
    edges = _multi_neighbor_edges(max_frequency=0)
    generator = torch.Generator().manual_seed(2)
    x = _random_field(field_type, (5,), generator=generator)

    class Ones(torch.nn.Module):
        def forward(self, source, edges):
            return {0: source[0]}

    conv.kernel = Ones()
    output = conv(x, edges)[0]
    weight = evaluate_aggregation_window(edges.distances, 1.0, "quartic") * edges.tangent_quadrature_weights
    for target in (0, 1):
        incident = edges.target_index == target
        expected = (weight[incident, None] * x[0][edges.source_index[incident]]).sum(0) / weight[incident].sum()
        assert torch.allclose(output[target], expected, atol=1e-12)


def test_rejects_a_nonpositive_eps() -> None:
    with pytest.raises(ValueError, match="eps"):
        SO2Convolution(SO2FieldType({0: 1}), SO2FieldType({0: 1}), cutoff=1.0, eps=0.0)
