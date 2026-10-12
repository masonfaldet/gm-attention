"""Tests for gm_attention.discretization.edge_geometry.

Covers the log map and parallel-transport geometry computed by
``compute_edge_geometry``, self-loop behavior, and the ``EdgeGeometry.select``
method (including the CSR ``row_ptr`` it must keep consistent).
"""

from __future__ import annotations

import math

import pytest
import torch

from gm_attention.discretization.discretization import SphericalDiscretization
from gm_attention.discretization.edge_geometry import (
    DEFAULT_MAX_FREQUENCY,
    EdgeGeometry,
    compute_edge_geometry,
    edge_geometry_from_index,
)
from gm_attention.discretization.frames import (
    build_tangent_frames,
    rotate_tangent_frames,
)
from gm_attention.discretization.grid import SphericalGrid
from gm_attention.discretization.grids.icosphere import build_icosphere_grid

ATOL = 1e-10

# Self-loops carry a geodesic distance of order 1e-17 rather than exactly
# zero, so "is this the origin" needs a tolerance.
ORIGIN_TOLERANCE = 1e-12


@pytest.fixture
def grid() -> SphericalGrid:
    return build_icosphere_grid(subdivisions=1)


@pytest.fixture
def frames(grid: SphericalGrid) -> torch.Tensor:
    return build_tangent_frames(grid.nodes)


@pytest.fixture
def edge_geometry(grid: SphericalGrid, frames: torch.Tensor) -> EdgeGeometry:
    return compute_edge_geometry(grid=grid, frames=frames, coverage_radius=1.0)


# --------------------------------------------------------------------------
# Log map / parallel transport geometry
# --------------------------------------------------------------------------


def test_log_coordinate_norm_equals_geodesic_distance(
    edge_geometry: EdgeGeometry,
) -> None:
    log_norm = torch.linalg.vector_norm(edge_geometry.log_coordinates, dim=-1)
    assert torch.allclose(log_norm, edge_geometry.distances, atol=ATOL)


def test_log_coordinates_reconstruct_source_node_via_exp_map(
    grid: SphericalGrid,
    frames: torch.Tensor,
    edge_geometry: EdgeGeometry,
) -> None:
    """log_coordinates are only meaningful relative to the specific frames
    they were computed against. Decoding them through frames[target] and
    applying the sphere's exponential map at the target node must
    reconstruct the source node exactly -- the direct check that frames and
    edge geometry agree on a common gauge.
    """
    target_frames = frames[edge_geometry.target_index]  # (n_edges, 3, 2)
    target_nodes = grid.nodes[edge_geometry.target_index]  # (n_edges, 3)
    source_nodes = grid.nodes[edge_geometry.source_index]  # (n_edges, 3)

    # E_i has orthonormal columns spanning T_{p_i}S^2, and the log vector
    # lies exactly in that plane, so E_i @ log_coordinates recovers the
    # ambient log vector without loss.
    ambient_log_vectors = torch.bmm(
        target_frames,
        edge_geometry.log_coordinates.unsqueeze(-1),
    ).squeeze(-1)  # (n_edges, 3)

    r = edge_geometry.distances  # (n_edges,)
    # exp_p(v) = cos(|v|) p + sinc(|v|) v, using sinc(x) = sin(pi x)/(pi x)
    # so sinc(r / pi) = sin(r) / r, well-defined (= 1) at r = 0.
    reconstructed_source_nodes = (
        torch.cos(r).unsqueeze(-1) * target_nodes
        + torch.sinc(r / math.pi).unsqueeze(-1) * ambient_log_vectors
    )

    assert torch.allclose(reconstructed_source_nodes, source_nodes, atol=ATOL)


def test_distances_are_within_coverage_radius(
    edge_geometry: EdgeGeometry,
) -> None:
    assert bool((edge_geometry.distances <= 1.0 + 1e-9).all())
    assert bool((edge_geometry.distances >= 0.0).all())


def test_tangent_quadrature_weights_are_positive(
    edge_geometry: EdgeGeometry,
) -> None:
    assert bool((edge_geometry.tangent_quadrature_weights > 0).all())


def test_edges_are_sorted_target_major_source_minor(
    edge_geometry: EdgeGeometry,
) -> None:
    target = edge_geometry.target_index
    source = edge_geometry.source_index
    same_target_ordered = source[1:] >= source[:-1]
    target_ordered = target[1:] >= target[:-1]
    new_target_started = target[1:] > target[:-1]
    assert bool((target_ordered).all())
    assert bool((same_target_ordered | new_target_started).all())


# --------------------------------------------------------------------------
# Self-loops (i == j)
# --------------------------------------------------------------------------


def test_self_loops_are_present_for_every_node(
    grid: SphericalGrid,
    edge_geometry: EdgeGeometry,
) -> None:
    self_loop_mask = edge_geometry.target_index == edge_geometry.source_index
    assert int(self_loop_mask.sum()) == grid.n_nodes


def test_self_loops_have_trivial_geometry(
    grid: SphericalGrid,
    edge_geometry: EdgeGeometry,
) -> None:
    self_loop_mask = edge_geometry.target_index == edge_geometry.source_index

    assert torch.allclose(
        edge_geometry.distances[self_loop_mask],
        torch.zeros_like(edge_geometry.distances[self_loop_mask]),
        atol=ATOL,
    )
    assert torch.allclose(
        edge_geometry.log_coordinates[self_loop_mask],
        torch.zeros_like(edge_geometry.log_coordinates[self_loop_mask]),
        atol=ATOL,
    )
    # Zero rotation: cos(0), sin(0) -> atan2(0, 1) == 0.
    assert torch.allclose(
        edge_geometry.transport_angles[self_loop_mask],
        torch.zeros_like(edge_geometry.transport_angles[self_loop_mask]),
        atol=ATOL,
    )
    # r / sin(r) -> 1 as r -> 0, so the self-loop tangent weight equals the
    # node's own spherical quadrature weight.
    self_loop_targets = edge_geometry.target_index[self_loop_mask]
    assert torch.allclose(
        edge_geometry.tangent_quadrature_weights[self_loop_mask],
        grid.quadrature_weights[self_loop_targets],
        atol=ATOL,
    )


# --------------------------------------------------------------------------
# coverage_radius validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad_radius", [0.0, -0.1, math.pi, math.pi + 0.1])
def test_compute_edge_geometry_rejects_invalid_coverage_radius(
    grid: SphericalGrid,
    bad_radius: float,
) -> None:
    frames = build_tangent_frames(grid.nodes)
    with pytest.raises(ValueError):
        compute_edge_geometry(
            grid=grid,
            frames=frames,
            coverage_radius=bad_radius,
        )


# --------------------------------------------------------------------------
# EdgeGeometry.select
# --------------------------------------------------------------------------


def test_select_with_boolean_mask_matches_manual_filtering(
    edge_geometry: EdgeGeometry,
) -> None:
    mask = edge_geometry.distances <= 0.5
    subset = edge_geometry.select(mask)

    assert subset.n_edges == int(mask.sum())
    assert torch.equal(subset.edge_index, edge_geometry.edge_index[:, mask])
    assert torch.equal(
        subset.log_coordinates,
        edge_geometry.log_coordinates[mask],
    )
    assert torch.equal(subset.distances, edge_geometry.distances[mask])
    assert torch.equal(
        subset.transport_angles,
        edge_geometry.transport_angles[mask],
    )
    assert torch.equal(
        subset.tangent_quadrature_weights,
        edge_geometry.tangent_quadrature_weights[mask],
    )


def test_select_with_boolean_mask_produces_consistent_row_ptr(
    grid: SphericalGrid,
    edge_geometry: EdgeGeometry,
) -> None:
    mask = edge_geometry.distances <= 0.5
    subset = edge_geometry.select(mask)

    assert subset.row_ptr.shape == (grid.n_nodes + 1,)
    assert subset.row_ptr[0].item() == 0
    assert subset.row_ptr[-1].item() == subset.n_edges

    manual = torch.bincount(
        subset.target_index,
        minlength=grid.n_nodes,
    ).cumsum(0)
    assert torch.equal(subset.row_ptr[1:], manual)


def test_select_with_sorted_long_indices(
    edge_geometry: EdgeGeometry,
) -> None:
    indices = torch.arange(0, edge_geometry.n_edges, 2, dtype=torch.long)
    subset = edge_geometry.select(indices)

    assert subset.n_edges == indices.shape[0]
    assert torch.equal(subset.edge_index, edge_geometry.edge_index[:, indices])
    assert subset.row_ptr[-1].item() == subset.n_edges


def test_select_all_false_mask_yields_empty_geometry(
    grid: SphericalGrid,
    edge_geometry: EdgeGeometry,
) -> None:
    mask = torch.zeros(edge_geometry.n_edges, dtype=torch.bool)
    subset = edge_geometry.select(mask)

    assert subset.n_edges == 0
    assert subset.row_ptr.shape == (grid.n_nodes + 1,)
    assert bool((subset.row_ptr == 0).all())


def test_select_rejects_wrong_length_boolean_mask(
    edge_geometry: EdgeGeometry,
) -> None:
    with pytest.raises(ValueError):
        edge_geometry.select(torch.zeros(edge_geometry.n_edges + 1, dtype=torch.bool))


def test_select_rejects_non_bool_non_long_dtype(
    edge_geometry: EdgeGeometry,
) -> None:
    with pytest.raises(TypeError):
        edge_geometry.select(
            torch.zeros(edge_geometry.n_edges, dtype=torch.float32)
        )


def test_select_rejects_multidimensional_selection(
    edge_geometry: EdgeGeometry,
) -> None:
    with pytest.raises(ValueError):
        edge_geometry.select(
            torch.zeros((edge_geometry.n_edges, 1), dtype=torch.bool)
        )


def test_select_rejects_unsorted_long_indices(
    edge_geometry: EdgeGeometry,
) -> None:
    # Reversing the index order breaks the target-major sort invariant that
    # row_ptr depends on, so this must fail loudly rather than silently
    # producing an inconsistent CSR structure.
    indices = torch.arange(
        edge_geometry.n_edges - 1, -1, -1, dtype=torch.long
    )
    with pytest.raises(ValueError):
        edge_geometry.select(indices)


def test_select_with_explicit_n_nodes_resizes_row_ptr(
    grid: SphericalGrid,
    edge_geometry: EdgeGeometry,
) -> None:
    """The bipartite-downsampling use case: narrow to edges targeting only a
    node prefix, and size row_ptr for that smaller target set, not the
    original grid's full node count."""
    n_coarse = grid.n_nodes // 2
    mask = edge_geometry.target_index < n_coarse
    subset = edge_geometry.select(mask, n_nodes=n_coarse)

    assert subset.row_ptr.shape == (n_coarse + 1,)
    assert subset.row_ptr[0].item() == 0
    assert subset.row_ptr[-1].item() == subset.n_edges
    assert bool((subset.target_index < n_coarse).all())

    manual = torch.bincount(subset.target_index, minlength=n_coarse).cumsum(0)
    assert torch.equal(subset.row_ptr[1:], manual)


def test_select_without_n_nodes_defaults_to_the_original_node_count(
    grid: SphericalGrid,
    edge_geometry: EdgeGeometry,
) -> None:
    mask = edge_geometry.distances <= 0.5
    assert edge_geometry.select(mask).row_ptr.shape == (grid.n_nodes + 1,)


def test_select_matches_discretization_edges_within(
    grid: SphericalGrid,
    edge_geometry: EdgeGeometry,
) -> None:
    # SphericalDiscretization.edges_within is select() driven by a radius
    # mask; cross-check the two independent code paths agree.
    discretization = SphericalDiscretization(
        grid=grid,
        frames=build_tangent_frames(grid.nodes),
        edge_geometry=edge_geometry,
        coverage_radius=1.0,
    )
    via_discretization = discretization.edges_within(0.6)
    via_select = edge_geometry.select(edge_geometry.distances <= 0.6)

    assert torch.equal(via_discretization.edge_index, via_select.edge_index)
    assert torch.equal(via_discretization.row_ptr, via_select.row_ptr)


# --------------------------------------------------------------------------
# Polar angles and tabulated angular harmonics
# --------------------------------------------------------------------------


def test_polar_angles_are_the_argument_of_the_log_coordinates(
    edge_geometry: EdgeGeometry,
) -> None:
    expected = torch.atan2(
        edge_geometry.log_coordinates[:, 1],
        edge_geometry.log_coordinates[:, 0],
    )
    assert torch.allclose(edge_geometry.polar_angles, expected, atol=ATOL)


def test_log_coordinates_recover_from_distance_and_polar_angle(
    edge_geometry: EdgeGeometry,
) -> None:
    """v = r (cos phi, sin phi) is the polar decomposition the kernel uses."""
    polar = torch.stack(
        (
            torch.cos(edge_geometry.polar_angles),
            torch.sin(edge_geometry.polar_angles),
        ),
        dim=-1,
    )
    reconstructed = edge_geometry.distances.unsqueeze(-1) * polar

    assert torch.allclose(
        reconstructed,
        edge_geometry.log_coordinates,
        atol=ATOL,
    )


def test_self_loops_have_zero_polar_angle_and_unit_harmonics(
    edge_geometry: EdgeGeometry,
) -> None:
    self_loops = edge_geometry.target_index == edge_geometry.source_index

    assert bool(self_loops.any())
    assert torch.allclose(
        edge_geometry.polar_angles[self_loops],
        torch.zeros_like(edge_geometry.polar_angles[self_loops]),
        atol=ATOL,
    )

    for order in range(edge_geometry.max_frequency + 1):
        harmonics = edge_geometry.angular_harmonic(order)[self_loops]
        assert torch.allclose(
            harmonics,
            torch.ones_like(harmonics),
            atol=ATOL,
        )


def test_default_max_frequency_is_four(edge_geometry: EdgeGeometry) -> None:
    assert edge_geometry.max_frequency == DEFAULT_MAX_FREQUENCY == 4


def test_phase_tables_have_the_documented_shape_and_dtype(
    edge_geometry: EdgeGeometry,
) -> None:
    expected_shape = (edge_geometry.n_edges, edge_geometry.max_frequency + 1)

    assert edge_geometry.angular_phases.shape == expected_shape
    assert edge_geometry.transport_phases.shape == expected_shape
    assert edge_geometry.angular_phases.dtype == torch.complex128
    assert edge_geometry.transport_phases.dtype == torch.complex128


@pytest.mark.parametrize("max_frequency", [0, 1, 3, 7])
def test_max_frequency_is_configurable(
    grid: SphericalGrid,
    frames: torch.Tensor,
    max_frequency: int,
) -> None:
    edges = compute_edge_geometry(
        grid=grid,
        frames=frames,
        coverage_radius=1.0,
        max_frequency=max_frequency,
    )

    assert edges.max_frequency == max_frequency
    assert edges.angular_phases.shape[1] == max_frequency + 1


def test_compute_edge_geometry_rejects_a_negative_max_frequency(
    grid: SphericalGrid,
    frames: torch.Tensor,
) -> None:
    with pytest.raises(ValueError):
        compute_edge_geometry(
            grid=grid,
            frames=frames,
            coverage_radius=1.0,
            max_frequency=-1,
        )


@pytest.mark.parametrize("order", [0, 1, 2, 3, 4])
def test_angular_harmonic_equals_the_expected_phase(
    edge_geometry: EdgeGeometry,
    order: int,
) -> None:
    expected = torch.polar(
        torch.ones_like(edge_geometry.polar_angles),
        order * edge_geometry.polar_angles,
    )
    assert torch.allclose(
        edge_geometry.angular_harmonic(order),
        expected,
        atol=ATOL,
    )


@pytest.mark.parametrize("order", [0, 1, 2, 3, 4])
def test_transport_harmonic_equals_the_expected_phase(
    edge_geometry: EdgeGeometry,
    order: int,
) -> None:
    expected = torch.polar(
        torch.ones_like(edge_geometry.transport_angles),
        order * edge_geometry.transport_angles,
    )
    assert torch.allclose(
        edge_geometry.transport_harmonic(order),
        expected,
        atol=ATOL,
    )


@pytest.mark.parametrize("order", [1, 2, 4])
def test_negative_orders_are_the_conjugate(
    edge_geometry: EdgeGeometry,
    order: int,
) -> None:
    assert torch.allclose(
        edge_geometry.angular_harmonic(-order),
        edge_geometry.angular_harmonic(order).conj(),
        atol=ATOL,
    )
    assert torch.allclose(
        edge_geometry.transport_harmonic(-order),
        edge_geometry.transport_harmonic(order).conj(),
        atol=ATOL,
    )


def test_harmonics_have_unit_modulus(edge_geometry: EdgeGeometry) -> None:
    for order in range(-edge_geometry.max_frequency, edge_geometry.max_frequency + 1):
        harmonics = edge_geometry.angular_harmonic(order)
        assert torch.allclose(
            harmonics.abs(),
            torch.ones_like(harmonics.abs()),
            atol=ATOL,
        )


@pytest.mark.parametrize("order", [5, -5, 17])
def test_untabulated_order_raises_with_an_actionable_message(
    edge_geometry: EdgeGeometry,
    order: int,
) -> None:
    with pytest.raises(ValueError, match="max_frequency"):
        edge_geometry.angular_harmonic(order)

    with pytest.raises(ValueError, match="max_frequency"):
        edge_geometry.transport_harmonic(order)


def test_harmonic_rejects_a_non_integer_order(
    edge_geometry: EdgeGeometry,
) -> None:
    with pytest.raises(TypeError):
        edge_geometry.angular_harmonic(1.0)  # type: ignore[arg-type]


def test_angular_harmonics_reproduce_the_steerable_kernel_bases(
    edge_geometry: EdgeGeometry,
) -> None:
    """The tabulated phases must BE the angular kernel bases of Table 8.3.

    For m, n >= 1 the basis is K_1 = r_{(m-n)phi}, K_2 = r_{(m-n)phi}J,
    K_3 = r_{(m+n)phi}S, K_4 = r_{(m+n)phi}JS. Under R^2 = C these are
    multiplication by w, i*w, and w' composed with conjugation, where
    w = exp(i(m-n)phi) and w' = exp(i(m+n)phi) are table lookups.
    """
    phi = edge_geometry.polar_angles  # (n_edges,)

    def multiplication_matrix(w: torch.Tensor) -> torch.Tensor:
        """Real 2x2 matrix of z -> w z, batched over edges."""
        return torch.stack(
            (
                torch.stack((w.real, -w.imag), dim=-1),
                torch.stack((w.imag, w.real), dim=-1),
            ),
            dim=-2,
        )  # (n_edges, 2, 2)

    conjugation = torch.tensor(
        [[1.0, 0.0], [0.0, -1.0]],
        dtype=phi.dtype,
    )
    imaginary_unit = torch.tensor(1j, dtype=torch.complex128)

    for m, n in [(1, 1), (2, 1), (1, 2), (2, 2)]:
        minus, plus = (m - n) * phi, (m + n) * phi
        cos_minus, sin_minus = torch.cos(minus), torch.sin(minus)
        cos_plus, sin_plus = torch.cos(plus), torch.sin(plus)

        def row(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.stack((a, b), dim=-1)

        # Table 8.3, transcribed literally.
        k1 = torch.stack((row(cos_minus, -sin_minus), row(sin_minus, cos_minus)), dim=-2)
        k2 = torch.stack((row(-sin_minus, -cos_minus), row(cos_minus, -sin_minus)), dim=-2)
        k3 = torch.stack((row(cos_plus, sin_plus), row(sin_plus, -cos_plus)), dim=-2)
        k4 = torch.stack((row(-sin_plus, cos_plus), row(cos_plus, sin_plus)), dim=-2)

        linear = edge_geometry.angular_harmonic(m - n)  # (n_edges,)
        antilinear = edge_geometry.angular_harmonic(m + n)  # (n_edges,)

        assert torch.allclose(multiplication_matrix(linear), k1, atol=ATOL)
        assert torch.allclose(
            multiplication_matrix(imaginary_unit * linear), k2, atol=ATOL
        )
        assert torch.allclose(
            multiplication_matrix(antilinear) @ conjugation, k3, atol=ATOL
        )
        assert torch.allclose(
            multiplication_matrix(imaginary_unit * antilinear) @ conjugation,
            k4,
            atol=ATOL,
        )


def test_angular_harmonics_are_gauge_covariant(
    grid: SphericalGrid,
    frames: torch.Tensor,
) -> None:
    """Regauging by theta_i must rotate the order-k harmonic by exp(-i k theta_i).

    This is the property that lets a steerable kernel absorb the gauge: the
    harmonic supplies exactly the phase that cancels the input and output
    transformation laws.

    The angular law is asserted only away from the origin. At r = 0 the polar
    angle is undefined and the stabilizer is all of SO(2), so the angular
    harmonics are not covariant there; see
    ``test_origin_angular_harmonics_are_gauge_invariant_not_covariant``.
    """
    coverage_radius = 1.0
    original = compute_edge_geometry(
        grid=grid, frames=frames, coverage_radius=coverage_radius
    )

    generator = torch.Generator().manual_seed(5)
    gauge_angles = torch.rand(
        grid.n_nodes, dtype=grid.dtype, generator=generator
    ) * (2 * math.pi)

    regauged = compute_edge_geometry(
        grid=grid,
        frames=rotate_tangent_frames(frames, gauge_angles),
        coverage_radius=coverage_radius,
    )

    target_angles = gauge_angles[original.target_index]  # (n_edges,)
    source_angles = gauge_angles[original.source_index]  # (n_edges,)

    # A self-loop's geodesic distance is not exactly zero -- atan2 of the
    # near-zero cross-product norm lands around 1e-17 -- so the origin must be
    # detected with a tolerance rather than by comparing against zero.
    away_from_origin = original.distances > ORIGIN_TOLERANCE  # (n_edges,) bool

    assert bool(away_from_origin.any())
    assert not bool(
        away_from_origin[original.target_index == original.source_index].any()
    )

    for order in range(1, original.max_frequency + 1):
        # phi' = phi - theta_i, for r > 0
        assert torch.allclose(
            regauged.angular_harmonic(order)[away_from_origin],
            (
                original.angular_harmonic(order)
                * torch.polar(
                    torch.ones_like(target_angles), -order * target_angles
                )
            )[away_from_origin],
            atol=1e-9,
        )

        # gamma' = gamma + theta_j - theta_i
        assert torch.allclose(
            regauged.transport_harmonic(order),
            original.transport_harmonic(order)
            * torch.polar(
                torch.ones_like(target_angles),
                order * (source_angles - target_angles),
            ),
            atol=1e-9,
        )


def test_select_subsets_the_phase_tables_consistently(
    edge_geometry: EdgeGeometry,
) -> None:
    mask = edge_geometry.distances <= 0.6
    subset = edge_geometry.select(mask)

    assert subset.max_frequency == edge_geometry.max_frequency
    assert torch.allclose(
        subset.polar_angles, edge_geometry.polar_angles[mask], atol=ATOL
    )

    for order in (-2, 0, 3):
        assert torch.allclose(
            subset.angular_harmonic(order),
            edge_geometry.angular_harmonic(order)[mask],
            atol=ATOL,
        )
        assert torch.allclose(
            subset.transport_harmonic(order),
            edge_geometry.transport_harmonic(order)[mask],
            atol=ATOL,
        )


def test_to_float32_narrows_the_phase_tables_to_complex64(
    edge_geometry: EdgeGeometry,
) -> None:
    narrowed = edge_geometry.to(dtype=torch.float32)

    assert narrowed.angular_phases.dtype == torch.complex64
    assert narrowed.transport_phases.dtype == torch.complex64
    assert narrowed.polar_angles.dtype == torch.float32
    assert torch.allclose(
        narrowed.angular_harmonic(2),
        edge_geometry.angular_harmonic(2).to(torch.complex64),
        atol=1e-6,
    )


def test_origin_angular_harmonics_are_gauge_invariant_not_covariant(
    grid: SphericalGrid,
    frames: torch.Tensor,
) -> None:
    """At r = 0 the polar angle is undefined, so no gauge law applies.

    The stabilizer of the origin is all of SO(2), so a steerable kernel must
    reduce to an intertwiner there rather than reading the angular harmonics.
    ``compute_edge_geometry`` assigns phi = 0 to self-loops in every gauge;
    this test pins that convention so a later change cannot silently make the
    self-interaction term gauge dependent.
    """
    coverage_radius = 1.0
    original = compute_edge_geometry(
        grid=grid, frames=frames, coverage_radius=coverage_radius
    )

    generator = torch.Generator().manual_seed(7)
    gauge_angles = torch.rand(
        grid.n_nodes, dtype=grid.dtype, generator=generator
    ) * (2 * math.pi)

    regauged = compute_edge_geometry(
        grid=grid,
        frames=rotate_tangent_frames(frames, gauge_angles),
        coverage_radius=coverage_radius,
    )

    self_loops = original.target_index == original.source_index

    for order in range(original.max_frequency + 1):
        assert torch.allclose(
            regauged.angular_harmonic(order)[self_loops],
            original.angular_harmonic(order)[self_loops],
            atol=ATOL,
        )


# --------------------------------------------------------------------------
# edge_geometry_from_index (the radius-agnostic derivation compute_edge_
# geometry is built from, and IcoDecode's upsample step reuses directly with
# a hand-specified, non-radius edge list)
# --------------------------------------------------------------------------


def test_from_index_matches_compute_edge_geometry_given_its_own_edge_index(
    grid: SphericalGrid,
    frames: torch.Tensor,
    edge_geometry: EdgeGeometry,
) -> None:
    """compute_edge_geometry is (radius search + tolerance filter) then this
    function -- so calling this function directly on that same, already-
    filtered edge_index must reproduce it exactly, not approximately: this is
    the refactor's own correctness check, independent of any of the other
    tests here continuing to pass."""
    rebuilt = edge_geometry_from_index(
        nodes=grid.nodes,
        frames=frames,
        spherical_quadrature_weights=grid.quadrature_weights,
        edge_index=edge_geometry.edge_index,
        n_target_nodes=grid.n_nodes,
        max_frequency=DEFAULT_MAX_FREQUENCY,
    )

    assert torch.equal(rebuilt.edge_index, edge_geometry.edge_index)
    assert torch.equal(rebuilt.row_ptr, edge_geometry.row_ptr)
    assert torch.allclose(rebuilt.distances, edge_geometry.distances, atol=ATOL)
    assert torch.allclose(rebuilt.log_coordinates, edge_geometry.log_coordinates, atol=ATOL)
    assert torch.allclose(rebuilt.transport_angles, edge_geometry.transport_angles, atol=ATOL)
    assert torch.allclose(
        rebuilt.tangent_quadrature_weights, edge_geometry.tangent_quadrature_weights, atol=ATOL
    )
    assert torch.allclose(rebuilt.angular_phases, edge_geometry.angular_phases, atol=ATOL)
    assert torch.allclose(rebuilt.transport_phases, edge_geometry.transport_phases, atol=ATOL)


def test_from_index_applies_no_radius_filtering_of_its_own(
    grid: SphericalGrid,
    frames: torch.Tensor,
) -> None:
    """A hand-specified edge list -- e.g. IcoDecode's exact parent-child
    pairs, which are not the output of any radius search at all -- must be
    trusted exactly as given, including an edge geodesically longer than any
    cutoff this module's own radius search would ever be asked to use."""
    long_edge = torch.tensor([[0], [grid.n_nodes - 1]], dtype=torch.long)  # (2, 1)

    result = edge_geometry_from_index(
        nodes=grid.nodes,
        frames=frames,
        spherical_quadrature_weights=grid.quadrature_weights,
        edge_index=long_edge,
        n_target_nodes=grid.n_nodes,
        max_frequency=DEFAULT_MAX_FREQUENCY,
    )

    assert result.n_edges == 1
    assert torch.isfinite(result.distances).all()


def test_from_index_sizes_row_ptr_from_n_target_nodes_not_edge_index(
    grid: SphericalGrid,
    frames: torch.Tensor,
) -> None:
    """The bipartite case this exists for: targets are a strict subset of the
    node set edge_index's sources are drawn from (e.g. IcoEncode's coarse
    output, a prefix of the fine input nodes) -- row_ptr must be sized for
    that smaller target count, not the full node set edge_index indexes into."""
    n_coarse = 3
    # Every fine node sends an edge to coarse node 0 -- deliberately not
    # something a radius search on this grid would produce.
    edge_index = torch.stack(
        [torch.zeros(grid.n_nodes, dtype=torch.long), torch.arange(grid.n_nodes)]
    )

    result = edge_geometry_from_index(
        nodes=grid.nodes,
        frames=frames,
        spherical_quadrature_weights=grid.quadrature_weights,
        edge_index=edge_index,
        n_target_nodes=n_coarse,
        max_frequency=DEFAULT_MAX_FREQUENCY,
    )

    assert result.row_ptr.shape == (n_coarse + 1,)
    assert result.row_ptr[0].item() == 0
    assert result.row_ptr[1].item() == grid.n_nodes  # every edge targets node 0
    assert torch.equal(result.row_ptr[1:], result.row_ptr[-1].expand(n_coarse))


def test_self_loop_distances_are_near_zero_but_not_exactly_zero(
    edge_geometry: EdgeGeometry,
) -> None:
    """Origin detection must use a tolerance, not ``== 0`` or ``> 0``.

    ``distances`` is ``atan2(|p x q|, <p, q>)``, and for a self-loop the
    cross-product norm is a rounding artifact rather than an exact zero. The
    resulting distance is around 1e-17: tiny, but strictly positive for most
    nodes. A steerable kernel selecting its self-interaction branch with
    ``distances == 0`` would therefore route almost every self-loop through
    the angular path, where the polar angle is meaningless.
    """
    self_loops = edge_geometry.target_index == edge_geometry.source_index
    self_loop_distances = edge_geometry.distances[self_loops]

    assert bool((self_loop_distances < ORIGIN_TOLERANCE).all())
    # Strictly positive for at least one node -- the point of this test.
    assert bool((self_loop_distances > 0).any())
