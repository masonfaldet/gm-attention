"""dtype/device transfer (`.to()`) tests across the discretization stack.

Covers SphericalGrid, EdgeGeometry, and SphericalDiscretization, including
that index tensors (edge_index, row_ptr) stay torch.long under a dtype
transfer meant for the floating-point fields, and that cached
``edges_within`` subsets travel along with the discretization.
"""

from __future__ import annotations

import pytest
import torch

from gm_attention.discretization.discretization import SphericalDiscretization
from gm_attention.discretization.edge_geometry import compute_edge_geometry
from gm_attention.discretization.frames import build_tangent_frames
from gm_attention.discretization.grids.icosphere import build_icosphere_grid

TOLERANCE = 1e-5  # crossing to float32 and back loses precision


# --------------------------------------------------------------------------
# SphericalGrid
# --------------------------------------------------------------------------


def test_grid_to_changes_dtype() -> None:
    grid = build_icosphere_grid(1)
    moved = grid.to(dtype=torch.float32)

    assert moved.dtype == torch.float32
    assert grid.dtype == torch.float64  # original untouched
    assert torch.allclose(
        moved.nodes.to(torch.float64),
        grid.nodes,
        atol=TOLERANCE,
    )


def test_grid_to_is_a_new_object() -> None:
    grid = build_icosphere_grid(0)
    moved = grid.to(dtype=torch.float64)
    assert moved is not grid
    assert torch.equal(moved.nodes, grid.nodes)


def test_grid_to_cpu_noop_preserves_values() -> None:
    grid = build_icosphere_grid(0)
    moved = grid.to(device="cpu")
    assert torch.equal(moved.nodes, grid.nodes)
    assert torch.equal(moved.quadrature_weights, grid.quadrature_weights)


# --------------------------------------------------------------------------
# EdgeGeometry
# --------------------------------------------------------------------------


def _edge_geometry(dtype: torch.dtype = torch.float64):
    grid = build_icosphere_grid(1, dtype=dtype)
    frames = build_tangent_frames(grid.nodes)
    return compute_edge_geometry(grid=grid, frames=frames, coverage_radius=1.0)


def test_edge_geometry_to_casts_float_fields_only() -> None:
    edge_geometry = _edge_geometry(torch.float64)
    moved = edge_geometry.to(dtype=torch.float32)

    assert moved.log_coordinates.dtype == torch.float32
    assert moved.distances.dtype == torch.float32
    assert moved.transport_angles.dtype == torch.float32
    assert moved.tangent_quadrature_weights.dtype == torch.float32

    # Index tensors must stay torch.long regardless of the float dtype arg.
    assert moved.edge_index.dtype == torch.long
    assert moved.row_ptr.dtype == torch.long
    assert torch.equal(moved.edge_index, edge_geometry.edge_index)
    assert torch.equal(moved.row_ptr, edge_geometry.row_ptr)


def test_edge_geometry_to_preserves_values_within_tolerance() -> None:
    edge_geometry = _edge_geometry(torch.float64)
    moved = edge_geometry.to(dtype=torch.float32)

    assert torch.allclose(
        moved.distances.to(torch.float64),
        edge_geometry.distances,
        atol=TOLERANCE,
    )
    assert torch.allclose(
        moved.log_coordinates.to(torch.float64),
        edge_geometry.log_coordinates,
        atol=TOLERANCE,
    )


def test_edge_geometry_to_default_dtype_matches_distances_dtype() -> None:
    edge_geometry = _edge_geometry(torch.float64)
    moved = edge_geometry.to()  # no dtype/device given
    assert moved.distances.dtype == edge_geometry.distances.dtype
    assert torch.equal(moved.edge_index, edge_geometry.edge_index)


# --------------------------------------------------------------------------
# SphericalDiscretization
# --------------------------------------------------------------------------


def test_discretization_to_casts_grid_frames_and_edges() -> None:
    grid = build_icosphere_grid(1)
    discretization = SphericalDiscretization.from_grid(grid, coverage_radius=1.0)

    moved = discretization.to(dtype=torch.float32)

    assert moved.dtype == torch.float32
    assert moved.frames.dtype == torch.float32
    assert moved.edge_geometry.distances.dtype == torch.float32
    assert moved.edge_geometry.edge_index.dtype == torch.long
    assert moved.edge_geometry.row_ptr.dtype == torch.long

    # original is untouched
    assert discretization.dtype == torch.float64


def test_discretization_to_is_still_internally_valid() -> None:
    # SphericalDiscretization.__init__ re-runs full validation, so simply
    # constructing successfully here re-checks frame orthonormality/
    # orientation, edge_index range, row_ptr consistency, and the coverage
    # radius bound on the moved copy.
    grid = build_icosphere_grid(1)
    discretization = SphericalDiscretization.from_grid(grid, coverage_radius=1.0)
    moved = discretization.to(dtype=torch.float32)
    assert moved.n_nodes == discretization.n_nodes
    assert moved.n_edges == discretization.n_edges


def test_discretization_to_carries_cached_radii() -> None:
    grid = build_icosphere_grid(1)
    discretization = SphericalDiscretization.from_grid(grid, coverage_radius=1.0)
    discretization.prepare_windows([0.3, 0.6])

    moved = discretization.to(dtype=torch.float32)

    assert moved.cached_radii == discretization.cached_radii
    for radius in (0.3, 0.6, 1.0):
        cached = moved.edges_within(radius)
        assert cached.distances.dtype == torch.float32


def test_discretization_to_noop_preserves_edge_count() -> None:
    grid = build_icosphere_grid(0)
    discretization = SphericalDiscretization.from_grid(grid, coverage_radius=1.0)
    moved = discretization.to()
    assert moved.n_edges == discretization.n_edges
    assert moved.n_nodes == discretization.n_nodes


def test_discretization_to_float32_round_trip_fails_strict_revalidation() -> None:
    # A float64 -> float32 -> float64 round trip does not recover exact
    # unit-norm nodes: float32 rounding leaves a ~1e-8 norm error, well
    # above SphericalGrid's 1e-10 float64 tolerance. Since `.to()`
    # reconstructs (and so re-validates) at every call, the final cast
    # correctly raises instead of silently accepting degraded geometry.
    grid = build_icosphere_grid(1)
    discretization = SphericalDiscretization.from_grid(grid, coverage_radius=1.0)

    downcast = discretization.to(dtype=torch.float32)
    with pytest.raises(ValueError, match="unit sphere"):
        downcast.to(dtype=torch.float64)
