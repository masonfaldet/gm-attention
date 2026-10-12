"""Tests for gm_attention.discretization.discretization.SphericalDiscretization.

Covers construction and its validation of geometric invariants (frame
orthonormality/orientation, edge_index range, row_ptr consistency, coverage
radius), ``edges_within`` caching, and ``prepare_windows``.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch

from gm_attention.discretization.discretization import SphericalDiscretization
from gm_attention.discretization.edge_geometry import EdgeGeometry
from gm_attention.discretization.frames import build_tangent_frames
from gm_attention.discretization.grid import SphericalGrid
from gm_attention.discretization.grids.icosphere import build_icosphere_grid


@pytest.fixture
def grid() -> SphericalGrid:
    return build_icosphere_grid(subdivisions=1)


def _corrupted(edge_geometry: EdgeGeometry, **overrides: torch.Tensor) -> EdgeGeometry:
    # dataclasses.replace keeps this helper correct as EdgeGeometry gains
    # fields, instead of silently dropping the ones it does not enumerate.
    return dataclasses.replace(edge_geometry, **overrides)


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


def test_from_grid_uses_deterministic_frames_by_default(
    grid: SphericalGrid,
) -> None:
    discretization = SphericalDiscretization.from_grid(grid, coverage_radius=1.0)
    expected_frames = build_tangent_frames(grid.nodes)
    assert torch.equal(discretization.frames, expected_frames)


def test_from_grid_accepts_user_supplied_frames(grid: SphericalGrid) -> None:
    frames = build_tangent_frames(grid.nodes)
    discretization = SphericalDiscretization.from_grid(
        grid,
        coverage_radius=1.0,
        frames=frames,
    )
    assert torch.equal(discretization.frames, frames)


def test_properties_match_underlying_grid_and_edge_geometry(
    grid: SphericalGrid,
) -> None:
    discretization = SphericalDiscretization.from_grid(grid, coverage_radius=1.0)
    assert discretization.n_nodes == grid.n_nodes
    assert discretization.n_edges == discretization.edge_geometry.n_edges
    assert torch.equal(discretization.nodes, grid.nodes)
    assert torch.equal(
        discretization.quadrature_weights,
        grid.quadrature_weights,
    )
    assert discretization.dtype == grid.dtype
    assert discretization.device == grid.device


def test_repr_smoke() -> None:
    grid = build_icosphere_grid(0)
    discretization = SphericalDiscretization.from_grid(grid, coverage_radius=1.0)
    text = repr(discretization)
    assert "SphericalDiscretization" in text
    assert f"n_nodes={grid.n_nodes}" in text


@pytest.mark.parametrize(
    "bad_radius",
    [0.0, -1.0, math.pi, math.pi + 0.5, float("inf"), float("nan")],
)
def test_from_grid_rejects_invalid_coverage_radius(
    grid: SphericalGrid,
    bad_radius: float,
) -> None:
    with pytest.raises(ValueError):
        SphericalDiscretization.from_grid(grid, coverage_radius=bad_radius)


def test_constructor_rejects_non_grid() -> None:
    grid = build_icosphere_grid(0)
    frames = build_tangent_frames(grid.nodes)
    discretization = SphericalDiscretization.from_grid(grid, coverage_radius=1.0)

    with pytest.raises(TypeError, match="SphericalGrid"):
        SphericalDiscretization(
            grid="not a grid",  # type: ignore[arg-type]
            frames=frames,
            edge_geometry=discretization.edge_geometry,
            coverage_radius=1.0,
        )


def test_constructor_rejects_wrong_frame_shape(grid: SphericalGrid) -> None:
    discretization = SphericalDiscretization.from_grid(grid, coverage_radius=1.0)
    bad_frames = torch.zeros(grid.n_nodes, 3, 3, dtype=grid.dtype)

    with pytest.raises(ValueError, match="frames must have shape"):
        SphericalDiscretization(
            grid=grid,
            frames=bad_frames,
            edge_geometry=discretization.edge_geometry,
            coverage_radius=1.0,
        )


def test_constructor_rejects_dtype_mismatch(grid: SphericalGrid) -> None:
    discretization = SphericalDiscretization.from_grid(grid, coverage_radius=1.0)
    mismatched_frames = discretization.frames.to(torch.float32)

    with pytest.raises(ValueError, match="same dtype"):
        SphericalDiscretization(
            grid=grid,
            frames=mismatched_frames,
            edge_geometry=discretization.edge_geometry,
            coverage_radius=1.0,
        )


def test_constructor_rejects_nonfinite_frames(grid: SphericalGrid) -> None:
    discretization = SphericalDiscretization.from_grid(grid, coverage_radius=1.0)
    bad_frames = discretization.frames.clone()
    bad_frames[0, 0, 0] = float("nan")

    with pytest.raises(ValueError, match="nonfinite"):
        SphericalDiscretization(
            grid=grid,
            frames=bad_frames,
            edge_geometry=discretization.edge_geometry,
            coverage_radius=1.0,
        )


def test_constructor_rejects_non_orthonormal_frames(grid: SphericalGrid) -> None:
    discretization = SphericalDiscretization.from_grid(grid, coverage_radius=1.0)
    bad_frames = discretization.frames.clone()
    bad_frames[0, :, 1] *= 2.0  # break unit norm of the second column

    with pytest.raises(ValueError, match="orthonormal"):
        SphericalDiscretization(
            grid=grid,
            frames=bad_frames,
            edge_geometry=discretization.edge_geometry,
            coverage_radius=1.0,
        )


def test_constructor_rejects_misoriented_frames(grid: SphericalGrid) -> None:
    discretization = SphericalDiscretization.from_grid(grid, coverage_radius=1.0)
    bad_frames = discretization.frames.clone()
    # Swap the two frame vectors at one node: still orthonormal, but now
    # cross(e_1, e_2) = -p at that node, flipping the orientation.
    bad_frames[0] = bad_frames[0].flip(-1)

    with pytest.raises(ValueError, match="positively oriented"):
        SphericalDiscretization(
            grid=grid,
            frames=bad_frames,
            edge_geometry=discretization.edge_geometry,
            coverage_radius=1.0,
        )


def test_constructor_rejects_out_of_range_edge_index(grid: SphericalGrid) -> None:
    discretization = SphericalDiscretization.from_grid(grid, coverage_radius=1.0)
    bad_edge_index = discretization.edge_geometry.edge_index.clone()
    bad_edge_index[0, 0] = grid.n_nodes  # one past the last valid index

    with pytest.raises(ValueError, match="edge_index contains"):
        SphericalDiscretization(
            grid=grid,
            frames=discretization.frames,
            edge_geometry=_corrupted(
                discretization.edge_geometry,
                edge_index=bad_edge_index,
            ),
            coverage_radius=1.0,
        )


def test_constructor_rejects_row_ptr_wrong_shape(grid: SphericalGrid) -> None:
    discretization = SphericalDiscretization.from_grid(grid, coverage_radius=1.0)

    with pytest.raises(ValueError, match="row_ptr must have shape"):
        SphericalDiscretization(
            grid=grid,
            frames=discretization.frames,
            edge_geometry=_corrupted(
                discretization.edge_geometry,
                row_ptr=discretization.edge_geometry.row_ptr[:-1],
            ),
            coverage_radius=1.0,
        )


def test_constructor_rejects_row_ptr_bad_endpoint(grid: SphericalGrid) -> None:
    discretization = SphericalDiscretization.from_grid(grid, coverage_radius=1.0)
    bad_row_ptr = discretization.edge_geometry.row_ptr.clone()
    bad_row_ptr[-1] += 1

    with pytest.raises(ValueError, match="non-decreasing"):
        SphericalDiscretization(
            grid=grid,
            frames=discretization.frames,
            edge_geometry=_corrupted(
                discretization.edge_geometry,
                row_ptr=bad_row_ptr,
            ),
            coverage_radius=1.0,
        )


def test_constructor_rejects_row_ptr_non_monotonic(grid: SphericalGrid) -> None:
    discretization = SphericalDiscretization.from_grid(grid, coverage_radius=1.0)
    bad_row_ptr = discretization.edge_geometry.row_ptr.clone()
    if bad_row_ptr.shape[0] > 2:
        bad_row_ptr[1] = bad_row_ptr[2] + 1

    with pytest.raises(ValueError, match="non-decreasing"):
        SphericalDiscretization(
            grid=grid,
            frames=discretization.frames,
            edge_geometry=_corrupted(
                discretization.edge_geometry,
                row_ptr=bad_row_ptr,
            ),
            coverage_radius=1.0,
        )


def test_constructor_rejects_edges_outside_coverage_radius(
    grid: SphericalGrid,
) -> None:
    wide = SphericalDiscretization.from_grid(grid, coverage_radius=1.5)

    with pytest.raises(ValueError, match="outside the declared"):
        SphericalDiscretization(
            grid=grid,
            frames=wide.frames,
            edge_geometry=wide.edge_geometry,
            coverage_radius=0.5,
        )


# --------------------------------------------------------------------------
# edges_within / prepare_windows caching
# --------------------------------------------------------------------------


def test_edges_within_caches_by_radius(grid: SphericalGrid) -> None:
    discretization = SphericalDiscretization.from_grid(grid, coverage_radius=1.0)

    first = discretization.edges_within(0.4)
    second = discretization.edges_within(0.4)
    assert first is second
    assert discretization.cached_radii == (
        discretization._radius_key(0.4),
        discretization._radius_key(1.0),
    )


def test_edges_within_matches_direct_select(grid: SphericalGrid) -> None:
    discretization = SphericalDiscretization.from_grid(grid, coverage_radius=1.0)
    subset = discretization.edges_within(0.6)
    expected = discretization.edge_geometry.select(
        discretization.edge_geometry.distances <= 0.6
    )
    assert torch.equal(subset.edge_index, expected.edge_index)


def test_edges_within_at_full_coverage_radius_returns_original(
    grid: SphericalGrid,
) -> None:
    discretization = SphericalDiscretization.from_grid(grid, coverage_radius=1.0)
    assert discretization.edges_within(1.0) is discretization.edge_geometry


def test_edges_within_rejects_radius_above_coverage(grid: SphericalGrid) -> None:
    discretization = SphericalDiscretization.from_grid(grid, coverage_radius=1.0)
    with pytest.raises(ValueError, match="exceeds"):
        discretization.edges_within(1.5)


@pytest.mark.parametrize("bad_radius", [0.0, -0.1])
def test_edges_within_rejects_nonpositive_radius(
    grid: SphericalGrid,
    bad_radius: float,
) -> None:
    discretization = SphericalDiscretization.from_grid(grid, coverage_radius=1.0)
    with pytest.raises(ValueError):
        discretization.edges_within(bad_radius)


def test_prepare_windows_populates_cache_for_all_radii(
    grid: SphericalGrid,
) -> None:
    discretization = SphericalDiscretization.from_grid(grid, coverage_radius=1.0)
    discretization.prepare_windows([0.2, 0.5, 0.8])

    for radius in (0.2, 0.5, 0.8, 1.0):
        assert discretization._radius_key(radius) in discretization._radius_cache
