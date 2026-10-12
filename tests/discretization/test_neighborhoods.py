"""Tests for the KD-tree-based neighbor discovery in edge_geometry.py.

Cross-checks ``compute_edge_geometry``'s edge set against a brute-force,
O(n^2) all-pairs geodesic-distance computation on small random point
clouds, and checks the target-major / source-minor ordering guarantee that
downstream code (row_ptr, edges_within) depends on.
"""

from __future__ import annotations

import math

import pytest
import torch

from gm_attention.discretization.edge_geometry import compute_edge_geometry
from gm_attention.discretization.frames import build_tangent_frames
from gm_attention.discretization.grid import SphericalGrid


def _random_unit_vectors(
    n: int,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    points = torch.randn(n, 3, dtype=torch.float64, generator=generator)
    return points / torch.linalg.vector_norm(points, dim=-1, keepdim=True)


def _random_grid(n: int, seed: int) -> SphericalGrid:
    generator = torch.Generator().manual_seed(seed)
    nodes = _random_unit_vectors(n, generator=generator)
    weights = torch.full(
        (n,),
        4.0 * math.pi / n,
        dtype=torch.float64,
    )
    return SphericalGrid(nodes, weights)


def _brute_force_edges(nodes: torch.Tensor, radius: float) -> set[tuple[int, int]]:
    dots = (nodes @ nodes.T).clamp(-1.0, 1.0)
    distances = torch.arccos(dots)
    mask = distances <= radius + 1e-9

    targets, sources = torch.nonzero(mask, as_tuple=True)
    return set(zip(targets.tolist(), sources.tolist()))


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("radius", [0.3, 0.7, 1.5])
def test_edge_set_matches_brute_force_search(
    seed: int,
    radius: float,
) -> None:
    grid = _random_grid(n=60, seed=seed)
    frames = build_tangent_frames(grid.nodes)

    edge_geometry = compute_edge_geometry(
        grid=grid,
        frames=frames,
        coverage_radius=radius,
    )

    actual_edges = set(
        zip(
            edge_geometry.target_index.tolist(),
            edge_geometry.source_index.tolist(),
        )
    )
    expected_edges = _brute_force_edges(grid.nodes, radius)

    assert actual_edges == expected_edges


def test_no_duplicate_edges() -> None:
    grid = _random_grid(n=80, seed=4)
    frames = build_tangent_frames(grid.nodes)

    edge_geometry = compute_edge_geometry(
        grid=grid,
        frames=frames,
        coverage_radius=1.0,
    )

    pairs = list(
        zip(
            edge_geometry.target_index.tolist(),
            edge_geometry.source_index.tolist(),
        )
    )
    assert len(pairs) == len(set(pairs))


def test_edges_are_target_major_source_minor_sorted() -> None:
    grid = _random_grid(n=80, seed=5)
    frames = build_tangent_frames(grid.nodes)

    edge_geometry = compute_edge_geometry(
        grid=grid,
        frames=frames,
        coverage_radius=1.0,
    )

    target = edge_geometry.target_index
    source = edge_geometry.source_index

    assert bool((target[1:] >= target[:-1]).all())

    same_target = target[1:] == target[:-1]
    assert bool((source[1:][same_target] > source[:-1][same_target]).all())


def test_larger_radius_yields_a_superset_of_edges() -> None:
    grid = _random_grid(n=60, seed=6)
    frames = build_tangent_frames(grid.nodes)

    small = compute_edge_geometry(grid=grid, frames=frames, coverage_radius=0.4)
    large = compute_edge_geometry(grid=grid, frames=frames, coverage_radius=1.2)

    small_edges = set(
        zip(small.target_index.tolist(), small.source_index.tolist())
    )
    large_edges = set(
        zip(large.target_index.tolist(), large.source_index.tolist())
    )
    assert small_edges <= large_edges


def test_every_node_is_its_own_neighbor_at_any_positive_radius() -> None:
    grid = _random_grid(n=30, seed=7)
    frames = build_tangent_frames(grid.nodes)

    edge_geometry = compute_edge_geometry(
        grid=grid,
        frames=frames,
        coverage_radius=1e-6,
    )
    self_loops = (edge_geometry.target_index == edge_geometry.source_index).sum()
    assert int(self_loops) == grid.n_nodes
