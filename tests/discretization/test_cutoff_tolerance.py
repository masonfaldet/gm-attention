"""Boundary shells are kept whole: one tolerant comparison behind every radius test.

On a structured grid a radius that is a multiple of the spacing puts whole
shells of nodes exactly on it; floating point then lands each shell node a
rounding error either side, and a strict ``distance <= radius`` splits the
shell by rounding luck -- differently for a node and its image under a
rotation that maps the grid onto itself. These tests pin the convention
that fixes it: the search, the cached radius subsets, the bipartite search
and the hard window all decide a shell the same way.
"""

from __future__ import annotations

import math

import pytest
import torch

from gm_attention.discretization.discretization import SphericalDiscretization
from gm_attention.discretization.edge_geometry import (
    compute_bipartite_edge_geometry,
    cutoff_tolerance,
    within_cutoff,
)
from gm_attention.discretization.frames import build_tangent_frames
from gm_attention.discretization.grids.torch_harmonics import build_torch_harmonics_grid
from gm_attention.nn.convolution import evaluate_aggregation_window

NLAT, NLON = 24, 48
ROW = math.pi / (NLAT - 1)


def _degrees_by_row(edges, n_targets: int, nlon: int) -> torch.Tensor:  # (nlat, nlon)
    return torch.bincount(edges.target_index, minlength=n_targets).reshape(-1, nlon)


def test_the_helpers_state_the_convention() -> None:
    assert cutoff_tolerance(torch.float64) == 1e-10
    assert cutoff_tolerance(torch.float32) == 1e-5
    d = torch.tensor([0.5 - 1e-7, 0.5, 0.5 + 1e-7, 0.5 + 1e-4], dtype=torch.float32)
    assert within_cutoff(d, 0.5).tolist() == [True, True, True, False]
    d = torch.tensor([0.5 - 1e-12, 0.5, 0.5 + 1e-12, 0.5 + 1e-9], dtype=torch.float64)
    assert within_cutoff(d, 0.5).tolist() == [True, True, True, False]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("rows", [1.0, 3.0])
def test_same_grid_shells_are_whole_in_every_column(dtype: torch.dtype, rows: float) -> None:
    """A radius of an integer number of row spacings: every node of a row
    must see the same number of neighbours, whatever its column, and the
    hard window must accept every edge the radius subset selected."""
    grid = build_torch_harmonics_grid("equiangular", NLAT, NLON, dtype=dtype)
    discretization = SphericalDiscretization.from_grid(grid, coverage_radius=3.0 * ROW)
    edges = discretization.edges_within(rows * ROW)
    degrees = _degrees_by_row(edges, grid.n_nodes, NLON)
    assert bool((degrees == degrees[:, :1]).all()), "a shell was split between columns"
    # The meridian shell itself is inside: the nodes exactly `rows` rows away.
    interior = degrees[int(rows) : NLAT - int(rows)]
    assert int(interior.min()) >= 2 * int(rows) + 1
    window = evaluate_aggregation_window(edges.distances, rows * ROW, "hard")
    assert bool((window == 1).all())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_bipartite_shells_are_whole_in_every_column(dtype: torch.dtype) -> None:
    """Coarse (12 x 24) <- fine (24 x 48) at a radius of two fine rows: the
    fine nodes two rows from a coarse node sit exactly on the radius."""
    fine = build_torch_harmonics_grid("equiangular", NLAT, NLON, dtype=dtype)
    coarse = build_torch_harmonics_grid("equiangular", NLAT // 2, NLON // 2, dtype=dtype)
    edges = compute_bipartite_edge_geometry(
        fine.nodes, build_tangent_frames(fine.nodes), fine.quadrature_weights,
        coarse.nodes, build_tangent_frames(coarse.nodes), cutoff=2.0 * ROW, max_frequency=1,
    )
    degrees = _degrees_by_row(edges, coarse.n_nodes, NLON // 2)
    assert bool((degrees == degrees[:, :1]).all()), "a shell was split between columns"
    window = evaluate_aggregation_window(edges.distances, 2.0 * ROW, "hard")
    assert bool((window == 1).all())
    assert bool(within_cutoff(edges.distances, 2.0 * ROW).all())


def test_the_legacy_rule_reproduces_the_historical_edge_sets(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shallow-water arms of the thesis tables were trained under the
    pre-2026-09-10 comparisons: the encode hop of every depth-1 arm (L5 -> L4
    at the L4 coarse edge length) has a 960-edge shell exactly on its
    radius, of which the old rule kept 480 by rounding luck (30,906 stored,
    30,882 with a nonzero hard window); the whole-shell rule keeps all 960.
    The legacy switch exists so those checkpoints are re-evaluated over the
    edges they trained with, and these are its anchors."""
    from gm_attention.discretization import edge_geometry
    from gm_attention.discretization.grids.icosphere import (
        build_icosphere_downsample_edges,
        build_icosphere_refinement,
    )

    coarse, fine, refinement = build_icosphere_refinement(4, dtype=torch.float32)
    frames = build_tangent_frames(fine.nodes)
    counts = {}
    for rule in ("whole-shell", "legacy"):
        monkeypatch.setattr(edge_geometry, "CUTOFF_RULE", rule)
        assert edge_geometry.cutoff_rule() == rule
        edges = build_icosphere_downsample_edges(fine, frames, coarse.n_nodes, cutoff=refinement.coarse_edge_length)
        window = evaluate_aggregation_window(edges.distances, refinement.coarse_edge_length, "hard")
        counts[rule] = (edges.n_edges, int((window > 0).sum()))
    assert counts["legacy"] == (30906, 30882)
    assert counts["whole-shell"] == (31362, 31362)
    monkeypatch.setattr(edge_geometry, "CUTOFF_RULE", "nonsense")
    with pytest.raises(ValueError, match="GM_CUTOFF_RULE"):
        edge_geometry.within_cutoff(torch.zeros(1), 1.0)


def test_membership_is_decided_from_the_float64_lattice() -> None:
    """A pair a few nanometres beyond a radius is out, and a few inside is
    in, whole orbit and all, in a float32 build: the decision comes from the
    builders' float64 coordinates, not from the stored float32 distances,
    whose error (~1e-7) would decide such a margin by luck. The float32 and
    float64 builds therefore agree edge for edge, bipartite sets included."""
    from gm_attention.discretization.edge_geometry import EXACT_SLACK

    grid32 = build_torch_harmonics_grid("equiangular", NLAT, NLON, dtype=torch.float32)
    grid64 = build_torch_harmonics_grid("equiangular", NLAT, NLON, dtype=torch.float64)
    assert grid32.exact_nodes is not None and torch.equal(grid32.exact_nodes, grid64.nodes)
    target, source = 6 * NLON + 0, 9 * NLON + 5  # rows 6 and 9, five columns apart: a generic pair
    p, q = grid32.exact_nodes[target], grid32.exact_nodes[source]
    exact = float(torch.atan2(torch.linalg.cross(p, q).norm(), p @ q))
    margin = 3e-9
    assert margin > 10 * EXACT_SLACK
    for radius, expected in ((exact - margin, False), (exact + margin, True)):
        sets = []
        for grid in (grid32, grid64):
            discretization = SphericalDiscretization.from_grid(grid, coverage_radius=exact + ROW)
            edges = discretization.edges_within(radius)
            pairs = set((edges.edge_index[0] * (1 << 32) + edges.edge_index[1]).tolist())
            sets.append(pairs)
            assert ((target << 32) + source in pairs) is expected
            # Whole orbit: the same decision in every column.
            degrees = _degrees_by_row(edges, grid.n_nodes, NLON)
            assert bool((degrees == degrees[:, :1]).all())
            window = evaluate_aggregation_window(edges.distances, radius, "hard")
            assert bool((window == 1).all()), "the window never rejects an edge the exact decision admitted"
        assert sets[0] == sets[1], "float32 and float64 builds give the same edges"
    coarse32 = build_torch_harmonics_grid("equiangular", NLAT // 2, NLON // 2, dtype=torch.float32)
    coarse64 = build_torch_harmonics_grid("equiangular", NLAT // 2, NLON // 2, dtype=torch.float64)
    bipartite = []
    for fine, coarse in ((grid32, coarse32), (grid64, coarse64)):
        edges = compute_bipartite_edge_geometry(
            fine.nodes, build_tangent_frames(fine.nodes), fine.quadrature_weights,
            coarse.nodes, build_tangent_frames(coarse.nodes), cutoff=exact + margin, max_frequency=1,
            source_exact_nodes=fine.exact_nodes, target_exact_nodes=coarse.exact_nodes,
        )
        bipartite.append(set((edges.edge_index[0] * (1 << 32) + edges.edge_index[1]).tolist()))
    assert bipartite[0] == bipartite[1]


def test_a_grid_without_exact_coordinates_falls_back_to_the_dtype_slack() -> None:
    from gm_attention.discretization.grid import SphericalGrid

    built = build_torch_harmonics_grid("equiangular", NLAT, NLON, dtype=torch.float32)
    plain = SphericalGrid(built.nodes, built.quadrature_weights)
    assert plain.exact_nodes is None
    discretization = SphericalDiscretization.from_grid(plain, coverage_radius=3.0 * ROW)
    assert discretization._exact_distances is None
    edges = discretization.edges_within(3.0 * ROW)
    degrees = _degrees_by_row(edges, plain.n_nodes, NLON)
    assert bool((degrees == degrees[:, :1]).all()), "exact shells are still whole under the 1e-5 slack"
    assert plain.clone().exact_nodes is None and built.to(device="cpu").exact_nodes is not None
