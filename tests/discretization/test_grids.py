"""Structural tests for the grid builders in gm_attention.discretization.grids.

These check the SphericalGrid postconditions (unit-norm nodes, weights
summing to 4*pi), shapes, dtype/device handling, and error cases for each
builder. Numerical *accuracy* of the resulting quadrature rules is covered
separately in test_quadrature.py.
"""

from __future__ import annotations

import math

import pytest
import torch

from gm_attention.discretization.frames import (
    build_tangent_frames,
    rotate_tangent_frames,
)
from gm_attention.discretization.grid import (
    SphericalGrid,
    median_nearest_neighbor_distance,
)
from gm_attention.discretization.grids.icosphere import (
    build_icosphere_downsample_edges,
    build_icosphere_grid,
    build_icosphere_multiresolution,
    build_icosphere_refinement,
    build_icosphere_upsample_edges,
)
from gm_attention.discretization.grids.torch_harmonics import (
    _GRID_RULES,
    build_torch_harmonics_grid,
)

FOUR_PI = 4.0 * math.pi


def _assert_valid_grid(
    grid: SphericalGrid,
    *,
    expected_n_nodes: int,
    dtype: torch.dtype,
) -> None:
    assert isinstance(grid, SphericalGrid)
    assert grid.n_nodes == expected_n_nodes
    assert grid.dtype == dtype
    assert grid.nodes.shape == (expected_n_nodes, 3)
    assert grid.quadrature_weights.shape == (expected_n_nodes,)

    tolerance = 1e-10 if dtype == torch.float64 else 1e-5

    norms = torch.linalg.vector_norm(grid.nodes, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=tolerance)

    assert bool((grid.quadrature_weights > 0).all())
    assert grid.total_quadrature_weight.item() == pytest.approx(
        FOUR_PI,
        abs=tolerance,
    )


# --------------------------------------------------------------------------
# torch_harmonics-backed grids
# --------------------------------------------------------------------------


@pytest.mark.parametrize("grid_type", sorted(_GRID_RULES))
def test_torch_harmonics_grid_is_valid(grid_type: str) -> None:
    n_latitude, n_longitude = 8, 16
    grid = build_torch_harmonics_grid(grid_type, n_latitude, n_longitude)
    _assert_valid_grid(
        grid,
        expected_n_nodes=n_latitude * n_longitude,
        dtype=torch.float64,
    )


def test_torch_harmonics_grid_dtype_and_device(
    dtype: torch.dtype,
) -> None:
    grid = build_torch_harmonics_grid(
        "legendre-gauss",
        6,
        12,
        dtype=dtype,
        device="cpu",
    )
    _assert_valid_grid(grid, expected_n_nodes=72, dtype=dtype)
    assert grid.device.type == "cpu"


def test_torch_harmonics_grid_single_node() -> None:
    # Only legendre-gauss is defined for a single latitude node; lobatto
    # always includes both poles and clenshaw-curtis requires n > 1.
    grid = build_torch_harmonics_grid("legendre-gauss", 1, 1)
    _assert_valid_grid(grid, expected_n_nodes=1, dtype=torch.float64)


@pytest.mark.parametrize("grid_type", ["lobatto", "equiangular"])
def test_torch_harmonics_grid_requires_at_least_two_latitudes(
    grid_type: str,
) -> None:
    with pytest.raises(ValueError, match="n_latitude"):
        build_torch_harmonics_grid(grid_type, 1, 4)


@pytest.mark.parametrize(
    "grid_type",
    ["equiangular-trapezoidal", "equidistant", "not-a-grid"],
)
def test_torch_harmonics_grid_rejects_unsupported_type(
    grid_type: str,
) -> None:
    with pytest.raises(ValueError, match="grid must be one of"):
        build_torch_harmonics_grid(grid_type, 4, 4)


@pytest.mark.parametrize("n_latitude,n_longitude", [(0, 4), (4, 0), (-1, 4)])
def test_torch_harmonics_grid_rejects_nonpositive_counts(
    n_latitude: int,
    n_longitude: int,
) -> None:
    with pytest.raises(ValueError):
        build_torch_harmonics_grid("equiangular", n_latitude, n_longitude)


# --------------------------------------------------------------------------
# Icosphere grids
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "subdivisions,expected_n_nodes",
    [(0, 12), (1, 42), (2, 162)],
)
def test_icosphere_grid_is_valid(
    subdivisions: int,
    expected_n_nodes: int,
) -> None:
    grid = build_icosphere_grid(subdivisions)
    _assert_valid_grid(
        grid,
        expected_n_nodes=expected_n_nodes,
        dtype=torch.float64,
    )


def test_icosphere_grid_dtype(dtype: torch.dtype) -> None:
    grid = build_icosphere_grid(2, dtype=dtype)
    _assert_valid_grid(grid, expected_n_nodes=162, dtype=dtype)


def test_icosphere_grid_weights_reflect_vertex_valence() -> None:
    # A subdivided icosahedron has exactly 12 valence-5 (pentagon) vertices
    # and the rest valence-6 (hexagon); the pentagon vertices touch one
    # fewer face and should carry systematically lighter dual-cell weight,
    # in a roughly 5:6 ratio to the rest.
    grid = build_icosphere_grid(3)
    sorted_weights, _ = torch.sort(grid.quadrature_weights)

    pentagon_weights = sorted_weights[:12]
    hexagon_weights = sorted_weights[12:]

    assert pentagon_weights.max() < hexagon_weights.min()

    ratio = (pentagon_weights.mean() / hexagon_weights.mean()).item()
    assert 0.7 < ratio < 5.0 / 6.0


def test_icosphere_grid_rejects_negative_subdivisions() -> None:
    with pytest.raises(ValueError):
        build_icosphere_grid(-1)


# --------------------------------------------------------------------------
# build_icosphere_refinement
# --------------------------------------------------------------------------


def test_coarse_edge_length_matches_a_known_regular_case() -> None:
    """coarse_subdivisions=0's coarse grid is a bare icosahedron -- every
    edge exactly the same length, arccos(1/sqrt(5)) (a standard closed
    form, already used to pin median_nearest_neighbor_distance above)."""
    _, _, refinement = build_icosphere_refinement(0, dtype=torch.float64)

    assert refinement.coarse_edge_length == pytest.approx(math.acos(1.0 / math.sqrt(5.0)), abs=1e-9)


@pytest.mark.parametrize("coarse_subdivisions", [2, 3, 4])
def test_coarse_edge_length_is_close_to_the_median_nearest_neighbor_distance(
    coarse_subdivisions: int,
) -> None:
    """coarse_edge_length is a single representative edge, not a robust
    statistic -- it should track median_nearest_neighbor_distance closely
    (this is its entire reason to exist: a cheap O(1) stand-in for an O(n
    log n) computation) without being asserted to equal it exactly."""
    coarse, _, refinement = build_icosphere_refinement(coarse_subdivisions, dtype=torch.float64)
    median = median_nearest_neighbor_distance(coarse)

    assert refinement.coarse_edge_length == pytest.approx(median, rel=0.02)


@pytest.mark.parametrize("coarse_subdivisions", [0, 1, 2, 3])
def test_refinement_grids_match_build_icosphere_grid_directly(
    coarse_subdivisions: int,
) -> None:
    coarse, fine, _ = build_icosphere_refinement(coarse_subdivisions, dtype=torch.float64)

    assert torch.equal(coarse.nodes, build_icosphere_grid(coarse_subdivisions, dtype=torch.float64).nodes)
    assert torch.equal(fine.nodes, build_icosphere_grid(coarse_subdivisions + 1, dtype=torch.float64).nodes)


@pytest.mark.parametrize("coarse_subdivisions", [2, 4])
def test_coarse_nodes_are_an_exact_prefix_of_fine_nodes(coarse_subdivisions: int) -> None:
    """atol, not torch.equal: two independent trimesh.creation.icosphere()
    calls (different total subdivide() counts, hence a different array-size
    trajectory even for their shared early steps) can differ by a single
    float64 ULP -- confirmed directly at coarse_subdivisions=4 (~8 of 2562
    nodes, ~2.2e-16), not present at coarse_subdivisions=2, which is why
    both are parametrized here rather than just the smaller, lucky case."""
    coarse, fine, refinement = build_icosphere_refinement(coarse_subdivisions, dtype=torch.float64)

    assert refinement.n_coarse_nodes == coarse.n_nodes
    assert torch.allclose(coarse.nodes, fine.nodes[: refinement.n_coarse_nodes], atol=1e-10)


def test_midpoint_parents_cover_every_remaining_fine_node_exactly_once() -> None:
    coarse, fine, refinement = build_icosphere_refinement(2, dtype=torch.float64)
    n_midpoints = fine.n_nodes - coarse.n_nodes

    assert refinement.midpoint_parents.shape == (n_midpoints, 2)
    assert bool((refinement.midpoint_parents >= 0).all())
    assert bool((refinement.midpoint_parents < coarse.n_nodes).all())
    # Every coarse edge produces a distinct new node -- no coarse-node pair
    # is recorded as the parent of more than one fine node.
    parent_pairs = {tuple(sorted(pair.tolist())) for pair in refinement.midpoint_parents}
    assert len(parent_pairs) == n_midpoints


@pytest.mark.parametrize("coarse_subdivisions", [2, 3, 4])
def test_midpoint_parents_exactly_reconstruct_the_fine_node(coarse_subdivisions: int) -> None:
    """The real correctness property: averaging (then re-normalizing) each
    recorded parent pair must reproduce the actual fine-grid node position
    to floating-point precision -- this is what makes IcoDecode's upsample
    step (nn/blocks.py) a real geometric fact, not an approximation.
    Parametrized because coarse_subdivisions=4 is where the one-ULP,
    array-size-dependent floating-point noise described above this section
    was actually caught -- exercising only a smaller, lucky level would not
    have caught it."""
    coarse, fine, refinement = build_icosphere_refinement(coarse_subdivisions, dtype=torch.float64)
    n_coarse = refinement.n_coarse_nodes

    parents = coarse.nodes[refinement.midpoint_parents]  # (n_midpoints, 2, 3)
    reconstructed = parents.mean(dim=1)
    reconstructed = reconstructed / torch.linalg.vector_norm(reconstructed, dim=-1, keepdim=True)

    assert torch.allclose(reconstructed, fine.nodes[n_coarse:], atol=1e-12)


def test_refinement_rejects_negative_coarse_subdivisions() -> None:
    with pytest.raises(ValueError):
        build_icosphere_refinement(-1)


def test_refinement_respects_dtype_and_device() -> None:
    coarse, fine, refinement = build_icosphere_refinement(1, dtype=torch.float32)

    assert coarse.dtype == torch.float32
    assert fine.dtype == torch.float32
    assert refinement.midpoint_parents.dtype == torch.long


# --------------------------------------------------------------------------
# build_icosphere_downsample_edges / build_icosphere_upsample_edges
# --------------------------------------------------------------------------


def _hierarchy(coarse_subdivisions: int = 2):
    coarse, fine, refinement = build_icosphere_refinement(coarse_subdivisions, dtype=torch.float64)
    fine_frames = build_tangent_frames(fine.nodes)
    return coarse, fine, refinement, fine_frames


def test_downsample_edges_row_ptr_is_sized_for_the_coarse_grid() -> None:
    coarse, fine, refinement, fine_frames = _hierarchy()
    cutoff = median_nearest_neighbor_distance(coarse)

    edges = build_icosphere_downsample_edges(
        fine, fine_frames, refinement.n_coarse_nodes, cutoff=cutoff
    )

    assert edges.row_ptr.shape == (refinement.n_coarse_nodes + 1,)
    assert bool((edges.target_index < refinement.n_coarse_nodes).all())
    assert bool((edges.source_index < fine.n_nodes).all())


def test_downsample_edges_every_coarse_node_has_a_self_loop() -> None:
    """A coarse node is itself one of the fine grid's nodes, so it must
    appear as its own neighbor -- the self-loop SO2SteerableKernel's origin
    branch depends on (kernels.py)."""
    coarse, fine, refinement, fine_frames = _hierarchy()
    cutoff = median_nearest_neighbor_distance(coarse)

    edges = build_icosphere_downsample_edges(
        fine, fine_frames, refinement.n_coarse_nodes, cutoff=cutoff
    )
    self_loops = edges.target_index == edges.source_index

    assert int(self_loops.sum()) == refinement.n_coarse_nodes


def test_downsample_edges_quadrature_weights_are_the_fine_grids_own() -> None:
    coarse, fine, refinement, fine_frames = _hierarchy()
    cutoff = median_nearest_neighbor_distance(coarse)

    edges = build_icosphere_downsample_edges(
        fine, fine_frames, refinement.n_coarse_nodes, cutoff=cutoff
    )
    # tangent_quadrature_weights = fine_grid.quadrature_weights[source] * (a
    # purely geometric r/sin(r) factor) -- so a self-loop's tangent weight
    # (r -> 0, factor -> 1) equals the fine node's own quadrature weight
    # exactly, never the coarse grid's (which, at the same physical node,
    # covers a different, larger dual-cell area).
    self_loops = edges.target_index == edges.source_index
    self_loop_targets = edges.target_index[self_loops]

    assert torch.allclose(
        edges.tangent_quadrature_weights[self_loops],
        fine.quadrature_weights[self_loop_targets],
        atol=1e-10,
    )


def test_downsample_edges_rejects_out_of_range_n_coarse_nodes() -> None:
    _, fine, _, fine_frames = _hierarchy()

    with pytest.raises(ValueError):
        build_icosphere_downsample_edges(fine, fine_frames, 0, cutoff=0.2)

    with pytest.raises(ValueError):
        build_icosphere_downsample_edges(fine, fine_frames, fine.n_nodes + 1, cutoff=0.2)


def test_upsample_edges_row_ptr_is_sized_for_the_fine_grid() -> None:
    _, fine, refinement, fine_frames = _hierarchy()

    edges = build_icosphere_upsample_edges(fine, fine_frames, refinement)

    assert edges.row_ptr.shape == (fine.n_nodes + 1,)
    assert edges.n_edges == 2 * refinement.midpoint_parents.shape[0]


def test_upsample_edges_inherited_nodes_have_no_incident_edges() -> None:
    """The first n_coarse_nodes rows are the "copy" region (IcoDecode's
    docstring) -- nothing should target them here at all; IcoDecode fills
    them in directly from the coarse field, not via this edge set."""
    _, fine, refinement, fine_frames = _hierarchy()

    edges = build_icosphere_upsample_edges(fine, fine_frames, refinement)
    incident_counts = edges.row_ptr[1 : refinement.n_coarse_nodes + 1] - edges.row_ptr[: refinement.n_coarse_nodes]

    assert bool((incident_counts == 0).all())


def test_upsample_edges_every_midpoint_has_exactly_its_two_parents() -> None:
    _, fine, refinement, fine_frames = _hierarchy()
    n_coarse = refinement.n_coarse_nodes

    edges = build_icosphere_upsample_edges(fine, fine_frames, refinement)
    incident_counts = edges.row_ptr[n_coarse + 1 :] - edges.row_ptr[n_coarse:-1]

    assert bool((incident_counts == 2).all())

    for midpoint_offset in range(refinement.midpoint_parents.shape[0]):
        node = n_coarse + midpoint_offset
        start, stop = edges.row_ptr[node].item(), edges.row_ptr[node + 1].item()
        sources = set(edges.source_index[start:stop].tolist())
        assert sources == set(refinement.midpoint_parents[midpoint_offset].tolist())


# --------------------------------------------------------------------------
# IcosphereMultiresolution / build_icosphere_multiresolution
# --------------------------------------------------------------------------


def test_multiresolution_depth_one_matches_direct_pairwise_calls() -> None:
    geometry = build_icosphere_multiresolution(3, depth=1, coarsest_cutoff=0.3, dtype=torch.float64)
    coarse, fine, refinement = build_icosphere_refinement(2, dtype=torch.float64)

    assert geometry.depth == 1
    assert len(geometry.levels) == 2
    assert torch.equal(geometry.finest.nodes, fine.nodes)
    assert torch.equal(geometry.coarsest.nodes, coarse.nodes)
    assert torch.equal(geometry.hops[0].refinement.midpoint_parents, refinement.midpoint_parents)
    assert geometry.hops[0].refinement.coarse_edge_length == refinement.coarse_edge_length


def test_multiresolution_depth_two_levels_are_transitively_prefixed() -> None:
    """levels[2]'s nodes must be an exact prefix of levels[0]'s -- not just
    of levels[1]'s -- since IcosphereRefinement's own prefix guarantee is
    meant to compose across any chain length, not just one hop."""
    geometry = build_icosphere_multiresolution(4, depth=2, coarsest_cutoff=0.5, dtype=torch.float64)

    assert geometry.levels[0].n_nodes > geometry.levels[1].n_nodes > geometry.levels[2].n_nodes
    n2 = geometry.levels[2].n_nodes
    assert torch.allclose(geometry.levels[0].nodes[:n2], geometry.levels[2].nodes, atol=1e-10)


def test_multiresolution_depth_two_hops_match_direct_pairwise_calls() -> None:
    geometry = build_icosphere_multiresolution(4, depth=2, coarsest_cutoff=0.5, dtype=torch.float64)

    _, _, refinement0 = build_icosphere_refinement(3, dtype=torch.float64)
    _, _, refinement1 = build_icosphere_refinement(2, dtype=torch.float64)

    assert torch.equal(geometry.hops[0].refinement.midpoint_parents, refinement0.midpoint_parents)
    assert torch.equal(geometry.hops[1].refinement.midpoint_parents, refinement1.midpoint_parents)


def test_multiresolution_default_cutoffs_match_coarse_edge_length() -> None:
    """Every level below the coarsest defaults to its own hop's
    coarse_edge_length -- the same value build_ushaped_geometry used to
    derive (and reuse for both roles) before IcosphereMultiresolution
    replaced it."""
    geometry = build_icosphere_multiresolution(3, depth=1, coarsest_cutoff=0.3, dtype=torch.float64)

    assert geometry.finest.coverage_radius == pytest.approx(geometry.hops[0].refinement.coarse_edge_length)
    assert geometry.coarsest.coverage_radius == pytest.approx(0.3)


def test_multiresolution_local_cutoffs_override_a_levels_own_discretization_only() -> None:
    """The actual fix this class exists for: overriding a level's own
    discretization cutoff must not change its hop's down_edges, which stay
    pinned to the geometrically-derived coarse_edge_length regardless --
    previously (build_ushaped_geometry) these were silently the same value,
    with nothing structurally keeping them independent."""
    default = build_icosphere_multiresolution(4, depth=2, coarsest_cutoff=0.5, dtype=torch.float64)
    overridden = build_icosphere_multiresolution(
        4, depth=2, coarsest_cutoff=0.5, local_cutoffs={0: 0.05}, dtype=torch.float64
    )

    assert overridden.levels[0].coverage_radius == pytest.approx(0.05)
    assert default.levels[0].coverage_radius != pytest.approx(0.05)
    assert torch.equal(default.hops[0].down_edges.edge_index, overridden.hops[0].down_edges.edge_index)
    assert torch.allclose(default.hops[0].down_edges.distances, overridden.hops[0].down_edges.distances)
    # Untouched levels are unaffected by an override elsewhere in the chain.
    assert default.levels[1].coverage_radius == overridden.levels[1].coverage_radius
    assert default.levels[2].coverage_radius == overridden.levels[2].coverage_radius


def test_multiresolution_rejects_depth_below_one() -> None:
    with pytest.raises(ValueError):
        build_icosphere_multiresolution(3, depth=0, coarsest_cutoff=0.3)


def test_multiresolution_rejects_local_cutoffs_key_outside_range_depth() -> None:
    with pytest.raises(ValueError):
        build_icosphere_multiresolution(4, depth=2, coarsest_cutoff=0.5, local_cutoffs={2: 0.1})

    with pytest.raises(ValueError):
        build_icosphere_multiresolution(4, depth=2, coarsest_cutoff=0.5, local_cutoffs={-1: 0.1})


def test_multiresolution_respects_finest_frames_override() -> None:
    """Mirrors build_ushaped_geometry's own former fine_frames override --
    needed for regauging in an equivariance check (see test_blocks.py)."""
    grid = build_icosphere_grid(3, dtype=torch.float64)
    default_frames = build_tangent_frames(grid.nodes)
    angles = torch.rand(grid.n_nodes, dtype=torch.float64) * 2 * math.pi
    rotated_frames = rotate_tangent_frames(default_frames, angles)

    geometry = build_icosphere_multiresolution(
        3, depth=1, coarsest_cutoff=0.3, finest_frames=rotated_frames, dtype=torch.float64
    )

    assert torch.equal(geometry.finest.frames, rotated_frames)
    assert torch.equal(geometry.coarsest.frames, rotated_frames[: geometry.coarsest.n_nodes])
    assert not torch.allclose(geometry.finest.frames, default_frames)


# --------------------------------------------------------------------------
# median_nearest_neighbor_distance
# --------------------------------------------------------------------------


def test_median_nearest_neighbor_distance_matches_a_known_regular_case() -> None:
    """An icosahedron's own 12 vertices (subdivisions=0) are maximally
    symmetric: every vertex has exactly 5 neighbors, all at the same
    distance, so this is directly hand-verifiable, not just "a plausible
    number." The regular icosahedron's edge central angle is
    arccos(1/sqrt(5)) (a standard closed form)."""
    grid = build_icosphere_grid(subdivisions=0, dtype=torch.float64)
    expected = math.acos(1.0 / math.sqrt(5.0))

    assert median_nearest_neighbor_distance(grid) == pytest.approx(expected, abs=1e-9)


def test_median_nearest_neighbor_distance_halves_with_each_subdivision() -> None:
    """Each icosphere subdivision bisects every edge, so node spacing should
    roughly halve -- loosely enough to be robust to the exact 5-vs-6 valence
    mix, tightly enough to catch a real error (e.g. an off-by-one level)."""
    coarse = median_nearest_neighbor_distance(build_icosphere_grid(subdivisions=3, dtype=torch.float64))
    fine = median_nearest_neighbor_distance(build_icosphere_grid(subdivisions=4, dtype=torch.float64))

    assert 1.8 < coarse / fine < 2.2


def test_median_nearest_neighbor_distance_rejects_a_single_node_grid() -> None:
    grid = SphericalGrid(
        nodes=torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64),
        quadrature_weights=torch.tensor([4 * math.pi], dtype=torch.float64),
    )
    with pytest.raises(ValueError):
        median_nearest_neighbor_distance(grid)
