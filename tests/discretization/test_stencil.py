"""Tests for the coordinate frames and the per-row stencil of a tensor-product grid.

The stencil's claim is that on a Gauss-Legendre grid in the coordinate gauge
the whole edge geometry is a function of ``(ho, hi, dw)``. The builder
checks that itself; the tests here check the builder from the outside, by
comparing every slot against the edges of randomly chosen target nodes of
the full edge set, check the fold check's calibration (a gauge error of one
node just above the tolerance is refused, one below is accepted), check that
the two cases where the claim fails (the default argmin-axis frames, pole
nodes) are refused rather than folded, and that the constructor refuses
tables that do not describe the same slots.
"""

from __future__ import annotations

import math

import pytest
import torch

from gm_attention.discretization.discretization import SphericalDiscretization
from gm_attention.discretization.edge_geometry import EdgeGeometry
from gm_attention.discretization.frames import (
    build_coordinate_frames,
    build_tangent_frames,
    planar_rotation_matrices,
    rotate_tangent_frames,
)
from gm_attention.discretization.grids.torch_harmonics import (
    build_torch_harmonics_grid,
)
from gm_attention.discretization.stencil import (
    FOLD_TOLERANCE,
    TensorGridStencil,
    _longitude_arcs,
    build_tensor_grid_discretization,
    build_tensor_grid_stencil,
    prepare_stencils,
    slot_classes,
)

SHAPE = (12, 24)
CUTOFF = 0.55


def _discretization(dtype: torch.dtype, shape: tuple[int, int] = SHAPE) -> SphericalDiscretization:
    return build_tensor_grid_discretization("legendre-gauss", *shape, coverage_radius=0.8, dtype=dtype)


def _stencil(dtype: torch.dtype = torch.float64, shape: tuple[int, int] = SHAPE, cutoff: float = CUTOFF) -> TensorGridStencil:
    return build_tensor_grid_stencil(_discretization(dtype, shape), cutoff, n_latitude=shape[0], n_longitude=shape[1])


# --------------------------------------------------------------------------
# Coordinate frames


def test_coordinate_frames_are_oriented_orthonormal(dtype: torch.dtype) -> None:
    grid = build_torch_harmonics_grid("legendre-gauss", *SHAPE, dtype=dtype)
    frames = build_coordinate_frames(grid.nodes)
    assert frames.shape == (grid.n_nodes, 3, 2)
    # The discretization's own validator: orthonormal columns, e_1 x e_2 = p.
    SphericalDiscretization._validate_oriented_orthonormal_frames(grid.nodes, frames)


def test_coordinate_frames_are_the_coordinate_tangents() -> None:
    grid = build_torch_harmonics_grid("legendre-gauss", *SHAPE, dtype=torch.float64)
    frames = build_coordinate_frames(grid.nodes)
    nodes = grid.nodes
    theta = torch.arccos(nodes[:, 2])
    phi = torch.atan2(nodes[:, 1], nodes[:, 0])
    e_theta = torch.stack([torch.cos(theta) * torch.cos(phi), torch.cos(theta) * torch.sin(phi), -torch.sin(theta)], -1)
    e_phi = torch.stack([-torch.sin(phi), torch.cos(phi), torch.zeros_like(phi)], -1)
    assert torch.allclose(frames[:, :, 0], e_theta, atol=1e-12)
    assert torch.allclose(frames[:, :, 1], e_phi, atol=1e-12)


def test_coordinate_frames_are_covariant_under_rotation_about_the_axis() -> None:
    """The frame at (h, w) is R_z(phi_w) applied to the frame at (h, 0)."""
    nlat, nlon = SHAPE
    grid = build_torch_harmonics_grid("legendre-gauss", nlat, nlon, dtype=torch.float64)
    frames = build_coordinate_frames(grid.nodes).reshape(nlat, nlon, 3, 2)
    for w in (1, 5, nlon // 2, nlon - 1):
        alpha = torch.tensor(2.0 * math.pi * w / nlon, dtype=torch.float64)
        rotation = torch.eye(3, dtype=torch.float64)
        rotation[:2, :2] = planar_rotation_matrices(alpha)
        rotated = torch.einsum("ab,hbk->hak", rotation, frames[:, 0])  # (nlat, 3, 2)
        assert torch.allclose(rotated, frames[:, w], atol=1e-12)


def test_default_frames_are_not_covariant_under_rotation_about_the_axis() -> None:
    """The argmin-axis frames switch reference axis with longitude: the
    reason the stencil needs the coordinate gauge."""
    nlat, nlon = SHAPE
    grid = build_torch_harmonics_grid("legendre-gauss", nlat, nlon, dtype=torch.float64)
    frames = build_tangent_frames(grid.nodes).reshape(nlat, nlon, 3, 2)
    w = nlon // 4
    alpha = torch.tensor(2.0 * math.pi * w / nlon, dtype=torch.float64)
    rotation = torch.eye(3, dtype=torch.float64)
    rotation[:2, :2] = planar_rotation_matrices(alpha)
    rotated = torch.einsum("ab,hbk->hak", rotation, frames[:, 0])
    assert not torch.allclose(rotated, frames[:, w], atol=1e-6)


def test_coordinate_frames_refuse_pole_nodes() -> None:
    grid = build_torch_harmonics_grid("equiangular", *SHAPE, dtype=torch.float64)  # both poles, nlon copies each
    with pytest.raises(ValueError, match="pole"):
        build_coordinate_frames(grid.nodes)


# --------------------------------------------------------------------------
# The stencil against the full edge set


@pytest.mark.parametrize("shape", [SHAPE, (7, 13), (5, 8)])
def test_stencil_folds_every_target_longitude(dtype: torch.dtype, shape: tuple[int, int]) -> None:
    """Every slot of row ho, compared against the edges of a random target
    (ho, wo) of the full edge set: same source classes, same geometry."""
    nlat, nlon = shape
    discretization = _discretization(dtype, shape)
    stencil = build_tensor_grid_stencil(discretization, CUTOFF, n_latitude=nlat, n_longitude=nlon)
    edges = discretization.edges_within(CUTOFF)
    classes = slot_classes(nlat, nlon, edges)  # (n_edges, 3)
    assert stencil.n_slots * nlon == edges.n_edges
    assert torch.equal(stencil.degrees() * nlon, torch.bincount(edges.target_index, minlength=nlat * nlon).reshape(nlat, nlon).sum(1))

    generator = torch.Generator().manual_seed(0)
    tolerance = FOLD_TOLERANCE[dtype]
    for _ in range(12):
        row = int(torch.randint(nlat, (1,), generator=generator))
        longitude = int(torch.randint(nlon, (1,), generator=generator))
        target = row * nlon + longitude
        first, last = int(edges.row_ptr[target]), int(edges.row_ptr[target + 1])
        edge_classes = classes[first:last]  # (deg, 3), source-minor order: (hi, wi) ascending
        slots = stencil.slots(row)
        slot_classes_ = torch.stack(
            (torch.full_like(stencil.source_row[slots], row), stencil.source_row[slots], stencil.offset[slots]), -1
        )
        # Match on (hi, dw): the edge order is by wi, the slot order by dw, so sort both.
        edge_key = edge_classes[:, 1] * nlon + edge_classes[:, 2]
        slot_key = slot_classes_[:, 1] * nlon + slot_classes_[:, 2]
        edge_order, slot_order = torch.argsort(edge_key), torch.argsort(slot_key)
        assert torch.equal(edge_key[edge_order], slot_key[slot_order])
        for name in ("distances", "log_coordinates", "tangent_quadrature_weights", "angular_phases", "transport_phases"):
            edge_table = getattr(edges, name)[first:last][edge_order]
            slot_table = getattr(stencil.geometry, name)[slots][slot_order]
            difference = (edge_table - slot_table).abs()
            if name == "angular_phases":  # arbitrary at the self-loop (distance 0), on both paths
                difference = difference[edges.distances[first:last][edge_order] > tolerance]
            if difference.numel():  # a row whose only neighbour is itself has no angular phase to compare
                assert float(difference.max()) <= tolerance, name
        # The source nodes the stencil names for this target are the edges' sources.
        sources = stencil.source_nodes(slots)[:, longitude]
        assert torch.equal(torch.sort(sources).values, torch.sort(edges.source_index[first:last]).values)


def test_stencil_representatives_have_target_longitude_zero() -> None:
    nlat, nlon = SHAPE
    stencil = _stencil()
    assert bool((stencil.geometry.target_index % nlon == 0).all())
    assert torch.equal(stencil.geometry.target_index // nlon, torch.repeat_interleave(torch.arange(nlat), stencil.degrees()))
    assert torch.equal(stencil.source_row, stencil.geometry.source_index // nlon)
    assert torch.equal(stencil.offset, stencil.geometry.source_index % nlon)
    # Every row has its self-loop slot: source row ho, offset 0, distance 0.
    for row in range(nlat):
        slots = stencil.slots(row)
        self_loop = (stencil.source_row[slots] == row) & (stencil.offset[slots] == 0)
        assert int(self_loop.sum()) == 1
        assert float(stencil.geometry.distances[slots][self_loop]) < 1e-12


def test_stencil_fold_check_is_calibrated() -> None:
    """One node's frame rotated by a gauge angle just above the tolerance is
    refused, naming the transport phases; one just below is accepted. The
    rotation is a change of gauge, so it breaks nothing but the fold."""
    nlat, nlon = SHAPE
    grid = build_torch_harmonics_grid("legendre-gauss", nlat, nlon, dtype=torch.float64)
    frames = build_coordinate_frames(grid.nodes)
    for angle, accepted in ((1e-11, True), (1e-8, False)):
        angles = torch.zeros(grid.n_nodes, dtype=torch.float64)
        angles[3 * nlon + 5] = angle
        perturbed = SphericalDiscretization.from_grid(grid, coverage_radius=0.8, frames=rotate_tangent_frames(frames, angles))
        if accepted:
            build_tensor_grid_stencil(perturbed, CUTOFF, n_latitude=nlat, n_longitude=nlon)
        else:
            with pytest.raises(ValueError, match="transport_phases"):
                build_tensor_grid_stencil(perturbed, CUTOFF, n_latitude=nlat, n_longitude=nlon)


def test_stencil_refuses_the_default_frames() -> None:
    nlat, nlon = SHAPE
    grid = build_torch_harmonics_grid("legendre-gauss", nlat, nlon, dtype=torch.float64)
    discretization = SphericalDiscretization.from_grid(grid, coverage_radius=0.8)  # build_tangent_frames
    with pytest.raises(ValueError, match="does not fold"):
        build_tensor_grid_stencil(discretization, CUTOFF, n_latitude=nlat, n_longitude=nlon)


def test_stencil_refuses_a_wrong_grid_shape() -> None:
    discretization = _discretization(torch.float64)
    with pytest.raises(ValueError, match="nodes"):
        build_tensor_grid_stencil(discretization, CUTOFF, n_latitude=SHAPE[0] + 1, n_longitude=SHAPE[1])
    with pytest.raises(ValueError, match="not invariant"):  # the right node count, the axes swapped
        build_tensor_grid_stencil(discretization, CUTOFF, n_latitude=SHAPE[1], n_longitude=SHAPE[0])


def test_stencil_refuses_a_non_invariant_edge_set() -> None:
    """Dropping one edge from one longitude breaks the fold, and the builder
    says so before comparing any geometry."""
    nlat, nlon = SHAPE
    discretization = _discretization(torch.float64)
    edges = discretization.edges_within(CUTOFF)
    keep = torch.ones(edges.n_edges, dtype=torch.bool)
    keep[int(edges.row_ptr[3 * nlon + 7])] = False  # the first edge of target (3, 7)
    broken = SphericalDiscretization(
        discretization.grid, discretization.frames, edges.select(keep), coverage_radius=CUTOFF
    )
    with pytest.raises(ValueError, match="not invariant"):
        build_tensor_grid_stencil(broken, CUTOFF, n_latitude=nlat, n_longitude=nlon)


# --------------------------------------------------------------------------
# The arcs


def test_stencil_arcs_are_contiguous_and_cover_the_slots() -> None:
    nlat, nlon = SHAPE
    stencil = _stencil()
    assert stencil.arcs_are_contiguous
    assert int(stencil.arcs[:, 3].sum()) == stencil.n_slots
    covered = set()
    for row_out, row_in, start, length in stencil.arcs.tolist():
        for step in range(length):
            covered.add((row_out, row_in, (start + step) % nlon))
    slots = {(int(r), int(h), int(d)) for r, h, d in zip(
        torch.repeat_interleave(torch.arange(nlat), stencil.degrees()), stencil.source_row, stencil.offset, strict=True
    )}
    assert covered == slots
    # Near the poles a cap contains whole latitude circles: full arcs of length nlon exist.
    assert int((stencil.arcs[:, 3] == nlon).sum()) > 0


@pytest.mark.parametrize(
    ("offsets", "n_longitude", "expected"),
    [
        ([0, 1, 5, 6], 8, [(0, 2), (5, 2)]),  # two runs
        ([0, 2, 4], 8, [(0, 1), (2, 1), (4, 1)]),  # isolated offsets
        ([3], 8, [(3, 1)]),  # a single offset
        ([0, 1, 2, 3, 4, 5, 6, 7], 8, [(0, 8)]),  # the full circle
        ([0, 1, 2, 6, 7], 8, [(6, 5)]),  # one run wrapping through 0
        ([0, 7], 8, [(7, 2)]),  # a two-slot wrap
        ([0, 2], 3, [(2, 2)]),  # wrap on an odd circle
        ([0, 1, 3, 5, 6, 7], 8, [(3, 1), (5, 5)]),  # two runs, one wrapping through 0
        ([0], 1, [(0, 1)]),  # n_longitude = 1
    ],
)
def test_longitude_arcs_are_the_maximal_circular_runs(offsets: list[int], n_longitude: int, expected: list[tuple[int, int]]) -> None:
    """The private run finder on synthetic slots of one (ho, hi) pair,
    beside a second pair with the full circle: every run maximal, every
    slot covered once, the wrap-around run reported from its start."""
    first = torch.tensor(sorted(offsets))
    full = torch.arange(n_longitude)
    row_ptr = torch.tensor([0, first.numel() + full.numel()])
    source_row = torch.cat([torch.zeros_like(first), torch.ones_like(full)])
    offset = torch.cat([first, full])
    arcs = _longitude_arcs(row_ptr, source_row, offset, n_longitude)
    got = [(int(s), int(n)) for r, h, s, n in arcs.tolist() if h == 0 and r == 0]
    assert sorted(got) == sorted(expected)
    assert [(int(s), int(n)) for r, h, s, n in arcs.tolist() if h == 1] == [(0, n_longitude)]
    assert int(arcs[:, 3].sum()) == first.numel() + full.numel()


def test_arcs_are_contiguous_reports_a_split_pair() -> None:
    """A stencil whose slots are a legitimate fold except for a cap that
    meets a latitude circle in two arcs: the property says so."""
    stencil = _stencil()
    nlat, nlon = SHAPE
    # Drop one interior slot of a long arc from the geometry and rebuild the
    # tables; the fold is not re-checked by the constructor, only consistency.
    arcs = stencil.arcs
    long_arc = int(torch.nonzero((arcs[:, 3] >= 5) & (arcs[:, 3] < nlon))[0])
    row_out, row_in, start, length = arcs[long_arc].tolist()
    drop = (start + length // 2) % nlon
    keep = ~(
        (torch.repeat_interleave(torch.arange(nlat), stencil.degrees()) == row_out)
        & (stencil.source_row == row_in) & (stencil.offset == drop)
    )
    geometry = stencil.geometry.select(keep)
    row_ptr = geometry.row_ptr[torch.arange(nlat + 1) * nlon]
    source_row, offset = geometry.source_index // nlon, geometry.source_index % nlon
    split = TensorGridStencil(nlat, nlon, CUTOFF, geometry, row_ptr, source_row, offset,
                              _longitude_arcs(row_ptr, source_row, offset, nlon))
    assert not split.arcs_are_contiguous
    assert split.arcs.shape[0] == arcs.shape[0] + 1


# --------------------------------------------------------------------------
# The object


def test_stencil_source_longitudes_wrap() -> None:
    nlat, nlon = SHAPE
    stencil = _stencil()
    longitudes = stencil.source_longitudes()
    assert longitudes.shape == (stencil.n_slots, nlon)
    expected = (stencil.offset.unsqueeze(-1) + torch.arange(nlon)) % nlon
    assert torch.equal(longitudes, expected)
    assert torch.equal(stencil.source_nodes()[:, 0], stencil.geometry.source_index)


def test_stencil_repeated_geometry_repeats_each_slot_consecutively() -> None:
    stencil = _stencil()
    repeated = stencil.repeated(3)
    assert repeated.n_edges == 3 * stencil.n_slots
    assert torch.equal(repeated.distances, stencil.geometry.distances.repeat_interleave(3))
    assert torch.equal(repeated.transport_phases, stencil.geometry.transport_phases.repeat_interleave(3, dim=0))
    assert torch.equal(repeated.target_index, stencil.geometry.target_index.repeat_interleave(3))
    with pytest.raises(ValueError):
        stencil.repeated(0)


def test_stencil_constructor_refuses_inconsistent_tables() -> None:
    """Each table checked against the geometry it must describe, and the
    self-loop every row relies on."""
    nlat, nlon = SHAPE
    stencil = _stencil()
    geometry = stencil.geometry
    arcs = stencil.arcs

    def build(**overrides: object) -> TensorGridStencil:
        fields = dict(n_latitude=nlat, n_longitude=nlon, cutoff=CUTOFF, geometry=geometry, row_ptr=stencil.row_ptr,
                      source_row=stencil.source_row, offset=stencil.offset, arcs=arcs)
        fields.update(overrides)
        return TensorGridStencil(**fields)  # type: ignore[arg-type]

    build()  # the builder's own tables pass
    with pytest.raises(ValueError, match="row_ptr"):
        tampered = stencil.row_ptr.clone()
        tampered[1] += 1
        build(row_ptr=tampered)
    with pytest.raises(ValueError, match="source_row"):
        build(source_row=stencil.source_row.roll(1))
    with pytest.raises(ValueError, match="offset"):
        build(offset=(stencil.offset + 1) % nlon)
    with pytest.raises(ValueError, match="length"):
        build(arcs=torch.tensor([[0, 0, 0, stencil.n_slots]]))
    with pytest.raises(ValueError, match="cover"):
        build(arcs=arcs[1:])
    with pytest.raises(ValueError, match="beyond the cutoff"):
        build(cutoff=0.1)
    with pytest.raises(ValueError, match="cutoff must lie"):
        build(cutoff=math.pi)
    with pytest.raises(TypeError, match="dtype torch.long"):
        build(offset=stencil.offset.to(torch.int32))
    with pytest.raises(ValueError, match="shape"):
        build(source_row=stencil.source_row[:-1])
    with pytest.raises(ValueError, match="longitude 0"):
        # The full edge set is not a fold: its targets span every longitude.
        full = _discretization(torch.float64).edges_within(CUTOFF)
        build(geometry=full, row_ptr=full.row_ptr[torch.arange(nlat + 1) * nlon],
              source_row=full.source_index // nlon, offset=full.source_index % nlon)
    with pytest.raises(ValueError, match="self-loop"):
        keep = ~((stencil.source_row == 2) & (stencil.offset == 0)
                 & (torch.repeat_interleave(torch.arange(nlat), stencil.degrees()) == 2))
        without = geometry.select(keep)
        row_ptr = without.row_ptr[torch.arange(nlat + 1) * nlon]
        source_row, offset = without.source_index // nlon, without.source_index % nlon
        build(geometry=without, row_ptr=row_ptr, source_row=source_row, offset=offset,
              arcs=_longitude_arcs(row_ptr, source_row, offset, nlon))


def test_stencil_to_moves_every_table() -> None:
    stencil = _stencil()
    moved = stencil.to(dtype=torch.float32, non_blocking=True)
    assert isinstance(moved, TensorGridStencil)
    assert moved.dtype == torch.float32
    assert torch.equal(moved.row_ptr, stencil.row_ptr)
    assert torch.allclose(moved.geometry.distances.double(), stencil.geometry.distances, atol=1e-6)
    assert "neighbours=" in repr(moved)


def test_prepare_stencils_keys_like_the_discretization_cache() -> None:
    nlat, nlon = SHAPE
    discretization = _discretization(torch.float64)
    stencils = prepare_stencils(discretization, [CUTOFF, 0.3], n_latitude=nlat, n_longitude=nlon)
    assert set(stencils) == {SphericalDiscretization._radius_key(CUTOFF), SphericalDiscretization._radius_key(0.3)}
    assert stencils[SphericalDiscretization._radius_key(0.3)].n_slots < stencils[SphericalDiscretization._radius_key(CUTOFF)].n_slots
    with pytest.raises(ValueError):  # edges_within refuses a radius of pi
        prepare_stencils(discretization, [math.pi], n_latitude=nlat, n_longitude=nlon)


def test_stencil_degree_is_constant_along_each_row_and_largest_at_the_poles() -> None:
    nlat, nlon = (16, 32)
    stencil = _stencil(shape=(nlat, nlon), cutoff=0.45)
    degrees = stencil.degrees()
    assert int(degrees.argmax()) in (0, nlat - 1)
    assert int(degrees[0]) == int(degrees[-1])  # the Gauss-Legendre rows are symmetric about the equator
    assert torch.equal(degrees, degrees.flip(0))


def test_stencil_geometry_is_an_edge_geometry_of_the_slots() -> None:
    stencil = _stencil()
    assert isinstance(stencil.geometry, EdgeGeometry)
    assert stencil.geometry.row_ptr.shape == (stencil.n_nodes + 1,)
