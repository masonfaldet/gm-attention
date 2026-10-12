"""The grid chain and radii are the released SegFormer's, not a paraphrase."""

from __future__ import annotations

import math

import pytest
import torch

from experiments.stanford_2d3ds.geometry import (
    SEGFORMER_KERNEL_LATITUDE,
    build_segformer_hierarchy,
    row_spacing,
    segformer_radii,
    segformer_stage_shapes,
)


def test_stage_shapes_follow_the_released_constructors_arithmetic() -> None:
    from torch_harmonics.examples.models import SphericalSegformer

    reference = SphericalSegformer(
        img_size=(128, 256), in_chans=3, out_chans=14, embed_dims=[4, 4, 4, 4], heads=[1, 1, 1, 1],
        depths=[1, 1, 1, 1], scale_factor=2, kernel_shape=(5, 4), filter_basis_type="piecewise linear",
        grid_internal="equiangular", attention_mode="neighborhood",
    )
    assert segformer_stage_shapes((128, 256), 2, 4) == tuple(block.out_shape for block in reference.blocks)
    assert segformer_stage_shapes((128, 256), 2, 4) == ((64, 128), (32, 64), (16, 32), (8, 16))
    with pytest.raises(ValueError):
        segformer_stage_shapes((16, 32), 2, 5)


def test_radii_transcribe_the_released_cutoffs() -> None:
    from torch_harmonics.examples.models.s2segformer import _compute_cutoff_radius

    shapes = segformer_stage_shapes((128, 256), 2, 4)
    radii = segformer_radii(shapes)
    for (nlat, _), merge, attention, mix in zip(shapes, radii.merge, radii.attention, radii.mix, strict=True):
        assert merge == pytest.approx(_compute_cutoff_radius(nlat, (SEGFORMER_KERNEL_LATITUDE, 4), "piecewise linear"))
        assert mix == merge
        # AttentionWrapper's default: (7 / sqrt(pi)) * pi / (nlat - 1).
        assert attention == pytest.approx((7.0 / math.sqrt(math.pi)) * math.pi / (nlat - 1))
    assert radii.interpolate == pytest.approx(tuple(math.pi / (nlat - 1) for nlat, _ in shapes))
    assert radii.readout is None
    with_readout = segformer_radii(shapes, readout_rows=1.0)
    assert with_readout.readout == pytest.approx(tuple(math.pi / (nlat - 1) for nlat, _ in shapes))
    with pytest.raises(ValueError, match="readout_rows"):
        segformer_radii(shapes, readout_rows=0.0)


def test_row_spacing_is_the_widest_gap_poles_included() -> None:
    from torch_harmonics.quadrature import _precompute_latitudes

    assert row_spacing(8, "equiangular") == pytest.approx(math.pi / 7)
    theta, _ = _precompute_latitudes(nlat=8, grid="legendre-gauss")
    theta = torch.as_tensor(theta, dtype=torch.float64)
    expected = max(float(theta[0]), float(torch.diff(theta).max()), float(math.pi - theta[-1]))
    assert row_spacing(8, "legendre-gauss") == pytest.approx(expected)


def test_the_full_size_chain_reaches_every_node_and_matches_the_counts_on_record() -> None:
    """The counts the study's design was sized by; a change here changes
    the memory and compute story, so it should be deliberate."""
    shapes = segformer_stage_shapes((128, 256), 2, 4)
    radii = segformer_radii(shapes, readout_rows=1.0)
    hierarchy = build_segformer_hierarchy((128, 256), shapes, radii, dtype=torch.float32)

    assert [level.n_nodes for level in hierarchy.levels] == [32768, 8192, 2048, 512, 128]
    assert hierarchy.finest.discretization is None
    readout = [
        level.discretization.edges_within(radius).n_edges
        for level, radius in zip(hierarchy.levels[1:], radii.readout, strict=True)
        if level.discretization is not None
    ]
    assert readout == [161792, 38400, 8960, 2048]  # boundary shells whole (cutoff tolerance)
    for hop in range(hierarchy.depth):
        up = torch.bincount(hierarchy.up_edges[hop].target_index, minlength=hierarchy.finest.n_nodes)
        down = torch.bincount(hierarchy.down_edges[hop].target_index, minlength=hierarchy.levels[hop + 1].n_nodes)
        assert int(up.min()) >= 1 and int(down.min()) >= 1
    attention = [
        level.discretization.edges_within(radius).n_edges
        for level, radius in zip(hierarchy.levels[1:], segformer_radii(shapes).attention, strict=True)
        if level.discretization is not None
    ]
    assert attention == [1152512, 244224, 49664, 9088]
    # Decided from the float64 lattice: the 512 stage-1 merge pairs at r + 8.2e-6 are out.
    assert [e.n_edges for e in hierarchy.down_edges] == [3042816, 659968, 138368, 27136]
    assert [e.n_edges for e in hierarchy.up_edges] == [542208, 448256, 375680, 336960]


def _shift_columns(index: torch.Tensor, shape: tuple[int, int], columns: int) -> torch.Tensor:
    nlon = shape[1]
    return (index // nlon) * nlon + (index % nlon + columns) % nlon


def _edge_keys(target: torch.Tensor, source: torch.Tensor) -> set[int]:
    return set((target.to(torch.int64) * (1 << 32) + source.to(torch.int64)).tolist())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_the_full_size_chain_is_symmetric_under_a_sixteen_column_shift(dtype: torch.dtype) -> None:
    """Sixteen input columns is one column of the coarsest stage, the
    rotation every level shares. Every edge set of the chain -- attention,
    mix and readout radii on each stage, the merges down and the
    interpolations up -- must map onto itself under it, in either working
    dtype; with a strict radius comparison the shells exactly on a radius
    did not."""
    img = (128, 256)
    shapes = segformer_stage_shapes(img, 2, 4)
    radii = segformer_radii(shapes, readout_rows=1.0)
    hierarchy = build_segformer_hierarchy(img, shapes, radii, dtype=dtype)
    columns = 16
    for i, (level, shape) in enumerate(zip(hierarchy.levels[1:], shapes, strict=True)):
        assert level.discretization is not None
        k = columns * shape[1] // img[1]
        for radius in (radii.attention[i], radii.mix[i], radii.readout[i]):
            edges = level.discretization.edges_within(radius)
            t, s = edges.edge_index[0], edges.edge_index[1]
            assert _edge_keys(_shift_columns(t, shape, k), _shift_columns(s, shape, k)) == _edge_keys(t, s)
    for hop in range(hierarchy.depth):
        source_shape = img if hop == 0 else shapes[hop - 1]
        down = hierarchy.down_edges[hop]
        t, s = down.edge_index[0], down.edge_index[1]
        moved = _edge_keys(
            _shift_columns(t, shapes[hop], columns * shapes[hop][1] // img[1]),
            _shift_columns(s, source_shape, columns * source_shape[1] // img[1]),
        )
        assert moved == _edge_keys(t, s)
        up = hierarchy.up_edges[hop]
        t, s = up.edge_index[0], up.edge_index[1]
        moved = _edge_keys(
            _shift_columns(t, img, columns), _shift_columns(s, shapes[hop], columns * shapes[hop][1] // img[1])
        )
        assert moved == _edge_keys(t, s)


def test_icosphere_stages_reach_every_node_and_match_the_counts_on_record() -> None:
    """The native-geometry arm's chain: equiangular 128x256 in and out, icosphere
    levels 5, 4, 3, 2 for the stages, the reference's radii throughout."""
    from experiments.stanford_2d3ds.geometry import (
        DEFAULT_ICOSPHERE_LEVELS,
        icosphere_spacing,
    )

    shapes = segformer_stage_shapes((128, 256), 2, 4)
    radii = segformer_radii(shapes, grid_internal="icosphere", interpolate_spacings=1.5, readout_rows=1.0)
    equiangular = segformer_radii(shapes, readout_rows=1.0)
    assert radii.merge == equiangular.merge and radii.attention == equiangular.attention
    assert radii.interpolate == pytest.approx(tuple(1.5 * icosphere_spacing(level) for level in DEFAULT_ICOSPHERE_LEVELS))

    hierarchy = build_segformer_hierarchy((128, 256), shapes, radii, grid_internal="icosphere", dtype=torch.float32)
    assert [level.n_nodes for level in hierarchy.levels] == [32768, 10242, 2562, 642, 162]
    assert [level.describe() for level in hierarchy.levels[:2]] == ["128x256 equiangular", "icosphere L5"]
    for hop in range(hierarchy.depth):
        up = torch.bincount(hierarchy.up_edges[hop].target_index, minlength=hierarchy.finest.n_nodes)
        down = torch.bincount(hierarchy.down_edges[hop].target_index, minlength=hierarchy.levels[hop + 1].n_nodes)
        assert int(up.min()) >= 1 and int(down.min()) >= 1
    attention = [
        level.discretization.edges_within(radius).n_edges
        for level, radius in zip(hierarchy.levels[1:], radii.attention, strict=True)
        if level.discretization is not None
    ]
    assert attention == [991182, 259062, 68142, 15462]
    assert [e.n_edges for e in hierarchy.down_edges] == [1872340, 603642, 158322, 40842]
    assert [e.n_edges for e in hierarchy.up_edges] == [229352, 229944, 229616, 225432]
    with pytest.raises(ValueError, match="icosphere_levels"):
        segformer_radii(shapes, grid_internal="icosphere", icosphere_levels=[5, 4])


def test_lateral_edges_and_the_stem_discretization_cover_their_targets_and_share_the_symmetry() -> None:
    """With ``lateral=True`` the chain carries one top-down transfer per hop
    after the first, stage k + 1 onto stage k at stage k + 1's interpolation
    radius; with ``stem_rows`` the input level carries a same-grid
    discretization at that many input rows. Every target is covered, and
    every new edge set maps onto itself under the sixteen-column shift."""
    img = (128, 256)
    shapes = segformer_stage_shapes(img, 2, 4)
    radii = segformer_radii(shapes, readout_rows=1.0, img_size=img, stem_rows=1.0)
    assert radii.stem == pytest.approx(math.pi / 127)
    hierarchy = build_segformer_hierarchy(img, shapes, radii, lateral=True, dtype=torch.float32)
    assert len(hierarchy.lateral_edges) == hierarchy.depth - 1
    columns = 16
    for k, lateral in enumerate(hierarchy.lateral_edges):
        target_shape, source_shape = shapes[k], shapes[k + 1]
        covered = torch.bincount(lateral.target_index, minlength=target_shape[0] * target_shape[1])
        assert int(covered.min()) >= 1
        t, s = lateral.edge_index[0], lateral.edge_index[1]
        moved = _edge_keys(
            _shift_columns(t, target_shape, columns * target_shape[1] // img[1]),
            _shift_columns(s, source_shape, columns * source_shape[1] // img[1]),
        )
        assert moved == _edge_keys(t, s)
    finest = hierarchy.finest
    assert finest.discretization is not None
    stem = finest.discretization.edges_within(radii.stem)
    degrees = torch.bincount(stem.target_index, minlength=finest.n_nodes)
    assert int(degrees.min()) >= 3  # a node, its two row neighbours at least; the pole rows more
    t, s = stem.edge_index[0], stem.edge_index[1]
    assert _edge_keys(_shift_columns(t, img, columns), _shift_columns(s, img, columns)) == _edge_keys(t, s)
    assert build_segformer_hierarchy(img, shapes, segformer_radii(shapes, readout_rows=1.0), dtype=torch.float32).lateral_edges == ()
