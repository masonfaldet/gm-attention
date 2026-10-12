"""Tests for the tensor-grid chain (gm_attention.discretization.grids.equiangular).

The supplement ships :func:`build_grid_hierarchy`, which takes prebuilt
grids; the development repository's ``build_tensor_grid_hierarchy``
convenience (shapes and quadrature kinds in) is not shipped. ``_build``
below is that convenience inlined: it builds one ``torch_harmonics`` grid
per shape and hands everything to the shipped builder, which is also how
the released SegFormer's geometry assembles its chain.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest
import torch
from torch import Tensor

from gm_attention.discretization.frames import rotate_tangent_frames
from gm_attention.discretization.grids.equiangular import (
    TensorGridHierarchy,
    build_grid_hierarchy,
)
from gm_attention.discretization.grids.torch_harmonics import build_torch_harmonics_grid

SHAPES = [(16, 32), (8, 16), (4, 8)]
KINDS = ["equiangular", "equiangular", "equiangular"]
KW = {
    "restrict_cutoffs": [0.5, 1.0],
    "interpolate_cutoffs": [math.pi / 7, math.pi / 3],
    "local_cutoffs": [None, [0.9, 0.5], [1.6, 1.0]],
    "dtype": torch.float64,
}


def _build(
    shapes: Sequence[tuple[int, int]],
    grid_kinds: Sequence[str],
    *,
    dtype: torch.dtype = torch.float64,
    frames: Sequence[Tensor | None] | None = None,
    **cutoffs,
) -> TensorGridHierarchy:
    """A hierarchy over ``torch_harmonics`` grids of ``shapes``, one per kind.

    The zip is deliberately not strict: a kinds/shapes length mismatch is
    left for :func:`build_grid_hierarchy`'s own per-level check to report.
    """
    grids = [
        build_torch_harmonics_grid(kind, nlat, nlon, dtype=dtype)
        for (nlat, nlon), kind in zip(shapes, grid_kinds)
    ]
    return build_grid_hierarchy(
        grids, list(grid_kinds), [tuple(int(v) for v in shape) for shape in shapes],
        frames=frames, **cutoffs,
    )


def test_levels_transfers_and_windows_have_the_shapes_of_the_chain() -> None:
    hierarchy = _build(SHAPES, KINDS, **KW)

    assert hierarchy.depth == 2
    assert [level.n_nodes for level in hierarchy.levels] == [512, 128, 32]
    assert hierarchy.finest.discretization is None
    for level, radii in zip(hierarchy.levels[1:], KW["local_cutoffs"][1:], strict=True):
        assert level.discretization is not None
        assert level.discretization.coverage_radius == pytest.approx(max(radii))
        assert set(level.discretization.cached_radii) >= {float(r) for r in radii}
    for hop in range(hierarchy.depth):
        down, up = hierarchy.down_edges[hop], hierarchy.up_edges[hop]
        assert down.row_ptr.shape[0] == hierarchy.levels[hop + 1].n_nodes + 1
        assert int(down.source_index.max()) < hierarchy.levels[hop].n_nodes
        assert up.row_ptr.shape[0] == hierarchy.finest.n_nodes + 1
        assert int(up.source_index.max()) < hierarchy.levels[hop + 1].n_nodes


def test_every_target_has_a_source_on_both_transfer_directions() -> None:
    hierarchy = _build(SHAPES, KINDS, **KW)
    for hop in range(hierarchy.depth):
        down = torch.bincount(hierarchy.down_edges[hop].target_index, minlength=hierarchy.levels[hop + 1].n_nodes)
        up = torch.bincount(hierarchy.up_edges[hop].target_index, minlength=hierarchy.finest.n_nodes)
        assert int(down.min()) >= 1
        assert int(up.min()) >= 1


def test_frame_overrides_are_honoured_per_level() -> None:
    base = _build(SHAPES, KINDS, **KW)
    generator = torch.Generator().manual_seed(0)
    frames = [
        rotate_tangent_frames(level.frames, torch.rand(level.n_nodes, dtype=torch.float64, generator=generator) * 2 * math.pi)
        for level in base.levels
    ]
    regauged = _build(SHAPES, KINDS, frames=frames, **KW)
    for level, override in zip(regauged.levels, frames, strict=True):
        assert torch.equal(level.frames, override)
    # Same edges, different gauge: the transport angles must have changed.
    assert torch.equal(base.down_edges[0].edge_index, regauged.down_edges[0].edge_index)
    assert not torch.allclose(base.down_edges[0].transport_angles, regauged.down_edges[0].transport_angles)


def test_rejects_inconsistent_arguments() -> None:
    # The shipped builder measures "coarser" in nodes (the dropped shape
    # convenience also demanded both axes shrink), so the offending level
    # here has as many nodes as its predecessor.
    with pytest.raises(ValueError, match="strictly coarser"):
        _build([(16, 32), (16, 32)], KINDS[:2], restrict_cutoffs=[0.5], interpolate_cutoffs=[0.5], local_cutoffs=[None, [0.5]])
    with pytest.raises(ValueError, match="one entry per hop"):
        _build(SHAPES, KINDS, restrict_cutoffs=[0.5], interpolate_cutoffs=[0.5, 0.5], local_cutoffs=KW["local_cutoffs"])
    with pytest.raises(ValueError, match="one entry per level"):
        _build(SHAPES, KINDS[:2], restrict_cutoffs=[0.5, 1.0], interpolate_cutoffs=[0.5, 0.5], local_cutoffs=KW["local_cutoffs"])
    with pytest.raises(ValueError, match="at least two levels"):
        _build([(16, 32)], ["equiangular"], restrict_cutoffs=[], interpolate_cutoffs=[], local_cutoffs=[None])
    with pytest.raises(ValueError, match="empty"):
        _build(SHAPES, KINDS, restrict_cutoffs=[0.5, 1.0], interpolate_cutoffs=[0.5, 1.1], local_cutoffs=[None, [], [1.0]])


def test_to_moves_every_tensor() -> None:
    hierarchy = _build(SHAPES, KINDS, **KW).to("cpu")
    assert hierarchy.levels[1].discretization is not None
    assert hierarchy.levels[1].discretization.device.type == "cpu"
    assert hierarchy.up_edges[0].distances.device.type == "cpu"
