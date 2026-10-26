"""The released spherical SegFormer's grid chain and receptive fields, as numbers.

``torch_harmonics``' ``SphericalSegformer`` (the ``s2nsegformer_sc2_layers4_e128``
registry entry, the S2 SegFormer of "Attention on the Sphere") reads an
equiangular ``128 x 256`` image and halves both axes at every stage,
``64 x 128 -> 32 x 64 -> 16 x 32 -> 8 x 16``, every internal grid
equiangular. At a stage of ``nlat`` rows its operators have the radii of the
pinned implementation:

* the overlap patch merging DISCO convolution and the Mix-FFN's depthwise
  DISCO convolution: ``(kernel_latitude + 1) * 0.5 * pi / (nlat - 1)`` with
  the registry's piecewise-linear ``kernel_shape=(5, 4)``, i.e. ``3 pi / (nlat - 1)``;
* the neighborhood attention: ``7 sqrt(pi) / (nlat - 1)``.

The gauge-equivariant model runs on the same chain, every operator at the
radius of the reference operator in the same role. The interpolation back to
full resolution (bilinear in the reference, with no radius) uses the coarse
grid's row spacing, the smallest radius that reaches every fine node, pole
rows included, from at least one coarse node.
"""

from __future__ import annotations

import dataclasses

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from torch_harmonics.quadrature import _precompute_latitudes

from experiments.shallow_water.cutoffs import s2_attention_cutoff, s2_disco_cutoff
from gm_attention.discretization.grid import (
    SphericalGrid,
    median_nearest_neighbor_distance,
)
from gm_attention.discretization.grids.equiangular import (
    TensorGridHierarchy,
    build_grid_hierarchy,
)
from gm_attention.discretization.frames import build_coordinate_frames
from gm_attention.discretization.grids.icosphere import build_icosphere_grid
from gm_attention.discretization.grids.torch_harmonics import build_torch_harmonics_grid
from gm_attention.nn.stencil_plan import build_stencil_plans

#: Latitude taps of the registry's ``kernel_shape=(5, 4)``: DISCO support ``(5 + 1) * 0.5 * pi / (nlat - 1)``.
SEGFORMER_KERNEL_LATITUDE = 5


def segformer_stage_shapes(
    img_size: tuple[int, int], scale_factor: int, num_stages: int
) -> tuple[tuple[int, int], ...]:
    """The internal grid of every stage, by the released constructor's own
    arithmetic: both axes floor-divided by ``scale_factor`` per stage."""
    if scale_factor < 2:
        raise ValueError(f"scale_factor must be at least 2, got {scale_factor}.")
    if num_stages < 1:
        raise ValueError(f"num_stages must be at least 1, got {num_stages}.")
    shapes: list[tuple[int, int]] = []
    nlat, nlon = int(img_size[0]), int(img_size[1])
    for _ in range(num_stages):
        nlat, nlon = nlat // scale_factor, nlon // scale_factor
        if nlat < 2 or nlon < 2:
            raise ValueError(
                f"img_size {tuple(img_size)} cannot be halved {num_stages} times."
            )
        shapes.append((nlat, nlon))
    return tuple(shapes)


def row_spacing(nlat: int, grid_kind: str) -> float:
    """The largest colatitude gap a grid leaves anywhere, poles included: the
    maximum of the row spacing and the gaps from either pole to its nearest
    row, which on a Legendre-Gauss grid (no row at the poles) can exceed the
    row spacing.
    """
    colatitudes, _ = _precompute_latitudes(nlat=nlat, grid=grid_kind)
    theta = torch.as_tensor(colatitudes, dtype=torch.float64)
    gaps = [float(theta[0]), float(math.pi - theta[-1])]
    if nlat > 1:
        gaps.append(float(torch.diff(theta).max()))
    return max(gaps)


#: Icosphere level per stage; node counts 10242/2562/642/162 against the released stages' 8192/2048/512/128 at 128x256.
DEFAULT_ICOSPHERE_LEVELS: tuple[int, ...] = (5, 4, 3, 2)


def icosphere_spacing(level: int) -> float:
    """The median nearest-neighbour geodesic distance of an icosphere level."""
    return median_nearest_neighbor_distance(build_icosphere_grid(level, dtype=torch.float64))


def stage_spacings(
    stage_shapes: Sequence[tuple[int, int]],
    *,
    grid_internal: str,
    icosphere_levels: Sequence[int] | None = None,
) -> tuple[float, ...]:
    """The typical node spacing of every stage grid: the row spacing of a
    tensor grid, the edge length of an icosphere level."""
    if grid_internal == "icosphere":
        levels = DEFAULT_ICOSPHERE_LEVELS if icosphere_levels is None else tuple(icosphere_levels)
        if len(levels) != len(stage_shapes):
            raise ValueError(
                f"icosphere_levels needs one entry per stage ({len(stage_shapes)}), got {len(levels)}."
            )
        return tuple(icosphere_spacing(int(level)) for level in levels)
    return tuple(row_spacing(nlat, grid_internal) for nlat, _ in stage_shapes)


@dataclass(frozen=True)
class SegformerRadii:
    """One radius per stage for each cross-node operator: the patch merging,
    the attention, the depthwise mix, the interpolation back to the input
    grid, and (``None`` for a pointwise readout) the convolution that turns
    a stage's features into scalars on the stage's own grid."""

    merge: tuple[float, ...]
    attention: tuple[float, ...]
    mix: tuple[float, ...]
    interpolate: tuple[float, ...]
    readout: tuple[float, ...] | None = None
    #: Radius of a stem convolution on the input grid (``stem_rows`` input rows), or ``None`` for no stem.
    stem: float | None = None

    @property
    def num_stages(self) -> int:
        return len(self.merge)


def segformer_radii(
    stage_shapes: Sequence[tuple[int, int]],
    *,
    grid_internal: str = "equiangular",
    icosphere_levels: Sequence[int] | None = None,
    kernel_latitude: int = SEGFORMER_KERNEL_LATITUDE,
    interpolate_spacings: float = 1.0,
    readout_rows: float | None = None,
    img_size: tuple[int, int] | None = None,
    grid: str = "equiangular",
    stem_rows: float | None = None,
) -> SegformerRadii:
    """The reference's supports at every stage (functions of the released
    stage shapes, whatever grid the stage runs on), the interpolation radius
    as ``interpolate_spacings`` stage spacings, and, when ``readout_rows`` is
    given, a readout radius of that many stage spacings."""
    merge = tuple(s2_disco_cutoff(nlat, kernel_latitude=kernel_latitude) for nlat, _ in stage_shapes)
    attention = tuple(s2_attention_cutoff(nlat) for nlat, _ in stage_shapes)
    spacings = stage_spacings(stage_shapes, grid_internal=grid_internal, icosphere_levels=icosphere_levels)
    if not interpolate_spacings > 0:
        raise ValueError(f"interpolate_spacings must be positive, got {interpolate_spacings}.")
    interpolate = tuple(float(interpolate_spacings) * spacing for spacing in spacings)
    readout: tuple[float, ...] | None = None
    if readout_rows is not None:
        if not readout_rows > 0:
            raise ValueError(f"readout_rows must be positive, got {readout_rows}.")
        readout = tuple(float(readout_rows) * spacing for spacing in spacings)
    stem: float | None = None
    if stem_rows is not None:
        if img_size is None:
            raise ValueError("stem_rows needs img_size, the input grid whose rows it counts.")
        if not stem_rows > 0:
            raise ValueError(f"stem_rows must be positive, got {stem_rows}.")
        stem = float(stem_rows) * row_spacing(int(img_size[0]), grid)
    return SegformerRadii(
        merge=merge, attention=attention, mix=merge, interpolate=interpolate, readout=readout, stem=stem
    )


def build_segformer_hierarchy(
    img_size: tuple[int, int],
    stage_shapes: Sequence[tuple[int, int]],
    radii: SegformerRadii,
    *,
    grid: str = "equiangular",
    grid_internal: str = "equiangular",
    icosphere_levels: Sequence[int] | None = None,
    frames: Sequence[Tensor | None] | None = None,
    lateral: bool = False,
    stencil_plans: bool = False,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> TensorGridHierarchy:
    """The chain ``img_size -> stage 1 -> ... -> stage N`` with every transfer
    and same-grid geometry the gauge-equivariant SegFormer consumes.

    The stages run on tensor grids of ``stage_shapes`` (the released chain)
    or, with ``grid_internal="icosphere"``, on the icosphere levels
    ``icosphere_levels``; the input level is the tensor grid either way,
    the grid the released loss and metrics read. Every transfer is a
    radius-based bipartite edge set at the reference's radius, integrating
    with the source grid's own quadrature (the equiangular weights, pole
    copies included, for the first merge). Level 0 carries a same-grid
    discretization only when ``radii.stem`` is set; otherwise its only
    operators are the first patch merging (source) and the interpolations
    (target). Each stage level prepares its attention and mix radii, and
    its readout radius when the readout convolves. ``lateral=True`` adds
    the top-down transfers of a fusing decoder (``hierarchy.lateral_edges``).
    ``stencil_plans=True`` gives every ``"legendre-gauss"`` level that
    carries a discretization the stencil plans of its radii
    (``TensorGridLevel.plans``), on which the model runs the level's blocks
    and readout on the dense fixed-grid layout; such levels then take the
    coordinate frames unless ``frames`` names theirs, since the stencil
    needs the frames covariant under the grid's rotations. Levels of other
    kinds (the equiangular input grid with its poles, icosphere stages) keep
    the graph path; at least one level must take plans.
    """
    if radii.num_stages != len(stage_shapes):
        raise ValueError(
            f"radii describe {radii.num_stages} stages, stage_shapes {len(stage_shapes)}."
        )
    fine = build_torch_harmonics_grid(grid, int(img_size[0]), int(img_size[1]), dtype=dtype, device=device)
    grids: list[SphericalGrid] = [fine]
    shapes: list[tuple[int, ...]] = [(int(img_size[0]), int(img_size[1]))]
    if grid_internal == "icosphere":
        levels = DEFAULT_ICOSPHERE_LEVELS if icosphere_levels is None else tuple(int(v) for v in icosphere_levels)
        if len(levels) != len(stage_shapes):
            raise ValueError(
                f"icosphere_levels needs one entry per stage ({len(stage_shapes)}), got {len(levels)}."
            )
        for level in levels:
            grids.append(build_icosphere_grid(level, dtype=dtype, device=device))
            shapes.append((level,))
    else:
        for nlat, nlon in stage_shapes:
            grids.append(build_torch_harmonics_grid(grid_internal, int(nlat), int(nlon), dtype=dtype, device=device))
            shapes.append((int(nlat), int(nlon)))
    kinds = [grid, *([grid_internal] * len(stage_shapes))]
    # Level 0 carries a same-grid discretization only for a stem convolution.
    local: list[Sequence[float] | None] = [None if radii.stem is None else [radii.stem]]
    readouts = radii.readout if radii.readout is not None else [None] * radii.num_stages
    for attention, mix, readout in zip(radii.attention, radii.mix, readouts, strict=True):
        local.append([attention, mix] if readout is None else [attention, mix, readout])
    foldable = [kind == "legendre-gauss" and cutoffs is not None for kind, cutoffs in zip(kinds, local, strict=True)]
    if stencil_plans:
        if not any(foldable):
            raise ValueError(
                "stencil_plans needs a 'legendre-gauss' level with operators on it; the stages are "
                f"{grid_internal!r} (pole nodes have no coordinate frame, icospheres no stencil)."
            )
        if frames is None:
            # The coordinate gauge, on which the stencil folds; the other levels keep the default frames.
            frames = [
                build_coordinate_frames(g.nodes) if kind == "legendre-gauss" else None
                for g, kind in zip(grids, kinds, strict=True)
            ]
    hierarchy = build_grid_hierarchy(
        grids, kinds, shapes,
        restrict_cutoffs=list(radii.merge),
        interpolate_cutoffs=list(radii.interpolate),
        local_cutoffs=local,
        frames=frames,
        # Top-down transfers: stage k + 1 onto stage k at stage k + 1's own
        # interpolation radius (a source-spacing radius covers any finer target).
        lateral_cutoffs=list(radii.interpolate[1:]) if lateral else None,
    )
    if not stencil_plans:
        return hierarchy
    levels = []
    for level, cutoffs, takes_plans in zip(hierarchy.levels, local, foldable, strict=True):
        if not takes_plans:
            levels.append(level)
            continue
        assert level.discretization is not None and cutoffs is not None
        nlat, nlon = level.shape
        plans = build_stencil_plans(level.discretization, cutoffs, n_latitude=int(nlat), n_longitude=int(nlon))
        levels.append(dataclasses.replace(level, plans=plans))
    return dataclasses.replace(hierarchy, levels=tuple(levels))
