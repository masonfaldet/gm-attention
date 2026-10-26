"""Grid chains over independent spherical grids.

A spherical SegFormer runs on a chain of grids, an equiangular input level
and coarser stage levels (equiangular or icosphere), precomputed by
:func:`build_grid_hierarchy` into a :class:`TensorGridHierarchy`. A coarse
equiangular grid is not a prefix of the fine one, so every transfer is a
radius-based bipartite edge set built by
:func:`~gm_attention.discretization.edge_geometry.compute_bipartite_edge_geometry`.
A radius rather than a fixed ``k``: a ``torch_harmonics`` equiangular grid
repeats each pole across every longitude and splits the pole's quadrature
weight among the copies, and a radius includes every colocated copy, so the
split weights sum back to the correct mass.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from gm_attention.discretization.discretization import SphericalDiscretization
from gm_attention.discretization.edge_geometry import (
    DEFAULT_MAX_FREQUENCY as DEFAULT_EDGE_MAX_FREQUENCY,
)
from gm_attention.discretization.edge_geometry import (
    EdgeGeometry,
    compute_bipartite_edge_geometry,
)
from gm_attention.discretization.frames import build_tangent_frames
from gm_attention.discretization.grid import SphericalGrid

if TYPE_CHECKING:  # the stencil plans live in the layer package, which imports this one
    from gm_attention.nn.stencil_plan import StencilPlans


@dataclass(frozen=True, slots=True)
class TensorGridLevel:
    """One level of a :class:`TensorGridHierarchy`.

    ``discretization`` is the same-grid geometry for the operators that run
    on this level (attention, depthwise convolution), or ``None`` for a
    level nothing runs on, such as the input/output level of a SegFormer
    chain, which only sources the first restriction and receives every
    interpolation. ``plans`` optionally holds the same operators' stencil
    plans (:class:`~gm_attention.nn.stencil_plan.StencilPlans`, one per
    radius of the discretization) for a tensor grid in the coordinate gauge,
    on which a model evaluates them on the dense fixed-grid layout instead;
    a level without plans runs on the graph path.
    """

    #: ``(nlat, nlon)`` for a tensor-product grid; ``(subdivisions,)`` for an icosphere.
    shape: tuple[int, ...]
    grid_kind: str
    grid: SphericalGrid
    frames: Tensor  # (n_nodes, 3, 2)
    discretization: SphericalDiscretization | None
    plans: StencilPlans | None = None

    @property
    def local(self) -> SphericalDiscretization | StencilPlans | None:
        """What a layer on this level aggregates over: the plans when the
        level has them, else the discretization; what a model hands to
        its blocks."""
        return self.plans if self.plans is not None else self.discretization

    @property
    def n_nodes(self) -> int:
        return self.grid.n_nodes

    def describe(self) -> str:
        """``64x128 equiangular`` or ``icosphere L5``, for logs."""
        if self.grid_kind == "icosphere":
            return f"icosphere L{self.shape[0]}"
        return f"{self.shape[0]}x{self.shape[1]} {self.grid_kind}"


@dataclass(frozen=True, slots=True)
class TensorGridHierarchy:
    """``depth + 1`` grids, finest first, with the transfers a
    SegFormer-shaped model needs between them.

    ``down_edges[k]`` restricts level ``k`` onto level ``k + 1`` (the
    overlap patch merging of stage ``k``); ``up_edges[k]`` interpolates
    level ``k + 1`` onto level ``0`` (the decoder upsamples every stage to
    full resolution). Both are radius-based bipartite edge sets built by
    :func:`~gm_attention.discretization.edge_geometry.compute_bipartite_edge_geometry`.
    """

    levels: tuple[TensorGridLevel, ...]
    down_edges: tuple[EdgeGeometry, ...]
    up_edges: tuple[EdgeGeometry, ...]
    #: ``lateral_edges[k]`` interpolates level ``k + 2`` onto ``k + 1``; empty without ``lateral_cutoffs``.
    lateral_edges: tuple[EdgeGeometry, ...] = ()

    @property
    def depth(self) -> int:
        return len(self.down_edges)

    @property
    def finest(self) -> TensorGridLevel:
        return self.levels[0]

    def tile(self, count: int) -> TensorGridHierarchy:
        """The hierarchy of the disjoint union of ``count`` samples: every
        level and every transfer tiled, so ``count`` panoramas run through a
        model as one graph, each aggregating within its own copy. Node
        ``b * n + i`` of a level is node ``i`` of sample ``b``.
        ``count == 1`` returns ``self``. The union is not a tensor grid, so
        the levels' stencil plans are dropped: a tiled hierarchy runs on the
        graph path."""
        count = int(count)
        if count == 1:
            return self
        levels = tuple(
            TensorGridLevel(
                shape=level.shape,
                grid_kind=level.grid_kind,
                grid=level.grid.tile(count),
                frames=level.frames.repeat(count, 1, 1),
                discretization=None if level.discretization is None else level.discretization.tile(count),
            )
            for level in self.levels
        )
        return TensorGridHierarchy(
            levels=levels,
            down_edges=tuple(
                edges.tile(count, n_sources=self.levels[k].n_nodes) for k, edges in enumerate(self.down_edges)
            ),
            up_edges=tuple(
                edges.tile(count, n_sources=self.levels[k + 1].n_nodes) for k, edges in enumerate(self.up_edges)
            ),
            lateral_edges=tuple(
                edges.tile(count, n_sources=self.levels[k + 2].n_nodes) for k, edges in enumerate(self.lateral_edges)
            ),
        )

    def to(self, device: torch.device | str) -> TensorGridHierarchy:
        """The same hierarchy on ``device``."""
        return TensorGridHierarchy(
            levels=tuple(
                TensorGridLevel(
                    shape=level.shape,
                    grid_kind=level.grid_kind,
                    grid=level.grid.to(device=device),
                    frames=level.frames.to(device=device),
                    discretization=(
                        None if level.discretization is None
                        else level.discretization.to(device=device)
                    ),
                    plans=None if level.plans is None else level.plans.to(device=device),
                )
                for level in self.levels
            ),
            down_edges=tuple(edges.to(device=device) for edges in self.down_edges),
            up_edges=tuple(edges.to(device=device) for edges in self.up_edges),
            lateral_edges=tuple(edges.to(device=device) for edges in self.lateral_edges),
        )


def build_grid_hierarchy(
    grids: Sequence[SphericalGrid],
    grid_kinds: Sequence[str],
    shapes: Sequence[tuple[int, ...]],
    *,
    restrict_cutoffs: Sequence[float],
    interpolate_cutoffs: Sequence[float],
    local_cutoffs: Sequence[Sequence[float] | None],
    frames: Sequence[Tensor | None] | None = None,
    max_frequency: int = DEFAULT_EDGE_MAX_FREQUENCY,
    lateral_cutoffs: Sequence[float] | None = None,
) -> TensorGridHierarchy:
    """The hierarchy over prebuilt grids of any kind, finest first.

    ``lateral_cutoffs``, one per hop after the first, adds the top-down
    transfers ``lateral_edges[k]`` from level ``k + 2`` onto level ``k + 1``
    at those radii; ``None`` builds no lateral edges. Grid kinds mix
    freely, since every transfer reads each side's own nodes, frames and
    quadrature. Every level must have strictly fewer nodes than its
    predecessor. ``shapes`` (``(nlat, nlon)`` or ``(subdivisions,)``) are
    descriptive only, for :meth:`TensorGridLevel.describe`.
    """
    n_levels = len(grids)
    if n_levels < 2:
        raise ValueError(f"a hierarchy needs at least two levels, got {n_levels}.")
    if len(grid_kinds) != n_levels or len(local_cutoffs) != n_levels or len(shapes) != n_levels:
        raise ValueError(
            "grid_kinds, shapes and local_cutoffs need one entry per level "
            f"({n_levels}), got {len(grid_kinds)}, {len(shapes)} and {len(local_cutoffs)}."
        )
    n_hops = n_levels - 1
    if len(restrict_cutoffs) != n_hops or len(interpolate_cutoffs) != n_hops:
        raise ValueError(
            f"restrict_cutoffs and interpolate_cutoffs need one entry per hop "
            f"({n_hops}), got {len(restrict_cutoffs)} and {len(interpolate_cutoffs)}."
        )
    for fine, coarse in itertools.pairwise(grids):
        if coarse.n_nodes >= fine.n_nodes:
            raise ValueError(
                f"levels must be strictly coarser than their predecessor, got "
                f"{fine.n_nodes} -> {coarse.n_nodes} nodes."
            )
    if frames is not None and len(frames) != n_levels:
        raise ValueError(f"frames needs one entry per level ({n_levels}), got {len(frames)}.")

    levels: list[TensorGridLevel] = []
    for index, (grid, kind, shape) in enumerate(zip(grids, grid_kinds, shapes, strict=True)):
        override: Tensor | None = frames[index] if frames is not None else None
        level_frames = override if override is not None else build_tangent_frames(grid.nodes)
        radii = local_cutoffs[index]
        discretization: SphericalDiscretization | None = None
        if radii is not None:
            if not radii:
                raise ValueError(f"local_cutoffs[{index}] is empty; pass None for no operator.")
            discretization = SphericalDiscretization.from_grid(
                grid, coverage_radius=max(radii), frames=level_frames,
                max_frequency=max_frequency,
            )
            discretization.prepare_windows(radii)
        levels.append(
            TensorGridLevel(
                shape=tuple(shape), grid_kind=kind, grid=grid,
                frames=level_frames, discretization=discretization,
            )
        )

    finest = levels[0]
    down_edges: list[EdgeGeometry] = []
    up_edges: list[EdgeGeometry] = []
    for hop in range(n_hops):
        upper, lower = levels[hop], levels[hop + 1]
        down_edges.append(
            compute_bipartite_edge_geometry(
                upper.grid.nodes, upper.frames, upper.grid.quadrature_weights,
                lower.grid.nodes, lower.frames,
                cutoff=float(restrict_cutoffs[hop]), max_frequency=max_frequency,
                source_exact_nodes=upper.grid.exact_nodes, target_exact_nodes=lower.grid.exact_nodes,
            )
        )
        up_edges.append(
            compute_bipartite_edge_geometry(
                lower.grid.nodes, lower.frames, lower.grid.quadrature_weights,
                finest.grid.nodes, finest.frames,
                cutoff=float(interpolate_cutoffs[hop]), max_frequency=max_frequency,
                source_exact_nodes=lower.grid.exact_nodes, target_exact_nodes=finest.grid.exact_nodes,
            )
        )

    lateral_edges: list[EdgeGeometry] = []
    if lateral_cutoffs is not None:
        if len(lateral_cutoffs) != n_hops - 1:
            raise ValueError(
                f"lateral_cutoffs needs one entry per hop after the first ({n_hops - 1}), "
                f"got {len(lateral_cutoffs)}."
            )
        for hop, cutoff in enumerate(lateral_cutoffs):
            coarse, fine_stage = levels[hop + 2], levels[hop + 1]
            lateral_edges.append(
                compute_bipartite_edge_geometry(
                    coarse.grid.nodes, coarse.frames, coarse.grid.quadrature_weights,
                    fine_stage.grid.nodes, fine_stage.frames,
                    cutoff=float(cutoff), max_frequency=max_frequency,
                    source_exact_nodes=coarse.grid.exact_nodes, target_exact_nodes=fine_stage.grid.exact_nodes,
                )
            )
    return TensorGridHierarchy(
        levels=tuple(levels), down_edges=tuple(down_edges), up_edges=tuple(up_edges),
        lateral_edges=tuple(lateral_edges),
    )
