"""Bucketed slot tables: a tensor-grid stencil arranged to evaluate every output row at once.

The per-row reference of :mod:`gm_attention.nn.fixed_grid` loops over the
output rows of a :class:`~gm_attention.discretization.stencil.TensorGridStencil`
because the rows have different numbers of slots: on a Gauss-Legendre grid
the longitude spacing shrinks like ``sin(colatitude)``, so a cap of fixed
geodesic radius holds a handful of neighbours at the equator and whole
latitude circles near the poles. A :class:`StencilPlan` removes the loop by
grouping the rows into a few *buckets* of similar slot count, padding every
row of a bucket to the bucket's width, and tabulating per bucket the slot
index and the flat source node of every ``(row, target longitude, padded
position)``. A layer then gathers one bucket's whole neighbourhood in one
``index_select`` and contracts it in one batched matmul, and the shapes it
sees are fixed once the plan is built: nothing inside a forward depends on
the data, which is what ``torch.compile`` needs.

Padding points at a *phantom slot*, index ``n_slots``, one past the last
real slot. Every per-slot table a layer reads through the plan is extended
by one row for it (:meth:`StencilPlan.pad`: a zero matrix, a ``-inf``
softmax bias, a zero window weight), so a gather produces neutral values at
the padded positions and no mask is needed in the hot loop; the phantom's
source node is the target node itself, a valid address whose value is
multiplied by zero.

The buckets are chosen by :func:`partition_degrees`: the rows sorted by slot
count are cut into at most ``max_buckets`` contiguous groups so that the
padded slot count ``sum_b n_rows_b * width_b`` is minimal (a dynamic program
over the distinct slot counts). Eight buckets bring the padding below a few
percent on the grids this module is built for; one bucket is the plain
padding to the maximal degree.

A model's layers run at several cutoffs on one grid. :class:`StencilPlans`
holds one plan per cutoff, keyed as a
:class:`~gm_attention.discretization.discretization.SphericalDiscretization`
keys its radius subsets, and a layer given the container resolves the plan
at its own cutoff (:func:`split_edges_and_plan`), so a block hands whatever it
received to each of its layers, as it does with a discretization.
:func:`build_stencil_plans` builds the container over a discretization, and
:func:`build_tensor_grid_geometry` builds both at once for a tensor grid in
the coordinate gauge: the discretization with its windows prepared at the
cutoffs, for the graph path, and the plans at the same cutoffs, for the
stencil path.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

import torch
from torch import Tensor

from gm_attention.discretization.discretization import SphericalDiscretization
from gm_attention.discretization.edge_geometry import DEFAULT_MAX_FREQUENCY
from gm_attention.discretization.stencil import (
    TensorGridStencil,
    build_tensor_grid_discretization,
    build_tensor_grid_stencil,
)

#: Default upper bound on the number of buckets of :func:`build_stencil_plan`.
DEFAULT_MAX_BUCKETS = 8


class StencilBucket:
    """The rows of one bucket and their padded slot tables.

    Parameters
    ----------
    rows
        Shape ``(n_rows,)``: the output rows of this bucket, ascending.
    slots
        Shape ``(n_rows, width)``: the slot index at every padded position,
        the row's slots in the stencil's order followed by the phantom
        slot ``n_slots`` up to the bucket's width.
    valid
        Shape ``(n_rows, width)``: ``True`` at the real slots.
    source
        Shape ``(n_rows, n_longitude, width)``: the flat source node of each
        position at each target longitude ``wo``,
        ``source_row * n_longitude + (wo + offset) mod n_longitude``; at a
        padded position the target node ``row * n_longitude + wo`` itself.
    """

    __slots__ = ("rows", "slots", "source", "valid")

    def __init__(self, rows: Tensor, slots: Tensor, valid: Tensor, source: Tensor) -> None:
        self.rows = rows
        self.slots = slots
        self.valid = valid
        self.source = source

    @property
    def n_rows(self) -> int:
        return int(self.rows.shape[0])

    @property
    def width(self) -> int:
        return int(self.slots.shape[1])

    def to(self, device: torch.device | str | None = None, non_blocking: bool = False) -> StencilBucket:
        return type(self)(
            rows=self.rows.to(device=device, non_blocking=non_blocking),
            slots=self.slots.to(device=device, non_blocking=non_blocking),
            valid=self.valid.to(device=device, non_blocking=non_blocking),
            source=self.source.to(device=device, non_blocking=non_blocking),
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n_rows={self.n_rows}, width={self.width})"


class StencilPlan:
    """A :class:`~gm_attention.discretization.stencil.TensorGridStencil` with its rows bucketed by slot count.

    Built by :func:`build_stencil_plan`. The stencil's own tables stay the
    source of truth: the plan adds the buckets, the order in which they list
    the rows and its inverse, so that a layer concatenating the buckets'
    outputs along the row axis and indexing with :attr:`inverse_order`
    recovers the grid's row order.

    Parameters
    ----------
    stencil
        The stencil the buckets index.
    buckets
        The buckets, every output row in exactly one of them.
    """

    __slots__ = ("buckets", "inverse_order", "row_order", "stencil")

    def __init__(self, stencil: TensorGridStencil, buckets: Sequence[StencilBucket]) -> None:
        if not isinstance(stencil, TensorGridStencil):
            raise TypeError(f"stencil must be a TensorGridStencil, got {type(stencil).__name__}.")
        buckets = tuple(buckets)
        if not buckets:
            raise ValueError("A plan needs at least one bucket.")
        device = stencil.device
        n_latitude, n_longitude, n_slots = stencil.n_latitude, stencil.n_longitude, stencil.n_slots
        for bucket in buckets:
            if not isinstance(bucket, StencilBucket):
                raise TypeError(f"Every bucket must be a StencilBucket, got {type(bucket).__name__}.")
            n_rows, width = bucket.n_rows, bucket.width
            if bucket.rows.ndim != 1 or n_rows < 1:
                raise ValueError("Every bucket must hold at least one row.")
            if tuple(bucket.slots.shape) != (n_rows, width) or tuple(bucket.valid.shape) != (n_rows, width):
                raise ValueError("slots and valid must have shape (n_rows, width).")
            if tuple(bucket.source.shape) != (n_rows, n_longitude, width):
                raise ValueError(
                    f"source must have shape (n_rows, n_longitude, width) = ({n_rows}, {n_longitude}, {width}), "
                    f"got {tuple(bucket.source.shape)}."
                )
            for name in ("rows", "slots", "source"):
                tensor = getattr(bucket, name)
                if tensor.dtype != torch.long:
                    raise TypeError(f"{name} must have dtype torch.long, got {tensor.dtype}.")
                if tensor.device != device:
                    raise ValueError(f"{name} must live on the stencil's device {device}, got {tensor.device}.")
            if bucket.valid.dtype != torch.bool:
                raise TypeError(f"valid must have dtype torch.bool, got {bucket.valid.dtype}.")
            if bool((bucket.slots < 0).any() or (bucket.slots > n_slots).any()):
                raise ValueError(f"slots must lie in [0, n_slots] = [0, {n_slots}] (n_slots is the phantom slot).")
            if not torch.equal(bucket.valid, bucket.slots < n_slots):
                raise ValueError("valid must mark exactly the positions holding a real slot.")
        row_order = torch.cat([bucket.rows for bucket in buckets])
        if row_order.shape[0] != n_latitude or not torch.equal(
            torch.sort(row_order).values, torch.arange(n_latitude, device=device)
        ):
            raise ValueError(f"The buckets must list every one of the {n_latitude} rows exactly once.")
        self.stencil = stencil
        self.buckets = buckets
        self.row_order = row_order
        self.inverse_order = torch.argsort(row_order)

    # Delegation to the stencil, so a layer needs the plan alone.

    @property
    def n_latitude(self) -> int:
        return self.stencil.n_latitude

    @property
    def n_longitude(self) -> int:
        return self.stencil.n_longitude

    @property
    def n_nodes(self) -> int:
        return self.stencil.n_nodes

    @property
    def n_slots(self) -> int:
        return self.stencil.n_slots

    @property
    def cutoff(self) -> float:
        return self.stencil.cutoff

    @property
    def device(self) -> torch.device:
        return self.stencil.device

    @property
    def dtype(self) -> torch.dtype:
        return self.stencil.dtype

    @property
    def n_buckets(self) -> int:
        return len(self.buckets)

    @property
    def n_padded(self) -> int:
        """``sum_b n_rows_b * width_b``: the slot positions a layer evaluates,
        padding included; at least :attr:`n_slots`."""
        return sum(bucket.n_rows * bucket.width for bucket in self.buckets)

    @property
    def padding_fraction(self) -> float:
        """The share of evaluated positions that are padding."""
        return 1.0 - self.n_slots / self.n_padded

    def pad(self, table: Tensor, fill: float, dim: int = 0) -> Tensor:
        """``table`` with one row for the phantom slot appended along ``dim``,
        holding ``fill``: the form in which a per-slot table of ``n_slots``
        rows is indexed by a bucket's :attr:`StencilBucket.slots`."""
        if table.shape[dim] != self.n_slots:
            raise ValueError(f"table must hold n_slots = {self.n_slots} rows along dim {dim}, got {table.shape[dim]}.")
        shape = list(table.shape)
        shape[dim] = 1
        phantom = torch.full(shape, fill, dtype=table.dtype, device=table.device)
        return torch.cat([table, phantom], dim=dim)

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None,
           non_blocking: bool = False) -> StencilPlan:
        """The same plan on a new device, or over the stencil moved to ``dtype``."""
        return type(self)(
            self.stencil.to(device=device, dtype=dtype, non_blocking=non_blocking),
            [bucket.to(device=device, non_blocking=non_blocking) for bucket in self.buckets],
        )

    def __repr__(self) -> str:
        widths = ", ".join(f"{bucket.n_rows}x{bucket.width}" for bucket in self.buckets)
        return (
            f"{type(self).__name__}(shape=({self.n_latitude}, {self.n_longitude}), n_slots={self.n_slots}, "
            f"buckets=[{widths}], padding={self.padding_fraction:.1%})"
        )


class StencilPlans:
    """One :class:`StencilPlan` per cutoff over one tensor grid.

    Parameters
    ----------
    plans
        ``{cutoff: plan}``; every plan over the same grid shape, dtype and
        device. The keys are stored as
        :meth:`SphericalDiscretization._radius_key` stores radii, so
        :meth:`for_cutoff` finds a plan by the cutoff a layer was built with.
    """

    __slots__ = ("_plans",)

    def __init__(self, plans: Mapping[float, StencilPlan]) -> None:
        items = {SphericalDiscretization._radius_key(cutoff): plan for cutoff, plan in plans.items()}
        if not items:
            raise ValueError("StencilPlans needs at least one plan.")
        first = next(iter(items.values()))
        for key, plan in items.items():
            if not isinstance(plan, StencilPlan):
                raise TypeError(f"Every value must be a StencilPlan, got {type(plan).__name__} at cutoff {key}.")
            if not math.isclose(plan.cutoff, key, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"The plan under cutoff {key:.12g} was built at cutoff {plan.cutoff:.12g}.")
            if (plan.n_latitude, plan.n_longitude, plan.dtype, plan.device) != (
                first.n_latitude, first.n_longitude, first.dtype, first.device
            ):
                raise ValueError("Every plan must share the grid shape, dtype and device.")
        self._plans = dict(sorted(items.items()))

    @property
    def cutoffs(self) -> tuple[float, ...]:
        """The cutoffs with a plan, ascending."""
        return tuple(self._plans)

    def __contains__(self, cutoff: object) -> bool:
        return isinstance(cutoff, (int, float)) and SphericalDiscretization._radius_key(float(cutoff)) in self._plans

    def __len__(self) -> int:
        return len(self._plans)

    def for_cutoff(self, cutoff: float) -> StencilPlan:
        """The plan at ``cutoff``; raises if none was built there."""
        key = SphericalDiscretization._radius_key(cutoff)
        try:
            return self._plans[key]
        except KeyError:
            available = ", ".join(f"{c:.6g}" for c in self._plans)
            raise ValueError(f"No stencil plan at cutoff {cutoff:.12g}; the plans hold {available}.") from None

    @property
    def n_latitude(self) -> int:
        return next(iter(self._plans.values())).n_latitude

    @property
    def n_longitude(self) -> int:
        return next(iter(self._plans.values())).n_longitude

    @property
    def n_nodes(self) -> int:
        return next(iter(self._plans.values())).n_nodes

    @property
    def dtype(self) -> torch.dtype:
        return next(iter(self._plans.values())).dtype

    @property
    def device(self) -> torch.device:
        return next(iter(self._plans.values())).device

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None,
           non_blocking: bool = False) -> StencilPlans:
        """Every plan moved; see :meth:`StencilPlan.to`."""
        return type(self)({c: p.to(device=device, dtype=dtype, non_blocking=non_blocking) for c, p in self._plans.items()})

    def __repr__(self) -> str:
        cutoffs = ", ".join(f"{c:.6g}" for c in self._plans)
        return f"{type(self).__name__}(shape=({self.n_latitude}, {self.n_longitude}), cutoffs=[{cutoffs}])"


def build_stencil_plans(
    discretization: SphericalDiscretization,
    cutoffs: Iterable[float],
    *,
    n_latitude: int,
    n_longitude: int,
    max_buckets: int = DEFAULT_MAX_BUCKETS,
    max_bucket_positions: int | None = None,
) -> StencilPlans:
    """A plan per cutoff over a tensor-product discretization in the coordinate gauge.

    Each cutoff's stencil comes from
    :func:`~gm_attention.discretization.stencil.build_tensor_grid_stencil`,
    which validates the fold (and so refuses frames that are not covariant or
    a grid with pole nodes), and is bucketed by :func:`build_stencil_plan`.
    """
    plans: dict[float, StencilPlan] = {}
    for cutoff in cutoffs:
        stencil = build_tensor_grid_stencil(discretization, float(cutoff), n_latitude=n_latitude, n_longitude=n_longitude)
        plans[float(cutoff)] = build_stencil_plan(stencil, max_buckets=max_buckets, max_bucket_positions=max_bucket_positions)
    return StencilPlans(plans)


def build_tensor_grid_geometry(
    grid: str,
    n_latitude: int,
    n_longitude: int,
    cutoffs: Iterable[float],
    *,
    max_frequency: int = DEFAULT_MAX_FREQUENCY,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
    max_buckets: int = DEFAULT_MAX_BUCKETS,
    max_bucket_positions: int | None = None,
) -> tuple[SphericalDiscretization, StencilPlans]:
    """A tensor grid's geometry for both paths: the discretization and the plans at every cutoff.

    The discretization is
    :func:`~gm_attention.discretization.stencil.build_tensor_grid_discretization`
    (``grid`` must have no pole nodes: ``"legendre-gauss"``) with the
    coverage radius ``max(cutoffs)`` and every cutoff's window prepared, so a
    layer can take either object: the graph path resolves its edges by
    ``edges_within``, the stencil path its plan by :meth:`StencilPlans.for_cutoff`,
    and the two evaluate the same function to summation order.
    """
    radii = [float(cutoff) for cutoff in cutoffs]
    if not radii:
        raise ValueError("cutoffs must hold at least one radius.")
    discretization = build_tensor_grid_discretization(
        grid, n_latitude, n_longitude, coverage_radius=max(radii), max_frequency=max_frequency, dtype=dtype, device=device
    )
    discretization.prepare_windows(radii)
    plans = build_stencil_plans(
        discretization, radii, n_latitude=n_latitude, n_longitude=n_longitude,
        max_buckets=max_buckets, max_bucket_positions=max_bucket_positions,
    )
    return discretization, plans


def split_edges_and_plan(edges: object, plan: object, cutoff: float) -> tuple[object, StencilPlan | None]:
    """Sort out what a layer's ``forward(x, edges=None, *, plan=None)`` received.

    Returns ``(edges, None)`` for a graph-path call and ``(None, plan)`` for a
    stencil-path call. A :class:`StencilPlan` or :class:`StencilPlans` passed
    in the ``edges`` position counts as the plan, so that a block can forward
    whichever it was given to its layers unchanged; a :class:`StencilPlans`
    is resolved at the layer's ``cutoff``. Exactly one of the two must be
    given.
    """
    if isinstance(edges, (StencilPlan, StencilPlans)):
        if plan is not None:
            raise TypeError("Pass the stencil plan once, either as edges or as plan.")
        edges, plan = None, edges
    if plan is not None:
        if isinstance(plan, StencilPlans):
            plan = plan.for_cutoff(cutoff)
        if not isinstance(plan, StencilPlan):
            raise TypeError(f"plan must be a StencilPlan or StencilPlans, got {type(plan).__name__}.")
        if edges is not None:
            raise TypeError("forward takes either edges or a plan, not both.")
        return None, plan
    if edges is None:
        raise TypeError("forward needs edges (an EdgeGeometry or a SphericalDiscretization) or plan (a StencilPlan).")
    return edges, None


def partition_degrees(degrees: Sequence[int], max_buckets: int) -> list[list[int]]:
    """Group indices by value into at most ``max_buckets`` groups with the least padding.

    Parameters
    ----------
    degrees
        One nonnegative integer per item (a row's slot count).
    max_buckets
        The largest number of groups allowed.

    Returns
    -------
    list[list[int]]
        The groups, each a list of item indices ascending, ordered by
        increasing value; every item in exactly one group. The groups
        minimize ``sum_g len(g) * max(degrees[i] for i in g)``, the number
        of positions a layer evaluates when each group is padded to its
        largest degree, over all partitions of the items sorted by degree
        into at most ``max_buckets`` contiguous runs (the optimal groups are
        contiguous in that order: swapping two items across groups never
        lowers the cost). A dynamic program over the distinct degrees,
        ``O(max_buckets * n_distinct^2)``.
    """
    max_buckets = int(max_buckets)
    if max_buckets < 1:
        raise ValueError(f"max_buckets must be at least 1, got {max_buckets}.")
    values = torch.as_tensor(list(degrees), dtype=torch.long)
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("degrees must hold at least one item.")
    if bool((values < 0).any()):
        raise ValueError("degrees must be nonnegative.")

    order = torch.argsort(values, stable=True)
    unique, counts = torch.unique_consecutive(values[order], return_counts=True)
    n_distinct = int(unique.numel())
    groups = min(max_buckets, n_distinct)
    prefix = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])  # (n_distinct + 1,)

    # best[g, j]: least cost covering the first j distinct degrees with g groups; choice: where the last group starts.
    infinity = math.inf
    best = torch.full((groups + 1, n_distinct + 1), infinity, dtype=torch.float64)
    choice = torch.zeros((groups + 1, n_distinct + 1), dtype=torch.long)
    best[0, 0] = 0.0
    for g in range(1, groups + 1):
        for j in range(g, n_distinct + 1):
            starts = torch.arange(g - 1, j)  # the last group is [start, j), padded to unique[j - 1]
            cost = best[g - 1, starts] + (prefix[j] - prefix[starts]).to(torch.float64) * float(unique[j - 1])
            k = int(torch.argmin(cost))
            best[g, j] = cost[k]
            choice[g, j] = starts[k]
    n_groups = int(torch.argmin(best[1:, n_distinct])) + 1

    bounds: list[tuple[int, int]] = []
    end = n_distinct
    for g in range(n_groups, 0, -1):
        start = int(choice[g, end])
        bounds.append((start, end))
        end = start
    bounds.reverse()

    result: list[list[int]] = []
    for start, end in bounds:
        members = order[int(prefix[start]) : int(prefix[end])]
        result.append(sorted(int(i) for i in members))
    return result


def build_stencil_plan(
    stencil: TensorGridStencil,
    *,
    max_buckets: int = DEFAULT_MAX_BUCKETS,
    max_bucket_positions: int | None = None,
) -> StencilPlan:
    """Bucket a stencil's rows by slot count and tabulate the padded gathers.

    Parameters
    ----------
    stencil
        The stencil, from :func:`~gm_attention.discretization.stencil.build_tensor_grid_stencil`.
    max_buckets
        The largest number of buckets; see :func:`partition_degrees`. One
        bucket pads every row to the largest degree. More buckets mean less
        padding and more (smaller) batched operations per forward.
    max_bucket_positions
        If given, a bucket whose ``n_rows * width`` exceeds this is split
        into runs of consecutive rows that each stay within it: a bound on
        the working set a layer materializes per bucket (``n_rows * width *
        n_longitude`` gathered features), at no cost in padding.

    Returns
    -------
    StencilPlan
    """
    if not isinstance(stencil, TensorGridStencil):
        raise TypeError(f"stencil must be a TensorGridStencil, got {type(stencil).__name__}.")
    if max_bucket_positions is not None and int(max_bucket_positions) < 1:
        raise ValueError(f"max_bucket_positions must be positive, got {max_bucket_positions}.")

    device = stencil.device
    n_longitude, n_slots = stencil.n_longitude, stencil.n_slots
    degrees = stencil.degrees()  # (n_latitude,)
    longitudes = torch.arange(n_longitude, device=device)

    buckets: list[StencilBucket] = []
    for members in partition_degrees(degrees.tolist(), max_buckets):
        width = max(int(degrees[row]) for row in members)
        if max_bucket_positions is None:
            rows_per_bucket = len(members)
        else:
            rows_per_bucket = max(1, int(max_bucket_positions) // max(width, 1))
        for start in range(0, len(members), rows_per_bucket):
            rows = torch.tensor(members[start : start + rows_per_bucket], dtype=torch.long, device=device)
            positions = torch.arange(width, device=device)
            valid = positions.unsqueeze(0) < degrees[rows].unsqueeze(1)  # (n_rows, width)
            slots = torch.where(valid, stencil.row_ptr[rows].unsqueeze(1) + positions.unsqueeze(0), n_slots)
            clamped = slots.clamp(max=n_slots - 1)
            source_row = torch.where(valid, stencil.source_row[clamped], rows.unsqueeze(1))  # (n_rows, width)
            offset = torch.where(valid, stencil.offset[clamped], 0)
            source = (
                source_row.unsqueeze(1) * n_longitude
                + (offset.unsqueeze(1) + longitudes.reshape(1, -1, 1)) % n_longitude
            )  # (n_rows, n_longitude, width)
            buckets.append(StencilBucket(rows=rows, slots=slots, valid=valid, source=source))
    return StencilPlan(stencil, buckets)
