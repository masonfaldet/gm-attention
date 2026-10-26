"""Per-row stencils: the edge geometry of a tensor-product grid folded along longitude.

On a tensor-product grid (:func:`~gm_attention.discretization.grids.torch_harmonics.build_torch_harmonics_grid`:
node ``h * n_longitude + w`` at colatitude row ``h`` and longitude ``w``) with the
coordinate frames of :func:`~gm_attention.discretization.frames.build_coordinate_frames`,
a rotation about the polar axis by a multiple of ``2 pi / n_longitude`` is an
isometry that maps the grid onto itself and every frame onto a frame. The
geometry of an edge ``(ho, wo) <- (hi, wi)``, its log coordinates, distance,
transport angle and tangent quadrature weight, therefore depends only on
``(ho, hi, (wi - wo) mod n_longitude)``: one *slot* of output row ``ho``.

:class:`TensorGridStencil` keeps one representative edge per slot (the edges
whose target has longitude ``0``) together with each slot's source row and
longitude offset, so that a layer on a dense ``(..., n_latitude, n_longitude,
channels)`` layout reads the source of slot ``s`` at target ``(ho, wo)`` from
``(source_row[s], (wo + offset[s]) mod n_longitude)``: the access pattern of
torch-harmonics' neighbourhood kernels, with ``n_longitude`` times fewer
geometry rows than the edge set. :func:`build_tensor_grid_stencil` checks the
folding against every edge of the full geometry and raises where it fails (a
frame field that is not covariant under the rotations, pole nodes, or an edge
set that is not itself shift invariant).
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import Tensor

from gm_attention.discretization.discretization import SphericalDiscretization
from gm_attention.discretization.edge_geometry import (
    DEFAULT_MAX_FREQUENCY,
    EdgeGeometry,
    cutoff_tolerance,
)
from gm_attention.discretization.frames import build_coordinate_frames
from gm_attention.discretization.grids.torch_harmonics import (
    build_torch_harmonics_grid,
)

#: Largest deviation tolerated between an edge and its slot's representative, per dtype.
FOLD_TOLERANCE = {torch.float64: 1e-9, torch.float32: 1e-4}


class TensorGridStencil:
    """The edge geometry of a tensor-product grid at one support radius, one row per slot.

    The constructor checks that the tables describe the same slots (a layer
    indexes them independently) and that every row carries its self-loop
    slot, on which a softmax over the slots relies to stay finite;
    :func:`build_tensor_grid_stencil` is the way to obtain a consistent one.

    Parameters
    ----------
    n_latitude, n_longitude
        The grid shape; the grid has ``n_latitude * n_longitude`` nodes,
        ``h * n_longitude + w`` at row ``h`` and longitude ``w``.
    cutoff
        The support radius the slots were selected at, in ``(0, pi)``; no
        slot's distance exceeds it (within
        :func:`~gm_attention.discretization.edge_geometry.cutoff_tolerance`).
    geometry
        The representative edges, one per slot: every edge of the radius
        subset whose target has longitude ``0``, in the edge set's own
        target-major, source-minor order, so that the slots of row ``ho``
        are contiguous and ordered by ``(source_row, offset)``. Its
        ``row_ptr`` runs over all ``n_latitude * n_longitude`` nodes, as the
        edge set's does; :attr:`row_ptr` is the per-row view.
    row_ptr
        Shape ``(n_latitude + 1,)``: row ``ho``'s slots occupy
        ``[row_ptr[ho], row_ptr[ho + 1])`` of every per-slot tensor; equal
        to ``geometry.row_ptr`` read at the nodes of longitude ``0``.
    source_row, offset
        Shape ``(n_slots,)``: the colatitude row of the slot's source, and
        its longitude offset ``dw`` in ``[0, n_longitude)``, so that the
        source of slot ``s`` at target ``(ho, wo)`` is node
        ``source_row[s] * n_longitude + (wo + offset[s]) mod n_longitude``;
        equal to ``geometry.source_index // n_longitude`` and
        ``geometry.source_index % n_longitude``.
    arcs
        Shape ``(n_arcs, 4)`` of ``(ho, hi, dw_start, length)``: the maximal
        runs of consecutive offsets (modulo ``n_longitude``) among the slots
        of row ``ho`` with source row ``hi``, the circular longitude windows
        a stencil kernel would loop over; together they cover every slot
        once. On the grids this module is built for every ``(ho, hi)`` pair
        is one arc (a geodesic cap meets a latitude circle in one arc),
        which :attr:`arcs_are_contiguous` reports.
    """

    __slots__ = (
        "arcs",
        "cutoff",
        "geometry",
        "n_latitude",
        "n_longitude",
        "offset",
        "row_ptr",
        "source_row",
    )

    def __init__(
        self,
        n_latitude: int,
        n_longitude: int,
        cutoff: float,
        geometry: EdgeGeometry,
        row_ptr: Tensor,  # (n_latitude + 1,)
        source_row: Tensor,  # (n_slots,)
        offset: Tensor,  # (n_slots,)
        arcs: Tensor,  # (n_arcs, 4)
    ) -> None:
        n_latitude, n_longitude = int(n_latitude), int(n_longitude)
        cutoff = float(cutoff)
        if n_latitude < 1 or n_longitude < 1:
            raise ValueError(f"The grid shape must be positive, got ({n_latitude}, {n_longitude}).")
        if not (math.isfinite(cutoff) and 0.0 < cutoff < math.pi):
            raise ValueError(f"cutoff must lie in (0, pi), got {cutoff}.")
        if not isinstance(geometry, EdgeGeometry):
            raise TypeError(f"geometry must be an EdgeGeometry, got {type(geometry).__name__}.")

        n_nodes = n_latitude * n_longitude
        n_slots = geometry.n_edges
        device = geometry.row_ptr.device

        for name, tensor, shape in (
            ("row_ptr", row_ptr, (n_latitude + 1,)),
            ("source_row", source_row, (n_slots,)),
            ("offset", offset, (n_slots,)),
        ):
            if not isinstance(tensor, Tensor):
                raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}.")
            if tensor.dtype != torch.long:
                raise TypeError(f"{name} must have dtype torch.long, got {tensor.dtype}.")
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}.")
            if tensor.device != device:
                raise ValueError(f"{name} and geometry must be on the same device, got {tensor.device} and {device}.")
        if not isinstance(arcs, Tensor):
            raise TypeError(f"arcs must be a torch.Tensor, got {type(arcs).__name__}.")
        if arcs.dtype != torch.long:
            raise TypeError(f"arcs must have dtype torch.long, got {arcs.dtype}.")
        if arcs.ndim != 2 or arcs.shape[1] != 4:
            raise ValueError(f"arcs must have shape (n_arcs, 4), got {tuple(arcs.shape)}.")
        if arcs.device != device:
            raise ValueError(f"arcs and geometry must be on the same device, got {arcs.device} and {device}.")

        # The geometry's rows are the slots: targets at longitude 0, in the
        # edge set's target-major order, with row_ptr over every grid node.
        if geometry.row_ptr.shape[0] != n_nodes + 1:
            raise ValueError(
                f"geometry.row_ptr must run over the {n_nodes} grid nodes, got {geometry.row_ptr.shape[0] - 1}."
            )
        if n_slots and bool((geometry.target_index % n_longitude != 0).any()):
            raise ValueError("Every representative edge must have a target at longitude 0.")
        longitude_zero = torch.arange(n_latitude + 1, device=device) * n_longitude
        if not torch.equal(row_ptr, geometry.row_ptr[longitude_zero]):
            raise ValueError("row_ptr must equal geometry.row_ptr at the nodes of longitude 0.")
        if not torch.equal(source_row, geometry.source_index // n_longitude):
            raise ValueError("source_row must equal geometry.source_index // n_longitude.")
        if not torch.equal(offset, geometry.source_index % n_longitude):
            raise ValueError("offset must equal geometry.source_index % n_longitude.")
        if n_slots and float(geometry.distances.max()) > cutoff + cutoff_tolerance(geometry.distances.dtype):
            raise ValueError(
                f"geometry contains a slot at distance {float(geometry.distances.max()):.12g} beyond the "
                f"cutoff {cutoff:.12g}."
            )

        # Every row carries exactly one self-loop slot (source row ho, offset 0).
        row_of_slot = torch.repeat_interleave(
            torch.arange(n_latitude, device=device), row_ptr[1:] - row_ptr[:-1]
        )
        self_loops = (source_row == row_of_slot) & (offset == 0)
        if not bool((torch.bincount(row_of_slot[self_loops], minlength=n_latitude) == 1).all()):
            raise ValueError("Every output row must carry exactly one self-loop slot (its own row at offset 0).")

        # The arcs cover the slots once, with every entry in range.
        if arcs.shape[0]:
            if bool((arcs[:, 0] < 0).any() or (arcs[:, 0] >= n_latitude).any() or (arcs[:, 1] < 0).any()
                    or (arcs[:, 1] >= n_latitude).any()):
                raise ValueError("arcs contains a row outside [0, n_latitude).")
            if bool((arcs[:, 2] < 0).any() or (arcs[:, 2] >= n_longitude).any()):
                raise ValueError("arcs contains an offset outside [0, n_longitude).")
            if bool((arcs[:, 3] < 1).any() or (arcs[:, 3] > n_longitude).any()):
                raise ValueError("Every arc must have a length in [1, n_longitude].")
        if int(arcs[:, 3].sum()) != n_slots:
            raise ValueError("The arcs must cover every slot exactly once.")

        self.n_latitude = n_latitude
        self.n_longitude = n_longitude
        self.cutoff = cutoff
        self.geometry = geometry
        self.row_ptr = row_ptr.contiguous()
        self.source_row = source_row.contiguous()
        self.offset = offset.contiguous()
        self.arcs = arcs.contiguous()

    @property
    def n_slots(self) -> int:
        return self.geometry.n_edges

    @property
    def n_nodes(self) -> int:
        return self.n_latitude * self.n_longitude

    @property
    def device(self) -> torch.device:
        return self.row_ptr.device

    @property
    def dtype(self) -> torch.dtype:
        return self.geometry.distances.dtype

    @property
    def arcs_are_contiguous(self) -> bool:
        """Whether every ``(ho, hi)`` pair's offsets form a single circular run."""
        pairs = torch.unique(self.arcs[:, 0] * self.n_latitude + self.arcs[:, 1])
        return int(pairs.numel()) == int(self.arcs.shape[0])

    def slots(self, row: int) -> slice:
        """The slice of every per-slot tensor holding output row ``row``'s slots."""
        if not 0 <= row < self.n_latitude:
            raise IndexError(f"row must lie in [0, {self.n_latitude}), got {row}.")
        return slice(int(self.row_ptr[row]), int(self.row_ptr[row + 1]))

    def degrees(self) -> Tensor:  # (n_latitude,)
        """The number of slots, equal to the number of neighbours, of each output row."""
        return self.row_ptr[1:] - self.row_ptr[:-1]

    def source_longitudes(self, slots: slice | Tensor | None = None) -> Tensor:  # (k, n_longitude)
        """``(wo + offset[s]) mod n_longitude`` for every target longitude ``wo``:
        the source longitude of each selected slot at each target longitude."""
        offset = self.offset if slots is None else self.offset[slots]
        longitudes = torch.arange(self.n_longitude, device=offset.device)
        return (offset.unsqueeze(-1) + longitudes) % self.n_longitude

    def source_nodes(self, slots: slice | Tensor | None = None) -> Tensor:  # (k, n_longitude)
        """The source node index of each selected slot at each target longitude."""
        source_row = self.source_row if slots is None else self.source_row[slots]
        return source_row.unsqueeze(-1) * self.n_longitude + self.source_longitudes(slots)

    def repeated(self, count: int) -> EdgeGeometry:
        """The representative geometry with every slot's row repeated ``count``
        times consecutively, ``(n_slots * count,)`` edges: the edge set on
        which a steerable kernel applied to the ``count = d_in`` basis
        vectors of its input type yields its matrix at every slot, the
        check :func:`gm_attention.nn.fixed_grid.steerable_kernel_matrices`
        is tested against. Built on each call from the current geometry."""
        count = int(count)
        if count < 1:
            raise ValueError(f"count must be at least 1, got {count}.")
        index = torch.arange(self.n_slots, device=self.device).repeat_interleave(count)
        return self.geometry.select(index)

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> TensorGridStencil:
        """The same stencil on a new device or floating-point dtype."""
        return type(self)(
            n_latitude=self.n_latitude,
            n_longitude=self.n_longitude,
            cutoff=self.cutoff,
            geometry=self.geometry.to(device=device, dtype=dtype, non_blocking=non_blocking),
            row_ptr=self.row_ptr.to(device=device, non_blocking=non_blocking),
            source_row=self.source_row.to(device=device, non_blocking=non_blocking),
            offset=self.offset.to(device=device, non_blocking=non_blocking),
            arcs=self.arcs.to(device=device, non_blocking=non_blocking),
        )

    def __repr__(self) -> str:
        degrees = self.degrees()
        return (
            f"{type(self).__name__}(shape=({self.n_latitude}, {self.n_longitude}), "
            f"cutoff={self.cutoff:.6g}, n_slots={self.n_slots}, "
            f"neighbours={int(degrees.min())}..{int(degrees.max())}, "
            f"arcs={int(self.arcs.shape[0])}, dtype={self.dtype}, device='{self.device}')"
        )


def build_tensor_grid_discretization(
    grid: str,
    n_latitude: int,
    n_longitude: int,
    coverage_radius: float,
    *,
    max_frequency: int = DEFAULT_MAX_FREQUENCY,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> SphericalDiscretization:
    """A :class:`SphericalDiscretization` of a tensor-product grid in the coordinate gauge.

    The grid of :func:`build_torch_harmonics_grid` with the frames of
    :func:`build_coordinate_frames`, so that every radius subset folds
    into a :class:`TensorGridStencil` by :func:`build_tensor_grid_stencil`.
    ``grid`` must have no pole nodes: ``"legendre-gauss"``.
    """
    spherical_grid = build_torch_harmonics_grid(grid, n_latitude, n_longitude, dtype=dtype, device=device)
    frames = build_coordinate_frames(spherical_grid.nodes)
    return SphericalDiscretization.from_grid(
        spherical_grid, coverage_radius, frames=frames, max_frequency=max_frequency
    )


def build_tensor_grid_stencil(
    discretization: SphericalDiscretization,
    cutoff: float,
    *,
    n_latitude: int,
    n_longitude: int,
    tolerance: float | None = None,
) -> TensorGridStencil:
    """Fold the edges within ``cutoff`` of a tensor-product discretization into per-row slots.

    Parameters
    ----------
    discretization
        Over a tensor-product grid of ``n_latitude * n_longitude`` nodes in
        the order of :func:`build_torch_harmonics_grid`, with frames
        covariant under rotations about the polar axis
        (:func:`build_coordinate_frames`), as
        :func:`build_tensor_grid_discretization` builds.
    cutoff
        The support radius; the slots are the fold of
        ``discretization.edges_within(cutoff)``.
    n_latitude, n_longitude
        The grid shape, which a :class:`~gm_attention.discretization.grid.SphericalGrid`
        does not record.
    tolerance
        Largest deviation allowed between an edge's geometry and its slot's
        representative, by default :data:`FOLD_TOLERANCE` of the dtype.

    Raises
    ------
    ValueError
        If the edge set is not invariant under the longitude shifts (some
        ``(ho, hi, dw)`` class has fewer than ``n_longitude`` edges), or if
        an edge's distance, log coordinates, tangent quadrature weight,
        transport phases, or angular phases weighed by ``r / cutoff`` (the
        least power a kernel scales them by; at ``r = 0`` the polar angle is
        rounding noise on every path) deviate from its representative's by
        more than ``tolerance``: the frames are not covariant (the default
        :func:`~gm_attention.discretization.frames.build_tangent_frames` are
        not) or the grid has pole nodes.
    """
    n_latitude, n_longitude = int(n_latitude), int(n_longitude)
    n_nodes = n_latitude * n_longitude
    if discretization.n_nodes != n_nodes:
        raise ValueError(
            f"The discretization has {discretization.n_nodes} nodes, not n_latitude * n_longitude = {n_nodes}."
        )
    edges = discretization.edges_within(cutoff)
    if tolerance is None:
        tolerance = FOLD_TOLERANCE[edges.distances.dtype]

    target, source = edges.target_index, edges.source_index
    row_out, longitude_out = target // n_longitude, target % n_longitude
    row_in, longitude_in = source // n_longitude, source % n_longitude
    offset = (longitude_in - longitude_out) % n_longitude
    key = (row_out * n_latitude + row_in) * n_longitude + offset  # (n_edges,)

    classes, inverse = torch.unique(key, return_inverse=True)
    class_sizes = torch.bincount(inverse, minlength=classes.numel())
    incomplete = class_sizes != n_longitude
    if bool(incomplete.any()):
        raise ValueError(
            f"The edge set is not invariant under the longitude shifts: {int(incomplete.sum())} of "
            f"{classes.numel()} (ho, hi, dw) classes do not hold one edge per target longitude."
        )

    # Every class holds exactly one edge whose target has longitude 0: the representative.
    representative_mask = longitude_out == 0
    representatives = torch.nonzero(representative_mask).squeeze(-1)  # (n_slots,) ascending
    class_to_slot = torch.empty(classes.numel(), dtype=torch.long, device=key.device)
    class_to_slot[inverse[representatives]] = torch.arange(representatives.numel(), device=key.device)
    representative_of = representatives[class_to_slot[inverse]]  # (n_edges,)

    # A polar angle is atan2 of the log coordinates, so its rounding error
    # grows like 1 / r and at r = 0 (the self-loops) it is noise. A kernel
    # reads the angular phase e^{ik phi} only through modes scaled by
    # (r / cutoff)^|k| with |k| >= 1, so the deviation that can reach a
    # layer is the phase deviation times r / cutoff: that is what is checked.
    angular_weight = (edges.distances / float(cutoff)).clamp(max=1.0).unsqueeze(-1)
    deviations: dict[str, float] = {}
    for name in ("distances", "log_coordinates", "tangent_quadrature_weights", "angular_phases", "transport_phases"):
        table = getattr(edges, name)
        difference = (table - table[representative_of]).abs()
        if name == "angular_phases":
            difference = difference * angular_weight
        deviations[name] = float(difference.max()) if difference.numel() else 0.0
    violated = {name: value for name, value in deviations.items() if not value <= tolerance}
    if violated:
        worst = ", ".join(f"{name} {value:.3g}" for name, value in violated.items())
        raise ValueError(
            "The edge geometry does not fold along longitude (largest deviation from the slot "
            f"representative above the tolerance {tolerance:g}: {worst}). The frames must be covariant "
            "under rotations about the polar axis (build_coordinate_frames) and the grid free of pole nodes."
        )

    geometry = edges.select(representative_mask)
    row_ptr = geometry.row_ptr[torch.arange(n_latitude + 1, device=key.device) * n_longitude]
    source_row = geometry.source_index // n_longitude
    slot_offset = geometry.source_index % n_longitude  # the target longitude is 0
    arcs = _longitude_arcs(row_ptr, source_row, slot_offset, n_longitude)

    return TensorGridStencil(
        n_latitude=n_latitude,
        n_longitude=n_longitude,
        cutoff=cutoff,
        geometry=geometry,
        row_ptr=row_ptr,
        source_row=source_row,
        offset=slot_offset,
        arcs=arcs,
    )


def _longitude_arcs(
    row_ptr: Tensor,  # (n_latitude + 1,)
    source_row: Tensor,  # (n_slots,)
    offset: Tensor,  # (n_slots,)
    n_longitude: int,
) -> Tensor:  # (n_arcs, 4)
    """The maximal circular runs of consecutive offsets per ``(ho, hi)``,
    as ``(ho, hi, dw_start, length)`` rows sorted by ``(ho, hi, dw_start)``."""
    rows: list[tuple[int, int, int, int]] = []
    n_latitude = row_ptr.shape[0] - 1
    source_cpu, offset_cpu = source_row.cpu(), offset.cpu()
    for row_out in range(n_latitude):
        first, last = int(row_ptr[row_out]), int(row_ptr[row_out + 1])
        if first == last:
            continue
        for row_in in torch.unique(source_cpu[first:last]).tolist():
            offsets = offset_cpu[first:last][source_cpu[first:last] == row_in].sort().values
            count = int(offsets.numel())
            if count == n_longitude:
                rows.append((row_out, row_in, 0, n_longitude))
                continue
            # A break after position k: the next offset (circularly) is not offset + 1.
            following = offsets.roll(-1)
            breaks = torch.nonzero((following - offsets) % n_longitude != 1).squeeze(-1).tolist()
            starts = sorted((k + 1) % count for k in breaks)
            for index, start in enumerate(starts):
                stop = starts[(index + 1) % len(starts)]
                length = (stop - start) % count or count
                rows.append((row_out, row_in, int(offsets[start]), length))
    rows.sort()
    arcs = torch.tensor(rows, dtype=torch.long).reshape(-1, 4)
    return arcs.to(row_ptr.device)


def slot_classes(
    n_latitude: int,
    n_longitude: int,
    edges: EdgeGeometry,
) -> Tensor:  # (n_edges, 3)
    """``(ho, hi, dw)`` of every edge of a tensor-product edge set: the class
    a slot represents, for tests and tools that compare the two layouts."""
    target, source = edges.target_index, edges.source_index
    row_out, longitude_out = target // n_longitude, target % n_longitude
    row_in, longitude_in = source // n_longitude, source % n_longitude
    return torch.stack((row_out, row_in, (longitude_in - longitude_out) % n_longitude), dim=-1)


def prepare_stencils(
    discretization: SphericalDiscretization,
    cutoffs: Iterable[float],
    *,
    n_latitude: int,
    n_longitude: int,
) -> dict[float, TensorGridStencil]:
    """One stencil per support radius, keyed as
    :attr:`SphericalDiscretization.cached_radii` keys its radius subsets;
    ``edges_within`` validates each radius."""
    stencils: dict[float, TensorGridStencil] = {}
    for cutoff in cutoffs:
        stencils[SphericalDiscretization._radius_key(cutoff)] = build_tensor_grid_stencil(
            discretization, cutoff, n_latitude=n_latitude, n_longitude=n_longitude
        )
    return stencils
