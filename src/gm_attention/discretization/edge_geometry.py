from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import Tensor

from gm_attention.discretization.grid import SphericalGrid
from gm_attention.representations.so2 import (
    complex_dtype,
    phase_powers,
    validate_frequency,
)

# A steerable kernel from type rho_n to type rho_m depends on the polar angle
# through the harmonics of order |m - n| and m + n; orders up to
# max(I_in) + max(I_out) cover every type pair. Default: frequencies <= 2.
DEFAULT_MAX_FREQUENCY = 4


@dataclass(frozen=True, slots=True)
class EdgeGeometry:
    """Precomputed data for directed edges i <- j.

    Edges are sorted target-major, source-minor, so ``row_ptr`` gives a
    CSR-style view of the same data: node ``i``'s incident edges occupy the
    contiguous half-open range ``[row_ptr[i], row_ptr[i + 1])``.

    ``log_coordinates``, ``polar_angles`` and ``transport_angles`` are
    expressed in the gauge of the ``frames`` passed to
    :func:`compute_edge_geometry`; the object does not store those frames.

    Angular harmonics
    -----------------
    ``angular_phases`` and ``transport_phases`` tabulate ``exp(i k phi)`` and
    ``exp(i k gamma)`` for every order ``k`` up to ``max_frequency``: the
    angular kernel bases and the transport action ``rho_k(r_gamma)``, stored
    as unit complex numbers. :meth:`angular_harmonic` and
    :meth:`transport_harmonic` serve negative orders as conjugates and raise
    for an order that was not tabulated.
    """

    # Edge e runs i <- j: target i = edge_index[0, e], source j = edge_index[1, e].
    edge_index: Tensor  # (2, n_edges)

    # CSR row offsets: edges with target i occupy
    # edge_index[:, row_ptr[i]:row_ptr[i + 1]].
    row_ptr: Tensor  # (n_target_nodes + 1,)

    # log_coordinates[e] = (x, y), the coefficients of log_{p_i}(p_j) in the
    # target's own tangent frame E_i, so log_{p_i}(p_j) = x E_i[:, 0] + y E_i[:, 1].
    log_coordinates: Tensor  # (n_edges, 2)

    # distances[e] = r_ij, the geodesic distance between p_i and p_j; equal
    # to ||log_coordinates[e]||.
    distances: Tensor  # (n_edges,)

    # polar_angles[e] = phi_ij = atan2(y, x), the direction of the source
    # as seen from the target, measured in the target's frame.
    polar_angles: Tensor  # (n_edges,)

    # transport_angles[e] = gamma_{i<-j}, the rotation carrying the source frame
    # E_j, parallel transported along the geodesic to p_i, onto the target frame E_i.
    transport_angles: Tensor  # (n_edges,)

    # tangent_quadrature_weights[e] = q_j * (r/sin r): the source node's spherical
    # weight as a Lebesgue weight on T_{p_i}S^2, so edge sums approximate integrals.
    tangent_quadrature_weights: Tensor  # (n_edges,)

    # angular_phases[e, k] = exp(i k phi_ij), the angular kernel basis.
    angular_phases: Tensor  # (n_edges, max_frequency + 1) complex

    # transport_phases[e, k] = exp(i k gamma_{i<-j}), the action
    # rho_k(r_gamma) of the transport rotation on a frequency-k irrep.
    transport_phases: Tensor  # (n_edges, max_frequency + 1) complex

    @property
    def n_edges(self) -> int:
        return self.edge_index.shape[1]

    @property
    def target_index(self) -> Tensor:
        return self.edge_index[0]  # (n_edges,)

    @property
    def source_index(self) -> Tensor:
        return self.edge_index[1]  # (n_edges,)

    @property
    def max_frequency(self) -> int:
        """Largest tabulated harmonic order."""
        return self.angular_phases.shape[-1] - 1

    def angular_harmonic(
        self,
        order: int,
    ) -> Tensor:  # (n_edges,) complex
        """Return ``exp(i * order * phi)`` for every edge.

        Parameters
        ----------
        order
            Harmonic order, which may be negative. A steerable kernel from
            type ``rho_n`` to type ``rho_m`` needs orders ``m - n`` and
            ``m + n``.

        Returns
        -------
        Tensor
            Complex tensor of shape ``(n_edges,)``.

        Notes
        -----
        Only nonnegative orders are stored; a negative order is served as the
        conjugate of its positive counterpart.
        """
        return self._harmonic(self.angular_phases, order, "angular_phases")

    def transport_harmonic(
        self,
        order: int,
    ) -> Tensor:  # (n_edges,) complex
        """Return ``exp(i * order * gamma)`` for every edge.

        Transporting a frequency ``n`` feature from the source gauge into the
        target gauge multiplies it by ``exp(i n gamma)``, so a layer consuming
        input frequency ``n`` needs order ``n``.
        """
        return self._harmonic(self.transport_phases, order, "transport_phases")

    def _harmonic(
        self,
        table: Tensor,  # (n_edges, max_frequency + 1) complex
        order: int,
        table_name: str,
    ) -> Tensor:  # (n_edges,) complex
        if isinstance(order, bool) or not isinstance(order, int):
            raise TypeError(
                f"order must be an int, got {type(order).__name__}."
            )

        magnitude = abs(order)

        if magnitude > self.max_frequency:
            raise ValueError(
                f"Harmonic order {order} is not tabulated: {table_name} "
                f"covers orders up to {self.max_frequency}. Rebuild the "
                f"discretization with max_frequency >= {magnitude}."
            )

        column = table[:, magnitude]  # (n_edges,) complex

        return column.conj() if order < 0 else column

    def select(
        self,
        selection: Tensor,  # (n_edges,) bool mask, or (k,) long indices
        *,
        n_nodes: int | None = None,
    ) -> EdgeGeometry:
        """Select an edge subset using a Boolean mask or edge indices.

        Parameters
        ----------
        selection
            Boolean edge mask, or a sorted array of edge indices.
        n_nodes
            Number of target nodes the new ``row_ptr`` is sized for; defaults
            to this object's own. Passed explicitly when the selection also
            restricts the targets to the subset ``0 .. n_nodes - 1``
            (bipartite downsampling). Every kept edge's target must be
            ``< n_nodes``; this is not checked.
        """
        if selection.ndim != 1:
            raise ValueError(
                "Edge selection must be one-dimensional."
            )

        if selection.dtype == torch.bool:
            if selection.shape[0] != self.n_edges:
                raise ValueError(
                    "A Boolean edge mask must have one entry per edge."
                )
        elif selection.dtype != torch.long:
            raise TypeError(
                "Edge selection must have dtype torch.bool or torch.long."
            )

        new_edge_index = self.edge_index[:, selection].contiguous()  # (2, k)

        if n_nodes is None:
            n_nodes = self.row_ptr.shape[0] - 1

        return type(self)(
            edge_index=new_edge_index,
            row_ptr=_build_row_ptr(new_edge_index[0], n_nodes),  # (n_nodes + 1,)
            log_coordinates=self.log_coordinates[selection].contiguous(),  # (k, 2)
            distances=self.distances[selection].contiguous(),  # (k,)
            polar_angles=self.polar_angles[selection].contiguous(),  # (k,)
            transport_angles=self.transport_angles[selection].contiguous(),  # (k,)
            tangent_quadrature_weights=(
                self.tangent_quadrature_weights[selection].contiguous()  # (k,)
            ),
            angular_phases=(
                self.angular_phases[selection].contiguous()  # (k, K + 1)
            ),
            transport_phases=(
                self.transport_phases[selection].contiguous()  # (k, K + 1)
            ),
        )

    def tile(self, count: int, *, n_sources: int) -> EdgeGeometry:
        """The disjoint union of ``count`` copies of this edge set: copy ``b``
        connects sources ``b * n_sources + j`` to targets ``b * n_targets + i``.

        Every per-edge table is repeated; ``row_ptr`` runs over the
        ``count * n_targets`` union targets, and the edges stay target-major
        because the copies follow each other. ``n_sources`` is the source
        grid's node count (the target count for a same-grid geometry, the
        fine grid's for a bipartite one). ``count == 1`` returns ``self``.
        """
        count = int(count)
        if count < 1:
            raise ValueError(f"count must be at least 1, got {count}.")
        if count == 1:
            return self
        n_targets = int(self.row_ptr.shape[0]) - 1
        n_edges = self.n_edges
        device = self.edge_index.device
        offsets = torch.arange(count, device=device, dtype=torch.long).unsqueeze(1)  # (count, 1)
        targets = (self.edge_index[0].unsqueeze(0) + offsets * n_targets).reshape(-1)
        sources = (self.edge_index[1].unsqueeze(0) + offsets * int(n_sources)).reshape(-1)
        row_ptr = torch.cat([self.row_ptr[:1], (self.row_ptr[1:].unsqueeze(0) + offsets * n_edges).reshape(-1)])

        def repeat(table: Tensor) -> Tensor:
            return table.repeat(count, *([1] * (table.ndim - 1)))

        return type(self)(
            edge_index=torch.stack([targets, sources]),
            row_ptr=row_ptr,
            log_coordinates=repeat(self.log_coordinates),
            distances=repeat(self.distances),
            polar_angles=repeat(self.polar_angles),
            transport_angles=repeat(self.transport_angles),
            tangent_quadrature_weights=repeat(self.tangent_quadrature_weights),
            angular_phases=repeat(self.angular_phases),
            transport_phases=repeat(self.transport_phases),
        )

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> EdgeGeometry:
        """Return the edge data on a new device or floating-point dtype."""
        target_dtype = (
            self.distances.dtype
            if dtype is None
            else dtype
        )

        # Phases follow the geometry dtype: float32 geometry carries
        # complex64 tables, float64 geometry complex128.
        target_complex_dtype = complex_dtype(target_dtype)

        return type(self)(
            edge_index=self.edge_index.to(  # (2, n_edges)
                device=device,
                non_blocking=non_blocking,
            ),
            row_ptr=self.row_ptr.to(  # (n_nodes + 1,)
                device=device,
                non_blocking=non_blocking,
            ),
            log_coordinates=self.log_coordinates.to(  # (n_edges, 2)
                device=device,
                dtype=target_dtype,
                non_blocking=non_blocking,
            ),
            distances=self.distances.to(  # (n_edges,)
                device=device,
                dtype=target_dtype,
                non_blocking=non_blocking,
            ),
            polar_angles=self.polar_angles.to(  # (n_edges,)
                device=device,
                dtype=target_dtype,
                non_blocking=non_blocking,
            ),
            transport_angles=self.transport_angles.to(  # (n_edges,)
                device=device,
                dtype=target_dtype,
                non_blocking=non_blocking,
            ),
            tangent_quadrature_weights=(
                self.tangent_quadrature_weights.to(  # (n_edges,)
                    device=device,
                    dtype=target_dtype,
                    non_blocking=non_blocking,
                )
            ),
            angular_phases=self.angular_phases.to(  # (n_edges, K + 1)
                device=device,
                dtype=target_complex_dtype,
                non_blocking=non_blocking,
            ),
            transport_phases=self.transport_phases.to(  # (n_edges, K + 1)
                device=device,
                dtype=target_complex_dtype,
                non_blocking=non_blocking,
            ),
        )


def compute_edge_geometry(
    grid: SphericalGrid,
    frames: Tensor,  # (n_nodes, 3, 2)
    coverage_radius: float,
    *,
    max_frequency: int = DEFAULT_MAX_FREQUENCY,
) -> EdgeGeometry:
    """Compute all directed edge data within a geodesic radius.

    An edge is written as ``i <- j``. Accordingly,

    - ``edge_index[0]`` contains target indices ``i``;
    - ``edge_index[1]`` contains source indices ``j``;
    - log coordinates are expressed in the target frame ``E_i``;
    - transport angles represent transport from ``p_j`` to ``p_i``;
    - quadrature weights approximate Lebesgue integration on ``T_{p_i}S^2``.

    Parameters
    ----------
    grid
        Nodes and spherical quadrature weights.
    frames
        Oriented orthonormal tangent frames of shape
        ``(n_nodes, 3, 2)``.
    coverage_radius
        Maximum geodesic edge distance in radians. Must lie in ``(0, pi)``.
    max_frequency
        Largest harmonic order to tabulate in ``angular_phases`` and
        ``transport_phases``. A layer mapping input types ``I_in`` to output
        types ``I_out`` requires ``max(I_in) + max(I_out)``. The default of
        ``DEFAULT_MAX_FREQUENCY`` accommodates fields of frequency up to two
        on both sides.

    Returns
    -------
    EdgeGeometry
        Precomputed geometry, angular harmonics, and tangent-space quadrature
        weights for every directed edge within ``coverage_radius``.
    """
    coverage_radius = float(coverage_radius)
    validate_frequency(max_frequency)
    nodes = grid.nodes.contiguous()  # (n_nodes, 3)
    spherical_quadrature_weights = grid.quadrature_weights.contiguous()  # (n_nodes,)

    _validate_inputs(
        nodes=nodes,
        frames=frames,
        coverage_radius=coverage_radius,
        spherical_quadrature_weights=spherical_quadrature_weights,
    )

    exact_nodes = exact_membership_nodes(grid)  # (n_nodes, 3) float64 on the CPU, or None
    edge_index = _compute_radius_edges(  # (2, n_edges)
        nodes=nodes if exact_nodes is None else exact_nodes,
        coverage_radius=coverage_radius,
        exact=exact_nodes is not None,
    ).to(nodes.device)

    # The tree query is widened by the slack; the geodesic condition decides
    # membership here, from the float64 lattice with EXACT_SLACK when the grid
    # carries one, else with the dtype-sized slack (tolerant under both rules).
    if exact_nodes is not None:
        distances = exact_geodesic_distances(exact_nodes, edge_index)  # (n_edges,) float64
        within_radius = (distances <= coverage_radius + EXACT_SLACK).to(nodes.device)  # (n_edges,) bool
    else:
        distances = _geodesic_distances(nodes, edge_index)  # (n_edges,)
        within_radius = distances <= coverage_radius + cutoff_tolerance(nodes.dtype)  # (n_edges,) bool
    edge_index = edge_index[:, within_radius]  # (2, n_edges)

    return edge_geometry_from_index(
        nodes=nodes,
        frames=frames,
        spherical_quadrature_weights=spherical_quadrature_weights,
        edge_index=edge_index,
        n_target_nodes=nodes.shape[0],
        max_frequency=max_frequency,
    )


#: Slack of a membership decision from float64 lattice coordinates; see :func:`cutoff_tolerance`.
EXACT_SLACK = 1e-10


def cutoff_tolerance(dtype: torch.dtype) -> float:
    """Absolute slack in a comparison of a stored geodesic distance against a radius.

    A structured grid places whole shells of nodes exactly on a radius, and
    rounding lands each shell node on either side of it. A strict
    ``distance <= radius`` then splits the shell by rounding, differently
    for a node and its image under a symmetry of the grid: the graph itself
    breaks equivariance.

    Membership is therefore decided from the grid's float64 lattice
    coordinates with :data:`EXACT_SLACK` wherever the grid carries them;
    ``EXACT_SLACK`` lies above the rounding of a lattice distance and below
    the gap between distinct lattice distances, so a shell is kept or left
    out whole. The dtype-sized slack returned here is used by the hard
    aggregation window and by the fallback for a grid without exact
    coordinates; it is wide enough that a window evaluated on stored float32
    distances never rejects an edge the exact decision admitted.
    """
    return 1e-10 if dtype == torch.float64 else 1e-5


#: The rule deciding a boundary shell (``GM_CUTOFF_RULE``, read at import); see :func:`cutoff_rule`.
CUTOFF_RULES = ("whole-shell", "legacy")
CUTOFF_RULE: str = os.environ.get("GM_CUTOFF_RULE", "whole-shell")
if CUTOFF_RULE not in CUTOFF_RULES:
    raise ValueError(f"GM_CUTOFF_RULE must be one of {CUTOFF_RULES}, got {CUTOFF_RULE!r}.")


def cutoff_rule() -> str:
    """The cutoff rule in force, one of ``CUTOFF_RULES``; recorded in protocols.

    ``"whole-shell"`` (the default): edge membership is decided from the
    grid's float64 lattice coordinates with ``EXACT_SLACK`` where the grid
    carries them, else by the tolerant comparison :func:`within_cutoff`, in
    every search, radius subset and hard window. ``"legacy"``: the rule the
    paper's shallow-water runs used (the SWE bundles' run.sh export it), so
    that they are evaluated over the same edge sets: radius subsets and the hard window compare
    strictly, the KD-tree queries are widened by a relative 1e-12 in float64
    and the bipartite search applies no geodesic filter. The coverage search
    is tolerant under both rules. ``GM_CUTOFF_RULE`` is read at import.
    """
    return CUTOFF_RULE


def within_cutoff(distances: Tensor, cutoff: float) -> Tensor:  # (...,) bool
    """``distances <= cutoff`` on stored distances, under the rule in force:
    strict under ``"legacy"``, with :func:`cutoff_tolerance` slack under
    ``"whole-shell"``.

    The comparison behind the hard aggregation window, and behind the radius
    subsets and the bipartite search of a grid without float64 lattice
    coordinates. The slack is wide enough that a window evaluated on stored
    float32 distances never rejects an edge the exact decision admitted, so
    an edge set and its window agree on every shell.
    """
    if CUTOFF_RULE == "legacy":
        return distances <= float(cutoff)
    if CUTOFF_RULE != "whole-shell":
        raise ValueError(f"GM_CUTOFF_RULE must be one of {CUTOFF_RULES}, got {CUTOFF_RULE!r}.")
    return distances <= float(cutoff) + cutoff_tolerance(distances.dtype)


def _query_chord_radius(cutoff: float, dtype: torch.dtype, *, exact: bool = False) -> float:
    """The chord radius ``2 sin((cutoff + slack) / 2)`` of a KD-tree query for a
    geodesic ``cutoff``, wide enough that the condition applied afterward
    decides every edge: slack ``EXACT_SLACK`` when ``exact`` (float64 lattice
    coordinates), else the dtype-sized one. Under ``"legacy"`` the chord
    radius ``2 sin(cutoff / 2)`` is instead widened by a relative 1e-12."""
    if CUTOFF_RULE == "legacy":
        return 2.0 * math.sin(cutoff / 2.0) * (1.0 + 1e-12)
    slack = EXACT_SLACK if exact else cutoff_tolerance(dtype)
    return 2.0 * math.sin((cutoff + slack) / 2.0)


def exact_membership_nodes(grid: SphericalGrid) -> Tensor | None:
    """The float64 coordinates edge membership is decided from, or ``None``
    when the grid has none or the legacy rule is in force (which decides
    from the working-dtype nodes)."""
    if CUTOFF_RULE == "legacy":
        return None
    return grid.exact_nodes


def exact_geodesic_distances(exact_nodes: Tensor, edge_index: Tensor) -> Tensor:  # (n_edges,) float64
    """Geodesic distances of the pairs in ``edge_index`` from float64 lattice
    coordinates, on the CPU: the numbers a membership decision is made from."""
    return _geodesic_distances(exact_nodes, edge_index.detach().cpu())


def _geodesic_distances(nodes: Tensor, edge_index: Tensor) -> Tensor:  # (n_edges,)
    """``atan2(|p x q|, <p, q>)`` for every edge -- more stable than ``acos``
    near coincident points."""
    target_nodes = nodes[edge_index[0]]  # (n_edges, 3)
    source_nodes = nodes[edge_index[1]]  # (n_edges, 3)

    dot_products = torch.sum(target_nodes * source_nodes, dim=-1).clamp(min=-1.0, max=1.0)
    cross_norms = torch.linalg.vector_norm(
        torch.linalg.cross(target_nodes, source_nodes, dim=-1), dim=-1
    )

    return torch.atan2(cross_norms, dot_products)  # (n_edges,)


def edge_geometry_from_index(
    nodes: Tensor,  # (n_nodes, 3)
    frames: Tensor,  # (n_nodes, 3, 2)
    spherical_quadrature_weights: Tensor,  # (n_nodes,)
    edge_index: Tensor,  # (2, n_edges), target-major sorted
    n_target_nodes: int,
    max_frequency: int,
) -> EdgeGeometry:
    """Derive the full edge geometry for a given set of directed edges.

    The derivation (log map, parallel transport, quadrature reweighting,
    angular harmonics) does not depend on how ``edge_index`` was decided:
    :func:`compute_edge_geometry` applies it to a radius search, and
    ``IcoDecode`` to each fine midpoint node's two coarse parents.

    ``n_target_nodes`` sizes the returned ``row_ptr`` and therefore the node
    count a consumer infers; it differs from ``nodes.shape[0]`` for a
    bipartite edge set whose targets are a prefix of ``nodes``
    (``IcoEncode``'s coarse nodes). Every target index must be
    ``< n_target_nodes``; this is not checked.
    """
    target_index = edge_index[0]  # (n_edges,)
    source_index = edge_index[1]  # (n_edges,)

    target_nodes = nodes[target_index]  # (n_edges, 3)
    source_nodes = nodes[source_index]  # (n_edges, 3)

    dot_products = torch.sum(  # (n_edges,)
        target_nodes * source_nodes,
        dim=-1,
    ).clamp(min=-1.0, max=1.0)

    distances = _geodesic_distances(nodes, edge_index)  # (n_edges,)

    target_frames = frames[target_index]  # (n_edges, 3, 2)
    source_frames = frames[source_index]  # (n_edges, 3, 2)

    # torch.sinc(r / pi) = sin(r) / r, whose reciprocal r / sin(r) is one at r = 0.
    inverse_exponential_volume_density = torch.reciprocal(  # (n_edges,)
        torch.sinc(distances / math.pi)
    )

    # Ambient logarithmic vector: log_p(q) = [r/sin(r)] [q - cos(r) p].
    ambient_log_vectors = (  # (n_edges, 3)
        inverse_exponential_volume_density.unsqueeze(-1)
        * (
            source_nodes
            - dot_products.unsqueeze(-1) * target_nodes
        )
    )

    # Express log_{p_i}(p_j) in the target frame E_i.
    log_coordinates = torch.bmm(  # (n_edges, 2)
        target_frames.transpose(1, 2),
        ambient_log_vectors.unsqueeze(-1),
    ).squeeze(-1)

    transport_rotations = _parallel_transport_rotations(  # (n_edges, 3, 3)
        source_nodes=source_nodes,
        target_nodes=target_nodes,
        dot_products=dot_products,
    )

    transported_source_frames = torch.bmm(  # (n_edges, 3, 2)
        transport_rotations,
        source_frames,
    )

    # Source-frame to target-frame coordinates: E_i^T P_{p_i <- p_j} E_j.
    coordinate_transport = torch.bmm(  # (n_edges, 2, 2)
        target_frames.transpose(1, 2),
        transported_source_frames,
    )

    # With r_gamma = [[cos(gamma), -sin(gamma)], [sin(gamma), cos(gamma)]],
    # gamma = atan2(r_gamma[1, 0], r_gamma[0, 0]).
    transport_angles = torch.atan2(  # (n_edges,)
        coordinate_transport[:, 1, 0],
        coordinate_transport[:, 0, 0],
    )

    # Spherical-area weights to tangent-plane Lebesgue weights: du = [r/sin(r)] dVol.
    tangent_quadrature_weights = (  # (n_edges,)
        spherical_quadrature_weights[source_index]
        * inverse_exponential_volume_density
    )

    # Polar angle of log_{p_i}(p_j) in the target frame. A self-loop has zero log
    # coordinates and phi = 0; the origin is fixed by all of SO(2), and a steerable
    # kernel handles it through its intertwiner branch, not the angular harmonics.
    polar_angles = torch.atan2(  # (n_edges,)
        log_coordinates[:, 1],
        log_coordinates[:, 0],
    )

    return EdgeGeometry(
        edge_index=edge_index.contiguous(),  # (2, n_edges)
        row_ptr=_build_row_ptr(target_index, n_target_nodes),  # (n_target_nodes + 1,)
        log_coordinates=log_coordinates.contiguous(),  # (n_edges, 2)
        distances=distances.contiguous(),  # (n_edges,)
        polar_angles=polar_angles.contiguous(),  # (n_edges,)
        transport_angles=transport_angles.contiguous(),  # (n_edges,)
        tangent_quadrature_weights=(
            tangent_quadrature_weights.contiguous()  # (n_edges,)
        ),
        angular_phases=phase_powers(  # (n_edges, max_frequency + 1)
            polar_angles,
            max_frequency,
        ).contiguous(),
        transport_phases=phase_powers(  # (n_edges, max_frequency + 1)
            transport_angles,
            max_frequency,
        ).contiguous(),
    )


def _build_row_ptr(
    target_index: Tensor,  # (n_edges,), sorted non-decreasing
    n_nodes: int,
) -> Tensor:
    """Build CSR-style row pointers from target-major sorted edges.

    ``target_index`` must be sorted in non-decreasing order, so that every
    node's incident edges form a contiguous block. Node ``i``'s edges then
    occupy ``[row_ptr[i], row_ptr[i + 1])``.
    """
    if target_index.numel() > 1 and not bool(
        torch.all(target_index[1:] >= target_index[:-1])
    ):
        raise ValueError(
            "target_index must be sorted in non-decreasing order to build "
            "CSR row pointers."
        )

    counts = torch.bincount(  # (n_nodes,)
        target_index,
        minlength=n_nodes,
    )

    row_ptr = torch.zeros(  # (n_nodes + 1,)
        n_nodes + 1,
        dtype=torch.long,
        device=target_index.device,
    )

    torch.cumsum(counts, dim=0, out=row_ptr[1:])  # row_ptr[1:] : (n_nodes,)

    return row_ptr


def _compute_radius_edges(
    nodes: Tensor,  # (n_nodes, 3)
    coverage_radius: float,
    *,
    exact: bool = False,
) -> Tensor:
    """Find directed node pairs inside a geodesic radius (``exact``: ``nodes``
    are the float64 lattice and the query carries ``EXACT_SLACK``)."""

    # float64 CPU coordinates make the search independent of the training device.
    cpu_nodes = (  # (n_nodes, 3), numpy
        nodes.detach()
        .to(device="cpu", dtype=torch.float64)
        .numpy()
    )

    tree = cKDTree(cpu_nodes)

    # The query radius carries the cutoff slack in the nodes' own precision; the
    # geodesic condition is applied by the caller.
    query_radius = _query_chord_radius(coverage_radius, nodes.dtype, exact=exact)

    neighbor_lists = tree.query_ball_point(  # length n_nodes, ragged
        cpu_nodes,
        r=query_radius,
        workers=-1,
    )

    counts = np.asarray(  # (n_nodes,), numpy
        [len(neighbors) for neighbors in neighbor_lists],
        dtype=np.int64,
    )

    target_indices = np.repeat(  # (n_edges,), numpy
        np.arange(nodes.shape[0], dtype=np.int64),
        counts,
    )

    source_indices = np.concatenate(  # (n_edges,), numpy
        [
            np.asarray(neighbors, dtype=np.int64)
            for neighbors in neighbor_lists
        ]
    )

    # Deterministic target-major, source-minor order.
    order = np.lexsort((source_indices, target_indices))  # (n_edges,), numpy
    target_indices = target_indices[order]  # (n_edges,), numpy
    source_indices = source_indices[order]  # (n_edges,), numpy

    edge_index = np.stack(  # (2, n_edges), numpy
        [target_indices, source_indices],
        axis=0,
    )

    return torch.as_tensor(  # (2, n_edges)
        edge_index,
        dtype=torch.long,
        device=nodes.device,
    )


def _parallel_transport_rotations(
    source_nodes: Tensor,  # (n_edges, 3)
    target_nodes: Tensor,  # (n_edges, 3)
    dot_products: Tensor,  # (n_edges,)
) -> Tensor:
    """Return rotations transporting tangent vectors source -> target.

    For source ``q`` and target ``p``, the minimal rotation mapping ``q`` to
    ``p`` is

        R = I + K + K^2 / (1 + <q, p>),

    where ``K`` is the cross-product matrix of ``q x p``.
    """
    cross_vectors = torch.linalg.cross(  # (n_edges, 3)
        source_nodes,
        target_nodes,
        dim=-1,
    )

    n_edges = source_nodes.shape[0]

    skew = torch.zeros(  # (n_edges, 3, 3)
        (n_edges, 3, 3),
        dtype=source_nodes.dtype,
        device=source_nodes.device,
    )

    x = cross_vectors[:, 0]  # (n_edges,)
    y = cross_vectors[:, 1]  # (n_edges,)
    z = cross_vectors[:, 2]  # (n_edges,)

    skew[:, 0, 1] = -z
    skew[:, 0, 2] = y
    skew[:, 1, 0] = z
    skew[:, 1, 2] = -x
    skew[:, 2, 0] = -y
    skew[:, 2, 1] = x

    identity = torch.eye(  # (n_edges, 3, 3)
        3,
        dtype=source_nodes.dtype,
        device=source_nodes.device,
    ).expand(n_edges, -1, -1)

    denominator = (  # (n_edges, 1, 1)
        1.0 + dot_products
    ).reshape(-1, 1, 1)

    return (  # (n_edges, 3, 3)
        identity
        + skew
        + torch.bmm(skew, skew) / denominator
    )


def _validate_inputs(
    nodes: Tensor,  # (n_nodes, 3)
    frames: Tensor,  # (n_nodes, 3, 2)
    coverage_radius: float,
    spherical_quadrature_weights: Tensor,  # (n_nodes,)
) -> None:
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError(
            "nodes must have shape (n_nodes, 3), "
            f"got {tuple(nodes.shape)}."
        )

    if frames.shape != (nodes.shape[0], 3, 2):
        raise ValueError(
            "frames must have shape (n_nodes, 3, 2), "
            f"got {tuple(frames.shape)}."
        )

    if spherical_quadrature_weights.shape != (
        nodes.shape[0],
    ):
        raise ValueError(
            "spherical_quadrature_weights must have shape "
            f"({nodes.shape[0]},), got "
            f"{tuple(spherical_quadrature_weights.shape)}."
        )

    if not math.isfinite(coverage_radius):
        raise ValueError("coverage_radius must be finite.")

    if not 0.0 < coverage_radius < math.pi:
        raise ValueError(
            "coverage_radius must lie strictly between 0 and pi."
        )

    if nodes.device != frames.device:
        raise ValueError(
            "nodes and frames must be on the same device."
        )

    if nodes.device != spherical_quadrature_weights.device:
        raise ValueError(
            "nodes and quadrature weights must be on the same device."
        )

    if nodes.dtype != frames.dtype:
        raise ValueError(
            "nodes and frames must have the same dtype."
        )

    if nodes.dtype != spherical_quadrature_weights.dtype:
        raise ValueError(
            "nodes and quadrature weights must have the same dtype."
        )


def compute_bipartite_edge_geometry(
    source_nodes: Tensor,  # (n_source, 3)
    source_frames: Tensor,  # (n_source, 3, 2)
    source_spherical_quadrature_weights: Tensor,  # (n_source,)
    target_nodes: Tensor,  # (n_target, 3)
    target_frames: Tensor,  # (n_target, 3, 2)
    *,
    cutoff: float,
    max_frequency: int,
    source_exact_nodes: Tensor | None = None,  # (n_source, 3) float64
    target_exact_nodes: Tensor | None = None,  # (n_target, 3) float64
) -> EdgeGeometry:
    """Edge geometry between two independent grids, target ``i`` <- source ``j``.

    Unlike :func:`compute_edge_geometry`, which reads targets and sources
    out of one node array, the two grids share no indexing (an equiangular
    grid restricted onto a coarser one). The returned geometry indexes each
    side in its own local numbering, ``target_index`` into ``target_nodes``
    and ``source_index`` into ``source_nodes``, with ``row_ptr`` sized
    ``n_target + 1``; source features are gathered with
    ``x[edges.source_index]`` as for a same-grid convolution.

    Quadrature weights come from the source side, the side integrated over.
    ``target_frames`` fixes the gauge every log coordinate and transport
    angle is expressed in, so the two frame arrays may be regauged
    independently and the result stays covariant.

    Implemented by concatenating the two grids, delegating to
    :func:`edge_geometry_from_index`, and rewriting ``source_index`` back
    to source-local numbering.

    Parameters
    ----------
    source_nodes, source_frames, source_spherical_quadrature_weights
        The grid being integrated over.
    target_nodes, target_frames
        The grid being integrated onto; ``target_frames`` sets the output
        gauge.
    cutoff
        Geodesic radius. Every target must have at least one source inside
        it; an uncovered target would aggregate to zero.
    max_frequency
        Highest angular/transport order to tabulate.
    source_exact_nodes, target_exact_nodes
        The two grids' float64 lattice coordinates (``SphericalGrid.exact_nodes``);
        when both are given, membership is decided from them with
        ``EXACT_SLACK``, else from the working-dtype nodes with the
        dtype-sized slack.

    Raises
    ------
    ValueError
        If any target has no source within ``cutoff``.
    """
    if cutoff <= 0.0:
        raise ValueError(f"cutoff must be positive, got {cutoff}.")

    n_target = target_nodes.shape[0]
    exact = (
        CUTOFF_RULE != "legacy"
        and source_exact_nodes is not None and target_exact_nodes is not None
    )
    if exact:
        assert source_exact_nodes is not None and target_exact_nodes is not None
        source_search, target_search = source_exact_nodes, target_exact_nodes
    else:
        source_search, target_search = source_nodes, target_nodes

    edge_index = _compute_bipartite_radius_edges(
        source_nodes=source_search, target_nodes=target_search, cutoff=cutoff, exact=exact
    ).to(source_nodes.device)

    # The tree query is widened by the slack; the geodesic condition (from the
    # float64 lattice when both grids carry one) decides membership here.
    if CUTOFF_RULE != "legacy":  # the legacy rule applies no geodesic filter
        if exact:
            distances = exact_geodesic_distances(
                torch.cat([target_search, source_search], dim=0),
                torch.stack([edge_index[0], edge_index[1] + n_target], dim=0),
            )  # (n_edges,) float64
            keep = (distances <= cutoff + EXACT_SLACK).to(edge_index.device)
        else:
            distances = _geodesic_distances(
                torch.cat([target_nodes, source_nodes], dim=0),
                torch.stack([edge_index[0], edge_index[1] + n_target], dim=0),
            )  # (n_edges,)
            keep = within_cutoff(distances, cutoff)
        edge_index = edge_index[:, keep]  # (2, n_edges)

    covered = torch.zeros(n_target, dtype=torch.bool, device=edge_index.device)
    covered[edge_index[0]] = True
    if not bool(covered.all()):
        missing = int((~covered).sum())
        raise ValueError(
            f"{missing} of {n_target} target nodes have no source within "
            f"cutoff={cutoff}. Widen the cutoff: an uncovered target "
            "aggregates to zero, which is indistinguishable from a genuine "
            "zero downstream."
        )

    # One combined array, targets first; sources are offset by n_target.
    nodes = torch.cat([target_nodes, source_nodes], dim=0)
    frames = torch.cat([target_frames, source_frames], dim=0)
    # Target weights are never read (weights are gathered by source index);
    # zeros align the array with `nodes` and make an accidental read detectable.
    weights = torch.cat(
        [
            torch.zeros(
                n_target,
                dtype=source_spherical_quadrature_weights.dtype,
                device=source_spherical_quadrature_weights.device,
            ),
            source_spherical_quadrature_weights,
        ],
        dim=0,
    )

    combined = edge_geometry_from_index(
        nodes,
        frames,
        weights,
        torch.stack([edge_index[0], edge_index[1] + n_target], dim=0),
        n_target,
        max_frequency,
    )

    return replace(
        combined,
        edge_index=torch.stack(
            [combined.edge_index[0], combined.edge_index[1] - n_target], dim=0
        ).contiguous(),
    )


def _compute_bipartite_radius_edges(
    source_nodes: Tensor,  # (n_source, 3)
    target_nodes: Tensor,  # (n_target, 3)
    cutoff: float,
    *,
    exact: bool = False,
) -> Tensor:
    """Directed target <- source pairs inside a geodesic radius, target-major:
    the cross-grid counterpart of :func:`_compute_radius_edges`, searched in
    float64 on the CPU."""
    cpu_source = source_nodes.detach().to(device="cpu", dtype=torch.float64).numpy()
    cpu_target = target_nodes.detach().to(device="cpu", dtype=torch.float64).numpy()

    tree = cKDTree(cpu_source)

    # Chord radius for the cutoff plus the slack, in the nodes' own precision;
    # the geodesic condition is applied by the caller.
    query_radius = _query_chord_radius(cutoff, source_nodes.dtype, exact=exact)

    neighbor_lists = tree.query_ball_point(cpu_target, r=query_radius, workers=-1)

    counts = np.asarray([len(n) for n in neighbor_lists], dtype=np.int64)
    target_indices = np.repeat(
        np.arange(cpu_target.shape[0], dtype=np.int64), counts
    )
    source_indices = (
        np.concatenate([np.asarray(n, dtype=np.int64) for n in neighbor_lists])
        if counts.sum()
        else np.zeros(0, dtype=np.int64)
    )

    order = np.lexsort((source_indices, target_indices))
    edge_index = np.stack([target_indices[order], source_indices[order]], axis=0)

    return torch.as_tensor(edge_index, dtype=torch.long, device=source_nodes.device)
