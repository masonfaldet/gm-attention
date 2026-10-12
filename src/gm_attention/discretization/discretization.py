from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import Tensor

from gm_attention.discretization.edge_geometry import (
    DEFAULT_MAX_FREQUENCY,
    EXACT_SLACK,
    EdgeGeometry,
    compute_edge_geometry,
    cutoff_tolerance,
    exact_geodesic_distances,
    exact_membership_nodes,
    within_cutoff,
)
from gm_attention.discretization.frames import build_tangent_frames
from gm_attention.discretization.grid import SphericalGrid
from gm_attention.representations.so2 import complex_dtype


class SphericalDiscretization:
    """Complete fixed geometry used by GM-aggregation layers.

    The discretization stores geometry for all directed node pairs within a
    maximum coverage radius. Individual layers query subsets corresponding to
    their aggregation-window support radii.

    Parameters
    ----------
    grid
        Nodes and spherical quadrature weights.
    frames
        Oriented orthonormal tangent frames with shape
        ``(n_nodes, 3, 2)``: the frames ``edge_geometry`` was computed
        against.
    edge_geometry
        Instance of ``EdgeGeometry`` containing:

        - edge indices with shape ``(2, n_edges)``, sorted target-major,
          source-minor;
        - CSR-style row pointers with shape ``(n_nodes + 1,)``, so that node
          ``i``'s incident edges occupy ``[row_ptr[i], row_ptr[i + 1])``;
        - log coordinates with shape ``(n_edges, 2)``;
        - induced tangent-space quadrature weights with shape
          ``(n_edges,)``;
        - geodesic distances with shape ``(n_edges,)``;
        - parallel-transport angles with shape ``(n_edges,)``.

        For an edge ``i <- j``, the log coordinates represent
        ``log_{p_i}(p_j)`` in the target frame at ``p_i``. The transport angle
        represents parallel transport from the source frame at ``p_j`` to the
        target frame at ``p_i``.
    coverage_radius
        Maximum geodesic radius covered by ``edge_geometry``. It must be
        strictly smaller than ``pi`` to avoid antipodal ambiguity.
    exact_distances
        Optional geodesic distances of the coverage edges computed from the
        grid's float64 lattice coordinates, shape ``(n_edges,)``. When given,
        :meth:`edges_within` decides each radius subset from them with
        ``EXACT_SLACK``; otherwise from the stored distances with
        :func:`within_cutoff`.

    Warning
    -------
    ``log_coordinates`` and ``transport_angles`` are expressed in the gauge
    of the ``frames`` from which ``edge_geometry`` was computed. ``_validate``
    checks that ``frames`` is orthonormal and positively oriented and that
    ``edge_geometry`` is internally consistent (index ranges, ``row_ptr``,
    distances within ``coverage_radius``); it does not check that the two
    were computed in the same gauge. :meth:`from_grid` builds both together.
    """

    __slots__ = (
        "_exact_distances",
        "_radius_cache",
        "coverage_radius",
        "edge_geometry",
        "frames",
        "grid",
    )

    def __init__(
        self,
        grid: SphericalGrid,
        frames: Tensor,  # (n_nodes, 3, 2)
        edge_geometry: EdgeGeometry,
        coverage_radius: float,
        exact_distances: Tensor | None = None,  # (n_edges,) float64 on the CPU
    ) -> None:
        coverage_radius = float(coverage_radius)

        self._validate(
            grid=grid,
            frames=frames,
            edge_geometry=edge_geometry,
            coverage_radius=coverage_radius,
        )

        self.grid = grid
        self.frames = frames.contiguous()  # (n_nodes, 3, 2)
        self.edge_geometry = edge_geometry
        self.coverage_radius = coverage_radius
        if exact_distances is not None and exact_distances.shape != (edge_geometry.n_edges,):
            raise ValueError(
                f"exact_distances must have shape ({edge_geometry.n_edges},), got {tuple(exact_distances.shape)}."
            )
        self._exact_distances = (
            None if exact_distances is None else exact_distances.detach().to(device="cpu", dtype=torch.float64)
        )

        self._radius_cache: dict[float, EdgeGeometry] = {
            self._radius_key(coverage_radius): edge_geometry
        }

    @classmethod
    def from_grid(
        cls,
        grid: SphericalGrid,
        coverage_radius: float,
        *,
        frames: Tensor | None = None,  # (n_nodes, 3, 2)
        max_frequency: int = DEFAULT_MAX_FREQUENCY,
    ) -> SphericalDiscretization:
        """Construct a discretization from a quadrature grid.

        Parameters
        ----------
        grid
            Spherical nodes and quadrature weights.
        coverage_radius
            Maximum support radius required by any GM layer.
        frames
            Optional tangent frames. If omitted, :func:`build_tangent_frames`
            supplies them.
        max_frequency
            Largest harmonic order to tabulate. This must be at least
            ``max(I_in) + max(I_out)`` over every layer that will consume this
            discretization; layers check it and raise otherwise. The default
            of ``DEFAULT_MAX_FREQUENCY`` accommodates fields of frequency up
            to two on both sides.
        """
        coverage_radius = float(coverage_radius)
        cls._validate_radius(coverage_radius)

        if frames is None:
            frames = build_tangent_frames(grid.nodes)  # (n_nodes, 3, 2)

        edge_geometry = compute_edge_geometry(
            grid=grid,
            frames=frames,
            coverage_radius=coverage_radius,
            max_frequency=max_frequency,
        )
        exact_nodes = exact_membership_nodes(grid)

        return cls(
            grid=grid,
            frames=frames,
            edge_geometry=edge_geometry,
            coverage_radius=coverage_radius,
            exact_distances=(
                None if exact_nodes is None
                else exact_geodesic_distances(exact_nodes, edge_geometry.edge_index)
            ),
        )

    @staticmethod
    def _radius_key(radius: float) -> float:
        """Return a stable dictionary key for a configured radius."""
        return round(float(radius), 12)

    @staticmethod
    def _validate_radius(radius: float) -> None:
        if not math.isfinite(radius):
            raise ValueError(
                f"Support radii must be finite, got {radius}."
            )

        if radius <= 0.0:
            raise ValueError(
                f"Support radii must be positive, got {radius}."
            )

        if radius >= math.pi:
            raise ValueError(
                "Support radii must be strictly smaller than pi because "
                "logarithmic coordinates and shortest-geodesic parallel "
                "transport are ambiguous at antipodal points."
            )

    @classmethod
    def _validate(
        cls,
        grid: SphericalGrid,
        frames: Tensor,  # (n_nodes, 3, 2)
        edge_geometry: EdgeGeometry,
        coverage_radius: float,
    ) -> None:
        cls._validate_radius(coverage_radius)

        if not isinstance(grid, SphericalGrid):
            raise TypeError(
                "grid must be a SphericalGrid, "
                f"got {type(grid).__name__}."
            )

        if not isinstance(frames, Tensor):
            raise TypeError(
                "frames must be a torch.Tensor, "
                f"got {type(frames).__name__}."
            )

        expected_frame_shape = (grid.n_nodes, 3, 2)

        if tuple(frames.shape) != expected_frame_shape:
            raise ValueError(
                f"frames must have shape {expected_frame_shape}, "
                f"got {tuple(frames.shape)}."
            )

        if frames.device != grid.device:
            raise ValueError(
                "frames and grid tensors must be on the same device, "
                f"got {frames.device} and {grid.device}."
            )

        if frames.dtype != grid.dtype:
            raise ValueError(
                "frames and grid tensors must have the same dtype, "
                f"got {frames.dtype} and {grid.dtype}."
            )

        if not bool(torch.isfinite(frames).all()):
            raise ValueError("frames contains nonfinite values.")

        cls._validate_oriented_orthonormal_frames(
            nodes=grid.nodes,
            frames=frames,
        )

        edge_index = edge_geometry.edge_index  # (2, n_edges)

        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                "edge_geometry.edge_index must have shape (2, n_edges), "
                f"got {tuple(edge_index.shape)}."
            )

        if edge_index.dtype != torch.long:
            raise TypeError(
                "edge_geometry.edge_index must have dtype torch.long, "
                f"got {edge_index.dtype}."
            )

        if edge_index.device != grid.device:
            raise ValueError(
                "edge geometry and grid tensors must be on the same device."
            )

        if edge_index.numel() > 0:
            minimum_index = edge_index.min().item()
            maximum_index = edge_index.max().item()

            if minimum_index < 0 or maximum_index >= grid.n_nodes:
                raise ValueError(
                    "edge_index contains a node index outside the valid "
                    f"range [0, {grid.n_nodes})."
                )

        row_ptr = edge_geometry.row_ptr  # (n_nodes + 1,)
        expected_row_ptr_shape = (grid.n_nodes + 1,)

        if tuple(row_ptr.shape) != expected_row_ptr_shape:
            raise ValueError(
                "edge_geometry.row_ptr must have shape "
                f"{expected_row_ptr_shape}, got {tuple(row_ptr.shape)}."
            )

        if row_ptr.dtype != torch.long:
            raise TypeError(
                "edge_geometry.row_ptr must have dtype torch.long, "
                f"got {row_ptr.dtype}."
            )

        if row_ptr.device != grid.device:
            raise ValueError(
                "edge_geometry.row_ptr and grid tensors must be on the "
                "same device."
            )

        valid_row_ptr = bool(
            row_ptr[0].item() == 0
            and row_ptr[-1].item() == edge_geometry.n_edges
            and torch.all(row_ptr[1:] >= row_ptr[:-1])
        )

        if not valid_row_ptr:
            raise ValueError(
                "edge_geometry.row_ptr must be non-decreasing, start at 0, "
                "and end at n_edges."
            )

        expected_complex_dtype = complex_dtype(grid.dtype)

        for name in ("angular_phases", "transport_phases"):
            table = getattr(edge_geometry, name)  # (n_edges, K + 1) complex

            if table.ndim != 2 or table.shape[0] != edge_geometry.n_edges:
                raise ValueError(
                    f"edge_geometry.{name} must have shape "
                    f"({edge_geometry.n_edges}, max_frequency + 1), got "
                    f"{tuple(table.shape)}."
                )

            if table.shape[1] == 0:
                raise ValueError(
                    f"edge_geometry.{name} must tabulate at least the "
                    "zeroth harmonic."
                )

            if table.dtype != expected_complex_dtype:
                raise TypeError(
                    f"edge_geometry.{name} must have dtype "
                    f"{expected_complex_dtype} to match the grid dtype "
                    f"{grid.dtype}, got {table.dtype}."
                )

            if table.device != grid.device:
                raise ValueError(
                    f"edge_geometry.{name} and grid tensors must be on the "
                    "same device."
                )

        if (
            edge_geometry.angular_phases.shape[1]
            != edge_geometry.transport_phases.shape[1]
        ):
            raise ValueError(
                "edge_geometry.angular_phases and "
                "edge_geometry.transport_phases must tabulate the same "
                "number of harmonic orders, got "
                f"{edge_geometry.angular_phases.shape[1]} and "
                f"{edge_geometry.transport_phases.shape[1]}."
            )

        if edge_geometry.distances.numel() > 0:
            tolerance = cutoff_tolerance(grid.dtype)
            maximum_distance = edge_geometry.distances.max().item()

            if maximum_distance > coverage_radius + tolerance:
                raise ValueError(
                    "edge_geometry contains an edge outside the declared "
                    f"coverage radius: maximum distance "
                    f"{maximum_distance:.12g}, coverage radius "
                    f"{coverage_radius:.12g}."
                )

    @staticmethod
    def _validate_oriented_orthonormal_frames(
        nodes: Tensor,  # (n_nodes, 3)
        frames: Tensor,  # (n_nodes, 3, 2)
    ) -> None:
        tolerance = 1e-10 if nodes.dtype == torch.float64 else 1e-5

        gram_matrices = torch.matmul(  # (n_nodes, 2, 2)
            frames.transpose(-1, -2),
            frames,
        )

        identity = torch.eye(  # (n_nodes, 2, 2)
            2,
            dtype=frames.dtype,
            device=frames.device,
        ).expand(frames.shape[0], -1, -1)

        if not bool(
            torch.allclose(
                gram_matrices,
                identity,
                atol=tolerance,
                rtol=0.0,
            )
        ):
            raise ValueError(
                "The columns of every frame must be orthonormal."
            )

        first_vectors = frames[:, :, 0]  # (n_nodes, 3)
        second_vectors = frames[:, :, 1]  # (n_nodes, 3)

        frame_normals = torch.linalg.cross(  # (n_nodes, 3)
            first_vectors,
            second_vectors,
            dim=-1,
        )

        if not bool(
            torch.allclose(
                frame_normals,
                nodes,
                atol=tolerance,
                rtol=0.0,
            )
        ):
            raise ValueError(
                "Every frame must be positively oriented: "
                "cross(e_1, e_2) must equal the corresponding sphere node."
            )

    @property
    def nodes(self) -> Tensor:
        return self.grid.nodes  # (n_nodes, 3)

    @property
    def quadrature_weights(self) -> Tensor:
        return self.grid.quadrature_weights  # (n_nodes,)

    @property
    def n_nodes(self) -> int:
        return self.grid.n_nodes

    @property
    def n_edges(self) -> int:
        return self.edge_geometry.n_edges

    @property
    def device(self) -> torch.device:
        return self.grid.device

    @property
    def dtype(self) -> torch.dtype:
        return self.grid.dtype

    @property
    def max_frequency(self) -> int:
        """Largest harmonic order tabulated on the edge geometry."""
        return self.edge_geometry.max_frequency

    @property
    def cached_radii(self) -> tuple[float, ...]:
        """Support radii with materialized edge-geometry subsets."""
        return tuple(sorted(self._radius_cache))

    def edges_within(
        self,
        radius: float,
    ) -> EdgeGeometry:
        """Return the geometry of the edges within a support radius.

        Membership is decided from the grid's float64 lattice with
        ``EXACT_SLACK`` when the grid carries one, else by
        :func:`within_cutoff` on the stored distances, so that a shell of
        nodes exactly on the radius is kept whole. The subset is materialized
        and cached on the first request for a radius; :meth:`prepare_windows`
        materializes every window before training.
        """
        radius = float(radius)
        self._validate_radius(radius)

        tolerance = 1e-12

        if radius > self.coverage_radius + tolerance:
            raise ValueError(
                f"Requested radius {radius:.12g} exceeds the "
                f"discretization coverage radius "
                f"{self.coverage_radius:.12g}."
            )

        key = self._radius_key(radius)

        if key not in self._radius_cache:
            if self._exact_distances is not None:
                mask = (self._exact_distances <= radius + EXACT_SLACK).to(self.edge_geometry.distances.device)
            else:
                mask = within_cutoff(self.edge_geometry.distances, radius)  # (n_edges,) bool
            self._radius_cache[key] = self.edge_geometry.select(mask)

        return self._radius_cache[key]

    def prepare_windows(
        self,
        support_radii: Iterable[float],
    ) -> None:
        """Materialize the edge subset of every support radius.

        Called before training or ``torch.compile``, so that no subset is
        decided inside a forward pass.
        """
        for radius in support_radii:
            self.edges_within(radius)

    def tile(self, count: int) -> SphericalDiscretization:
        """The disjoint union of ``count`` copies: ``count`` samples on the
        same grid as one graph with no edge between copies, so that a layer
        aggregates each sample within its own copy. The grid, frames and
        coverage edges are tiled (:meth:`SphericalGrid.tile`,
        :meth:`EdgeGeometry.tile`), and every cached radius subset is tiled
        into the union's cache, so no membership is re-decided.
        ``count == 1`` returns ``self``."""
        count = int(count)
        if count == 1:
            return self
        union = type(self)(
            grid=self.grid.tile(count),
            frames=self.frames.repeat(count, 1, 1),  # (count * n_nodes, 3, 2)
            edge_geometry=self.edge_geometry.tile(count, n_sources=self.n_nodes),
            coverage_radius=self.coverage_radius,
            exact_distances=None if self._exact_distances is None else self._exact_distances.repeat(count),
        )
        coverage_key = self._radius_key(self.coverage_radius)
        for key, edges in self._radius_cache.items():
            if key != coverage_key:
                union._radius_cache[key] = edges.tile(count, n_sources=self.n_nodes)
        return union

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> SphericalDiscretization:
        """Return the complete discretization on a new device or dtype."""
        target_dtype = self.dtype if dtype is None else dtype

        moved = type(self)(
            grid=self.grid.to(
                device=device,
                dtype=target_dtype,
                non_blocking=non_blocking,
            ),
            frames=self.frames.to(  # (n_nodes, 3, 2)
                device=device,
                dtype=target_dtype,
                non_blocking=non_blocking,
            ),
            edge_geometry=self.edge_geometry.to(
                device=device,
                dtype=target_dtype,
                non_blocking=non_blocking,
            ),
            coverage_radius=self.coverage_radius,
            exact_distances=self._exact_distances,
        )

        coverage_key = self._radius_key(self.coverage_radius)

        for key, edges in self._radius_cache.items():
            if key == coverage_key:
                continue

            moved._radius_cache[key] = edges.to(
                device=device,
                dtype=target_dtype,
                non_blocking=non_blocking,
            )

        return moved

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"n_nodes={self.n_nodes}, "
            f"n_edges={self.n_edges}, "
            f"coverage_radius={self.coverage_radius:.6g}, "
            f"cached_radii={self.cached_radii}, "
            f"dtype={self.dtype}, "
            f"device='{self.device}')"
        )
