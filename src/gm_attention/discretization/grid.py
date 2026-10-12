from __future__ import annotations

import math

import torch
from scipy.spatial import cKDTree
from torch import Tensor


class SphericalGrid:
    """Nodes and quadrature weights for a discretization of the unit sphere.

    Parameters
    ----------
    nodes
        Shape ``(n_nodes, 3)``. ``nodes[i]`` is the ambient Cartesian
        position of node ``i`` on the unit sphere, so ``|nodes[i]| = 1``.
    quadrature_weights
        Shape ``(n_nodes,)``. ``quadrature_weights[i]`` is the spherical
        area associated with node ``i``, so that
        ``sum_i quadrature_weights[i] * f(nodes[i])`` approximates
        ``integral_{S^2} f dA``. Positive, and summing to ``4 * pi``.
    exact_nodes
        Optional float64 coordinates of the same nodes, as the grid builder
        computed them before the cast to the working dtype; stored on the
        CPU. Edge membership is decided from these when present, so that a
        shell of nodes exactly on a radius is decided by the lattice rather
        than by rounding in the working dtype. A grid without them falls
        back to a dtype-sized tolerance.
    """

    __slots__ = ("exact_nodes", "nodes", "quadrature_weights")

    def __init__(
        self,
        nodes: Tensor,  # (n_nodes, 3)
        quadrature_weights: Tensor,  # (n_nodes,)
        *,
        exact_nodes: Tensor | None = None,  # (n_nodes, 3), float64
    ) -> None:
        self._validate(nodes, quadrature_weights)

        self.nodes = nodes.contiguous()  # (n_nodes, 3)
        self.quadrature_weights = quadrature_weights.contiguous()  # (n_nodes,)
        if exact_nodes is not None:
            if exact_nodes.shape != nodes.shape:
                raise ValueError(f"exact_nodes must have shape {tuple(nodes.shape)}, got {tuple(exact_nodes.shape)}.")
            exact_nodes = exact_nodes.detach().to(device="cpu", dtype=torch.float64).contiguous()
        self.exact_nodes = exact_nodes  # (n_nodes, 3) float64 on the CPU, or None

    @staticmethod
    def _validate(
        nodes: Tensor,  # (n_nodes, 3)
        quadrature_weights: Tensor,  # (n_nodes,)
    ) -> None:
        if not isinstance(nodes, Tensor):
            raise TypeError(
                f"nodes must be a torch.Tensor, got {type(nodes).__name__}."
            )

        if not isinstance(quadrature_weights, Tensor):
            raise TypeError(
                "quadrature_weights must be a torch.Tensor, "
                f"got {type(quadrature_weights).__name__}."
            )

        if nodes.ndim != 2 or nodes.shape[1] != 3:
            raise ValueError(
                "nodes must have shape (n_nodes, 3), "
                f"got {tuple(nodes.shape)}."
            )

        if nodes.shape[0] == 0:
            raise ValueError("A spherical grid must contain at least one node.")

        if quadrature_weights.ndim != 1:
            raise ValueError(
                "quadrature_weights must have shape (n_nodes,), "
                f"got {tuple(quadrature_weights.shape)}."
            )

        if quadrature_weights.shape[0] != nodes.shape[0]:
            raise ValueError(
                "The number of quadrature weights must equal the number of "
                f"nodes, got {quadrature_weights.shape[0]} weights and "
                f"{nodes.shape[0]} nodes."
            )

        if nodes.device != quadrature_weights.device:
            raise ValueError(
                "nodes and quadrature_weights must be on the same device, "
                f"got {nodes.device} and {quadrature_weights.device}."
            )

        if nodes.dtype != quadrature_weights.dtype:
            raise ValueError(
                "nodes and quadrature_weights must have the same dtype, "
                f"got {nodes.dtype} and {quadrature_weights.dtype}."
            )

        if nodes.dtype not in (torch.float32, torch.float64):
            raise TypeError(
                "Spherical-grid geometry must use torch.float32 or "
                f"torch.float64, got {nodes.dtype}."
            )

        if not bool(torch.isfinite(nodes).all()):
            raise ValueError("nodes contains nonfinite values.")

        if not bool(torch.isfinite(quadrature_weights).all()):
            raise ValueError("quadrature_weights contains nonfinite values.")

        if not bool((quadrature_weights > 0).all()):
            raise ValueError("All quadrature weights must be strictly positive.")

        if nodes.dtype == torch.float64:
            norm_atol = 1e-10
            area_atol = 1e-10
            area_rtol = 1e-10
        else:
            norm_atol = 1e-5
            area_atol = 1e-5
            area_rtol = 1e-4

        node_norms = torch.linalg.vector_norm(nodes, dim=-1)  # (n_nodes,)
        unit_norms = torch.ones_like(node_norms)  # (n_nodes,)

        if not bool(
            torch.allclose(
                node_norms,
                unit_norms,
                atol=norm_atol,
                rtol=0.0,
            )
        ):
            maximum_error = torch.max(torch.abs(node_norms - 1)).item()
            raise ValueError(
                "Every node must lie on the unit sphere. "
                f"Maximum norm error: {maximum_error:.3e}."
            )

        total_weight = quadrature_weights.sum()  # scalar
        sphere_area = quadrature_weights.new_tensor(4.0 * math.pi)  # scalar

        if not bool(
            torch.isclose(
                total_weight,
                sphere_area,
                atol=area_atol,
                rtol=area_rtol,
            )
        ):
            raise ValueError(
                "Quadrature weights must sum approximately to 4*pi for the "
                f"unit sphere, got {total_weight.item():.12g}."
            )

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the grid."""
        return self.nodes.shape[0]

    @property
    def device(self) -> torch.device:
        """Device on which the grid tensors are stored."""
        return self.nodes.device

    @property
    def dtype(self) -> torch.dtype:
        """Floating-point dtype used by the grid."""
        return self.nodes.dtype

    @property
    def total_quadrature_weight(self) -> Tensor:
        """Sum of the quadrature weights."""
        return self.quadrature_weights.sum()  # scalar

    def __len__(self) -> int:
        return self.n_nodes

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> SphericalGrid:
        """Return a copy of the grid on the requested device and dtype."""
        target_dtype = self.dtype if dtype is None else dtype

        return type(self)(
            nodes=self.nodes.to(  # (n_nodes, 3)
                device=device,
                dtype=target_dtype,
                non_blocking=non_blocking,
            ),
            quadrature_weights=self.quadrature_weights.to(  # (n_nodes,)
                device=device,
                dtype=target_dtype,
                non_blocking=non_blocking,
            ),
            exact_nodes=self.exact_nodes,
        )

    def tile(self, count: int) -> SphericalGrid:
        """The node set of the disjoint union of ``count`` copies of the grid,
        copy after copy. The quadrature weights are repeated and divided by
        ``count`` so that the union integrates to the sphere area; the
        per-sample measure is carried by the edge geometries' tangent
        weights, which tile unchanged
        (:meth:`~gm_attention.discretization.edge_geometry.EdgeGeometry.tile`).
        ``count == 1`` returns ``self``."""
        count = int(count)
        if count < 1:
            raise ValueError(f"count must be at least 1, got {count}.")
        if count == 1:
            return self
        return type(self)(
            nodes=self.nodes.repeat(count, 1),  # (count * n_nodes, 3)
            quadrature_weights=self.quadrature_weights.repeat(count) / count,  # (count * n_nodes,)
            exact_nodes=None if self.exact_nodes is None else self.exact_nodes.repeat(count, 1),
        )

    def clone(self) -> SphericalGrid:
        """Return a tensor-wise copy of the grid."""
        return type(self)(
            nodes=self.nodes.clone(),  # (n_nodes, 3)
            quadrature_weights=self.quadrature_weights.clone(),  # (n_nodes,)
            exact_nodes=None if self.exact_nodes is None else self.exact_nodes.clone(),
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"n_nodes={self.n_nodes}, "
            f"dtype={self.dtype}, "
            f"device='{self.device}')"
        )


def median_nearest_neighbor_distance(grid: SphericalGrid) -> float:
    """Median geodesic distance from each node to its nearest neighbor.

    A grid-derived estimate of the node spacing, from which cutoffs are set
    relative to a grid's own resolution.

    Parameters
    ----------
    grid
        Must have at least two nodes.
    """
    if grid.n_nodes < 2:
        raise ValueError(
            f"grid must have at least two nodes to have a nearest-neighbor "
            f"distance, got {grid.n_nodes}."
        )

    cpu_nodes = grid.nodes.detach().to(device="cpu", dtype=torch.float64).numpy()
    tree = cKDTree(cpu_nodes)
    # k=2: the nearest result is the point itself at distance 0; its nearest
    # neighbor is the second.
    chord_distances, _ = tree.query(cpu_nodes, k=2)
    nearest_chord = chord_distances[:, 1]

    # Chord to geodesic distance on the unit sphere: chord = 2 sin(geodesic / 2).
    geodesic = 2.0 * torch.asin(
        torch.as_tensor(nearest_chord / 2.0, dtype=torch.float64).clamp(max=1.0)
    )

    return float(torch.median(geodesic))
