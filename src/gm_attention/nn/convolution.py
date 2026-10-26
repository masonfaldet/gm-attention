"""GM-convolution: gauge-equivariant, unnormalized neighborhood aggregation.

``SO2Convolution`` evaluates the discretized trivial-affinity GM-aggregation

    (A^c_V f)_i := sum_{j in N_i} c(v_ij) w_ij  V(v_ij) rho_in(g_{i<-j}) f_j,

the GM-convolution specialized to ``SO(2)``. The per-edge value transform
``V(v_ij) rho_in(g_{i<-j}) f_j``, transport included, is
:class:`gm_attention.nn.kernels.SO2SteerableKernel`; this layer supplies the
gather of node features to edges (``edges.source_index``), the aggregation
window ``c``, the tangent-space quadrature weight ``w_ij``
(``edges.tangent_quadrature_weights``, the discrete measure approximating
``dv`` in the continuum integral over ``T_pM``) and the segment sum of the
weighted messages over each target's incident edges (``edges.row_ptr``). No
affinity and no normalization: every neighbor's weight is independent of
the feature values on either side of the edge.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Sequence
from typing import Literal, cast

import torch
import torch.utils.checkpoint
from torch import Tensor, nn

from gm_attention.discretization.discretization import SphericalDiscretization
from gm_attention.discretization.edge_geometry import EdgeGeometry, within_cutoff
from gm_attention.nn import fused_attention, fused_factored, kernels
from gm_attention.nn._scatter import gather_to_edges, scatter_add_to_nodes
from gm_attention.nn.depthwise import SO2DepthwiseKernel, stacked_coefficients
from gm_attention.nn.kernels import RadialBasisKind, SO2SteerableKernel
from gm_attention.nn.linear import SO2Linear
from gm_attention.nn.stencil_plan import StencilPlan, StencilPlans, split_edges_and_plan
from gm_attention.representations.field_type import (
    SO2FeatureField,
    SO2FieldType,
)

WindowKind = Literal["hard", "quartic"]
# The integral alone ("none") or its sum with a pointwise intertwiner ("linear").
SelfInteractionKind = Literal["none", "linear"]


def evaluate_aggregation_window(
    distances: Tensor,  # (...,)
    radius: float,
    kind: WindowKind,
) -> Tensor:  # (...,)
    """Evaluate a GM-aggregation window at the given geodesic distances.

    Coordinate independence requires the aggregation-window template
    ``c : R^d -> R_{>=0}`` to be gauge-invariant, ``c(gu) = c(u)`` for every
    ``g`` in the structure group. For ``G = SO(2)`` this forces ``c`` to
    depend on ``v`` only through ``r = ||v||``, so both windows are
    functions of the geodesic distance ``r`` alone.

    Parameters
    ----------
    distances
        Geodesic distances ``r``, arbitrary shape.
    radius
        The support radius: both windows are exactly zero for ``r > radius``.
    kind
        - ``"hard"``: ``c(r) = 1[r <= radius]``, the indicator that
          restricting to the edges within ``radius`` already computes.
        - ``"quartic"``: ``c(r) = (1 - (r/radius)^2)^2`` for ``r <= radius``,
          else ``0``. Continuously differentiable, with ``c(radius) = 0``
          and ``c'(radius) = 0``, so a neighbor crossing the boundary of
          the window enters or leaves without a jump.

    Returns
    -------
    Tensor
        Same shape as ``distances``.
    """
    if not (math.isfinite(radius) and radius > 0.0):
        raise ValueError(f"radius must be finite and positive, got {radius}.")

    if kind == "hard":
        # The same tolerant comparison that selected the edges (within_cutoff),
        # so the window never re-decides a boundary shell.
        return within_cutoff(distances, radius).to(distances.dtype)

    if kind == "quartic":
        ratio_squared = (distances / radius) ** 2  # (...,)
        return torch.relu(1.0 - ratio_squared) ** 2  # (...,)

    raise ValueError(f"window must be 'hard' or 'quartic', got {kind!r}.")


def _checkpointed(module: nn.Module, compute: Callable[[], dict[int, Tensor]]) -> dict[int, Tensor]:
    """One chunk, rematerialized in backward when training (non-reentrant
    checkpoint: closures and arbitrary returns are fine, RNG preserved)."""
    if module.training and torch.is_grad_enabled():
        return torch.utils.checkpoint.checkpoint(compute, use_reentrant=False)
    return compute()


class SO2Convolution(nn.Module):
    r"""A gauge-equivariant convolution over a node's geodesic neighborhood.

    Parameters
    ----------
    field_type_in, field_type_out
        Passed to the internal :class:`SO2SteerableKernel`.
    cutoff
        Support radius, shared between the kernel's radial basis and the
        aggregation window.
    window
        The aggregation window weighting the neighbors; see
        :func:`evaluate_aggregation_window`. ``"hard"`` is a plain
        restriction to the edges within ``cutoff``.
    self_interaction
        ``"linear"`` adds a learned pointwise intertwiner ``W_self x_i`` to
        each node's output, on top of the integral:

        .. math::

            y_i = \underbrace{\sum_j q_{ij} c(v_{ij}) V(v_{ij})
                  T_{i \leftarrow j} x_j}_{\text{discretized integral}}
                  + \underbrace{W_{\text{self}} x_i}_{\text{pointwise}}

        The pointwise term is added after aggregation and is never
        multiplied by the quadrature weight, the window or the
        normalization: it is not a sample of the integrand, so it does not
        shrink under mesh refinement as the integral's terms do. ``"none"``
        leaves the layer a pure discretized integral operator.
    normalize
        ``True`` divides the integral term by each target's window mass
        ``sum_j c(v_ij) w_ij`` over its incident edges (the denominator of
        :class:`~gm_attention.nn.blocks.IcoLowPassRestriction`), a weighted
        mean of kernel-transformed neighbors instead of a Riemann sum. One
        invariant scalar per target, shared by every frequency and
        coordinate, so equivariance is preserved. The self-interaction is
        not divided.
    depthwise
        ``True`` replaces the general kernel by
        :class:`~gm_attention.nn.depthwise.SO2DepthwiseKernel`: one steerable
        kernel per irrep copy and no mixing between copies. Requires
        ``field_type_in == field_type_out``. Window, quadrature weight,
        scatter, chunking and self-interaction are unchanged.
    eps
        Floor on the per-target window mass when ``normalize`` is set.
    n_radial_basis, radial_hidden_features, radial_basis, bias, dtype, device
        Passed to the internal kernel.
    """

    def __init__(
        self,
        field_type_in: SO2FieldType,
        field_type_out: SO2FieldType,
        *,
        cutoff: float,
        window: WindowKind = "hard",
        self_interaction: SelfInteractionKind = "none",
        depthwise: bool = False,
        normalize: bool = False,
        n_radial_basis: int = 8,
        radial_hidden_features: Sequence[int] = (16,),
        radial_basis: RadialBasisKind = "gaussian",
        bias: bool = True,
        eps: float = 1e-12,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        if window not in ("hard", "quartic"):
            raise ValueError(f"window must be 'hard' or 'quartic', got {window!r}.")
        if not eps > 0:
            raise ValueError(f"eps must be positive, got {eps}.")

        if depthwise and field_type_in != field_type_out:
            raise ValueError(
                "a depthwise convolution maps every irrep copy to itself, so its "
                f"input and output types must agree; got {field_type_in!r} -> "
                f"{field_type_out!r}."
            )

        if self_interaction not in ("none", "linear"):
            raise ValueError(
                f"self_interaction must be 'none' or 'linear', got {self_interaction!r}."
            )

        self.field_type_in = field_type_in
        self.field_type_out = field_type_out
        self.cutoff = float(cutoff)
        self.window: WindowKind = window
        self.self_interaction: SelfInteractionKind = self_interaction
        self.depthwise = bool(depthwise)
        self.normalize = bool(normalize)
        self.eps = float(eps)

        self.kernel: SO2SteerableKernel | SO2DepthwiseKernel
        if depthwise:
            self.kernel = SO2DepthwiseKernel(
                field_type_in,
                cutoff=cutoff,
                n_radial_basis=n_radial_basis,
                radial_hidden_features=radial_hidden_features,
                radial_basis=radial_basis,
                bias=bias,
                dtype=dtype,
                device=device,
            )
        else:
            self.kernel = SO2SteerableKernel(
                field_type_in,
                field_type_out,
                cutoff=cutoff,
                n_radial_basis=n_radial_basis,
                radial_hidden_features=radial_hidden_features,
                radial_basis=radial_basis,
                bias=bias,
                dtype=dtype,
                device=device,
            )

        # bias=False: the self-intertwiner is a linear map, as the layer is.
        self.self_linear = (
            SO2Linear(
                field_type_in,
                field_type_out,
                bias=False,
                dtype=dtype,
                device=device,
            )
            if self_interaction == "linear"
            else None
        )

    def forward(
        self,
        x: SO2FeatureField,
        edges: EdgeGeometry | SphericalDiscretization | StencilPlan | StencilPlans | None = None,
        *,
        plan: StencilPlan | StencilPlans | None = None,
    ) -> SO2FeatureField:
        """Aggregate node features over each node's incoming neighborhood.

        Parameters
        ----------
        x
            Node-level feature field of type ``field_type_in``, i.e.
            ``x.batch_shape == (n_nodes,)`` with ``n_nodes`` inferred from
            the edge geometry's ``row_ptr``.
        edges
            Either the neighborhood geometry directly (an ``EdgeGeometry``)
            or a ``SphericalDiscretization``, from which the layer takes
            ``edges_within(self.cutoff)``, served from the discretization's
            radius cache (``prepare_windows`` populates it ahead of the
            first forward). A :class:`~gm_attention.nn.stencil_plan.StencilPlan`
            or :class:`~gm_attention.nn.stencil_plan.StencilPlans` in this
            position counts as ``plan``, as a block forwards it to its mixer.
        plan
            Instead of ``edges``: a plan over a stencil built at
            ``self.cutoff``, or a ``StencilPlans`` holding one, on which the
            layer is evaluated by
            :func:`~gm_attention.nn.stencil_convolution.stencil_convolution`
            on the dense fixed-grid layout, the same values to summation
            order; ``x`` may then carry leading batch axes.

        Returns
        -------
        SO2FeatureField
            Node-level feature field of type ``field_type_out``, batch shape
            ``(n_nodes,)``.
        """
        edges, plan = split_edges_and_plan(edges, plan, self.cutoff)
        if plan is not None:
            from gm_attention.nn.stencil_convolution import stencil_convolution

            return stencil_convolution(self, x, plan)
        if isinstance(edges, SphericalDiscretization):
            edges = edges.edges_within(self.cutoff)

        if x.field_type != self.field_type_in:
            raise ValueError(
                f"Expected a field of type {self.field_type_in!r}, got "
                f"{x.field_type!r}."
            )

        n_nodes = edges.row_ptr.shape[0] - 1

        if x.batch_shape != (n_nodes,):
            raise ValueError(
                f"x.batch_shape must equal (n_nodes,) = ({n_nodes},), inferred "
                f"from edges.row_ptr, got {tuple(x.batch_shape)}."
            )

        # y_i = sum_{j in N(i)} c(v_ij) w_ij K(v_ij) rho(g_{i<-j}) x_j:
        # gather x_j to edges, kernel (transport included), weighted scatter.
        return self._finish(x, self._aggregate(x, edges))

    def _factored(self) -> bool:
        """Whether the kernel is evaluated through its factored path (the
        switch is on and the kernel is a steerable one)."""
        return kernels.factored_enabled() and isinstance(self.kernel, SO2SteerableKernel)

    def _node_products(self, x: SO2FeatureField) -> dict[tuple[int, int, str], Tensor] | None:
        return self.kernel.node_products(x) if self._factored() else None

    def _chunk_groups(self, x: SO2FeatureField, edges: EdgeGeometry, first: int, last: int, chunk: EdgeGeometry):
        """The chunk's edges grouped by source for the fused factored kernel,
        cached against the persistent geometry; None when not on that path."""
        if not (self._factored() and fused_factored.enabled(self.kernel, x, chunk)):
            return None
        return fused_factored.source_groups(chunk.source_index, fused_factored.group_key(edges, first, last), owner=edges.edge_index)

    def _aggregate(
        self, x: SO2FeatureField, edges: EdgeGeometry,
        products: dict[tuple[int, int, str], Tensor] | None = None,
        groups: tuple[Tensor, Tensor, Tensor] | None = None,
    ) -> dict[int, Tensor]:
        """Gather, apply the kernel, scatter. Shared by both forward paths.
        Under the factored switch the kernel reads the node field and its
        node-side mixes (``products``, shared across chunks when given);
        a depthwise layer under ``GM_FUSED_ATTENTION`` runs the whole
        aggregation as one fused op per direction."""
        if self.depthwise and fused_attention.enabled_depthwise(self, x, edges):
            return self._aggregate_fused(x, edges)

        # c(v_ij) w_ij: aggregation window times tangent-space quadrature
        # weight, shared by every output frequency.
        weight = evaluate_aggregation_window(
            edges.distances, self.cutoff, self.window
        ) * edges.tangent_quadrature_weights  # (n_edges,)

        # Window mass per target, an invariant scalar; exact per chunk too,
        # since chunks hold whole neighborhoods.
        denominator = (
            scatter_add_to_nodes(weight, edges.row_ptr).clamp_min(self.eps)  # (n_nodes,)
            if self.normalize
            else None
        )

        if self._factored():
            # The kernel in the factored order, its messages weighted and
            # summed per target in one step (one op per direction under the
            # fused kernel: no weighted per-edge tensor).
            summed_by_frequency = cast(SO2SteerableKernel, self.kernel).forward_factored_aggregated(
                x, edges, weight, products, groups
            ).coefficients
        else:
            source = gather_to_edges(x, edges.source_index)  # batch=(n_edges,)
            messages = self.kernel(source, edges)  # batch=(n_edges,), type_out
            summed_by_frequency = {}
            for frequency in self.field_type_out.frequencies:
                # (n_edges, b_n) at frequency 0, (n_edges, b_n, 2) above it.
                message = messages[frequency]
                # One scalar per edge, broadcast over the multiplicity and
                # coordinate axes; a scalar commutes with every rho_n.
                per_edge_weight = weight.reshape(
                    edges.n_edges, *([1] * (message.ndim - 1))
                )  # (n_edges, 1[, 1])
                # Segment sum over each target's incident edges, per row_ptr.
                summed_by_frequency[frequency] = scatter_add_to_nodes(
                    per_edge_weight * message, edges.row_ptr
                )  # (n_nodes, b_n[, 2])

        outputs: dict[int, Tensor] = {}
        for frequency in self.field_type_out.frequencies:
            summed = summed_by_frequency[frequency]
            if denominator is not None:
                summed = summed / denominator.reshape(-1, *([1] * (summed.ndim - 1)))
            outputs[frequency] = summed

        return outputs

    def forward_chunked(
        self,
        x: SO2FeatureField,
        edges: EdgeGeometry | SphericalDiscretization,
        *,
        chunk_size: int,
    ) -> SO2FeatureField:
        """:meth:`forward`, evaluating the edge set a slice at a time.

        Chunks are node ranges. Edges are sorted by target with ``row_ptr``
        marking each target's segment, so slicing at segment boundaries
        keeps every target's whole neighborhood inside one chunk, and each
        chunk sums exactly the terms :meth:`forward` sums for its targets;
        the chunks are disjoint in their targets, so their sum is the full
        result. Agreement with :meth:`forward` is to floating-point
        precision rather than bitwise, since the segment sum's accumulation
        order is implementation-defined. In training mode with gradients
        enabled each chunk runs under a non-reentrant checkpoint, so
        backward rematerializes one chunk at a time.

        Parameters
        ----------
        chunk_size
            Target nodes per chunk.
        """
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be at least 1, got {chunk_size}.")

        if isinstance(edges, (StencilPlan, StencilPlans)):
            raise TypeError("A StencilPlan evaluates the layer through forward only; chunked evaluation runs on the graph path.")
        if isinstance(edges, SphericalDiscretization):
            edges = edges.edges_within(self.cutoff)

        if x.field_type != self.field_type_in:
            raise ValueError(
                f"Expected a field of type {self.field_type_in!r}, got "
                f"{x.field_type!r}."
            )

        n_nodes = edges.row_ptr.shape[0] - 1

        if x.batch_shape != (n_nodes,):
            raise ValueError(
                f"x.batch_shape must equal (n_nodes,) = ({n_nodes},), inferred "
                f"from edges.row_ptr, got {tuple(x.batch_shape)}."
            )

        totals: dict[int, Tensor] = {}
        products = self._node_products(x)  # the factored kernel's node-side mixes, once for every chunk
        for start in range(0, n_nodes, chunk_size):
            stop = min(start + chunk_size, n_nodes)
            # Segment boundaries, so the slice holds whole neighborhoods.
            first, last = int(edges.row_ptr[start]), int(edges.row_ptr[stop])
            if first == last:  # every target in this range is isolated
                continue

            span = torch.arange(first, last, device=edges.row_ptr.device)
            chunk = edges.select(span)
            fused_attention.register_chunk(chunk, edges, first, last)
            groups = self._chunk_groups(x, edges, first, last, chunk)
            # Sized for the full node set: each chunk scatters into the same
            # shape and contributes zero outside its own range.
            partial = _checkpointed(
                self, functools.partial(self._aggregate, x, chunk, products, groups)
            )

            for frequency, value in partial.items():
                if frequency in totals:
                    totals[frequency] = totals[frequency] + value
                else:
                    totals[frequency] = value

        if not totals:  # no edges at all; forward would scatter into zeros
            return self._finish(x, self._aggregate(x, edges))

        return self._finish(x, totals)

    def _aggregate_fused(self, x: SO2FeatureField, edges: EdgeGeometry) -> dict[int, Tensor]:
        """:meth:`_aggregate` for a depthwise kernel through the fused walk
        (``fused_attention.Depthwise``): the radial coefficients in PyTorch,
        then transport, coefficient products, weighting and the segment sum
        in one op; the same values to summation order."""
        kernel = self.kernel
        assert isinstance(kernel, SO2DepthwiseKernel)
        real_coefficients, complex_coefficients = stacked_coefficients([kernel], edges)
        flat = fused_attention.forward_depthwise(self, x.to_flat(), real_coefficients, complex_coefficients, edges)
        return SO2FeatureField.from_flat(self.field_type_out, flat).coefficients

    def _finish(
        self, x: SO2FeatureField, outputs: dict[int, Tensor]
    ) -> SO2FeatureField:
        """Add the pointwise self-interaction, if this layer has one."""
        aggregated = SO2FeatureField(self.field_type_out, outputs)

        if self.self_linear is None:
            return aggregated

        # Added after the scatter, untouched by `weight`: a pointwise map,
        # not a quadrature sample.
        pointwise = self.self_linear(x)
        return SO2FeatureField(
            self.field_type_out,
            {
                frequency: aggregated[frequency] + pointwise[frequency]
                for frequency in self.field_type_out.frequencies
            },
        )

    def extra_repr(self) -> str:
        return (
            f"field_type_in={self.field_type_in!r}, "
            f"field_type_out={self.field_type_out!r}, "
            f"cutoff={self.cutoff:.6g}, window={self.window!r}, "
            f"self_interaction={self.self_interaction!r}, depthwise={self.depthwise}, "
            f"normalize={self.normalize}"
        )
