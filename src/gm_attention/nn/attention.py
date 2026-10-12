"""Gauge-equivariant self-attention over geodesic neighborhoods on the sphere.

:class:`SO2Attention` is single-head GM-self-attention specialized to
``SO(2)``: intertwiner query and key maps, a steerable value kernel, and a
softmax over each node's neighborhood weighted by the aggregation window and
the tangent quadrature weights. :class:`SO2MHSA` splits a field type across
heads, evaluated head by head or in one batched pass over the heads' stacked
parameters. Both support exact chunked evaluation over ranges of target nodes.
"""

from __future__ import annotations

import functools
import math
import os
from collections.abc import Callable, Sequence
from typing import Literal, cast

import torch
import torch.utils.checkpoint
from torch import Tensor, nn

from gm_attention.discretization.discretization import SphericalDiscretization
from gm_attention.discretization.edge_geometry import EdgeGeometry
from gm_attention.nn import fused_attention, fused_factored, kernels
from gm_attention.nn._scatter import (
    gather_to_edges,
    scatter_add_to_nodes,
    scatter_max_to_nodes,
)
from gm_attention.nn.convolution import (
    SelfInteractionKind,
    WindowKind,
    evaluate_aggregation_window,
)
from gm_attention.nn.kernels import (
    RadialBasisKind,
    SO2SteerableKernel,
    split_k_linear,
    transport_field,
)
from gm_attention.nn.linear import SO2Linear
from gm_attention.representations import so2
from gm_attention.representations.field_type import (
    SO2FeatureField,
    SO2FieldType,
)

#: ``GM_CHUNK_REMATERIALIZE=0`` keeps every chunk's saved tensors instead of recomputing its forward in the backward.
REMATERIALIZE_CHUNKS = os.environ.get("GM_CHUNK_REMATERIALIZE", "1") != "0"


def _checkpointed(module: nn.Module, compute: Callable[[], dict[int, Tensor]]) -> dict[int, Tensor]:
    """Run one chunk's computation, rematerialized in the backward when training.

    Non-reentrant checkpointing preserves the RNG state across the recompute,
    so dropout draws identically. Outside training, with gradients disabled,
    or under ``GM_CHUNK_REMATERIALIZE=0``, the closure simply runs.
    """
    if module.training and torch.is_grad_enabled() and REMATERIALIZE_CHUNKS:
        return torch.utils.checkpoint.checkpoint(compute, use_reentrant=False)
    return compute()


def _resolve_edges(
    edges: EdgeGeometry | SphericalDiscretization,
    cutoff: float,
) -> EdgeGeometry:
    if isinstance(edges, SphericalDiscretization):
        return edges.edges_within(cutoff)

    return edges


def _validate_node_field(
    x: SO2FeatureField,
    field_type: SO2FieldType,
    edges: EdgeGeometry,
) -> None:
    """Shared forward()-entry validation."""
    if x.field_type != field_type:
        raise ValueError(f"Expected a field of type {field_type!r}, got {x.field_type!r}.")

    n_nodes = edges.row_ptr.shape[0] - 1

    if x.batch_shape != (n_nodes,):
        raise ValueError(
            f"x.batch_shape must equal (n_nodes,) = ({n_nodes},), inferred "
            f"from edges.row_ptr, got {tuple(x.batch_shape)}."
        )


#: How a neighbor's features become a message; ``"steerable"``: one steerable kernel ``rho_in -> rho_out`` per edge.
ValueKind = Literal["steerable"]


class SO2Attention(nn.Module):
    """Gauge-equivariant scaled dot-product attention over a geodesic neighborhood.

    With ``W_Q, W_K`` intertwiners ``rho_in -> rho_attn``
    (:class:`~gm_attention.nn.linear.SO2Linear`, ``W rho_in(g) = rho_attn(g) W``)
    and ``V`` a steerable kernel ``rho_in -> rho_out``
    (:class:`~gm_attention.nn.kernels.SO2SteerableKernel`), the output at
    node ``i`` is

        s_ij     = <W_Q f_i, rho_attn(g_{i<-j}) W_K f_j> / sqrt(dim rho_attn)
        alpha_ij = c(v_ij) w_ij exp(s_ij) / sum_k c(v_ik) w_ik exp(s_ik)
        y_i      = sum_j alpha_ij V(v_ij) rho_in(g_{i<-j}) f_j

    over the neighbors ``j`` within ``cutoff``, with ``c`` the aggregation
    window, ``w_ij`` the tangent quadrature weight and ``g_{i<-j}`` the
    transport along the edge. ``rho_attn`` is a direct sum of real ``SO(2)``
    irreps, hence orthogonal, so the Euclidean dot product is gauge
    invariant and the layer is gauge equivariant. Keys are projected per
    node and transported per edge, which is exact because ``W_K``
    intertwines: ``W_K(rho_in(g) v) = rho_attn(g) (W_K v)``.

    Parameters
    ----------
    field_type_in
        Input field type.
    field_type_attn
        The type queries and keys are projected into (``rho_attn``).
    field_type_out
        Output field type, the value kernel's output type.
    cutoff
        Support radius, shared between the value kernel's radial basis and
        the aggregation window.
    window
        The aggregation window weighting neighbors before the softmax
        normalization; see
        :func:`gm_attention.nn.convolution.evaluate_aggregation_window`.
    self_interaction
        ``"linear"`` adds a learned pointwise intertwiner of each node's own
        input to its output, outside the attention normalization; ``"none"``
        (default) adds nothing. Inside a
        :class:`~gm_attention.nn.blocks.SO2TransformerBlock` the residual
        update already provides the pointwise path.
    n_radial_basis, radial_hidden_features, radial_basis
        Passed to the value kernel (:class:`SO2SteerableKernel`).
    value_kind
        See :data:`ValueKind`.
    bias
        Whether the query, key and value maps carry a bias on their
        frequency-``0`` channels, the only place equivariance permits one
        (see :class:`SO2Linear`); also controls the value kernel's
        radial-MLP biases (see :class:`SO2SteerableKernel`).
    dtype, device
        Floating-point dtype and device of the parameters.
    """

    def __init__(
        self,
        field_type_in: SO2FieldType,
        field_type_attn: SO2FieldType,
        field_type_out: SO2FieldType,
        *,
        cutoff: float,
        window: WindowKind = "hard",
        self_interaction: SelfInteractionKind = "none",
        n_radial_basis: int = 8,
        radial_hidden_features: Sequence[int] = (16,),
        radial_basis: RadialBasisKind = "gaussian",
        bias: bool = True,
        value_kind: ValueKind = "steerable",
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        if window not in ("hard", "quartic"):
            raise ValueError(f"window must be 'hard' or 'quartic', got {window!r}.")

        if self_interaction not in ("none", "linear"):
            raise ValueError(
                f"self_interaction must be 'none' or 'linear', got {self_interaction!r}."
            )

        if value_kind != "steerable":
            raise ValueError(f"value_kind must be 'steerable', got {value_kind!r}.")
        self.value_kind: ValueKind = value_kind

        self.field_type_in = field_type_in
        self.field_type_attn = field_type_attn
        self.field_type_out = field_type_out
        self.cutoff = float(cutoff)
        self.window: WindowKind = window
        self.self_interaction: SelfInteractionKind = self_interaction

        self.query = SO2Linear(
            field_type_in, field_type_attn, bias=bias, dtype=dtype, device=device
        )
        self.key = SO2Linear(
            field_type_in, field_type_attn, bias=bias, dtype=dtype, device=device
        )
        self.value = SO2SteerableKernel(
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

        self.self_linear = (
            SO2Linear(
                field_type_in, field_type_out, bias=False, dtype=dtype, device=device
            )
            if self_interaction == "linear"
            else None
        )

        # S(q, k) = <q, k> / sqrt(dim(rho_attn)).
        self._score_scale = 1.0 / math.sqrt(field_type_attn.dimension)

    def forward(
        self,
        x: SO2FeatureField,
        edges: EdgeGeometry | SphericalDiscretization,
    ) -> SO2FeatureField:
        """Attend over each node's incoming neighborhood.

        Parameters
        ----------
        x
            Node-level feature field of type ``field_type_in``.
        edges
            Either an ``EdgeGeometry``, or a ``SphericalDiscretization``,
            from which ``edges_within(self.cutoff)`` is taken.

        Returns
        -------
        SO2FeatureField
            Node-level feature field of type ``field_type_out``.
        """
        edges = _resolve_edges(edges, self.cutoff)
        _validate_node_field(x, self.field_type_in, edges)
        with torch.profiler.record_function("gm::attention/attend"):
            attended = self._attend(x, edges)
        with torch.profiler.record_function("gm::attention/finish"):
            return self._finish(x, attended)

    def _attend(
        self, x: SO2FeatureField, edges: EdgeGeometry,
        products: dict[tuple[int, int, str], Tensor] | None = None,
        groups: tuple[Tensor, Tensor, Tensor] | None = None,
    ) -> dict[int, Tensor]:
        """Score, normalize and aggregate; shared by :meth:`forward` and :meth:`forward_chunked`.

        Every step is per target (the softmax maximum, its normalizer and the
        value scatter key on ``row_ptr``), so restricting ``edges`` to a
        contiguous run of targets gives those targets the full result.
        """
        # Queries: one projection per node, evaluated at the target's own feature.
        query_nodes = self.query(x)  # SO2FeatureField[attn], (n_nodes,)

        # Keys: project per node, then transport per edge (exact: W_K intertwines).
        key_nodes = self.key(x)  # type rho_attn, batch=(n_nodes,)

        if fused_attention.softmax_head_enabled(self, x, edges):
            # The scores, the window and quadrature bias and the per-target
            # softmax in one op per direction (fused_attention.softmax_head).
            attention_weight = fused_attention.softmax_head(self, query_nodes.to_flat(), key_nodes.to_flat(), edges)
        else:
            attention_weight = self._softmax_weight(query_nodes, key_nodes, edges)

        # y_i = sum_j alpha_ij V(v_ij) rho(g_{i<-j}) x_j; the kernel transports internally.
        if self._factored():
            # The value kernel in the factored order, its messages weighted
            # and summed per target in one step (one op per direction under
            # the fused kernel: no weighted per-edge tensor).
            return cast(SO2SteerableKernel, self.value).forward_factored_aggregated(
                x, edges, attention_weight, products, groups
            ).coefficients
        messages = self._messages(x, edges)  # type_out, (n_edges,)

        outputs: dict[int, Tensor] = {}

        for frequency in self.field_type_out.frequencies:
            message = messages[frequency]
            weight = attention_weight.reshape(
                edges.n_edges, *([1] * (message.ndim - 1))
            )
            outputs[frequency] = scatter_add_to_nodes(
                weight * message, edges.row_ptr
            )

        return outputs

    def _softmax_weight(self, query_nodes: SO2FeatureField, key_nodes: SO2FeatureField, edges: EdgeGeometry) -> Tensor:
        """The attention weights ``(n_edges,)`` in eager PyTorch: the gathered
        queries, the gathered and transported keys, the scores, the window
        and quadrature bias and the per-target softmax."""
        query_edges = gather_to_edges(query_nodes, edges.target_index)
        key_edges = transport_field(  # type rho_attn, batch=(n_edges,)
            gather_to_edges(key_nodes, edges.source_index), edges
        )

        # s_ij = <q_i, rho(g_{i<-j}) k_j> / sqrt(dim rho_attn), summed over
        # frequencies in the packed real layout (rho_n orthogonal, so invariant).
        score = torch.zeros_like(edges.distances)
        for frequency in query_edges.field_type.frequencies:
            product = query_edges[frequency] * key_edges[frequency]
            score = score + product.reshape(product.shape[0], -1).sum(dim=-1)
        score = score * self._score_scale  # (n_edges,)

        window = evaluate_aggregation_window(edges.distances, self.cutoff, self.window)

        # Log-space weight ell_ij = s_ij + log c(v_ij) + log w_ij, shifted by
        # each target's maximum before exponentiating (window included, so a
        # near-zero window on the largest score cannot underflow the rest).
        tiny = torch.finfo(score.dtype).tiny
        inside_window = window > 0.0
        log_weight = (
            score
            + torch.log(window.clamp_min(tiny))
            + torch.log(edges.tangent_quadrature_weights.clamp_min(tiny))
        )  # (n_edges,)

        # clamp_min keeps a masked edge's exponent finite (no -inf - -inf); the
        # hard zero below enforces compact support. The shift is detached: the
        # softmax is invariant to it, so its gradient is exactly zero.
        node_max = scatter_max_to_nodes(log_weight, edges.row_ptr).detach()
        unnormalized_weight = torch.where(
            inside_window,
            torch.exp(log_weight - node_max.index_select(0, edges.target_index)),
            torch.zeros_like(log_weight),
        )  # (n_edges,)

        # Every node carries its own self-loop with strictly positive weight
        # (window(0) = 1, positive quadrature weight), so the sum is > 0.
        normalization = scatter_add_to_nodes(
            unnormalized_weight, edges.row_ptr
        )  # (n_nodes,)
        # alpha_ij: a partition of unity over each target's neighborhood.
        return unnormalized_weight / normalization.index_select(0, edges.target_index)

    def forward_chunked(
        self,
        x: SO2FeatureField,
        edges: EdgeGeometry | SphericalDiscretization,
        *,
        chunk_size: int,
    ) -> SO2FeatureField:
        """:meth:`forward`, evaluating the edge set one range of target nodes at a time.

        Chunks are node ranges: every step of the attention keys on
        ``row_ptr``, so a contiguous run of targets carries its whole
        neighborhood and its softmax is complete within the chunk, and the
        chunked result equals :meth:`forward` up to floating-point summation
        order. In training mode with gradients enabled each chunk runs under
        non-reentrant ``torch.utils.checkpoint``, so the backward
        rematerializes one chunk at a time and peak memory is one chunk's
        per-edge intermediates.

        Parameters
        ----------
        chunk_size
            Target nodes per chunk.
        """
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be at least 1, got {chunk_size}.")

        edges = _resolve_edges(edges, self.cutoff)
        _validate_node_field(x, self.field_type_in, edges)

        n_nodes = edges.row_ptr.shape[0] - 1
        totals: dict[int, Tensor] = {}
        # The factored kernel's node-side mixes, shared by every chunk.
        products = self._node_products(x)

        for start in range(0, n_nodes, chunk_size):
            stop = min(start + chunk_size, n_nodes)
            first, last = int(edges.row_ptr[start]), int(edges.row_ptr[stop])
            if first == last:
                continue

            span = torch.arange(first, last, device=edges.row_ptr.device)
            chunk = edges.select(span)
            fused_attention.register_chunk(chunk, edges, first, last)
            groups = self._chunk_groups(x, edges, first, last, chunk)
            # Bound eagerly (partial, not a closure): the checkpoint recomputes
            # this in backward, after the loop has moved on.
            partial = _checkpointed(
                self, functools.partial(self._attend, x, chunk, products, groups)
            )

            for frequency, value in partial.items():
                totals[frequency] = (
                    value if frequency not in totals else totals[frequency] + value
                )

        if not totals:
            return self._finish(x, self._attend(x, edges))

        return self._finish(x, totals)

    def _factored(self) -> bool:
        """Whether the value kernel is evaluated through its factored path
        (``GM_FACTORED_KERNEL``; :func:`~gm_attention.nn.kernels.factored_enabled`)."""
        return kernels.factored_enabled() and isinstance(self.value, SO2SteerableKernel)

    def _node_products(self, x: SO2FeatureField) -> dict[tuple[int, int, str], Tensor] | None:
        return self.value.node_products(x) if self._factored() else None

    def _chunk_groups(self, x: SO2FeatureField, edges: EdgeGeometry, first: int, last: int, chunk: EdgeGeometry):
        """The chunk's edges grouped by source for the fused factored kernel,
        cached against the persistent geometry; None when not on that path."""
        if not (self._factored() and fused_factored.enabled(self.value, x, chunk)):
            return None
        return fused_factored.source_groups(chunk.source_index, fused_factored.group_key(edges, first, last), owner=edges.edge_index)

    def _messages(self, x: SO2FeatureField, edges: EdgeGeometry) -> SO2FeatureField:
        """The per-edge value messages, ``rho_out`` on ``(n_edges,)``: the
        steerable kernel on the gathered input. (Under the factored switch
        the messages are never formed on their own: :meth:`_attend` sums
        them per target as it goes.)"""
        return self.value(gather_to_edges(x, edges.source_index), edges)

    def _finish(
        self, x: SO2FeatureField, outputs: dict[int, Tensor]
    ) -> SO2FeatureField:
        """Add the pointwise self-interaction, if this layer has one."""
        attended = SO2FeatureField(self.field_type_out, outputs)

        if self.self_linear is None:
            return attended

        # Outside the attention normalization, as SO2Convolution adds its own
        # outside the quadrature sum: a pointwise map, not a softmax term.
        pointwise = self.self_linear(x)
        return SO2FeatureField(
            self.field_type_out,
            {
                frequency: attended[frequency] + pointwise[frequency]
                for frequency in self.field_type_out.frequencies
            },
        )

    def extra_repr(self) -> str:
        return (
            f"field_type_in={self.field_type_in!r}, "
            f"field_type_attn={self.field_type_attn!r}, "
            f"field_type_out={self.field_type_out!r}, "
            f"cutoff={self.cutoff:.6g}, window={self.window!r}, "
            f"self_interaction={self.self_interaction!r}, value_kind={self.value_kind!r}"
        )


def _concatenate_heads(
    fields: Sequence[SO2FeatureField],
    field_type: SO2FieldType,
) -> SO2FeatureField:
    """Direct-sum same-typed fields along the multiplicity axis.

    Each field in ``fields`` must share the same field type, with every
    frequency's multiplicity exactly ``field_type.multiplicity(n) /
    len(fields)``, e.g. one head's worth of ``rho_attn``.
    """
    coefficients: dict[int, Tensor] = {}

    for frequency in field_type.frequencies:
        multiplicity_axis = -1 if frequency == 0 else -2
        coefficients[frequency] = torch.cat(
            [field[frequency] for field in fields], dim=multiplicity_axis
        )

    return SO2FeatureField(field_type, coefficients)


class SO2MHSA(nn.Module):
    """Multi-head gauge-equivariant self-attention.

    Splits ``field_type_in`` evenly across ``num_heads`` heads, each an
    :class:`SO2Attention` with
    ``rho_attn = rho_out = (+)_n (a_n / num_heads) rho_n``, concatenates the
    heads' outputs along each frequency's multiplicity axis (head-major) and
    applies a learned intertwiner ``field_type_in -> field_type_in``.

    Parameters
    ----------
    field_type_in
        Input and output field type. Every present frequency's multiplicity
        ``a_n`` must be divisible by ``num_heads``; the check is per
        frequency rather than on the total dimension, so that no head splits
        the two real coordinates of one ``rho_n`` copy or receives different
        representation content from another.
    num_heads
        Number of attention heads.
    cutoff, window
        Shared across every head; see :class:`SO2Attention`.
    n_radial_basis, radial_hidden_features, radial_basis, bias, dtype, device
        Passed to every head's :class:`SO2SteerableKernel` and to the output
        intertwiner.
    batched
        Evaluate every head in one pass over the heads' stacked parameters
        (:func:`_attend_batched`) instead of head after head: the same
        function and parameters, differing only in summation order. Off by
        default.
    """

    def __init__(
        self,
        field_type_in: SO2FieldType,
        num_heads: int,
        *,
        cutoff: float,
        window: WindowKind = "hard",
        n_radial_basis: int = 8,
        radial_hidden_features: Sequence[int] = (16,),
        radial_basis: RadialBasisKind = "gaussian",
        bias: bool = True,
        batched: bool = False,
        value_kind: ValueKind = "steerable",
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        if num_heads < 1:
            raise ValueError(f"num_heads must be at least 1, got {num_heads}.")

        self.batched = bool(batched)
        self.value_kind: ValueKind = value_kind

        offending = {
            frequency: multiplicity
            for frequency, multiplicity in field_type_in.items()
            if multiplicity % num_heads != 0
        }

        if offending:
            raise ValueError(
                f"num_heads ({num_heads}) must divide the multiplicity of "
                f"every present frequency; offending frequencies (as "
                f"frequency: multiplicity): {offending}."
            )

        self.field_type_in = field_type_in
        self.num_heads = num_heads
        self.cutoff = float(cutoff)

        # Per-frequency integer division: each head keeps the full spectrum
        # at 1/num_heads of each frequency's own width.
        self.field_type_attn = SO2FieldType(
            {
                frequency: multiplicity // num_heads
                for frequency, multiplicity in field_type_in.items()
            }
        )

        self.heads = nn.ModuleList(
            [
                SO2Attention(
                    field_type_in,
                    self.field_type_attn,
                    self.field_type_attn,
                    cutoff=cutoff,
                    window=window,
                    n_radial_basis=n_radial_basis,
                    radial_hidden_features=radial_hidden_features,
                    radial_basis=radial_basis,
                    bias=bias,
                    value_kind=value_kind,
                    dtype=dtype,
                    device=device,
                )
                for _ in range(num_heads)
            ]
        )

        self.output_projection = SO2Linear(
            field_type_in, field_type_in, bias=bias, dtype=dtype, device=device
        )

    def forward(
        self,
        x: SO2FeatureField,
        edges: EdgeGeometry | SphericalDiscretization,
    ) -> SO2FeatureField:
        """Apply every head and fuse the results.

        Parameters
        ----------
        x
            Node-level feature field of type ``field_type_in``.
        edges
            As in :meth:`SO2Attention.forward`, resolved once and passed as
            an ``EdgeGeometry`` to every head.

        Returns
        -------
        SO2FeatureField
            Node-level feature field of type ``field_type_in``.
        """
        edges = _resolve_edges(edges, self.cutoff)

        if self.batched:
            _validate_node_field(x, self.field_type_in, edges)
            return self.output_projection(
                SO2FeatureField(self.field_type_in, _attend_batched(self, x, edges))
            )

        head_outputs = [head(x, edges) for head in self.heads]
        fused = _concatenate_heads(head_outputs, self.field_type_in)

        return self.output_projection(fused)

    def forward_chunked(
        self,
        x: SO2FeatureField,
        edges: EdgeGeometry | SphericalDiscretization,
        *,
        chunk_size: int,
    ) -> SO2FeatureField:
        """:meth:`forward` with each head chunked; see :meth:`SO2Attention.forward_chunked`.

        The concatenation and output projection are pointwise and hold
        nothing per edge.
        """
        edges = _resolve_edges(edges, self.cutoff)

        head_outputs = [
            cast(SO2Attention, head).forward_chunked(
                x, edges, chunk_size=chunk_size
            )
            for head in self.heads
        ]
        fused = _concatenate_heads(head_outputs, self.field_type_in)

        return self.output_projection(fused)

    def extra_repr(self) -> str:
        return (
            f"field_type_in={self.field_type_in!r}, "
            f"num_heads={self.num_heads}, "
            f"field_type_attn={self.field_type_attn!r}, "
            f"cutoff={self.cutoff:.6g}, batched={self.batched}, value_kind={self.value_kind!r}"
        )


# ---------------------------------------------------------------------------
# The batched multi-head path: every head of an SO2MHSA in one pass; see
# _attend_batched.


def _stacked_projection(linears: Sequence[SO2Linear], x: SO2FeatureField) -> SO2FeatureField:
    """The heads' intertwiners applied at once: ``x`` projected onto the
    direct sum of their output types, head-major within each frequency."""
    outputs: dict[int, Tensor] = {}
    for frequency in x.field_type.frequencies:
        key = str(frequency)
        weight = torch.cat([linear.weight[key] for linear in linears], dim=0)  # (a, a[, 2])
        if frequency == 0:
            output = x[0] @ weight.transpose(-1, -2)  # (n_nodes, a)
            if linears[0].has_bias:
                output = output + torch.cat([linear.bias for linear in linears])
            outputs[0] = output
        else:
            outputs[frequency] = so2.as_real(
                so2.as_complex(x[frequency]) @ so2.as_complex(weight).transpose(-1, -2)
            )  # (n_nodes, a, 2)
    return SO2FeatureField(x.field_type, outputs)


def _stacked(parameters: Sequence[Tensor | None]) -> Tensor | None:
    """``torch.stack`` over heads, or ``None`` when the layer has no such parameter."""
    if parameters[0] is None:
        return None
    return torch.stack([cast(Tensor, p) for p in parameters])


def _stacked_kernels(
    kernels: Sequence[SO2SteerableKernel],
    x: SO2FeatureField,
    edges: EdgeGeometry,
) -> dict[int, Tensor]:
    """Every head's value kernel on the same edges, with a leading head axis.

    ``x`` is the input field gathered to edge shape. Returns, per output
    frequency, ``(n_heads, n_edges, b_m)`` (frequency 0) or
    ``(n_heads, n_edges, b_m, 2)``: the arithmetic of
    :meth:`SO2SteerableKernel._forward_general` with a head axis, one radial
    embedding shared by every head, one transport of the input, per-head
    trunks and readouts as grouped matmuls over the stacked parameters.
    """
    first = kernels[0]
    first.validate_call(x, edges)
    n_heads = len(kernels)
    n_edges = edges.n_edges
    real_dtype = x.dtype
    complex_dtype = so2.complex_dtype(real_dtype)

    with torch.profiler.record_function("gm::kernel/radial"):
        embedding = first.radial_basis(edges.distances)  # (n_edges, n_basis)
        hidden = embedding.unsqueeze(0).expand(n_heads, -1, -1)  # (n_heads, n_edges, n_basis)
        trunks = [
            [module for module in kernel.radial_trunk if isinstance(module, nn.Linear)]
            for kernel in kernels
        ]
        for layer_index in range(len(trunks[0])):
            weight = torch.stack([layers[layer_index].weight for layers in trunks])  # (n_heads, out, in)
            bias = _stacked([layers[layer_index].bias for layers in trunks])
            hidden = torch.nn.functional.silu(split_k_linear(hidden, weight, bias))  # (n_heads, n_edges, out)

    def readout(heads: Sequence[nn.Linear | None]) -> Tensor | None:
        if heads[0] is None:
            return None
        weight = torch.stack([cast(nn.Linear, head).weight for head in heads])  # (n_heads, width, trunk)
        bias = _stacked([cast(nn.Linear, head).bias for head in heads])
        return split_k_linear(hidden, weight, bias)  # (n_heads, n_edges, width)

    complex_coefficients = readout([kernel.complex_head for kernel in kernels])
    real_coefficients = readout([kernel.real_head for kernel in kernels])
    coefficient_chunks = (
        dict(
            zip(
                first._coefficient_order,
                torch.split(complex_coefficients, first._coefficient_widths, dim=2),
                strict=True,
            )
        )
        if complex_coefficients is not None
        else None
    )
    radial_powers = (
        first._radial_powers(edges.distances) if complex_coefficients is not None else None
    )

    transported_field = transport_field(x, edges)
    transported: dict[int, Tensor] = {
        frequency: (
            transported_field[0] if frequency == 0
            else so2.as_complex(transported_field[frequency])
        )
        for frequency in first.field_type_in.frequencies
    }

    def apply(matrices: Tensor, vectors: Tensor) -> Tensor:
        # (n_heads, n_edges, b, a) blocks on one (n_edges, a) vector per edge.
        return (matrices * vectors.unsqueeze(0).unsqueeze(-2)).sum(-1)  # (n_heads, n_edges, b)

    def scale(frequency_n: int, frequency_m: int, role: str) -> Tensor:
        assert radial_powers is not None
        return radial_powers[:, first._pair_powers[(frequency_n, frequency_m, role)]].unsqueeze(-1)

    outputs: dict[int, Tensor] = {}
    with torch.profiler.record_function("gm::kernel/pairs"):
        for frequency_m in first.field_type_out.frequencies:
            multiplicity_out = first.field_type_out.multiplicity(frequency_m)
            accumulator = torch.zeros(
                n_heads, n_edges, multiplicity_out,
                dtype=real_dtype if frequency_m == 0 else complex_dtype, device=x.device,
            )
            for frequency_n in first.field_type_in.frequencies:
                multiplicity_in = first.field_type_in.multiplicity(frequency_n)
                if frequency_n == 0 and frequency_m == 0:
                    assert real_coefficients is not None
                    matrix = real_coefficients.reshape(n_heads, n_edges, multiplicity_out, multiplicity_in)
                    accumulator = accumulator + apply(matrix, transported[0])
                    continue
                assert coefficient_chunks is not None
                linear_matrix = so2.as_complex(
                    coefficient_chunks[(frequency_n, frequency_m, "linear")].reshape(
                        n_heads, n_edges, multiplicity_out, multiplicity_in, 2
                    )
                )
                if frequency_n == 0:
                    source = transported[0].to(complex_dtype)
                    phase = edges.angular_harmonic(frequency_m).unsqueeze(-1)
                    contribution = phase * apply(linear_matrix, source) * scale(frequency_n, frequency_m, "linear")
                elif frequency_m == 0:
                    phase = edges.angular_harmonic(-frequency_n).unsqueeze(-1)
                    rotated = phase * transported[frequency_n]
                    contribution = apply(linear_matrix, rotated.conj()).real * scale(frequency_n, frequency_m, "linear")
                else:
                    antilinear_matrix = so2.as_complex(
                        coefficient_chunks[(frequency_n, frequency_m, "antilinear")].reshape(
                            n_heads, n_edges, multiplicity_out, multiplicity_in, 2
                        )
                    )
                    linear_phase = edges.angular_harmonic(frequency_m - frequency_n).unsqueeze(-1)
                    antilinear_phase = edges.angular_harmonic(frequency_m + frequency_n).unsqueeze(-1)
                    contribution = (
                        apply(linear_matrix, linear_phase * transported[frequency_n])
                        * scale(frequency_n, frequency_m, "linear")
                        + apply(antilinear_matrix, antilinear_phase * transported[frequency_n].conj())
                        * scale(frequency_n, frequency_m, "antilinear")
                    )
                accumulator = accumulator + contribution
            outputs[frequency_m] = accumulator if frequency_m == 0 else so2.as_real(accumulator)
    return outputs


def _aggregated_factored(mhsa: SO2MHSA, x: SO2FeatureField, edges: EdgeGeometry, attention_weight: Tensor) -> dict[int, Tensor]:
    """Every head's steerable value kernel in the factored order, each head's
    messages weighted by its attention weights ``attention_weight[:, h]``
    and summed per target (one op per direction through the fused kernel
    where :func:`fused_factored.enabled`), the heads' node fields then
    concatenated head-major within each frequency: what
    :func:`_attend_batched`'s weighting and segment sum form from stacked
    per-edge messages, without the per-edge weighted tensor. Each head
    reads the node field and its own node-side mixes; the source grouping
    of the edges is one computation for every head, cached against the
    persistent geometry."""
    heads = [cast(SO2Attention, head) for head in mhsa.heads]
    fields: list[SO2FeatureField] = []
    groups = None
    for head, weights in zip(heads, attention_weight.unbind(1), strict=True):
        kernel = cast(SO2SteerableKernel, head.value)
        fused = fused_factored.enabled(kernel, x, edges)
        if fused and groups is None:
            groups = fused_factored.groups_for(edges, edges.source_index)
        fields.append(kernel.forward_factored_aggregated(x, edges, weights.contiguous(), kernel.node_products(x),
                                                         groups if fused else None))
    return _concatenate_heads(fields, mhsa.field_type_in).coefficients


def _softmax_weights(heads: Sequence[SO2Attention], queries: SO2FeatureField, keys: SO2FeatureField, edges: EdgeGeometry,
                     n_heads: int) -> Tensor:
    """The heads' attention weights ``(n_edges, n_heads)`` in eager PyTorch:
    the gathered queries, the gathered and transported keys, the scores,
    the window and quadrature bias and the per-target softmax."""
    field_type = queries.field_type
    n_edges = edges.n_edges
    query_edges = gather_to_edges(queries, edges.target_index)
    key_edges = transport_field(gather_to_edges(keys, edges.source_index), edges)

    # s_ij^(h) = <q_i^(h), rho(g_{i<-j}) k_j^(h)> / sqrt(dim rho_attn); each
    # head's copies are a contiguous run of the frequency's multiplicity axis.
    score = torch.zeros(n_edges, n_heads, dtype=query_edges[field_type.frequencies[0]].dtype, device=edges.row_ptr.device)
    for frequency in field_type.frequencies:
        product = query_edges[frequency] * key_edges[frequency]  # (n_edges, a[, 2])
        score = score + product.reshape(n_edges, n_heads, -1).sum(-1)
    score = score * heads[0]._score_scale

    window = evaluate_aggregation_window(edges.distances, heads[0].cutoff, heads[0].window)
    tiny = torch.finfo(score.dtype).tiny
    inside_window = (window > 0.0).unsqueeze(-1)  # (n_edges, 1)
    log_weight = score + (
        torch.log(window.clamp_min(tiny))
        + torch.log(edges.tangent_quadrature_weights.clamp_min(tiny))
    ).unsqueeze(-1)  # (n_edges, n_heads)
    node_max = scatter_max_to_nodes(log_weight, edges.row_ptr).detach()  # (n_nodes, n_heads); detached as in _attend
    unnormalized_weight = torch.where(
        inside_window,
        torch.exp(log_weight - node_max.index_select(0, edges.target_index)),
        torch.zeros_like(log_weight),
    )
    normalization = scatter_add_to_nodes(unnormalized_weight, edges.row_ptr)  # (n_nodes, n_heads)
    return unnormalized_weight / normalization.index_select(0, edges.target_index)  # (n_edges, n_heads)


def _attend_batched(mhsa: SO2MHSA, x: SO2FeatureField, edges: EdgeGeometry) -> dict[int, Tensor]:
    """Score, normalize and aggregate every head at once.

    The heads' query (key) intertwiners stacked along the output
    multiplicity form one intertwiner of the full type whose output copies,
    split within each frequency's multiplicity axis, are the heads' queries
    (keys); the value kernels stack the same way with per-head radial trunks
    and readouts, so the function and parameters are exactly the sequential
    path's. Scores form an ``(n_edges, n_heads)`` tensor whose per-target
    maxima and normalizers are taken per head, with the window and
    quadrature weight broadcast across heads; a contiguous target range is
    self-contained as in :meth:`SO2Attention._attend`. No parameter is
    owned here.
    """
    heads = [cast(SO2Attention, head) for head in mhsa.heads]
    n_heads = len(heads)
    n_edges = edges.n_edges
    field_type = mhsa.field_type_in

    with torch.profiler.record_function("gm::attention/attend"):
        queries = _stacked_projection([head.query for head in heads], x)
        keys = _stacked_projection([head.key for head in heads], x)
        if fused_attention.softmax_enabled(mhsa, x, edges):
            # The scores, the window and quadrature bias and the per-target
            # softmax in one op per direction; the messages below come from
            # the value kernels as before.
            attention_weight = fused_attention.softmax(mhsa, queries.to_flat(), keys.to_flat(), edges)  # (n_edges, n_heads)
        else:
            attention_weight = _softmax_weights(heads, queries, keys, edges, n_heads)

        if heads[0]._factored():
            # The factored order: weighted and summed per target head by
            # head, no stacked per-edge messages.
            return _aggregated_factored(mhsa, x, edges, attention_weight)
        else:
            source = gather_to_edges(x, edges.source_index)
            messages = _stacked_kernels(
                [cast(SO2SteerableKernel, head.value) for head in heads], source, edges
            )

        outputs: dict[int, Tensor] = {}
        for frequency in field_type.frequencies:
            message = messages[frequency]  # (n_heads, n_edges, b[, 2])
            weight = attention_weight.transpose(0, 1).reshape(
                n_heads, n_edges, *([1] * (message.ndim - 2))
            )
            weighted = (weight * message).transpose(0, 1)  # (n_edges, n_heads, b[, 2])
            # Head-major within the frequency's multiplicity axis: the order
            # _concatenate_heads produces.
            merged = weighted.reshape(n_edges, field_type.multiplicity(frequency), *message.shape[3:])
            outputs[frequency] = scatter_add_to_nodes(merged, edges.row_ptr)
    return outputs
