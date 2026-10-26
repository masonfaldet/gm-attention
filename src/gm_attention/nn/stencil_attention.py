"""Gauge-equivariant attention on a fixed tensor-product grid, every output row at once.

:func:`stencil_attention` and :func:`stencil_mhsa` evaluate
:class:`~gm_attention.nn.attention.SO2Attention` and
:class:`~gm_attention.nn.attention.SO2MHSA` on the dense
``(..., n_latitude, n_longitude, channels)`` layout from a
:class:`~gm_attention.nn.stencil_plan.StencilPlan`, the same arithmetic as
the per-row reference :func:`~gm_attention.nn.fixed_grid.attention_fixed_grid`
without its Python loop over rows. A forward has two stages:

1. :func:`attention_tables` evaluates everything that depends on the
   parameters but not on the field, once per slot: the value kernel's
   ``(d_out, d_in)`` matrix (:func:`~gm_attention.nn.stencil_kernel.steerable_slot_matrices`,
   the values of the reference's ``steerable_kernel_matrices`` assembled by
   two gathers, stored transposed, with a zero row for the plan's phantom
   slot), the
   softmax bias ``log c(v_s) + log w_s`` (``-inf`` outside the window and
   at the phantom), and the cosine and sine of every attention frequency's
   transport angle. These are ``n_slots`` rows, ``n_longitude`` times fewer
   than the graph path's edges, and the only complex arithmetic.
2. :func:`attend` is the dense part, in real arithmetic with static shapes:
   per bucket of the plan, one ``index_select`` gathers every row's whole
   neighbourhood as ``(batch, heads, rows, n_longitude, width, d)``, the
   scores are a batched matmul of the gathered keys against the queries (the
   transport ``<q, rho(g_s) k>`` is ``cos(f gamma_s) <q, k> + sin(f gamma_s)
   <q_perp, k>`` per frequency, so the keys are never rotated), a softmax
   over the slot axis, and the aggregation is one matmul of the
   attention-weighted gathered sources, ``(rows, n_longitude, width * d_in)``,
   against the stacked slot matrices ``(rows, width * d_in, d_out)``: the
   contraction over slots and input channels is a single GEMM with
   ``width * d_in`` inner dimension, and no per-edge message tensor exists.

Both functions agree with the graph path and the reference to summation
order, gradients included (``tests/nn/test_stencil_attention.py``), and are
exactly equivariant to longitude shifts. :func:`attend` compiles with
``torch.compile(fullgraph=True)``: the plan's tables are inputs and every
shape is fixed at trace time. The per-edge arithmetic of the steerable value
kernel remains what it is; what this path removes is the per-edge geometry,
radial MLP, gather and scatter of the graph path and the per-row launches
of the reference.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple, cast

import torch
from torch import Tensor

from gm_attention.nn.attention import SO2MHSA, SO2Attention
from gm_attention.nn.convolution import evaluate_aggregation_window
from gm_attention.nn.stencil_kernel import steerable_slot_matrices
from gm_attention.nn.kernels import SO2SteerableKernel
from gm_attention.nn.stencil_plan import StencilPlan
from gm_attention.representations.field_type import SO2FeatureField, SO2FieldType


class AttentionTables(NamedTuple):
    """The per-slot tables of :func:`attention_tables`, phantom slot included."""

    #: ``(n_heads, n_slots + 1, d_in, d_out)``: the transpose of the value kernel's matrix at each slot, zero at the phantom.
    matrices: Tensor
    #: ``(n_slots + 1,)``: ``log c(v_s) + log w_s`` inside the window, ``-inf`` outside it and at the phantom.
    bias: Tensor
    #: ``(n_phases, n_slots + 1)``: ``cos(f gamma_s)`` for each nonzero attention frequency ``f``, ascending.
    cosines: Tensor
    #: ``(n_phases, n_slots + 1)``: ``sin(f gamma_s)`` likewise.
    sines: Tensor


def _validate_field(x: SO2FeatureField, field_type: SO2FieldType, plan: StencilPlan, cutoff: float) -> int:
    """Shared entry checks; returns the number of leading batch axes."""
    if x.field_type != field_type:
        raise ValueError(f"Expected a field of type {field_type!r}, got {x.field_type!r}.")
    if x.batch_shape[-1:] != (plan.n_nodes,):
        raise ValueError(
            f"x.batch_shape must end in (n_nodes,) = ({plan.n_nodes},) for the "
            f"{plan.n_latitude} x {plan.n_longitude} grid, got {tuple(x.batch_shape)}."
        )
    if plan.dtype != x.dtype:
        raise ValueError(f"The plan's stencil is {plan.dtype} and x is {x.dtype}; move one with .to().")
    if not math.isclose(cutoff, plan.cutoff, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"The stencil was built at cutoff {plan.cutoff:.12g}, the layer has {cutoff:.12g}.")
    return len(x.batch_shape) - 1


def attention_tables(heads: Sequence[SO2Attention], plan: StencilPlan) -> AttentionTables:
    """The per-slot tables of one or more attention layers sharing a neighbourhood.

    Parameters
    ----------
    heads
        The layers, one for :func:`stencil_attention` or the heads of an
        :class:`~gm_attention.nn.attention.SO2MHSA`; they must share the
        cutoff, the window and the attention and input types, as the heads
        of an MHSA do.
    plan
        Over a stencil built at the layers' cutoff.

    Returns
    -------
    AttentionTables
        Differentiable with respect to every head's parameters.
    """
    if not heads:
        raise ValueError("heads must hold at least one layer.")
    first = heads[0]
    for head in heads[1:]:
        if (head.cutoff, head.window, head.field_type_in, head.field_type_attn, head.field_type_out) != (
            first.cutoff, first.window, first.field_type_in, first.field_type_attn, first.field_type_out
        ):
            raise ValueError("Every head must share the cutoff, window and field types.")
    if not math.isclose(first.cutoff, plan.cutoff, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"The stencil was built at cutoff {plan.cutoff:.12g}, the layer has {first.cutoff:.12g}.")

    stencil = plan.stencil
    geometry = stencil.geometry
    real_dtype = geometry.distances.dtype

    window = evaluate_aggregation_window(geometry.distances, first.cutoff, first.window)  # (n_slots,)
    tiny = torch.finfo(real_dtype).tiny
    bias = torch.log(window.clamp_min(tiny)) + torch.log(geometry.tangent_quadrature_weights.clamp_min(tiny))
    bias = bias.masked_fill(~(window > 0.0), -math.inf)

    matrices = torch.stack(
        [steerable_slot_matrices(cast(SO2SteerableKernel, head.value), stencil, transpose=True, pad=True) for head in heads]
    )  # (n_heads, n_slots + 1, d_in, d_out), zero at the phantom

    frequencies = [f for f in first.field_type_attn.frequencies if f != 0]
    if frequencies:
        phases = torch.stack([geometry.transport_harmonic(f) for f in frequencies])  # (n_phases, n_slots) complex
        cosines, sines = phases.real.contiguous(), phases.imag.contiguous()
    else:
        cosines = torch.zeros(0, stencil.n_slots, dtype=real_dtype, device=stencil.device)
        sines = torch.zeros_like(cosines)

    return AttentionTables(
        matrices=matrices,
        bias=plan.pad(bias, -math.inf),
        cosines=plan.pad(cosines, 1.0, dim=1),
        sines=plan.pad(sines, 0.0, dim=1),
    )


def _perpendicular(vectors: Tensor) -> Tensor:
    """``(x, y) -> (y, -x)`` on the packed ``(..., a, 2)`` pairs of a flat
    ``(..., 2a)`` slice: ``<q_perp, k>`` is the ``sin`` coefficient of
    ``<q, rho_f(gamma) k> = cos(f gamma) <q, k> + sin(f gamma) <q_perp, k>``."""
    pairs = vectors.reshape(*vectors.shape[:-1], -1, 2)
    return torch.stack((pairs[..., 1], -pairs[..., 0]), dim=-1).reshape(vectors.shape)


def attend(
    plan: StencilPlan,
    queries: Tensor,  # (batch, n_heads, n_nodes, d_attn)
    keys: Tensor,  # (batch, n_heads, n_nodes, d_attn)
    source: Tensor,  # (batch, n_nodes, d_in)
    tables: AttentionTables,
    field_type_attn: SO2FieldType,
    scale: float,
) -> Tensor:  # (batch, n_heads, n_nodes, d_out)
    """Score, normalize and aggregate every head on every output row at once.

    Parameters
    ----------
    plan
        The bucketed stencil.
    queries, keys
        The projected fields in the packed layout of ``field_type_attn``,
        one per head, nodes in the grid's row-major order.
    source
        The input field in its packed layout, shared by the heads.
    tables
        From :func:`attention_tables`, over the same plan.
    field_type_attn
        The type of the queries and keys: its frequencies and slices.
    scale
        ``1 / sqrt(dim rho_attn)``.

    Returns
    -------
    Tensor
        The attended values ``sum_j alpha_ij V(v_ij) rho_in(g_ij) x_j`` per
        head in the packed layout of the value kernel's output type.
    """
    batch, n_heads, n_nodes, d_attn = queries.shape
    d_in = source.shape[-1]
    d_out = tables.matrices.shape[-1]
    n_latitude, n_longitude = plan.n_latitude, plan.n_longitude
    slices = field_type_attn.slices()
    phase_index = {f: i for i, f in enumerate(f for f in field_type_attn.frequencies if f != 0)}

    dense_queries = queries.reshape(batch, n_heads, n_latitude, n_longitude, d_attn)
    outputs: list[Tensor] = []
    for bucket in plan.buckets:
        n_rows, width = bucket.n_rows, bucket.width
        flat_slots = bucket.slots.reshape(-1)  # (n_rows * width,)
        flat_source = bucket.source.reshape(-1)  # (n_rows * n_longitude * width,)

        query_rows = dense_queries.index_select(2, bucket.rows)  # (batch, n_heads, n_rows, n_longitude, d_attn)
        gathered_keys = keys.index_select(2, flat_source).reshape(
            batch, n_heads, n_rows, n_longitude, width, d_attn
        )

        # s = <q, rho(g_s) k> per frequency: cos(f gamma_s) <q, k> + sin(f gamma_s) <q_perp, k>. A
        # broadcast product and a reduction rather than a matmul: the contraction is over d_f alone
        # with one query per (row, longitude), a batch of tiny GEMMs on which CPU BLAS loops.
        score: Tensor | None = None
        for frequency in field_type_attn.frequencies:
            channels = slices[frequency]
            key_part = gathered_keys[..., channels]  # (batch, n_heads, n_rows, n_longitude, width, d_f)
            query_part = query_rows[..., channels].unsqueeze(-2)  # (batch, n_heads, n_rows, n_longitude, 1, d_f)
            dot = (key_part * query_part).sum(-1)  # (batch, n_heads, n_rows, n_longitude, width)
            if frequency == 0:
                term = dot
            else:
                cross = (key_part * _perpendicular(query_part)).sum(-1)
                cosine = tables.cosines[phase_index[frequency]].index_select(0, flat_slots).reshape(1, 1, n_rows, 1, width)
                sine = tables.sines[phase_index[frequency]].index_select(0, flat_slots).reshape(1, 1, n_rows, 1, width)
                term = dot * cosine + cross * sine
            score = term if score is None else score + term
        assert score is not None
        del gathered_keys

        bias = tables.bias.index_select(0, flat_slots).reshape(1, 1, n_rows, 1, width)
        # -inf outside the window and at the padding; every row holds its self-loop inside, so the softmax is finite.
        attention = torch.softmax(score * scale + bias, dim=-1)  # (batch, n_heads, n_rows, n_longitude, width)

        gathered_source = source.index_select(1, flat_source).reshape(batch, 1, n_rows, n_longitude, width, d_in)
        weighted = (attention.unsqueeze(-1) * gathered_source).reshape(
            batch, n_heads, n_rows, n_longitude, width * d_in
        )
        matrices = tables.matrices.index_select(1, flat_slots).reshape(n_heads, n_rows, width * d_in, d_out)
        outputs.append(torch.matmul(weighted, matrices))  # (batch, n_heads, n_rows, n_longitude, d_out)

    attended = torch.cat(outputs, dim=2).index_select(2, plan.inverse_order)  # rows back in grid order
    return attended.reshape(batch, n_heads, n_nodes, d_out)


def stencil_attention(module: SO2Attention, x: SO2FeatureField, plan: StencilPlan) -> SO2FeatureField:
    """:meth:`SO2Attention.forward` from a stencil plan on the dense layout, all rows at once.

    Parameters
    ----------
    module
        The layer; its parameters are used, nothing is owned here.
    x
        A field of ``module.field_type_in`` with ``batch_shape``
        ``(*lead, n_latitude * n_longitude)``, nodes in the grid's row-major
        order; the leading axes are independent samples.
    plan
        Over a stencil built at ``module.cutoff`` over the discretization the
        graph path would receive, in ``x``'s dtype.

    Returns
    -------
    SO2FeatureField
        Of ``module.field_type_out`` with ``x``'s batch shape; equal to the
        graph path's output, sample by sample, to summation order.
    """
    lead = _validate_field(x, module.field_type_in, plan, module.cutoff)
    batch = math.prod(x.batch_shape[:lead])
    n_nodes = plan.n_nodes

    tables = attention_tables([module], plan)
    queries = module.query(x).to_flat().reshape(batch, 1, n_nodes, -1)
    keys = module.key(x).to_flat().reshape(batch, 1, n_nodes, -1)
    source = x.to_flat().reshape(batch, n_nodes, -1)
    attended = attend(plan, queries, keys, source, tables, module.field_type_attn, module._score_scale)
    flat = attended.reshape(*x.batch_shape, -1)  # (*lead, n_nodes, d_out)
    return module._finish(x, SO2FeatureField.from_flat(module.field_type_out, flat).coefficients)


def stencil_mhsa(module: SO2MHSA, x: SO2FeatureField, plan: StencilPlan) -> SO2FeatureField:
    """:meth:`SO2MHSA.forward` from a stencil plan: every head in one pass of
    :func:`attend` over the heads' stacked queries, keys and slot matrices,
    the heads' outputs concatenated head-major within each frequency and
    passed through the output intertwiner, as the graph path."""
    lead = _validate_field(x, module.field_type_in, plan, module.cutoff)
    batch = math.prod(x.batch_shape[:lead])
    n_nodes = plan.n_nodes
    heads = [cast(SO2Attention, head) for head in module.heads]
    n_heads = len(heads)
    field_type_attn = module.field_type_attn

    tables = attention_tables(heads, plan)
    queries = torch.stack([head.query(x).to_flat() for head in heads], dim=-3).reshape(batch, n_heads, n_nodes, -1)
    keys = torch.stack([head.key(x).to_flat() for head in heads], dim=-3).reshape(batch, n_heads, n_nodes, -1)
    source = x.to_flat().reshape(batch, n_nodes, -1)
    attended = attend(plan, queries, keys, source, tables, field_type_attn, heads[0]._score_scale)

    # Head-major within each frequency: the layout of _concatenate_heads.
    slices = field_type_attn.slices()
    coefficients: dict[int, Tensor] = {}
    for frequency, multiplicity in field_type_attn.items():
        piece = attended[..., slices[frequency]]  # (batch, n_heads, n_nodes, a_f * d_f)
        if frequency == 0:
            coefficients[0] = piece.permute(0, 2, 1, 3).reshape(*x.batch_shape, n_heads * multiplicity)
        else:
            coefficients[frequency] = (
                piece.reshape(batch, n_heads, n_nodes, multiplicity, 2)
                .permute(0, 2, 1, 3, 4)
                .reshape(*x.batch_shape, n_heads * multiplicity, 2)
            )
    return module.output_projection(SO2FeatureField(module.field_type_in, coefficients))
