"""A Triton forward kernel for the stencil attention, in torch-harmonics' style. UNTESTED.

.. warning::

    This module was written on a machine without a GPU and has never been
    run. It is kept beside the tested PyTorch path of
    :mod:`gm_attention.nn.stencil_attention` as the sketch of the fused
    kernel that path is designed to feed, not as a supported feature. Treat
    every line as unverified until ``tests/nn/test_stencil_attention.py`` is
    extended with a CUDA comparison against :func:`~gm_attention.nn.stencil_attention.attend`.

The kernel follows the neighbourhood attention of torch-harmonics: one
program per ``(batch * head, output row, longitude tile)``, a loop over the
row's slots with an online softmax (running maximum, running normalizer, a
``(tile, d_out)`` accumulator rescaled when the maximum moves), the source
tile of every slot read at ``(source_row[s], (wo + offset[s]) mod
n_longitude)``, and the value kernel's ``(d_in, d_out)`` matrix read from
the per-slot table of :func:`~gm_attention.nn.stencil_attention.attention_tables`
and applied with one ``tl.dot`` per slot. The transport of the keys is
``cos(f gamma_s) <q, k> + sin(f gamma_s) <q_perp, k>`` as in the PyTorch
path; the host expands the per-frequency tables to one cosine and one sine
per channel and precomputes ``q_perp`` per node. No per-edge tensor exists
in either the forward or, when one is written, the backward. Only the
forward is provided: calling the function under autograd raises, since no
backward kernel exists yet.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from gm_attention.nn.stencil_attention import AttentionTables
from gm_attention.nn.stencil_plan import StencilPlan
from gm_attention.representations.field_type import SO2FieldType

try:  # pragma: no cover - the import is environment dependent
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


def triton_available() -> bool:
    """Whether Triton is importable and a CUDA device is present."""
    return triton is not None and torch.cuda.is_available()


if triton is not None:  # pragma: no cover - never executed on a machine without Triton

    @triton.jit
    def _stencil_attention_forward_kernel(
        # fields, contiguous: queries/perpendicular/keys (BH, n_nodes, D_ATTN), source (B, n_nodes, D_IN), out (BH, n_nodes, D_OUT)
        queries_ptr, perpendicular_ptr, keys_ptr, source_ptr, out_ptr,
        # per-slot tables: cosines/sines (S + 1, D_ATTN), bias (S + 1,), matrices (H, S + 1, D_IN, D_OUT)
        cosines_ptr, sines_ptr, bias_ptr, matrices_ptr,
        # stencil: row_ptr (n_latitude + 1,), source_row (S,), offset (S,)
        row_ptr_ptr, source_row_ptr, offset_ptr,
        n_heads, n_nodes, n_longitude, n_slots_padded, scale,
        D_ATTN: tl.constexpr, D_IN: tl.constexpr, D_OUT: tl.constexpr,
        BLOCK_W: tl.constexpr, BLOCK_ATTN: tl.constexpr, BLOCK_IN: tl.constexpr, BLOCK_OUT: tl.constexpr,
    ):
        program_bh = tl.program_id(0)  # batch * n_heads + head
        row = tl.program_id(1)  # output row ho
        tile = tl.program_id(2)  # longitude tile
        batch = program_bh // n_heads
        head = program_bh % n_heads

        longitudes = tile * BLOCK_W + tl.arange(0, BLOCK_W)  # (BLOCK_W,)
        longitude_mask = longitudes < n_longitude
        targets = row * n_longitude + longitudes  # (BLOCK_W,) target nodes

        attn_channels = tl.arange(0, BLOCK_ATTN)
        attn_mask = attn_channels < D_ATTN
        in_channels = tl.arange(0, BLOCK_IN)
        in_mask = in_channels < D_IN
        out_channels = tl.arange(0, BLOCK_OUT)
        out_mask = out_channels < D_OUT

        # The queries of the tile, (BLOCK_W, BLOCK_ATTN), and their perpendiculars.
        query_offsets = (program_bh * n_nodes + targets)[:, None] * D_ATTN + attn_channels[None, :]
        tile_mask = longitude_mask[:, None] & attn_mask[None, :]
        queries = tl.load(queries_ptr + query_offsets, mask=tile_mask, other=0.0)
        perpendicular = tl.load(perpendicular_ptr + query_offsets, mask=tile_mask, other=0.0)

        running_max = tl.full((BLOCK_W,), -1.0e30, dtype=tl.float32)
        running_sum = tl.zeros((BLOCK_W,), dtype=tl.float32)
        accumulator = tl.zeros((BLOCK_W, BLOCK_OUT), dtype=tl.float32)

        first = tl.load(row_ptr_ptr + row)
        last = tl.load(row_ptr_ptr + row + 1)
        for slot in range(first, last):
            source_row = tl.load(source_row_ptr + slot)
            offset = tl.load(offset_ptr + slot)
            sources = source_row * n_longitude + (longitudes + offset) % n_longitude  # (BLOCK_W,)

            # s_w = scale * sum_c k[c] (cos_c q[c] + sin_c q_perp[c]) + bias_s
            cosines = tl.load(cosines_ptr + slot * D_ATTN + attn_channels, mask=attn_mask, other=0.0)
            sines = tl.load(sines_ptr + slot * D_ATTN + attn_channels, mask=attn_mask, other=0.0)
            rotated = queries * cosines[None, :] + perpendicular * sines[None, :]  # (BLOCK_W, BLOCK_ATTN)
            key_offsets = (program_bh * n_nodes + sources)[:, None] * D_ATTN + attn_channels[None, :]
            keys = tl.load(keys_ptr + key_offsets, mask=tile_mask, other=0.0)
            score = tl.sum(keys * rotated, axis=1) * scale + tl.load(bias_ptr + slot)  # (BLOCK_W,)

            # Online softmax: rescale the running sums when the maximum moves.
            new_max = tl.maximum(running_max, score)
            correction = tl.exp(running_max - new_max)
            weight = tl.exp(score - new_max)  # 0 at a -inf bias (outside the window)
            running_sum = running_sum * correction + weight
            running_max = new_max

            # The slot's message on the tile: x_tile @ V_s^T, (BLOCK_W, BLOCK_OUT).
            source_offsets = (batch * n_nodes + sources)[:, None] * D_IN + in_channels[None, :]
            source_tile = tl.load(source_ptr + source_offsets, mask=longitude_mask[:, None] & in_mask[None, :], other=0.0)
            matrix_offsets = ((head * n_slots_padded + slot) * D_IN + in_channels)[:, None] * D_OUT + out_channels[None, :]
            matrix = tl.load(matrices_ptr + matrix_offsets, mask=in_mask[:, None] & out_mask[None, :], other=0.0)
            message = tl.dot(source_tile, matrix)  # (BLOCK_W, BLOCK_OUT)
            accumulator = accumulator * correction[:, None] + weight[:, None] * message

        result = accumulator / running_sum[:, None]
        out_offsets = (program_bh * n_nodes + targets)[:, None] * D_OUT + out_channels[None, :]
        tl.store(out_ptr + out_offsets, result, mask=longitude_mask[:, None] & out_mask[None, :])


def _per_channel_phases(tables: AttentionTables, field_type_attn: SO2FieldType) -> tuple[Tensor, Tensor]:
    """``(n_slots + 1, d_attn)`` cosines and sines: each channel's frequency
    table, ``cos = 1, sin = 0`` on the frequency-0 channels."""
    n_slots_padded = tables.bias.shape[0]
    d_attn = field_type_attn.dimension
    cosines = torch.ones(n_slots_padded, d_attn, dtype=tables.bias.dtype, device=tables.bias.device)
    sines = torch.zeros_like(cosines)
    phase_index = 0
    for frequency, channels in field_type_attn.slices().items():
        if frequency == 0:
            continue
        cosines[:, channels] = tables.cosines[phase_index].unsqueeze(1)
        sines[:, channels] = tables.sines[phase_index].unsqueeze(1)
        phase_index += 1
    return cosines, sines


def _perpendicular(queries: Tensor, field_type_attn: SO2FieldType) -> Tensor:
    """``(x, y) -> (y, -x)`` on every non-scalar copy, zero on the scalar channels."""
    perpendicular = torch.zeros_like(queries)
    for frequency, channels in field_type_attn.slices().items():
        if frequency == 0:
            continue
        pairs = queries[..., channels].reshape(*queries.shape[:-1], -1, 2)
        perpendicular[..., channels] = torch.stack((pairs[..., 1], -pairs[..., 0]), dim=-1).reshape(pairs.shape[:-2] + (-1,))
    return perpendicular


def attend_triton(
    plan: StencilPlan,
    queries: Tensor,  # (batch, n_heads, n_nodes, d_attn)
    keys: Tensor,  # (batch, n_heads, n_nodes, d_attn)
    source: Tensor,  # (batch, n_nodes, d_in)
    tables: AttentionTables,
    field_type_attn: SO2FieldType,
    scale: float,
    *,
    block_longitudes: int = 32,
) -> Tensor:  # (batch, n_heads, n_nodes, d_out)
    """:func:`~gm_attention.nn.stencil_attention.attend` through the Triton kernel. UNTESTED, forward only.

    Raises
    ------
    RuntimeError
        Without Triton or a CUDA device, or if any input requires a
        gradient (no backward kernel exists).
    """
    if not triton_available():
        raise RuntimeError("attend_triton needs Triton and a CUDA device; use gm_attention.nn.stencil_attention.attend.")
    if torch.is_grad_enabled() and any(t.requires_grad for t in (queries, keys, source, tables.matrices, tables.bias)):
        raise RuntimeError("attend_triton has no backward kernel; call it under torch.no_grad().")
    device = queries.device
    batch, n_heads, n_nodes, d_attn = queries.shape
    d_in = source.shape[-1]
    d_out = tables.matrices.shape[-1]
    n_longitude = plan.n_longitude
    stencil = plan.stencil

    dtype = torch.float32
    queries32 = queries.reshape(batch * n_heads, n_nodes, d_attn).to(dtype).contiguous()
    perpendicular32 = _perpendicular(queries32, field_type_attn).contiguous()
    keys32 = keys.reshape(batch * n_heads, n_nodes, d_attn).to(dtype).contiguous()
    source32 = source.to(dtype).contiguous()
    cosines, sines = _per_channel_phases(tables, field_type_attn)
    out = torch.empty(batch * n_heads, n_nodes, d_out, dtype=dtype, device=device)

    def block(size: int) -> int:
        return max(16, 1 << (size - 1).bit_length())

    grid = (batch * n_heads, plan.n_latitude, math.ceil(n_longitude / block_longitudes))
    _stencil_attention_forward_kernel[grid](
        queries32, perpendicular32, keys32, source32, out,
        cosines.to(dtype).contiguous(), sines.to(dtype).contiguous(), tables.bias.to(dtype).contiguous(),
        tables.matrices.to(dtype).contiguous(),
        stencil.row_ptr.to(torch.int32), stencil.source_row.to(torch.int32), stencil.offset.to(torch.int32),
        n_heads, n_nodes, n_longitude, tables.matrices.shape[1], float(scale),
        D_ATTN=d_attn, D_IN=d_in, D_OUT=d_out,
        BLOCK_W=block_longitudes, BLOCK_ATTN=block(d_attn), BLOCK_IN=block(d_in), BLOCK_OUT=block(d_out),
    )
    return out.reshape(batch, n_heads, n_nodes, d_out).to(queries.dtype)
