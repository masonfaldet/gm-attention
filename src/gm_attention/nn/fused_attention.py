"""The scores-and-softmax op and the depthwise convolution as fused kernels
(``nn/fused/attention.h``).

Both run one thread block per target node. :class:`Softmax` (for the
attention, whose value messages come from the factored steerable kernel)
stages the node's query, scores its edges into shared memory under the
window and quadrature bias and normalizes there: the attention weights
``alpha`` out, and from ``d alpha`` the softmax adjoint and the score's
backward, ``dq`` and ``dk``. :class:`Depthwise` (the blocks' mix) walks the
neighbourhood one thread per output component; per edge only ``v_j``, the
depthwise coefficients, the phases and the aggregation weight (window times
quadrature weight, optionally normalized by the target's window mass) move.
The radial MLP producing the coefficients stays in PyTorch, so the ops
differentiate with respect to the flat node fields and the coefficients
only; the weights and phases are the geometry's and get no gradient. The
source-side gradients (``dk``, ``dv``) accumulate with atomic adds on CUDA,
so their summation order is not fixed run to run. Off unless
``GM_FUSED_ATTENTION=1`` (CUDA tensors) or ``GM_FUSED_ATTENTION=cpu`` (also
the CPU driver); built on first use with ``torch.utils.cpp_extension.load``.
``GM_FUSED_ATTENTION_VERBOSE=1`` shows the compiler output.
"""

from __future__ import annotations

import functools
import os
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from gm_attention.nn._scatter import scatter_add_to_nodes

MAX_H, MAX_F, MAX_D, MAX_SLOTS, DESC = 8, 8, 512, 8192, 5
#: The CUDA kernels' block (attention_cuda.cu THREADS) and the dynamic
#: shared memory a V100 grants one block (the backward stages two copies
#: of the neighbourhood's slots beside the query and gradient rows).
THREADS, MAX_SMEM = 128, 96 * 1024
#: ``"0"`` off, ``"1"`` on for CUDA tensors, ``"cpu"`` on everywhere.
MODE = os.environ.get("GM_FUSED_ATTENTION", "0")


@functools.cache
def extension(cuda: bool = False) -> Any:
    """Build (once, cached by torch) and import the extension: the CPU driver,
    or the CPU driver plus the CUDA kernels."""
    from torch.utils.cpp_extension import load

    source_dir = Path(__file__).parent / "fused"
    verbose = os.environ.get("GM_FUSED_ATTENTION_VERBOSE", "0") == "1"
    sources = [str(source_dir / "attention_cpu.cpp")]
    if not cuda:
        return load(name="gm_fused_attention_cpu", sources=sources, extra_cflags=["-O3"], verbose=verbose)
    defines = ["-DGM_ATTENTION_WITH_CUDA"]
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", ";".join(_visible_archs()))
    return load(
        name="gm_fused_attention_cuda",
        sources=[*sources, str(source_dir / "attention_cuda.cu")],
        extra_cflags=["-O3", *defines],
        extra_cuda_cflags=["-O3", *defines, *(["--ptxas-options=-v"] if verbose else [])],
        verbose=verbose,
    )


def _visible_archs() -> list[str]:
    archs = {torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())}
    return [f"{major}.{minor}" for major, minor in sorted(archs)]


@dataclass(frozen=True)
class Plan:
    """The heads' layout as the kernel reads it: one descriptor row per
    frequency ``(n, a_n, flat offset, coefficient offset, dim)``."""

    rows: tuple[tuple[int, int, int, int, int], ...]
    heads: int
    width: int  # D, the flat width of the heads' direct sum
    a_0: int  # scalar copies per head
    complex_width: int  # W, a head's complex coefficients per edge: 4 a_n per frequency n >= 1
    max_order: int  # the highest phase order read: max(n, 2n)


def plan(mhsa: Any) -> Plan:
    """The multi-head layer's plan: each head's type ``field_type_attn``
    inside the heads' direct sum ``field_type_in``."""
    return _plan(mhsa, mhsa.field_type_attn, mhsa.field_type_in, mhsa.num_heads)


def plan_depthwise(conv: Any) -> Plan:
    """A depthwise convolution's plan: one kernel over its own type."""
    return _plan(conv, conv.kernel.field_type, conv.kernel.field_type, 1)


def _plan(owner: Any, attn: Any, full: Any, heads: int) -> Plan:
    cached = owner.__dict__.get("_fused_attention_plan")
    if cached is not None:
        return cached  # type: ignore[no-any-return]
    slices = full.slices()
    rows: list[tuple[int, int, int, int, int]] = []
    coff = 0
    for n, a_n in attn.items():
        if n == 0:
            rows.append((0, a_n, slices[0].start, 0, 1))
        else:
            rows.append((n, a_n, slices[n].start, coff, 2))
            coff += 4 * a_n
    result = Plan(tuple(rows), heads, full.dimension, attn.multiplicity(0) if 0 in attn else 0, coff,
                  max((2 * n for n, _ in attn.items()), default=0))
    owner.__dict__["_fused_attention_plan"] = result
    return result


def descriptors(owner: Any, p: Plan, device: torch.device) -> Tensor:
    key = ("_fused_attention_desc", str(device))
    cached = owner.__dict__.get(key)
    if cached is None:
        cached = torch.tensor([list(row) for row in p.rows], dtype=torch.int64, device=device)
        owner.__dict__[key] = cached
    return cached  # type: ignore[no-any-return]


@dataclass
class Geometry:
    """The per-edge constants of one edge set in one dtype under one cutoff
    and window: the phase tables as real pairs, ``r / cutoff``, the
    aggregation weights ``c(r) w`` (and, on demand, their normalized form
    per ``eps``), the softmax bias ``log c(r) + log w`` with its support
    mask, and the largest neighbourhood."""

    transport: Tensor  # (n_edges, orders, 2)
    angular: Tensor  # (n_edges, orders, 2)
    rho: Tensor  # (n_edges,)
    weights: Tensor  # (n_edges,)
    bias: Tensor  # (n_edges,)
    inside: Tensor  # (n_edges,) uint8
    row_ptr: Tensor  # (n_nodes + 1,) int64
    source: Tensor  # (n_edges,) int64
    max_degree: int
    normalized: dict[float, Tensor] = field(default_factory=dict)
    #: a chunk's geometry: views into its parent's tables over ``span``
    base: Geometry | None = None
    span: tuple[int, int] | None = None
    #: the largest neighbourhood of each chunk taken of this (persistent) geometry
    chunk_degrees: dict[tuple[int, int], int] = field(default_factory=dict)

    def aggregation_weights(self, normalize: bool, eps: float) -> Tensor:
        """``c(r) w``, divided by each target's window mass (clamped at
        ``eps``) when ``normalize``; SO2Convolution._aggregate's weighting.
        A chunk holds whole neighbourhoods, so its masses are its parent's."""
        if self.base is not None and self.span is not None:
            return self.base.aggregation_weights(normalize, eps)[self.span[0]:self.span[1]]
        if not normalize:
            return self.weights
        cached = self.normalized.get(eps)
        if cached is None:
            mass = scatter_add_to_nodes(self.weights, self.row_ptr).clamp_min(eps)  # (n_nodes,)
            target = torch.repeat_interleave(torch.arange(mass.shape[0], device=mass.device), self.row_ptr[1:] - self.row_ptr[:-1])
            cached = (self.weights / mass.index_select(0, target)).contiguous()
            self.normalized[eps] = cached
        return cached


_GEOMETRIES: OrderedDict[tuple, tuple[Any, Geometry]] = OrderedDict()
_GEOMETRY_CACHE = 64
#: The chunks of persistent edge sets now alive (``register_chunk``): the
#: chunk's ``row_ptr`` identity -> (its weak reference, the parent, first,
#: last), so a chunk's geometry is its parent's sliced rather than rebuilt
#: and held per chunk (an entry per chunk per forward once filled a card).
_CHUNKS: dict[int, tuple[Any, Any, int, int]] = {}


def register_chunk(chunk: Any, parent: Any, first: int, last: int) -> None:
    """Record that ``chunk`` is ``parent.select(arange(first, last))``, the
    edges ``first .. last`` of ``parent`` in order (the chunking of
    ``forward_chunked``), so :func:`geometry` serves it from the parent's."""
    _purge()
    _CHUNKS[id(chunk.row_ptr)] = (weakref.ref(chunk.row_ptr), parent, int(first), int(last))


def chunk_of(edges: Any) -> tuple[Any, int, int] | None:
    """``edges``'s registration as a chunk (:func:`register_chunk`): its
    parent and edge range ``(parent, first, last)``, or ``None`` when it is
    not a live registered chunk."""
    registered = _CHUNKS.get(id(edges.row_ptr))
    if registered is None or registered[0]() is not edges.row_ptr:
        return None
    _, parent, first, last = registered
    return parent, first, last


def _purge() -> None:
    """Drop the entries whose edge set has died."""
    for key in [k for k, (held, _) in _GEOMETRIES.items() if held() is None]:
        del _GEOMETRIES[key]
    for key in [k for k, (held, _, _, _) in _CHUNKS.items() if held() is None]:
        del _CHUNKS[key]


def geometry(edges: Any, cutoff: float, window: str, dtype: torch.dtype) -> Geometry:
    """The constants for ``edges``, kept for as long as the edge set lives
    (keyed by its ``row_ptr`` tensor, checked by weak reference) so a
    hierarchy's edge sets are prepared once. A registered chunk
    (:func:`register_chunk`) gets views of its parent's tables, its own
    ``row_ptr`` and sources, and a largest neighbourhood remembered per
    range on the parent; nothing is copied or cached per chunk."""
    _purge()
    registered = _CHUNKS.get(id(edges.row_ptr))
    if registered is not None and registered[0]() is edges.row_ptr:
        _, parent, first, last = registered
        base = geometry(parent, cutoff, window, dtype)
        degree = base.chunk_degrees.get((first, last))
        if degree is None:
            row_ptr = edges.row_ptr
            degree = int((row_ptr[1:] - row_ptr[:-1]).max()) if row_ptr.numel() > 1 else 0
            base.chunk_degrees[(first, last)] = degree
        return Geometry(
            base.transport[first:last], base.angular[first:last], base.rho[first:last], base.weights[first:last],
            base.bias[first:last], base.inside[first:last], edges.row_ptr.contiguous(), edges.source_index.contiguous(),
            degree, base=base, span=(first, last),
        )
    key = (id(edges.row_ptr), float(cutoff), window, dtype, str(edges.row_ptr.device))
    held = _GEOMETRIES.get(key)
    if held is not None and held[0]() is edges.row_ptr:
        _GEOMETRIES.move_to_end(key)
        return held[1]
    from gm_attention.nn.convolution import (
        evaluate_aggregation_window,  # here, not at import: convolution imports this module
    )

    tiny = torch.finfo(dtype).tiny
    distances = edges.distances.to(dtype)
    quadrature = edges.tangent_quadrature_weights.to(dtype)
    window_values = evaluate_aggregation_window(distances, cutoff, window)  # type: ignore[arg-type]
    bias = torch.log(window_values.clamp_min(tiny)) + torch.log(quadrature.clamp_min(tiny))
    row_ptr = edges.row_ptr.contiguous()
    max_degree = int((row_ptr[1:] - row_ptr[:-1]).max()) if row_ptr.numel() > 1 else 0
    result = Geometry(
        torch.view_as_real(edges.transport_phases.contiguous()).to(dtype).contiguous(),
        torch.view_as_real(edges.angular_phases.contiguous()).to(dtype).contiguous(),
        (distances / cutoff).contiguous(),
        (window_values * quadrature).contiguous(),
        bias.contiguous(),
        (window_values > 0.0).to(torch.uint8).contiguous(),
        row_ptr,
        edges.source_index.contiguous(),
        max_degree,
    )
    _GEOMETRIES[key] = (weakref.ref(edges.row_ptr), result)
    while len(_GEOMETRIES) > _GEOMETRY_CACHE:
        _GEOMETRIES.popitem(last=False)
    return result


def softmax_enabled(mhsa: Any, x: Any, edges: Any) -> bool:
    """Whether :func:`attention._attend_batched` takes the scores-and-softmax
    op for a layer whose values are not separable (steerable value kernels,
    whose messages come from the factored kernel): the mode allows the
    device, the dtype is float32/64, and the layout fits the bounds. Only
    the transport phases of the heads' own orders are read."""
    if MODE == "0":
        return False
    if MODE != "cpu" and x.device.type != "cuda":
        return False
    if x.dtype not in (torch.float32, torch.float64):
        return False
    if mhsa.num_heads > MAX_H:
        return False
    p = plan(mhsa)
    if len(p.rows) > MAX_F or p.width > MAX_D or max((n for n, _, _, _, _ in p.rows), default=0) > edges.max_frequency:
        return False
    if edges.n_edges == 0:
        return False
    slots = geometry(edges, mhsa.cutoff, mhsa.heads[0].window, x.dtype).max_degree * mhsa.num_heads
    smem_backward = (p.width + 2 * slots + (THREADS // 32) * mhsa.num_heads) * torch.finfo(x.dtype).bits // 8
    limit = MAX_SMEM
    if x.device.type == "cuda":
        limit = getattr(torch.cuda.get_device_properties(x.device), "shared_memory_per_block_optin", MAX_SMEM)
    return slots <= MAX_SLOTS and smem_backward <= limit


def plan_head(head: Any) -> Plan:
    """One head's plan: its own type ``field_type_attn`` as the whole
    (one head), for the per-head attention path."""
    return _plan(head, head.field_type_attn, head.field_type_attn, 1)


def softmax_head_enabled(head: Any, x: Any, edges: Any) -> bool:
    """:func:`softmax_enabled` for one head (``SO2Attention._attend``)."""
    if MODE == "0":
        return False
    if MODE != "cpu" and x.device.type != "cuda":
        return False
    if x.dtype not in (torch.float32, torch.float64):
        return False
    p = plan_head(head)
    if len(p.rows) > MAX_F or p.width > MAX_D or max((n for n, _, _, _, _ in p.rows), default=0) > edges.max_frequency:
        return False
    if edges.n_edges == 0:
        return False
    slots = geometry(edges, head.cutoff, head.window, x.dtype).max_degree
    smem_backward = (p.width + 2 * slots + (THREADS // 32)) * torch.finfo(x.dtype).bits // 8
    limit = MAX_SMEM
    if x.device.type == "cuda":
        limit = getattr(torch.cuda.get_device_properties(x.device), "shared_memory_per_block_optin", MAX_SMEM)
    return slots <= MAX_SLOTS and smem_backward <= limit


def enabled_depthwise(conv: Any, x: Any, edges: Any) -> bool:
    """Whether ``SO2Convolution._aggregate`` takes this path for a depthwise
    layer: the mode allows the device, the dtype is float32/64, and the
    layout fits the kernel's bounds (no neighbourhood bound: the walk holds
    nothing per edge)."""
    if MODE == "0":
        return False
    if MODE != "cpu" and x.device.type != "cuda":
        return False
    if x.dtype not in (torch.float32, torch.float64):
        return False
    p = plan_depthwise(conv)
    if len(p.rows) > MAX_F or p.width > MAX_D or p.max_order > edges.max_frequency:
        return False
    return edges.n_edges > 0


class Depthwise(torch.autograd.Function):
    """The depthwise convolution's forward and backward as one differentiable
    op on the flat node field ``v`` and the per-edge coefficients
    ``(rc, cc)``; the weights, the geometry and the descriptors are constants."""

    @staticmethod
    def forward(ctx: Any, v: Tensor, rc: Tensor, cc: Tensor, transport: Tensor, angular: Tensor, rho: Tensor,
                weights: Tensor, desc: Tensor, row_ptr: Tensor, source: Tensor, a_0: int, complex_width: int) -> Tensor:
        ext = extension(cuda=v.is_cuda)
        with torch.profiler.record_function("gm::conv/fused"):
            out = ext.conv_forward(v, rc, cc, transport, angular, rho, weights, desc, row_ptr, source, a_0, complex_width)
        ctx.save_for_backward(v, rc, cc, transport, angular, rho, weights, desc, row_ptr, source)
        ctx.ints = (a_0, complex_width)
        return out  # type: ignore[no-any-return]

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx: Any, grad_out: Tensor) -> tuple:  # type: ignore[override]
        v, rc, cc, transport, angular, rho, weights, desc, row_ptr, source = ctx.saved_tensors
        a_0, complex_width = ctx.ints
        ext = extension(cuda=v.is_cuda)
        with torch.profiler.record_function("gm::conv/fused_backward"):
            dv, drc, dcc = ext.conv_backward(grad_out.contiguous(), v, rc, cc, transport, angular, rho, weights, desc,
                                             row_ptr, source, a_0, complex_width)
        return dv, drc, dcc, None, None, None, None, None, None, None, None, None


def forward_depthwise(conv: Any, v: Tensor, real_coefficients: Tensor | None, complex_coefficients: Tensor | None,
                      edges: Any) -> Tensor:
    """The depthwise convolution over ``edges`` on the flat node field
    ``(n_nodes, D)`` with the kernel's coefficients ``(1, n_edges, a_0)`` and
    ``(1, n_edges, W)``; returns the aggregated flat field ``(n_nodes, D)``."""
    p = plan_depthwise(conv)
    g = geometry(edges, conv.cutoff, conv.window, v.dtype)
    empty = v.new_zeros(0)
    rc = empty if real_coefficients is None else real_coefficients.contiguous()
    cc = empty if complex_coefficients is None else complex_coefficients.contiguous()
    weights = g.aggregation_weights(bool(conv.normalize), float(conv.eps))
    return Depthwise.apply(v.contiguous(), rc, cc, g.transport, g.angular, g.rho, weights,  # type: ignore[no-any-return]
                           descriptors(conv, p, v.device), g.row_ptr, g.source, p.a_0, p.complex_width)


class Softmax(torch.autograd.Function):
    """The heads' scores and neighbourhood softmax as one differentiable op
    on the flat node fields ``(q, k)``: ``alpha (n_edges, H)`` out, and from
    ``d alpha`` the gradients ``dq``, ``dk`` (the softmax adjoint and the
    score's backward, per target). The geometry and the descriptors are
    constants. The key-side gradient accumulates with atomic adds on CUDA."""

    @staticmethod
    def forward(ctx: Any, q: Tensor, k: Tensor, transport: Tensor, bias: Tensor, inside: Tensor, desc: Tensor,
                row_ptr: Tensor, source: Tensor, heads: int, scale: float, max_degree: int) -> Tensor:
        ext = extension(cuda=q.is_cuda)
        with torch.profiler.record_function("gm::attention/fused_softmax"):
            alpha = ext.softmax_forward(q, k, transport, bias, inside, desc, row_ptr, source, heads, scale, max_degree)
        ctx.save_for_backward(q, k, transport, bias, inside, desc, row_ptr, source, alpha)
        ctx.ints = (heads, scale, max_degree)
        return alpha  # type: ignore[no-any-return]

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx: Any, grad_alpha: Tensor) -> tuple:  # type: ignore[override]
        q, k, transport, bias, inside, desc, row_ptr, source, alpha = ctx.saved_tensors
        heads, scale, max_degree = ctx.ints
        ext = extension(cuda=q.is_cuda)
        with torch.profiler.record_function("gm::attention/fused_softmax_backward"):
            dq, dk = ext.softmax_backward(grad_alpha.contiguous(), q, k, transport, bias, inside, desc, row_ptr, source,
                                          alpha, heads, scale, max_degree)
        return dq, dk, None, None, None, None, None, None, None, None, None


def softmax(mhsa: Any, q: Tensor, k: Tensor, edges: Any) -> Tensor:
    """The heads' attention weights ``(n_edges, n_heads)`` over ``edges``
    from the flat projected queries and keys (``(n_nodes, D)`` each,
    head-major within each frequency): the scores, the window and
    quadrature bias and the per-target softmax of
    :func:`attention._attend_batched`, in one op per direction."""
    p = plan(mhsa)
    g = geometry(edges, mhsa.cutoff, mhsa.heads[0].window, q.dtype)
    scale = float(mhsa.heads[0]._score_scale)
    return Softmax.apply(q.contiguous(), k.contiguous(), g.transport, g.bias, g.inside, descriptors(mhsa, p, q.device),  # type: ignore[no-any-return]
                         g.row_ptr, g.source, p.heads, scale, max(g.max_degree, 1))


def softmax_head(head: Any, q: Tensor, k: Tensor, edges: Any) -> Tensor:
    """One head's attention weights ``(n_edges,)`` over ``edges`` from its
    flat projected queries and keys (``(n_nodes, dim rho_attn)`` each):
    :func:`softmax` with one head, for ``SO2Attention._attend``."""
    p = plan_head(head)
    g = geometry(edges, head.cutoff, head.window, q.dtype)
    scale = float(head._score_scale)
    alpha = Softmax.apply(q.contiguous(), k.contiguous(), g.transport, g.bias, g.inside, descriptors(head, p, q.device),
                          g.row_ptr, g.source, 1, scale, max(g.max_degree, 1))
    return alpha.reshape(-1)  # type: ignore[no-any-return]
