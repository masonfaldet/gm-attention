"""The factored steerable kernel as one fused kernel per call (``nn/fused/factored.h``).

The edges are grouped by source; one thread block stages the source's mixes
``Z(j)`` in shared memory, and each of its edges forms every block's
``sum_t h~_t Z_t(j)`` in registers under its own phases and radial powers,
writing only its output row. The kernel reads the trunk's own hidden
features (the constant feature of ``h~ = (1, h)`` is implicit) and forms
each block's radial power from the edge's ``r / cutoff`` in registers, so no
per-edge concatenation or power table is built for it. :class:`Factored` is
an autograd function whose backward is the same grouping with the roles
swapped; its gradients reach the trunk features and the node-side mixes,
from which autograd handles the trunk, the heads' weights and the input
field. The phases and the radius ratio are the geometry's and get no
gradient.

What every layer does with the messages is sum them per target under
per-edge weights (the attention weights, or a convolution's window times
quadrature weight). :class:`FactoredAggregate` (:func:`forward_aggregated`,
:meth:`SO2SteerableKernel.forward_factored_aggregated`) is that op whole:
the rows once, a weighted segment sum over them, and a backward that reads
each target's gradient through the edge's target index, scales it by the
edge's weight and forms the weight's gradient from the rows the forward
kept, so no weighted per-edge tensor, gathered gradient or per-edge
multiply pass exists in either direction. Off unless ``GM_FUSED_FACTORED=1``
(CUDA tensors) or ``GM_FUSED_FACTORED=cpu`` (also the CPU driver); built on
first use with ``torch.utils.cpp_extension.load`` (``ninja``, and ``nvcc``
for CUDA). ``GM_FUSED_FACTORED_VERBOSE=1`` shows the compiler output.
"""

from __future__ import annotations

import functools
import os
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from gm_attention.representations.field_type import SO2FeatureField

MAX_K, MAX_OUT, MAX_BM, MAX_C, DESC = 20, 128, 32, 320, 8
#: The routines come in two tiers (``factored.h``): narrow for blocks up to 16 wide and mixes up to 256,
#: wide up to ``MAX_BM`` and ``MAX_C``; the drivers pick the tier per call from the plan's widest block
#: and the packed width, so a narrow layer runs the same code either way.
NARROW_BM, NARROW_C = 16, 256
#: ``"0"`` off, ``"1"`` on for CUDA tensors, ``"cpu"`` on everywhere.
MODE = os.environ.get("GM_FUSED_FACTORED", "0")


@functools.cache
def extension(cuda: bool = False) -> Any:
    """Build (once, cached by torch) and import the extension: the CPU driver,
    or the CPU driver plus the CUDA kernel."""
    from torch.utils.cpp_extension import load

    source_dir = Path(__file__).parent / "fused"
    verbose = os.environ.get("GM_FUSED_FACTORED_VERBOSE", "0") == "1"
    sources = [str(source_dir / "factored_cpu.cpp")]
    if not cuda:
        return load(name="gm_factored_cpu", sources=sources, extra_cflags=["-O3"], verbose=verbose)
    defines = ["-DGM_FACTORED_WITH_CUDA"]
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", ";".join(_visible_archs()))
    return load(
        name="gm_factored_cuda",
        sources=[*sources, str(source_dir / "factored_cuda.cu")],
        extra_cflags=["-O3", *defines],
        extra_cuda_cflags=["-O3", *defines, *(["--ptxas-options=-v"] if verbose else [])],
        verbose=verbose,
    )


def _visible_archs() -> list[str]:
    archs = {torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())}
    return [f"{major}.{minor}" for major, minor in sorted(archs)]


@dataclass(frozen=True)
class Block:
    """One coefficient block's descriptor row (``factored.h`` Column)."""

    block: tuple[int, int, str]
    kind: int
    z_offset: int
    b_m: int
    angular: int
    transport: int
    power: int
    out_offset: int
    take_real: int

    def row(self) -> list[int]:
        return [self.kind, self.z_offset, self.b_m, self.angular, self.transport, self.power, self.out_offset, self.take_real]


@dataclass(frozen=True)
class Plan:
    blocks: tuple[Block, ...]
    wc: int  # complex mixes per node per t
    wr: int  # real mixes per node per t
    d_out: int
    max_bm: int  # the widest block's output multiplicity: the forward's tier


def plan(kernel: Any) -> Plan:
    """The kernel's blocks as descriptor rows, grouped by output frequency;
    the complex blocks' mixes packed side by side in that order."""
    cached = kernel.__dict__.get("_fused_factored_plan")
    if cached is not None:
        return cached  # type: ignore[no-any-return]
    slices = kernel.field_type_out.slices()
    # Stable sort by output frequency: the routine accumulates a group in
    # registers and writes each output slot once; the mixes pack in this order.
    ordered = sorted(kernel._blocks(), key=lambda block: slices[block[1]].start)
    blocks: list[Block] = []
    wc = 0
    wr = 0
    for n, m, role in ordered:
        b_m = kernel.field_type_out.multiplicity(m)
        if role == "real":
            blocks.append(Block((n, m, role), 0, 0, b_m, 0, 0, 0, slices[0].start, 0))
            wr = b_m
            continue
        if n == 0:
            angular, transport = m, 0
        elif m == 0:
            angular, transport = n, -n
        elif role == "linear":
            angular, transport = m - n, n
        else:
            angular, transport = m + n, -n
        blocks.append(Block((n, m, role), 1, wc, b_m, angular, transport, int(kernel._pair_powers[(n, m, role)]),
                            slices[m].start, int(m == 0)))
        wc += b_m
    result = Plan(tuple(blocks), wc, wr, kernel.field_type_out.dimension, max((b.b_m for b in blocks), default=1))
    kernel.__dict__["_fused_factored_plan"] = result
    return result


def descriptors(kernel: Any, device: torch.device) -> Tensor:
    """The plan as an int32 ``(n_blocks, 8)`` tensor, cached per device."""
    cache = kernel.__dict__.setdefault("_fused_factored_descriptors", {})
    key = str(device)
    if key not in cache:
        cache[key] = torch.tensor([b.row() for b in plan(kernel).blocks], dtype=torch.int32, device=device)
    return cache[key]  # type: ignore[no-any-return]


#: Key under which a ``node_products`` dict caches its packed form, shared across a forward's chunks.
PACKED = "_packed"


def pack(kernel: Any, products: dict) -> tuple[Tensor, Tensor]:
    """``node_products`` as the extension's ``zc`` ``(n_nodes, K, wc, 2)`` and
    ``zr`` ``(n_nodes, K, wr)``, remembered on the dict under :data:`PACKED`."""
    packed = products.get(PACKED)
    if packed is not None:
        return packed  # type: ignore[no-any-return]
    with torch.profiler.record_function("gm::kernel/pack"):
        packed = _pack(kernel, products)
    products[PACKED] = packed
    return packed


def _pack(kernel: Any, products: dict) -> tuple[Tensor, Tensor]:
    p = plan(kernel)
    complex_blocks = [b.block for b in p.blocks if b.kind == 1]
    first = next(v for key, v in products.items() if key != PACKED)
    n_nodes, k = first.shape[0], first.shape[1]
    real_dtype = torch.float64 if first.dtype in (torch.complex128, torch.float64) else torch.float32
    if complex_blocks:
        zc = torch.view_as_real(torch.cat([products[b] for b in complex_blocks], dim=-1).contiguous()).contiguous()
    else:
        zc = torch.zeros(n_nodes, k, 0, 2, dtype=real_dtype, device=first.device)
    real = [b.block for b in p.blocks if b.kind == 0]
    zr = products[real[0]].contiguous() if real else torch.zeros(n_nodes, k, 0, dtype=real_dtype, device=first.device)
    return zc, zr


#: Bounded cache of source groupings keyed by caller; each entry holds a weak reference to its owner, since ids and addresses recur.
_GROUPS: OrderedDict[tuple, tuple[Any, tuple[Tensor, Tensor, Tensor]]] = OrderedDict()
GROUP_CACHE_SIZE = 512


def source_groups(source_index: Tensor, key: tuple | None = None,
                  owner: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
    """``(order, group_ptr, group_source)``: the edges sorted by source, the
    half-open ranges of each source's run in that order, and the sources.
    Cached under ``key`` when given (a chunk of a persistent geometry), the
    entry tied to ``owner`` (its edge tensor) when given."""
    if key is not None:
        hit = _GROUPS.get(key)
        if hit is not None:
            held, groups = hit
            if _same_owner(held, owner) and groups[0].device == source_index.device:
                _GROUPS.move_to_end(key)
                return groups
    order = torch.argsort(source_index, stable=True)
    sorted_sources = source_index[order]
    group_source, counts = torch.unique_consecutive(sorted_sources, return_counts=True)
    group_ptr = torch.zeros(group_source.shape[0] + 1, dtype=torch.long, device=source_index.device)
    group_ptr[1:] = torch.cumsum(counts, dim=0)
    result = (order.contiguous(), group_ptr.contiguous(), group_source.contiguous())
    if key is not None:
        _GROUPS[key] = (None if owner is None else weakref.ref(owner), result)
        while len(_GROUPS) > GROUP_CACHE_SIZE:
            _GROUPS.popitem(last=False)
    return result


def _same_owner(held: Any, owner: Tensor | None) -> bool:
    """Whether a cached entry's owner (a weak reference, or None) is the
    caller's: an entry made without an owner serves ownerless calls only, and
    one made for a tensor serves that tensor while it lives."""
    if held is None:
        return owner is None
    referent = held()
    return referent is not None and referent is owner


def group_key(parent: Any, first: int, last: int) -> tuple:
    """A cache key for the chunk ``[first, last)`` of a persistent geometry:
    its identity, its edge tensor's address and size, and the range."""
    return (id(parent), parent.edge_index.data_ptr(), int(parent.n_edges), int(first), int(last))


def enabled(kernel: Any, x: SO2FeatureField, edges: Any) -> bool:
    """Whether :meth:`forward_factored` takes this path: the mode allows the
    device, the dtype is float32/64, and the kernel's widths fit the
    routine's register and shared-memory arrays."""
    if MODE == "0":
        return False
    if MODE != "cpu" and x.device.type != "cuda":
        return False
    if x.dtype not in (torch.float32, torch.float64):
        return False
    p = plan(kernel)
    trunk = kernel.complex_head.in_features if kernel.complex_head is not None else kernel.real_head.in_features
    if trunk + 1 > MAX_K or p.d_out > MAX_OUT or p.max_bm > MAX_BM or 2 * p.wc + p.wr > MAX_C:
        return False
    return edges.n_edges > 0


class Factored(torch.autograd.Function):
    """The extension's forward and backward as one differentiable op on
    ``(h, zc, zr)``: the rows ``(n_edges, d_out)`` out; the phases, the
    radius ratio, descriptors and groups are constants of the geometry and
    the kernel."""

    @staticmethod
    def forward(ctx: Any, h: Tensor, zc: Tensor, zr: Tensor, angular: Tensor, transport: Tensor, rho: Tensor, desc: Tensor,
                source: Tensor, order: Tensor, group_ptr: Tensor, group_source: Tensor, d_out: int, max_bm: int) -> Tensor:
        ext = extension(cuda=h.is_cuda)
        with torch.profiler.record_function("gm::kernel/pairs"):
            out = ext.forward(h, zc, zr, angular, transport, rho, desc, source, order, group_ptr, group_source, d_out, max_bm)
        ctx.save_for_backward(h, zc, zr, angular, transport, rho, desc, source, order, group_ptr, group_source)
        return out  # type: ignore[no-any-return]

    @staticmethod
    @torch.autograd.function.once_differentiable  # a double backward raises instead of returning detached gradients
    def backward(ctx: Any, grad_out: Tensor) -> tuple:  # type: ignore[override]
        h, zc, zr, angular, transport, rho, desc, source, order, group_ptr, group_source = ctx.saved_tensors
        ext = extension(cuda=h.is_cuda)
        with torch.profiler.record_function("gm::kernel/pairs_backward"):
            dh, dzc, dzr = ext.backward(grad_out.contiguous(), h, zc, zr, angular, transport, rho, desc, source, order,
                                        group_ptr, group_source)
        return dh, dzc, dzr, None, None, None, None, None, None, None, None, None, None


class FactoredAggregate(torch.autograd.Function):
    """The rows summed per target under per-edge ``weights`` as one
    differentiable op on ``(h, zc, zr, weights)``: ``(n_targets, d_out)``
    out. The forward keeps the rows; the backward takes the targets'
    gradient and returns the weights' gradient alongside the kernel's."""

    @staticmethod
    def forward(ctx: Any, h: Tensor, zc: Tensor, zr: Tensor, weights: Tensor, angular: Tensor, transport: Tensor, rho: Tensor,
                desc: Tensor, source: Tensor, target: Tensor, row_ptr: Tensor, order: Tensor, group_ptr: Tensor,
                group_source: Tensor, d_out: int, max_bm: int) -> Tensor:
        ext = extension(cuda=h.is_cuda)
        with torch.profiler.record_function("gm::kernel/pairs"):
            rows = ext.forward(h, zc, zr, angular, transport, rho, desc, source, order, group_ptr, group_source, d_out, max_bm)
        with torch.profiler.record_function("gm::kernel/aggregate"):
            out = ext.aggregate(rows, weights, row_ptr)
        ctx.save_for_backward(h, zc, zr, weights, rows, angular, transport, rho, desc, source, target, order, group_ptr, group_source)
        return out  # type: ignore[no-any-return]

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx: Any, grad_out: Tensor) -> tuple:  # type: ignore[override]
        h, zc, zr, weights, rows, angular, transport, rho, desc, source, target, order, group_ptr, group_source = ctx.saved_tensors
        ext = extension(cuda=h.is_cuda)
        with torch.profiler.record_function("gm::kernel/pairs_backward"):
            dh, dzc, dzr, dweights = ext.backward_aggregated(grad_out.contiguous(), target, weights, rows, h, zc, zr, angular,
                                                             transport, rho, desc, source, order, group_ptr, group_source)
        return dh, dzc, dzr, dweights, None, None, None, None, None, None, None, None, None, None, None, None


@dataclass(frozen=True)
class Inputs:
    """One call's kernel inputs: the trunk's features, the packed mixes, the
    phase tables and radius ratio in the features' dtype, the descriptors,
    the sources and the plan."""

    h: Tensor  # (n_edges, T)
    zc: Tensor
    zr: Tensor
    angular: Tensor
    transport: Tensor
    rho: Tensor  # (n_edges,)
    desc: Tensor
    source: Tensor
    plan: Plan


def inputs(kernel: Any, x: SO2FeatureField, edges: Any, products: dict[tuple[int, int, str], Tensor] | None) -> Inputs:
    """The extension's per-call inputs for ``kernel`` over ``edges`` on the
    node field ``x`` (its mixes ``products`` when given), the phase tables
    checked against the blocks' orders."""
    if products is None:
        products = kernel.node_products(x)
    h = kernel._trunk_hidden(edges)
    zc, zr = pack(kernel, products)
    rho = (edges.distances / kernel.radial_basis.cutoff).to(h.dtype).contiguous()
    angular = torch.view_as_real(edges.angular_phases.contiguous()).contiguous().to(h.dtype)
    transport = torch.view_as_real(edges.transport_phases.contiguous()).contiguous().to(h.dtype)
    p = plan(kernel)
    order_needed = max((max(abs(b.angular), abs(b.transport)) for b in p.blocks), default=0)
    if order_needed >= angular.shape[1] or order_needed >= transport.shape[1]:
        raise ValueError(f"the geometry's phase tables ({angular.shape[1]} orders) are narrower than the kernel's blocks need "
                         f"(order {order_needed})")
    return Inputs(h.contiguous(), zc, zr, angular, transport, rho, descriptors(kernel, h.device), edges.source_index.contiguous(), p)


def groups_for(edges: Any, source: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """The source grouping of ``edges`` from the cache, keyed on the
    persistent geometry: the registered parent and range of a chunk
    (:func:`fused_attention.chunk_of`), else the edge set itself over all
    its edges. What a caller that holds no grouping of its own gets, so an
    unchunked layer's forward neither re-sorts its edges nor synchronizes
    on their unique sources each call."""
    from gm_attention.nn import (
        fused_attention,  # here, not at import: attention imports both modules
    )

    registered = fused_attention.chunk_of(edges)
    parent, first, last = registered if registered is not None else (edges, 0, int(edges.n_edges))
    return source_groups(source, group_key(parent, first, last), owner=parent.edge_index)


def forward(kernel: Any, x: SO2FeatureField, edges: Any, products: dict[tuple[int, int, str], Tensor] | None = None,
            groups: tuple[Tensor, Tensor, Tensor] | None = None) -> SO2FeatureField:
    """:meth:`SO2SteerableKernel.forward_factored` through the extension:
    the messages on the edges, ``(n_edges,)`` of the kernel's output type."""
    i = inputs(kernel, x, edges, products)
    order, group_ptr, group_source = groups_for(edges, i.source) if groups is None else groups
    flat = Factored.apply(i.h, i.zc, i.zr, i.angular, i.transport, i.rho, i.desc, i.source, order, group_ptr, group_source,
                          i.plan.d_out, i.plan.max_bm)
    return SO2FeatureField.from_flat(kernel.field_type_out, flat)


def forward_aggregated(kernel: Any, x: SO2FeatureField, edges: Any, weights: Tensor,
                       products: dict[tuple[int, int, str], Tensor] | None = None,
                       groups: tuple[Tensor, Tensor, Tensor] | None = None) -> SO2FeatureField:
    """:meth:`SO2SteerableKernel.forward_factored_aggregated` through the
    extension: each target's weighted sum of its messages, ``(n_targets,)``
    of the kernel's output type (``n_targets`` from ``edges.row_ptr``)."""
    i = inputs(kernel, x, edges, products)
    order, group_ptr, group_source = groups_for(edges, i.source) if groups is None else groups
    flat = FactoredAggregate.apply(i.h, i.zc, i.zr, weights.to(i.h.dtype).contiguous(), i.angular, i.transport, i.rho, i.desc,
                                   i.source, edges.target_index.contiguous(), edges.row_ptr.contiguous(), order, group_ptr,
                                   group_source, i.plan.d_out, i.plan.max_bm)
    return SO2FeatureField.from_flat(kernel.field_type_out, flat)
