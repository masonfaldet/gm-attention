"""The steerable kernel's pair loop as fused terms (``nn/fused/pairs.h``).

Each ``(n, m, role)`` term of
:meth:`~gm_attention.nn.kernels.SO2SteerableKernel._forward_general` is one
kernel launch that forms its ``b_m x a_n`` block in registers from the trunk
output and the readout rows, applies phase and radial scale, and accumulates
into the output. The backward is one launch per term that produces the input
and trunk gradients in registers and its slice of the coefficient gradient;
the weight gradient is then a single wide GEMM
(``_blocked_weight_gradient``). Same values and gradients as the Python path
to summation order. Off unless ``GM_FUSED_PAIRS=1`` (CUDA tensors) or
``GM_FUSED_PAIRS=cpu`` (also the CPU driver); the extensions are built on
first use with ``torch.utils.cpp_extension.load`` (``ninja``, and ``nvcc``
for CUDA), one CUDA build per block size and trunk width met, one CPU build
of every size in ``SUPPORTED``. ``GM_FUSED_PAIRS_VERBOSE=1`` shows the
compiler output and ptxas' register and spill counts.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from gm_attention.representations import so2
from gm_attention.representations.field_type import SO2FeatureField

MATRIX_COMPLEX, U_REAL, CONJ_U, OUT_REAL = 1, 2, 4, 8
MAX_MULT = 16
MAX_TRUNK = 16
#: Block sides the CPU build instantiates (``GM_PAIRS_SIZES`` in pairs.h); a CUDA build is per size met, any side in 1..MAX_MULT.
SUPPORTED = (1, 2, 3, 4, 8, 16)

#: ``"0"`` off, ``"1"`` on for CUDA tensors, ``"cpu"`` on everywhere.
MODE = os.environ.get("GM_FUSED_PAIRS", "0")


@functools.cache
def extension(size: tuple[int, int, int] | None = None) -> Any:
    """Build (once, cached by torch) and import an extension: the CUDA kernels
    of one block size and trunk width ``size = (a, b, trunk)``, or, without a
    size, the CPU driver of every size in ``SUPPORTED``."""
    from torch.utils.cpp_extension import load

    source_dir = Path(__file__).parent / "fused"
    verbose = os.environ.get("GM_FUSED_PAIRS_VERBOSE", "0") == "1"
    sources = [str(source_dir / "pairs_cpu.cpp")]
    if size is None:
        return load(name="gm_pairs_cpu", sources=sources, extra_cflags=["-O3"], verbose=verbose)
    a, b, trunk = size
    defines = ["-DGM_PAIRS_WITH_CUDA", f"-DGM_PAIRS_A={a}", f"-DGM_PAIRS_B={b}", f"-DGM_PAIRS_K={trunk}"]
    # Compile for the visible devices only.  Left unset, torch adds a PTX
    # pass for the newest of them, which doubles the device compile for a
    # portability this cache never needs.
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", ";".join(_visible_archs()))
    return load(
        name=f"gm_pairs_cuda_{a}x{b}_k{trunk}",
        sources=[*sources, str(source_dir / "pairs_cuda.cu")],
        extra_cflags=["-O3", *defines],
        extra_cuda_cflags=["-O3", *defines, *(["--ptxas-options=-v"] if verbose else [])],
        verbose=verbose,
    )


def _extension_for(hidden: Tensor, term: Term) -> Any:
    """The extension serving ``term`` on ``hidden``'s device."""
    if hidden.is_cuda:
        return extension((term.a, term.b, hidden.shape[1]))
    return extension()


def _visible_archs() -> list[str]:
    """Compute capabilities of the visible devices, as ``TORCH_CUDA_ARCH_LIST`` entries."""
    archs = {torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())}
    return [f"{major}.{minor}" for major, minor in sorted(archs)]


@dataclass(frozen=True)
class Term:
    n: int
    m: int
    role: str
    flags: int
    a: int
    b: int
    real_head: bool  # rows of the real head (the (0, 0) block) or the complex head
    rows: slice
    pre_order: int | None
    post_order: int | None
    power: int | None


def plan(kernel: Any) -> list[Term]:
    """The terms of ``kernel._forward_general``'s pair loop, in its order."""
    cached = kernel.__dict__.get("_fused_plan")
    if cached is not None:
        return cached  # type: ignore[no-any-return]
    terms: list[Term] = []
    for m in kernel.field_type_out.frequencies:
        b = kernel.field_type_out.multiplicity(m)
        for n in kernel.field_type_in.frequencies:
            a = kernel.field_type_in.multiplicity(n)
            if n == 0 and m == 0:
                terms.append(Term(n, m, "real", U_REAL | OUT_REAL, a, b, True, slice(0, a * b), None, None, None))
            elif n == 0:
                terms.append(Term(n, m, "linear", MATRIX_COMPLEX | U_REAL, a, b, False,
                                  kernel._pair_ranges[(n, m, "linear")], None, m, kernel._pair_powers[(n, m, "linear")]))
            elif m == 0:
                # Re[A conj(e^{-in phi} z)] = Re[A (e^{in phi} conj z)]
                terms.append(Term(n, m, "linear", MATRIX_COMPLEX | CONJ_U | OUT_REAL, a, b, False,
                                  kernel._pair_ranges[(n, m, "linear")], n, None, kernel._pair_powers[(n, m, "linear")]))
            else:
                terms.append(Term(n, m, "linear", MATRIX_COMPLEX, a, b, False,
                                  kernel._pair_ranges[(n, m, "linear")], m - n, None, kernel._pair_powers[(n, m, "linear")]))
                terms.append(Term(n, m, "antilinear", MATRIX_COMPLEX | CONJ_U, a, b, False,
                                  kernel._pair_ranges[(n, m, "antilinear")], m + n, None, kernel._pair_powers[(n, m, "antilinear")]))
    kernel.__dict__["_fused_plan"] = terms
    return terms


def enabled(kernel: Any, x: SO2FeatureField, edges: Any) -> bool:
    """Whether this call takes the fused path: the mode allows the device,
    every block and the trunk fit the kernels' register caps, the dtype is
    float32/64, and every phase order is tabulated on ``edges``."""
    if MODE == "0":
        return False
    if MODE != "cpu" and x.device.type != "cuda":
        return False
    if x.dtype not in (torch.float32, torch.float64):
        return False
    trunk = kernel.complex_head.in_features if kernel.complex_head is not None else kernel.real_head.in_features
    if not 1 <= trunk <= MAX_TRUNK:
        return False
    per_size = x.device.type == "cuda"  # a CUDA build per size met; the CPU build has its fixed set
    for term in plan(kernel):
        if per_size:
            if not (1 <= term.a <= MAX_MULT and 1 <= term.b <= MAX_MULT):
                return False
        elif term.a not in SUPPORTED or term.b not in SUPPORTED:
            return False
        for order in (term.pre_order, term.post_order):
            if order is not None and abs(order) > edges.max_frequency:
                return False
    return True


def _empty(like: Tensor) -> Tensor:
    return like.new_empty(0)


class _FusedPairs(torch.autograd.Function):
    """All terms of one kernel call; see the module docstring."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any, hidden: Tensor, complex_weight: Tensor | None, complex_bias: Tensor | None,
        real_weight: Tensor | None, real_bias: Tensor | None, meta: dict[str, Any],
        phases: tuple[Tensor, ...], scales: tuple[Tensor, ...], *vectors: Tensor,
    ) -> tuple[Tensor, ...]:
        terms: list[Term] = meta["terms"]
        n_edges = hidden.shape[0]
        outputs: list[Tensor] = []
        for m in meta["out_frequencies"]:
            b = meta["out_multiplicity"][m]
            outputs.append(
                torch.zeros(n_edges, b, dtype=hidden.dtype if m == 0 else so2.complex_dtype(hidden.dtype), device=hidden.device)
            )
        for term, (phase_pre, phase_post, scale) in zip(terms, meta["tables"], strict=True):
            weight, bias = _rows(term, complex_weight, complex_bias, real_weight, real_bias)
            _extension_for(hidden, term).forward(
                hidden, weight, bias if bias is not None else _empty(hidden), vectors[meta["in_index"][term.n]],
                phases[phase_pre] if phase_pre is not None else _empty(hidden),
                phases[phase_post] if phase_post is not None else _empty(hidden),
                scales[scale] if scale is not None else _empty(hidden),
                outputs[meta["out_index"][term.m]], term.a, term.b, term.flags,
            )
        ctx.save_for_backward(hidden, complex_weight, complex_bias, real_weight, real_bias, *phases, *scales, *vectors)
        ctx.meta = meta
        ctx.counts = (len(phases), len(scales), len(vectors))
        return tuple(outputs)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor | None) -> tuple[Any, ...]:  # type: ignore[override]
        meta = ctx.meta
        n_phases, n_scales, _n_vectors = ctx.counts
        saved = ctx.saved_tensors
        hidden, complex_weight, complex_bias, real_weight, real_bias = saved[:5]
        phases = saved[5:5 + n_phases]
        scales = saved[5 + n_phases:5 + n_phases + n_scales]
        vectors = saved[5 + n_phases + n_scales:]
        terms: list[Term] = meta["terms"]
        n_edges = hidden.shape[0]
        grads_out = [
            g if g is not None else torch.zeros(n_edges, meta["out_multiplicity"][m], dtype=hidden.dtype if m == 0 else so2.complex_dtype(hidden.dtype), device=hidden.device)
            for g, m in zip(grad_outputs, meta["out_frequencies"], strict=True)
        ]
        grads_out = [g.contiguous() for g in grads_out]
        grad_hidden = torch.zeros_like(hidden)
        grad_vectors = [torch.zeros_like(v) for v in vectors]
        # Coefficient gradients, coefficient-major (width, n_edges): a term's
        # rows are a contiguous slab, and a warp's stores land in one line.
        grad_complex = hidden.new_empty(complex_weight.shape[0], n_edges) if complex_weight is not None else None
        grad_real = hidden.new_empty(real_weight.shape[0], n_edges) if real_weight is not None else None
        for term, (phase_pre, phase_post, scale) in zip(terms, meta["tables"], strict=True):
            weight, bias = _rows(term, complex_weight, complex_bias, real_weight, real_bias)
            grad_k = (grad_real if term.real_head else grad_complex)
            assert grad_k is not None
            _extension_for(hidden, term).backward(
                hidden, weight, bias if bias is not None else _empty(hidden), vectors[meta["in_index"][term.n]],
                phases[phase_pre] if phase_pre is not None else _empty(hidden),
                phases[phase_post] if phase_post is not None else _empty(hidden),
                scales[scale] if scale is not None else _empty(hidden),
                grads_out[meta["out_index"][term.m]], grad_vectors[meta["in_index"][term.n]], grad_hidden,
                grad_k[term.rows], term.a, term.b, term.flags,
            )
        from gm_attention.nn.kernels import SPLIT_K_ROWS, _blocked_weight_gradient

        # The bias gradient is the coefficient gradient summed over edges: a
        # column of ones on the trunk output folds it into the weight GEMM.
        with_bias = complex_bias is not None or real_bias is not None
        operand = torch.cat((hidden, hidden.new_ones(n_edges, 1)), dim=1) if with_bias else hidden

        def head_gradients(grad_k: Tensor | None, weight: Tensor | None, bias: Tensor | None) -> tuple[Tensor | None, Tensor | None]:
            if grad_k is None or weight is None:
                return None, None
            grad = _blocked_weight_gradient(grad_k.transpose(0, 1), operand, SPLIT_K_ROWS)  # a view: no copy
            if not with_bias:
                return grad, None
            return grad[:, :-1], grad[:, -1] if bias is not None else None

        grad_cw, grad_cb = head_gradients(grad_complex, complex_weight, complex_bias)
        grad_rw, grad_rb = head_gradients(grad_real, real_weight, real_bias)
        return (grad_hidden, grad_cw, grad_cb, grad_rw, grad_rb, None, None, None, *grad_vectors)


def _rows(term: Term, cw: Tensor | None, cb: Tensor | None, rw: Tensor | None, rb: Tensor | None) -> tuple[Tensor, Tensor | None]:
    weight, bias = (rw, rb) if term.real_head else (cw, cb)
    assert weight is not None
    return weight[term.rows].contiguous(), None if bias is None else bias[term.rows].contiguous()


def forward_general(kernel: Any, x: SO2FeatureField, edges: Any) -> SO2FeatureField:
    """``kernel._forward_general(x, edges)`` through the fused terms."""
    from gm_attention.nn.kernels import apply_radial_trunk, transport_field

    terms = plan(kernel)
    with torch.profiler.record_function("gm::kernel/radial"):
        embedding = kernel.radial_basis(edges.distances)
        hidden = apply_radial_trunk(kernel.radial_trunk, embedding).contiguous()
    transported_field = transport_field(x, edges)
    in_frequencies = list(kernel.field_type_in.frequencies)
    out_frequencies = list(kernel.field_type_out.frequencies)
    vectors = tuple(
        (transported_field[0] if n == 0 else so2.as_complex(transported_field[n])).contiguous() for n in in_frequencies
    )
    orders = sorted({o for t in terms for o in (t.pre_order, t.post_order) if o is not None})
    phases = tuple(edges.angular_harmonic(o).contiguous() for o in orders)
    powers_needed = sorted({t.power for t in terms if t.power is not None})
    radial_powers = kernel._radial_powers(edges.distances) if powers_needed else None
    scales = tuple(radial_powers[:, p].contiguous() for p in powers_needed)  # type: ignore[index]
    order_index = {o: i for i, o in enumerate(orders)}
    power_index = {p: i for i, p in enumerate(powers_needed)}
    meta = {
        "terms": terms,
        "tables": [
            (
                order_index[t.pre_order] if t.pre_order is not None else None,
                order_index[t.post_order] if t.post_order is not None else None,
                power_index[t.power] if t.power is not None else None,
            )
            for t in terms
        ],
        "in_index": {n: i for i, n in enumerate(in_frequencies)},
        "out_index": {m: i for i, m in enumerate(out_frequencies)},
        "out_frequencies": out_frequencies,
        "out_multiplicity": {m: kernel.field_type_out.multiplicity(m) for m in out_frequencies},
    }
    with torch.profiler.record_function("gm::kernel/pairs"):
        outputs = _FusedPairs.apply(
            hidden,
            kernel.complex_head.weight if kernel.complex_head is not None else None,
            kernel.complex_head.bias if kernel.complex_head is not None else None,
            kernel.real_head.weight if kernel.real_head is not None else None,
            kernel.real_head.bias if kernel.real_head is not None else None,
            meta, phases, scales, *vectors,
        )
    return SO2FeatureField(
        kernel.field_type_out,
        {m: (out if m == 0 else so2.as_real(out)) for m, out in zip(out_frequencies, outputs, strict=True)},
    )


__all__ = ["MODE", "Term", "enabled", "extension", "forward_general", "plan"]
