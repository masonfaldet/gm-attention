"""GM-convolution on a fixed tensor-product grid as one fixed linear operator per bucket.

:func:`stencil_convolution` evaluates
:class:`~gm_attention.nn.convolution.SO2Convolution`, with its general
:class:`~gm_attention.nn.kernels.SO2SteerableKernel` or its depthwise
:class:`~gm_attention.nn.depthwise.SO2DepthwiseKernel`, on the dense
``(..., n_latitude, n_longitude, channels)`` layout from a
:class:`~gm_attention.nn.stencil_plan.StencilPlan`. Without a softmax the
aggregation has no data-dependent weights, so once the parameters are fixed
the whole stencil contraction is a fixed linear operator in the sense of a
DISCO convolution: a table of per-slot kernel matrices, each multiplied by
its slot's window value and tangent quadrature weight, applied to the
gathered sources and summed over the slots. :func:`convolution_tables` builds
the table from one evaluation of the radial MLP per slot (through
:func:`~gm_attention.nn.stencil_kernel.steerable_slot_matrices` for the general
kernel and :func:`depthwise_kernel_matrices` for the depthwise one, transport
included) and the per-row window mass of a normalized layer; :func:`aggregate`
is the dense part with static shapes: per bucket one ``index_select`` of the
neighbourhood and, for the general kernel, one matmul of the gathered sources
``(rows, n_longitude, width * d_in)`` against the stacked slot matrices
``(rows, width * d_in, d_out)``, the contraction over slots and channels as a
single GEMM. The depthwise kernel acts on each irrep copy alone, by one real
scalar per slot and scalar copy and one real ``2 x 2`` matrix per slot and
non-scalar copy, so its contraction is per copy and costs ``2 d_in`` rather
than ``d_out d_in`` multiplications per slot; it is the mix of the
segmentation model's blocks.

The self-interaction is added pointwise outside the sum and the
normalization divides by each row's window mass, as the graph path does. The
result agrees with :meth:`SO2Convolution.forward` to summation order,
gradients included (``tests/nn/test_stencil_convolution.py``), and is
exactly equivariant to longitude shifts; :func:`aggregate` compiles with
``torch.compile(fullgraph=True)``.
"""

from __future__ import annotations

import math
from typing import NamedTuple, cast

import torch
from torch import Tensor

from gm_attention.discretization.stencil import TensorGridStencil
from gm_attention.nn.convolution import SO2Convolution, evaluate_aggregation_window
from gm_attention.nn.depthwise import SO2DepthwiseKernel
from gm_attention.nn.stencil_kernel import steerable_slot_matrices
from gm_attention.nn.kernels import SO2SteerableKernel, apply_radial_trunk, split_k_linear
from gm_attention.nn.stencil_plan import StencilPlan
from gm_attention.representations import so2
from gm_attention.representations.field_type import SO2FeatureField, SO2FieldType


def _real_2x2(linear: Tensor, antilinear: Tensor) -> Tensor:  # (..., 2, 2)
    """The real matrix of ``z -> P z + Q conj(z)`` on ``(re, im)`` for complex
    ``P`` (``linear``) and ``Q`` (``antilinear``) of a common shape: the
    input ``re`` maps to ``P + Q``, the input ``im`` to ``i (P - Q)``."""
    total, difference = linear + antilinear, linear - antilinear
    first_row = torch.stack((total.real, -difference.imag), dim=-1)
    second_row = torch.stack((total.imag, difference.real), dim=-1)
    return torch.stack((first_row, second_row), dim=-2)


def depthwise_kernel_matrices(kernel: SO2DepthwiseKernel, stencil: TensorGridStencil) -> dict[int, Tensor]:
    """The depthwise kernel's value map at every slot, per frequency, transport included.

    Returns ``{0: (n_slots, a_0)}`` real coefficients ``C_a(r)`` for the
    scalar copies and, for every frequency ``n >= 1`` of the type, the
    ``(n_slots, a_n, 2, 2)`` real matrices of ``x -> P x + Q conj(x)`` with
    ``P = A_a(r) e^{i n gamma}`` and ``Q = (r/c)^{2n} B_a(r) e^{2 i n phi}
    e^{-i n gamma}``: the message of :meth:`SO2DepthwiseKernel.forward` on
    an edge of that slot as a function of the untransported source copy. The
    radial MLP runs once per slot, and the tables differentiate with respect
    to the kernel's parameters as the forward does.
    """
    geometry = stencil.geometry
    kernel.validate_edges(geometry)
    field_type = kernel.field_type
    n_slots = geometry.n_edges

    hidden = apply_radial_trunk(kernel.radial_trunk, kernel.radial_basis(geometry.distances))  # (n_slots, width)
    tables: dict[int, Tensor] = {}
    if kernel.real_head is not None:
        tables[0] = split_k_linear(hidden, kernel.real_head.weight, kernel.real_head.bias)  # (n_slots, a_0)
    if kernel.complex_head is not None:
        chunks = torch.split(
            split_k_linear(hidden, kernel.complex_head.weight, kernel.complex_head.bias), kernel._complex_widths, dim=1
        )
        normalized = geometry.distances / kernel.cutoff
        for frequency, chunk in zip(kernel._complex_frequencies, chunks, strict=True):
            multiplicity = field_type.multiplicity(frequency)
            coefficients = chunk.reshape(n_slots, 2, multiplicity, 2)
            linear = so2.as_complex(coefficients[:, 0])  # A, (n_slots, a_n)
            antilinear = so2.as_complex(coefficients[:, 1])  # B, (n_slots, a_n)
            # (r/c)^{2n} by repeated multiplication, as the kernel: differentiable at r = 0.
            scale = torch.ones_like(normalized)
            for _ in range(2 * frequency):
                scale = scale * normalized
            transport = geometry.transport_harmonic(frequency).unsqueeze(-1)  # e^{i n gamma}, (n_slots, 1)
            phase = (scale * geometry.angular_harmonic(2 * frequency)).unsqueeze(-1)  # (r/c)^{2n} e^{2 i n phi}
            tables[frequency] = _real_2x2(linear * transport, phase * antilinear * transport.conj())
    return tables


class ConvolutionTables(NamedTuple):
    """The per-slot tables of :func:`convolution_tables`, phantom slot included.

    A depthwise layer's tables follow the packed layout: the scalar copies
    first, then every non-scalar copy of every frequency as one contiguous
    run of ``(re, im)`` pairs, so one table covers all of them.
    """

    #: General kernel: ``(n_slots + 1, d_in, d_out)``, ``c(v_s) w_s`` times the transposed kernel matrix, zero at the phantom; ``None`` for a depthwise layer.
    matrices: Tensor | None
    #: Depthwise kernel: ``(n_slots + 1, a_0)``, ``c(v_s) w_s`` times the scalar copies' coefficients, zero at the phantom; ``None`` without scalar copies or for a general layer.
    depthwise_scalar: Tensor | None
    #: Depthwise kernel: ``(n_slots + 1, sum_{n >= 1} a_n, 2, 2)``, ``c(v_s) w_s`` times the real matrix of every non-scalar copy, frequencies ascending, zero at the phantom; ``None`` without such copies or for a general layer.
    depthwise_pairs: Tensor | None
    #: ``(n_latitude,)``: each row's window mass ``sum_s c(v_s) w_s`` clamped at ``eps``, or ``None`` for an unnormalized layer.
    denominator: Tensor | None


def convolution_tables(module: SO2Convolution, plan: StencilPlan) -> ConvolutionTables:
    """The per-slot tables of a convolution layer over a plan built at its cutoff.

    Differentiable with respect to the kernel's parameters; the window, the
    quadrature weights and the normalization carry none.
    """
    if not math.isclose(module.cutoff, plan.cutoff, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"The stencil was built at cutoff {plan.cutoff:.12g}, the layer has {module.cutoff:.12g}.")
    stencil = plan.stencil
    geometry = stencil.geometry

    # c(v_s) w_s: the window times the tangent quadrature weight, one scalar per slot.
    weight = evaluate_aggregation_window(geometry.distances, module.cutoff, module.window) * geometry.tangent_quadrature_weights

    denominator: Tensor | None = None
    if module.normalize:
        row_of_slot = torch.repeat_interleave(
            torch.arange(stencil.n_latitude, device=stencil.device), stencil.degrees()
        )
        mass = torch.zeros(stencil.n_latitude, dtype=weight.dtype, device=weight.device).index_add(0, row_of_slot, weight)
        denominator = mass.clamp_min(module.eps)

    if isinstance(module.kernel, SO2DepthwiseKernel):
        tables = depthwise_kernel_matrices(module.kernel, stencil)
        scalar = tables.get(0)
        pairs = [tables[frequency] for frequency in module.field_type_in.frequencies if frequency != 0]
        return ConvolutionTables(
            matrices=None,
            depthwise_scalar=None if scalar is None else plan.pad(scalar * weight.unsqueeze(-1), 0.0),
            depthwise_pairs=None if not pairs else plan.pad(torch.cat(pairs, dim=1) * weight.reshape(-1, 1, 1, 1), 0.0),
            denominator=denominator,
        )

    # c(v_s) w_s V(v_s)^T per slot, (n_slots + 1, d_in, d_out) with a zero matrix at the phantom.
    matrices = steerable_slot_matrices(
        cast(SO2SteerableKernel, module.kernel), stencil, transpose=True, pad=True, scale=weight
    )
    return ConvolutionTables(matrices=matrices, depthwise_scalar=None, depthwise_pairs=None, denominator=denominator)


def aggregate(
    plan: StencilPlan,
    source: Tensor,  # (batch, n_nodes, d_in)
    tables: ConvolutionTables,
    field_type_in: SO2FieldType,
) -> Tensor:  # (batch, n_nodes, d_out)
    """The weighted sum over every target's slots, all rows at once.

    Parameters
    ----------
    plan
        The bucketed stencil.
    source
        The input field in its packed layout, nodes in the grid's row-major
        order.
    tables
        From :func:`convolution_tables`, over the same plan.
    field_type_in
        The input type: how many scalar copies lead the packed layout (the
        depthwise contraction treats them apart from the pairs).

    Returns
    -------
    Tensor
        ``sum_j c(v_ij) w_ij V(v_ij) rho_in(g_ij) x_j`` per node in the
        packed layout of the kernel's output type, divided by the row's
        window mass when the layer normalizes.
    """
    batch, n_nodes, d_in = source.shape
    n_latitude, n_longitude = plan.n_latitude, plan.n_longitude
    n_scalar = field_type_in.multiplicity(0) if 0 in field_type_in else 0
    n_pairs = (d_in - n_scalar) // 2

    outputs: list[Tensor] = []
    for bucket in plan.buckets:
        n_rows, width = bucket.n_rows, bucket.width
        flat_slots = bucket.slots.reshape(-1)
        flat_source = bucket.source.reshape(-1)
        gathered = source.index_select(1, flat_source).reshape(batch, n_rows, n_longitude, width, d_in)

        if tables.matrices is not None:
            matrices = tables.matrices.index_select(0, flat_slots).reshape(n_rows, width * d_in, -1)
            outputs.append(torch.matmul(gathered.reshape(batch, n_rows, n_longitude, width * d_in), matrices))
            continue

        parts: list[Tensor] = []
        if tables.depthwise_scalar is not None:
            coefficients = tables.depthwise_scalar.index_select(0, flat_slots).reshape(1, n_rows, 1, width, n_scalar)
            parts.append((gathered[..., :n_scalar] * coefficients).sum(dim=-2))  # (batch, n_rows, n_longitude, a_0)
        if tables.depthwise_pairs is not None:
            # y_i = sum_k sum_j D[k, a, i, j] x[k, a, j] over every non-scalar copy at once: four broadcast
            # products of one coordinate plane against one entry of the 2 x 2 blocks, each reduced over the
            # slots. A 2 x 2 block per copy is too small for a batched matmul, and products over fewer axes
            # iterate faster than one product over (k, a, i, j).
            pairs = gathered[..., n_scalar:].reshape(batch, n_rows, n_longitude, width, n_pairs, 2)
            real, imaginary = pairs[..., 0], pairs[..., 1]  # (batch, n_rows, n_longitude, width, n_pairs)
            entries = tables.depthwise_pairs.index_select(0, flat_slots).reshape(1, n_rows, 1, width, n_pairs, 4)
            first = (real * entries[..., 0]).sum(-2) + (imaginary * entries[..., 1]).sum(-2)
            second = (real * entries[..., 2]).sum(-2) + (imaginary * entries[..., 3]).sum(-2)
            mixed = torch.stack((first, second), dim=-1)  # (batch, n_rows, n_longitude, n_pairs, 2)
            parts.append(mixed.reshape(batch, n_rows, n_longitude, 2 * n_pairs))
        outputs.append(torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0])

    aggregated = torch.cat(outputs, dim=1).index_select(1, plan.inverse_order)  # (batch, n_latitude, n_longitude, d_out)
    if tables.denominator is not None:
        aggregated = aggregated / tables.denominator.reshape(1, n_latitude, 1, 1)
    return aggregated.reshape(batch, n_nodes, -1)


def stencil_convolution(module: SO2Convolution, x: SO2FeatureField, plan: StencilPlan) -> SO2FeatureField:
    """:meth:`SO2Convolution.forward` from a stencil plan on the dense layout, all rows at once.

    Parameters
    ----------
    module
        The layer, general or depthwise, with its window, normalization and
        self-interaction; its parameters are used, nothing is owned here.
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
    if x.field_type != module.field_type_in:
        raise ValueError(f"Expected a field of type {module.field_type_in!r}, got {x.field_type!r}.")
    if x.batch_shape[-1:] != (plan.n_nodes,):
        raise ValueError(
            f"x.batch_shape must end in (n_nodes,) = ({plan.n_nodes},) for the "
            f"{plan.n_latitude} x {plan.n_longitude} grid, got {tuple(x.batch_shape)}."
        )
    if plan.dtype != x.dtype:
        raise ValueError(f"The plan's stencil is {plan.dtype} and x is {x.dtype}; move one with .to().")
    lead = len(x.batch_shape) - 1
    batch = math.prod(x.batch_shape[:lead])

    tables = convolution_tables(module, plan)
    source = x.to_flat().reshape(batch, plan.n_nodes, -1)
    aggregated = aggregate(plan, source, tables, module.field_type_in).reshape(*x.batch_shape, -1)
    return module._finish(x, SO2FeatureField.from_flat(module.field_type_out, aggregated).coefficients)
