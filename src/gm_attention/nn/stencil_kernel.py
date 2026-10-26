"""The steerable kernel's slot matrices assembled by two gathers instead of per-block stacking.

:func:`~gm_attention.nn.fixed_grid.steerable_kernel_matrices` builds the real
``(d_out, d_in)`` matrix of a :class:`~gm_attention.nn.kernels.SO2SteerableKernel`
at every slot block by block: for each pair of frequencies it multiplies the
block's complex coefficients by its phase, splits real and imaginary parts
into the rows and columns of the packed layout and concatenates the blocks.
That is a few dozen small operations per block, and with the released field
type the assembly costs more than the radial MLP it follows: a third of the
stencil attention's forward at 64x128.

:func:`steerable_slot_matrices` computes the same matrices in a handful of
operations over the whole table. Every entry of the real matrix is the sum of
at most two terms, each ``+-`` the real or imaginary part of one phased
complex coefficient: two terms where the block maps ``rho_n -> rho_m`` with
``n, m >= 1`` and carries a linear and an antilinear coefficient, one term
elsewhere, and the plain real coefficient on the ``rho_0 -> rho_0`` block.
So the function phases every complex column of the complex head's output
with its block's phase times radial power in one complex multiply, writes
the result, the real head's output and a zero column into one *pool*, forms
the pool with the minus signs applied to the columns that are ever negated,
and gathers the two terms of every entry, one from each pool, with two
``torch.gather`` calls and one addition. Which columns feed which entry
depends only on the two field types; that plan is built once per type pair
and cached. The values agree with the reference's to rounding: the phase and
the radial power are multiplied together before the coefficient where the
reference multiplies the coefficient by the power first, and every other
operation is the same one on the same numbers. The function also returns the
layout the stencil path wants directly: the transposed ``(d_in, d_out)``
matrices, a zero row for the plan's phantom slot, and an optional per-slot
factor (a convolution's window times quadrature weight).
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

from gm_attention.discretization.edge_geometry import EdgeGeometry
from gm_attention.discretization.stencil import TensorGridStencil
from gm_attention.nn.kernels import SO2SteerableKernel, apply_radial_trunk, split_k_linear
from gm_attention.representations import so2
from gm_attention.representations.field_type import SO2FeatureField, SO2FieldType


class _Assembly(NamedTuple):
    """Where every entry of the real matrix reads its two terms."""

    #: ``(d_out * d_in,)`` (or its transpose, flattened): the pool column of the term read from the signed pool.
    first: Tensor
    #: The pool column of the term read from the plain pool; the zero column where the entry has one term.
    second: Tensor
    #: ``(pool_width,)``: ``-1`` on the columns some entry reads negated, ``+1`` elsewhere.
    signs: Tensor
    #: ``(n_complex_columns,)``: the coefficient block of each complex column of the complex head.
    column_block: Tensor
    #: ``(n_blocks,)``: the radial power of each coefficient block.
    block_powers: Tensor
    #: Columns of the pool: the complex head's, the real head's, one zero column.
    pool_width: int


_ASSEMBLIES: dict[tuple[SO2FieldType, SO2FieldType, bool], _Assembly] = {}


def _assembly(kernel: SO2SteerableKernel, transpose: bool) -> _Assembly:
    """The gather plan of a kernel's type pair, built once and cached on the CPU."""
    type_in, type_out = kernel.field_type_in, kernel.field_type_out
    key = (type_in, type_out, transpose)
    cached = _ASSEMBLIES.get(key)
    if cached is not None:
        return cached

    complex_width = kernel.complex_head.out_features if kernel.complex_head is not None else 0
    real_width = kernel.real_head.out_features if kernel.real_head is not None else 0
    zero = complex_width + real_width
    pool_width = zero + 1
    d_in, d_out = type_in.dimension, type_out.dimension
    slices_in, slices_out = type_in.slices(), type_out.slices()

    # The signed pool negates the imaginary columns of a two-term block's linear coefficient P and the
    # real columns of its antilinear coefficient Q: in [[Re P + Re Q, -Im P + Im Q], [Im P + Im Q, Re P - Re Q]]
    # those are the quantities that appear with a minus sign. An entry reads its negated term, or when it
    # has none a term whose column is not negated, from the signed pool and the other from the plain one.
    signs = torch.ones(pool_width)
    first = torch.full((d_out, d_in), zero, dtype=torch.long)
    second = torch.full((d_out, d_in), zero, dtype=torch.long)
    for frequency_m, b in type_out.items():
        row = slices_out[frequency_m].start
        for frequency_n, a in type_in.items():
            column = slices_in[frequency_n].start
            beta = torch.arange(b).unsqueeze(1)  # (b, 1) output copies
            alpha = torch.arange(a).unsqueeze(0)  # (1, a) input copies
            if frequency_n == 0 and frequency_m == 0:
                # The real head, rows ordered (b_0, a_0), no phase: one positive term.
                first[row : row + b, column : column + a] = complex_width + beta * a + alpha
                continue
            linear = kernel._pair_ranges[(frequency_n, frequency_m, "linear")].start + (beta * a + alpha) * 2
            if frequency_n == 0:
                # (0 -> m >= 1): the rows (re, im) of e^{i m phi} A on one real input coordinate.
                first[row : row + 2 * b : 2, column : column + a] = linear
                first[row + 1 : row + 2 * b : 2, column : column + a] = linear + 1
            elif frequency_m == 0:
                # (n >= 1 -> 0): Re[Q conj(z)] on the input (re, im) reads (Re Q, Im Q).
                first[row : row + b, column : column + 2 * a : 2] = linear
                first[row : row + b, column + 1 : column + 2 * a : 2] = linear + 1
            else:
                antilinear = kernel._pair_ranges[(frequency_n, frequency_m, "antilinear")].start + (beta * a + alpha) * 2
                signs[linear.reshape(-1) + 1] = -1.0  # Im P
                signs[antilinear.reshape(-1)] = -1.0  # Re Q
                rows_re, rows_im = slice(row, row + 2 * b, 2), slice(row + 1, row + 2 * b, 2)
                cols_re, cols_im = slice(column, column + 2 * a, 2), slice(column + 1, column + 2 * a, 2)
                first[rows_re, cols_re], second[rows_re, cols_re] = linear, antilinear  # Re P + Re Q
                first[rows_re, cols_im], second[rows_re, cols_im] = linear + 1, antilinear + 1  # -Im P + Im Q
                first[rows_im, cols_re], second[rows_im, cols_re] = antilinear + 1, linear + 1  # Im Q + Im P
                first[rows_im, cols_im], second[rows_im, cols_im] = antilinear, linear  # -Re Q + Re P
    if transpose:
        first, second = first.transpose(0, 1), second.transpose(0, 1)

    widths = torch.tensor(kernel._coefficient_widths, dtype=torch.long) // 2
    column_block = torch.repeat_interleave(torch.arange(len(kernel._coefficient_order)), widths)
    block_powers = torch.tensor([kernel._pair_powers[block] for block in kernel._coefficient_order], dtype=torch.long)
    assembly = _Assembly(
        first.reshape(-1).contiguous(), second.reshape(-1).contiguous(), signs, column_block, block_powers, pool_width
    )
    _ASSEMBLIES[key] = assembly
    return assembly


def _block_phase(geometry: EdgeGeometry, block: tuple[int, int, str]) -> Tensor:  # (n_slots,) complex
    """The phase the reference multiplies a block's coefficients by, in its order of factors."""
    frequency_n, frequency_m, role = block
    if frequency_n == 0:
        return geometry.angular_harmonic(frequency_m)
    if frequency_m == 0:
        return geometry.angular_harmonic(frequency_n) * geometry.transport_harmonic(-frequency_n)
    if role == "linear":
        return geometry.angular_harmonic(frequency_m - frequency_n) * geometry.transport_harmonic(frequency_n)
    return geometry.angular_harmonic(frequency_m + frequency_n) * geometry.transport_harmonic(-frequency_n)


def steerable_slot_matrices(
    kernel: SO2SteerableKernel,
    stencil: TensorGridStencil,
    *,
    transpose: bool = False,
    pad: bool = False,
    scale: Tensor | None = None,
) -> Tensor:
    """The kernel's value map at every slot as a real matrix, assembled by two gathers.

    Parameters
    ----------
    kernel
        The value kernel.
    stencil
        Its slots; the kernel's entry checks run on ``stencil.geometry``.
    transpose
        Return ``(d_in, d_out)`` matrices, the operand of the stencil path's
        contraction, instead of ``(d_out, d_in)``.
    pad
        Append a zero matrix for the phantom slot of a
        :class:`~gm_attention.nn.stencil_plan.StencilPlan`: ``n_slots + 1``
        rows.
    scale
        Shape ``(n_slots,)``: a real factor per slot applied to every
        coefficient before the assembly (a convolution's window times
        quadrature weight).

    Returns
    -------
    Tensor
        ``(n_slots [+ 1], d_out, d_in)`` or its transpose; the values of
        :func:`~gm_attention.nn.fixed_grid.steerable_kernel_matrices` to
        rounding, differentiable with respect to the kernel's parameters as
        the forward is.
    """
    geometry = stencil.geometry
    n_slots = geometry.n_edges
    type_in, type_out = kernel.field_type_in, kernel.field_type_out
    reference = kernel.radial_basis.centers
    dtype, device = reference.dtype, reference.device
    kernel.validate_call(
        SO2FeatureField.from_flat(type_in, torch.zeros(n_slots, type_in.dimension, dtype=dtype, device=device)), geometry
    )
    if scale is not None and tuple(scale.shape) != (n_slots,):
        raise ValueError(f"scale must have shape (n_slots,) = ({n_slots},), got {tuple(scale.shape)}.")
    assembly = _assembly(kernel, transpose)
    rows = n_slots + 1 if pad else n_slots

    hidden = apply_radial_trunk(kernel.radial_trunk, kernel.radial_basis(geometry.distances))  # (n_slots, T)
    pieces: list[Tensor] = []
    if kernel.complex_head is not None:
        coefficients = split_k_linear(hidden, kernel.complex_head.weight, kernel.complex_head.bias)  # (n_slots, W)
        # Each block's phase times its radial power (r / cutoff)^p, spread over the block's complex columns.
        radial_powers = kernel._radial_powers(geometry.distances)  # (n_slots, max_power + 1)
        block_factors = torch.stack([_block_phase(geometry, block) for block in kernel._coefficient_order], dim=1)
        block_factors = block_factors * radial_powers.index_select(1, assembly.block_powers.to(device))
        factors = torch.gather(block_factors, 1, assembly.column_block.to(device).unsqueeze(0).expand(n_slots, -1))
        phased = factors * so2.as_complex(coefficients.reshape(n_slots, -1, 2))  # (n_slots, W / 2) complex
        pieces.append(so2.as_real(phased).reshape(n_slots, -1))
    if kernel.real_head is not None:
        pieces.append(split_k_linear(hidden, kernel.real_head.weight, kernel.real_head.bias))  # (n_slots, R)

    # The plain pool: the phased complex columns, the real columns, a zero column; a zero row for the phantom.
    pool = torch.zeros(rows, assembly.pool_width, dtype=dtype, device=device)
    pool[:n_slots, : assembly.pool_width - 1] = torch.cat(pieces, dim=1) if len(pieces) > 1 else pieces[0]
    if scale is not None:
        pool = pool * torch.cat([scale, scale.new_zeros(1)]).unsqueeze(-1) if pad else pool * scale.unsqueeze(-1)
    signed = pool * assembly.signs.to(dtype=dtype, device=device)

    first = assembly.first.to(device).unsqueeze(0).expand(rows, -1)
    second = assembly.second.to(device).unsqueeze(0).expand(rows, -1)
    matrices = torch.gather(signed, 1, first) + torch.gather(pool, 1, second)
    shape = (type_in.dimension, type_out.dimension) if transpose else (type_out.dimension, type_in.dimension)
    return matrices.reshape(rows, *shape)
