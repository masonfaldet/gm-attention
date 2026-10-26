"""Gauge-equivariant attention on a fixed tensor-product grid: the dense reference.

The graph path of :class:`~gm_attention.nn.attention.SO2Attention` reads one
geometry row per edge. On a tensor-product grid in the coordinate gauge the
rows fold into the per-row slots of a
:class:`~gm_attention.discretization.stencil.TensorGridStencil`, and the
functions here evaluate the same layer on the dense
``(..., n_latitude, n_longitude, channels)`` layout from those slots alone:

- :func:`steerable_kernel_matrices` turns a
  :class:`~gm_attention.nn.kernels.SO2SteerableKernel` into one real
  ``(d_out, d_in)`` matrix per slot, ``V(v_s) rho_in(g_s)`` on the packed
  layout, from one evaluation of the kernel's radial coefficients per slot:
  the radial MLP, the readout heads, the phases and the radial powers run on
  the ``n_slots`` representative rows, ``n_longitude`` times fewer than once
  per edge, and the block formula of the kernel is assembled into a real
  matrix. The kernel is linear in its input, so the tests check these
  matrices against the kernel's own forward on the basis vectors of its
  input type (:meth:`TensorGridStencil.repeated`).
- :func:`attention_fixed_grid` scores, normalizes and aggregates one output
  row at a time: the keys of the row's slots gathered along longitude with
  the slot's offset, transported by the slot's phase, dotted with the row's
  queries, the window and quadrature bias added, a softmax over the slots,
  and the slot matrices applied to the gathered input features under the
  attention weights. :func:`mhsa_fixed_grid` runs every head of an
  :class:`~gm_attention.nn.attention.SO2MHSA` and fuses them as the graph
  path does.

Both agree with the graph path on the same discretization to floating-point
summation order (``tests/nn/test_fixed_grid_attention.py``; an oracle
written from the layer's definition in ``tests/nn/test_fixed_grid_oracle.py``),
including the gradients, and are exactly equivariant to the grid's own
rotations: shifting the input along longitude shifts the output. They are a
reference, not a fast path: the per-row loop materializes each row's
gathered neighbourhood, ``(*lead, k, n_longitude, d_in)`` at a time, and the
slot matrices hold ``d_out * d_in`` reals per slot. What they establish is
the arithmetic a stencil kernel on this layout has to reproduce.
"""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor

from gm_attention.discretization.stencil import TensorGridStencil

# The attention's reference shares the attention's own pieces (the heads'
# concatenation, the pointwise finish, the score scale), as the fused paths do.
from gm_attention.nn.attention import SO2MHSA, SO2Attention, _concatenate_heads
from gm_attention.nn.convolution import evaluate_aggregation_window
from gm_attention.nn.kernels import (
    SO2SteerableKernel,
    apply_radial_trunk,
    split_k_linear,
)
from gm_attention.representations import so2
from gm_attention.representations.field_type import SO2FeatureField


def _real_columns(linear: Tensor, antilinear: Tensor | None) -> Tensor:  # (k, b, 2, a, 2)
    """The real matrix of the map ``z -> P z + Q conj(z)`` between complex
    coordinates, on the packed ``(copy, re/im)`` layout of both sides.

    ``linear`` is ``P`` of shape ``(k, b, a)``, ``antilinear`` is ``Q`` of
    the same shape or ``None`` for ``Q = 0``. The input coordinate ``re``
    (``z = e_a``) maps to the column ``P[:, a] + Q[:, a]``, the coordinate
    ``im`` (``z = i e_a``, ``conj(z) = -i e_a``) to ``i (P[:, a] - Q[:, a])``;
    the output rows are the real and imaginary parts.
    """
    if antilinear is None:
        columns = torch.stack((linear, 1j * linear), dim=-1)  # (k, b, a, 2): c = re, im
    else:
        columns = torch.stack((linear + antilinear, 1j * (linear - antilinear)), dim=-1)
    return torch.stack((columns.real, columns.imag), dim=2)  # (k, b, 2, a, 2): r = re, im


def steerable_kernel_matrices(
    kernel: SO2SteerableKernel,
    stencil: TensorGridStencil,
) -> Tensor:  # (n_slots, d_out, d_in)
    """The kernel's value map at every slot as a real matrix on the packed layout.

    ``matrices[s] @ x_j`` equals the message the kernel forms on an edge of
    slot ``s`` from the source feature ``x_j`` (packed by
    :meth:`~gm_attention.representations.field_type.SO2FeatureField.to_flat`),
    transport included: the block formula of
    :meth:`SO2SteerableKernel._forward_general`, with the radial
    coefficients, phases and radial powers of ``stencil.geometry`` computed
    once per slot and each ``(n, m)`` block written as a real matrix. The
    matrices differentiate with respect to the kernel's parameters as the
    forward does.
    """
    geometry = stencil.geometry
    n_slots = geometry.n_edges
    type_in, type_out = kernel.field_type_in, kernel.field_type_out
    reference = kernel.radial_basis.centers
    # The kernel's own entry checks (harmonic orders, the cutoff), on a field of the slots' shape.
    kernel.validate_call(
        SO2FeatureField.from_flat(
            type_in, torch.zeros(n_slots, type_in.dimension, dtype=reference.dtype, device=reference.device)
        ),
        geometry,
    )

    hidden = apply_radial_trunk(kernel.radial_trunk, kernel.radial_basis(geometry.distances))  # (n_slots, T)
    complex_coefficients = (
        split_k_linear(hidden, kernel.complex_head.weight, kernel.complex_head.bias)
        if kernel.complex_head is not None else None
    )
    chunks = (
        dict(zip(kernel._coefficient_order, torch.split(complex_coefficients, kernel._coefficient_widths, dim=1),
                 strict=True))
        if complex_coefficients is not None else {}
    )
    radial_powers = kernel._radial_powers(geometry.distances) if complex_coefficients is not None else None
    real_coefficients = (
        split_k_linear(hidden, kernel.real_head.weight, kernel.real_head.bias)
        if kernel.real_head is not None else None
    )

    def block(frequency_n: int, frequency_m: int) -> Tensor:  # (n_slots, b_m d_m, a_n d_n)
        a_n, b_m = type_in.multiplicity(frequency_n), type_out.multiplicity(frequency_m)
        if frequency_n == 0 and frequency_m == 0:
            assert real_coefficients is not None
            return real_coefficients.reshape(n_slots, b_m, a_n)

        def coefficient(role: str) -> Tensor:  # (n_slots, b_m, a_n) complex, scaled by (r / cutoff)^|k|
            assert radial_powers is not None
            matrix = so2.as_complex(chunks[(frequency_n, frequency_m, role)].reshape(n_slots, b_m, a_n, 2))
            return matrix * radial_powers[:, kernel._pair_powers[(frequency_n, frequency_m, role)]].reshape(-1, 1, 1)

        def phase(angular: int, transport: int) -> Tensor:  # (n_slots, 1, 1) complex
            value = geometry.angular_harmonic(angular)
            if transport:
                value = value * geometry.transport_harmonic(transport)
            return value.reshape(-1, 1, 1)

        if frequency_n == 0:
            # (0 -> m >= 1): e^{i m phi} A x_0 on a real input: one real input coordinate per copy.
            columns = phase(frequency_m, 0) * coefficient("linear")  # (n_slots, b_m, a_0)
            return torch.stack((columns.real, columns.imag), dim=2).reshape(n_slots, 2 * b_m, a_n)
        if frequency_m == 0:
            # (n >= 1 -> 0): Re[A conj(e^{-i n phi} e^{i n gamma} z)] = Re[Q conj(z)], Q = A e^{i n phi} e^{-i n gamma}.
            antilinear = phase(frequency_n, -frequency_n) * coefficient("linear")  # (n_slots, b_0, a_n)
            real_part = _real_columns(torch.zeros_like(antilinear), antilinear)[:, :, 0]  # (n_slots, b_0, a_n, 2)
            return real_part.reshape(n_slots, b_m, 2 * a_n)
        # (n, m >= 1): A e^{i (m - n) phi} e^{i n gamma} z + B e^{i (m + n) phi} e^{-i n gamma} conj(z).
        linear = phase(frequency_m - frequency_n, frequency_n) * coefficient("linear")
        antilinear = phase(frequency_m + frequency_n, -frequency_n) * coefficient("antilinear")
        return _real_columns(linear, antilinear).reshape(n_slots, 2 * b_m, 2 * a_n)

    # Frequencies ascend on both sides, the order of the packed layout's slices.
    return torch.cat(
        [torch.cat([block(n, m) for n in type_in.frequencies], dim=-1) for m in type_out.frequencies], dim=-2
    )


def _dense(tensor: Tensor, lead: int, n_latitude: int, n_longitude: int) -> Tensor:
    """``(*lead, n_nodes, *trailing) -> (*lead, n_latitude, n_longitude, *trailing)``."""
    return tensor.reshape(*tensor.shape[:lead], n_latitude, n_longitude, *tensor.shape[lead + 1 :])


def _row(tensor: Tensor, row: int, trailing: int) -> Tensor:
    """Output row ``row`` of a dense tensor with ``trailing`` type axes:
    ``(*lead, n_latitude, n_longitude, *type) -> (*lead, n_longitude, *type)``."""
    return tensor.select(-(trailing + 2), row)


def _gather(tensor: Tensor, source_row: Tensor, source_longitude: Tensor, trailing: int) -> Tensor:
    """The slots' sources at every target longitude:
    ``(*lead, n_latitude, n_longitude, *type) -> (*lead, k, n_longitude, *type)``
    for ``source_row`` of shape ``(k,)`` and ``source_longitude`` of shape ``(k, n_longitude)``."""
    index = (Ellipsis, source_row.unsqueeze(-1), source_longitude) + (slice(None),) * trailing
    return tensor[index]


def _validate(module: SO2Attention, x: SO2FeatureField, stencil: TensorGridStencil) -> int:
    """Shared entry checks; returns the number of leading batch axes."""
    if x.field_type != module.field_type_in:
        raise ValueError(f"Expected a field of type {module.field_type_in!r}, got {x.field_type!r}.")
    if x.batch_shape[-1:] != (stencil.n_nodes,):
        raise ValueError(
            f"x.batch_shape must end in (n_nodes,) = ({stencil.n_nodes},) for the "
            f"{stencil.n_latitude} x {stencil.n_longitude} grid, got {tuple(x.batch_shape)}."
        )
    if stencil.dtype != x.dtype:
        raise ValueError(f"The stencil is {stencil.dtype} and x is {x.dtype}; move one with .to().")
    if not math.isclose(module.cutoff, stencil.cutoff, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"The stencil was built at cutoff {stencil.cutoff:.12g}, the layer has {module.cutoff:.12g}.")
    return len(x.batch_shape) - 1


def attention_fixed_grid(
    module: SO2Attention,
    x: SO2FeatureField,
    stencil: TensorGridStencil,
) -> SO2FeatureField:
    """:meth:`SO2Attention.forward` evaluated from the stencil on the dense layout.

    Parameters
    ----------
    module
        The layer; its parameters are used, nothing is owned here.
    x
        A field of ``module.field_type_in`` with ``batch_shape``
        ``(*lead, n_latitude * n_longitude)``, nodes in the grid's
        row-major order; the leading axes are independent samples.
    stencil
        Built at ``module.cutoff`` over the discretization the graph path
        would receive, in ``x``'s dtype.

    Returns
    -------
    SO2FeatureField
        Of ``module.field_type_out`` with ``x``'s batch shape; equal to the
        graph path's output, sample by sample, to summation order.
    """
    lead = _validate(module, x, stencil)
    n_latitude, n_longitude = stencil.n_latitude, stencil.n_longitude
    geometry = stencil.geometry
    real_dtype = x.dtype

    # The per-slot part of the softmax: log c(v_s) + log w_s, -inf outside the window.
    window = evaluate_aggregation_window(geometry.distances, module.cutoff, module.window)  # (n_slots,)
    tiny = torch.finfo(real_dtype).tiny
    bias = torch.log(window.clamp_min(tiny)) + torch.log(geometry.tangent_quadrature_weights.clamp_min(tiny))
    inside = window > 0.0  # (n_slots,)

    matrices = steerable_kernel_matrices(cast(SO2SteerableKernel, module.value), stencil)  # (n_slots, d_out, d_in)

    # Queries and keys per node, as the graph path; the keys are transported per slot below.
    queries = module.query(x)
    keys = module.key(x)
    frequencies = module.field_type_attn.frequencies
    dense_queries = {f: _dense(queries[f], lead, n_latitude, n_longitude) for f in frequencies}
    dense_keys = {f: _dense(keys[f], lead, n_latitude, n_longitude) for f in frequencies}
    dense_source = _dense(x.to_flat(), lead, n_latitude, n_longitude)  # (*lead, nlat, nlon, d_in)

    rows_out: list[Tensor] = []
    for row in range(n_latitude):
        slots = stencil.slots(row)
        source_row = stencil.source_row[slots]  # (k,)
        source_longitude = stencil.source_longitudes(slots)  # (k, n_longitude)
        k = source_row.shape[0]

        # s_ij = <q_i, rho(g_s) k_j> / sqrt(dim rho_attn), over the packed real coordinates.
        score = torch.zeros(*x.batch_shape[:lead], k, n_longitude, dtype=real_dtype, device=x.device)
        for frequency in frequencies:
            trailing = 1 if frequency == 0 else 2
            key_sources = _gather(dense_keys[frequency], source_row, source_longitude, trailing)  # (*lead, k, nlon, a[, 2])
            if frequency != 0:
                phase = geometry.transport_harmonic(frequency)[slots].reshape(k, 1, 1)  # (k, 1, 1) complex
                key_sources = so2.as_real(phase * so2.as_complex(key_sources))
            query_row = _row(dense_queries[frequency], row, trailing).unsqueeze(-(trailing + 2))  # (*lead, 1, nlon, a[, 2])
            product = query_row * key_sources
            score = score + product.reshape(*product.shape[: lead + 2], -1).sum(-1)
        score = score * module._score_scale + bias[slots].unsqueeze(-1)
        score = score.masked_fill(~inside[slots].unsqueeze(-1), float("-inf"))
        # Every row holds its self-loop slot inside the window (the stencil checks it), so the softmax is finite.
        attention = torch.softmax(score, dim=-2)  # (*lead, k, n_longitude), summing to 1 over the slots

        sources = _gather(dense_source, source_row, source_longitude, 1)  # (*lead, k, nlon, d_in)
        messages = torch.einsum("kod,...knd->...kno", matrices[slots], sources)  # (*lead, k, nlon, d_out)
        rows_out.append((attention.unsqueeze(-1) * messages).sum(dim=-3))  # (*lead, nlon, d_out)

    out = torch.stack(rows_out, dim=-3).reshape(*x.batch_shape, -1)  # (*lead, n_nodes, d_out)
    attended = SO2FeatureField.from_flat(module.field_type_out, out)
    return module._finish(x, attended.coefficients)


def mhsa_fixed_grid(
    module: SO2MHSA,
    x: SO2FeatureField,
    stencil: TensorGridStencil,
) -> SO2FeatureField:
    """:meth:`SO2MHSA.forward` from the stencil: every head through
    :func:`attention_fixed_grid`, concatenated head-major within each
    frequency and passed through the output intertwiner, as the graph path."""
    if x.field_type != module.field_type_in:
        raise ValueError(f"Expected a field of type {module.field_type_in!r}, got {x.field_type!r}.")
    heads = [attention_fixed_grid(cast(SO2Attention, head), x, stencil) for head in module.heads]
    return module.output_projection(_concatenate_heads(heads, module.field_type_in))
