"""Dense reference implementation of ``SO(2)``-steerable angular kernel bases.

This module is the literal implementation for the matrix valued kernels. For each pair of
irreducible types ``(rho_n, rho_m)``, a basis
``{K_i^{m<-n}(phi)}_{i=1}^{d_m d_n}`` of the space of steerable value
transformations, satisfying

    K_i^{m<-n}(phi) = rho_m(r_phi) A_i rho_n(r_phi)^{-1}

for a basis ``{A_i}`` of ``Hom(R^{d_n}, R^{d_m})``. A general steerable block
is then the matrix-valued radial expansion

    V_{m<-n}(r, phi) = sum_i C_i^{m<-n}(r) (x) K_i^{m<-n}(phi).

Every function here works with dense matrices and real trigonometry and is
written for transparency rather than speed. It provides a reference
implementation against which the optimized complex-arithmetic kernel in
``nn/kernels.py`` is checked, and is not used in a forward pass. That fast path replaces the four
matrices below with two complex numbers read from a precomputed table; see
:meth:`gm_attention.discretization.edge_geometry.EdgeGeometry.angular_harmonic`.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

from gm_attention.representations import so2
from gm_attention.representations.field_type import SO2FieldType


def steerable_basis_dimension(frequency_n: int, frequency_m: int) -> int:
    """Dimension ``d_m * d_n`` of the steerable-kernel basis for ``(n, m)``."""
    so2.validate_frequency(frequency_n)
    so2.validate_frequency(frequency_m)

    return so2.irrep_dimension(frequency_m) * so2.irrep_dimension(frequency_n)


def steerable_basis(
    frequency_n: int,
    frequency_m: int,
    angles: Tensor,  # (...)
) -> Tensor:  # (..., dim, d_m, d_n)
    """Angular kernel basis ``{K_i^{m<-n}(phi)}`` of the steerable-transformation table
    (docs/math/so2_representations.md).

    Parameters
    ----------
    frequency_n, frequency_m
        Nonnegative integer frequencies of the input and output types.
    angles
        Polar angles ``phi`` in radians, arbitrary shape ``(...)``.

    Returns
    -------
    Tensor
        Shape ``(..., steerable_basis_dimension(n, m), d_m, d_n)``, ordered
        to match that table: for ``m, n >= 1`` the basis is
        ``(K1, K2, K3, K4)`` built from ``(I, J, S, JS)``.
    """
    so2.validate_frequency(frequency_n)
    so2.validate_frequency(frequency_m)
    _validate_angles(angles)

    if frequency_n == 0 and frequency_m == 0:
        return torch.ones(  # (..., 1, 1, 1)
            (*angles.shape, 1, 1, 1),
            dtype=angles.dtype,
            device=angles.device,
        )

    if frequency_n == 0:  # frequency_m >= 1
        cosine = torch.cos(frequency_m * angles)  # (...)
        sine = torch.sin(frequency_m * angles)  # (...)

        # Column vectors: K_i has shape (..., d_m, d_n) = (..., 2, 1).
        k1 = torch.stack((cosine, sine), dim=-1).unsqueeze(-1)  # (..., 2, 1)
        k2 = torch.stack((-sine, cosine), dim=-1).unsqueeze(-1)  # (..., 2, 1)

        # The new "which basis element" axis goes immediately after the
        # batch dims, i.e. right before (d_m, d_n) -- three axes from the end
        # of the *output*, hence dim=-3.
        return torch.stack((k1, k2), dim=-3)  # (..., 2, 2, 1)

    if frequency_m == 0:  # frequency_n >= 1
        cosine = torch.cos(frequency_n * angles)  # (...)
        sine = torch.sin(frequency_n * angles)  # (...)

        # Row vectors: K_i has shape (..., d_m, d_n) = (..., 1, 2).
        k1 = torch.stack((cosine, sine), dim=-1).unsqueeze(-2)  # (..., 1, 2)
        k2 = torch.stack((-sine, cosine), dim=-1).unsqueeze(-2)  # (..., 1, 2)

        return torch.stack((k1, k2), dim=-3)  # (..., 2, 1, 2)

    # frequency_n >= 1 and frequency_m >= 1
    minus_angles = (frequency_m - frequency_n) * angles  # (...)
    plus_angles = (frequency_m + frequency_n) * angles  # (...)

    cos_minus = torch.cos(minus_angles)  # (...)
    sin_minus = torch.sin(minus_angles)  # (...)
    cos_plus = torch.cos(plus_angles)  # (...)
    sin_plus = torch.sin(plus_angles)  # (...)

    k1 = _matrix2(cos_minus, -sin_minus, sin_minus, cos_minus)  # (..., 2, 2)
    k2 = _matrix2(-sin_minus, -cos_minus, cos_minus, -sin_minus)  # (..., 2, 2)
    k3 = _matrix2(cos_plus, sin_plus, sin_plus, -cos_plus)  # (..., 2, 2)
    k4 = _matrix2(-sin_plus, cos_plus, cos_plus, sin_plus)  # (..., 2, 2)

    return torch.stack((k1, k2, k3, k4), dim=-3)  # (..., 4, 2, 2)


def dense_steerable_block(
    frequency_n: int,
    frequency_m: int,
    radial_coefficients: Tensor,  # (..., dim)
    angles: Tensor,  # (...)
) -> Tensor:  # (..., d_m, d_n)
    """Combine radial coefficients with the angular basis at fixed multiplicity one.

    Implements ``V_{m<-n}(r, phi) = sum_i gamma_i(r) K_i^{m<-n}(phi)``
    (the un-tensored form of the assembled steerable block, i.e. before
    mixing multiple multiplicity copies).

    Parameters
    ----------
    frequency_n, frequency_m
        Nonnegative integer frequencies.
    radial_coefficients
        Real tensor of shape ``(..., steerable_basis_dimension(n, m))``,
        broadcastable against ``angles``.
    angles
        Polar angles in radians, broadcastable against
        ``radial_coefficients``'s leading shape.

    Returns
    -------
    Tensor
        Shape ``(..., d_m, d_n)``, broadcasting ``radial_coefficients`` and
        ``angles`` together.
    """
    dimension = steerable_basis_dimension(frequency_n, frequency_m)

    if radial_coefficients.shape[-1:] != (dimension,):
        raise ValueError(
            "radial_coefficients must have shape (..., "
            f"{dimension}) for the pair (n={frequency_n}, m={frequency_m}), "
            f"got {tuple(radial_coefficients.shape)}."
        )

    basis = steerable_basis(  # (..., dim, d_m, d_n)
        frequency_n,
        frequency_m,
        angles,
    )

    return torch.einsum("...i,...ipq->...pq", radial_coefficients, basis)


def dense_field_type_steerable_kernel(
    field_type_in: SO2FieldType,
    field_type_out: SO2FieldType,
    angles: Tensor,  # (...)
    radial_coefficients: Mapping[tuple[int, int], Tensor],
) -> Tensor:  # (..., field_type_out.dimension, field_type_in.dimension)
    """Assemble a full steerable value transformation from radial coefficients.

    This is the dense ground truth for a steerable value transformation  used to check the fast
    complex-arithmetic kernel in ``nn/kernels.py``. Unlike an intertwiner, a
    steerable kernel may have nonzero blocks between *inequivalent* irrep
    types, so every ``(n, m)`` pair may contribute.

    Parameters
    ----------
    field_type_in, field_type_out
        Input and output field types.
    angles
        Polar angles ``phi`` in radians, arbitrary shape ``(...)`` -- e.g.
        one angle per edge. Only meaningful away from the origin; a
        steerable kernel's value at ``r = 0`` is a plain intertwiner (see
        :mod:`tests.support.intertwiners`), not covered here.
    radial_coefficients
        Maps a ``(n, m)`` pair, with ``n`` in ``field_type_in.frequencies``
        and ``m`` in ``field_type_out.frequencies``, to a real tensor of
        shape ``(*coefficient_batch, steerable_basis_dimension(n, m),
        b_m, a_n)`` -- the stacked ``C_i^{m<-n}(r)`` matrices of
        the assembled steerable block. ``coefficient_batch`` must be
        broadcastable against ``angles.shape``. A pair absent from the
        mapping contributes an all-zero block.

    Returns
    -------
    Tensor
        Shape ``(*batch_shape, field_type_out.dimension,
        field_type_in.dimension)``, where ``batch_shape`` is the broadcast of
        ``angles.shape`` with every supplied coefficient's leading shape.
    """
    _validate_angles(angles)

    in_slices = field_type_in.slices()
    out_slices = field_type_out.slices()

    batch_shape = angles.shape

    for (frequency_n, frequency_m), coefficients in radial_coefficients.items():
        _validate_pair(field_type_in, field_type_out, frequency_n, frequency_m)
        dimension = steerable_basis_dimension(frequency_n, frequency_m)
        expected_tail = (
            dimension,
            field_type_out.multiplicity(frequency_m),
            field_type_in.multiplicity(frequency_n),
        )

        if (
            not isinstance(coefficients, Tensor)
            or coefficients.shape[-3:] != expected_tail
        ):
            raise ValueError(
                "radial_coefficients[("
                f"{frequency_n}, {frequency_m})] must have shape "
                f"(*batch, {', '.join(map(str, expected_tail))}), got "
                f"{coefficients.shape if isinstance(coefficients, Tensor) else type(coefficients)}."
            )

        batch_shape = torch.broadcast_shapes(
            batch_shape,
            coefficients.shape[:-3],
        )

    matrix = torch.zeros(  # (*batch_shape, out_dim, in_dim)
        (*batch_shape, field_type_out.dimension, field_type_in.dimension),
        dtype=angles.dtype,
        device=angles.device,
    )

    for (frequency_n, frequency_m), coefficients in radial_coefficients.items():
        basis = steerable_basis(  # (*batch_shape, dim, d_m, d_n)
            frequency_n,
            frequency_m,
            angles.expand(batch_shape),
        )

        # out[..., beta, p, alpha, q] = sum_i C_i[..., beta, alpha] K_i[..., p, q]
        block = torch.einsum(  # (*batch_shape, b_m, d_m, a_n, d_n)
            "...iba,...ipq->...bpaq",
            coefficients,
            basis,
        )

        b_m, d_m, a_n, d_n = block.shape[-4:]
        block = block.reshape(  # (*batch_shape, b_m * d_m, a_n * d_n)
            (*batch_shape, b_m * d_m, a_n * d_n)
        )

        matrix[..., out_slices[frequency_m], in_slices[frequency_n]] = block

    return matrix


def _validate_pair(
    field_type_in: SO2FieldType,
    field_type_out: SO2FieldType,
    frequency_n: int,
    frequency_m: int,
) -> None:
    if frequency_n not in field_type_in:
        raise ValueError(
            f"radial_coefficients has a key with input frequency "
            f"{frequency_n}, which is not present in field_type_in "
            f"{sorted(field_type_in.frequencies)}."
        )

    if frequency_m not in field_type_out:
        raise ValueError(
            f"radial_coefficients has a key with output frequency "
            f"{frequency_m}, which is not present in field_type_out "
            f"{sorted(field_type_out.frequencies)}."
        )


def _matrix2(
    a: Tensor,  # (...)
    b: Tensor,  # (...)
    c: Tensor,  # (...)
    d: Tensor,  # (...)
) -> Tensor:  # (..., 2, 2)
    """Stack four scalar tensors into batched [[a, b], [c, d]] matrices."""
    first_row = torch.stack((a, b), dim=-1)  # (..., 2)
    second_row = torch.stack((c, d), dim=-1)  # (..., 2)

    return torch.stack((first_row, second_row), dim=-2)  # (..., 2, 2)


def _validate_angles(angles: Tensor) -> None:
    if not isinstance(angles, Tensor):
        raise TypeError(
            f"angles must be a torch.Tensor, got {type(angles).__name__}."
        )

    if not angles.is_floating_point():
        raise TypeError("angles must have a floating-point dtype.")
