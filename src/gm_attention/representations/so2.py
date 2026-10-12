"""Irreducible real representations of ``SO(2)`` and their complex form.

Up to equivalence the finite-dimensional irreducible real representations of
``SO(2)`` are the trivial representation ``rho_0(r_theta) = 1`` and, for
every integer frequency ``n >= 1``, the rotation representation

    rho_n(r_theta) = [[cos(n theta), -sin(n theta)],
                      [sin(n theta),  cos(n theta)]].

Under ``R^2 = C``, ``(x, y) <-> x + iy``, ``rho_n(r_theta)`` acts as
multiplication by ``exp(i n theta)``; each routine has a dense real-matrix
form and a complex-scalar form. A feature field of type ``rho_n`` is stored
as its coefficient vector with respect to a node's tangent gauge.
Consistently with
:func:`gm_attention.discretization.frames.rotate_tangent_frames`, a change
of gauge ``E' = E r(theta)`` transforms coefficients by the inverse
representation, ``f' = rho_n(r_theta)^{-1} f = rho_n(r_{-theta}) f``, so
that the underlying physical object is unchanged. The functions here
implement the bare action ``rho_n(r_theta)``; the sign of a regauging is
applied by
:meth:`gm_attention.representations.field_type.SO2FeatureField.gauge_transform`.
"""

from __future__ import annotations

import torch
from torch import Tensor

_COMPLEX_DTYPES: dict[torch.dtype, torch.dtype] = {
    torch.float32: torch.complex64,
    torch.float64: torch.complex128,
}

_REAL_DTYPES: dict[torch.dtype, torch.dtype] = {
    complex_dtype: real_dtype
    for real_dtype, complex_dtype in _COMPLEX_DTYPES.items()
}


def irrep_dimension(frequency: int) -> int:
    """Return the dimension of the irreducible representation ``rho_n``.

    Parameters
    ----------
    frequency
        Nonnegative integer frequency ``n``.

    Returns
    -------
    int
        ``1`` for the trivial representation ``n = 0`` and ``2`` otherwise.
    """
    validate_frequency(frequency)

    return 1 if frequency == 0 else 2


def validate_frequency(frequency: int) -> None:
    """Raise if ``frequency`` is not a valid irrep label for ``SO(2)``."""
    if isinstance(frequency, bool) or not isinstance(frequency, int):
        raise TypeError(
            "An SO(2) irrep frequency must be an int, "
            f"got {type(frequency).__name__}."
        )

    if frequency < 0:
        raise ValueError(
            "An SO(2) irrep frequency must be nonnegative, "
            f"got {frequency}."
        )


def complex_dtype(real_dtype: torch.dtype) -> torch.dtype:
    """Return the complex dtype paired with a supported real dtype."""
    if real_dtype not in _COMPLEX_DTYPES:
        raise TypeError(
            "Representation arithmetic supports torch.float32 and "
            f"torch.float64, got {real_dtype}."
        )

    return _COMPLEX_DTYPES[real_dtype]


def real_dtype(complex_dtype_: torch.dtype) -> torch.dtype:
    """Return the real dtype paired with a supported complex dtype."""
    if complex_dtype_ not in _REAL_DTYPES:
        raise TypeError(
            "Representation arithmetic supports torch.complex64 and "
            f"torch.complex128, got {complex_dtype_}."
        )

    return _REAL_DTYPES[complex_dtype_]


def irrep_matrix(
    frequency: int,
    angles: Tensor,  # (...)
) -> Tensor:  # (..., d_n, d_n)
    """Build the dense real matrices ``rho_n(r_theta)``.

    :func:`irrep_phase` represents the same action as one complex number.

    Parameters
    ----------
    frequency
        Nonnegative integer frequency ``n``.
    angles
        Rotation angles in radians with arbitrary shape ``(...)``.

    Returns
    -------
    Tensor
        Real matrices of shape ``(..., d_n, d_n)`` where
        ``d_n = irrep_dimension(frequency)``.
    """
    dimension = irrep_dimension(frequency)
    _validate_angles(angles)

    if dimension == 1:
        return torch.ones(  # (..., 1, 1)
            (*angles.shape, 1, 1),
            dtype=angles.dtype,
            device=angles.device,
        )

    scaled_angles = frequency * angles  # (...)

    cosine = torch.cos(scaled_angles)  # (...)
    sine = torch.sin(scaled_angles)  # (...)

    first_row = torch.stack((cosine, -sine), dim=-1)  # (..., 2)
    second_row = torch.stack((sine, cosine), dim=-1)  # (..., 2)

    return torch.stack((first_row, second_row), dim=-2)  # (..., 2, 2)


def irrep_phase(
    frequency: int,
    angles: Tensor,  # (...)
) -> Tensor:  # (...) complex
    """Return ``exp(i n theta)``, the complex form of ``rho_n(r_theta)``.

    Parameters
    ----------
    frequency
        Nonnegative integer frequency ``n``.
    angles
        Rotation angles in radians with arbitrary shape ``(...)``.

    Returns
    -------
    Tensor
        Unit-modulus complex tensor of shape ``(...)``. For ``n = 0`` this is
        identically one, matching the trivial representation.
    """
    validate_frequency(frequency)
    _validate_angles(angles)

    scaled_angles = frequency * angles  # (...)

    return torch.polar(  # (...) complex
        torch.ones_like(scaled_angles),
        scaled_angles,
    )


def phase_powers(
    angles: Tensor,  # (...)
    max_frequency: int,
) -> Tensor:  # (..., max_frequency + 1) complex
    """Precompute ``exp(i k theta)`` for every ``k`` up to ``max_frequency``.

    A steerable kernel between types ``rho_n`` and ``rho_m`` depends on the
    polar angle only through the harmonics ``exp(i(m - n)phi)`` and
    ``exp(i(m + n)phi)``; for a fixed discretization they are tabulated once
    and shared by every layer and frequency pair. The table is one
    ``torch.polar`` over the outer product ``theta (x) (0, ..., max_frequency)``.

    Parameters
    ----------
    angles
        Angles in radians with arbitrary shape ``(...)``.
    max_frequency
        Largest harmonic to tabulate. Must be nonnegative.

    Returns
    -------
    Tensor
        Complex tensor of shape ``(..., max_frequency + 1)`` whose ``k``-th
        entry along the final axis is ``exp(i k theta)``.

    Notes
    -----
    Only nonnegative orders are tabulated: ``exp(-i k theta) =
    conj(exp(i k theta))``, so the harmonic ``exp(i(m - n)phi)`` with
    ``n > m`` is ``table[..., n - m].conj()``. A table up to the largest
    ``m + n`` in use holds every harmonic a steerable kernel needs.
    """
    validate_frequency(max_frequency)
    _validate_angles(angles)

    exponents = torch.arange(  # (max_frequency + 1,)
        max_frequency + 1,
        dtype=angles.dtype,
        device=angles.device,
    )

    scaled_angles = angles.unsqueeze(-1) * exponents  # (..., K + 1)

    return torch.polar(  # (..., K + 1) complex
        torch.ones_like(scaled_angles),
        scaled_angles,
    )


def as_complex(
    coefficients: Tensor,  # (..., 2)
) -> Tensor:  # (...) complex
    """View real ``rho_n`` coefficients (``n >= 1``) as complex numbers.

    Parameters
    ----------
    coefficients
        Real tensor of shape ``(..., 2)`` holding the two coefficients of a
        frequency ``n >= 1`` irrep along the final axis.

    Returns
    -------
    Tensor
        Complex tensor of shape ``(...)``.

    Notes
    -----
    A view, with gradients flowing to the real tensor, whenever
    ``coefficients`` is contiguous and starts at an even element offset
    within its storage, as ``torch.view_as_complex`` requires; a contiguous
    tensor at an odd offset (a slice such as ``buffer[1:3]``) is cloned
    first.
    """
    if not isinstance(coefficients, Tensor):
        raise TypeError(
            "coefficients must be a torch.Tensor, "
            f"got {type(coefficients).__name__}."
        )

    if coefficients.ndim == 0 or coefficients.shape[-1] != 2:
        raise ValueError(
            "coefficients must have shape (..., 2), "
            f"got {tuple(coefficients.shape)}."
        )

    if coefficients.dtype not in _COMPLEX_DTYPES:
        raise TypeError(
            "coefficients must use torch.float32 or torch.float64, "
            f"got {coefficients.dtype}."
        )

    coefficients = coefficients.contiguous()

    if coefficients.storage_offset() % 2 != 0:
        coefficients = coefficients.clone()

    return torch.view_as_complex(coefficients)  # (...)


def as_real(
    values: Tensor,  # (...) complex
) -> Tensor:  # (..., 2)
    """Inverse of :func:`as_complex`.

    Parameters
    ----------
    values
        Complex tensor of shape ``(...)``.

    Returns
    -------
    Tensor
        Real tensor of shape ``(..., 2)``.
    """
    if not isinstance(values, Tensor):
        raise TypeError(
            f"values must be a torch.Tensor, got {type(values).__name__}."
        )

    if values.dtype not in _REAL_DTYPES:
        raise TypeError(
            "values must use torch.complex64 or torch.complex128, "
            f"got {values.dtype}."
        )

    return torch.view_as_real(values.contiguous())  # (..., 2)


def rotate_irrep_coefficients(
    frequency: int,
    coefficients: Tensor,  # (..., d_n)
    angles: Tensor,  # broadcastable to (...)
) -> Tensor:  # (..., d_n)
    """Apply ``rho_n(r_theta)`` to irrep coefficients.

    Parameters
    ----------
    frequency
        Nonnegative integer frequency ``n``.
    coefficients
        For ``n >= 1``, a real tensor of shape ``(..., 2)``. For ``n = 0``
        the action is the identity and the shape is unconstrained.
    angles
        Rotation angles in radians, broadcastable against the leading shape
        ``(...)`` of ``coefficients``.

    Returns
    -------
    Tensor
        Rotated coefficients with the same shape as ``coefficients``.

    Notes
    -----
    The bare group action; a change of gauge by ``theta`` acts on
    coefficients by ``rho_n(r_{-theta})``.
    """
    validate_frequency(frequency)

    if frequency == 0:
        return coefficients

    values = as_complex(coefficients)  # (...) complex
    phases = irrep_phase(frequency, angles)  # broadcastable to (...) complex

    return as_real(values * phases)  # (..., 2)


def _validate_angles(
    angles: Tensor,  # (...)
) -> None:
    if not isinstance(angles, Tensor):
        raise TypeError(
            f"angles must be a torch.Tensor, got {type(angles).__name__}."
        )

    if angles.dtype not in _COMPLEX_DTYPES:
        raise TypeError(
            "angles must use torch.float32 or torch.float64, "
            f"got {angles.dtype}."
        )
