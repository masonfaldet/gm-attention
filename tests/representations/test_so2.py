"""Tests for gm_attention.representations.so2.

The central contract is that the dense real form ``irrep_matrix`` and the
complex form ``irrep_phase`` describe the *same* action, since every layer
uses the second while every reference implementation uses the first.
"""

from __future__ import annotations

import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from gm_attention.representations.so2 import (
    as_complex,
    as_real,
    complex_dtype,
    irrep_dimension,
    irrep_matrix,
    irrep_phase,
    phase_powers,
    real_dtype,
    rotate_irrep_coefficients,
    validate_frequency,
)

ATOL = 1e-10

_angles = st.floats(
    min_value=-10 * math.pi,
    max_value=10 * math.pi,
    allow_nan=False,
    allow_infinity=False,
)

_frequencies = st.integers(min_value=0, max_value=6)


# --------------------------------------------------------------------------
# irrep_dimension / validate_frequency
# --------------------------------------------------------------------------


def test_irrep_dimension_is_one_for_the_trivial_representation() -> None:
    assert irrep_dimension(0) == 1


@pytest.mark.parametrize("frequency", [1, 2, 3, 17])
def test_irrep_dimension_is_two_for_nontrivial_frequencies(
    frequency: int,
) -> None:
    assert irrep_dimension(frequency) == 2


def test_validate_frequency_rejects_negative() -> None:
    with pytest.raises(ValueError):
        validate_frequency(-1)


@pytest.mark.parametrize("frequency", [1.0, "1", True, None])
def test_validate_frequency_rejects_non_int(frequency: object) -> None:
    with pytest.raises(TypeError):
        validate_frequency(frequency)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# irrep_matrix
# --------------------------------------------------------------------------


def test_irrep_matrix_trivial_representation_is_one() -> None:
    angles = torch.linspace(0, 2 * math.pi, 7, dtype=torch.float64)
    matrices = irrep_matrix(0, angles)

    assert matrices.shape == (7, 1, 1)
    assert torch.allclose(matrices, torch.ones_like(matrices), atol=ATOL)


def test_irrep_matrix_frequency_one_is_the_standard_rotation() -> None:
    angle = torch.tensor(math.pi / 3, dtype=torch.float64)
    matrix = irrep_matrix(1, angle)

    expected = torch.tensor(
        [
            [math.cos(math.pi / 3), -math.sin(math.pi / 3)],
            [math.sin(math.pi / 3), math.cos(math.pi / 3)],
        ],
        dtype=torch.float64,
    )

    assert torch.allclose(matrix, expected, atol=ATOL)


def test_irrep_matrix_frequency_n_rotates_n_times_faster() -> None:
    angles = torch.linspace(0, 2 * math.pi, 13, dtype=torch.float64)

    for frequency in (1, 2, 3, 5):
        assert torch.allclose(
            irrep_matrix(frequency, angles),
            irrep_matrix(1, frequency * angles),
            atol=ATOL,
        )


def test_irrep_matrix_generator_at_quarter_turn_is_the_complex_structure() -> (
    None
):
    # rho_n(r_{pi / 2n}) = J, the standard complex structure. This is the
    # step that makes rho_n irreducible over the reals.
    complex_structure = torch.tensor(
        [[0.0, -1.0], [1.0, 0.0]],
        dtype=torch.float64,
    )

    for frequency in (1, 2, 4):
        angle = torch.tensor(
            math.pi / (2 * frequency),
            dtype=torch.float64,
        )

        assert torch.allclose(
            irrep_matrix(frequency, angle),
            complex_structure,
            atol=ATOL,
        )


@given(frequency=_frequencies, a=_angles, b=_angles)
@settings(max_examples=50)
def test_irrep_matrix_is_a_group_homomorphism(
    frequency: int,
    a: float,
    b: float,
) -> None:
    first = irrep_matrix(frequency, torch.tensor(a, dtype=torch.float64))
    second = irrep_matrix(frequency, torch.tensor(b, dtype=torch.float64))

    composed = first @ second
    expected = irrep_matrix(
        frequency,
        torch.tensor(a + b, dtype=torch.float64),
    )

    assert torch.allclose(composed, expected, atol=1e-8)


def test_irrep_matrix_is_orthogonal_with_unit_determinant() -> None:
    angles = torch.linspace(0, 4 * math.pi, 23, dtype=torch.float64)

    for frequency in (1, 2, 3):
        matrices = irrep_matrix(frequency, angles)
        gram = matrices.transpose(-1, -2) @ matrices

        identity = torch.eye(2, dtype=torch.float64).expand_as(gram)
        assert torch.allclose(gram, identity, atol=ATOL)

        determinant = torch.linalg.det(matrices)
        assert torch.allclose(
            determinant,
            torch.ones_like(determinant),
            atol=ATOL,
        )


def test_irrep_matrix_is_two_pi_periodic() -> None:
    angles = torch.linspace(0, 2 * math.pi, 11, dtype=torch.float64)

    for frequency in (0, 1, 4):
        assert torch.allclose(
            irrep_matrix(frequency, angles),
            irrep_matrix(frequency, angles + 2 * math.pi),
            atol=1e-9,
        )


def test_irrep_matrix_rejects_non_float_angles() -> None:
    with pytest.raises(TypeError):
        irrep_matrix(1, torch.zeros(3, dtype=torch.long))


# --------------------------------------------------------------------------
# irrep_phase and its agreement with irrep_matrix
# --------------------------------------------------------------------------


def test_irrep_phase_has_unit_modulus() -> None:
    angles = torch.linspace(-3.0, 3.0, 17, dtype=torch.float64)

    for frequency in (0, 1, 2, 5):
        phases = irrep_phase(frequency, angles)
        assert torch.allclose(
            phases.abs(),
            torch.ones_like(angles),
            atol=ATOL,
        )


def test_irrep_phase_of_the_trivial_representation_is_one() -> None:
    angles = torch.linspace(-3.0, 3.0, 5, dtype=torch.float64)
    phases = irrep_phase(0, angles)

    assert torch.allclose(
        phases,
        torch.ones_like(phases),
        atol=ATOL,
    )


@given(frequency=st.integers(min_value=1, max_value=6), angle=_angles)
@settings(max_examples=50)
def test_irrep_phase_agrees_with_irrep_matrix(
    frequency: int,
    angle: float,
) -> None:
    """The load-bearing equivalence: complex multiply == 2x2 matmul."""
    angle_tensor = torch.tensor(angle, dtype=torch.float64)
    vectors = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.3, -1.7], [-2.5, 0.9]],
        dtype=torch.float64,
    )  # (4, 2)

    matrix = irrep_matrix(frequency, angle_tensor)  # (2, 2)
    by_matrix = vectors @ matrix.transpose(-1, -2)  # (4, 2)

    by_phase = as_real(
        as_complex(vectors) * irrep_phase(frequency, angle_tensor)
    )  # (4, 2)

    assert torch.allclose(by_matrix, by_phase, atol=1e-9)


# --------------------------------------------------------------------------
# phase_powers
# --------------------------------------------------------------------------


def test_phase_powers_has_expected_shape_and_entries() -> None:
    angles = torch.linspace(0.0, 2.0, 9, dtype=torch.float64)
    max_frequency = 4

    table = phase_powers(angles, max_frequency)

    assert table.shape == (9, max_frequency + 1)
    assert table.dtype == complex_dtype(angles.dtype)

    for frequency in range(max_frequency + 1):
        assert torch.allclose(
            table[..., frequency],
            irrep_phase(frequency, angles),
            atol=ATOL,
        )


def test_phase_powers_zeroth_entry_is_one() -> None:
    angles = torch.randn(6, dtype=torch.float64)
    table = phase_powers(angles, 3)

    assert torch.allclose(
        table[..., 0],
        torch.ones_like(table[..., 0]),
        atol=ATOL,
    )


def test_phase_powers_supports_max_frequency_zero() -> None:
    angles = torch.randn(4, dtype=torch.float64)
    table = phase_powers(angles, 0)

    assert table.shape == (4, 1)


def test_phase_powers_preserves_leading_shape() -> None:
    angles = torch.randn(3, 5, dtype=torch.float64)
    assert phase_powers(angles, 2).shape == (3, 5, 3)


def test_phase_powers_rejects_negative_max_frequency() -> None:
    with pytest.raises(ValueError):
        phase_powers(torch.zeros(3, dtype=torch.float64), -1)


# --------------------------------------------------------------------------
# as_complex / as_real
# --------------------------------------------------------------------------


def test_as_complex_and_as_real_round_trip() -> None:
    coefficients = torch.randn(7, 3, 2, dtype=torch.float64)

    assert torch.equal(as_real(as_complex(coefficients)), coefficients)


def test_as_complex_maps_components_to_real_and_imaginary_parts() -> None:
    coefficients = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    values = as_complex(coefficients)

    assert torch.allclose(values.real, torch.tensor([1.0], dtype=torch.float64))
    assert torch.allclose(values.imag, torch.tensor([2.0], dtype=torch.float64))


def test_as_complex_survives_a_non_contiguous_input() -> None:
    # Slicing a wider buffer yields a strided view that view_as_complex
    # cannot consume directly.
    buffer = torch.randn(4, 6, dtype=torch.float64)
    strided = buffer[:, 1:3]

    assert not strided.is_contiguous()
    assert torch.allclose(as_real(as_complex(strided)), strided, atol=ATOL)


def test_as_complex_survives_a_contiguous_odd_offset_slice() -> None:
    """A 1-D slice can be contiguous while starting at an odd offset.

    ``buffer[1:3]`` is a slice along the tensor's only dimension, so its
    strides already match a fresh tensor's -- ``is_contiguous()`` is True --
    but its storage offset is 1. ``view_as_complex`` requires an even offset
    regardless of contiguity, and ``.contiguous()`` alone is a no-op here
    (nothing about strides needs fixing), so this exercises the fallback
    clone rather than the non-contiguous branch above.
    """
    buffer = torch.randn(6, dtype=torch.float64)
    odd_offset = buffer[1:3]

    assert odd_offset.is_contiguous()
    assert odd_offset.storage_offset() % 2 != 0
    assert torch.allclose(as_real(as_complex(odd_offset)), odd_offset, atol=ATOL)


def test_as_complex_gradients_flow_back_to_the_real_tensor() -> None:
    coefficients = torch.randn(5, 2, dtype=torch.float64, requires_grad=True)

    values = as_complex(coefficients)
    (values.abs() ** 2).sum().backward()

    assert coefficients.grad is not None
    assert torch.allclose(
        coefficients.grad,
        2.0 * coefficients.detach(),
        atol=1e-9,
    )


def test_as_complex_rejects_a_trailing_dimension_other_than_two() -> None:
    with pytest.raises(ValueError):
        as_complex(torch.randn(4, 3, dtype=torch.float64))


def test_as_complex_rejects_integer_dtypes() -> None:
    with pytest.raises(TypeError):
        as_complex(torch.zeros(4, 2, dtype=torch.long))


def test_as_real_rejects_real_input() -> None:
    with pytest.raises(TypeError):
        as_real(torch.randn(4, dtype=torch.float64))


@pytest.mark.parametrize(
    ("real", "complex_"),
    [(torch.float32, torch.complex64), (torch.float64, torch.complex128)],
)
def test_dtype_pairing_is_a_bijection(
    real: torch.dtype,
    complex_: torch.dtype,
) -> None:
    assert complex_dtype(real) == complex_
    assert real_dtype(complex_) == real


def test_complex_dtype_rejects_unsupported_dtype() -> None:
    with pytest.raises(TypeError):
        complex_dtype(torch.float16)


# --------------------------------------------------------------------------
# rotate_irrep_coefficients
# --------------------------------------------------------------------------


def test_rotate_irrep_coefficients_is_the_identity_at_frequency_zero() -> None:
    coefficients = torch.randn(4, 3, dtype=torch.float64)
    angles = torch.randn(4, dtype=torch.float64)

    assert torch.equal(
        rotate_irrep_coefficients(0, coefficients, angles),
        coefficients,
    )


def test_rotate_irrep_coefficients_matches_the_dense_matrix_action() -> None:
    coefficients = torch.randn(6, 2, dtype=torch.float64)
    angles = torch.randn(6, dtype=torch.float64)

    for frequency in (1, 2, 3):
        rotated = rotate_irrep_coefficients(frequency, coefficients, angles)

        matrices = irrep_matrix(frequency, angles)  # (6, 2, 2)
        expected = torch.einsum("nij,nj->ni", matrices, coefficients)

        assert torch.allclose(rotated, expected, atol=1e-9)


def test_rotate_irrep_coefficients_composes_by_angle_addition() -> None:
    coefficients = torch.randn(5, 2, dtype=torch.float64)
    first = torch.randn(5, dtype=torch.float64)
    second = torch.randn(5, dtype=torch.float64)

    twice = rotate_irrep_coefficients(
        3,
        rotate_irrep_coefficients(3, coefficients, first),
        second,
    )
    once = rotate_irrep_coefficients(3, coefficients, first + second)

    assert torch.allclose(twice, once, atol=1e-9)


def test_rotate_irrep_coefficients_preserves_the_norm() -> None:
    coefficients = torch.randn(8, 2, dtype=torch.float64)
    angles = torch.randn(8, dtype=torch.float64)

    rotated = rotate_irrep_coefficients(2, coefficients, angles)

    assert torch.allclose(
        torch.linalg.vector_norm(rotated, dim=-1),
        torch.linalg.vector_norm(coefficients, dim=-1),
        atol=1e-9,
    )


def test_rotate_irrep_coefficients_broadcasts_scalar_angles() -> None:
    coefficients = torch.randn(4, 3, 2, dtype=torch.float64)
    angle = torch.tensor(0.7, dtype=torch.float64)

    rotated = rotate_irrep_coefficients(1, coefficients, angle)

    assert rotated.shape == coefficients.shape


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rotate_irrep_coefficients_supports_both_dtypes(
    dtype: torch.dtype,
) -> None:
    coefficients = torch.randn(4, 2, dtype=dtype)
    angles = torch.randn(4, dtype=dtype)

    rotated = rotate_irrep_coefficients(2, coefficients, angles)

    assert rotated.dtype == dtype
    assert rotated.shape == coefficients.shape
