"""Tests for gm_attention.representations.intertwiners.

Two things matter here, independently of each other:

* ``hom_basis`` / ``dense_intertwiner_block`` must produce matrices that
  really do satisfy the intertwiner constraint -- checked against
  ``is_intertwiner``, which tests the defining equation directly via
  ``so2.irrep_matrix`` rather than via basis membership; and
* ``dense_field_type_intertwiner`` must assemble those blocks into a matrix
  that both intertwines the *full* representations and matches the channel
  ordering ``SO2FeatureField`` actually uses.
"""

from __future__ import annotations

import math

import pytest
import torch

from gm_attention.representations.field_type import (
    SO2FeatureField,
    SO2FieldType,
)
from tests.support.intertwiners import (
    dense_field_type_intertwiner,
    dense_intertwiner_block,
    hom_basis,
    hom_dimension,
    is_intertwiner,
)
from gm_attention.representations.so2 import irrep_matrix

ATOL = 1e-9


def _block_diagonal_representation(
    field_type: SO2FieldType,
    angle: torch.Tensor,
) -> torch.Tensor:
    """Dense ``sum_n I_{a_n} (x) rho_n(r_theta)`` in the packed layout."""
    blocks = [
        irrep_matrix(frequency, angle)
        for frequency, multiplicity in field_type.items()
        for _ in range(multiplicity)
    ]

    return torch.block_diag(*blocks)


# --------------------------------------------------------------------------
# hom_dimension / hom_basis
# --------------------------------------------------------------------------


def test_hom_dimension_trivial_case() -> None:
    assert hom_dimension(0, 0) == 1


@pytest.mark.parametrize("frequency", [1, 2, 5])
def test_hom_dimension_matching_nontrivial_frequencies(frequency: int) -> None:
    assert hom_dimension(frequency, frequency) == 2


@pytest.mark.parametrize(
    ("n", "m"), [(0, 1), (1, 0), (1, 2), (2, 1), (3, 5)]
)
def test_hom_dimension_is_zero_for_inequivalent_types(n: int, m: int) -> None:
    assert hom_dimension(n, m) == 0


def test_hom_basis_trivial_case_is_the_scalar_one() -> None:
    basis = hom_basis(0, 0, dtype=torch.float64)
    assert basis.shape == (1, 1, 1)
    assert torch.allclose(basis, torch.ones(1, 1, 1, dtype=torch.float64))


def test_hom_basis_nontrivial_case_is_i_and_j() -> None:
    basis = hom_basis(3, 3, dtype=torch.float64)
    expected = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]], [[0.0, -1.0], [1.0, 0.0]]],
        dtype=torch.float64,
    )
    assert torch.allclose(basis, expected)


def test_hom_basis_is_empty_for_inequivalent_types() -> None:
    basis = hom_basis(1, 2, dtype=torch.float64)
    assert basis.shape == (0, 2, 2)


# --------------------------------------------------------------------------
# is_intertwiner: the independent defining-equation check
# --------------------------------------------------------------------------


def test_identity_is_an_intertwiner_of_any_type_with_itself() -> None:
    for frequency in (0, 1, 2, 4):
        dimension = 1 if frequency == 0 else 2
        identity = torch.eye(dimension, dtype=torch.float64)
        assert is_intertwiner(frequency, frequency, identity)


def test_zero_matrix_is_always_an_intertwiner() -> None:
    assert is_intertwiner(1, 3, torch.zeros(2, 2, dtype=torch.float64))


def test_basis_elements_are_intertwiners() -> None:
    for frequency in (1, 2, 3):
        basis = hom_basis(frequency, frequency, dtype=torch.float64)
        for element in basis:
            assert is_intertwiner(frequency, frequency, element)


def test_a_generic_matrix_is_not_an_intertwiner_of_distinct_frequencies() -> None:
    matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    assert not is_intertwiner(1, 2, matrix)


def test_a_generic_matrix_is_not_an_intertwiner_of_equal_frequencies() -> None:
    # A generic 2x2 matrix is not of the form MI + NJ (e.g. not symmetric
    # under swapping the off-diagonal signs the way I and J are).
    matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    assert not is_intertwiner(2, 2, matrix)


def test_is_intertwiner_rejects_the_wrong_shape() -> None:
    with pytest.raises(ValueError):
        is_intertwiner(1, 1, torch.zeros(3, 3))


# --------------------------------------------------------------------------
# dense_intertwiner_block, cross-checked against is_intertwiner
# --------------------------------------------------------------------------


@pytest.mark.parametrize("frequency", [0, 1, 2, 4])
def test_dense_intertwiner_block_is_always_an_intertwiner(frequency: int) -> None:
    dimension = hom_dimension(frequency, frequency)
    generator = torch.Generator().manual_seed(0)
    coefficients = torch.randn(dimension, dtype=torch.float64, generator=generator)

    block = dense_intertwiner_block(frequency, frequency, coefficients)
    assert is_intertwiner(frequency, frequency, block)


def test_dense_intertwiner_block_is_zero_for_inequivalent_types() -> None:
    block = dense_intertwiner_block(1, 2, torch.zeros(0, dtype=torch.float64))
    assert block.shape == (2, 2)
    assert torch.allclose(block, torch.zeros(2, 2, dtype=torch.float64))


def test_dense_intertwiner_block_broadcasts_a_batch_of_coefficients() -> None:
    coefficients = torch.randn(5, 3, 2, dtype=torch.float64)
    block = dense_intertwiner_block(2, 2, coefficients)
    assert block.shape == (5, 3, 2, 2)


def test_dense_intertwiner_block_rejects_the_wrong_trailing_dimension() -> None:
    with pytest.raises(ValueError):
        dense_intertwiner_block(2, 2, torch.zeros(3, dtype=torch.float64))


def test_dense_intertwiner_block_matches_the_hom_basis_expansion() -> None:
    coefficients = torch.tensor([0.3, -1.7], dtype=torch.float64)
    block = dense_intertwiner_block(2, 2, coefficients)

    basis = hom_basis(2, 2, dtype=torch.float64)
    expected = coefficients[0] * basis[0] + coefficients[1] * basis[1]

    assert torch.allclose(block, expected, atol=ATOL)


# --------------------------------------------------------------------------
# dense_field_type_intertwiner
# --------------------------------------------------------------------------


def test_dense_field_type_intertwiner_has_the_expected_shape() -> None:
    field_type_in = SO2FieldType({0: 2, 1: 3, 2: 1})
    field_type_out = SO2FieldType({0: 1, 1: 2, 3: 1})

    weights = {
        0: torch.randn(1, 2, dtype=torch.float64),
        1: torch.randn(2, 3, dtype=torch.complex128),
    }

    matrix = dense_field_type_intertwiner(field_type_in, field_type_out, weights)

    assert matrix.shape == (
        field_type_out.dimension,
        field_type_in.dimension,
    )


def test_dense_field_type_intertwiner_vanishes_between_inequivalent_types() -> (
    None
):
    field_type_in = SO2FieldType({0: 1, 2: 1})
    field_type_out = SO2FieldType({0: 1, 3: 1})

    matrix = dense_field_type_intertwiner(
        field_type_in,
        field_type_out,
        {0: torch.randn(1, 1, dtype=torch.float64)},
    )

    out_slices = field_type_out.slices()
    in_slices = field_type_in.slices()

    # The only common frequency is 0; every other block must be exactly zero.
    cross_block = matrix[out_slices[3], in_slices[2]]
    assert torch.equal(cross_block, torch.zeros_like(cross_block))


def test_dense_field_type_intertwiner_satisfies_the_global_constraint() -> None:
    """W = rho_out(g) W rho_in(g)^{-1} for every g, the load-bearing property."""
    field_type_in = SO2FieldType({0: 2, 1: 3, 2: 1})
    field_type_out = SO2FieldType({0: 1, 1: 2, 2: 4, 3: 1})

    generator = torch.Generator().manual_seed(1)
    weights = {
        0: torch.randn(1, 2, dtype=torch.float64, generator=generator),
        1: torch.randn(2, 3, dtype=torch.complex128, generator=generator),
        2: torch.randn(4, 1, dtype=torch.complex128, generator=generator),
    }

    matrix = dense_field_type_intertwiner(field_type_in, field_type_out, weights)

    for angle_value in (0.0, 0.7391, math.pi / 2, 4.0):
        angle = torch.tensor(angle_value, dtype=torch.float64)

        rep_in = _block_diagonal_representation(field_type_in, angle)
        rep_out = _block_diagonal_representation(field_type_out, angle)

        lhs = matrix
        rhs = rep_out @ matrix @ torch.linalg.inv(rep_in)

        assert torch.allclose(lhs, rhs, atol=1e-8)


def test_dense_field_type_intertwiner_matches_so2_feature_field_layout() -> None:
    """Applying the dense matrix to to_flat() must match a per-block apply."""
    field_type_in = SO2FieldType({0: 2, 1: 2})
    field_type_out = SO2FieldType({0: 1, 1: 3})

    generator = torch.Generator().manual_seed(2)
    weights = {
        0: torch.randn(1, 2, dtype=torch.float64, generator=generator),
        1: torch.randn(3, 2, dtype=torch.complex128, generator=generator),
    }

    matrix = dense_field_type_intertwiner(field_type_in, field_type_out, weights)

    field = SO2FeatureField.from_flat(
        field_type_in,
        torch.randn(field_type_in.dimension, dtype=torch.float64, generator=generator),
    )

    via_matrix = matrix @ field.to_flat()

    # Per-frequency, apply the complex weight directly and compare.
    from gm_attention.representations.so2 import as_complex, as_real

    expected_freq_0 = weights[0] @ field[0]
    expected_freq_1 = as_real(weights[1] @ as_complex(field[1])).flatten()

    expected = torch.cat((expected_freq_0, expected_freq_1))

    assert torch.allclose(via_matrix, expected, atol=1e-9)


def test_dense_field_type_intertwiner_rejects_missing_weights() -> None:
    field_type_in = SO2FieldType({0: 1, 1: 1})
    field_type_out = SO2FieldType({0: 1, 1: 1})

    with pytest.raises(ValueError):
        dense_field_type_intertwiner(field_type_in, field_type_out, {0: torch.zeros(1, 1)})


def test_dense_field_type_intertwiner_rejects_extra_weights() -> None:
    field_type_in = SO2FieldType({0: 1})
    field_type_out = SO2FieldType({0: 1})

    with pytest.raises(ValueError):
        dense_field_type_intertwiner(
            field_type_in,
            field_type_out,
            {0: torch.zeros(1, 1, dtype=torch.float64), 1: torch.zeros(1, 1, dtype=torch.complex128)},
        )


def test_dense_field_type_intertwiner_rejects_a_real_weight_at_nonzero_frequency() -> (
    None
):
    field_type_in = SO2FieldType({1: 1})
    field_type_out = SO2FieldType({1: 1})

    with pytest.raises(TypeError):
        dense_field_type_intertwiner(
            field_type_in,
            field_type_out,
            {1: torch.zeros(1, 1, dtype=torch.float64)},
        )


def test_dense_field_type_intertwiner_rejects_a_complex_weight_at_frequency_zero() -> (
    None
):
    field_type_in = SO2FieldType({0: 1})
    field_type_out = SO2FieldType({0: 1})

    with pytest.raises(TypeError):
        dense_field_type_intertwiner(
            field_type_in,
            field_type_out,
            {0: torch.zeros(1, 1, dtype=torch.complex128)},
        )


def test_dense_field_type_intertwiner_rejects_the_wrong_weight_shape() -> None:
    field_type_in = SO2FieldType({0: 2})
    field_type_out = SO2FieldType({0: 3})

    with pytest.raises(ValueError):
        dense_field_type_intertwiner(
            field_type_in,
            field_type_out,
            {0: torch.zeros(3, 5, dtype=torch.float64)},
        )


def test_dense_field_type_intertwiner_with_no_common_frequencies_needs_dtype_and_device() -> (
    None
):
    field_type_in = SO2FieldType({1: 1})
    field_type_out = SO2FieldType({2: 1})

    with pytest.raises(ValueError):
        dense_field_type_intertwiner(field_type_in, field_type_out, {})

    matrix = dense_field_type_intertwiner(
        field_type_in,
        field_type_out,
        {},
        dtype=torch.float64,
        device=torch.device("cpu"),
    )

    assert matrix.shape == (2, 2)
    assert torch.equal(matrix, torch.zeros(2, 2, dtype=torch.float64))


def test_dense_field_type_intertwiner_infers_dtype_and_device_from_weights() -> None:
    field_type_in = SO2FieldType({0: 1, 1: 1})
    field_type_out = SO2FieldType({0: 1, 1: 1})

    weights = {
        0: torch.zeros(1, 1, dtype=torch.float32),
        1: torch.zeros(1, 1, dtype=torch.complex64),
    }

    matrix = dense_field_type_intertwiner(field_type_in, field_type_out, weights)
    assert matrix.dtype == torch.float32


def test_dense_field_type_intertwiner_rejects_a_conflicting_dtype() -> None:
    field_type_in = SO2FieldType({0: 1})
    field_type_out = SO2FieldType({0: 1})

    with pytest.raises(ValueError):
        dense_field_type_intertwiner(
            field_type_in,
            field_type_out,
            {0: torch.zeros(1, 1, dtype=torch.float64)},
            dtype=torch.float32,
        )
