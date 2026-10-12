"""Tests for gm_attention.representations.bases.

The load-bearing check is that ``steerable_basis`` satisfies its defining
equation

    K_i(phi) = rho_m(r_phi) A_i rho_n(r_phi)^{-1},  A_i := K_i(0),

checked against ``so2.irrep_matrix``, an independent code path
from the literal trigonometry ``steerable_basis`` itself uses.
``dense_field_type_steerable_kernel`` is then checked against the actual
steerability constraint
``V(gv) = rho_out(g) V(v) rho_in(g)^{-1}`` for the assembled,
multi-frequency, multi-multiplicity case.
"""

from __future__ import annotations

import math

import pytest
import torch

from tests.support.bases import (
    dense_field_type_steerable_kernel,
    dense_steerable_block,
    steerable_basis,
    steerable_basis_dimension,
)
from gm_attention.representations.field_type import SO2FieldType
from gm_attention.representations.so2 import irrep_matrix

ATOL = 1e-9


def _block_diagonal_representation(
    field_type: SO2FieldType,
    angle: torch.Tensor,
) -> torch.Tensor:
    blocks = [
        irrep_matrix(frequency, angle)
        for frequency, multiplicity in field_type.items()
        for _ in range(multiplicity)
    ]

    return torch.block_diag(*blocks)


# --------------------------------------------------------------------------
# steerable_basis_dimension
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "m", "expected"),
    [(0, 0, 1), (0, 3, 2), (3, 0, 2), (1, 1, 4), (2, 5, 4)],
)
def test_steerable_basis_dimension_is_d_m_times_d_n(
    n: int,
    m: int,
    expected: int,
) -> None:
    assert steerable_basis_dimension(n, m) == expected


# --------------------------------------------------------------------------
# steerable_basis: shape, and the defining steerability equation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "m", "d_m", "d_n"),
    [(0, 0, 1, 1), (0, 2, 2, 1), (3, 0, 1, 2), (2, 3, 2, 2)],
)
def test_steerable_basis_has_the_expected_shape(
    n: int,
    m: int,
    d_m: int,
    d_n: int,
) -> None:
    angles = torch.linspace(0.0, 2 * math.pi, 5, dtype=torch.float64)
    basis = steerable_basis(n, m, angles)

    assert basis.shape == (5, steerable_basis_dimension(n, m), d_m, d_n)


def test_steerable_basis_preserves_leading_batch_shape() -> None:
    angles = torch.randn(3, 4, dtype=torch.float64)
    basis = steerable_basis(2, 1, angles)
    assert basis.shape == (3, 4, 4, 2, 2)


@pytest.mark.parametrize(
    ("n", "m"), [(0, 0), (0, 1), (0, 3), (1, 0), (2, 0), (1, 1), (1, 2), (2, 3)]
)
def test_steerable_basis_satisfies_its_defining_equation(n: int, m: int) -> None:
    """K_i(phi) = rho_m(r_phi) A_i rho_n(r_phi)^{-1}, independently verified.

    A_i is read off as K_i(0); the equation is then checked at other angles
    using so2.irrep_matrix, a completely different code path from the
    trigonometric identities steerable_basis is built from.
    """
    zero = torch.zeros(1, dtype=torch.float64)
    reference_basis = steerable_basis(n, m, zero)[0]  # (dim, d_m, d_n)

    angles = torch.tensor(
        [0.1, 0.7391, math.pi / 2, 2.3, -1.4],
        dtype=torch.float64,
    )
    basis = steerable_basis(n, m, angles)  # (5, dim, d_m, d_n)

    output_matrices = irrep_matrix(m, angles)  # (5, d_m, d_m)
    input_inverses = irrep_matrix(n, -angles)  # (5, d_n, d_n)

    # expected[s, i] = output_matrices[s] @ reference_basis[i] @ input_inverses[s]
    expected = torch.einsum(
        "sqp,ipr,srt->siqt",
        output_matrices,
        reference_basis,
        input_inverses,
    )

    assert torch.allclose(basis, expected, atol=1e-8)


def test_steerable_basis_reference_matrices_match_table_1_conventions() -> None:
    """K_i(0) = A_i should be the elementary maps named in the text."""
    zero = torch.zeros(1, dtype=torch.float64)

    # n = 0, m = 0: A_1 = 1.
    assert torch.allclose(
        steerable_basis(0, 0, zero)[0, 0],
        torch.ones(1, 1, dtype=torch.float64),
    )

    # n = 0, m >= 1: A_1 = e_1, A_2 = e_2.
    basis = steerable_basis(0, 2, zero)[0]
    assert torch.allclose(basis[0], torch.tensor([[1.0], [0.0]], dtype=torch.float64))
    assert torch.allclose(basis[1], torch.tensor([[0.0], [1.0]], dtype=torch.float64))

    # n >= 1, m = 0: A_1 = e_1^T, A_2 = e_2^T.
    basis = steerable_basis(2, 0, zero)[0]
    assert torch.allclose(basis[0], torch.tensor([[1.0, 0.0]], dtype=torch.float64))
    assert torch.allclose(basis[1], torch.tensor([[0.0, 1.0]], dtype=torch.float64))

    # n, m >= 1: A_1..A_4 = I, J, S, JS.
    basis = steerable_basis(1, 1, zero)[0]
    identity = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    j = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float64)
    s = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.float64)
    assert torch.allclose(basis[0], identity)
    assert torch.allclose(basis[1], j)
    assert torch.allclose(basis[2], s)
    assert torch.allclose(basis[3], j @ s)


def test_steerable_basis_rejects_non_floating_angles() -> None:
    with pytest.raises(TypeError):
        steerable_basis(1, 1, torch.zeros(3, dtype=torch.long))


# --------------------------------------------------------------------------
# dense_steerable_block
# --------------------------------------------------------------------------


def test_dense_steerable_block_matches_a_manual_expansion() -> None:
    angles = torch.tensor([0.3, 1.1], dtype=torch.float64)
    coefficients = torch.tensor(
        [[1.0, -2.0, 0.5, 3.0], [0.2, 0.1, -1.0, 0.4]],
        dtype=torch.float64,
    )

    block = dense_steerable_block(1, 2, coefficients, angles)

    basis = steerable_basis(1, 2, angles)  # (2, 4, 2, 2)
    expected = torch.einsum("si,sipq->spq", coefficients, basis)

    assert torch.allclose(block, expected, atol=ATOL)


def test_dense_steerable_block_rejects_the_wrong_trailing_dimension() -> None:
    with pytest.raises(ValueError):
        dense_steerable_block(
            1, 1, torch.zeros(3, dtype=torch.float64), torch.zeros(1, dtype=torch.float64)
        )


# --------------------------------------------------------------------------
# dense_field_type_steerable_kernel
# --------------------------------------------------------------------------


def test_dense_field_type_steerable_kernel_has_the_expected_shape() -> None:
    field_type_in = SO2FieldType({0: 1, 1: 2})
    field_type_out = SO2FieldType({0: 2, 1: 1, 2: 1})

    angles = torch.linspace(0.0, 1.0, 3, dtype=torch.float64)
    radial_coefficients = {
        (1, 1): torch.randn(4, 1, 2, dtype=torch.float64),
        (0, 2): torch.randn(2, 1, 1, dtype=torch.float64),
    }

    kernel = dense_field_type_steerable_kernel(
        field_type_in, field_type_out, angles, radial_coefficients
    )

    assert kernel.shape == (
        3,
        field_type_out.dimension,
        field_type_in.dimension,
    )


def test_dense_field_type_steerable_kernel_places_blocks_at_the_right_slices() -> (
    None
):
    field_type_in = SO2FieldType({0: 1, 1: 1})
    field_type_out = SO2FieldType({1: 1, 2: 1})

    angles = torch.tensor([0.4], dtype=torch.float64)
    coefficients_0_1 = torch.randn(2, 1, 1, dtype=torch.float64)  # (n=0, m=1)

    kernel = dense_field_type_steerable_kernel(
        field_type_in,
        field_type_out,
        angles,
        {(0, 1): coefficients_0_1},
    )[0]

    out_slices = field_type_out.slices()
    in_slices = field_type_in.slices()

    # dense_steerable_block operates at multiplicity one, i.e. (..., dim)
    # rather than the (..., dim, b_m, a_n) shape a field-type-level pair
    # coefficient carries; squeeze out the singleton multiplicities to
    # compare.
    expected_block = dense_steerable_block(
        0, 1, coefficients_0_1.squeeze(-1).squeeze(-1), angles
    )[0]
    assert torch.allclose(
        kernel[out_slices[1], in_slices[0]],
        expected_block,
        atol=ATOL,
    )

    # Every other block (unlisted pairs) must be exactly zero.
    other_block = kernel[out_slices[2], in_slices[0]]
    assert torch.equal(other_block, torch.zeros_like(other_block))
    other_block = kernel[out_slices[1], in_slices[1]]
    assert torch.equal(other_block, torch.zeros_like(other_block))


def test_dense_field_type_steerable_kernel_allows_cross_frequency_blocks() -> None:
    """Unlike an intertwiner, m != n blocks are allowed to be nonzero."""
    field_type_in = SO2FieldType({1: 1})
    field_type_out = SO2FieldType({3: 1})

    angles = torch.tensor([0.4], dtype=torch.float64)
    coefficients = torch.randn(4, 1, 1, dtype=torch.float64)

    kernel = dense_field_type_steerable_kernel(
        field_type_in,
        field_type_out,
        angles,
        {(1, 3): coefficients},
    )

    assert not torch.allclose(kernel, torch.zeros_like(kernel))


def test_dense_field_type_steerable_kernel_satisfies_the_steerability_constraint() -> (
    None
):
    """V(gv) = rho_out(g) V(v) rho_in(g)^{-1}, the load-bearing property.

    Checked at v = r r_phi e_1 for fixed r and varying phi, against g = r_theta.
    """
    field_type_in = SO2FieldType({0: 1, 1: 2})
    field_type_out = SO2FieldType({0: 2, 1: 1, 2: 1})

    generator = torch.Generator().manual_seed(3)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, dtype=torch.float64, generator=generator)

    radial_coefficients = {
        (0, 0): randn(1, 2, 1),
        (1, 0): randn(2, 2, 2),
        (0, 1): randn(2, 1, 1),
        (1, 1): randn(4, 1, 2),
        (1, 2): randn(4, 1, 2),
    }

    phi = torch.tensor(0.5, dtype=torch.float64)
    theta = torch.tensor(1.1, dtype=torch.float64)

    def kernel_at(angle: torch.Tensor) -> torch.Tensor:
        return dense_field_type_steerable_kernel(
            field_type_in,
            field_type_out,
            angle.reshape(1),
            radial_coefficients,
        )[0]

    lhs = kernel_at(phi + theta)

    rep_in = _block_diagonal_representation(field_type_in, theta)
    rep_out = _block_diagonal_representation(field_type_out, theta)
    rhs = rep_out @ kernel_at(phi) @ torch.linalg.inv(rep_in)

    assert torch.allclose(lhs, rhs, atol=1e-7)


def test_dense_field_type_steerable_kernel_broadcasts_batched_coefficients() -> (
    None
):
    field_type_in = SO2FieldType({1: 1})
    field_type_out = SO2FieldType({1: 1})

    angles = torch.linspace(0.0, 1.0, 6, dtype=torch.float64)
    # Coefficients that vary per edge, e.g. as a radial network's output.
    coefficients = torch.randn(6, 4, 1, 1, dtype=torch.float64)

    kernel = dense_field_type_steerable_kernel(
        field_type_in, field_type_out, angles, {(1, 1): coefficients}
    )

    assert kernel.shape == (6, 2, 2)

    for edge in range(6):
        expected = dense_steerable_block(
            1,
            1,
            coefficients[edge].squeeze(-1).squeeze(-1),
            angles[edge].reshape(()),
        )
        assert torch.allclose(kernel[edge], expected, atol=ATOL)


def test_dense_field_type_steerable_kernel_rejects_an_unknown_input_frequency() -> (
    None
):
    field_type_in = SO2FieldType({0: 1})
    field_type_out = SO2FieldType({0: 1})

    with pytest.raises(ValueError):
        dense_field_type_steerable_kernel(
            field_type_in,
            field_type_out,
            torch.zeros(1, dtype=torch.float64),
            {(1, 0): torch.zeros(2, 1, 1, dtype=torch.float64)},
        )


def test_dense_field_type_steerable_kernel_rejects_an_unknown_output_frequency() -> (
    None
):
    field_type_in = SO2FieldType({0: 1})
    field_type_out = SO2FieldType({0: 1})

    with pytest.raises(ValueError):
        dense_field_type_steerable_kernel(
            field_type_in,
            field_type_out,
            torch.zeros(1, dtype=torch.float64),
            {(0, 1): torch.zeros(2, 1, 1, dtype=torch.float64)},
        )


def test_dense_field_type_steerable_kernel_rejects_the_wrong_coefficient_shape() -> (
    None
):
    field_type_in = SO2FieldType({0: 2})
    field_type_out = SO2FieldType({0: 3})

    with pytest.raises(ValueError):
        dense_field_type_steerable_kernel(
            field_type_in,
            field_type_out,
            torch.zeros(1, dtype=torch.float64),
            {(0, 0): torch.zeros(1, 5, 5, dtype=torch.float64)},
        )


def test_dense_field_type_steerable_kernel_with_no_coefficients_is_zero() -> None:
    field_type_in = SO2FieldType({0: 1})
    field_type_out = SO2FieldType({0: 1})

    kernel = dense_field_type_steerable_kernel(
        field_type_in,
        field_type_out,
        torch.zeros(3, dtype=torch.float64),
        {},
    )

    assert torch.equal(kernel, torch.zeros(3, 1, 1, dtype=torch.float64))
