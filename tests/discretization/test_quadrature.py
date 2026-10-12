"""Numerical accuracy of the spherical quadrature rules on test functions.

Each grid builder in gm_attention.discretization.grids returns a
SphericalGrid whose (node, weight) pairs are meant to approximate

    integral_{S^2} f dOmega ~= sum_i w_i f(x_i).

This file checks that approximation against closed-form reference
integrals: exact low-degree monomials (which every rule here reproduces to
machine precision) and a smooth, non-polynomial test function (which is
only approximated, so we check that the error shrinks as the grid is
refined).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import partial

import pytest
import torch
from torch import Tensor

from gm_attention.discretization.grid import SphericalGrid
from gm_attention.discretization.grids.icosphere import build_icosphere_grid
from gm_attention.discretization.grids.torch_harmonics import (
    build_torch_harmonics_grid,
)

FOUR_PI = 4.0 * math.pi

# Reference: integral_{S^2} exp(p . v) dOmega(v) = 4 * pi * sinh(|p|) / |p|
# for a constant vector p, here p = (1, 0, 0).
REFERENCE_EXP_X = FOUR_PI * math.sinh(1.0)

# A representative grid of "moderate" resolution from each family, used for
# the exactness checks below.
MODERATE_GRIDS: dict[str, Callable[[], SphericalGrid]] = {
    "icosphere": lambda: build_icosphere_grid(2),
    "legendre-gauss": lambda: build_torch_harmonics_grid(
        "legendre-gauss", 8, 16
    ),
    "lobatto": lambda: build_torch_harmonics_grid("lobatto", 8, 16),
    "equiangular": lambda: build_torch_harmonics_grid("equiangular", 8, 16),
}

# Increasing-resolution sequences per family, used for the convergence
# check on a non-polynomial function.
REFINEMENT_SEQUENCES: dict[str, list[Callable[[], SphericalGrid]]] = {
    "icosphere": [partial(build_icosphere_grid, s) for s in (0, 1, 2, 3)],
    "legendre-gauss": [
        partial(build_torch_harmonics_grid, "legendre-gauss", n, 2 * n)
        for n in (2, 4, 8, 16)
    ],
}


def integrate(grid: SphericalGrid, f: Callable[[Tensor], Tensor]) -> float:
    return (grid.quadrature_weights * f(grid.nodes)).sum().item()


def _constant(nodes: Tensor) -> Tensor:
    return torch.ones(nodes.shape[0], dtype=nodes.dtype, device=nodes.device)


def _x_squared(nodes: Tensor) -> Tensor:
    return nodes[:, 0] ** 2


def _z(nodes: Tensor) -> Tensor:
    return nodes[:, 2]


def _xy(nodes: Tensor) -> Tensor:
    return nodes[:, 0] * nodes[:, 1]


def _exp_x(nodes: Tensor) -> Tensor:
    return torch.exp(nodes[:, 0])


# --------------------------------------------------------------------------
# Exactness on low-degree monomials
# --------------------------------------------------------------------------


@pytest.mark.parametrize("grid_name", sorted(MODERATE_GRIDS))
def test_quadrature_is_exact_for_constant(grid_name: str) -> None:
    grid = MODERATE_GRIDS[grid_name]()
    assert integrate(grid, _constant) == pytest.approx(FOUR_PI, abs=1e-10)


@pytest.mark.parametrize("grid_name", sorted(MODERATE_GRIDS))
def test_quadrature_is_exact_for_x_squared(grid_name: str) -> None:
    # integral x^2 dOmega = 4*pi/3 by symmetry with y^2 and z^2, which sum
    # to the exact integral of 1.
    grid = MODERATE_GRIDS[grid_name]()
    assert integrate(grid, _x_squared) == pytest.approx(
        FOUR_PI / 3.0,
        abs=1e-9,
    )


@pytest.mark.parametrize("grid_name", sorted(MODERATE_GRIDS))
@pytest.mark.parametrize("odd_function", [_z, _xy], ids=["z", "xy"])
def test_quadrature_annihilates_odd_symmetric_functions(
    grid_name: str,
    odd_function: Callable[[Tensor], Tensor],
) -> None:
    # Every grid here is closed under the icosahedral or product-grid
    # symmetries that make z and xy integrate to exactly zero on the
    # continuous sphere.
    grid = MODERATE_GRIDS[grid_name]()
    assert integrate(grid, odd_function) == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# Convergence on a smooth, non-polynomial function
# --------------------------------------------------------------------------


@pytest.mark.parametrize("family", sorted(REFINEMENT_SEQUENCES))
def test_quadrature_error_shrinks_with_refinement(family: str) -> None:
    errors = [
        abs(integrate(build(), _exp_x) - REFERENCE_EXP_X)
        for build in REFINEMENT_SEQUENCES[family]
    ]

    # Error should be (weakly) monotonically decreasing, and the finest
    # grid should be accurate to many digits.
    assert all(
        errors[i + 1] <= errors[i] + 1e-12 for i in range(len(errors) - 1)
    )
    assert errors[-1] < 1e-6


def test_finer_grids_are_more_accurate_than_coarser_ones() -> None:
    coarse = build_icosphere_grid(0)
    fine = build_icosphere_grid(4)

    coarse_error = abs(integrate(coarse, _exp_x) - REFERENCE_EXP_X)
    fine_error = abs(integrate(fine, _exp_x) - REFERENCE_EXP_X)

    assert fine_error < coarse_error
