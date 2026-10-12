"""Tests for gm_attention.training.losses.

The centerpiece is the cross-check against the actual reference
implementation this project is benchmarking against
(`torch_harmonics.examples.losses`) -- not a re-derivation of the same
formula, but a direct numerical comparison on an equiangular grid, which is
the one point set where both implementations are simultaneously well-defined.
That is what makes the icosphere case (which the reference implementation
cannot even represent) trustworthy by extension.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch_harmonics.examples.losses import L1LossS2 as ReferenceL1LossS2
from torch_harmonics.examples.losses import L2LossS2 as ReferenceL2LossS2
from torch_harmonics.examples.losses import SquaredL2LossS2 as ReferenceSquaredL2LossS2
from torch_harmonics.examples.losses import get_quadrature_weights

from gm_attention.training.losses import (
    L1LossS2,
    L2LossS2,
    SphericalLossBase,
    SquaredL2LossS2,
)

NLAT, NLON, BATCH, CHANNELS = 16, 32, 3, 2


def _equiangular_point_major_weights() -> torch.Tensor:
    """Flattened (n_points,) weights matching get_quadrature_weights's own
    per-point convention (tile=True materializes exactly what the reference
    loss's (nlat, 1)-broadcast already applies implicitly)."""
    tiled = get_quadrature_weights(NLAT, NLON, grid="equiangular", tile=True, normalized=True)
    return tiled.reshape(-1).to(torch.float64)


def _channel_first_to_point_major(x: torch.Tensor) -> torch.Tensor:
    # (batch, channels, nlat, nlon) -> (batch, nlat * nlon, channels)
    return x.permute(0, 2, 3, 1).reshape(x.shape[0], NLAT * NLON, CHANNELS)


# ==========================================================================
# Construction validation
# ==========================================================================


def test_rejects_non_tensor_quadrature_weights() -> None:
    with pytest.raises(TypeError):
        SquaredL2LossS2([1.0, 2.0, 3.0])  # type: ignore[arg-type]


def test_rejects_non_1d_quadrature_weights() -> None:
    with pytest.raises(ValueError):
        SquaredL2LossS2(torch.ones(4, 4))


def test_rejects_nonpositive_quadrature_weights() -> None:
    weights = torch.ones(4)
    weights[0] = 0.0
    with pytest.raises(ValueError):
        SquaredL2LossS2(weights)


def test_normalizes_weights_by_their_own_sum() -> None:
    loss = SquaredL2LossS2(torch.full((10,), 7.0, dtype=torch.float64))
    assert torch.allclose(loss.quadrature_weights.sum(), torch.tensor(1.0, dtype=torch.float64))
    assert torch.allclose(
        loss.quadrature_weights, torch.full((10,), 0.1, dtype=torch.float64)
    )


# ==========================================================================
# Cross-check against the actual reference implementation
# ==========================================================================


@pytest.mark.parametrize(
    ("ours", "reference"),
    [
        (SquaredL2LossS2, ReferenceSquaredL2LossS2),
        (L1LossS2, ReferenceL1LossS2),
        (L2LossS2, ReferenceL2LossS2),
    ],
)
def test_matches_the_reference_loss_on_an_equiangular_grid(ours, reference) -> None:
    torch.manual_seed(0)
    prediction = torch.randn(BATCH, CHANNELS, NLAT, NLON, dtype=torch.float64)
    target = torch.randn(BATCH, CHANNELS, NLAT, NLON, dtype=torch.float64)

    reference_loss = reference(NLAT, NLON, grid="equiangular")
    expected = reference_loss(prediction.float(), target.float())

    our_loss = ours(_equiangular_point_major_weights())
    actual = our_loss(
        _channel_first_to_point_major(prediction), _channel_first_to_point_major(target)
    )

    assert torch.allclose(actual.double(), expected.double(), atol=1e-5, rtol=1e-4)


def test_zero_error_gives_exactly_zero_for_every_variant() -> None:
    weights = _equiangular_point_major_weights()
    x = torch.randn(BATCH, NLAT * NLON, CHANNELS, dtype=torch.float64)

    for cls in (SquaredL2LossS2, L1LossS2, L2LossS2):
        assert cls(weights)(x, x).item() == pytest.approx(0.0, abs=1e-12)


# ==========================================================================
# Class hierarchy: L2 is SquaredL2 plus a square root
# ==========================================================================


def test_l2_is_the_square_root_of_squared_l2() -> None:
    """L2LossS2 takes the square root *per batch-and-channel element*,
    before the final mean over every remaining dimension -- so this
    identity only holds globally with a single batch element and a single
    channel (else sqrt and mean don't commute: sqrt(mean(x)) != mean(sqrt(x))
    in general, e.g. for two different per-element values). With exactly one
    element there is nothing to average over first, so the two really are
    the same number."""
    torch.manual_seed(1)
    weights = _equiangular_point_major_weights()
    prediction = torch.randn(1, NLAT * NLON, 1, dtype=torch.float64)
    target = torch.randn(1, NLAT * NLON, 1, dtype=torch.float64)

    squared = SquaredL2LossS2(weights)(prediction, target)
    rooted = L2LossS2(weights)(prediction, target)

    assert torch.allclose(rooted, torch.sqrt(squared))


def test_l2_loss_s2_subclasses_squared_l2_loss_s2() -> None:
    assert issubclass(L2LossS2, SquaredL2LossS2)


# ==========================================================================
# Works on a genuinely non-grid point set (the actual point of this module)
# ==========================================================================


def test_works_on_an_icosphere_shaped_point_set() -> None:
    from gm_attention.discretization.grids.icosphere import build_icosphere_grid

    grid = build_icosphere_grid(subdivisions=1, dtype=torch.float64)  # 42 nodes
    loss = SquaredL2LossS2(grid.quadrature_weights)

    prediction = torch.randn(grid.n_nodes, 3, dtype=torch.float64)
    target = torch.randn(grid.n_nodes, 3, dtype=torch.float64)

    value = loss(prediction, target)
    assert value.item() > 0.0
    assert torch.isfinite(value)


# ==========================================================================
# Autograd
# ==========================================================================


@pytest.mark.parametrize("cls", [SquaredL2LossS2, L1LossS2, L2LossS2])
def test_gradients_flow_to_the_prediction(cls: type[SphericalLossBase]) -> None:
    weights = _equiangular_point_major_weights()
    loss = cls(weights)

    prediction = torch.randn(
        BATCH, NLAT * NLON, CHANNELS, dtype=torch.float64, requires_grad=True
    )
    target = torch.randn(BATCH, NLAT * NLON, CHANNELS, dtype=torch.float64)

    loss(prediction, target).backward()

    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


# ==========================================================================
# Componentwise losses are gauge-dependent
# ==========================================================================


def _rotate_velocity(field: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Rotate the (u, w) coordinates of a (..., n_points, 3) field nodewise."""
    cos, sin = torch.cos(angles), torch.sin(angles)
    u, w = field[..., 1], field[..., 2]
    return torch.stack([field[..., 0], cos * u - sin * w, sin * u + cos * w], dim=-1)


def test_componentwise_l1_is_not_rotation_invariant() -> None:
    """Documents exactly why paper-style metrics need a common gauge: every
    channel is scored as an independent scalar, so a nodewise rotation of
    the velocity coordinates changes the value."""
    torch.manual_seed(1)
    weights = torch.rand(16, dtype=torch.float64) + 0.1
    prediction = torch.randn(16, 3, dtype=torch.float64)
    target = torch.zeros(16, 3, dtype=torch.float64)
    angles = torch.full((16,), math.pi / 4, dtype=torch.float64)

    loss = L1LossS2(weights)
    assert not torch.allclose(
        loss(prediction, target),
        loss(_rotate_velocity(prediction, angles), _rotate_velocity(target, angles)),
    )
