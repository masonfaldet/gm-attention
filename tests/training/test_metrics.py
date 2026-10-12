"""Tests for gm_attention.training.metrics.

The shipped metric family is "Attention on the Sphere"'s componentwise one,
and the load-bearing property asserted here is that it is *not* invariant
under a change of tangent frame: ``paper_l1`` and ``paper_l2`` change under a
nodewise regauging, and only ``paper_squared_l2`` happens not to. That is
why these metrics must only ever be fed tensors expressed in one common
basis, and the tests pin both the non-invariance and the agreement once both
sides share a gauge.
"""

from __future__ import annotations

import math

import torch

from gm_attention.training.losses import (
    L1LossS2,
    L2LossS2,
    SquaredL2LossS2,
)
from gm_attention.training.metrics import PaperSphericalRegressionMetrics

PAPER_KEYS = {"paper_squared_l2", "paper_l1", "paper_l2"}


def _rotate_velocity(field: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    cos, sin = torch.cos(angles), torch.sin(angles)
    u, w = field[..., 1], field[..., 2]
    return torch.stack([field[..., 0], cos * u - sin * w, sin * u + cos * w], dim=-1)


def _random_case(
    n_points: int = 20, batch: tuple[int, ...] = (4,)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weights = torch.rand(n_points, dtype=torch.float64) + 0.1  # nonuniform quadrature
    prediction = torch.randn(*batch, n_points, 3, dtype=torch.float64)
    target = torch.randn(*batch, n_points, 3, dtype=torch.float64)
    return weights, prediction, target


# ==========================================================================
# PaperSphericalRegressionMetrics
# ==========================================================================


def test_paper_metrics_match_each_componentwise_loss_computed_separately() -> None:
    torch.manual_seed(6)
    weights, prediction, target = _random_case()

    result = PaperSphericalRegressionMetrics(weights)(prediction, target)

    assert set(result) == PAPER_KEYS
    assert torch.allclose(result["paper_squared_l2"], SquaredL2LossS2(weights)(prediction, target))
    assert torch.allclose(result["paper_l1"], L1LossS2(weights)(prediction, target))
    assert torch.allclose(result["paper_l2"], L2LossS2(weights)(prediction, target))


def test_paper_component_metrics_generally_change_under_regauging() -> None:
    """Why they must only ever be fed common-basis tensors.

    ``paper_squared_l2`` is exempt: a sum of squared components *is* a
    squared norm, so it happens to be invariant. The other two are not, and
    that is exactly the trap this naming and these tests exist to flag.
    """
    torch.manual_seed(7)
    weights, prediction, target = _random_case()
    angles = torch.rand(20, dtype=torch.float64) * 2 * math.pi

    metrics = PaperSphericalRegressionMetrics(weights)
    original = metrics(prediction, target)
    regauged = metrics(_rotate_velocity(prediction, angles), _rotate_velocity(target, angles))

    assert not torch.allclose(original["paper_l1"], regauged["paper_l1"])
    assert not torch.allclose(original["paper_l2"], regauged["paper_l2"])
    # Coincidentally invariant -- documented, not relied on.
    assert torch.allclose(original["paper_squared_l2"], regauged["paper_squared_l2"])


def test_paper_metrics_agree_once_both_sides_share_one_gauge() -> None:
    """Two models working in different internal gauges score identically
    after both are converted into the same common basis -- the whole reason
    the conversion is mandatory before these are compared."""
    torch.manual_seed(8)
    weights, prediction, target = _random_case()
    metrics = PaperSphericalRegressionMetrics(weights)

    # One "model" whose native gauge is rotated relative to the other's.
    angles = torch.rand(20, dtype=torch.float64) * 2 * math.pi
    in_model_gauge = _rotate_velocity(prediction, angles)

    # Converting it back into the common basis recovers the same numbers.
    back_to_common = _rotate_velocity(in_model_gauge, -angles)

    original = metrics(prediction, target)
    converted = metrics(back_to_common, target)

    for key in PAPER_KEYS:
        assert torch.allclose(original[key], converted[key]), key


def test_paper_metrics_handle_leading_batch_dimensions_and_nonuniform_quadrature() -> None:
    torch.manual_seed(9)
    weights, prediction, target = _random_case(batch=(2, 3))

    result = PaperSphericalRegressionMetrics(weights)(prediction, target)

    for key in PAPER_KEYS:
        assert result[key].ndim == 0, key
        assert torch.isfinite(result[key]), key
