"""SWE evaluation metrics: "Attention on the Sphere"'s componentwise quantities.

:class:`PaperSphericalRegressionMetrics` reports the paper's sphere-integrated
``L1``, ``L2^2`` and ``L2`` over the channels of a ``(..., n_points, channels)``
tensor. Each channel is treated as an independent scalar, so the quantities
are not gauge invariant: predictions holding tangent-vector components must
be expressed in one common basis before they are compared across models.
"""

from __future__ import annotations

from torch import Tensor, nn

from gm_attention.training.losses import (
    L1LossS2,
    L2LossS2,
    SquaredL2LossS2,
)


class PaperSphericalRegressionMetrics(nn.Module):
    """"Attention on the Sphere"'s componentwise quantities, under ``paper_``-prefixed names.

    For the error ``e_ic`` at point ``i`` in channel ``c``, ``C`` channels
    and normalized quadrature weights ``q_i``:

    - ``paper_l1``: ``(1/C) sum_c sum_i q_i |e_ic|``
    - ``paper_squared_l2``: ``(1/C) sum_c sum_i q_i e_ic^2``
    - ``paper_l2``: ``(1/C) sum_c sqrt(sum_i q_i e_ic^2)``

    Not gauge invariant: each treats the channels as independent scalars,
    so a nodewise rotation of the tangent frames changes ``paper_l1`` and
    ``paper_l2`` when channels are vector components (``paper_squared_l2``
    is unchanged only because a sum of squared components is a squared
    norm).

    Parameters
    ----------
    quadrature_weights
        Passed to each underlying componentwise loss.
    """

    def __init__(self, quadrature_weights: Tensor) -> None:
        super().__init__()
        self.squared_l2 = SquaredL2LossS2(quadrature_weights)
        self.l1 = L1LossS2(quadrature_weights)
        self.l2 = L2LossS2(quadrature_weights)

    def forward(self, prediction: Tensor, target: Tensor) -> dict[str, Tensor]:
        """Compute all three paper-style metrics.

        Parameters
        ----------
        prediction, target
            Shape ``(..., n_points, channels)``, already in one common basis.
        """
        return {
            "paper_squared_l2": self.squared_l2(prediction, target),
            "paper_l1": self.l1(prediction, target),
            "paper_l2": self.l2(prediction, target),
        }
