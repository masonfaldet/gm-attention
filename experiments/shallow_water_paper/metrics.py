"""Losses and per-channel RMSE over the three-scalar shallow-water state.

The state holds three scalars and no vector, so the sphere-integrated
``L1``/``L2`` of Bonev et al. (Eq. 13-14, :mod:`gm_attention.training.losses`)
averaged over the three channels apply componentwise. ``paper_squared_l2``
is the training objective and the monitored validation metric.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from experiments.shallow_water_paper.data import CHANNELS
from gm_attention.training.metrics import PaperSphericalRegressionMetrics

#: The training objective and the monitored validation metric.
MONITOR_METRIC = "paper_squared_l2"


class ScalarStateMetrics(nn.Module):
    """``paper_squared_l2`` / ``paper_l1`` / ``paper_l2`` over the three
    channels, and ``<channel>_rmse`` for each of :data:`CHANNELS`."""

    weights: Tensor

    def __init__(self, quadrature_weights: Tensor) -> None:
        super().__init__()
        self.paper = PaperSphericalRegressionMetrics(quadrature_weights)
        weights = quadrature_weights.reshape(-1).to(torch.float64)
        self.register_buffer("weights", (weights / weights.sum()).to(quadrature_weights.dtype))

    def forward(self, prediction: Tensor, target: Tensor) -> dict[str, Tensor]:
        """``prediction``, ``target``: ``(..., n_points, 3)`` in :data:`CHANNELS` order."""
        out = dict(self.paper(prediction, target))
        squared = ((prediction - target).square() * self.weights.unsqueeze(-1)).sum(dim=-2)
        # Leading batch dimensions average before the root: one RMSE over the
        # sphere and the batch, for batched and single-sample steps alike.
        squared = squared.reshape(-1, len(CHANNELS)).mean(dim=0)
        for channel, name in enumerate(CHANNELS):
            out[f"{name}_rmse"] = squared[channel].sqrt()
        return out
