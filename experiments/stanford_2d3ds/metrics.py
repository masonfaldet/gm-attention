"""The released segmentation metrics, transcribed: quadrature-weighted IoU and accuracy.

``torch_harmonics.examples.metrics`` (v0.8.0), the module the released
segmentation script scores with, does not import: the docstring of its
``_predict_classes`` is mis-indented and raises an ``IndentationError``.
The four definitions below are that module's.
    torch-harmonics -- https://github.com/NVIDIA/torch-harmonics
    Copyright (c) 2025 The torch-harmonics Authors. BSD-3-Clause license.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_harmonics.examples.losses import get_quadrature_weights


def _get_stats_multiclass(
    output: torch.LongTensor,
    target: torch.LongTensor,
    num_classes: int,
    quad_weights: torch.Tensor,
    ignore_index: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-class (tp, fp, fn, tn) areas on the sphere, per batch item."""
    batch_size, *_dims = output.shape

    if ignore_index is not None:
        ignore = target == ignore_index
        output = torch.where(ignore, -1, output)  # type: ignore[assignment]
        target = torch.where(ignore, -1, target)  # type: ignore[assignment]

    tp_count = torch.zeros(batch_size, num_classes, dtype=torch.float32, device=output.device)
    fp_count = torch.zeros(batch_size, num_classes, dtype=torch.float32, device=output.device)
    fn_count = torch.zeros(batch_size, num_classes, dtype=torch.float32, device=output.device)

    matched = target == output
    not_matched = target != output
    for i in range(batch_size):
        matched_i = matched[i, ...]
        not_matched_i = not_matched[i, ...]
        target_i = target[i, ...]
        output_i = output[i, ...]
        for c in range(num_classes):
            qwt_c = quad_weights[target_i == c]
            qwo_c = quad_weights[output_i == c]
            tp_count[i, c] = torch.sum(matched_i[target_i == c] * qwt_c)
            fp_count[i, c] = torch.sum(not_matched_i[output_i == c] * qwo_c)
            fn_count[i, c] = torch.sum(not_matched_i[target_i == c] * qwt_c)

    # An ignored pixel is neither a target nor a prediction of any class, so
    # its area counts as a true negative of every class (released behaviour).
    tn_count = torch.sum(quad_weights) - tp_count - fp_count - fn_count
    return tp_count, fp_count, fn_count, tn_count


def _predict_classes(logits: torch.Tensor) -> torch.Tensor:
    """Class predictions: argmax of the softmax over the class axis."""
    return torch.argmax(torch.softmax(logits, dim=1), dim=1, keepdim=False)


class BaseMetricS2(nn.Module):
    """Quadrature-weighted multiclass statistics on an ``(nlat, nlon)`` grid."""

    quad_weights: torch.Tensor

    def __init__(self, nlat: int, nlon: int, grid: str = "equiangular", weight: torch.Tensor | None = None, ignore_index: int = -100, mode: str = "micro"):
        super().__init__()

        self.ignore_index = ignore_index
        self.mode = mode

        q = get_quadrature_weights(nlat=nlat, nlon=nlon, grid=grid, tile=True)
        self.register_buffer("quad_weights", q)

        if weight is None:
            self.weight = None
        else:
            self.register_buffer("weight", weight.unsqueeze(0))

    def _forward(self, pred: torch.Tensor, truth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_class = _predict_classes(pred)

        tp, fp, fn, tn = _get_stats_multiclass(pred_class, truth, pred.shape[1], self.quad_weights, self.ignore_index)  # type: ignore[arg-type]

        if self.mode == "micro":
            if self.weight is not None:
                tp = torch.sum(tp * self.weight)
                fp = torch.sum(fp * self.weight)
                fn = torch.sum(fn * self.weight)
                tn = torch.sum(tn * self.weight)
            else:
                tp = torch.mean(tp)
                fp = torch.mean(fp)
                fn = torch.mean(fn)
                tn = torch.mean(tn)
        else:
            tp = torch.mean(tp, dim=0)
            fp = torch.mean(fp, dim=0)
            fn = torch.mean(fn, dim=0)
            tn = torch.mean(tn, dim=0)

        return tp, fp, fn, tn


class IntersectionOverUnionS2(BaseMetricS2):
    """Intersection over union on the sphere, Eq. (20) of "Attention on the Sphere"."""

    def forward(self, pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        tp, fp, fn, _tn = self._forward(pred, truth)

        score = tp / (tp + fp + fn)

        if self.mode == "macro":
            # A class absent from both target and prediction scores nan; counted as 0.
            score = torch.where(torch.isnan(score), 0.0, score)

            if self.weight is not None:
                score = torch.sum(score * self.weight)
            else:
                score = torch.mean(score)

        return score


class AccuracyS2(BaseMetricS2):
    """Accuracy on the sphere, Eq. (23) of "Attention on the Sphere"."""

    def forward(self, pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        tp, fp, fn, tn = self._forward(pred, truth)

        score = (tp + tn) / (tp + fp + fn + tn)

        if self.mode == "macro":
            # A class absent from both target and prediction scores nan; counted as 0.
            score = torch.where(torch.isnan(score), 0.0, score)

            if self.weight is not None:
                score = torch.sum(score * self.weight)
            else:
                score = torch.mean(score)

        return score
