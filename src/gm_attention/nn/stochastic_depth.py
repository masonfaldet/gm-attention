"""Stochastic depth for feature fields: one Bernoulli draw per residual branch.

:class:`SO2DropPath` [Huang et al. 2016] drops a residual branch's whole
output with probability ``p`` in training and rescales survivors by
``1 / (1 - p)``. The output is multiplied by a single scalar, a scalar
multiple of the identity on every irrep copy, so the map commutes with every
``rho_n(r_theta)``. The draw is made once per sample, not per node: a forward
call carries one sample (one graph) unless :func:`copies` declares its fields
a sample-major disjoint union of ``count`` graphs along the leading batch
axis (the segmentation training step's ``training.panoramas_per_forward``),
in which case each sample's block of nodes receives its own draw.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import nn

from gm_attention.representations.field_type import SO2FeatureField

#: Samples the current forward's fields are a sample-major union of along the leading batch axis; ``1`` is one per call.
_COPIES: int = 1


def current_copies() -> int:
    """The number of samples the current forward's fields are a union of."""
    return _COPIES


@contextmanager
def copies(count: int) -> Iterator[None]:
    """Declare the fields of the enclosed forward a disjoint union of ``count``
    samples along the leading batch axis, so every :class:`SO2DropPath`
    draws once per sample. Nests; the previous value is restored on exit."""
    global _COPIES  # read at call time
    if int(count) != count or count < 1:
        raise ValueError(f"count must be a positive integer, got {count!r}.")
    previous = _COPIES
    _COPIES = int(count)
    try:
        yield
    finally:
        _COPIES = previous


class SO2DropPath(nn.Module):
    """Drop a residual branch's whole output with probability ``p`` in training.

    Parameters
    ----------
    p
        Drop probability, ``0 <= p < 1``. ``0`` (and eval mode) make this
        the exact identity.
    """

    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError(f"p must satisfy 0 <= p < 1, got {p}.")
        self.p = float(p)

    def forward(self, x: SO2FeatureField) -> SO2FeatureField:
        if not self.training or self.p == 0.0:
            return x
        keep_probability = 1.0 - self.p
        count = _COPIES
        if count == 1:
            # One draw for the whole branch; a kept branch is rescaled so the
            # expectation is unchanged.
            keep = torch.rand((), device=x.device, dtype=x.dtype) < keep_probability
            scale = keep.to(x.dtype) / keep_probability
            return SO2FeatureField(
                x.field_type,
                {frequency: tensor * scale for frequency, tensor in x.coefficients.items()},
            )
        # A union of `count` samples: one draw per sample, spread over its
        # block of the leading batch axis (sample-major).
        if not x.batch_shape or x.batch_shape[0] % count != 0:
            raise ValueError(
                f"A union of {count} samples needs a leading batch axis divisible by "
                f"{count}, got batch shape {tuple(x.batch_shape)}."
            )
        nodes = x.batch_shape[0]
        keep = torch.rand((count,), device=x.device, dtype=x.dtype) < keep_probability
        scale = (keep.to(x.dtype) / keep_probability).repeat_interleave(nodes // count)
        return SO2FeatureField(
            x.field_type,
            {
                frequency: tensor * scale.view(nodes, *([1] * (tensor.ndim - 1)))
                for frequency, tensor in x.coefficients.items()
            },
        )

    def extra_repr(self) -> str:
        return f"p={self.p}"
