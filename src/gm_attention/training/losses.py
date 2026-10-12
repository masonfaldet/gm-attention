"""Quadrature-weighted spherical loss functions on any point set.

``SquaredL2LossS2``, ``L1LossS2`` and ``L2LossS2`` are the ``L1LossS2``/
``L2LossS2`` family of ``torch_harmonics.examples.losses`` generalized from
a fixed equiangular ``(nlat, nlon)`` grid to any 1-D quadrature-weight
tensor, in the point-major layout ``(..., n_points, channels)``. With an
equiangular grid's flattened weights they reduce to the reference
computation; with an icosphere's quadrature weights they are the same
metric on that point set. For real signals ``u, u*`` and a quadrature rule
``{(x_i, q_i)}`` (Bonev et al., arXiv:2505.11157, Eq. 13-14),

    L1[u, u*]   = (1 / 4*pi) * sum_i |u(x_i) - u*(x_i)| * q_i
    L2^2[u, u*] = (1 / 4*pi) * sum_i (u(x_i) - u*(x_i))^2 * q_i

so the constructor normalizes the given weights by their own sum, and
weights summing to ``4*pi`` or to ``1`` give the same loss. Every channel
is treated as an independent scalar and the channels are averaged
uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


def _normalized_quadrature_weights(quadrature_weights: Tensor) -> Tensor:
    """Validate 1-D positive weights and rescale them to sum to ``1``."""
    if not isinstance(quadrature_weights, Tensor):
        raise TypeError(
            "quadrature_weights must be a torch.Tensor, got "
            f"{type(quadrature_weights).__name__}."
        )

    if quadrature_weights.ndim != 1:
        raise ValueError(
            "quadrature_weights must have shape (n_points,), got "
            f"{tuple(quadrature_weights.shape)}."
        )

    if not bool((quadrature_weights > 0).all()):
        raise ValueError("quadrature_weights must be strictly positive.")

    return quadrature_weights / quadrature_weights.sum()


class SphericalLossBase(nn.Module, ABC):
    """Integrates a pointwise loss term over the sphere, then averages.

    Parameters
    ----------
    quadrature_weights
        Positive weights, shape ``(n_points,)``, one per point of whatever
        discretization ``prediction``/``target`` live on. Normalized by
        their own sum at construction time, so any positive quadrature
        convention is accepted.
    """

    quadrature_weights: Tensor

    def __init__(self, quadrature_weights: Tensor) -> None:
        super().__init__()
        self.register_buffer(
            "quadrature_weights", _normalized_quadrature_weights(quadrature_weights)
        )

    def _integrate(self, term: Tensor) -> Tensor:
        # sum_i q_i t[i, c] with q summing to one: the quadrature approximation of
        # (1 / 4pi) integral_{S^2} t_c dA; (..., n_points, channels) -> (..., channels).
        return (term * self.quadrature_weights.unsqueeze(-1)).sum(dim=-2)

    @abstractmethod
    def _loss_term(self, prediction: Tensor, target: Tensor) -> Tensor:
        """The pointwise (pre-integration) loss term."""

    def _post_integration(self, integrated: Tensor) -> Tensor:
        """Hook applied after integration, before the final batch/channel mean."""
        return integrated

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Compute the loss.

        Parameters
        ----------
        prediction, target
            Shape ``(..., n_points, channels)``, matching
            ``quadrature_weights``'s ``n_points``.

        Returns
        -------
        Tensor
            Scalar: the sphere-integrated loss, averaged over every
            remaining (batch and channel) dimension.
        """
        term = self._loss_term(prediction, target)
        integrated = self._integrate(term)
        integrated = self._post_integration(integrated)
        return integrated.mean()


class SquaredL2LossS2(SphericalLossBase):
    """Sphere-integrated mean squared error, averaged over channels uniformly."""

    def _loss_term(self, prediction: Tensor, target: Tensor) -> Tensor:
        return (prediction - target).square()


class L1LossS2(SphericalLossBase):
    """Sphere-integrated mean absolute error."""

    def _loss_term(self, prediction: Tensor, target: Tensor) -> Tensor:
        return (prediction - target).abs()


class L2LossS2(SquaredL2LossS2):
    """Sphere-integrated root mean squared error (:class:`SquaredL2LossS2`, rooted)."""

    def _post_integration(self, integrated: Tensor) -> Tensor:
        return torch.sqrt(integrated)
