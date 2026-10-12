"""Field-aware normalization and dropout for real ``SO(2)`` feature fields.

Two self-maps ``field_type -> field_type`` that commute with the ``SO(2)``
gauge action: :class:`SO2LayerNorm`, a merged equivariant layer
normalization with one pooled denominator shared by every frequency, and
:class:`SO2Dropout`, an irrep-wise dropout that masks both real coordinates
of an irrep copy together. Both sit inside a pre-norm residual block,
``x' = x + Drop(Attention(Norm(x)))``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from gm_attention.representations.field_type import (
    SO2FeatureField,
    SO2FieldType,
)


class SO2LayerNorm(nn.Module):
    """Merged equivariant layer normalization [Equiformer], ``field_type -> field_type``.

    At a node write the feature one block per present frequency,
    ``x^{(0)} in R^{a_0}`` and ``x^{(n)} in R^{a_n x 2}`` for ``n > 0``.
    Every frequency shares one scalar denominator pooled from the
    per-frequency mean-square magnitudes, and only frequency ``0`` is
    centered:

        mu^{(0)}        = mean_a x_a^{(0)}
        (sigma^{(0)})^2 = mean_a (x_a^{(0)} - mu^{(0)})^2
        (sigma^{(n)})^2 = mean_a (1/2) |x_a^{(n)}|^2,             n > 0
        sigma_rho(x)^2  = mean over present frequencies of (sigma^{(n)})^2
        d_rho(x)        = sqrt(sigma_rho(x)^2 + eps)

        Norm(x)_a^{(0)} = gamma_a^{(0)} (x_a^{(0)} - mu^{(0)}) / d_rho(x) + beta_a^{(0)}
        Norm(x)_a^{(n)} = gamma_a^{(n)} x_a^{(n)} / d_rho(x),      n > 0

    with ``gamma^{(n)} in R^{a_n}`` for every present frequency and
    ``beta^{(0)} in R^{a_0}``; a bias at ``n > 0`` would have to be a fixed
    point of ``rho_n``, and only ``rho_0`` has a nonzero one. Each ``rho_n``
    is orthogonal, so every ``|x_a^{(n)}|`` and hence ``d_rho(x)`` is gauge
    invariant, and ``x^{(0)}`` is invariant outright; dividing the
    equivariant ``x^{(n)}`` by an invariant scalar commutes with the group
    action, ``Norm(rho(r_theta) x) = rho(r_theta) Norm(x)``. The
    per-frequency statistics are pooled with equal weight per frequency, so
    multiplicities may differ between frequencies.

    Parameters
    ----------
    field_type
        The shared input and output field type.
    eps
        Added to the pooled mean-square magnitude before the square root, as
        in ``torch.nn.LayerNorm``.
    bias
        Whether to add a learned bias to the frequency-``0`` channels, if
        present.
    dtype, device
        Floating-point dtype and device of the parameters.
    """

    def __init__(
        self,
        field_type: SO2FieldType,
        *,
        eps: float = 1e-5,
        bias: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        if not isinstance(field_type, SO2FieldType):
            raise TypeError(
                f"field_type must be an SO2FieldType, got {type(field_type).__name__}."
            )

        if not (eps > 0.0):
            raise ValueError(f"eps must be positive, got {eps}.")

        self.field_type = field_type
        self.eps = float(eps)

        # One gain per copy, sized by the frequency's own multiplicity.
        self.gamma = nn.ParameterDict(
            {
                str(frequency): nn.Parameter(
                    torch.empty(multiplicity, dtype=dtype, device=device)
                )
                for frequency, multiplicity in field_type.items()
            }
        )

        self.has_bias = bias and 0 in field_type

        if self.has_bias:
            self.beta = nn.Parameter(
                torch.empty(
                    field_type.multiplicity(0), dtype=dtype, device=device
                )
            )
        else:
            self.register_parameter("beta", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize every ``gamma`` to one and ``beta`` (if present) to zero."""
        for parameter in self.gamma.values():
            nn.init.ones_(parameter)

        if self.has_bias:
            nn.init.zeros_(self.beta)

    def forward(self, x: SO2FeatureField) -> SO2FeatureField:
        """Apply the normalization.

        Parameters
        ----------
        x
            Feature field of type ``field_type``, any batch shape.

        Returns
        -------
        SO2FeatureField
            Feature field of the same type ``field_type``.
        """
        if x.field_type != self.field_type:
            raise ValueError(
                f"Expected a field of type {self.field_type!r}, got {x.field_type!r}."
            )

        # v_0 = mean_a (x^{(0)}_a - mu_0)^2;  v_n = (1/2) mean_a ||x^{(n)}_a||^2, n >= 1.
        # Only frequency 0 is centered: a mean shift does not commute with rho_n.
        centered0: Tensor | None = None
        statistics: list[Tensor] = []

        if 0 in self.field_type:
            x0 = x[0]  # (*batch, a_0)
            mean0 = x0.mean(dim=-1, keepdim=True)  # mu_0, (*batch, 1)
            centered0 = x0 - mean0  # (*batch, a_0)
            statistics.append(centered0.pow(2).mean(dim=-1))  # v_0, (*batch,)

        for frequency in self.field_type.frequencies:
            if frequency == 0:
                continue

            xn = x[frequency]  # (*batch, a_n, 2)
            # ||x^{(n)}_a||^2 is gauge invariant: rho_n is orthogonal.
            squared_norm = xn.pow(2).sum(dim=-1)  # (*batch, a_n)
            statistics.append(0.5 * squared_norm.mean(dim=-1))  # v_n, (*batch,)

        # The merge: one denominator d = sqrt(mean_n v_n + eps) for every
        # frequency, pooled with equal weight per frequency.
        merged_variance = torch.stack(statistics, dim=-1).mean(dim=-1)  # (*batch,)
        denominator = torch.sqrt(merged_variance + self.eps)  # (*batch,)

        outputs: dict[int, Tensor] = {}

        if centered0 is not None:
            normalized0 = centered0 / denominator.unsqueeze(-1)  # (*batch, C)
            outputs[0] = self.gamma["0"] * normalized0

            if self.has_bias:
                outputs[0] = outputs[0] + self.beta

        for frequency in self.field_type.frequencies:
            if frequency == 0:
                continue

            xn = x[frequency]
            normalized = xn / denominator.unsqueeze(-1).unsqueeze(-1)  # (*batch, C, 2)
            outputs[frequency] = self.gamma[str(frequency)].unsqueeze(-1) * normalized

        return SO2FeatureField(self.field_type, outputs)

    def extra_repr(self) -> str:
        return f"field_type={self.field_type!r}, eps={self.eps:.3g}, bias={self.has_bias}"


class SO2Dropout(nn.Module):
    """Irrep-wise structured dropout, ``field_type -> field_type``.

    For ``x^{(n)} in R^{a_n x d_n}`` (``d_0 = 1``, ``d_n = 2`` for
    ``n > 0``), one Bernoulli mask per multiplicity copy, drawn
    independently over copies and batch positions, scales both coordinates
    of the copy:

        m_a^{(n)} ~ Bernoulli(1 - p),   Drop(x)_{a,:}^{(n)} = m_a^{(n)} / (1 - p) * x_{a,:}^{(n)}.

    At ``n = 0`` this is ordinary inverted dropout. At ``n > 0`` the scale
    ``(m_a^{(n)} / (1 - p)) I_2`` is a scalar matrix and commutes with every
    ``rho_n(r_theta)``, so the map is gauge equivariant; dropping the two
    coordinates of a copy independently would not commute. Masks drawn per
    batch position make node-permutation equivariance hold in distribution,
    not pathwise.

    Parameters
    ----------
    field_type
        The shared input and output field type.
    p
        Drop probability, shared across every frequency; ``0 <= p < 1``
        (``p = 1`` would divide by zero in the scale ``1 / (1 - p)``).

    Notes
    -----
    Stochastic only in training mode; in eval mode, or with ``p == 0``, the
    exact identity.
    """

    def __init__(
        self,
        field_type: SO2FieldType,
        p: float = 0.5,
    ) -> None:
        super().__init__()

        if not isinstance(field_type, SO2FieldType):
            raise TypeError(
                f"field_type must be an SO2FieldType, got {type(field_type).__name__}."
            )

        if not (0.0 <= p < 1.0):
            raise ValueError(f"p must satisfy 0 <= p < 1, got {p}.")

        self.field_type = field_type
        self.p = float(p)

    def forward(self, x: SO2FeatureField) -> SO2FeatureField:
        """Apply irrep-wise dropout.

        Parameters
        ----------
        x
            Feature field of type ``field_type``, any batch shape.

        Returns
        -------
        SO2FeatureField
            Feature field of the same type ``field_type``. Returned
            unchanged (the same tensors) outside training mode, or if
            ``p == 0``.
        """
        if x.field_type != self.field_type:
            raise ValueError(
                f"Expected a field of type {self.field_type!r}, got {x.field_type!r}."
            )

        if not self.training or self.p == 0.0:
            return x

        keep_probability = 1.0 - self.p
        outputs: dict[int, Tensor] = {}

        for frequency in self.field_type.frequencies:
            xn = x[frequency]  # (*batch, C) or (*batch, C, 2)
            mask_shape = xn.shape if frequency == 0 else xn.shape[:-1]  # (*batch, C)

            # One draw per copy, shared across the d_n = 2 coordinates of a
            # non-scalar copy, so the pair drops as a unit.
            mask = torch.bernoulli(
                torch.full(
                    mask_shape, keep_probability, dtype=xn.dtype, device=xn.device
                )
            )
            scale = mask / keep_probability

            outputs[frequency] = (
                xn * scale if frequency == 0 else xn * scale.unsqueeze(-1)
            )

        return SO2FeatureField(self.field_type, outputs)

    def extra_repr(self) -> str:
        return f"field_type={self.field_type!r}, p={self.p:.3g}"
