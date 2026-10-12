"""Gauge-equivariant pointwise nonlinearities for real ``SO(2)`` fields.

:class:`SO2ScalarGatedNonlinearity` is a pointwise self-map ``field_type ->
field_type``: an ordinary ``SiLU`` on the scalar (frequency-``0``) channels
and, for every nonzero frequency, one gate per multiplicity copy computed
from the scalar channels and applied to both real coordinates of the copy.
Multiplying an irrep copy by a gauge-invariant scalar commutes with the
orthogonal ``rho_n(r_theta)``, so the map is gauge equivariant, and being
pointwise it preserves the ``Isom_GM`` equivariance of a stack.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from gm_attention.representations.field_type import (
    SO2FeatureField,
    SO2FieldType,
)

__all__ = [
    "SO2ScalarGatedNonlinearity",
]


def _validate_init(field_type: SO2FieldType, eps: float) -> None:
    """Validate the constructor arguments."""
    if not isinstance(field_type, SO2FieldType):
        raise TypeError(
            f"field_type must be an SO2FieldType, got {type(field_type).__name__}."
        )

    if not (eps > 0.0):
        raise ValueError(f"eps must be positive, got {eps}.")


def _validate_input(x: SO2FeatureField, field_type: SO2FieldType) -> None:
    """Check that an input feature field has the expected field type."""
    if x.field_type != field_type:
        raise ValueError(
            f"Expected a field of type {field_type!r}, got {x.field_type!r}."
        )


class SO2ScalarGatedNonlinearity(nn.Module):
    """Scalar-conditioned gates, ``field_type -> field_type``.

    Scalar (frequency-``0``) channels pass through ``SiLU``. Every nonzero
    frequency ``n`` is gated by a learned affine map of the scalar channels:

        g_n(x_0) = gate(A_n x_0 + b_n)          (..., a_n)
        y_{n,a}  = g_{n,a}(x_0) * x_{n,a}       one gate per multiplicity copy

    ``rho_0`` is trivial, so ``x_0`` and every function of it, including
    ``g_n``, are gauge invariant; multiplying a real irrep copy (both
    coordinates) by one invariant scalar commutes with ``rho_n(g)``, so the
    map is equivariant. Gating the two coordinates of a copy differently
    would be a non-uniform scaling of ``R^2`` and would not commute with
    rotation. Requires frequency-``0`` channels in ``field_type``.

    Parameters
    ----------
    field_type
        The shared input and output field type. Must contain frequency 0.
    gate
        ``"silu"`` (default): gates in ``(-0.28, inf)``, able to suppress,
        pass, amplify and mildly invert. ``"double_sigmoid"``:
        ``2 * sigmoid(...)``, strictly positive in ``(0, 2)``.
    eps
        Accepted for signature compatibility and unused: no norm is
        computed.
    dtype, device
        Floating-point dtype and device of the gate parameters.
    """

    def __init__(
        self,
        field_type: SO2FieldType,
        *,
        gate: Literal["silu", "double_sigmoid"] = "silu",
        eps: float = 1e-6,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        _validate_init(field_type, eps)

        if gate not in ("silu", "double_sigmoid"):
            raise ValueError(
                f"gate must be 'silu' or 'double_sigmoid', got {gate!r}."
            )

        if 0 not in field_type:
            raise ValueError(
                "SO2ScalarGatedNonlinearity requires frequency-0 channels to "
                f"condition its gates on, but got {field_type!r}."
            )

        self.field_type = field_type
        self.gate = gate
        self.eps = float(eps)

        scalar_multiplicity = field_type.multiplicity(0)

        # One nn.Linear per nonzero frequency, scalar channels -> one gate per
        # copy of that frequency; ModuleDict keys are strings.
        self.gates = nn.ModuleDict(
            {
                str(frequency): nn.Linear(
                    scalar_multiplicity,
                    field_type.multiplicity(frequency),
                    dtype=dtype,
                    device=device,
                )
                for frequency in field_type.frequencies
                if frequency != 0
            }
        )

    def _activate(self, pre_activation: Tensor) -> Tensor:
        if self.gate == "double_sigmoid":
            return 2.0 * torch.sigmoid(pre_activation)
        return F.silu(pre_activation)

    def forward(self, x: SO2FeatureField) -> SO2FeatureField:
        """Apply scalar-conditioned gating pointwise to a feature field.

        Parameters
        ----------
        x
            Feature field of type ``field_type``, any batch shape.

        Returns
        -------
        SO2FeatureField
            Feature field of the same type ``field_type``.
        """
        _validate_input(x, self.field_type)

        # y^{(0)} = SiLU(x^{(0)});  for n >= 1,
        #     g_n = act(A_n x^{(0)} + b_n),   y^{(n)}_a = g_{n,a} x^{(n)}_a.
        scalars = x[0]  # (*batch, a_0), gauge invariant
        outputs: dict[int, Tensor] = {0: F.silu(scalars)}  # (*batch, a_0)

        for frequency in self.field_type.frequencies:
            if frequency == 0:
                continue

            # Gates are computed from the raw scalars, not the activated ones.
            gate = self._activate(
                self.gates[str(frequency)](scalars)
            )  # g_n, (*batch, a_n)

            # unsqueeze(-1): one gate per copy, shared across both real coordinates.
            outputs[frequency] = x[frequency] * gate.unsqueeze(
                -1
            )  # (*batch, a_n, 2)

        return SO2FeatureField(self.field_type, outputs)

    def extra_repr(self) -> str:
        return f"field_type={self.field_type!r}, gate={self.gate!r}"
