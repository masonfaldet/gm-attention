"""Gauge-equivariant linear layers.

``SO2Linear`` is a linear map ``W`` between ``SO2FeatureField``s satisfying
the intertwiner constraint ``W rho_in(g) = rho_out(g) W`` for all ``g`` in
``SO(2)``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from gm_attention.representations import so2
from gm_attention.representations.field_type import (
    SO2FeatureField,
    SO2FieldType,
)


class SO2Linear(nn.Module):
    """A gauge-equivariant linear layer between ``SO(2)`` field types.

    Parameters
    ----------
    field_type_in, field_type_out
        Input and output field types.
    bias
        Whether to add a learned bias to the frequency-``0`` output
        channels, if any are present. Ignored (no parameter is created) if
        ``field_type_out`` has no frequency-``0`` channels.
    dtype, device
        Floating-point dtype and device of the parameters. ``dtype``
        defaults to ``torch.get_default_dtype()``.

    Notes
    -----
    Frequencies present in ``field_type_in`` but absent from
    ``field_type_out`` contribute nothing: the intertwiner constraint forces
    their block into every other output frequency to be zero, so no
    parameter is allocated for them. Frequencies present in
    ``field_type_out`` but absent from ``field_type_in`` are output as zero
    (or as the bias alone, at frequency zero).
    """

    def __init__(
        self,
        field_type_in: SO2FieldType,
        field_type_out: SO2FieldType,
        *,
        bias: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        if not isinstance(field_type_in, SO2FieldType):
            raise TypeError(
                "field_type_in must be an SO2FieldType, "
                f"got {type(field_type_in).__name__}."
            )

        if not isinstance(field_type_out, SO2FieldType):
            raise TypeError(
                "field_type_out must be an SO2FieldType, "
                f"got {type(field_type_out).__name__}."
            )

        self.field_type_in = field_type_in
        self.field_type_out = field_type_out
        self.common_frequencies = field_type_in.common_frequencies(field_type_out)

        self.weight = nn.ParameterDict()

        for frequency in self.common_frequencies:
            multiplicity_in = field_type_in.multiplicity(frequency)
            multiplicity_out = field_type_out.multiplicity(frequency)

            # Frequency 0: a real (b_0, a_0) matrix. Frequency n >= 1: the
            # complex (b_n, a_n) matrix A_n = M_n + i N_n as (M_n, N_n) on a
            # trailing axis of size 2, the layout so2.as_complex views.
            shape = (
                (multiplicity_out, multiplicity_in)
                if frequency == 0
                else (multiplicity_out, multiplicity_in, 2)
            )

            self.weight[str(frequency)] = nn.Parameter(
                torch.empty(shape, dtype=dtype, device=device)
            )

        self.has_bias = bias and 0 in field_type_out

        if self.has_bias:
            self.bias = nn.Parameter(
                torch.empty(
                    field_type_out.multiplicity(0), dtype=dtype, device=device
                )
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights and bias uniformly in ``[-1/sqrt(fan_in), 1/sqrt(fan_in)]``.

        Each frequency block is initialized independently with the real
        input dimension ``a_n * d_n`` it contracts over as its fan-in, the
        ``nn.Linear`` rule applied per block.
        """
        for frequency in self.common_frequencies:
            fan_in = self.field_type_in.multiplicity(frequency) * so2.irrep_dimension(
                frequency
            )
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.weight[str(frequency)], -bound, bound)

        if self.has_bias:
            if 0 in self.common_frequencies:
                fan_in = self.field_type_in.multiplicity(0)
            else:
                # No frequency-0 input: the bias is the whole output there.
                fan_in = 1

            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: SO2FeatureField) -> SO2FeatureField:
        """Apply the layer.

        Parameters
        ----------
        x
            Feature field of type ``field_type_in``.

        Returns
        -------
        SO2FeatureField
            Feature field of type ``field_type_out``.
        """
        if x.field_type != self.field_type_in:
            raise ValueError(
                f"Expected a field of type {self.field_type_in!r}, got "
                f"{x.field_type!r}."
            )

        outputs: dict[int, Tensor] = {}

        # An intertwiner is block diagonal in frequency: each rho_n isotypic
        # component maps to the rho_n component of the output only.
        for frequency in self.field_type_out.frequencies:
            multiplicity_out = self.field_type_out.multiplicity(frequency)

            # A frequency the input lacks has an identically zero output block.
            if frequency not in self.common_frequencies:
                shape = (
                    (*x.batch_shape, multiplicity_out)
                    if frequency == 0
                    else (*x.batch_shape, multiplicity_out, 2)
                )
                outputs[frequency] = torch.zeros(
                    shape,
                    dtype=x.dtype,
                    device=x.device,
                )
                continue

            weight = self.weight[str(frequency)]

            if frequency == 0:
                # rho_0 is trivial: y_b = sum_a W_{ba} x_a for any real W.
                outputs[0] = x[0] @ weight.transpose(-1, -2)  # (*batch, b_0)
            else:
                # For n >= 1 the commutant of rho_n on R^2 is C, so the block
                # is a complex (b_n, a_n) matrix: z_b = sum_a W_{ba} z_a.
                weight_complex = so2.as_complex(weight)  # (b_n, a_n) complex
                input_complex = so2.as_complex(x[frequency])  # (*batch, a_n) complex

                output_complex = (  # (*batch, b_n) complex
                    input_complex @ weight_complex.transpose(-1, -2)
                )
                outputs[frequency] = so2.as_real(output_complex)  # (*batch, b_n, 2)

        # A bias is equivariant only on rho_0 (no nonzero fixed vector of rho_n).
        if self.has_bias:
            outputs[0] = outputs[0] + self.bias  # (*batch, b_0)

        return SO2FeatureField(self.field_type_out, outputs)

    def extra_repr(self) -> str:
        return (
            f"field_type_in={self.field_type_in!r}, "
            f"field_type_out={self.field_type_out!r}, "
            f"bias={self.has_bias}"
        )
