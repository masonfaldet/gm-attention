"""Depthwise (per-copy) steerable value transform.

``SO2DepthwiseKernel`` restricts the general ``rho_n -> rho_m`` kernel of
:class:`~gm_attention.nn.kernels.SO2SteerableKernel` to ``m = n`` and to the
diagonal copy pairs ``a -> a``: one steerable kernel per irrep copy, acting
on that copy alone, the equivariant counterpart of a ``groups = channels``
convolution. Per copy ``a`` of frequency ``n``, with ``c`` the cutoff,

    n = 0:   V_a(r)      x_a  = C_a(r) x_a                                 (real)
    n >= 1:  V_a(r, phi) z_a  = A_a(r) z_a + (r/c)^{2n} B_a(r) e^{2 i n phi} conj(z_a)

with ``A_a, B_a`` complex-valued radial functions and ``C_a`` real. The
steerability constraint is linear in the kernel and holds block by block,
so zeroing every off-diagonal block of a steerable kernel leaves a
steerable kernel. The factor ``(r/c)^{2n}`` on the antilinear term is the
``r^{|k|}`` regularity rule of the general kernel (a mode ``e^{ik phi}``
vanishes like ``r^{|k|}`` at the origin) with ``k = 2n``; the linear term
carries ``k = 0`` and no factor. At ``r = 0`` the kernel is ``C_a(0)`` on a
scalar copy and ``A_a(0)`` on every other copy, a per-copy intertwiner.
The parameterization is one real radial function per scalar copy and two
complex ones (four real) per non-scalar copy, each a head over the shared
radial trunk, so the parameter count is linear in the multiplicity.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor, nn

from gm_attention.discretization.edge_geometry import EdgeGeometry, cutoff_tolerance
from gm_attention.nn.kernels import (
    RadialBasisKind,
    SO2SteerableKernel,
    apply_radial_trunk,
    split_k_linear,
    transport_field,
)
from gm_attention.representations import so2
from gm_attention.representations.field_type import SO2FeatureField, SO2FieldType


class SO2DepthwiseKernel(nn.Module):
    """One steerable kernel per irrep copy, ``field_type -> field_type``.

    Parameters
    ----------
    field_type
        The shared input and output type; a depthwise map cannot change it.
    cutoff, n_radial_basis, radial_hidden_features, radial_basis, bias
        As :class:`~gm_attention.nn.kernels.SO2SteerableKernel`; the radial
        embedding and trunk are built identically, and the two kernels
        differ only in which blocks exist.
    dtype, device
        Floating-point dtype and device of the parameters.
    """

    def __init__(
        self,
        field_type: SO2FieldType,
        *,
        cutoff: float,
        n_radial_basis: int = 8,
        radial_hidden_features: Sequence[int] = (16,),
        radial_basis: RadialBasisKind = "gaussian",
        bias: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        if not isinstance(field_type, SO2FieldType):
            raise TypeError(
                f"field_type must be an SO2FieldType, got {type(field_type).__name__}."
            )

        self.field_type = field_type
        # The attribute names the kernel consumers read; same type on both sides.
        self.field_type_in = field_type
        self.field_type_out = field_type

        # The radial embedding and trunk of a scalar 1 -> 1 general kernel.
        template = SO2SteerableKernel(
            SO2FieldType({0: 1}),
            SO2FieldType({0: 1}),
            cutoff=cutoff,
            n_radial_basis=n_radial_basis,
            radial_hidden_features=radial_hidden_features,
            radial_basis=radial_basis,
            bias=bias,
            dtype=dtype,
            device=device,
        )
        self.radial_basis = template.radial_basis
        self.radial_trunk = template.radial_trunk
        trunk_width = [n_radial_basis, *radial_hidden_features][-1]

        scalar_width = field_type.multiplicity(0) if 0 in field_type else 0
        # Per non-scalar copy: A = (A_re, A_im) and B = (B_re, B_im).
        self._complex_widths = [
            4 * multiplicity
            for frequency, multiplicity in field_type.items()
            if frequency != 0
        ]
        self._complex_frequencies = [f for f in field_type.frequencies if f != 0]
        complex_width = sum(self._complex_widths)

        self.real_head = (
            nn.Linear(trunk_width, scalar_width, bias=bias, dtype=dtype, device=device)
            if scalar_width > 0
            else None
        )
        self.complex_head = (
            nn.Linear(trunk_width, complex_width, bias=bias, dtype=dtype, device=device)
            if complex_width > 0
            else None
        )
        self._validated_edge_geometry_ids: set[int] = set()

    @property
    def cutoff(self) -> float:
        return float(self.radial_basis.cutoff)

    def validate_edges(self, edges: EdgeGeometry) -> None:
        """The edge-side checks of :meth:`forward`: harmonics tabulated to
        twice this kernel's top frequency, no edge beyond the cutoff. The
        distance check runs once per edge-geometry object."""
        required = 2 * self.field_type.max_frequency
        if edges.max_frequency < required:
            raise ValueError(
                f"edges tabulates harmonics up to order {edges.max_frequency}, but a "
                f"depthwise kernel at frequency {self.field_type.max_frequency} needs "
                f"order {required}."
            )
        if edges.distances.numel() > 0 and id(edges) not in self._validated_edge_geometry_ids:
            maximum = float(edges.distances.max())
            tolerance = cutoff_tolerance(edges.distances.dtype)
            if maximum > self.cutoff + tolerance:
                raise ValueError(
                    f"edges contains a distance ({maximum:.6g}) beyond this kernel's "
                    f"cutoff ({self.cutoff:.6g})."
                )
            self._validated_edge_geometry_ids.add(id(edges))

    def forward(self, x: SO2FeatureField, edges: EdgeGeometry) -> SO2FeatureField:
        """One message per edge from features already gathered to edge shape.

        ``x.batch_shape`` must equal ``(edges.n_edges,)``; the gather is the
        caller's, as for the general kernel. Transport ``rho(g_{i<-j})`` is
        applied before the kernel.
        """
        if x.field_type != self.field_type:
            raise ValueError(
                f"Expected a field of type {self.field_type!r}, got {x.field_type!r}."
            )
        if x.batch_shape != (edges.n_edges,):
            raise ValueError(
                "x.batch_shape must equal (edges.n_edges,) = "
                f"({edges.n_edges},), got {tuple(x.batch_shape)}."
            )
        self.validate_edges(edges)

        hidden = apply_radial_trunk(self.radial_trunk, self.radial_basis(edges.distances))  # (n_edges, width)
        transported = transport_field(x, edges)

        outputs: dict[int, Tensor] = {}

        if self.real_head is not None:
            outputs[0] = split_k_linear(hidden, self.real_head.weight, self.real_head.bias) * transported[0]

        if self.complex_head is not None:
            chunks = torch.split(
                split_k_linear(hidden, self.complex_head.weight, self.complex_head.bias),
                self._complex_widths, dim=1,
            )
            normalized = edges.distances / self.cutoff  # (n_edges,)
            for frequency, chunk in zip(self._complex_frequencies, chunks, strict=True):
                multiplicity = self.field_type.multiplicity(frequency)
                coefficients = chunk.reshape(edges.n_edges, 2, multiplicity, 2)
                linear = so2.as_complex(coefficients[:, 0])  # A, (n_edges, a_n)
                antilinear = so2.as_complex(coefficients[:, 1])  # B, (n_edges, a_n)
                z = so2.as_complex(transported[frequency])  # (n_edges, a_n)
                # (r/c)^{2n} by repeated multiplication: differentiable at r = 0.
                scale = torch.ones_like(normalized)
                for _ in range(2 * frequency):
                    scale = scale * normalized
                phase = edges.angular_harmonic(2 * frequency)  # e^{2 i n phi}
                message = linear * z + (scale * phase).unsqueeze(-1) * antilinear * z.conj()
                outputs[frequency] = so2.as_real(message)

        return SO2FeatureField(self.field_type, outputs)

    def extra_repr(self) -> str:
        return f"field_type={self.field_type!r}, cutoff={self.cutoff:.6g}"


def _stacked(parameters: Sequence[Tensor | None]) -> Tensor | None:
    """``torch.stack`` over kernels, or ``None`` when the layer has no such parameter."""
    if parameters[0] is None:
        return None
    return torch.stack([cast(Tensor, p) for p in parameters])


def stacked_coefficients(
    kernels: Sequence[SO2DepthwiseKernel],
    edges: EdgeGeometry,
) -> tuple[Tensor | None, Tensor | None]:
    """Every kernel's depthwise coefficients on the edges, with a leading
    kernel axis: the real ``(n_kernels, n_edges, a_0)`` and the complex
    ``(n_kernels, n_edges, sum_n 4 a_n)`` readouts of one radial embedding
    through the kernels' trunks, as grouped matmuls over the stacked
    parameters (``None`` for a readout the type does not have).
    """
    first = kernels[0]
    first.validate_edges(edges)
    n_kernels = len(kernels)
    with torch.profiler.record_function("gm::kernel/radial"):
        embedding = first.radial_basis(edges.distances)  # (n_edges, n_basis)
        hidden = embedding.unsqueeze(0).expand(n_kernels, -1, -1)  # (n_kernels, n_edges, n_basis)
        trunks = [
            [module for module in kernel.radial_trunk if isinstance(module, nn.Linear)]
            for kernel in kernels
        ]
        for layer_index in range(len(trunks[0])):
            weight = torch.stack([layers[layer_index].weight for layers in trunks])  # (n_kernels, out, in)
            bias = _stacked([layers[layer_index].bias for layers in trunks])
            hidden = torch.nn.functional.silu(split_k_linear(hidden, weight, bias))  # (n_kernels, n_edges, out)

    def readout(heads: Sequence[nn.Linear | None]) -> Tensor | None:
        if heads[0] is None:
            return None
        weight = torch.stack([cast(nn.Linear, head).weight for head in heads])  # (n_kernels, width, trunk)
        bias = _stacked([cast(nn.Linear, head).bias for head in heads])
        return split_k_linear(hidden, weight, bias)  # (n_kernels, n_edges, width)

    return readout([kernel.real_head for kernel in kernels]), readout([kernel.complex_head for kernel in kernels])
