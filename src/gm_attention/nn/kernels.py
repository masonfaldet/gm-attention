"""Gauge-equivariant steerable kernels: the value transformation template.

``SO2SteerableKernel`` is a position-dependent linear map ``W_V(v)`` between ``SO2FeatureField``s,
evaluated per edge, satisfying

    W_V(gv) = rho_out(g) W_V(v) rho_in(g)^{-1}  for all g in SO(2).
"""

from __future__ import annotations

import itertools
import math
import os
from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor, nn

from gm_attention.discretization.edge_geometry import EdgeGeometry, cutoff_tolerance
from gm_attention.representations import so2

#: Parameterization of the radial coefficient functions; ``"gaussian"`` is a Gaussian basis followed by an MLP.
RadialBasisKind = Literal["gaussian"]

from gm_attention.representations.field_type import (
    SO2FeatureField,
    SO2FieldType,
)

#: Rows per block of the blocked weight gradient (see :class:`_SplitKLinear`).
SPLIT_K_ROWS = 16384
#: ``GM_SPLIT_K_BACKWARD=0`` routes :func:`split_k_linear` through the plain autograd linear map instead of the blocked weight gradient.
SPLIT_K_BACKWARD = os.environ.get("GM_SPLIT_K_BACKWARD", "1") != "0"

#: ``GM_APPLY_BACKWARD=0`` gives the block contraction :func:`_apply` autograd's backward instead of :class:`_BlockApply`'s (same gradients to summation order).
APPLY_BACKWARD = os.environ.get("GM_APPLY_BACKWARD", "1") != "0"

#: ``GM_FACTORED_KERNEL=1`` evaluates the layers' kernels through :meth:`SO2SteerableKernel.forward_factored`; the layers consult :func:`factored_enabled` at call time.
FACTORED = os.environ.get("GM_FACTORED_KERNEL", "0") != "0"


def factored_enabled() -> bool:
    """Whether :class:`SO2Attention` and :class:`SO2Convolution` evaluate
    their steerable kernels through :meth:`SO2SteerableKernel.forward_factored`."""
    return FACTORED


def _blocked_weight_gradient(grad_out: Tensor, x: Tensor, rows: int) -> Tensor:
    """``grad_out^T @ x`` over the row axis, as one batched product per block of
    ``rows`` rows plus a remainder, summed: ``(..., n, w) x (..., n, k) -> (..., w, k)``.
    Row blocks are views of the operands."""
    n = x.shape[-2]
    blocks = n // rows
    if blocks < 2:
        return torch.matmul(grad_out.transpose(-1, -2), x)
    lead = x.shape[:-2]
    full = blocks * rows
    go = grad_out[..., :full, :].reshape(*lead, blocks, rows, grad_out.shape[-1])
    xb = x[..., :full, :].reshape(*lead, blocks, rows, x.shape[-1])
    grad_w = torch.matmul(go.transpose(-1, -2), xb).sum(dim=-3)  # (..., w, k)
    if full < n:
        grad_w = grad_w + torch.matmul(grad_out[..., full:, :].transpose(-1, -2), x[..., full:, :])
    return grad_w


class _SplitKLinear(torch.autograd.Function):
    """``x @ weight^T + bias`` with a blocked weight gradient.

    Forward is the ordinary linear map (``F.linear`` for a 2-D weight, so
    values are bitwise those of ``nn.Linear``; a matmul plus bias for a
    weight with leading axes). Backward differs from autograd's only in
    the weight gradient's summation order, see :func:`_blocked_weight_gradient`.
    """

    @staticmethod
    def forward(ctx: object, x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:  # type: ignore[override]
        ctx.save_for_backward(x, weight)  # type: ignore[attr-defined]
        ctx.has_bias = bias is not None  # type: ignore[attr-defined]
        if weight.ndim == 2:
            return torch.nn.functional.linear(x, weight, bias)
        out = torch.matmul(x, weight.transpose(-1, -2))
        return out if bias is None else out + bias.unsqueeze(-2)

    @staticmethod
    def backward(ctx: object, grad_out: Tensor) -> tuple[Tensor | None, Tensor | None, Tensor | None]:  # type: ignore[override]
        x, weight = ctx.saved_tensors  # type: ignore[attr-defined]
        needs_x, needs_w, needs_b = ctx.needs_input_grad  # type: ignore[attr-defined]
        grad_x = torch.matmul(grad_out, weight) if needs_x else None
        grad_w = _blocked_weight_gradient(grad_out, x, SPLIT_K_ROWS) if needs_w else None
        grad_b = grad_out.sum(dim=-2) if (needs_b and ctx.has_bias) else None  # type: ignore[attr-defined]
        return grad_x, grad_w, grad_b


def split_k_linear(x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
    """``x @ weight^T + bias`` on ``(..., n, k)`` with the blocked weight gradient
    of :class:`_SplitKLinear` when :data:`SPLIT_K_BACKWARD` is set, the plain
    autograd map otherwise. A weight with leading axes applies per leading
    index (the batched attention's stacked per-head weights)."""
    if SPLIT_K_BACKWARD:
        return _SplitKLinear.apply(x, weight, bias)  # type: ignore[no-any-return]
    if weight.ndim == 2:
        return torch.nn.functional.linear(x, weight, bias)
    out = torch.matmul(x, weight.transpose(-1, -2))
    return out if bias is None else out + bias.unsqueeze(-2)


def apply_radial_trunk(trunk: nn.Sequential, embedding: Tensor) -> Tensor:
    """A radial trunk (``Linear``/``SiLU`` pairs) on ``(..., n_edges, n_basis)``,
    every linear layer through :func:`split_k_linear`."""
    hidden = embedding
    for layer in trunk:
        if isinstance(layer, nn.Linear):
            hidden = split_k_linear(hidden, layer.weight, layer.bias)
        else:
            hidden = layer(hidden)
    return hidden


class _BlockApply(torch.autograd.Function):
    """``(matrices * vectors.unsqueeze(-2)).sum(-1)`` with a backward that
    never conjugates the block.

    For ``y_b = sum_a M_ba v_a`` and an upstream ``g``, PyTorch's complex
    convention gives ``grad_M = g conj(v)^T`` and
    ``grad_v = M^H g = conj(M^T conj(g))``; the latter is evaluated with the
    vector conjugated rather than the block.
    """

    @staticmethod
    def forward(ctx: object, matrices: Tensor, vectors: Tensor) -> Tensor:  # type: ignore[override]
        ctx.save_for_backward(matrices, vectors)  # type: ignore[attr-defined]
        return (matrices * vectors.unsqueeze(-2)).sum(-1)

    @staticmethod
    def backward(ctx: object, grad_out: Tensor) -> tuple[Tensor | None, Tensor | None]:  # type: ignore[override]
        matrices, vectors = ctx.saved_tensors  # type: ignore[attr-defined]
        needs_m, needs_v = ctx.needs_input_grad  # type: ignore[attr-defined]
        grad_m = grad_out.unsqueeze(-1) * vectors.conj().unsqueeze(-2) if needs_m else None
        grad_v = (matrices * grad_out.conj().unsqueeze(-1)).sum(-2).conj() if needs_v else None
        return grad_m, grad_v


def _apply(matrices: Tensor, vectors: Tensor) -> Tensor:
    """One small matrix per edge applied to one vector per edge.

    ``matrices`` is ``(n_edges, b, a)``, ``vectors`` ``(n_edges, a)``; the
    result is ``(n_edges, b)``, as a broadcast multiply and a reduction, with
    the backward of :class:`_BlockApply` when :data:`APPLY_BACKWARD` is set.
    """
    if APPLY_BACKWARD:
        return _BlockApply.apply(matrices, vectors)  # type: ignore[no-any-return]
    return (matrices * vectors.unsqueeze(-2)).sum(-1)


def transport_field(
    field: SO2FeatureField,
    edges: EdgeGeometry,
) -> SO2FeatureField:
    """Parallel-transport an edge-level field from its source gauge to its target gauge.

    ``z -> e^{i n gamma} z`` at frequency ``n``, the identity at frequency
    ``0`` since ``rho_0 = 1``; ``e^{i n gamma}`` is read from
    ``edges.transport_harmonic``.

    Parameters
    ----------
    field
        A field gathered to edge shape, ``field.batch_shape == (edges.n_edges,)``
        (typically ``node_features[edges.source_index]``).
    edges
        Geometry for the same edges.

    Returns
    -------
    SO2FeatureField
        Same field type and batch shape as ``field``.
    """
    with torch.profiler.record_function("gm::transport"):
        transported: dict[int, Tensor] = {}

        for frequency, tensor in field.coefficients.items():
            if frequency == 0:
                transported[0] = tensor  # rho_0 = 1: transport is the identity
            else:
                phase = edges.transport_harmonic(frequency).unsqueeze(-1)
                transported[frequency] = so2.as_real(phase * so2.as_complex(tensor))

        return SO2FeatureField(field.field_type, transported)

class _GaussianRadialBasis(nn.Module):
    """Fixed Gaussian radial-basis embedding of ``s = (r/cutoff)^2``.

    Centers are evenly spaced over ``[0, 1]`` in ``s``; the width is the
    center spacing. The parameterization in ``r^2`` rather than ``r`` makes
    the radial functions built on it smooth in Cartesian tangent coordinates
    through the origin (module docstring).
    """

    centers: Tensor

    def __init__(
        self,
        n_basis: int,
        cutoff: float,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        if n_basis < 1:
            raise ValueError(f"n_basis must be at least 1, got {n_basis}.")

        if not (math.isfinite(cutoff) and cutoff > 0.0):
            raise ValueError(f"cutoff must be finite and positive, got {cutoff}.")

        self.cutoff = float(cutoff)
        self.n_basis = n_basis
        self.width = 1.0 / max(n_basis - 1, 1)

        centers = torch.linspace(0.0, 1.0, n_basis, dtype=dtype, device=device)
        self.register_buffer("centers", centers)  # (n_basis,)

    def forward(self, r: Tensor) -> Tensor:  # (...,) -> (..., n_basis)
        # phi_k(s) = exp(-(s - mu_k)^2 / (2 sigma^2)), s = (r/cutoff)^2. The
        # product rather than .pow(2) keeps the derivative at r = 0 a finite 0.
        normalized = r / self.cutoff
        s = normalized * normalized
        difference = s.unsqueeze(-1) - self.centers  # (..., n_basis)
        return torch.exp(-0.5 * (difference / self.width) ** 2)


class SO2SteerableKernel(nn.Module):
    """A gauge-equivariant, geometric offset-dependent value transformation.

    Parameters
    ----------
    field_type_in, field_type_out
        Input and output field types. Every pair ``(n, m)`` with ``n`` in
        ``field_type_in.frequencies`` and ``m`` in
        ``field_type_out.frequencies`` contributes; the steerability
        constraint does not forbid connecting inequivalent types.
    cutoff
        The maximum geodesic distance the kernel is evaluated at, the span
        of the radial basis; equal to the ``cutoff`` of the
        :class:`SO2Convolution` or :class:`SO2Attention` layer and the radius
        of the ``EdgeGeometry`` passed to :meth:`forward`.
    n_radial_basis
        Number of Gaussian radial-basis functions.
    radial_hidden_features
        Widths of the hidden layers of the radial MLP that turns the radial
        basis into the ``C_i^{m<-n}(r)`` coefficients. May be empty, in
        which case the coefficients are an affine function of the radial
        basis.
    radial_basis
        The radial embedding; only ``"gaussian"`` is accepted.
    bias
        Whether the radial MLP's layers (trunk and both coefficient heads)
        carry biases. A radial bias is compatible with steerability: the
        coefficients multiply the constrained angular basis, so an offset
        in ``C_i(r)`` is another admissible kernel.
    dtype, device
        Floating-point dtype and device of the parameters.

    Notes
    -----
    The kernel is the whole value transformation, evaluated uniformly on
    every edge including coincident points; no branch is selected by
    ``target_index == source_index``. A learned pointwise self map is added
    by the aggregation layer outside the quadrature sum
    (``SO2Convolution``'s ``self_interaction``).
    """

    _column_powers: Tensor

    def __init__(
        self,
        field_type_in: SO2FieldType,
        field_type_out: SO2FieldType,
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

        # Edge geometries are immutable; validate_call checks each object
        # once, keyed by id (EdgeGeometry has no __weakref__ slot).
        self._validated_edge_geometry_ids: set[int] = set()

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
        self.pairs = tuple(
            (n, m)
            for n in field_type_in.frequencies
            for m in field_type_out.frequencies
        )

        if radial_basis != "gaussian":
            raise ValueError(
                f"radial_basis must be 'gaussian', got {radial_basis!r}."
            )
        self.radial_basis_kind: RadialBasisKind = radial_basis
        self.radial_basis = _GaussianRadialBasis(
            n_radial_basis,
            cutoff,
            dtype=dtype,
            device=device,
        )

        widths = [n_radial_basis, *radial_hidden_features]

        if any(width < 1 for width in widths[1:]):
            raise ValueError(
                "radial_hidden_features must contain only positive widths, "
                f"got {tuple(radial_hidden_features)}."
            )

        layers: list[nn.Module] = []

        for in_features, out_features in itertools.pairwise(widths):
            layers.append(
                nn.Linear(in_features, out_features, bias=bias, dtype=dtype, device=device)
            )
            layers.append(nn.SiLU())

        self.radial_trunk = nn.Sequential(*layers)
        trunk_width = widths[-1]

        # The complex head's output is laid out as consecutive (n, m, role)
        # ranges of 2 b_m a_n reals, rows ordered (b_m, a_n, re/im): the
        # "linear" coefficient A = C_1 + i C_2 and, for n, m >= 1, the
        # "antilinear" coefficient B = C_3 + i C_4. The (0, 0) block has no
        # phase and is a separate real head of b_0 a_0 rows ordered (b_0, a_0).
        self._pair_ranges: dict[tuple[int, int, str], slice] = {}
        # The same layout as an ordered partition, for torch.split.
        self._coefficient_order: list[tuple[int, int, str]] = []
        self._coefficient_widths: list[int] = []
        # |k| of the harmonic e^{ik phi} each range multiplies, the power of
        # (r/cutoff) its coefficient carries: per role and per column.
        self._pair_powers: dict[tuple[int, int, str], int] = {}
        column_powers: list[int] = []
        real_width = 0
        offset = 0

        for frequency_n, frequency_m in self.pairs:
            multiplicity_in = field_type_in.multiplicity(frequency_n)
            multiplicity_out = field_type_out.multiplicity(frequency_m)

            if frequency_n == 0 and frequency_m == 0:
                # k = 0: no angular dependence, no radial power.
                real_width = multiplicity_out * multiplicity_in
                continue

            width = 2 * multiplicity_out * multiplicity_in
            self._pair_ranges[(frequency_n, frequency_m, "linear")] = slice(
                offset, offset + width
            )
            self._coefficient_order.append((frequency_n, frequency_m, "linear"))
            self._coefficient_widths.append(width)
            # k = m for (0 -> m), -n for (n -> 0), m - n otherwise: |k| = |m - n|
            # in every case.
            column_powers.extend([abs(frequency_m - frequency_n)] * width)
            self._pair_powers[(frequency_n, frequency_m, "linear")] = abs(
                frequency_m - frequency_n
            )
            offset += width

            if frequency_n >= 1 and frequency_m >= 1:
                self._pair_ranges[(frequency_n, frequency_m, "antilinear")] = slice(
                    offset, offset + width
                )
                self._coefficient_order.append(
                    (frequency_n, frequency_m, "antilinear")
                )
                self._coefficient_widths.append(width)
                # k = m + n >= 2: the antilinear term vanishes at the origin.
                column_powers.extend([frequency_m + frequency_n] * width)
                self._pair_powers[(frequency_n, frequency_m, "antilinear")] = (
                    frequency_m + frequency_n
                )
                offset += width

        complex_width = offset

        self.register_buffer(
            "_column_powers",
            torch.tensor(column_powers, dtype=torch.long, device=device),
            persistent=False,
        )
        self._max_column_power = max(column_powers, default=0)

        self.complex_head = (
            nn.Linear(trunk_width, complex_width, bias=bias, dtype=dtype, device=device)
            if complex_width > 0
            else None
        )
        self.real_head = (
            nn.Linear(trunk_width, real_width, bias=bias, dtype=dtype, device=device)
            if real_width > 0
            else None
        )

    def forward(self, x: SO2FeatureField, edges: EdgeGeometry) -> SO2FeatureField:
        """Apply the kernel to source features, producing one message per edge.

        Parameters
        ----------
        x
            Feature field of type ``field_type_in`` gathered to edge shape,
            ``x.batch_shape == (edges.n_edges,)``, with ``x[frequency][e]``
            the source node's feature for edge ``e`` (typically
            ``node_features[edges.source_index]``).
        edges
            Geometry for the same edges, supplying ``distances``,
            ``angular_harmonic`` and ``transport_harmonic``.

        Returns
        -------
        SO2FeatureField
            Feature field of type ``field_type_out``, batch shape
            ``(edges.n_edges,)``: one message per edge, not yet aggregated.
        """
        self.validate_call(x, edges)
        return self._forward_general(x, edges)

    def validate_call(self, x: SO2FeatureField, edges: EdgeGeometry) -> None:
        """The checks :meth:`forward` makes before computing, for callers that
        evaluate this kernel through another path (the batched multi-head
        attention evaluates several kernels' parameters at once)."""
        if x.field_type != self.field_type_in:
            raise ValueError(
                f"Expected a field of type {self.field_type_in!r}, got "
                f"{x.field_type!r}."
            )

        if x.batch_shape != (edges.n_edges,):
            raise ValueError(
                "x.batch_shape must equal (edges.n_edges,) = "
                f"({edges.n_edges},), got {tuple(x.batch_shape)}. Gather node "
                "features to edge shape (e.g. via edges.source_index) before "
                "calling this kernel."
            )

        required_frequency = (
            self.field_type_in.max_frequency + self.field_type_out.max_frequency
        )

        if edges.max_frequency < required_frequency:
            raise ValueError(
                f"edges tabulates harmonics up to order {edges.max_frequency}, "
                f"but connecting frequency {self.field_type_in.max_frequency} "
                f"input to frequency {self.field_type_out.max_frequency} "
                f"output needs order {required_frequency}. Rebuild the "
                f"discretization with max_frequency >= {required_frequency}."
            )

        if edges.distances.numel() > 0 and id(edges) not in self._validated_edge_geometry_ids:
            maximum_distance = float(edges.distances.max())
            # The tolerance edge_geometry uses against a declared radius, so
            # an edge set built at exactly this cutoff is not rejected on noise.
            tolerance = cutoff_tolerance(edges.distances.dtype)

            if maximum_distance > self.radial_basis.cutoff + tolerance:
                raise ValueError(
                    "edges contains a distance "
                    f"({maximum_distance:.6g}) beyond this kernel's cutoff "
                    f"({self.radial_basis.cutoff:.6g}); pass edges restricted "
                    "to this kernel's support radius (e.g. via "
                    "discretization.edges_within(cutoff))."
                )

            self._validated_edge_geometry_ids.add(id(edges))

    # ------------------------------------------------------------------
    # The factored evaluation: channel mixing on nodes, combination on edges
    # ------------------------------------------------------------------

    def _blocks(self) -> list[tuple[int, int, str]]:
        """Every coefficient block in readout order, the real ``(0, 0)`` block
        (when the types have it) first under the role ``"real"``."""
        blocks: list[tuple[int, int, str]] = []
        if self.real_head is not None:
            blocks.append((0, 0, "real"))
        blocks.extend(self._coefficient_order)
        return blocks

    def _block_weights(self, block: tuple[int, int, str]) -> Tensor:
        """``W_0, W_1, ..., W_T`` of one block as ``(b_m, a_n, T + 1)``: the
        readout's weight columns for that block's coefficients, the bias
        (or zeros without one) first, so that the block's coefficient
        matrix on an edge is ``sum_t h~_t(e) W_t`` with ``h~ = (1, h)``.
        Complex for the phase-carrying blocks, real for ``(0, 0)``."""
        frequency_n, frequency_m, role = block
        multiplicity_in = self.field_type_in.multiplicity(frequency_n)
        multiplicity_out = self.field_type_out.multiplicity(frequency_m)
        if role == "real":
            assert self.real_head is not None
            weight = self.real_head.weight  # (b_0 a_0, T), rows ordered (b_0, a_0)
            trunk_width = weight.shape[1]
            columns = weight.reshape(multiplicity_out, multiplicity_in, trunk_width)
            bias = (
                self.real_head.bias.reshape(multiplicity_out, multiplicity_in)
                if self.real_head.bias is not None
                else torch.zeros(multiplicity_out, multiplicity_in, dtype=weight.dtype, device=weight.device)
            )
            return torch.cat([bias.unsqueeze(-1), columns], dim=-1)
        assert self.complex_head is not None
        rows = self._pair_ranges[block]
        weight = self.complex_head.weight[rows]  # (2 b_m a_n, T), rows ordered (b_m, a_n, re/im)
        trunk_width = weight.shape[1]
        columns = so2.as_complex(
            weight.reshape(multiplicity_out, multiplicity_in, 2, trunk_width).permute(0, 1, 3, 2).contiguous()
        )  # (b_m, a_n, T) complex
        if self.complex_head.bias is not None:
            bias = so2.as_complex(self.complex_head.bias[rows].reshape(multiplicity_out, multiplicity_in, 2).contiguous())
        else:
            bias = torch.zeros(multiplicity_out, multiplicity_in, dtype=columns.dtype, device=columns.device)
        return torch.cat([bias.unsqueeze(-1), columns], dim=-1)

    @staticmethod
    def _input_vector_key(block: tuple[int, int, str]) -> tuple[int, str]:
        """Which node vector a block reads: ``(n, "real" | "plain" | "conj")``."""
        frequency_n, frequency_m, role = block
        if role == "real":
            return (0, "real")
        if frequency_n == 0:
            return (0, "plain")
        return (frequency_n, "conj" if (frequency_m == 0 or role == "antilinear") else "plain")

    def _stacked_block_weights(self, device: torch.device) -> list[tuple[tuple[int, str], list[tuple[int, int, str]], Tensor]]:
        """The blocks grouped by the node vector they read, each group's
        ``W_t`` stacked along the output side as ``(sum b_m, a_n, T + 1)``.
        Under gradients the stack is rebuilt every call as part of the
        forward's graph; without gradients it is cached against the heads'
        parameter versions."""
        groups: dict[tuple[int, str], list[tuple[int, int, str]]] = {}
        for block in self._blocks():
            groups.setdefault(self._input_vector_key(block), []).append(block)

        def build() -> list[tuple[tuple[int, str], list[tuple[int, int, str]], Tensor]]:
            return [(vector_key, blocks, torch.cat([self._block_weights(b) for b in blocks], dim=0).to(device))
                    for vector_key, blocks in groups.items()]

        if torch.is_grad_enabled():
            return build()
        heads = [h for h in (self.complex_head, self.real_head) if h is not None]
        version = tuple((p._version, p.data_ptr()) for h in heads for p in (h.weight, h.bias) if p is not None)
        cache = self.__dict__.setdefault("_stacked_weights", {})
        key = (str(device), version)
        if key in cache:
            return cache[key]  # type: ignore[no-any-return]
        stacked = build()
        cache.clear()  # one version at a time
        cache[key] = stacked
        return stacked

    def node_products(self, x: SO2FeatureField) -> dict[tuple[int, int, str], Tensor]:
        """``W_t v_j`` for every block and every ``t``, on the nodes:
        ``(n_nodes, T + 1, b_m)`` per block, complex except the real ``(0, 0)``
        block. ``v_j`` is the block's input vector before any per-edge
        factor: ``x_0`` for the scalar rows, ``x_n`` for a linear block onto
        ``m >= 1``, ``conj(x_n)`` for the antilinear blocks and for the
        ``(n >= 1, 0)`` rows, whose contraction conjugates the transported
        input. Computed once per forward and reused by every chunk of edges;
        the blocks that read one vector are one product."""
        if x.field_type != self.field_type_in:
            raise ValueError(f"Expected a field of type {self.field_type_in!r}, got {x.field_type!r}.")
        complex_dtype = so2.complex_dtype(x.dtype)
        products: dict[tuple[int, int, str], Tensor] = {}
        with torch.profiler.record_function("gm::kernel/node_products"):
            for (frequency_n, kind), blocks, weights in self._stacked_block_weights(x.device):
                if kind == "real":
                    vector = x[0]  # (n_nodes, a_0) real
                elif frequency_n == 0:
                    vector = x[0].to(complex_dtype)
                else:
                    vector = so2.as_complex(x[frequency_n])  # (n_nodes, a_n)
                    if kind == "conj":
                        vector = vector.conj()
                # (n_nodes, a_n) x (sum b_m, a_n, T + 1) -> (n_nodes, T + 1, sum b_m)
                stacked = torch.einsum("ja,bat->jtb", vector, weights.to(vector.dtype))
                widths = [self.field_type_out.multiplicity(m) for _, m, _ in blocks]
                for block, piece in zip(blocks, torch.split(stacked, widths, dim=-1), strict=True):
                    products[block] = piece
        return products

    def _trunk_hidden(self, edges: EdgeGeometry) -> Tensor:
        """The radial trunk's features on the edges, ``(n_edges, T)``."""
        return apply_radial_trunk(self.radial_trunk, self.radial_basis(edges.distances))

    def _trunk_features(self, edges: EdgeGeometry) -> tuple[Tensor, Tensor | None]:
        """The per-edge ``h~ = (1, h)`` of the radial trunk, ``(n_edges, T + 1)``
        (the bias rides on the constant feature), and the radial powers
        ``(n_edges, max_power + 1)`` (``None`` without complex blocks)."""
        hidden = self._trunk_hidden(edges)  # (n_edges, T)
        ones = torch.ones(hidden.shape[0], 1, dtype=hidden.dtype, device=hidden.device)
        h = torch.cat([ones, hidden], dim=-1)
        radial_powers = self._radial_powers(edges.distances) if self.complex_head is not None else None
        return h, radial_powers

    def _edge_factors(self, edges: EdgeGeometry) -> tuple[Tensor, dict[tuple[int, int, str], Tensor | None]]:
        """The per-edge ``h~`` and every phase-carrying block's scalar
        ``e^{ik phi} e^{+-i n gamma} (r / cutoff)^p`` (``None`` for the real
        block): exactly the factors :meth:`_forward_general` applies to that
        block's input and output."""
        h, radial_powers = self._trunk_features(edges)
        factors: dict[tuple[int, int, str], Tensor | None] = {}
        for block in self._blocks():
            frequency_n, frequency_m, role = block
            if role == "real":
                factors[block] = None
                continue
            assert radial_powers is not None
            scale = radial_powers[:, self._pair_powers[block]]  # (n_edges,) real
            if frequency_n == 0:  # (0 -> m >= 1): e^{i m phi} (A x_0)
                phase = edges.angular_harmonic(frequency_m)
            elif frequency_m == 0:  # (n >= 1 -> 0): Re[ e^{i n phi} e^{-i n gamma} (A conj(x_n)) ]
                phase = edges.angular_harmonic(frequency_n) * edges.transport_harmonic(-frequency_n)
            elif role == "linear":  # A e^{i (m - n) phi} e^{i n gamma} x_n
                phase = edges.angular_harmonic(frequency_m - frequency_n) * edges.transport_harmonic(frequency_n)
            else:  # B e^{i (m + n) phi} e^{-i n gamma} conj(x_n)
                phase = edges.angular_harmonic(frequency_m + frequency_n) * edges.transport_harmonic(-frequency_n)
            factors[block] = phase * scale
        return h, factors

    def forward_factored(
        self,
        x: SO2FeatureField,
        edges: EdgeGeometry,
        products: dict[tuple[int, int, str], Tensor] | None = None,
        groups: tuple[Tensor, Tensor, Tensor] | None = None,
    ) -> SO2FeatureField:
        """:meth:`forward` on ``x[edges.source_index]``, evaluated the other
        way round: the node-side mixes :meth:`node_products` (given, or
        computed here) gathered to the edges and combined with the edge's
        ``(1, h)`` under its phases and radial powers.

        Same terms as :meth:`_forward_general` regrouped by associativity,
        ``(sum_t h_t W_t) v = sum_t h_t (W_t v)``, so the two agree to
        summation order; per edge the readout of ``2 b_m a_n`` coefficients
        from ``T`` features is replaced by a ``T + 1``-term combination of
        ``b_m``-vectors, and the mixing ``W_t v_j`` is paid once per node.
        ``x`` is the node field (``batch_shape == (n_nodes,)``), unlike
        :meth:`forward`'s edge-gathered input. Under ``GM_FUSED_FACTORED``
        the call is one fused kernel, forward and backward
        (:mod:`gm_attention.nn.fused_factored`), the edges grouped by source
        (``groups``, cached per chunk by the callers).
        """
        if x.field_type != self.field_type_in:
            raise ValueError(f"Expected a field of type {self.field_type_in!r}, got {x.field_type!r}.")
        from gm_attention.nn import fused_factored

        if fused_factored.enabled(self, x, edges):
            return fused_factored.forward(self, x, edges, products, groups)
        if products is None:
            products = self.node_products(x)
        n_edges = edges.n_edges
        real_dtype = x.dtype
        complex_dtype = so2.complex_dtype(real_dtype)
        source = edges.source_index
        with torch.profiler.record_function("gm::kernel/radial"):
            h, factors = self._edge_factors(edges)
        outputs: dict[int, Tensor] = {}
        with torch.profiler.record_function("gm::kernel/pairs"):
            for frequency_m in self.field_type_out.frequencies:
                multiplicity_out = self.field_type_out.multiplicity(frequency_m)
                accumulator = torch.zeros(
                    n_edges, multiplicity_out,
                    dtype=real_dtype if frequency_m == 0 else complex_dtype, device=x.device,
                )
                for block in self._blocks():
                    if block[1] != frequency_m:
                        continue
                    mixed = products[block][source]  # (n_edges, T + 1, b_m)
                    combined = (h.unsqueeze(-1) * mixed).sum(dim=1)  # (n_edges, b_m): sum_t h~_t (W_t v_j)
                    factor = factors[block]
                    if factor is None:  # the real (0, 0) block: no phase, no radial power
                        accumulator = accumulator + combined
                        continue
                    term = factor.unsqueeze(-1) * combined
                    accumulator = accumulator + (term.real if frequency_m == 0 else term)
                outputs[frequency_m] = accumulator if frequency_m == 0 else so2.as_real(accumulator)
        return SO2FeatureField(self.field_type_out, outputs)

    def forward_factored_aggregated(
        self,
        x: SO2FeatureField,
        edges: EdgeGeometry,
        weights: Tensor,
        products: dict[tuple[int, int, str], Tensor] | None = None,
        groups: tuple[Tensor, Tensor, Tensor] | None = None,
    ) -> SO2FeatureField:
        """Each target's weighted sum of :meth:`forward_factored`'s messages,
        ``sum_{e -> i} weights_e message_e``: the node field ``(n_targets,)``
        of the output type (``n_targets`` from ``edges.row_ptr``), what a
        layer forms from the messages under its attention weights or its
        window times quadrature weight. Under ``GM_FUSED_FACTORED`` the
        rows, the weighted segment sum and the backward through the target
        index are one op (:class:`~gm_attention.nn.fused_factored.FactoredAggregate`:
        no weighted per-edge tensor exists in either direction); otherwise
        the messages are weighted and summed here, per frequency.
        """
        from gm_attention.nn import fused_factored
        from gm_attention.nn._scatter import scatter_add_to_nodes

        if fused_factored.enabled(self, x, edges):
            return fused_factored.forward_aggregated(self, x, edges, weights, products, groups)
        messages = self.forward_factored(x, edges, products, groups)
        outputs: dict[int, Tensor] = {}
        for frequency in self.field_type_out.frequencies:
            message = messages[frequency]  # (n_edges, b_m[, 2])
            per_edge_weight = weights.reshape(edges.n_edges, *([1] * (message.ndim - 1)))
            outputs[frequency] = scatter_add_to_nodes(per_edge_weight * message, edges.row_ptr)
        return SO2FeatureField(self.field_type_out, outputs)

    def _radial_powers(self, distances: Tensor) -> Tensor:
        """``(r/cutoff)^p`` for every ``p`` this kernel needs, ``(n_edges, max_power + 1)``.

        One column per distinct power: each role's slice carries a single
        power (``|m - n|`` for a linear block, ``m + n`` for an antilinear
        one), applied to that role's contracted ``(n_edges, b_m)`` output,
        since a per-edge scalar commutes with the contraction,
        ``(s A) v = s (A v)``. Built by repeated multiplication rather than
        ``pow``, whose backward for ``x ** 0`` is ``nan`` at ``r = 0``.
        """
        normalized = distances / self.radial_basis.cutoff  # (n_edges,)

        table = [torch.ones_like(normalized)]
        for _ in range(self._max_column_power):
            table.append(table[-1] * normalized)

        return torch.stack(table, dim=-1)  # (n_edges, max_power + 1)

    def _forward_general(
        self,
        x: SO2FeatureField,
        edges: EdgeGeometry,
    ) -> SO2FeatureField:
        """The steerable block formula of the module docstring, in complex
        form, on every edge: uniform over the support including ``r = 0``,
        where the ``r^|k|`` scaling leaves an intertwiner."""
        from gm_attention.nn import fused_pairs

        if fused_pairs.enabled(self, x, edges):
            return fused_pairs.forward_general(self, x, edges)

        n_edges = edges.n_edges
        real_dtype = x.dtype
        complex_dtype = so2.complex_dtype(real_dtype)

        with torch.profiler.record_function("gm::kernel/radial"):
            embedding = self.radial_basis(edges.distances)  # (n_edges, n_basis)
            hidden = apply_radial_trunk(self.radial_trunk, embedding)  # (n_edges, trunk_width)

        complex_coefficients = (
            split_k_linear(hidden, self.complex_head.weight, self.complex_head.bias)
            if self.complex_head is not None
            else None
        )
        coefficient_chunks = (
            dict(
                zip(
                    self._coefficient_order,
                    torch.split(complex_coefficients, self._coefficient_widths, dim=1),
                    strict=True,
                )
            )
            if complex_coefficients is not None
            else None
        )
        # The regularity condition: an e^{ik phi} mode's coefficient vanishes
        # like r^|k|, applied to each role's contracted output (_radial_powers).
        radial_powers = (
            self._radial_powers(edges.distances)
            if complex_coefficients is not None
            else None
        )

        # The real head is the k = 0 block and needs no such factor.
        real_coefficients = (
            split_k_linear(hidden, self.real_head.weight, self.real_head.bias)
            if self.real_head is not None
            else None
        )

        # Transport every input frequency into the target gauge once, up front.
        transported_field = transport_field(x, edges)

        transported: dict[int, Tensor] = {
            frequency_n: (
                transported_field[0]  # (n_edges, a_0) real
                if frequency_n == 0
                else so2.as_complex(transported_field[frequency_n])  # (n_edges, a_n) complex
            )
            for frequency_n in self.field_type_in.frequencies
        }

        def role_scale(frequency_n: int, frequency_m: int, role: str) -> Tensor:
            """``(r/cutoff)^p`` for one role, shaped to broadcast over ``b_m``."""
            assert radial_powers is not None
            power = self._pair_powers[(frequency_n, frequency_m, role)]
            return radial_powers[:, power].unsqueeze(-1)  # (n_edges, 1)

        outputs: dict[int, Tensor] = {}

        pair_scope = torch.profiler.record_function("gm::kernel/pairs")
        pair_scope.__enter__()
        for frequency_m in self.field_type_out.frequencies:
            multiplicity_out = self.field_type_out.multiplicity(frequency_m)
            accumulator = torch.zeros(
                n_edges,
                multiplicity_out,
                dtype=real_dtype if frequency_m == 0 else complex_dtype,
                device=x.device,
            )

            for frequency_n in self.field_type_in.frequencies:
                multiplicity_in = self.field_type_in.multiplicity(frequency_n)

                if frequency_n == 0 and frequency_m == 0:
                    assert real_coefficients is not None
                    matrix = real_coefficients.reshape(
                        n_edges, multiplicity_out, multiplicity_in
                    )
                    contribution = _apply(
                        matrix, transported[0]
                    )  # (n_edges, b_0) real
                    accumulator = accumulator + contribution
                    continue

                assert coefficient_chunks is not None
                linear_matrix = so2.as_complex(
                    coefficient_chunks[(frequency_n, frequency_m, "linear")].reshape(
                        n_edges, multiplicity_out, multiplicity_in, 2
                    )
                )  # (n_edges, b_m, a_n) complex

                if frequency_n == 0:  # frequency_m >= 1
                    # (n = 0, m >= 1): V(phi) x0 = e^{i m phi} (A x0), A = C_1 + i C_2;
                    # x0 is real (rho_0 trivial) and is promoted to complex.
                    source = transported[0].to(complex_dtype)  # (n_edges, a_0)
                    phase = edges.angular_harmonic(frequency_m).unsqueeze(-1)
                    contribution = phase * _apply(
                        linear_matrix, source
                    )  # (n_edges, b_m) complex
                    contribution = contribution * role_scale(
                        frequency_n, frequency_m, "linear"
                    )
                elif frequency_m == 0:  # frequency_n >= 1
                    # (n >= 1, m = 0): V(phi) z = Re[ conj(A) e^{-i n phi} z ], the real
                    # pairing against the row basis {e_1^T rho_n(r_phi)^{-1}, e_2^T rho_n(r_phi)^{-1}}.
                    phase = edges.angular_harmonic(-frequency_n).unsqueeze(-1)
                    rotated = phase * transported[frequency_n]  # (n_edges, a_n)
                    # Re[conj(A) r] = Re[A conj(r)]: the narrow input is
                    # conjugated, not the block.
                    contribution = _apply(
                        linear_matrix, rotated.conj()
                    ).real  # (n_edges, b_0) real
                    # A real scalar commutes with taking the real part.
                    contribution = contribution * role_scale(
                        frequency_n, frequency_m, "linear"
                    )
                else:  # frequency_n, frequency_m >= 1
                    # V_{m<-n}(r,phi) z = A(r) e^{i(m-n)phi} z + B(r) e^{i(m+n)phi} conj(z):
                    # a real linear map C -> C is a z + b conj(z) for some a, b in C.
                    antilinear_matrix = so2.as_complex(
                        coefficient_chunks[
                            (frequency_n, frequency_m, "antilinear")
                        ].reshape(n_edges, multiplicity_out, multiplicity_in, 2)
                    )  # (n_edges, b_m, a_n) complex

                    linear_phase = edges.angular_harmonic(
                        frequency_m - frequency_n
                    ).unsqueeze(-1)
                    antilinear_phase = edges.angular_harmonic(
                        frequency_m + frequency_n
                    ).unsqueeze(-1)

                    # |m - n| and m + n differ: each term takes its own radial scalar.
                    linear_term = _apply(
                        linear_matrix, linear_phase * transported[frequency_n]
                    ) * role_scale(frequency_n, frequency_m, "linear")
                    antilinear_term = _apply(
                        antilinear_matrix,
                        antilinear_phase * transported[frequency_n].conj(),
                    ) * role_scale(frequency_n, frequency_m, "antilinear")

                    contribution = linear_term + antilinear_term  # (n_edges, b_m)

                accumulator = accumulator + contribution

            outputs[frequency_m] = (
                accumulator if frequency_m == 0 else so2.as_real(accumulator)
            )
        pair_scope.__exit__(None, None, None)

        return SO2FeatureField(self.field_type_out, outputs)

    def extra_repr(self) -> str:
        return (
            f"field_type_in={self.field_type_in!r}, "
            f"field_type_out={self.field_type_out!r}, "
            f"cutoff={self.radial_basis.cutoff:.6g}, "
            f"radial_basis={self.radial_basis_kind}"
        )
