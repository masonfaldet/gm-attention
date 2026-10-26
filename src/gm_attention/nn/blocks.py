"""Blocks assembled from the equivariant primitives of this package.

``SO2TransformerBlock`` is the pre-norm transformer encoder layer

    x'  = x  + Drop(SO2MHSA(SO2LayerNorm(x), edges))
    x'' = x' + Drop(SO2Linear(sigma(SO2Linear(SO2LayerNorm(x')))))

with every piece an equivariant layer of this package, so the block is
equivariant as their composition. ``IcoEncode`` and ``IcoDecode`` move a
field between one icosphere subdivision level and the next through
bipartite edge sets built once with the discretizations;
``IcoLowPassRestriction`` is a fixed normalized restriction and
``EquiEncode`` the learned restriction between two arbitrary grids.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import Literal

import torch
from torch import Tensor, nn

from gm_attention.discretization.discretization import SphericalDiscretization
from gm_attention.discretization.edge_geometry import EdgeGeometry
from gm_attention.nn import fused_factored
from gm_attention.nn._scatter import gather_to_edges, scatter_add_to_nodes
from gm_attention.nn.attention import SO2MHSA, ValueKind
from gm_attention.nn.convolution import (
    SelfInteractionKind,
    SO2Convolution,
    WindowKind,
    evaluate_aggregation_window,
)
from gm_attention.nn.kernels import (
    RadialBasisKind,
    SO2SteerableKernel,
    factored_enabled,
    transport_field,
)
from gm_attention.nn.linear import SO2Linear
from gm_attention.nn.nonlinearities import SO2ScalarGatedNonlinearity
from gm_attention.nn.normalization import SO2Dropout, SO2LayerNorm
from gm_attention.representations.field_type import SO2FeatureField, SO2FieldType

#: The operator mixing information across nodes in a block: multi-head attention, or a steerable convolution over the same neighborhood.
MixerKind = Literal["attention", "convolution"]

NonlinearityKind = Literal[
    "scalar_gate",
    "scalar_gate_double_sigmoid",
]

# The ffn_multiplier default, compared against when hidden_field_type is also given.
_FFN_MULTIPLIER_DEFAULT = 2

# Every entry takes (field_type, *, dtype, device).
_NONLINEARITIES: dict[str, Callable[..., nn.Module]] = {
    # The two entries differ only in the gate's activation: silu, or
    # 2*sigmoid (strictly positive, bounded in (0, 2)).
    "scalar_gate": SO2ScalarGatedNonlinearity,
    "scalar_gate_double_sigmoid": partial(
        SO2ScalarGatedNonlinearity, gate="double_sigmoid"
    ),
}


def build_nonlinearity(
    kind: NonlinearityKind,
    field_type: SO2FieldType,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> nn.Module:
    """The equivariant activation ``kind`` names, at ``field_type``.

    The single reader of the name-to-module table, so a model assembled
    outside this module offers the same choices as
    :class:`SO2TransformerBlock`.
    """
    if kind not in _NONLINEARITIES:
        raise ValueError(
            f"nonlinearity must be one of {tuple(_NONLINEARITIES)}, got {kind!r}."
        )
    return _NONLINEARITIES[kind](field_type, dtype=dtype, device=device)


def _add_fields(a: SO2FeatureField, b: SO2FeatureField) -> SO2FeatureField:
    """Sum two feature fields of the same type (a residual-stream addition)."""
    return SO2FeatureField(
        a.field_type,
        {frequency: a[frequency] + b[frequency] for frequency in a.field_type.frequencies},
    )


class SO2TransformerBlock(nn.Module):
    """A pre-norm transformer block, ``field_type -> field_type``.

    The feed-forward branch widens ``field_type`` to ``hidden_field_type``
    with an :class:`~gm_attention.nn.linear.SO2Linear`, applies the
    equivariant activation ``nonlinearity`` selects, and narrows back.
    ``field_type`` must be uniform (see
    :meth:`~gm_attention.representations.field_type.SO2FieldType.uniform_multiplicity`),
    as ``SO2LayerNorm`` and ``SO2MHSA`` require.

    Parameters
    ----------
    field_type
        The shared input, output, and residual-stream field type;
        ``num_heads`` must divide each multiplicity (see
        :class:`~gm_attention.nn.attention.SO2MHSA`).
    num_heads
        Number of self-attention heads; see :class:`~gm_attention.nn.attention.SO2MHSA`.
    ffn_multiplier
        Feed-forward width as a multiple of ``field_type``: the hidden type
        is ``field_type.scaled(ffn_multiplier)``, so the ratio between
        frequencies is preserved (``{0: 8, 1: 4}`` with ``ffn_multiplier=2``
        gives ``{0: 16, 1: 8}``). Ignored when ``hidden_field_type`` is
        given.
    hidden_field_type
        The feed-forward hidden type, stated exactly. Mutually exclusive
        with a non-default ``ffn_multiplier``.
    cutoff
        Support radius for the attention value kernel and its aggregation
        window; see :class:`~gm_attention.nn.attention.SO2MHSA`.
    window
        Aggregation window kind; see
        :func:`gm_attention.nn.convolution.evaluate_aggregation_window`.
    n_radial_basis, radial_hidden_features, radial_basis
        Passed to every attention head's internal steerable kernel; see
        :class:`~gm_attention.nn.attention.SO2MHSA`.
    dropout
        Drop probability applied independently to the attention branch's
        output and to the feed-forward branch's output, each through its
        own :class:`~gm_attention.nn.normalization.SO2Dropout`. Zero makes
        both an exact identity.
    eps
        Passed to both internal :class:`~gm_attention.nn.normalization.SO2LayerNorm`
        instances.
    bias
        Whether the linear/attention/kernel sub-layers carry a bias where
        one is equivariant; see :class:`~gm_attention.nn.linear.SO2Linear`.
    nonlinearity
        The equivariant activation of the feed-forward branch.

        - ``"scalar_gate"``: gates each nonzero frequency by a learned
          function of the scalar channels
          (:class:`~gm_attention.nn.nonlinearities.SO2ScalarGatedNonlinearity`),
          with a ``silu`` gate. Requires frequency-0 channels in the hidden
          type.
        - ``"scalar_gate_double_sigmoid"`` (the default): the same gate
          with a ``2 * sigmoid`` activation, strictly positive and bounded
          in ``(0, 2)``.
    mixer
        ``"attention"`` installs :class:`~gm_attention.nn.attention.SO2MHSA`;
        ``"convolution"`` installs a type-preserving
        :class:`~gm_attention.nn.convolution.SO2Convolution` over the same
        neighborhood and window, without feature-dependent weights
        (``num_heads`` is then ignored).
    attention_batched
        Evaluate the attention's heads in one pass; see
        :class:`~gm_attention.nn.attention.SO2MHSA`.
    value_kind
        The attention's value transform; see
        :class:`~gm_attention.nn.attention.SO2MHSA`.
    dtype, device
        Floating-point dtype and device of every parameter.
    """

    def __init__(
        self,
        field_type: SO2FieldType,
        num_heads: int,
        *,
        ffn_multiplier: int = 2,
        hidden_field_type: SO2FieldType | None = None,
        cutoff: float,
        window: WindowKind = "hard",
        n_radial_basis: int = 8,
        radial_hidden_features: Sequence[int] = (16,),
        radial_basis: RadialBasisKind = "gaussian",
        dropout: float = 0.0,
        eps: float = 1e-5,
        bias: bool = True,
        nonlinearity: NonlinearityKind = "scalar_gate_double_sigmoid",
        mixer: MixerKind = "attention",
        attention_batched: bool = False,
        value_kind: ValueKind = "steerable",
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        if not isinstance(field_type, SO2FieldType):
            raise TypeError(
                f"field_type must be an SO2FieldType, got {type(field_type).__name__}."
            )

        if hidden_field_type is not None and ffn_multiplier != _FFN_MULTIPLIER_DEFAULT:
            raise ValueError(
                "Pass either ffn_multiplier or hidden_field_type, not both -- "
                f"got ffn_multiplier={ffn_multiplier} and "
                f"hidden_field_type={hidden_field_type!r}, which would "
                "usually disagree."
            )

        if hidden_field_type is not None and not isinstance(
            hidden_field_type, SO2FieldType
        ):
            raise TypeError(
                "hidden_field_type must be an SO2FieldType, got "
                f"{type(hidden_field_type).__name__}."
            )

        if nonlinearity not in _NONLINEARITIES:
            raise ValueError(
                f"nonlinearity must be one of {tuple(_NONLINEARITIES)}, got "
                f"{nonlinearity!r}."
            )

        self.nonlinearity: NonlinearityKind = nonlinearity
        self.field_type = field_type
        # scaled() validates ffn_multiplier (positive int, bool rejected).
        self.hidden_field_type = (
            hidden_field_type
            if hidden_field_type is not None
            else field_type.scaled(ffn_multiplier)
        )

        if mixer not in ("attention", "convolution"):
            raise ValueError(
                f"mixer must be 'attention' or 'convolution', got {mixer!r}."
            )
        self.mixer: MixerKind = mixer

        self.norm1 = SO2LayerNorm(field_type, eps=eps, bias=bias, dtype=dtype, device=device)
        # Both mixers share the (x, edges) forward and forward_chunked, so the
        # residual wiring addresses `self.attention` under either.
        self.attention: nn.Module
        if mixer == "attention":
            self.attention = SO2MHSA(
                field_type,
                num_heads,
                cutoff=cutoff,
                window=window,
                n_radial_basis=n_radial_basis,
                radial_hidden_features=radial_hidden_features,
                radial_basis=radial_basis,
                bias=bias,
                batched=attention_batched,
                value_kind=value_kind,
                dtype=dtype,
                device=device,
            )
        else:
            # Same neighborhood and window, no feature-dependent weights;
            # self_interaction="none" leaves the identity to the outer residual.
            self.attention = SO2Convolution(
                field_type,
                field_type,
                cutoff=cutoff,
                window=window,
                self_interaction="none",
                n_radial_basis=n_radial_basis,
                radial_hidden_features=radial_hidden_features,
                radial_basis=radial_basis,
                bias=bias,
                dtype=dtype,
                device=device,
            )
        self.attention_dropout = SO2Dropout(field_type, p=dropout)

        self.norm2 = SO2LayerNorm(field_type, eps=eps, bias=bias, dtype=dtype, device=device)
        self.linear1 = SO2Linear(
            field_type, self.hidden_field_type, bias=bias, dtype=dtype, device=device
        )
        self.activation = build_nonlinearity(
            nonlinearity, self.hidden_field_type, dtype=dtype, device=device
        )
        self.linear2 = SO2Linear(
            self.hidden_field_type, field_type, bias=bias, dtype=dtype, device=device
        )
        self.mlp_dropout = SO2Dropout(field_type, p=dropout)

    def forward(
        self,
        x: SO2FeatureField,
        edges: EdgeGeometry | SphericalDiscretization,
    ) -> SO2FeatureField:
        """Apply the block.

        Parameters
        ----------
        x
            Node-level feature field of type ``field_type``.
        edges
            Passed straight through to the mixer; either an
            ``EdgeGeometry`` or a ``SphericalDiscretization``, or a
            :class:`~gm_attention.nn.stencil_plan.StencilPlan` over a stencil
            built at the block's cutoff, which evaluates the mixer on the
            dense fixed-grid layout (see :meth:`SO2MHSA.forward`).

        Returns
        -------
        SO2FeatureField
            Node-level feature field of type ``field_type``.
        """
        # x <- x + Drop(MHSA(LN(x))); x <- x + Drop(W_2 sigma(W_1 LN(x))).
        # Every term is type-preserving, so each residual sum adds fields of
        # the same type in the same per-node gauge.
        return self._run(x, edges, chunk_size=None)

    def forward_chunked(
        self,
        x: SO2FeatureField,
        edges: EdgeGeometry | SphericalDiscretization,
        *,
        chunk_size: int,
    ) -> SO2FeatureField:
        """:meth:`forward` with the mixer chunked over target ranges.

        Only the mixer holds anything per edge; the norms, the feed-forward
        and both residuals are pointwise. See
        :meth:`~gm_attention.nn.attention.SO2Attention.forward_chunked`.
        """
        return self._run(x, edges, chunk_size=chunk_size)

    def _run(
        self,
        x: SO2FeatureField,
        edges: EdgeGeometry | SphericalDiscretization,
        *,
        chunk_size: int | None,
    ) -> SO2FeatureField:
        attended = (
            self.attention(self.norm1(x), edges)
            if chunk_size is None
            # Both mixers provide forward_chunked; nn.Module's type does not
            # know that, hence the ignore rather than a cast to one of them.
            else self.attention.forward_chunked(  # type: ignore[operator]
                self.norm1(x), edges, chunk_size=chunk_size
            )
        )  # type field_type
        x = _add_fields(x, self.attention_dropout(attended))

        hidden = self.linear1(self.norm2(x))  # type hidden_field_type
        hidden = self.activation(hidden)  # type hidden_field_type
        hidden = self.linear2(hidden)  # type field_type
        x = _add_fields(x, self.mlp_dropout(hidden))

        return x

    def extra_repr(self) -> str:
        return (
            f"field_type={self.field_type!r}, "
            f"hidden_field_type={self.hidden_field_type!r}, "
            f"nonlinearity={self.nonlinearity!r}"
        )


class IcoEncode(nn.Module):
    """Downsamples a fine icosphere feature field to the next-coarser level.

    The bipartite form of :class:`~gm_attention.nn.convolution.SO2Convolution`'s
    sum: sources are fine-grid nodes, targets the coarse-grid nodes, and
    the per-edge term ``V(v_ij) rho_in(g_{i<-j}) x_j`` (transport included)
    is :class:`~gm_attention.nn.kernels.SO2SteerableKernel`'s, so the
    equivariance argument is that of the single-grid convolution. The
    grids are the pair
    :func:`~gm_attention.discretization.grids.icosphere.build_icosphere_refinement`
    produces: the coarse nodes are the fine grid's first ``n_coarse``
    nodes, with the same positions and the same tangent frames.

    Parameters
    ----------
    field_type_in
        Field type on the fine grid.
    field_type_out
        Field type produced on the coarse grid.
    cutoff
        Aggregation radius, shared between the kernel's radial basis and
        the aggregation window; must equal the radius the ``edges`` passed
        to :meth:`forward` were built with
        (:func:`~gm_attention.discretization.grids.icosphere.build_icosphere_downsample_edges`),
        which :meth:`forward` cannot check.
    window
        Aggregation window kind; see
        :func:`gm_attention.nn.convolution.evaluate_aggregation_window`.
    n_radial_basis, radial_hidden_features, radial_basis, bias, dtype, device
        Passed to the internal ``SO2SteerableKernel``.
    """

    def __init__(
        self,
        field_type_in: SO2FieldType,
        field_type_out: SO2FieldType,
        *,
        cutoff: float,
        window: WindowKind = "hard",
        n_radial_basis: int = 8,
        radial_hidden_features: Sequence[int] = (16,),
        radial_basis: RadialBasisKind = "gaussian",
        bias: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        if window not in ("hard", "quartic"):
            raise ValueError(f"window must be 'hard' or 'quartic', got {window!r}.")

        self.field_type_in = field_type_in
        self.field_type_out = field_type_out
        self.cutoff = float(cutoff)
        self.window: WindowKind = window

        self.kernel = SO2SteerableKernel(
            field_type_in,
            field_type_out,
            cutoff=cutoff,
            n_radial_basis=n_radial_basis,
            radial_hidden_features=radial_hidden_features,
            radial_basis=radial_basis,
            bias=bias,
            dtype=dtype,
            device=device,
        )

    def forward(self, x: SO2FeatureField, edges: EdgeGeometry) -> SO2FeatureField:
        """Aggregate fine-grid features down to the coarse grid.

        Parameters
        ----------
        x
            Fine-grid feature field of type ``field_type_in``,
            ``x.batch_shape == (n_fine,)``. ``n_fine`` is not validated
            against ``edges`` (``edges.row_ptr`` describes the coarse
            targets); an ``x`` too small for ``edges.source_index`` fails
            as an out-of-range index.
        edges
            Bipartite fine -> coarse edges, from
            :func:`~gm_attention.discretization.grids.icosphere.build_icosphere_downsample_edges`.
            ``edges.row_ptr`` must be sized ``(n_coarse + 1,)``.

        Returns
        -------
        SO2FeatureField
            Coarse-grid feature field of type ``field_type_out``,
            ``batch_shape == (n_coarse,)`` where
            ``n_coarse = edges.row_ptr.shape[0] - 1``.
        """
        if x.field_type != self.field_type_in:
            raise ValueError(
                f"Expected a field of type {self.field_type_in!r}, got "
                f"{x.field_type!r}."
            )

        # y_i = sum_{j in N(i)} c(v_ij) w_ij K(v_ij) rho(g_{i<-j}) x_j with
        # i a coarse target and j a fine source. Under the factored switch
        # the kernel reads the fine field and its node-side mixes instead of
        # the edge-gathered sources; the bipartite edge set is one chunk.
        # c(v_ij) w_ij: aggregation window times tangent-space quadrature
        # weight, the discrete measure approximating dv.
        weight = evaluate_aggregation_window(
            edges.distances, self.cutoff, self.window
        ) * edges.tangent_quadrature_weights  # (n_edges,)

        if self._factored():
            products = self.kernel.node_products(x)
            groups = (
                fused_factored.source_groups(
                    edges.source_index, fused_factored.group_key(edges, 0, int(edges.n_edges)), owner=edges.edge_index
                )
                if fused_factored.enabled(self.kernel, x, edges) else None
            )
            # The messages weighted and summed per coarse target in one step
            # (one op per direction under the fused kernel).
            return self.kernel.forward_factored_aggregated(x, edges, weight, products, groups)

        source = gather_to_edges(x, edges.source_index)  # type_in, (n_edges,)
        messages = self.kernel(source, edges)  # type_out, (n_edges,)

        outputs: dict[int, Tensor] = {}

        for frequency in self.field_type_out.frequencies:
            # (n_edges, b_n) at frequency 0, (n_edges, b_n, 2) above it.
            message = messages[frequency]
            per_edge_weight = weight.reshape(
                edges.n_edges, *([1] * (message.ndim - 1))
            )  # (n_edges, 1[, 1])
            outputs[frequency] = scatter_add_to_nodes(
                per_edge_weight * message, edges.row_ptr
            )  # (n_coarse, b_n[, 2])

        return SO2FeatureField(self.field_type_out, outputs)

    def _factored(self) -> bool:
        """Whether the kernel is evaluated through its factored path
        (``GM_FACTORED_KERNEL``; :func:`~gm_attention.nn.kernels.factored_enabled`)."""
        return factored_enabled()

    def extra_repr(self) -> str:
        return (
            f"field_type_in={self.field_type_in!r}, "
            f"field_type_out={self.field_type_out!r}, "
            f"cutoff={self.cutoff:.6g}, window={self.window!r}"
        )


class IcoLowPassRestriction(nn.Module):
    r"""Normalized, constant-preserving restriction from a fine grid to a coarse one.

    A fixed (unlearned) normalized average of transported neighbors, the
    discrete analogue of a low-pass filter applied before decimation:

    .. math::

        (R f)_i = \frac{\sum_{j \in N_i} \eta(r_{ij})\, q_{ij}\,
                        \rho(g_{i \leftarrow j}) f_j}
                       {\sum_{j \in N_i} \eta(r_{ij})\, q_{ij}}

    with ``eta`` the window and ``q_ij`` the tangent quadrature weight. The
    denominator is one scalar per target, applied identically to every
    representation coordinate, so equivariance is preserved. Scalar
    constants are reproduced exactly; a non-scalar field that is
    covariantly constant over a neighborhood transports to one value and
    averages back to it. The type cannot change: a normalized average of
    ``rho_n`` coefficients is still ``rho_n``.

    Parameters
    ----------
    field_type
        The shared input and output field type.
    cutoff
        Low-pass radius, tied to the refinement geometry: wide enough to
        cover the fine nodes belonging to one coarse cell.
    window
        Low-pass profile; see
        :func:`~gm_attention.nn.convolution.evaluate_aggregation_window`.
        Defaults to ``"quartic"``, a smooth profile.
    eps
        Floor on the per-target denominator, for a target collecting no
        positive weight.
    """

    def __init__(
        self,
        field_type: SO2FieldType,
        *,
        cutoff: float,
        window: WindowKind = "quartic",
        eps: float = 1e-12,
    ) -> None:
        super().__init__()

        if window not in ("hard", "quartic"):
            raise ValueError(f"window must be 'hard' or 'quartic', got {window!r}.")

        self.field_type = field_type
        self.cutoff = float(cutoff)
        self.window: WindowKind = window
        self.eps = float(eps)

    def forward(self, x: SO2FeatureField, edges: EdgeGeometry) -> SO2FeatureField:
        """Restrict fine-grid features to the coarse grid.

        Parameters
        ----------
        x
            Fine-grid feature field of type ``field_type``.
        edges
            Bipartite fine -> coarse edges, as :class:`IcoEncode` takes.

        Returns
        -------
        SO2FeatureField
            Coarse-grid feature field of the *same* type.
        """
        if x.field_type != self.field_type:
            raise ValueError(
                f"Expected a field of type {self.field_type!r}, got {x.field_type!r}."
            )

        # (R x)_i = [sum_j eta(r_ij) q_ij rho(g_{i<-j}) x_j] / [sum_j eta(r_ij) q_ij].
        # Transport first: coefficients in different frames cannot be averaged.
        source = gather_to_edges(x, edges.source_index)  # (n_edges,)
        transported = transport_field(source, edges)  # (n_edges,), target gauge

        # eta(r_ij) q_ij, the unnormalized nonnegative weight.
        weight = evaluate_aggregation_window(
            edges.distances, self.cutoff, self.window
        ) * edges.tangent_quadrature_weights  # (n_edges,)

        # One scalar denominator per target, shared across every frequency
        # and coordinate; a scalar commutes with every rho_n.
        denominator = scatter_add_to_nodes(
            weight, edges.row_ptr
        ).clamp_min(self.eps)  # (n_coarse,)

        outputs: dict[int, Tensor] = {}

        for frequency in self.field_type.frequencies:
            value = transported[frequency]
            per_edge_weight = weight.reshape(
                edges.n_edges, *([1] * (value.ndim - 1))
            )
            numerator = scatter_add_to_nodes(per_edge_weight * value, edges.row_ptr)
            outputs[frequency] = numerator / denominator.reshape(
                -1, *([1] * (numerator.ndim - 1))
            )

        return SO2FeatureField(self.field_type, outputs)

    def extra_repr(self) -> str:
        return (
            f"field_type={self.field_type!r}, cutoff={self.cutoff:.6g}, "
            f"window={self.window!r}"
        )


class IcoDecode(nn.Module):
    """Upsamples a coarse icosphere feature field to the next-finer level.

    Two steps. (1) Geometric upsample: a fine node that is also a coarse
    node (the first ``n_coarse`` fine nodes) copies its coarse feature
    (same position, same frame, no transport); a new fine node ``k`` (the
    sphere-normalized midpoint of two coarse nodes) takes the average of
    its two parents' features, each transported into its gauge,
    ``x~_k = (1/2) sum_{p in parents(k)} rho(g_{k<-p}) x_p``. (2) A
    fine-to-fine :class:`~gm_attention.nn.convolution.SO2Convolution` maps
    the upsampled ``field_type_in`` field to ``field_type_out``.

    Step 1 is gauge equivariant on its own: regauging every node by
    ``theta`` sends the transport angle ``gamma_{k<-p}`` to
    ``gamma_{k<-p} + theta_p - theta_k``, so a parent's regauged and
    transported contribution is ``rho(gamma_{k<-p} + theta_p - theta_k)
    rho(-theta_p) x_p = rho(-theta_k) rho(gamma_{k<-p}) x_p``, the output
    node's own ``rho(-theta_k)`` applied to the original contribution;
    averaging is linear, so the average transforms the same way. Step 2
    carries the convolution's own equivariance.

    Parameters
    ----------
    field_type_in
        Field type on the coarse grid, and of the geometric upsample's
        output on the fine grid.
    field_type_out
        Field type produced on the fine grid by the internal convolution.
    cutoff
        Support radius of the internal fine-to-fine convolution.
    window
        Aggregation window kind for that convolution; see
        :func:`gm_attention.nn.convolution.evaluate_aggregation_window`.
    self_interaction
        Self-interaction mode of the internal convolution.
    n_radial_basis, radial_hidden_features, radial_basis, bias, dtype, device
        Passed to the internal ``SO2Convolution``.
    """

    def __init__(
        self,
        field_type_in: SO2FieldType,
        field_type_out: SO2FieldType,
        *,
        cutoff: float,
        window: WindowKind = "hard",
        self_interaction: SelfInteractionKind = "none",
        n_radial_basis: int = 8,
        radial_hidden_features: Sequence[int] = (16,),
        radial_basis: RadialBasisKind = "gaussian",
        bias: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        self.field_type_in = field_type_in
        self.field_type_out = field_type_out

        self.conv = SO2Convolution(
            field_type_in,
            field_type_out,
            cutoff=cutoff,
            window=window,
            self_interaction=self_interaction,
            n_radial_basis=n_radial_basis,
            radial_hidden_features=radial_hidden_features,
            radial_basis=radial_basis,
            bias=bias,
            dtype=dtype,
            device=device,
        )

    def forward(
        self,
        x: SO2FeatureField,
        upsample_edges: EdgeGeometry,
        fine_edges: EdgeGeometry | SphericalDiscretization,
    ) -> SO2FeatureField:
        """Upsample coarse-grid features to the fine grid.

        Parameters
        ----------
        x
            Coarse-grid feature field of type ``field_type_in``,
            ``x.batch_shape == (n_coarse,)``; ``n_coarse`` is the number of
            inherited fine nodes (see
            :class:`~gm_attention.discretization.grids.icosphere.IcosphereRefinement`).
        upsample_edges
            Each new fine node's two parent edges, from
            :func:`~gm_attention.discretization.grids.icosphere.build_icosphere_upsample_edges`.
            ``upsample_edges.row_ptr`` must be sized ``(n_fine + 1,)``, with
            its first ``n_coarse`` rows carrying no incident edges.
        fine_edges
            Fine-to-fine edges or discretization for the internal
            :class:`~gm_attention.nn.convolution.SO2Convolution`.

        Returns
        -------
        SO2FeatureField
            Fine-grid feature field of type ``field_type_out``.
        """
        if x.field_type != self.field_type_in:
            raise ValueError(
                f"Expected a field of type {self.field_type_in!r}, got "
                f"{x.field_type!r}."
            )

        n_coarse = x.batch_shape[-1]

        # Fine nodes [0, n_coarse) are inherited (same positions as the coarse
        # nodes); [n_coarse, n_fine) are midpoints with exactly two parents,
        # x~_i = (1/2) sum_{j in parents(i)} rho(g_{i<-j}) x_j. Transport
        # first: the parents' coefficients are in their own frames.
        source = gather_to_edges(x, upsample_edges.source_index)  # (n_edges,)
        transported = transport_field(source, upsample_edges)  # target gauge

        averaged: dict[int, Tensor] = {
            frequency: scatter_add_to_nodes(0.5 * tensor, upsample_edges.row_ptr)
            for frequency, tensor in transported.coefficients.items()
        }  # (n_fine, a_n[, 2]); rows below n_coarse are zero

        # Inherited nodes take their coarse value unchanged: identical frames,
        # so no transport. upsample_edges carries no edges into that region,
        # so the concatenation is a partition rather than a sum.
        upsampled = SO2FeatureField(
            self.field_type_in,
            {
                frequency: torch.cat(
                    (x[frequency], averaged[frequency][n_coarse:]), dim=0
                )  # (n_fine, a_n[, 2])
                for frequency in self.field_type_in.frequencies
            },
        )

        # Learned fine-grid refinement, which also performs the type change.
        return self.conv(upsampled, fine_edges)  # type_out, (n_fine,)

    def _interpolate(
        self, x: SO2FeatureField, upsample_edges: EdgeGeometry
    ) -> SO2FeatureField:
        """The parent average and inherited-node concatenation of
        :meth:`forward`, without the refinement convolution."""
        n_coarse = x.batch_shape[0]
        source = gather_to_edges(x, upsample_edges.source_index)
        transported = transport_field(source, upsample_edges)
        averaged: dict[int, Tensor] = {
            frequency: scatter_add_to_nodes(0.5 * tensor, upsample_edges.row_ptr)
            for frequency, tensor in transported.coefficients.items()
        }
        return SO2FeatureField(
            self.field_type_in,
            {
                frequency: torch.cat(
                    (x[frequency], averaged[frequency][n_coarse:]), dim=0
                )
                for frequency in self.field_type_in.frequencies
            },
        )

    def forward_chunked(
        self,
        x: SO2FeatureField,
        upsample_edges: EdgeGeometry,
        fine_edges: EdgeGeometry | SphericalDiscretization,
        *,
        chunk_size: int,
    ) -> SO2FeatureField:
        """:meth:`forward` with the fine-grid convolution chunked.

        The parent interpolation, two edges per new fine node, is not
        chunked.
        """
        upsampled = self._interpolate(x, upsample_edges)
        return self.conv.forward_chunked(upsampled, fine_edges, chunk_size=chunk_size)

    def extra_repr(self) -> str:
        return f"field_type_in={self.field_type_in!r}, field_type_out={self.field_type_out!r}"


class EquiEncode(nn.Module):
    """Gauge-equivariant fine -> coarse restriction between two arbitrary grids.

    The cross-grid counterpart of :class:`IcoEncode`, for a pair of grids
    that share no indexing (an equiangular ``92 x 184`` onto ``46 x 92``,
    say). The computation is :class:`IcoEncode`'s, a steerable-kernel
    integral over bipartite edges that reads nothing grid-specific from its
    ``EdgeGeometry``; the edges come from
    :func:`~gm_attention.discretization.grids.equiangular.build_grid_hierarchy`.

    Parameters
    ----------
    field_type_in, field_type_out
        Fine-grid input and coarse-grid output types.
    cutoff
        Aggregation radius; must equal the radius ``down_edges`` was built with.
    mode
        ``"learned_integral"``, the only accepted value.
    window, n_radial_basis, radial_hidden_features, radial_basis, bias, dtype, device
        As :class:`IcoEncode`.
    """

    def __init__(
        self,
        field_type_in: SO2FieldType,
        field_type_out: SO2FieldType,
        *,
        cutoff: float,
        mode: Literal["learned_integral"] = "learned_integral",
        window: WindowKind = "quartic",
        n_radial_basis: int = 8,
        radial_hidden_features: Sequence[int] = (16,),
        radial_basis: RadialBasisKind = "gaussian",
        bias: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        if mode != "learned_integral":
            raise ValueError(
                f"mode must be 'learned_integral', got {mode!r}."
            )

        self.field_type_in = field_type_in
        self.field_type_out = field_type_out
        self.mode = mode
        self.cutoff = float(cutoff)

        self.integral = IcoEncode(
            field_type_in, field_type_out, cutoff=cutoff, window=window,
            n_radial_basis=n_radial_basis,
            radial_hidden_features=radial_hidden_features,
            radial_basis=radial_basis,
            bias=bias, dtype=dtype, device=device,
        )

    def forward(
        self,
        x: SO2FeatureField,
        down_edges: EdgeGeometry,
    ) -> SO2FeatureField:
        """Restrict ``x`` from the fine grid onto the coarse one."""
        return self.integral(x, down_edges)

    def extra_repr(self) -> str:
        return (
            f"{self.field_type_in!r} -> {self.field_type_out!r}, "
            f"mode={self.mode!r}, cutoff={self.cutoff:.6g}"
        )

