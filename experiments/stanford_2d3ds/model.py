"""The gauge-equivariant SegFormer: the released spherical SegFormer, component by component.

``torch_harmonics``' ``SphericalSegformer`` is a four-stage hierarchical
encoder with a lightweight all-MLP decoder. Stage ``i`` is

    OverlapPatchMerging  : DISCO conv (previous grid -> this grid) + LayerNorm
    depths[i] x [ LayerNorm -> NeighborhoodAttentionS2 -> (+ residual, DropPath)
                  LayerNorm -> 1x1 -> depthwise DISCO -> GELU -> 1x1 -> (+ residual, DropPath) ]
    LayerNorm

and the decoder applies, per stage, a ``1x1`` map followed by bilinear
resampling back to the input grid, concatenates the four results, and reads
the class logits off with one more ``1x1``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, TypedDict, cast

import torch
from torch import Tensor, nn

from gm_attention.discretization.discretization import SphericalDiscretization
from gm_attention.discretization.grids.equiangular import (
    TensorGridHierarchy,
    TensorGridLevel,
)
from gm_attention.nn.attention import SO2MHSA, ValueKind
from gm_attention.nn.blocks import (
    EquiEncode,
    IcoLowPassRestriction,
    NonlinearityKind,
    build_nonlinearity,
)
from gm_attention.nn.convolution import SO2Convolution, WindowKind
from gm_attention.nn.kernels import RadialBasisKind
from gm_attention.nn.linear import SO2Linear
from gm_attention.nn.normalization import SO2LayerNorm
from gm_attention.nn.stochastic_depth import SO2DropPath
from gm_attention.representations.field_type import (
    SO2FeatureField,
    SO2FieldType,
    scale_field_type,
)

ReadoutKind = Literal["convolution"]
DecoderKind = Literal["fusion"]
StemKind = Literal["pointwise"]

#: Legendre polynomials P_1 .. P_4 of <p, e_z>: the vertical conditioning's features.
_N_VERTICAL_FEATURES = 4


def vertical_features(nodes: Tensor) -> Tensor:  # (n_nodes, 4)
    """``P_1(z) .. P_4(z)`` for ``z = <p, e_z>`` of unit-sphere ``nodes`` -- an
    orthogonal basis on ``[-1, 1]`` of the functions of latitude alone, so a
    linear map of them is a smooth latitude profile per scalar channel."""
    z = nodes[:, 2]
    z2 = z * z
    return torch.stack(
        [z, 0.5 * (3.0 * z2 - 1.0), 0.5 * (5.0 * z2 - 3.0) * z, 0.125 * (35.0 * z2 * z2 - 30.0 * z2 + 3.0)],
        dim=-1,
    )


#: The tangent gravity field: a ``rho_1`` copy (the tangent part of ``e_z``) and a ``rho_2`` copy (its complex square).
GRAVITY_FIELD_TYPE = SO2FieldType({1: 1, 2: 1})


def gravity_features(nodes: Tensor, frames: Tensor) -> dict[int, Tensor]:
    """The tangent part of gravity in each node's frame, as ``rho_1`` and
    ``rho_2`` coefficients.

    ``e_z`` at ``p`` has the normal component ``z = <p, e_z>`` (a scalar,
    carried by :func:`vertical_features`) and the tangent component
    ``g = e_z - z p``, of length ``sqrt(1 - z^2)``, pointing at the zenith.
    Its frame coordinates ``(<g, e_1>, <g, e_2>)`` are a ``rho_1`` copy:
    under the regauging ``E' = E r(theta)`` they pick up ``r(-theta)``.
    Their complex square ``(c_1 + i c_2)^2`` picks up ``r(-2 theta)`` and is
    the ``rho_2`` copy, the vertical as an undirected axis. Both vanish at
    the poles, so the field is smooth everywhere.

    Parameters
    ----------
    nodes
        Unit-sphere nodes, ``(n_nodes, 3)``.
    frames
        Their oriented orthonormal tangent frames, ``(n_nodes, 3, 2)``.

    Returns
    -------
    dict
        ``{1: (n_nodes, 1, 2), 2: (n_nodes, 1, 2)}`` coefficient tensors.
    """
    z = nodes[:, 2:3]  # (n_nodes, 1)
    tangent = torch.zeros_like(nodes)
    tangent[:, 2] = 1.0
    tangent = tangent - z * nodes  # e_z - <p, e_z> p, (n_nodes, 3)
    c = torch.einsum("nk,nkd->nd", tangent, frames)  # (n_nodes, 2)
    square = torch.stack([c[:, 0] * c[:, 0] - c[:, 1] * c[:, 1], 2.0 * c[:, 0] * c[:, 1]], dim=-1)
    return {1: c.unsqueeze(-2), 2: square.unsqueeze(-2)}


def gravity_invariants(feature: SO2FeatureField, field: SO2FeatureField) -> Tensor:
    """Every ``rho_n`` copy of ``feature`` read against the gravity field's
    ``rho_n`` copy: the real and imaginary parts of ``f_c * conj(g_n)``, the
    inner product ``<f_c, g_n>`` and the cross product ``f_c x g_n``. Both
    factors pick up the same phase under a regauging, so the products are
    invariant scalars: the feature's component along the vertical and
    across it, with ``|g_n|`` as the scale (vanishing at the poles). Shape
    ``(n_nodes, 2 * sum_n m_n)`` over the frequencies the two share, in the
    order of ``feature``'s frequencies."""
    parts = []
    for frequency in feature.field_type.frequencies:
        if frequency == 0 or frequency not in field.field_type:
            continue
        f = torch.view_as_complex(feature[frequency].contiguous())  # (n, m)
        g = torch.view_as_complex(field[frequency].contiguous())  # (n, 1)
        parts.append(torch.view_as_real(f * g.conj()).flatten(-2))  # (n, 2m)
    return torch.cat(parts, dim=-1)


def _gravity_directional(field_type: SO2FieldType) -> dict[int, int]:
    """The ``rho_1`` and ``rho_2`` multiplicities of ``field_type`` (the
    frequencies the gravity field has)."""
    return {f: field_type.multiplicity(f) for f in GRAVITY_FIELD_TYPE.frequencies if f in field_type}


def _gravity_copies(field_type: SO2FieldType) -> int:
    """How many copies of ``field_type`` the gravity field can read."""
    return sum(_gravity_directional(field_type).values())


def directional_norms(feature: SO2FeatureField) -> Tensor:
    """The squared norm of every ``rho_n`` copy, ``n >= 1``: invariant
    scalars, shape ``(n_nodes, sum_{n >= 1} m_n)`` in the order of the
    frequencies."""
    return torch.cat([(feature[f] * feature[f]).sum(-1) for f in feature.field_type.frequencies if f != 0], dim=-1)


class _KernelSettings(TypedDict):
    """The steerable-kernel parameterization every kernel of the model shares."""

    n_radial_basis: int
    radial_hidden_features: Sequence[int]
    radial_basis: RadialBasisKind
    bias: bool
    dtype: torch.dtype | None
    device: torch.device | str | None


def add_fields(a: SO2FeatureField, b: SO2FeatureField) -> SO2FeatureField:
    """Residual sum of two same-typed fields."""
    if a.field_type != b.field_type:
        raise ValueError(f"cannot add fields of types {a.field_type!r} and {b.field_type!r}.")
    return SO2FeatureField(
        a.field_type,
        {frequency: a[frequency] + b[frequency] for frequency in a.field_type.frequencies},
    )


class GmSegformerBlock(nn.Module):
    """One attention + Mix-FFN pair of a stage, pre-norm, with drop path.

    Mirrors ``AttentionWrapper(pre_norm=True)`` followed by ``MixFFN``:

        x = x + Drop(MHSA(LN(x)))
        x = x + Drop(W_out sigma(DWConv(W_in LN(x))))

    With ``ffn_expansion=1`` the Mix-FFN has the reference's width (its
    ``use_mlp=False`` path maps channels to channels with no expansion)
    and its operator order.
    """

    def __init__(
        self,
        field_type: SO2FieldType,
        num_heads: int,
        *,
        attention_cutoff: float,
        mix_cutoff: float,
        window: WindowKind,
        nonlinearity: NonlinearityKind,
        drop_path: float,
        n_radial_basis: int,
        radial_hidden_features: Sequence[int],
        radial_basis: RadialBasisKind,
        bias: bool,
        attention_batched: bool = False,
        value_kind: ValueKind = "steerable",
        ffn_expansion: int = 1,
        eps: float = 1e-5,
        normalized_mix: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.field_type = field_type
        self.norm1 = SO2LayerNorm(field_type, eps=eps, bias=bias, dtype=dtype, device=device)
        self.attention = SO2MHSA(
            field_type, num_heads, cutoff=attention_cutoff, window=window,
            n_radial_basis=n_radial_basis, radial_hidden_features=radial_hidden_features,
            radial_basis=radial_basis, bias=bias, batched=attention_batched,
            value_kind=value_kind, dtype=dtype, device=device,
        )
        self.drop_path1 = SO2DropPath(drop_path)
        self.norm2 = SO2LayerNorm(field_type, eps=eps, bias=bias, dtype=dtype, device=device)
        if ffn_expansion < 1:
            raise ValueError(f"ffn_expansion must be at least 1, got {ffn_expansion}.")
        # The Mix-FFN's hidden type: ffn_expansion copies of the stage type.
        wide = scale_field_type(dict(field_type.items()), int(ffn_expansion))
        self.hidden_field_type = wide
        self.linear_in = SO2Linear(field_type, wide, bias=bias, dtype=dtype, device=device)
        self.mix = SO2Convolution(
            wide, wide, cutoff=mix_cutoff, window=window,
            self_interaction="none", depthwise=True, normalize=normalized_mix,
            n_radial_basis=n_radial_basis, radial_hidden_features=radial_hidden_features,
            radial_basis=radial_basis, bias=bias, dtype=dtype, device=device,
        )
        self.activation = build_nonlinearity(nonlinearity, wide, dtype=dtype, device=device)
        self.linear_out = SO2Linear(wide, field_type, bias=bias, dtype=dtype, device=device)
        self.drop_path2 = SO2DropPath(drop_path)

    def forward(
        self,
        x: SO2FeatureField,
        discretization: SphericalDiscretization,
    ) -> SO2FeatureField:
        normed = self.norm1(x)
        attended = self.attention(normed, discretization)
        x = add_fields(x, self.drop_path1(attended))
        hidden = self.linear_in(self.norm2(x))
        hidden = self.mix(hidden, discretization)
        hidden = self.activation(hidden)
        hidden = self.linear_out(hidden)
        return add_fields(x, self.drop_path2(hidden))


class GmSegformer(nn.Module):
    """The gauge-equivariant SegFormer; see the module docstring for the map.

    Parameters
    ----------
    in_channels
        Scalar input channels (RGB: 3), read as ``{0: in_channels}``.
    num_classes
        Output logits, ``{0: num_classes}`` on the input grid.
    base_field_type, multiplicities
        Stage ``i`` works at ``base_field_type`` scaled by
        ``multiplicities[i]``; every resulting multiplicity must be divisible
        by ``heads[i]``.
    heads, depths
        Attention heads and block count per stage, as the reference.
    merge_cutoffs, attention_cutoffs, mix_cutoffs, interpolate_cutoffs
        One radius per stage for the patch merging, the attention, the
        depthwise convolution and the interpolation back to the input grid,
        taken from the grid chain (``geometry.segformer_radii``) rather
        than configured.
    window
        Aggregation window of every steerable operator.
    nonlinearity
        The Mix-FFN activation, in place of GELU.
    drop_path_rate
        Final stochastic-depth rate; ramps linearly from 0 over all
        ``sum(depths)`` blocks, the attention and FFN branch of one block
        sharing a rate, exactly as the reference.
    readout
        ``"convolution"``, the only value: each stage's decoder projection
        is a steerable convolution on the stage's own grid from the stage
        type onto ``readout_channels[i]`` scalars, with a linear
        self-interaction; only scalars are then interpolated, and the head
        is an intertwiner over every fused scalar channel, as the released
        head's ``1x1``.
    readout_cutoffs
        Per-stage support radius of the convolutional readout. Each stage
        level's discretization must be prepared at its radius.
    readout_channels
        Per-stage scalar width of the convolutional readout; ``None`` gives
        every stage as many scalar channels as it has irrep copies.
    attention_batched
        Evaluate every attention layer's heads in one pass
        (:class:`~gm_attention.nn.attention.SO2MHSA`'s ``batched``), the
        same function.
    value_kind
        How every attention head forms its value messages
        (:data:`~gm_attention.nn.attention.ValueKind`): ``"steerable"``,
        one position-dependent steerable kernel from the stage type onto
        the head's slice.
    ffn_expansion
        Width of the Mix-FFN's hidden type in copies of the stage type; the
        reference's ``use_mlp=False`` path is 1.
    final_stage_scalars
        If given, the last stage's type is ``{0: final_stage_scalars}``
        instead of the scaled base type: a purely invariant coarsest stage.
        The merge into it is the type-changing steerable integral as for
        every stage.
    stage_field_types
        One field type per stage, given directly (multiplicity 1 of each),
        in place of ``base_field_type`` times ``multiplicities``. Exclusive
        with ``final_stage_scalars``.
    gravity_input
        Append the tangent part of gravity to the input beside the RGB, as
        one ``rho_1`` and one ``rho_2`` channel (:func:`gravity_features`),
        so the first merge's steerable kernel can form products of image
        gradients with the vertical: the scalar-only ablation's way of
        seeing the vertical, since ``gravity_conditioning`` needs
        directional channels.
    vertical_conditioning
        Add, after each stage's merge normalization, a learned linear map of
        four Legendre polynomials of the stage nodes' ``<p, e_z>`` to the
        stage's scalar channels (zero-initialized, so training starts from
        the unconditioned model).
    gravity_conditioning
        After each stage's merge normalization, a zero-initialized
        equivariant linear map of the stage nodes' gravity field
        (:func:`gravity_features`) into the stage's ``rho_1`` and ``rho_2``
        channels (the frequencies it has); the frequency-1 and -2
        counterpart of ``vertical_conditioning``. A scalar-only stage has
        nothing to emit into and is skipped; a model with no directional
        stage at all is refused. This is the emitting half of the
        conditioning: it adds ``W g`` to the directional channels; the
        scalar gate then forms ``sigma(s) W g``, a gravity-aligned vector
        of image-dependent magnitude, which the next steerable kernel's
        spatial harmonics turn into the derivative of ``s`` along the
        vertical.
    gravity_projection
        The reading half: after each stage's merge normalization, the
        invariants of the stage's ``rho_1`` and ``rho_2`` copies against the
        gravity field (:func:`gravity_invariants`: each copy's component
        along the vertical and across it) through a zero-initialized linear
        map into the stage's scalar channels.

        At a scalar-only stage (``final_stage_scalars``) the read moves to
        the merge's input: the previous stage's completed ``rho_1`` and
        ``rho_2`` copies are read against the gravity field of the source
        level and added to its scalars before the merge integrates them. A
        scalar-only stage whose source has no copies (or no scalars) is
        skipped; a model where no stage can be read is refused.
    gravity_readout
        The reading half at the end of every stage: a pointwise residual
        network after each convolutional readout, fed the readout ``r_i``,
        the four Legendre polynomials of ``<p, e_z>``, the completed stage's
        ``rho_1`` and ``rho_2`` copies read against the gravity field (inner
        and cross products) and their squared norms; one hidden layer of
        ``gravity_readout_expansion`` times the readout width, GELU, and a
        zero-initialized output added to ``r_i`` (the readout keeps its
        width and its spatial evidence).
    gravity_readout_expansion
        Hidden width of that network in multiples of the readout width.
    normalized_mix, normalized_readout
        ``normalize`` of the depthwise mix and of the readout convolutions:
        each target's integral is divided by its window mass, a weighted
        mean over the neighbourhood instead of a Riemann sum.
    head_hidden
        If given, the head is a scalar MLP with this hidden width (GELU) on
        the fused channels and the stem together, in place of the ``1x1``
        intertwiner.
    decoder
        ``"fusion"``, the only value: a top-down path on the stage grids. The
        coarsest readout is mapped to ``decoder_channels`` scalars; at each
        finer stage the fused channels are interpolated onto that stage's
        grid (``hierarchy.lateral_edges``), concatenated with the stage's
        own readout and mapped back to ``decoder_channels`` through a GELU;
        the finest fused channels are interpolated to the input grid,
        joined by the stem if any, passed through ``decoder_dropout`` and
        read out. Everything after the readout convolutions is scalar, so
        the maps are plain linear layers.
    decoder_channels, decoder_dropout
        The fused width and the dropout rate before the head.
    stem_channels, stem_kind, stem_cutoff
        ``stem_channels`` scalars of the input on the input grid, joined to
        the decoder's output before the head. ``stem_kind`` is
        ``"pointwise"``, the only value: a linear map of each node's input
        channels through a GELU, scalars in, scalars out. ``stem_cutoff``
        is accepted (the experiment passes ``radii.stem``) and unused by
        the pointwise stem.
    gradient_checkpointing
        Must be ``False``: every block, merge and interpolation runs
        directly.
    evaluation_chunk_size
        Must be ``None``: the attention is never chunked.
    n_radial_basis, radial_hidden_features, radial_basis, bias
        The steerable-kernel parameterization, shared by every kernel.
    eps
        Epsilon of every ``SO2LayerNorm``.
    """

    def __init__(
        self,
        *,
        in_channels: int = 3,
        num_classes: int,
        base_field_type: Mapping[int, int] | Mapping[str, int],
        multiplicities: Sequence[int],
        heads: Sequence[int],
        depths: Sequence[int],
        merge_cutoffs: Sequence[float],
        attention_cutoffs: Sequence[float],
        mix_cutoffs: Sequence[float],
        interpolate_cutoffs: Sequence[float],
        window: WindowKind = "hard",
        nonlinearity: NonlinearityKind = "scalar_gate_double_sigmoid",
        drop_path_rate: float = 0.1,
        readout: ReadoutKind = "convolution",
        readout_cutoffs: Sequence[float] | None = None,
        readout_channels: Sequence[int] | None = None,
        attention_batched: bool = False,
        value_kind: ValueKind = "steerable",
        ffn_expansion: int = 1,
        final_stage_scalars: int | None = None,
        stage_field_types: Sequence[Mapping[int, int] | Mapping[str, int]] | None = None,
        vertical_conditioning: bool = False,
        gravity_input: bool = False,
        gravity_conditioning: bool = False,
        gravity_projection: bool = False,
        gravity_readout: bool = False,
        gravity_readout_expansion: int = 2,
        normalized_mix: bool = False,
        normalized_readout: bool = False,
        head_hidden: int | None = None,
        decoder: DecoderKind = "fusion",
        decoder_channels: int = 32,
        decoder_dropout: float = 0.0,
        stem_channels: int = 0,
        stem_kind: StemKind = "pointwise",
        stem_cutoff: float | None = None,
        gradient_checkpointing: bool = False,
        evaluation_chunk_size: int | None = None,
        n_radial_basis: int = 8,
        radial_hidden_features: Sequence[int] = (16,),
        radial_basis: RadialBasisKind = "gaussian",
        bias: bool = True,
        eps: float = 1e-5,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        num_stages = len(multiplicities)
        if num_stages < 1:
            raise ValueError("at least one stage is required.")
        for name, values in (
            ("heads", heads), ("depths", depths), ("merge_cutoffs", merge_cutoffs),
            ("attention_cutoffs", attention_cutoffs), ("mix_cutoffs", mix_cutoffs),
            ("interpolate_cutoffs", interpolate_cutoffs),
        ):
            if len(values) != num_stages:
                raise ValueError(
                    f"{name} needs one entry per stage ({num_stages}), got {len(values)}."
                )
        if in_channels < 1 or num_classes < 1:
            raise ValueError("in_channels and num_classes must be positive.")
        if any(d < 1 for d in depths):
            raise ValueError(f"every stage needs at least one block, got depths={list(depths)}.")
        if readout != "convolution":
            raise ValueError(f"readout must be 'convolution', got {readout!r}.")
        if readout_cutoffs is None or len(readout_cutoffs) != num_stages:
            raise ValueError(
                "readout='convolution' needs one readout_cutoffs entry per stage "
                f"({num_stages}), got {readout_cutoffs!r}."
            )
        if readout_channels is not None and len(readout_channels) != num_stages:
            raise ValueError(
                f"readout_channels needs one entry per stage ({num_stages}), got {len(readout_channels)}."
            )
        if not (0.0 <= drop_path_rate < 1.0):
            raise ValueError(f"drop_path_rate must satisfy 0 <= p < 1, got {drop_path_rate}.")
        if evaluation_chunk_size is not None:
            raise ValueError(f"evaluation_chunk_size must be None (the attention is never chunked), got {evaluation_chunk_size!r}.")
        if decoder != "fusion":
            raise ValueError(f"decoder must be 'fusion', got {decoder!r}.")
        if decoder_channels < 1:
            raise ValueError(f"decoder_channels must be positive, got {decoder_channels}.")
        if not (0.0 <= decoder_dropout < 1.0):
            raise ValueError(f"decoder_dropout must satisfy 0 <= p < 1, got {decoder_dropout}.")
        if stem_channels < 0:
            raise ValueError(f"stem_channels must be non-negative, got {stem_channels}.")
        if stem_kind != "pointwise":
            raise ValueError(f"stem_kind must be 'pointwise', got {stem_kind!r}.")
        if gradient_checkpointing:
            raise ValueError("gradient_checkpointing must be False (blocks, merges and interpolations run directly).")
        if final_stage_scalars is not None and final_stage_scalars < 1:
            raise ValueError(f"final_stage_scalars must be positive if given, got {final_stage_scalars}.")

        self.input_field_type = SO2FieldType({0: int(in_channels)})
        self.gravity_input = bool(gravity_input)
        # What the first merge reads: the input plus, under gravity_input,
        # the tangent gravity field (rho_1 + rho_2).
        encoder_input: dict[int, int] = {0: int(in_channels)}
        if gravity_input:
            for frequency in GRAVITY_FIELD_TYPE.frequencies:
                encoder_input[frequency] = GRAVITY_FIELD_TYPE.multiplicity(frequency)
        self.encoder_input_type = SO2FieldType(encoder_input)
        self.output_field_type = SO2FieldType({0: int(num_classes)})
        stage_types = [scale_field_type(base_field_type, int(m)) for m in multiplicities]
        if stage_field_types is not None:
            if final_stage_scalars is not None:
                raise ValueError("pass stage_field_types or final_stage_scalars, not both.")
            if len(stage_field_types) != num_stages:
                raise ValueError(
                    f"stage_field_types needs one type per stage ({num_stages}), got {len(stage_field_types)}."
                )
            stage_types = [scale_field_type(t, 1) for t in stage_field_types]
        if final_stage_scalars is not None:
            stage_types[-1] = SO2FieldType({0: int(final_stage_scalars)})
        self.stage_field_types: tuple[SO2FieldType, ...] = tuple(stage_types)
        self.final_stage_scalars = None if final_stage_scalars is None else int(final_stage_scalars)
        self.ffn_expansion = int(ffn_expansion)
        self.vertical_conditioning = bool(vertical_conditioning)
        self.gravity_conditioning = bool(gravity_conditioning)
        self.gravity_projection = bool(gravity_projection)
        self.gravity_readout = bool(gravity_readout)
        self.gravity_readout_expansion = int(gravity_readout_expansion)
        self.normalized_mix = bool(normalized_mix)
        self.normalized_readout = bool(normalized_readout)
        self.head_hidden = None if head_hidden is None else int(head_hidden)
        if self.head_hidden is not None and self.head_hidden < 1:
            raise ValueError(f"head_hidden must be positive, got {head_hidden}.")
        if gravity_readout and self.gravity_readout_expansion < 1:
            raise ValueError(f"gravity_readout_expansion must be positive, got {gravity_readout_expansion}.")
        self.decoder: DecoderKind = decoder
        self.decoder_channels = int(decoder_channels)
        self.stem_channels = int(stem_channels)
        self.stem_kind: StemKind = stem_kind
        self.stem_cutoff = None if stem_cutoff is None else float(stem_cutoff)
        self.heads = tuple(int(h) for h in heads)
        self.depths = tuple(int(d) for d in depths)
        self.readout: ReadoutKind = readout
        self.value_kind: ValueKind = value_kind
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.evaluation_chunk_size = evaluation_chunk_size
        self.merge_cutoffs = tuple(float(c) for c in merge_cutoffs)
        self.attention_cutoffs = tuple(float(c) for c in attention_cutoffs)
        self.mix_cutoffs = tuple(float(c) for c in mix_cutoffs)
        self.interpolate_cutoffs = tuple(float(c) for c in interpolate_cutoffs)

        kernel = _KernelSettings(
            n_radial_basis=n_radial_basis, radial_hidden_features=radial_hidden_features,
            radial_basis=radial_basis, bias=bias, dtype=dtype, device=device,
        )

        # The reference's stochastic-depth schedule: linspace(0, rate) over
        # every block, one rate per block shared by its two branches.
        rates = [float(r) for r in torch.linspace(0.0, drop_path_rate, sum(self.depths))]

        self.merges = nn.ModuleList()
        self.merge_norms = nn.ModuleList()
        self.stages = nn.ModuleList()
        self.stage_norms = nn.ModuleList()
        previous = self.encoder_input_type
        cursor = 0
        for i, field_type in enumerate(self.stage_field_types):
            self.merges.append(
                EquiEncode(
                    previous, field_type, cutoff=self.merge_cutoffs[i],
                    mode="learned_integral", window=window, **kernel,
                )
            )
            self.merge_norms.append(
                SO2LayerNorm(field_type, eps=eps, bias=bias, dtype=dtype, device=device)
            )
            self.stages.append(
                nn.ModuleList(
                    [
                        GmSegformerBlock(
                            field_type, self.heads[i],
                            attention_cutoff=self.attention_cutoffs[i],
                            mix_cutoff=self.mix_cutoffs[i], window=window,
                            normalized_mix=self.normalized_mix,
                            nonlinearity=nonlinearity, drop_path=rates[cursor + j],
                            attention_batched=attention_batched, value_kind=value_kind,
                            ffn_expansion=ffn_expansion, eps=eps, **kernel,
                        )
                        for j in range(self.depths[i])
                    ]
                )
            )
            self.stage_norms.append(
                SO2LayerNorm(field_type, eps=eps, bias=bias, dtype=dtype, device=device)
            )
            cursor += self.depths[i]
            previous = field_type

        # Vertical conditioning: Legendre features of <p, e_z> mapped to the
        # stage's scalar channels; zero-initialized.
        self.conditioners = nn.ModuleList()
        if vertical_conditioning:
            for i, field_type in enumerate(self.stage_field_types):
                scalars = field_type.multiplicity(0) if 0 in field_type else 0
                if scalars < 1:
                    raise ValueError(f"vertical_conditioning needs scalar channels; stage {i + 1} has none.")
                linear = nn.Linear(_N_VERTICAL_FEATURES, scalars, dtype=dtype, device=device)
                nn.init.zeros_(linear.weight)
                nn.init.zeros_(linear.bias)
                self.conditioners.append(linear)
        # Gravity conditioning into the rho_1 and rho_2 channels, zero-initialized;
        # keyed by stage index, a scalar-only stage having none.
        self.gravity_conditioners = nn.ModuleDict()
        if gravity_conditioning:
            for i, field_type in enumerate(self.stage_field_types):
                directional = _gravity_directional(field_type)
                if not directional:
                    continue  # a scalar-only stage: nothing to emit into
                linear = SO2Linear(GRAVITY_FIELD_TYPE, SO2FieldType(directional), bias=False, dtype=dtype, device=device)
                for weight in linear.weight.values():
                    nn.init.zeros_(weight)
                self.gravity_conditioners[str(i)] = linear
            if not self.gravity_conditioners:
                raise ValueError("gravity_conditioning needs rho_1 or rho_2 channels at some stage; every stage is scalar-only.")
        # Gravity projection: invariants against the gravity field mapped to scalars,
        # zero-initialized; read at the stage or, if scalar-only, at its merge's input.
        self.gravity_projectors = nn.ModuleDict()
        self.gravity_projection_sites: dict[int, Literal["stage", "source"]] = {}
        if gravity_projection:
            for i, field_type in enumerate(self.stage_field_types):
                site: Literal["stage", "source"] | None = None
                read_type = field_type
                if _gravity_copies(field_type) and 0 in field_type:
                    site = "stage"
                elif not _gravity_copies(field_type) and i > 0:
                    source_type = self.stage_field_types[i - 1]
                    if _gravity_copies(source_type) and (0 in source_type):
                        site, read_type = "source", source_type
                if site is None:
                    continue
                linear = nn.Linear(2 * _gravity_copies(read_type), read_type.multiplicity(0), dtype=dtype, device=device)
                nn.init.zeros_(linear.weight)
                nn.init.zeros_(linear.bias)
                self.gravity_projectors[str(i)] = linear
                self.gravity_projection_sites[i] = site
            if not self.gravity_projectors:
                raise ValueError(
                    "gravity_projection needs a stage with rho_1 or rho_2 channels and scalar channels, "
                    "or a scalar-only stage merging from one; none has."
                )
        self.readout_cutoffs: tuple[float, ...] = tuple(float(c) for c in readout_cutoffs)
        # Per-stage readout: a steerable convolution onto scalars on the stage's own grid.
        projected_types: list[SO2FieldType] = []
        self.upsample_linears = nn.ModuleList()
        for i, field_type in enumerate(self.stage_field_types):
            channels = (
                int(readout_channels[i]) if readout_channels is not None
                else sum(field_type.multiplicity(f) for f in field_type.frequencies)
            )
            if channels < 1:
                raise ValueError(f"readout_channels[{i}] must be positive, got {channels}.")
            scalars = SO2FieldType({0: channels})
            projected_types.append(scalars)
            self.upsample_linears.append(
                SO2Convolution(
                    field_type, scalars, cutoff=self.readout_cutoffs[i], window=window,
                    self_interaction="linear", normalize=self.normalized_readout, **kernel,
                )
            )
        self.projected_field_types: tuple[SO2FieldType, ...] = tuple(projected_types)
        # Gravity readout: a pointwise residual network per stage on
        # [readout, Legendre(z), gravity invariants, squared norms].
        self.readout_heads = nn.ModuleList()
        if gravity_readout:
            for i, (field_type, projected_type) in enumerate(zip(self.stage_field_types, self.projected_field_types, strict=True)):
                channels = projected_type.multiplicity(0)
                directional = sum(field_type.multiplicity(f) for f in field_type.frequencies if f != 0)
                in_features = channels + _N_VERTICAL_FEATURES + 3 * directional
                hidden = self.gravity_readout_expansion * channels
                head = nn.Sequential(
                    nn.Linear(in_features, hidden, dtype=dtype, device=device),
                    nn.GELU(),
                    nn.Linear(hidden, channels, dtype=dtype, device=device),
                )
                nn.init.zeros_(cast(nn.Linear, head[2]).weight)
                nn.init.zeros_(cast(nn.Linear, head[2]).bias)
                self.readout_heads.append(head)
        # The top-down path interpolates stage to stage; only the finest
        # fused channels travel to the input grid.
        width = self.decoder_channels
        fused_type = SO2FieldType({0: width})
        self.fused_scalar_type = fused_type
        channels = [t.multiplicity(0) for t in self.projected_field_types]
        self.fusion_in = nn.Linear(channels[-1], width, dtype=dtype, device=device)
        # fusions[j] and lateral_interpolations[j] serve stage k = num_stages - 2 - j,
        # the order the top-down pass visits them.
        self.fusions = nn.ModuleList(
            [
                nn.Linear(channels[k] + width, width, dtype=dtype, device=device)
                for k in range(self.num_stages - 2, -1, -1)
            ]
        )
        self.lateral_interpolations = nn.ModuleList(
            [
                IcoLowPassRestriction(fused_type, cutoff=self.interpolate_cutoffs[k + 1], window="quartic")
                for k in range(self.num_stages - 2, -1, -1)
            ]
        )
        self.final_interpolation = IcoLowPassRestriction(
            fused_type, cutoff=self.interpolate_cutoffs[0], window="quartic"
        )
        self.stem: nn.Module | None = None
        if stem_channels:
            self.stem = nn.Linear(int(in_channels), stem_channels, dtype=dtype, device=device)
        self.decoder_dropout = nn.Dropout(decoder_dropout)
        head_in = SO2FieldType({0: width + stem_channels})
        self.head_field_type = head_in
        # The head: the 1x1 intertwiner or, with head_hidden, a scalar MLP on the fused channels and the stem.
        self.head: nn.Module
        if self.head_hidden is None:
            self.head = SO2Linear(head_in, self.output_field_type, bias=bias, dtype=dtype, device=device)
        else:
            self.head = nn.Sequential(
                nn.Linear(head_in.multiplicity(0), self.head_hidden, bias=bias, dtype=dtype, device=device),
                nn.GELU(),
                nn.Linear(self.head_hidden, num_classes, bias=bias, dtype=dtype, device=device),
            )

    @property
    def num_stages(self) -> int:
        return len(self.stage_field_types)

    def _rematerialize(self, module: nn.Module, *args: object) -> SO2FeatureField:
        """Run a module."""
        return module(*args)  # type: ignore[no-any-return]

    def _run_block(
        self, block: nn.Module, x: SO2FeatureField, discretization: SphericalDiscretization
    ) -> SO2FeatureField:
        return self._rematerialize(block, x, discretization)

    def encode(
        self, x: SO2FeatureField, hierarchy: TensorGridHierarchy
    ) -> list[SO2FeatureField]:
        """The per-stage features on their own grids, finest stage first."""
        if x.field_type != self.input_field_type:
            raise ValueError(
                f"Expected an input of type {self.input_field_type!r}, got {x.field_type!r}."
            )
        if hierarchy.depth != self.num_stages:
            raise ValueError(
                f"the hierarchy has {hierarchy.depth} hops but the model {self.num_stages} stages."
            )
        if x.batch_shape != (hierarchy.finest.n_nodes,):
            raise ValueError(
                f"x.batch_shape must be ({hierarchy.finest.n_nodes},), the finest level's "
                f"node count, got {tuple(x.batch_shape)}."
            )

        features: list[SO2FeatureField] = []
        feature = self._encoder_input(x, hierarchy.finest)
        for i in range(self.num_stages):
            level = hierarchy.levels[i + 1]
            if level.discretization is None:
                raise ValueError(f"hierarchy level {i + 1} carries no same-grid discretization.")
            if self.gravity_projection_sites.get(i) == "source":
                feature = self._read_gravity(i, feature, hierarchy.levels[i])
            feature = self._rematerialize(self.merges[i], feature, hierarchy.down_edges[i])
            feature = self.merge_norms[i](feature)
            if self.vertical_conditioning:
                feature = self._condition(i, feature, level)
            if str(i) in self.gravity_conditioners or self.gravity_projection_sites.get(i) == "stage":
                feature = self._condition_gravity(i, feature, level)
            for block in cast(nn.ModuleList, self.stages[i]):
                feature = self._run_block(block, feature, level.discretization)
            feature = self.stage_norms[i](feature)
            features.append(feature)
        return features

    def _encoder_input(self, x: SO2FeatureField, level: TensorGridLevel) -> SO2FeatureField:
        """``x`` as the first merge reads it: with the tangent gravity field
        appended as ``rho_1`` and ``rho_2`` channels under ``gravity_input``."""
        if not self.gravity_input:
            return x
        scalars = x[0]
        coefficients: dict[int, Tensor] = {0: scalars}
        for frequency, tensor in self._gravity(level, scalars).items():
            coefficients[frequency] = tensor
        return SO2FeatureField(self.encoder_input_type, coefficients)

    @staticmethod
    def _gravity(level: TensorGridLevel, like: Tensor) -> dict[int, Tensor]:
        """The level's tangent gravity field in ``like``'s dtype and device."""
        nodes = level.grid.nodes.to(dtype=like.dtype, device=like.device)
        frames = level.frames.to(dtype=like.dtype, device=like.device)
        return gravity_features(nodes, frames)

    def _condition_gravity(self, stage: int, feature: SO2FeatureField, level: TensorGridLevel) -> SO2FeatureField:
        """Both halves of the gravity conditioning from the same feature, at
        a stage with directional copies: the conditioner's equivariant map of
        the level nodes' gravity field added to the ``rho_1`` and ``rho_2``
        channels, and the projector's map of the feature's invariants against
        that field added to the scalars."""
        like = next(iter(feature.coefficients.values()))
        field = SO2FeatureField(GRAVITY_FIELD_TYPE, self._gravity(level, like))
        outputs = {f: feature[f] for f in feature.field_type.frequencies}
        key = str(stage)
        if key in self.gravity_conditioners:
            offsets = cast(SO2Linear, self.gravity_conditioners[key])(field)
            for f in offsets.field_type.frequencies:
                outputs[f] = outputs[f] + offsets[f]
        if self.gravity_projection_sites.get(stage) == "stage":
            projector = cast(nn.Linear, self.gravity_projectors[key])
            outputs[0] = outputs[0] + projector(gravity_invariants(feature, field))
        return SO2FeatureField(feature.field_type, outputs)

    def _read_gravity(self, stage: int, feature: SO2FeatureField, source: TensorGridLevel) -> SO2FeatureField:
        """The read of a scalar-only stage, taken at its merge's input: the
        previous stage's completed copies against the source level's gravity
        field, added to the source scalars the merge then integrates."""
        field = SO2FeatureField(GRAVITY_FIELD_TYPE, self._gravity(source, feature[0]))
        projector = cast(nn.Linear, self.gravity_projectors[str(stage)])
        outputs = {f: feature[f] for f in feature.field_type.frequencies}
        outputs[0] = outputs[0] + projector(gravity_invariants(feature, field))
        return SO2FeatureField(feature.field_type, outputs)

    def _condition(self, stage: int, feature: SO2FeatureField, level: TensorGridLevel) -> SO2FeatureField:
        """The stage's scalar channels plus the conditioner's map of the
        level nodes' vertical Legendre features."""
        scalars = feature[0]
        offsets = self.conditioners[stage](
            vertical_features(level.grid.nodes).to(dtype=scalars.dtype, device=scalars.device)
        )
        outputs = {f: (scalars + offsets if f == 0 else feature[f]) for f in feature.field_type.frequencies}
        return SO2FeatureField(feature.field_type, outputs)

    def _readouts(
        self, features: Sequence[SO2FeatureField], hierarchy: TensorGridHierarchy
    ) -> list[SO2FeatureField]:
        """Every stage's convolutional readout on its own grid: scalars."""
        projected: list[SO2FeatureField] = []
        for i, feature in enumerate(features):
            level = hierarchy.levels[i + 1]
            if level.discretization is None:
                raise ValueError(f"hierarchy level {i + 1} carries no same-grid discretization.")
            readout = self._rematerialize(self.upsample_linears[i], feature, level.discretization)
            if self.gravity_readout:
                readout = self._read_against_gravity(i, readout, feature, level)
            projected.append(readout)
        return projected

    def _read_against_gravity(
        self, stage: int, readout: SO2FeatureField, feature: SO2FeatureField, level: TensorGridLevel
    ) -> SO2FeatureField:
        """The readout plus the stage's pointwise head on [readout,
        Legendre(z), the completed stage's gravity invariants, its squared
        norms]."""
        scalars = readout[0]
        parts = [scalars, vertical_features(level.grid.nodes).to(dtype=scalars.dtype, device=scalars.device)]
        if any(f != 0 for f in feature.field_type.frequencies):
            field = SO2FeatureField(GRAVITY_FIELD_TYPE, self._gravity(level, scalars))
            parts.append(gravity_invariants(feature, field))
            parts.append(directional_norms(feature))
        offsets = self.readout_heads[stage](torch.cat(parts, dim=-1))
        return SO2FeatureField(readout.field_type, {0: scalars + offsets})

    def decode(
        self,
        features: Sequence[SO2FeatureField],
        hierarchy: TensorGridHierarchy,
        x: SO2FeatureField | None = None,
    ) -> SO2FeatureField:
        """Project every stage, fuse across stages, read out on the input grid.

        ``x`` is the model input, needed only by a stem (``stem_channels``).
        """
        readouts = self._readouts(features, hierarchy)
        if len(hierarchy.lateral_edges) != self.num_stages - 1:
            raise ValueError(
                f"decoder='fusion' needs {self.num_stages - 1} lateral edge sets; the hierarchy has "
                f"{len(hierarchy.lateral_edges)} (build it with lateral=True)."
            )
        gelu = torch.nn.functional.gelu
        fused_tensor = gelu(self.fusion_in(readouts[-1][0]))  # (n_coarsest, width)
        for j, k in enumerate(range(self.num_stages - 2, -1, -1)):
            carried = SO2FeatureField(self.fused_scalar_type, {0: fused_tensor})
            up = self._rematerialize(self.lateral_interpolations[j], carried, hierarchy.lateral_edges[k])
            fused_tensor = gelu(self.fusions[j](torch.cat([readouts[k][0], up[0]], dim=-1)))
        carried = SO2FeatureField(self.fused_scalar_type, {0: fused_tensor})
        fine = self._rematerialize(self.final_interpolation, carried, hierarchy.up_edges[0])[0]  # (n_input, width)
        if self.stem is not None:
            if x is None:
                raise ValueError("decode needs the model input x for the stem.")
            stem_out = gelu(self.stem(x[0]))  # (n_input, stem_channels)
            fine = torch.cat([fine, stem_out], dim=-1)
        fine = self.decoder_dropout(fine)
        if self.head_hidden is not None:
            return SO2FeatureField(self.output_field_type, {0: self.head(fine)})
        return self.head(SO2FeatureField(self.head_field_type, {0: fine}))

    def forward(self, x: SO2FeatureField, hierarchy: TensorGridHierarchy) -> SO2FeatureField:
        """Class logits, ``{0: num_classes}`` on the hierarchy's finest level.

        Parameters
        ----------
        x
            ``{0: in_channels}`` on ``hierarchy.finest``.
        hierarchy
            Built for this model's radii (``geometry.build_segformer_hierarchy``);
            its per-level radii must match the cutoffs the model was
            constructed with.
        """
        return self.decode(self.encode(x, hierarchy), hierarchy, x)

    def extra_repr(self) -> str:
        return (
            f"input={self.input_field_type!r}, stages={[repr(t) for t in self.stage_field_types]}, "
            f"heads={self.heads}, depths={self.depths}, output={self.output_field_type!r}, "
            f"readout={self.readout!r}, value_kind={self.value_kind!r}, "
            f"ffn_expansion={self.ffn_expansion}, "
            f"decoder={self.decoder!r}, stem_channels={self.stem_channels}, stem_kind={self.stem_kind!r}, "
            f"vertical_conditioning={self.vertical_conditioning}, gravity_input={self.gravity_input}, "
            f"gravity_conditioning={self.gravity_conditioning}, gravity_projection={self.gravity_projection}, "
            f"gravity_readout={self.gravity_readout}, normalized_mix={self.normalized_mix}, "
            f"normalized_readout={self.normalized_readout}, head_hidden={self.head_hidden}, "
            f"gradient_checkpointing={self.gradient_checkpointing}"
        )


def real_parameter_count(model: nn.Module) -> int:
    """Parameters as real degrees of freedom (a complex weight counts twice)."""
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())


__all__ = [
    "GRAVITY_FIELD_TYPE",
    "GmSegformer",
    "GmSegformerBlock",
    "add_fields",
    "directional_norms",
    "gravity_features",
    "gravity_invariants",
    "real_parameter_count",
    "vertical_features",
]

