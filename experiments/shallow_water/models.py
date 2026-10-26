"""Models for the shallow-water benchmark.

Two GM models and two reference builders, all consuming the conditioned
input ``(state, coriolis)`` and predicting the prognostic state:
:class:`GmAttentionDepth1` (encode, N transformer blocks and decode across
one icosphere hop, plus a zero-initialized fine skip),
:class:`GmAttentionSingleLevel` (convolution, N blocks and convolution on
one icosphere level), :func:`build_reference_transformer`
(``torch_harmonics``' ``SphericalTransformer``) and
:func:`build_sfno_reference` (its ``SphericalFourierNeuralOperator``).
Every GM model takes one graph per call: a feature field's ``batch_shape``
is ``(n_nodes,)`` and batching is handled by the training loop.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, cast

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch_harmonics.examples.models.s2transformer import SphericalTransformer
from torch_harmonics.examples.models.sfno import SphericalFourierNeuralOperator

from gm_attention.discretization.discretization import SphericalDiscretization
from gm_attention.nn.stencil_plan import StencilPlans
from gm_attention.discretization.grids.icosphere import IcosphereMultiresolution
from gm_attention.nn.blocks import (
    IcoDecode,
    IcoEncode,
    MixerKind,
    NonlinearityKind,
    SO2TransformerBlock,
)
from gm_attention.nn.convolution import (
    SelfInteractionKind,
    SO2Convolution,
    WindowKind,
)
from gm_attention.nn.kernels import RadialBasisKind
from gm_attention.nn.linear import SO2Linear
from gm_attention.representations.field_type import (
    SO2FeatureField,
    SO2FieldType,
    scale_field_type,
)

# The fine -> coarse path of the depth-1 hop: IcoEncode's gauge-steerable integral.
EncodeModeKind = Literal["learned_integral"]
# Initialization of the fine skip. The residual objective adds the model
# input outside the network, so the skip starts at zero.
FineSkipInit = Literal["zero"]


def _add_fields(a: SO2FeatureField, b: SO2FeatureField) -> SO2FeatureField:
    """Coefficient-wise sum of two same-typed fields in a shared gauge.

    Equivariant because both operands live in the same per-node gauge; not
    a general SO2FeatureField operator.
    """
    return SO2FeatureField(
        a.field_type,
        {frequency: a[frequency] + b[frequency] for frequency in a.field_type.frequencies},
    )


def _initialize_fine_skip(skip: SO2Linear, init: FineSkipInit) -> None:
    """Zero the weights and bias of a fine-skip ``SO2Linear``.

    The residual objective adds the model input outside the network; an
    identity-initialized skip would be a second identity path.
    """
    if init != "zero":
        raise ValueError(f"fine_skip_init must be 'zero', got {init!r}.")

    with torch.no_grad():
        for name, parameter in skip.weight.items():
            parameter.zero_()
            del name

        if skip.bias is not None:
            skip.bias.zero_()


#: Prognostic state: height as rho_0, velocity as one rho_1 copy.
PROGNOSTIC_FIELD_TYPE = SO2FieldType({0: 1, 1: 1})

#: Conditioned input: [normalized_height, coriolis_c] at frequency 0, velocity as rho_1.
CONDITIONED_FIELD_TYPE = SO2FieldType({0: 2, 1: 1})


def build_reference_transformer(
    *,
    nlat: int = 256,
    nlon: int = 512,
    in_chans: int = 4,
    out_chans: int = 3,
    grid_internal: str = "legendre-gauss",
    grid: str = "equiangular",
    pos_embed: str = "spectral",
    filter_basis_type: str = "morlet",
    encoder_kernel_shape: tuple[int, int] = (3, 3),
    theta_cutoff: float | None = None,
    residual_prediction: bool = False,
    embed_dim: int = 128,
    num_layers: int = 4,
    num_heads: int = 1,
    mlp_ratio: float = 2.0,
    attention_mode: str = "neighborhood",
    scale_factor: int = 4,
    **overrides: object,
) -> SphericalTransformer:
    """The reference S2 Transformer: ``torch_harmonics``'
    ``SphericalTransformer`` constructed directly on its native tensor grid.

    The defaults are the shared architecture of "Attention on the Sphere"
    (Appendix C.1: ``embed_dim=128, num_layers=4, num_heads=1,
    mlp_ratio=2.0``, a 3x3 encoder kernel, neighborhood attention); further
    ``SphericalTransformer`` keyword arguments pass through ``overrides``.
    The result has no ``dtype``/``device`` constructor argument and is
    moved with ``.to(...)``.

    ``in_chans=4``/``out_chans=3``
        The conditioned input ``(h, u, v, c)`` and the prognostic output
        ``(h, u, v)``; the Coriolis channel is supplied and never predicted.
    ``grid``
        The grid the model reads and writes, ``"equiangular"`` or
        ``"legendre-gauss"``; the DISCO encoder's quadrature follows it.
    ``grid_internal``
        Quadrature rule of the latent grid.
    ``pos_embed="spectral"``
        The paper's spectral positional embedding, kept in every comparison.
    ``scale_factor``
        Downsampling factor of the DISCO encoder and decoder; ``nlon`` must
        be divisible by ``nlon // scale_factor``.
    ``theta_cutoff``
        Attention radius in radians. ``None`` lets each block derive it
        from its latent row count (:mod:`experiments.shallow_water.cutoffs`);
        a number fixes a shared radius. The encoder and decoder DISCO radii
        are separate quantities and are not set by it.
    ``residual_prediction=False``
        The residual objective is applied outside the model over the
        prognostic channels only; an internal residual would add the
        Coriolis channel into a prognostic output.
    """
    return SphericalTransformer(
        img_size=(nlat, nlon),
        grid=grid,
        grid_internal=grid_internal,
        in_chans=in_chans,
        out_chans=out_chans,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        encoder_kernel_shape=encoder_kernel_shape,
        filter_basis_type=filter_basis_type,
        pos_embed=pos_embed,
        attention_mode=attention_mode,
        scale_factor=scale_factor,
        theta_cutoff=theta_cutoff,
        residual_prediction=residual_prediction,
        **overrides,
    )


def build_sfno_reference(
    *,
    nlat: int = 92,
    nlon: int = 184,
    in_chans: int = 4,
    out_chans: int = 3,
    embed_dim: int = 52,
    num_layers: int = 4,
    scale_factor: int = 2,
    residual_prediction: bool = False,
    **overrides: object,
) -> SphericalFourierNeuralOperator:
    """The reference SFNO: ``torch_harmonics``'
    ``SphericalFourierNeuralOperator`` constructed directly on the
    equiangular grid.

    A global spectral convolution, so every output point depends on every
    input point; the second reference next to the finite-support S2
    Transformer. Apart from the arguments below the library defaults stand
    (``grid_internal="legendre-gauss"``, ``pos_embed="none"``,
    ``bias=False``, ``normalization_layer="none"``, ``mlp_ratio=2.0``,
    ``hard_thresholding_fraction=1.0``, ``activation_function="gelu"``);
    further keyword arguments pass through ``overrides``. The result has no
    ``dtype``/``device`` constructor argument and is moved with ``.to(...)``.

    ``nlat``/``nlon`` and ``scale_factor``
        The observation grid and the downsampling factor to the latent
        grid; ``nlon`` must be divisible by ``nlon // scale_factor``.
    ``in_chans=4``/``out_chans=3``
        The conditioned input ``(h, u, v, c)`` and the prognostic output
        ``(h, u, v)``; the Coriolis channel is supplied and never predicted.
    ``embed_dim``
        The capacity knob, set per configuration for a parameter match
        against the S2 reference. The spectral weights are complex, which
        ``numel`` counts once each, and their number scales with the
        retained spherical-harmonic modes, so a match holds only at the
        resolution it was made at.
    ``residual_prediction=False``
        The residual objective is applied outside the model over the
        prognostic channels only.
    """
    return SphericalFourierNeuralOperator(
        img_size=(nlat, nlon),
        grid="equiangular",
        in_chans=in_chans,
        out_chans=out_chans,
        embed_dim=embed_dim,
        num_layers=num_layers,
        scale_factor=scale_factor,
        residual_prediction=residual_prediction,
        **overrides,
    )


class GmAttentionDepth1(nn.Module):
    """One-hop GM model: input and output on the fine level of a depth-1
    icosphere multiresolution, the transformer stack one level down.

    ``fine -> encode -> coarse -> N x SO2TransformerBlock -> decode -> fine``,
    plus a pointwise fine skip carrying the conditioned input to the output.
    The hidden type is ``base_field_type`` scaled by ``multiplicity``. Input
    and output types are fixed by the physics and never inferred from
    ``base_field_type``: a conditioned ``{0: 2, 1: 1}`` in (``[height,
    coriolis]`` at frequency 0, velocity as one ``rho_1`` copy) and a
    prognostic ``{0: 1, 1: 1}`` out, so that adding the conditioned input
    to a prognostic prediction is a type error.

    Parameters
    ----------
    base_field_type
        Hidden spectrum, a mapping from frequency to base multiplicity,
        scaled by ``multiplicity`` through
        :func:`~gm_attention.representations.field_type.scale_field_type`.
    multiplicity
        Hidden width multiplier. Every resulting per-frequency multiplicity
        must be divisible by ``num_heads``.
    num_layers
        Number of ``SO2TransformerBlock``s on the coarse level.
    num_heads
        Attention heads per block; must divide every frequency's
        multiplicity in the hidden type.
    ffn_multiplier
        Feed-forward width as a multiple of the hidden type, preserving the
        frequency ratio through the FFN.
    encode_mode
        ``"learned_integral"``, the only value:
        :class:`~gm_attention.nn.blocks.IcoEncode`, the gauge-steerable
        fine-to-coarse integral.
    encode_cutoff
        Radius of the fine-to-coarse aggregation: the hop's
        ``refinement.coarse_edge_length``, supplied by the experiment setup.
    deep_cutoff
        Attention radius of every coarse-level transformer block. The
        matched configurations set it to the reference's
        neighborhood-attention radius
        (:func:`~experiments.shallow_water.cutoffs.s2_attention_cutoff`).
    decode_refine_cutoff
        Radius of the fine-grid convolution inside the decoder, distinct
        from ``encode_cutoff``. The matched configurations set it to the
        reference decoder's DISCO radius
        (:func:`~experiments.shallow_water.cutoffs.s2_disco_cutoff` at the
        latent row count).
    deep_mixer
        ``"attention"`` (default) or ``"convolution"``: the operator mixing
        information across coarse nodes inside each block. The convolution
        variant keeps the block structure, cutoff and window and removes
        only the feature-dependent attention weights.
    fine_skip_init
        ``"zero"``, the only value: the residual objective adds the
        prognostic input outside the network, so the skip starts at zero.
    gradient_checkpointing
        Rematerialize each coarse block in backward while training.
    evaluation_chunk_size
        Target nodes per chunk for the deep stack and the decoder outside
        training, or ``None`` to hold each edge set whole; for evaluating a
        fitted model on a finer mesh than it was trained on. Each target's
        whole neighborhood stays in one chunk, so its softmax remains a
        partition of unity, and the result differs from the unchunked one
        only by floating-point summation order. Inactive while training.
    decode_self_interaction
        Self-interaction mode of the decoder's refinement convolution:
        ``"none"``, or ``"linear"`` for a pointwise intertwiner alongside
        the integral, as for the deep convolutions.
    window, nonlinearity, dropout, n_radial_basis, radial_hidden_features,
    radial_basis, bias, dtype, device
        Passed through to every ``SO2Convolution``/``SO2TransformerBlock``:
        the kernel window, the block nonlinearity, dropout rate, the radial
        basis and its node count, the radial MLP's hidden widths, whether
        biases are learned, and the parameter dtype/device.
    input_field_type, output_field_type
        Override the conditioned and prognostic types for another state
        representation, such as ``{0: 4}`` in and ``{0: 3}`` out for the
        three-scalar state of ``experiments/shallow_water_paper``; ``None``
        keeps the (height, velocity) types above. Encoder, decoder and fine
        skip are built from the two types.
    """

    def __init__(
        self,
        *,
        base_field_type: Mapping[int, int] | Mapping[str, int],
        multiplicity: int,
        num_layers: int,
        num_heads: int,
        ffn_multiplier: int = 2,
        encode_mode: EncodeModeKind = "learned_integral",
        encode_cutoff: float,
        deep_cutoff: float,
        decode_refine_cutoff: float,
        fine_skip_init: FineSkipInit = "zero",
        window: WindowKind = "quartic",
        deep_mixer: MixerKind = "attention",
        nonlinearity: NonlinearityKind = "scalar_gate",
        dropout: float = 0.0,
        gradient_checkpointing: bool = False,
        evaluation_chunk_size: int | None = None,
        decode_self_interaction: SelfInteractionKind = "none",
        n_radial_basis: int = 8,
        radial_hidden_features: Sequence[int] = (16,),
        radial_basis: RadialBasisKind = "gaussian",
        bias: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        input_field_type: Mapping[int, int] | Mapping[str, int] | None = None,
        output_field_type: Mapping[int, int] | Mapping[str, int] | None = None,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError(f"num_layers must be at least 1, got {num_layers}.")

        if encode_mode != "learned_integral":
            raise ValueError(
                f"encode_mode must be 'learned_integral', got {encode_mode!r}."
            )

        self.input_field_type = (
            CONDITIONED_FIELD_TYPE
            if input_field_type is None
            else scale_field_type(input_field_type, 1)
        )
        self.output_field_type = (
            PROGNOSTIC_FIELD_TYPE
            if output_field_type is None
            else scale_field_type(output_field_type, 1)
        )
        self.hidden_field_type = scale_field_type(base_field_type, multiplicity)
        self.encode_mode: EncodeModeKind = encode_mode
        self.encode_cutoff = float(encode_cutoff)
        self.deep_cutoff = float(deep_cutoff)
        self.decode_refine_cutoff = float(decode_refine_cutoff)

        self.encode: nn.Module = IcoEncode(
            self.input_field_type, self.hidden_field_type,
            cutoff=encode_cutoff, window=window,
            n_radial_basis=n_radial_basis,
            radial_hidden_features=radial_hidden_features,
            radial_basis=radial_basis,
            bias=bias, dtype=dtype, device=device,
        )

        self.blocks = nn.ModuleList(
            [
                SO2TransformerBlock(
                    self.hidden_field_type, num_heads,
                    ffn_multiplier=ffn_multiplier, cutoff=deep_cutoff,
                    mixer=deep_mixer,
                    window=window, nonlinearity=nonlinearity,
                    n_radial_basis=n_radial_basis,
                    radial_hidden_features=radial_hidden_features,
                    radial_basis=radial_basis,
                    dropout=dropout, bias=bias, dtype=dtype, device=device,
                )
                for _ in range(num_layers)
            ]
        )

        self.decode = IcoDecode(
            self.hidden_field_type, self.output_field_type,
            cutoff=decode_refine_cutoff, window=window,
            self_interaction=decode_self_interaction,
            n_radial_basis=n_radial_basis,
            radial_hidden_features=radial_hidden_features,
            radial_basis=radial_basis,
            bias=bias, dtype=dtype, device=device,
        )

        self.fine_skip = SO2Linear(
            self.input_field_type, self.output_field_type,
            bias=False, dtype=dtype, device=device,
        )
        _initialize_fine_skip(self.fine_skip, fine_skip_init)
        self.fine_skip_init: FineSkipInit = fine_skip_init
        self.gradient_checkpointing = bool(gradient_checkpointing)
        if evaluation_chunk_size is not None and evaluation_chunk_size < 1:
            raise ValueError(
                "evaluation_chunk_size must be at least 1 if given, got "
                f"{evaluation_chunk_size}."
            )
        self.evaluation_chunk_size = evaluation_chunk_size

    def forward(
        self, x: SO2FeatureField, geometry: IcosphereMultiresolution
    ) -> SO2FeatureField:
        """Apply the model.

        Parameters
        ----------
        x
            Conditioned field of type ``{0: 2, 1: 1}`` on ``geometry``'s
            finest level.
        geometry
            A **depth-1**
            :class:`~gm_attention.discretization.grids.icosphere.IcosphereMultiresolution`,
            built once from the same grid ``x`` lives on. Its
            ``levels[1]`` coverage radius and ``hops[0]`` edge geometry must
            match the cutoffs this model was constructed with; nothing here
            re-derives or checks that.

        Returns
        -------
        SO2FeatureField
            Prognostic field of type ``{0: 1, 1: 1}`` on the finest level.
        """
        if x.field_type != self.input_field_type:
            raise ValueError(
                f"Expected a conditioned field of type {self.input_field_type!r}, "
                f"got {x.field_type!r}."
            )

        if geometry.depth != 1:
            raise ValueError(
                f"GmAttentionDepth1 needs a depth-1 multiresolution, got depth "
                f"{geometry.depth}."
            )

        coarse = self.encode(x, geometry.hops[0].down_edges)
        for block in self.blocks:
            coarse = self._run_block(block, coarse, geometry.levels[1])

        decoded = (
            self.decode(coarse, geometry.hops[0].up_edges, geometry.levels[0])
            if self.evaluation_chunk_size is None or self.training
            else self.decode.forward_chunked(
                coarse, geometry.hops[0].up_edges, geometry.levels[0],
                chunk_size=self.evaluation_chunk_size,
            )
        )
        return _add_fields(decoded, self.fine_skip(x))

    def _run_block(
        self,
        block: nn.Module,
        x: SO2FeatureField,
        discretization: SphericalDiscretization,
    ) -> SO2FeatureField:
        """One transformer block: chunked by target range outside training
        when ``evaluation_chunk_size`` is set, rematerialized in backward
        when ``gradient_checkpointing`` is set and the model is training,
        applied whole otherwise. The two are never combined."""
        if self.evaluation_chunk_size is not None and not self.training:
            return cast(SO2TransformerBlock, block).forward_chunked(
                x, discretization, chunk_size=self.evaluation_chunk_size
            )

        if not (self.gradient_checkpointing and self.training):
            return block(x, discretization)

        return checkpoint(  # type: ignore[no-any-return]
            block, x, discretization, use_reentrant=False
        )

    def extra_repr(self) -> str:
        return (
            f"input={self.input_field_type!r}, hidden={self.hidden_field_type!r}, "
            f"output={self.output_field_type!r}, encode_mode={self.encode_mode!r}, "
            f"gradient_checkpointing={self.gradient_checkpointing}, "
            f"evaluation_chunk_size={self.evaluation_chunk_size}"
        )


class GmAttentionSingleLevel(nn.Module):
    """``conv -> SO2TransformerBlock x N -> conv`` on one icosphere level.

    :class:`GmAttentionDepth1` without the hop: the same conditioned input
    ``{0: 2, 1: 1}``, the same prognostic output, the same zero-initialized
    fine skip and the same deep stack, with ordinary same-level convolutions
    as encoder and decoder in place of ``IcoEncode``/``IcoDecode``, so that
    every operator lives on a single discretization.

    Parameters
    ----------
    deep_cutoff, conv_cutoff
        The attention support of the blocks and the support of the encoder
        and decoder convolutions, both cutoffs on the one discretization.
        The matched configurations set them to the reference's
        neighborhood-attention radius and DISCO radius respectively
        (:mod:`experiments.shallow_water.cutoffs`).
    self_interaction, encoder_self_interaction
        Self-interaction modes of the decoder and encoder convolutions,
        ``"linear"`` and ``"none"`` by default, so that the encoder stays
        the pure integral ``IcoEncode`` is.
    chunk_size
        Target nodes per chunk for every edge-aggregating layer, in
        training and evaluation alike, or ``None`` to hold each edge set
        whole. The function is the same either way; in training the
        chunked forward checkpoints each chunk, so peak memory is one
        chunk's intermediates.
    base_field_type, multiplicity, num_layers, num_heads, ffn_multiplier,
    fine_skip_init, window, deep_mixer, nonlinearity, dropout,
    n_radial_basis, radial_hidden_features, radial_basis, bias, dtype,
    device, input_field_type, output_field_type
        As :class:`GmAttentionDepth1`.
    """

    def __init__(
        self,
        *,
        base_field_type: Mapping[int, int] | Mapping[str, int],
        multiplicity: int,
        num_layers: int,
        num_heads: int,
        ffn_multiplier: int = 2,
        deep_cutoff: float,
        conv_cutoff: float,
        fine_skip_init: FineSkipInit = "zero",
        window: WindowKind = "quartic",
        deep_mixer: MixerKind = "attention",
        nonlinearity: NonlinearityKind = "scalar_gate",
        dropout: float = 0.0,
        self_interaction: SelfInteractionKind = "linear",
        encoder_self_interaction: SelfInteractionKind = "none",
        chunk_size: int | None = None,
        n_radial_basis: int = 8,
        radial_hidden_features: Sequence[int] = (16,),
        radial_basis: RadialBasisKind = "gaussian",
        bias: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        input_field_type: Mapping[int, int] | Mapping[str, int] | None = None,
        output_field_type: Mapping[int, int] | Mapping[str, int] | None = None,
        attention_batched: bool = False,
    ) -> None:
        """``attention_batched`` runs every block's heads in one pass
        (``SO2MHSA(batched=True)``): the same function up to floating-point
        order. Off by default."""
        super().__init__()

        if num_layers < 1:
            raise ValueError(f"num_layers must be at least 1, got {num_layers}.")
        if chunk_size is not None and chunk_size < 1:
            raise ValueError(f"chunk_size must be at least 1 if given, got {chunk_size}.")

        self.input_field_type = (
            CONDITIONED_FIELD_TYPE
            if input_field_type is None
            else scale_field_type(input_field_type, 1)
        )
        self.output_field_type = (
            PROGNOSTIC_FIELD_TYPE
            if output_field_type is None
            else scale_field_type(output_field_type, 1)
        )
        self.hidden_field_type = scale_field_type(base_field_type, multiplicity)
        self.deep_cutoff = float(deep_cutoff)
        self.conv_cutoff = float(conv_cutoff)
        self.chunk_size = chunk_size

        self.encoder = SO2Convolution(
            self.input_field_type, self.hidden_field_type,
            cutoff=conv_cutoff, window=window,
            self_interaction=encoder_self_interaction,
            n_radial_basis=n_radial_basis,
            radial_hidden_features=radial_hidden_features,
            radial_basis=radial_basis,
            bias=bias, dtype=dtype, device=device,
        )
        self.blocks = nn.ModuleList(
            [
                SO2TransformerBlock(
                    self.hidden_field_type, num_heads,
                    ffn_multiplier=ffn_multiplier, cutoff=deep_cutoff,
                    mixer=deep_mixer,
                    window=window, nonlinearity=nonlinearity,
                    n_radial_basis=n_radial_basis,
                    radial_hidden_features=radial_hidden_features,
                    radial_basis=radial_basis,
                    dropout=dropout, bias=bias, dtype=dtype, device=device,
                    attention_batched=attention_batched,
                )
                for _ in range(num_layers)
            ]
        )
        self.decoder = SO2Convolution(
            self.hidden_field_type, self.output_field_type,
            cutoff=conv_cutoff, window=window,
            self_interaction=self_interaction,
            n_radial_basis=n_radial_basis,
            radial_hidden_features=radial_hidden_features,
            radial_basis=radial_basis,
            bias=bias, dtype=dtype, device=device,
        )
        self.fine_skip = SO2Linear(
            self.input_field_type, self.output_field_type,
            bias=False, dtype=dtype, device=device,
        )
        _initialize_fine_skip(self.fine_skip, fine_skip_init)
        self.fine_skip_init: FineSkipInit = fine_skip_init

    def _run_layer(
        self, module: nn.Module, x: SO2FeatureField, discretization: SphericalDiscretization | StencilPlans
    ) -> SO2FeatureField:
        """Apply ``module`` chunked when ``chunk_size`` is set, whole
        otherwise; the same function either way. Stencil plans run whole:
        the dense fixed-grid path holds nothing per edge to chunk."""
        if self.chunk_size is None or isinstance(discretization, StencilPlans):
            return module(x, discretization)
        return module.forward_chunked(  # type: ignore[operator]
            x, discretization, chunk_size=self.chunk_size
        )

    def forward(
        self, x: SO2FeatureField, discretization: SphericalDiscretization | StencilPlans
    ) -> SO2FeatureField:
        """Apply the model.

        Parameters
        ----------
        x
            Conditioned field of type ``{0: 2, 1: 1}`` on ``discretization``.
        discretization
            The one level every layer aggregates over; each layer resolves
            its own radius through ``edges_within``, so both cutoffs must
            have been prepared on it. Or the
            :class:`~gm_attention.nn.stencil_plan.StencilPlans` of a tensor
            grid in the coordinate gauge at both cutoffs
            (:func:`~gm_attention.nn.stencil_plan.build_tensor_grid_geometry`),
            on which every layer runs on the dense fixed-grid layout, the
            same function to summation order.
        """
        if x.field_type != self.input_field_type:
            raise ValueError(
                f"Expected a conditioned field of type {self.input_field_type!r}, "
                f"got {x.field_type!r}."
            )

        hidden = self._run_layer(self.encoder, x, discretization)
        for block in self.blocks:
            hidden = self._run_layer(block, hidden, discretization)
        decoded = self._run_layer(self.decoder, hidden, discretization)
        return _add_fields(decoded, self.fine_skip(x))
