"""The GM SegFormer: exactly gauge equivariant, sized as recorded, wired as the reference.

The shipped model carries the reported design's options only: the
convolutional readout, the top-down fusion decoder (so every hierarchy here
is built with its lateral edges), steerable values and the pointwise stem.
"""

from __future__ import annotations

import math

import pytest
import torch

from experiments.stanford_2d3ds.geometry import (
    build_segformer_hierarchy,
    segformer_radii,
    segformer_stage_shapes,
)
from experiments.stanford_2d3ds.model import (
    GRAVITY_FIELD_TYPE,
    GmSegformer,
    GmSegformerBlock,
    gravity_features,
    real_parameter_count,
)
from gm_attention.discretization.frames import rotate_tangent_frames
from gm_attention.nn.depthwise import SO2DepthwiseKernel
from gm_attention.representations.field_type import SO2FeatureField, SO2FieldType

IMG = (32, 64)
STAGES = segformer_stage_shapes(IMG, 2, 2)  # (16, 32), (8, 16)
RADII = segformer_radii(STAGES, readout_rows=1.0)


def _hierarchy(frames=None):
    # The fusion decoder (the shipped model's only decoder) reads the top-down lateral edges.
    return build_segformer_hierarchy(IMG, STAGES, RADII, frames=frames, lateral=True, dtype=torch.float64)


def _model(**overrides):
    # The shipped defaults: convolutional readout, fusion decoder, steerable values, no checkpointing.
    settings: dict = {
        "num_classes": 5, "base_field_type": {0: 1, 1: 1, 2: 1}, "multiplicities": [2, 4],
        "heads": [1, 2], "depths": [1, 2], "merge_cutoffs": RADII.merge,
        "attention_cutoffs": RADII.attention, "mix_cutoffs": RADII.mix,
        "interpolate_cutoffs": RADII.interpolate, "readout_cutoffs": RADII.readout,
        "drop_path_rate": 0.1, "dtype": torch.float64,
    }
    settings.update(overrides)
    torch.manual_seed(0)
    return GmSegformer(**settings)


def _input(n_nodes: int, seed: int = 0) -> SO2FeatureField:
    generator = torch.Generator().manual_seed(seed)
    return SO2FeatureField(SO2FieldType({0: 3}), {0: torch.randn(n_nodes, 3, dtype=torch.float64, generator=generator)})


def test_output_is_a_scalar_logit_field_on_the_input_grid() -> None:
    hierarchy = _hierarchy()
    model = _model().eval()
    logits = model(_input(hierarchy.finest.n_nodes), hierarchy)
    assert logits.field_type == SO2FieldType({0: 5})
    assert logits.batch_shape == (hierarchy.finest.n_nodes,)
    assert torch.isfinite(logits[0]).all()


def test_logits_are_exactly_invariant_under_independent_regauging_of_every_level() -> None:
    """Regauge the input grid and each stage grid by separate random angles:
    scalars are invariant, so the logits must not move. On the convolutional
    readout, the shipped model's only one."""
    base = _hierarchy()
    generator = torch.Generator().manual_seed(1)
    frames = [
        rotate_tangent_frames(level.frames, torch.rand(level.n_nodes, dtype=torch.float64, generator=generator) * 2 * math.pi)
        for level in base.levels
    ]
    regauged = _hierarchy(frames=frames)
    model = _model().eval()
    x = _input(base.finest.n_nodes)
    torch.testing.assert_close(model(x, regauged)[0], model(x, base)[0], atol=1e-11, rtol=0.0)


def test_drop_path_is_stochastic_in_training_and_silent_in_eval() -> None:
    hierarchy = _hierarchy()
    x = _input(hierarchy.finest.n_nodes)
    model = _model(drop_path_rate=0.5)
    model.train()
    outputs = {tuple(model(x, hierarchy)[0].flatten()[:8].tolist()) for _ in range(6)}
    assert len(outputs) > 1
    model.eval()
    assert torch.equal(model(x, hierarchy)[0], model(x, hierarchy)[0])
    rates = [block.drop_path1.p for stage in model.stages for block in stage]
    assert rates == pytest.approx([float(r) for r in torch.linspace(0.0, 0.5, 3)])
    assert all(block.drop_path1.p == block.drop_path2.p for stage in model.stages for block in stage)


def test_blocks_are_wired_as_the_references_attention_and_mix_ffn() -> None:
    """The stage blocks, merges and interpolations at the chain's radii. The
    shipped decoder reads every stage through its readout convolution (as
    many scalars as the stage has irrep copies unless ``readout_channels``
    says otherwise) and fuses top-down, so the head sees the fused width."""
    model = _model()
    block = model.stages[1][0]
    assert isinstance(block, GmSegformerBlock)
    assert block.attention.num_heads == 2
    assert block.attention.cutoff == pytest.approx(RADII.attention[1])
    assert isinstance(block.mix.kernel, SO2DepthwiseKernel)
    assert block.mix.cutoff == pytest.approx(RADII.mix[1])
    assert block.linear_in.field_type_in == block.linear_in.field_type_out == model.stage_field_types[1]
    assert model.merges[1].cutoff == pytest.approx(RADII.merge[1])
    assert model.lateral_interpolations[0].cutoff == pytest.approx(RADII.interpolate[1])
    assert model.final_interpolation.cutoff == pytest.approx(RADII.interpolate[0])
    assert model.head.field_type_out == SO2FieldType({0: 5})
    assert model.projected_field_types == (SO2FieldType({0: 6}), SO2FieldType({0: 12}))
    assert model.fused_scalar_type == model.head_field_type == SO2FieldType({0: 32})
    assert model.upsample_linears[1].cutoff == pytest.approx(RADII.readout[1])
    assert model.upsample_linears[1].self_interaction == "linear"
    narrow = _model(readout_channels=[1, 2])
    assert narrow.projected_field_types == (SO2FieldType({0: 1}), SO2FieldType({0: 2}))


def test_the_configured_arm_has_the_count_on_record() -> None:
    """The two shipped configurations at the dataset's 128 x 256 and 14
    classes have the README's counts: 593,518 for ``configs/model/sos2_segformer``
    (icosphere stages of types 6 + 2 + 2 .. 24 + 8 + 8, depths 3 / 4 / 6 / 3,
    steerable values, gravity conditioning, projection and readout heads,
    normalized mix and readout, the 64-wide scalar head) and 599,158 for
    ``sos2_segformer_scalar`` (scalar stages 14 / 26 / 38 / 48 of depths
    3 / 4 / 7 / 3, the gravity field as input channels); against the released
    609,246. The radii do not enter the count."""
    shapes = segformer_stage_shapes((128, 256), 2, 4)
    radii = segformer_radii(shapes, grid_internal="icosphere", icosphere_levels=[5, 4, 3, 2], interpolate_spacings=1.5, readout_rows=1.5)
    common = dict(
        num_classes=14, base_field_type={0: 3, 1: 1, 2: 1}, heads=[1, 1, 2, 4],
        merge_cutoffs=radii.merge, attention_cutoffs=radii.attention, mix_cutoffs=radii.mix,
        interpolate_cutoffs=radii.interpolate, readout="convolution", readout_cutoffs=radii.readout,
        readout_channels=[8, 16, 32, 32], attention_batched=True, n_radial_basis=8, radial_hidden_features=[14],
        value_kind="steerable", ffn_expansion=2, vertical_conditioning=True, decoder="fusion", decoder_channels=32,
        decoder_dropout=0.1, stem_channels=8, stem_kind="pointwise", gravity_readout=True, gravity_readout_expansion=2,
        normalized_mix=True, normalized_readout=True, head_hidden=64,
    )
    arm = GmSegformer(
        multiplicities=[2, 4, 6, 8], depths=[3, 4, 6, 3], drop_path_rate=0.2,
        gravity_conditioning=True, gravity_projection=True, **common,
    )
    assert real_parameter_count(arm) == 593_518
    assert [t.dimension for t in arm.stage_field_types] == [14, 28, 42, 56]
    scalar = GmSegformer(
        multiplicities=[2, 4, 6, 6], depths=[3, 4, 7, 3], drop_path_rate=0.1,
        stage_field_types=[{0: 14}, {0: 26}, {0: 38}, {0: 48}], gravity_input=True, **common,
    )
    assert real_parameter_count(scalar) == 599_158
    assert [t.dimension for t in scalar.stage_field_types] == [14, 26, 38, 48]


def test_train_mode_with_drop_path_is_still_gauge_equivariant() -> None:
    """The same Bernoulli draws on both sides (seeded): the regauged forward
    must reproduce the logits exactly."""
    base = _hierarchy()
    generator = torch.Generator().manual_seed(2)
    frames = [
        rotate_tangent_frames(level.frames, torch.rand(level.n_nodes, dtype=torch.float64, generator=generator) * 2 * math.pi)
        for level in base.levels
    ]
    regauged = _hierarchy(frames=frames)
    model = _model(drop_path_rate=0.5).train()
    x = _input(base.finest.n_nodes)
    torch.manual_seed(11)
    plain = model(x, base)
    torch.manual_seed(11)
    moved = model(x, regauged)
    torch.testing.assert_close(moved[0], plain[0], atol=1e-11, rtol=0.0)


def test_rejects_inconsistent_configurations() -> None:
    with pytest.raises(ValueError, match="one entry per stage"):
        _model(heads=[1])
    with pytest.raises(ValueError, match="divide"):
        _model(multiplicities=[2, 3])
    with pytest.raises(ValueError, match="readout"):
        _model(readout="norms")
    with pytest.raises(ValueError, match="readout_cutoffs"):
        _model(readout="convolution", readout_cutoffs=None)
    with pytest.raises(ValueError, match="readout_channels"):
        _model(readout="convolution", readout_channels=[3])
    hierarchy = _hierarchy()
    model = _model().eval()
    with pytest.raises(ValueError, match="type"):
        model(SO2FeatureField(SO2FieldType({0: 4}), {0: torch.zeros(hierarchy.finest.n_nodes, 4, dtype=torch.float64)}), hierarchy)


@pytest.mark.parametrize("dtype, tolerance", [(torch.float64, 1e-12), (torch.float32, 1e-5)])
def test_logits_shift_with_a_longitude_shift_of_the_input(dtype: torch.dtype, tolerance: float) -> None:
    """A rotation about the axis that maps every level onto itself (four
    input columns: one column of the coarsest stage) must shift the logits
    and nothing else. This is the symmetry a strict radius comparison broke
    by splitting the boundary shells of the neighbourhoods by rounding luck;
    it is not covered by the regauging tests, which keep the graph fixed."""
    hierarchy = build_segformer_hierarchy(IMG, STAGES, RADII, lateral=True, dtype=dtype)
    model = _model(readout="convolution", attention_batched=True, drop_path_rate=0.0, dtype=dtype).eval()
    generator = torch.Generator().manual_seed(0)
    image = torch.randn(3, *IMG, dtype=dtype, generator=generator)

    def run(im: torch.Tensor) -> torch.Tensor:
        x = SO2FeatureField(SO2FieldType({0: 3}), {0: im.permute(1, 2, 0).reshape(-1, 3)})
        return model(x, hierarchy)[0].reshape(*IMG, -1)

    shift = IMG[1] // STAGES[-1][1]
    logits = run(image)
    shifted = run(torch.roll(image, shift, dims=-1))
    torch.testing.assert_close(torch.roll(logits, shift, dims=-2), shifted, atol=tolerance * float(logits.abs().max()), rtol=0.0)


def test_logits_are_exactly_invariant_with_icosphere_stages() -> None:
    """The same model on the mixed hierarchy: an equiangular input level,
    icosphere stage levels, every level regauged independently; with the
    steerable values, the shipped value kind."""
    radii = segformer_radii(STAGES, grid_internal="icosphere", icosphere_levels=[3, 2], interpolate_spacings=1.5, readout_rows=1.0)
    base = build_segformer_hierarchy(IMG, STAGES, radii, grid_internal="icosphere", icosphere_levels=[3, 2], lateral=True, dtype=torch.float64)
    assert [level.n_nodes for level in base.levels] == [2048, 642, 162]
    generator = torch.Generator().manual_seed(3)
    frames = [
        rotate_tangent_frames(level.frames, torch.rand(level.n_nodes, dtype=torch.float64, generator=generator) * 2 * math.pi)
        for level in base.levels
    ]
    regauged = build_segformer_hierarchy(
        IMG, STAGES, radii, grid_internal="icosphere", icosphere_levels=[3, 2], frames=frames, lateral=True, dtype=torch.float64
    )
    model = _model(readout="convolution", readout_cutoffs=radii.readout, interpolate_cutoffs=radii.interpolate, attention_batched=True).eval()
    x = _input(base.finest.n_nodes)
    logits = model(x, base)
    assert logits.batch_shape == (2048,)
    torch.testing.assert_close(model(x, regauged)[0], logits[0], atol=1e-11, rtol=0.0)


def _fusion_model(radii, **overrides):
    # The reported design at toy size: steerable values (the shipped value kind) and the pointwise stem.
    settings: dict = {
        "num_classes": 5, "base_field_type": {0: 4, 1: 1, 2: 1}, "multiplicities": [2, 4], "heads": [1, 1],
        "depths": [1, 2], "merge_cutoffs": radii.merge, "attention_cutoffs": radii.attention, "mix_cutoffs": radii.mix,
        "interpolate_cutoffs": radii.interpolate, "readout": "convolution", "readout_cutoffs": radii.readout,
        "readout_channels": [4, 8], "ffn_expansion": 2, "final_stage_scalars": 24,
        "vertical_conditioning": True, "decoder": "fusion", "decoder_channels": 8, "decoder_dropout": 0.1,
        "stem_channels": 4, "stem_kind": "pointwise", "stem_cutoff": radii.stem, "attention_batched": True, "drop_path_rate": 0.0,
        "dtype": torch.float64,
    }
    settings.update(overrides)
    torch.manual_seed(0)
    return GmSegformer(**settings)


def test_the_second_design_composes_and_keeps_both_exact_symmetries() -> None:
    """FFN expansion, a scalar final stage, vertical conditioning, the fusing
    decoder with lateral interpolation, stem and dropout: the logits are
    still exactly invariant under independent regauging of every level and
    exactly equivariant under a longitude shift; dropout is the identity in
    eval mode and stochastic in train mode. The pointwise stem, the only
    shipped kind, needs no geometry on the input grid."""
    radii = segformer_radii(STAGES, readout_rows=1.0)
    base = build_segformer_hierarchy(IMG, STAGES, radii, lateral=True, dtype=torch.float64)
    assert len(base.lateral_edges) == 1
    assert base.finest.discretization is None
    model = _fusion_model(radii).eval()
    assert model.stage_field_types[-1] == SO2FieldType({0: 24})
    assert model.stages[0][0].hidden_field_type == SO2FieldType({0: 16, 1: 4, 2: 4})
    assert model.head_field_type == SO2FieldType({0: 8 + 4})
    assert real_parameter_count(model) > 0
    x = _input(base.finest.n_nodes)
    logits = model(x, base)
    assert logits.field_type == SO2FieldType({0: 5}) and logits.batch_shape == (base.finest.n_nodes,)
    assert torch.isfinite(logits[0]).all()
    # eval mode is deterministic; train mode's dropout is not
    torch.testing.assert_close(model(x, base)[0], logits[0], atol=0.0, rtol=0.0)
    model.train()
    a, b = model(x, base)[0], model(x, base)[0]
    assert not torch.allclose(a, b)
    model.eval()
    # independent regauging of every level, the input level included
    generator = torch.Generator().manual_seed(5)
    frames = [
        rotate_tangent_frames(level.frames, torch.rand(level.n_nodes, dtype=torch.float64, generator=generator) * 2 * math.pi)
        for level in base.levels
    ]
    regauged = build_segformer_hierarchy(IMG, STAGES, radii, frames=frames, lateral=True, dtype=torch.float64)
    torch.testing.assert_close(model(x, regauged)[0], logits[0], atol=1e-11, rtol=0.0)
    # a longitude shift by the coarsest column
    nlat, nlon = IMG
    shifted = SO2FeatureField(SO2FieldType({0: 3}), {0: torch.roll(x[0].reshape(nlat, nlon, 3), 4, dims=1).reshape(-1, 3)})
    expected = torch.roll(logits[0].reshape(nlat, nlon, -1), 4, dims=1).reshape(-1, 5)
    torch.testing.assert_close(model(shifted, base)[0], expected, atol=1e-10, rtol=0.0)
    # the conditioners start at zero, so at initialization the conditioned model equals the unconditioned one
    torch.manual_seed(0)
    plain = _fusion_model(radii, vertical_conditioning=False).eval()
    plain_state = plain.state_dict()
    model.load_state_dict({**model.state_dict(), **plain_state})
    torch.testing.assert_close(model(x, base)[0], plain(x, base)[0], atol=1e-12, rtol=0.0)
    # the vertical features are the Legendre polynomials of z
    from experiments.stanford_2d3ds.model import vertical_features
    z = base.levels[1].grid.nodes[:, 2]
    torch.testing.assert_close(vertical_features(base.levels[1].grid.nodes)[:, 1], 0.5 * (3 * z * z - 1))


def test_the_second_design_keeps_its_exact_invariance_on_icosphere_stages() -> None:
    """The fusing decoder on icosphere levels: the lateral transfer, the final
    interpolation and the vertical conditioning all read icosphere geometry,
    and the logits stay exactly invariant under independent regauging of
    every level. The pointwise stem needs no geometry on the input grid."""
    levels = [3, 2]
    radii = segformer_radii(STAGES, grid_internal="icosphere", icosphere_levels=levels, interpolate_spacings=1.5, readout_rows=1.0)
    build = lambda frames=None: build_segformer_hierarchy(  # noqa: E731
        IMG, STAGES, radii, grid_internal="icosphere", icosphere_levels=levels, frames=frames, lateral=True, dtype=torch.float64
    )
    base = build()
    assert [level.describe() for level in base.levels] == ["32x64 equiangular", "icosphere L3", "icosphere L2"]
    assert len(base.lateral_edges) == 1
    lateral = base.lateral_edges[0]
    assert torch.bincount(lateral.target_index, minlength=base.levels[1].n_nodes).min() > 0
    assert lateral.distances.max() <= radii.interpolate[1] + 1e-12
    model = _fusion_model(radii, stem_kind="pointwise", stem_cutoff=None).eval()
    x = _input(base.finest.n_nodes)
    logits = model(x, base)
    assert logits.field_type == SO2FieldType({0: 5}) and logits.batch_shape == (base.finest.n_nodes,)
    assert torch.isfinite(logits[0]).all()
    generator = torch.Generator().manual_seed(9)
    frames = [
        rotate_tangent_frames(level.frames, torch.rand(level.n_nodes, dtype=torch.float64, generator=generator) * 2 * math.pi)
        for level in base.levels
    ]
    torch.testing.assert_close(model(x, build(frames))[0], logits[0], atol=1e-11, rtol=0.0)


FULL_SPECTRUM_TWO_STAGES = [{0: 8, 1: 2, 2: 2}, {0: 16, 1: 4, 2: 4}]


def _gravity_model(radii, **overrides):
    settings: dict = dict(
        stem_kind="pointwise", stem_cutoff=None, final_stage_scalars=None, stage_field_types=FULL_SPECTRUM_TWO_STAGES,
        gravity_conditioning=True, gravity_projection=True,
    )
    settings.update(overrides)
    return _fusion_model(radii, **settings)


GRAVITY_FORMS = {
    "grav": {},  # the arm: emit + read at every stage, nothing at the input
    "gravro": dict(gravity_readout=True),  # plus the pointwise gravity readout head after every stage
    "gravin": dict(gravity_input=True, gravity_projection=False),  # the field appended at the input (the scalar ablation's way in), no per-stage read
    # gravro with the Mix-FFN and readout integrals normalized by their window mass and the scalar MLP head
    "gravro_norm": dict(gravity_readout=True, normalized_mix=True, normalized_readout=True, head_hidden=6),
}


def _wake_zero_initialized(model: GmSegformer) -> None:
    """Give every zero-initialized gravity map a nonzero value so the tests
    exercise the field rather than the identity it starts from."""
    for conditioner in model.gravity_conditioners.values():
        for weight in conditioner.weight.values():
            torch.nn.init.normal_(weight, std=0.5)
    for projector in model.gravity_projectors.values():
        torch.nn.init.normal_(projector.weight, std=0.5)
    for head in model.readout_heads:
        torch.nn.init.normal_(head[2].weight, std=0.5)


def test_gravity_invariants_are_the_components_along_and_across_the_vertical_and_are_invariant() -> None:
    radii = segformer_radii(STAGES, readout_rows=1.0)
    hierarchy = build_segformer_hierarchy(IMG, STAGES, radii, lateral=True, dtype=torch.float64)
    level = hierarchy.levels[1]
    field = SO2FeatureField(GRAVITY_FIELD_TYPE, gravity_features(level.grid.nodes, level.frames))
    feature_type = SO2FieldType({0: 3, 1: 2, 2: 1})
    generator = torch.Generator().manual_seed(1)
    feature = SO2FeatureField(feature_type, {
        0: torch.randn(level.n_nodes, 3, dtype=torch.float64, generator=generator),
        1: torch.randn(level.n_nodes, 2, 2, dtype=torch.float64, generator=generator),
        2: torch.randn(level.n_nodes, 1, 2, dtype=torch.float64, generator=generator),
    })
    from experiments.stanford_2d3ds.model import gravity_invariants
    values = gravity_invariants(feature, field)
    assert values.shape == (level.n_nodes, 2 * (2 + 1))
    g = field[1][:, 0, :]
    f = feature[1][:, 0, :]
    torch.testing.assert_close(values[:, 0], (f * g).sum(-1), atol=1e-12, rtol=0.0)  # <f, g>
    torch.testing.assert_close(values[:, 1], f[:, 1] * g[:, 0] - f[:, 0] * g[:, 1], atol=1e-12, rtol=0.0)  # f x g
    angles = torch.rand(level.n_nodes, dtype=torch.float64, generator=generator) * 2 * math.pi
    rotated_field = SO2FeatureField(GRAVITY_FIELD_TYPE, gravity_features(level.grid.nodes, rotate_tangent_frames(level.frames, angles)))
    torch.testing.assert_close(gravity_invariants(feature.gauge_transform(angles), rotated_field), values, atol=1e-12, rtol=0.0)


def test_gravity_features_are_the_tangent_gravity_in_the_frame_and_transform_covariantly() -> None:
    """``rho_1``: the frame coordinates of ``g = e_z - <p, e_z> p``, so they
    reconstruct ``g`` and have squared length ``1 - z^2``; ``rho_2``: their
    complex square. Under a regauging both pick up exactly the field
    transformation law, so the field re-read from rotated frames equals the
    gauge transform of the field read from the original ones."""
    radii = segformer_radii(STAGES, readout_rows=1.0)
    hierarchy = build_segformer_hierarchy(IMG, STAGES, radii, lateral=True, dtype=torch.float64)
    level = hierarchy.levels[1]
    nodes, frames = level.grid.nodes, level.frames
    features = gravity_features(nodes, frames)
    assert set(features) == {1, 2} and features[1].shape == (level.n_nodes, 1, 2) and features[2].shape == (level.n_nodes, 1, 2)
    c = features[1][:, 0, :]  # (n, 2)
    z = nodes[:, 2]
    g = torch.zeros_like(nodes); g[:, 2] = 1.0; g = g - z.unsqueeze(-1) * nodes
    torch.testing.assert_close(torch.einsum("nd,nkd->nk", c, frames), g, atol=1e-12, rtol=0.0)
    torch.testing.assert_close((c * c).sum(-1), 1.0 - z * z, atol=1e-12, rtol=0.0)
    torch.testing.assert_close(features[2][:, 0, 0], c[:, 0] ** 2 - c[:, 1] ** 2, atol=1e-12, rtol=0.0)
    torch.testing.assert_close(features[2][:, 0, 1], 2.0 * c[:, 0] * c[:, 1], atol=1e-12, rtol=0.0)
    # the poles carry no tangent gravity
    polar = z.abs() > 1.0 - 1e-12
    if polar.any():
        assert features[1][polar].abs().max() < 1e-12 and features[2][polar].abs().max() < 1e-12
    # covariance: re-reading from rotated frames == gauge-transforming the field
    generator = torch.Generator().manual_seed(3)
    angles = torch.rand(level.n_nodes, dtype=torch.float64, generator=generator) * 2 * math.pi
    rotated = gravity_features(nodes, rotate_tangent_frames(frames, angles))
    field = SO2FeatureField(GRAVITY_FIELD_TYPE, features).gauge_transform(angles)
    for frequency in (1, 2):
        torch.testing.assert_close(rotated[frequency], field[frequency], atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("form", sorted(GRAVITY_FORMS))
@pytest.mark.parametrize("grid_internal", ["equiangular", "icosphere"])
def test_the_gravity_arm_keeps_exact_gauge_equivariance_and_the_longitude_shift(grid_internal: str, form: str) -> None:
    """With the gravity field appended at the input and injected at every
    stage, the logits are still exactly invariant under independent
    regauging of every level (the field is re-read from the rotated frames,
    so the test is the joint transformation of data and field) and, on the
    equiangular chain, exactly equivariant under a longitude shift, which
    fixes ``e_z``."""
    grids = dict(grid_internal="icosphere", icosphere_levels=[3, 2]) if grid_internal == "icosphere" else {}
    spacing = dict(interpolate_spacings=1.5) if grid_internal == "icosphere" else {}
    radii = segformer_radii(STAGES, readout_rows=1.0, **grids, **spacing)
    build = lambda frames=None: build_segformer_hierarchy(IMG, STAGES, radii, frames=frames, lateral=True, dtype=torch.float64, **grids)  # noqa: E731
    base = build()
    model = _gravity_model(radii, **GRAVITY_FORMS[form]).eval()
    assert model.encoder_input_type == (SO2FieldType({0: 3, 1: 1, 2: 1}) if form == "gravin" else SO2FieldType({0: 3}))
    assert len(model.gravity_conditioners) == 2 and len(model.gravity_projectors) == (0 if form == "gravin" else 2)
    assert len(model.readout_heads) == (2 if form in ("gravro", "gravro_norm") else 0)
    # the maps that read and emit the field start at zero
    assert all(weight.abs().max() == 0 for c in model.gravity_conditioners.values() for weight in c.weight.values())
    assert all(parameter.abs().max() == 0 for projector in model.gravity_projectors.values() for parameter in projector.parameters())
    assert all(parameter.abs().max() == 0 for head in model.readout_heads for parameter in head[2].parameters())
    _wake_zero_initialized(model)
    x = _input(base.finest.n_nodes)
    logits = model(x, base)
    assert logits.field_type == SO2FieldType({0: 5}) and torch.isfinite(logits[0]).all()
    generator = torch.Generator().manual_seed(13)
    frames = [
        rotate_tangent_frames(level.frames, torch.rand(level.n_nodes, dtype=torch.float64, generator=generator) * 2 * math.pi)
        for level in base.levels
    ]
    torch.testing.assert_close(model(x, build(frames))[0], logits[0], atol=1e-11, rtol=0.0)
    if grid_internal == "equiangular":
        nlat, nlon = IMG
        shifted = SO2FeatureField(SO2FieldType({0: 3}), {0: torch.roll(x[0].reshape(nlat, nlon, 3), 4, dims=1).reshape(-1, 3)})
        expected = torch.roll(logits[0].reshape(nlat, nlon, -1), 4, dims=1).reshape(-1, 5)
        torch.testing.assert_close(model(shifted, base)[0], expected, atol=1e-10, rtol=0.0)


@pytest.mark.parametrize("grid_internal", ["equiangular", "icosphere"])
def test_the_scalar_last_stage_read_keeps_exact_symmetries_and_starts_from_zero(grid_internal: str) -> None:
    """The base allocation's last stage is scalar-only: the emitter skips it
    and its read happens at the merge's input, on the source level's frames.
    With every gravity map woken, the logits stay exactly invariant under
    independent regauging of every level (the source read included) and, on
    the equiangular chain, equivariant under a longitude shift; at zero the
    model equals the unconditioned one."""
    grids = dict(grid_internal="icosphere", icosphere_levels=[3, 2]) if grid_internal == "icosphere" else {}
    spacing = dict(interpolate_spacings=1.5) if grid_internal == "icosphere" else {}
    radii = segformer_radii(STAGES, readout_rows=1.0, **grids, **spacing)
    build = lambda frames=None: build_segformer_hierarchy(IMG, STAGES, radii, frames=frames, lateral=True, dtype=torch.float64, **grids)  # noqa: E731
    base = build()
    x = _input(base.finest.n_nodes)
    torch.manual_seed(0)
    model = _fusion_model(radii, stem_kind="pointwise", stem_cutoff=None, gravity_conditioning=True, gravity_projection=True, gravity_readout=True).eval()
    assert model.stage_field_types[-1] == SO2FieldType({0: 24}) and model.gravity_projection_sites == {0: "stage", 1: "source"}
    assert [h[0].in_features for h in model.readout_heads] == [4 + 4 + 3 * 4, 8 + 4]
    torch.manual_seed(0)
    plain = _fusion_model(radii, stem_kind="pointwise", stem_cutoff=None).eval()
    model.load_state_dict({**model.state_dict(), **plain.state_dict()})
    torch.testing.assert_close(model(x, base)[0], plain(x, base)[0], atol=1e-12, rtol=0.0)
    _wake_zero_initialized(model)
    logits = model(x, base)
    assert not torch.allclose(logits[0], plain(x, base)[0])
    generator = torch.Generator().manual_seed(23)
    frames = [rotate_tangent_frames(level.frames, torch.rand(level.n_nodes, dtype=torch.float64, generator=generator) * 2 * math.pi) for level in base.levels]
    torch.testing.assert_close(model(x, build(frames))[0], logits[0], atol=1e-11, rtol=0.0)
    if grid_internal == "equiangular":
        nlat, nlon = IMG
        shifted = SO2FeatureField(SO2FieldType({0: 3}), {0: torch.roll(x[0].reshape(nlat, nlon, 3), 4, dims=1).reshape(-1, 3)})
        expected = torch.roll(logits[0].reshape(nlat, nlon, -1), 4, dims=1).reshape(-1, 5)
        torch.testing.assert_close(model(shifted, base)[0], expected, atol=1e-10, rtol=0.0)


def test_gravity_conditioning_starts_from_the_unconditioned_model_and_needs_directional_channels() -> None:
    radii = segformer_radii(STAGES, readout_rows=1.0)
    base = build_segformer_hierarchy(IMG, STAGES, radii, lateral=True, dtype=torch.float64)
    x = _input(base.finest.n_nodes)
    # emit, read and the readout heads all zero-initialized, so the model equals the plain one
    torch.manual_seed(0)
    conditioned = _gravity_model(radii, gravity_readout=True).eval()
    torch.manual_seed(0)
    plain = _gravity_model(radii, gravity_conditioning=False, gravity_projection=False).eval()
    conditioned.load_state_dict({**conditioned.state_dict(), **plain.state_dict()})
    torch.testing.assert_close(conditioned(x, base)[0], plain(x, base)[0], atol=1e-12, rtol=0.0)
    # the readout head sees [readout, 4 Legendre features, 2 invariants and 1 norm per directional copy]
    assert [head[0].in_features for head in conditioned.readout_heads] == [4 + 4 + 3 * 4, 8 + 4 + 3 * 8]
    # once the maps are nonzero the field acts, and it acts equivariantly
    _wake_zero_initialized(conditioned)
    moved = conditioned(x, base)[0]
    assert not torch.allclose(moved, plain(x, base)[0])
    generator = torch.Generator().manual_seed(17)
    frames = [rotate_tangent_frames(level.frames, torch.rand(level.n_nodes, dtype=torch.float64, generator=generator) * 2 * math.pi) for level in base.levels]
    torch.testing.assert_close(conditioned(x, build_segformer_hierarchy(IMG, STAGES, radii, frames=frames, lateral=True, dtype=torch.float64))[0], moved, atol=1e-11, rtol=0.0)
    # a scalar-only last stage is skipped by the emitter and read at its merge's input by the projector
    scalar_last = _fusion_model(radii, stem_kind="pointwise", stem_cutoff=None, gravity_conditioning=True, gravity_projection=True)
    assert list(scalar_last.gravity_conditioners) == ["0"] and scalar_last.gravity_projection_sites == {0: "stage", 1: "source"}
    assert scalar_last.gravity_projectors["1"].in_features == 2 * 4 and scalar_last.gravity_projectors["1"].out_features == 8
    # a model with no directional stage at all has nothing to condition or to read
    with pytest.raises(ValueError, match="gravity_conditioning needs rho_1 or rho_2"):
        _fusion_model(radii, stem_kind="pointwise", stem_cutoff=None, final_stage_scalars=None, stage_field_types=[{0: 8}, {0: 16}], gravity_conditioning=True)
    with pytest.raises(ValueError, match="gravity_projection needs"):
        _fusion_model(radii, stem_kind="pointwise", stem_cutoff=None, final_stage_scalars=None, stage_field_types=[{0: 8}, {0: 16}], gravity_projection=True)
    # a full-size four-stage allocation with every stage directional: the gravity maps' sizes, keys and inputs
    # (they do not depend on the value kernels or the rest of the trunk)
    shapes = segformer_stage_shapes((128, 256), 2, 4)
    full = segformer_radii(shapes, readout_rows=1.0)
    arm = GmSegformer(
        num_classes=14, base_field_type={0: 4, 1: 1, 2: 1}, multiplicities=[2, 4, 8, 8], heads=[1, 1, 2, 4], depths=[3, 4, 6, 3],
        merge_cutoffs=full.merge, attention_cutoffs=full.attention, mix_cutoffs=full.mix, interpolate_cutoffs=full.interpolate,
        readout="convolution", readout_cutoffs=full.readout, readout_channels=[8, 16, 32, 32], ffn_expansion=2,
        vertical_conditioning=True, decoder="fusion", decoder_channels=32, decoder_dropout=0.1, stem_channels=8, stem_kind="pointwise",
        window="hard", n_radial_basis=8, radial_hidden_features=[14], final_stage_scalars=None,
        stage_field_types=[{0: 8, 1: 2, 2: 2}, {0: 16, 1: 4, 2: 4}, {0: 32, 1: 8, 2: 8}, {0: 64, 1: 16, 2: 16}],
        gravity_conditioning=True, gravity_projection=True,
    )
    assert sum(p.numel() * (2 if p.is_complex() else 1) for p in arm.gravity_conditioners.parameters()) == 120
    assert sum(p.numel() for p in arm.gravity_projectors.parameters()) == 5_560
    with_heads = GmSegformer(
        num_classes=14, base_field_type={0: 4, 1: 1, 2: 1}, multiplicities=[2, 4, 8, 8], heads=[1, 1, 2, 4], depths=[3, 4, 6, 3],
        merge_cutoffs=full.merge, attention_cutoffs=full.attention, mix_cutoffs=full.mix, interpolate_cutoffs=full.interpolate,
        readout="convolution", readout_cutoffs=full.readout, readout_channels=[8, 16, 32, 32], ffn_expansion=2,
        vertical_conditioning=True, decoder="fusion", decoder_channels=32, decoder_dropout=0.1, stem_channels=8, stem_kind="pointwise",
        window="hard", n_radial_basis=8, radial_hidden_features=[14], final_stage_scalars=None,
        stage_field_types=[{0: 8, 1: 2, 2: 2}, {0: 16, 1: 4, 2: 4}, {0: 32, 1: 8, 2: 8}, {0: 64, 1: 16, 2: 16}],
        gravity_conditioning=True, gravity_projection=True, gravity_readout=True,
    )
    assert [sum(p.numel() for p in head.parameters()) for head in with_heads.readout_heads] == [536, 1_968, 7_520, 10_592]
    # the maps are keyed by stage, so a model with every stage directional keeps the names of a list
    assert list(with_heads.gravity_conditioners) == list(with_heads.gravity_projectors) == ["0", "1", "2", "3"]
    assert all(key.startswith(("gravity_conditioners.0.", "gravity_conditioners.1.", "gravity_conditioners.2.", "gravity_conditioners.3."))
               for key in with_heads.state_dict() if key.startswith("gravity_conditioners."))
    # a 96-scalar stage 4: emitters at stages 1-3, the stage-4 read at its merge's input
    scalar_four = GmSegformer(
        num_classes=14, base_field_type={0: 4, 1: 1, 2: 1}, multiplicities=[2, 4, 8, 8], heads=[1, 1, 2, 4], depths=[3, 4, 6, 3],
        merge_cutoffs=full.merge, attention_cutoffs=full.attention, mix_cutoffs=full.mix, interpolate_cutoffs=full.interpolate,
        readout="convolution", readout_cutoffs=full.readout, readout_channels=[8, 16, 32, 32], ffn_expansion=2,
        vertical_conditioning=True, decoder="fusion", decoder_channels=32, decoder_dropout=0.1, stem_channels=8, stem_kind="pointwise",
        window="hard", n_radial_basis=8, radial_hidden_features=[14], final_stage_scalars=96,
        gravity_conditioning=True, gravity_projection=True, gravity_readout=True,
    )
    assert list(scalar_four.gravity_conditioners) == ["0", "1", "2"] and scalar_four.gravity_projection_sites == {0: "stage", 1: "stage", 2: "stage", 3: "source"}
    assert [h[0].in_features for h in scalar_four.readout_heads] == [24, 44, 84, 32 + 4]


def test_the_second_design_rejects_inconsistent_settings() -> None:
    """The shipped model has one readout, one decoder and one stem kind and
    refuses the others outright; its fusion decoder needs lateral edges."""
    radii = segformer_radii(STAGES, readout_rows=1.0, img_size=IMG, stem_rows=1.0)
    with pytest.raises(ValueError, match="readout must be 'convolution'"):
        _fusion_model(radii, readout="scalar", readout_cutoffs=None)
    with pytest.raises(ValueError, match="decoder must be 'fusion'"):
        _fusion_model(radii, decoder="mlp")
    with pytest.raises(ValueError, match="stem_kind must be 'pointwise'"):
        _fusion_model(radii, stem_kind="convolution", stem_cutoff=None)
    assert _fusion_model(radii, stem_kind="pointwise", stem_cutoff=None).stem is not None
    with pytest.raises(ValueError, match="lateral"):
        plain = build_segformer_hierarchy(IMG, STAGES, radii, dtype=torch.float64)
        _fusion_model(radii).eval()(_input(plain.finest.n_nodes), plain)


def test_explicit_stage_types_replace_the_scaled_ones() -> None:
    radii = segformer_radii(STAGES, readout_rows=1.0)
    lower = _fusion_model(radii, stem_kind="pointwise", stem_cutoff=None, final_stage_scalars=None,
                          stage_field_types=[{0: 8, 1: 2, 2: 2}, {0: 16, 1: 8}]).eval()
    assert lower.stage_field_types == (SO2FieldType({0: 8, 1: 2, 2: 2}), SO2FieldType({0: 16, 1: 8}))
    with pytest.raises(ValueError, match="not both"):
        _fusion_model(radii, stem_kind="pointwise", stem_cutoff=None, stage_field_types=[{0: 8, 1: 2, 2: 2}, {0: 16, 1: 8}])
    with pytest.raises(ValueError, match="one type per stage"):
        _fusion_model(radii, stem_kind="pointwise", stem_cutoff=None, final_stage_scalars=None, stage_field_types=[{0: 8}])
    # exact symmetries survive an explicit allocation
    base = build_segformer_hierarchy(IMG, STAGES, radii, lateral=True, dtype=torch.float64)
    x = _input(base.finest.n_nodes)
    logits = lower(x, base)
    generator = torch.Generator().manual_seed(7)
    frames = [rotate_tangent_frames(level.frames, torch.rand(level.n_nodes, dtype=torch.float64, generator=generator) * 2 * math.pi) for level in base.levels]
    torch.testing.assert_close(lower(x, build_segformer_hierarchy(IMG, STAGES, radii, frames=frames, lateral=True, dtype=torch.float64))[0], logits[0], atol=1e-11, rtol=0.0)
