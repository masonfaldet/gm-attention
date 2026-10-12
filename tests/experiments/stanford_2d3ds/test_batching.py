"""Several panoramas through the GM SegFormer as one graph: the tiled hierarchy
is a disjoint union, and the batched step reproduces the single-panorama
step to summation order, forward and backward."""

from __future__ import annotations

import pytest
import torch

from experiments.stanford_2d3ds.data import (
    image_to_field,
    images_to_field,
    logits_to_image,
    logits_to_images,
)
from experiments.stanford_2d3ds.geometry import (
    build_segformer_hierarchy,
    segformer_radii,
    segformer_stage_shapes,
)
from experiments.stanford_2d3ds.model import GmSegformer
from experiments.stanford_2d3ds.train import SegmentationScorer, gm_step
from gm_attention.nn import stochastic_depth
from gm_attention.representations.field_type import SO2FeatureField, SO2FieldType

IMG = (32, 64)
STAGES = segformer_stage_shapes(IMG, 2, 2)  # (16, 32), (8, 16)
LEVELS = [3, 2]
RADII = segformer_radii(
    STAGES, grid_internal="icosphere", icosphere_levels=LEVELS, interpolate_spacings=1.5, readout_rows=1.0,
    img_size=IMG, grid="equiangular", stem_rows=1.0,
)
CPU = torch.device("cpu")


def _hierarchy():
    return build_segformer_hierarchy(
        IMG, STAGES, RADII, grid_internal="icosphere", icosphere_levels=LEVELS, lateral=True, dtype=torch.float64
    )


def _model(**overrides) -> GmSegformer:
    """The shipped design on icosphere stages: fused decoder, vertical conditioning,
    pointwise stem and convolutional readouts, so every transfer kind of a hierarchy is read."""
    settings: dict = {
        "num_classes": 5, "base_field_type": {0: 4, 1: 1, 2: 1}, "multiplicities": [2, 4], "heads": [1, 1],
        "depths": [1, 2], "merge_cutoffs": RADII.merge, "attention_cutoffs": RADII.attention, "mix_cutoffs": RADII.mix,
        "interpolate_cutoffs": RADII.interpolate, "readout": "convolution", "readout_cutoffs": RADII.readout,
        "readout_channels": [4, 8], "value_kind": "steerable", "ffn_expansion": 2, "final_stage_scalars": 24,
        "vertical_conditioning": True, "decoder": "fusion", "decoder_channels": 8, "decoder_dropout": 0.0,
        "stem_channels": 4, "stem_kind": "pointwise", "stem_cutoff": RADII.stem, "attention_batched": True,
        "drop_path_rate": 0.0, "dtype": torch.float64,
    }
    settings.update(overrides)
    torch.manual_seed(0)
    return GmSegformer(**settings)


def _images(count: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(count, 3, *IMG, dtype=torch.float64, generator=generator)
    targets = torch.randint(0, 5, (count, *IMG), generator=generator)
    return images, targets


def test_a_tiled_edge_set_is_the_disjoint_union_of_its_copies() -> None:
    hierarchy = _hierarchy()
    for edges, n_sources in [
        (hierarchy.levels[1].discretization.edges_within(RADII.attention[0]), hierarchy.levels[1].n_nodes),
        (hierarchy.down_edges[0], hierarchy.levels[0].n_nodes),
        (hierarchy.up_edges[1], hierarchy.levels[2].n_nodes),
        (hierarchy.lateral_edges[0], hierarchy.levels[2].n_nodes),
    ]:
        n_targets = edges.row_ptr.shape[0] - 1
        union = edges.tile(3, n_sources=n_sources)
        assert union.n_edges == 3 * edges.n_edges and union.row_ptr.shape == (3 * n_targets + 1,)
        for b in range(3):
            block = slice(b * edges.n_edges, (b + 1) * edges.n_edges)
            assert torch.equal(union.edge_index[0, block], edges.edge_index[0] + b * n_targets)
            assert torch.equal(union.edge_index[1, block], edges.edge_index[1] + b * n_sources)
            assert torch.equal(union.row_ptr[b * n_targets: (b + 1) * n_targets + 1], edges.row_ptr + b * edges.n_edges)
            for name in ("log_coordinates", "distances", "polar_angles", "transport_angles",
                         "tangent_quadrature_weights", "angular_phases", "transport_phases"):
                assert torch.equal(getattr(union, name)[block], getattr(edges, name))
        # target-major, as every consumer assumes
        assert torch.all(union.edge_index[0][1:] >= union.edge_index[0][:-1])
    assert hierarchy.tile(1) is hierarchy
    with pytest.raises(ValueError):
        hierarchy.down_edges[0].tile(0, n_sources=1)


def test_the_tiled_hierarchy_answers_every_cached_radius_and_keeps_the_grids() -> None:
    hierarchy = _hierarchy()
    union = hierarchy.tile(2)
    for level, tiled in zip(hierarchy.levels, union.levels, strict=True):
        assert tiled.n_nodes == 2 * level.n_nodes and tiled.describe() == level.describe()
        assert torch.equal(tiled.grid.nodes, level.grid.nodes.repeat(2, 1))
        assert torch.equal(tiled.frames, level.frames.repeat(2, 1, 1))
        if level.discretization is not None:
            assert tiled.discretization.cached_radii == level.discretization.cached_radii
            for radius in level.discretization.cached_radii:
                assert tiled.discretization.edges_within(radius).n_edges == 2 * level.discretization.edges_within(radius).n_edges
    assert union.depth == hierarchy.depth and len(union.lateral_edges) == len(hierarchy.lateral_edges)


def test_the_image_conversions_round_trip_sample_major() -> None:
    images, _ = _images(3)
    field = images_to_field(images)
    assert field.batch_shape == (3 * IMG[0] * IMG[1],)
    for b in range(3):
        single = image_to_field(images[b])
        assert torch.equal(field[0][b * IMG[0] * IMG[1]: (b + 1) * IMG[0] * IMG[1]], single[0])
    logits = SO2FeatureField(SO2FieldType({0: 3}), {0: field[0]})
    assert torch.equal(logits_to_images(logits, IMG), images)
    assert torch.equal(logits_to_images(logits, IMG)[1:2], logits_to_image(image_to_field(images[1]), IMG))
    with pytest.raises(ValueError):
        logits_to_images(SO2FeatureField(SO2FieldType({0: 3}), {0: field[0][:-1]}), IMG)


def test_the_union_forward_is_the_single_forwards() -> None:
    """Three panoramas as one graph give the same logits as three calls, and
    the batched loss is the mean of the single losses."""
    hierarchy = _hierarchy()
    model = _model().eval()
    images, targets = _images(3)
    scorer = SegmentationScorer.build(IMG, grid="equiangular", label_smoothing=0.0, device=CPU)
    with torch.no_grad():
        singles = [logits_to_image(model(image_to_field(images[b]), hierarchy), IMG) for b in range(3)]
        batched = logits_to_images(model(images_to_field(images), hierarchy.tile(3)), IMG)
    for b in range(3):
        torch.testing.assert_close(batched[b: b + 1], singles[b], rtol=1e-10, atol=1e-10)
    losses = [float(scorer(singles[b], targets[b: b + 1]).loss) for b in range(3)]
    assert float(scorer(batched, targets).loss) == pytest.approx(sum(losses) / 3, rel=1e-10)


def test_the_batched_step_accumulates_the_same_gradient() -> None:
    """gm_step on a (B, C, nlat, nlon) batch equals the mean of B single steps:
    the same loss and the same parameter gradients to summation order."""
    hierarchy = _hierarchy()
    model = _model().train()
    scorer = SegmentationScorer.build(IMG, grid="equiangular", label_smoothing=0.0, device=CPU)
    step = gm_step(hierarchy, IMG, scorer)
    images, targets = _images(3, seed=1)

    model.zero_grad(set_to_none=True)
    singles = [step(model, (images[b], targets[b]), CPU).loss for b in range(3)]
    (sum(singles) / 3).backward()
    reference = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}

    model.zero_grad(set_to_none=True)
    output = step(model, (images, targets), CPU)
    assert float(output.loss) == pytest.approx(float(sum(singles)) / 3, rel=1e-10)
    output.loss.backward()
    batched = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    assert set(batched) == set(reference)
    for name, expected in reference.items():
        torch.testing.assert_close(batched[name], expected, rtol=1e-9, atol=1e-9, msg=name)
    # the tiled hierarchy is built once per batch size and reused
    assert step(model, (images[:2], targets[:2]), CPU).loss.shape == ()


def test_the_batched_step_draws_drop_path_per_panorama(monkeypatch: pytest.MonkeyPatch) -> None:
    """In training the union step declares its B panoramas to stochastic depth,
    so every drop-path layer draws once per panorama, not once for the union."""
    hierarchy = _hierarchy()
    model = _model(drop_path_rate=0.5).train()
    scorer = SegmentationScorer.build(IMG, grid="equiangular", label_smoothing=0.0, device=CPU)
    step = gm_step(hierarchy, IMG, scorer)
    images, targets = _images(3)
    seen: list[int] = []
    original = stochastic_depth.SO2DropPath.forward

    def recording(self: stochastic_depth.SO2DropPath, x: SO2FeatureField) -> SO2FeatureField:
        seen.append(stochastic_depth.current_copies())
        return original(self, x)

    monkeypatch.setattr(stochastic_depth.SO2DropPath, "forward", recording)
    step(model, (images, targets), CPU)
    assert seen and set(seen) == {3}
    seen.clear()
    step(model, (images[0], targets[0]), CPU)
    assert seen and set(seen) == {1}
    assert stochastic_depth.current_copies() == 1


def test_panoramas_per_forward_must_divide_the_batch_size() -> None:
    from omegaconf import OmegaConf

    from experiments.stanford_2d3ds.train import panoramas_per_forward

    assert panoramas_per_forward(OmegaConf.create({"training": {"batch_size": 8}})) == 1
    assert panoramas_per_forward(OmegaConf.create({"training": {"batch_size": 8, "panoramas_per_forward": 4}})) == 4
    with pytest.raises(ValueError):
        panoramas_per_forward(OmegaConf.create({"training": {"batch_size": 8, "panoramas_per_forward": 3}}))
    with pytest.raises(ValueError):
        panoramas_per_forward(OmegaConf.create({"training": {"batch_size": 8, "panoramas_per_forward": 0}}))
