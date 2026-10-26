"""The GM SegFormer with its stages on Gauss-Legendre grids: the stencil path through the model.

``build_segformer_hierarchy(..., stencil_plans=True)`` gives every
Gauss-Legendre stage level the plans of its attention, mix and readout radii
in the coordinate gauge, and the model runs those operators on the dense
fixed-grid layout; the merges and interpolations stay on their bipartite
edges. The checks: the logits agree with the same hierarchy stripped of its
plans (the graph path on the same discretizations) in float64 and float32,
gradients agree, the input level keeps the graph path, tiling drops the
plans, moving keeps them, and equiangular stages are refused.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from experiments.stanford_2d3ds.geometry import (
    build_segformer_hierarchy,
    segformer_radii,
    segformer_stage_shapes,
)
from experiments.stanford_2d3ds.model import GmSegformer
from gm_attention.discretization.frames import build_coordinate_frames
from gm_attention.discretization.grids.equiangular import TensorGridHierarchy
from gm_attention.nn.stencil_plan import StencilPlans
from gm_attention.representations.field_type import SO2FeatureField, SO2FieldType

IMG = (32, 64)
STAGES = segformer_stage_shapes(IMG, 2, 2)  # (16, 32), (8, 16)
RADII = segformer_radii(STAGES, readout_rows=1.0)
ATOL = {torch.float64: 1e-10, torch.float32: 2e-4}


def _hierarchy(dtype: torch.dtype, **overrides) -> TensorGridHierarchy:
    settings: dict = {"grid_internal": "legendre-gauss", "lateral": True, "stencil_plans": True, "dtype": dtype}
    settings.update(overrides)
    return build_segformer_hierarchy(IMG, STAGES, RADII, **settings)


def _without_plans(hierarchy: TensorGridHierarchy) -> TensorGridHierarchy:
    """The same discretizations, frames and transfers, on the graph path."""
    return dataclasses.replace(hierarchy, levels=tuple(dataclasses.replace(level, plans=None) for level in hierarchy.levels))


def _model(dtype: torch.dtype, **overrides) -> GmSegformer:
    settings: dict = {
        "num_classes": 5, "base_field_type": {0: 1, 1: 1, 2: 1}, "multiplicities": [2, 4],
        "heads": [1, 2], "depths": [1, 2], "merge_cutoffs": RADII.merge,
        "attention_cutoffs": RADII.attention, "mix_cutoffs": RADII.mix,
        "interpolate_cutoffs": RADII.interpolate, "readout_cutoffs": RADII.readout,
        "drop_path_rate": 0.0, "normalized_mix": True, "dtype": dtype,
    }
    settings.update(overrides)
    torch.manual_seed(0)
    return GmSegformer(**settings).eval()


def _input(n_nodes: int, dtype: torch.dtype, seed: int = 0) -> SO2FeatureField:
    generator = torch.Generator().manual_seed(seed)
    return SO2FeatureField(SO2FieldType({0: 3}), {0: torch.randn(n_nodes, 3, dtype=dtype, generator=generator)})


@pytest.fixture(params=[torch.float64, torch.float32])
def dtype(request: pytest.FixtureRequest) -> torch.dtype:
    return request.param


def test_the_stage_levels_carry_plans_at_their_radii_and_the_input_level_does_not() -> None:
    hierarchy = _hierarchy(torch.float64)
    assert hierarchy.levels[0].plans is None and hierarchy.levels[0].local is hierarchy.levels[0].discretization
    for stage, level in enumerate(hierarchy.levels[1:]):
        assert isinstance(level.plans, StencilPlans) and level.local is level.plans
        radii = sorted((RADII.attention[stage], RADII.mix[stage], RADII.readout[stage]))
        assert level.plans.cutoffs == pytest.approx(radii, abs=1e-12)  # keyed as the discretization keys its radii
        assert all(radius in level.plans for radius in radii)
        assert (level.plans.n_latitude, level.plans.n_longitude) == tuple(level.shape)
        assert torch.allclose(level.frames, build_coordinate_frames(level.grid.nodes))
        assert level.discretization is not None and level.discretization.cached_radii == level.plans.cutoffs


def test_logits_agree_between_the_graph_path_and_the_stencil_plans(dtype: torch.dtype) -> None:
    hierarchy = _hierarchy(dtype)
    model = _model(dtype)
    x = _input(hierarchy.finest.n_nodes, dtype)
    expected = model(x, _without_plans(hierarchy))[0]
    actual = model(x, hierarchy)[0]
    assert torch.allclose(actual, expected, atol=ATOL[dtype] * float(expected.abs().max()), rtol=0.0), (
        f"max |diff| {float((actual - expected).abs().max()):.3e}"
    )


def test_gradients_agree_between_the_two_paths() -> None:
    hierarchy = _hierarchy(torch.float64)
    model = _model(torch.float64)
    flat = _input(hierarchy.finest.n_nodes, torch.float64).to_flat()
    probe = torch.randn(hierarchy.finest.n_nodes, 5, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    parameters = list(model.parameters())

    def run(geometry: TensorGridHierarchy) -> list[torch.Tensor]:
        leaf = flat.clone().requires_grad_(True)
        logits = model(SO2FeatureField(SO2FieldType({0: 3}), {0: leaf}), geometry)[0]
        return list(torch.autograd.grad((logits * probe).sum(), [leaf, *parameters], allow_unused=True))

    for name, a, b in zip(["input", *(n for n, _ in model.named_parameters())], run(hierarchy), run(_without_plans(hierarchy)), strict=True):
        if a is None or b is None:
            assert a is None and b is None, name
            continue
        scale = max(1.0, float(b.abs().max()))
        assert torch.allclose(a, b, atol=1e-9 * scale, rtol=0.0), f"{name}: max |diff| {float((a - b).abs().max()):.3e}"


def test_tiling_drops_the_plans_and_moving_keeps_them() -> None:
    hierarchy = _hierarchy(torch.float64)
    tiled = hierarchy.tile(2)
    assert all(level.plans is None for level in tiled.levels)
    assert tiled.levels[1].discretization is not None and tiled.levels[1].n_nodes == 2 * hierarchy.levels[1].n_nodes
    moved = hierarchy.to("cpu")
    assert all((a.plans is None) == (b.plans is None) for a, b in zip(moved.levels, hierarchy.levels, strict=True))
    assert moved.levels[1].plans is not None and moved.levels[1].plans.cutoffs == hierarchy.levels[1].plans.cutoffs


def test_plans_need_gauss_legendre_stages_and_explicit_frames_are_honoured() -> None:
    with pytest.raises(ValueError, match="legendre-gauss"):
        _hierarchy(torch.float64, grid_internal="equiangular")
    with pytest.raises(ValueError, match="legendre-gauss"):
        radii = segformer_radii(STAGES, grid_internal="icosphere", icosphere_levels=[3, 2], interpolate_spacings=1.5, readout_rows=1.0)
        build_segformer_hierarchy(IMG, STAGES, radii, grid_internal="icosphere", icosphere_levels=[3, 2], lateral=True, stencil_plans=True, dtype=torch.float64)
    # Frames given by the caller are kept; the default tangent frames do not fold, and the builder says so.
    plain = _hierarchy(torch.float64, stencil_plans=False)
    with pytest.raises(ValueError, match="covariant|fold"):
        _hierarchy(torch.float64, frames=[level.frames for level in plain.levels])
