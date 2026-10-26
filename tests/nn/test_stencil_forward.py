"""Tests for the layers' ``forward(x, plan=...)``: the stencil path through the layers' own entry point.

``SO2Attention``, ``SO2MHSA`` and ``SO2Convolution`` accept a
``StencilPlan`` in place of their edges, as a keyword or in the ``edges``
position, and then evaluate ``stencil_attention``, ``stencil_mhsa`` and
``stencil_convolution``. The checks: the plan route returns exactly what the
stencil function returns, agrees with the graph path to summation order in
float64 and float32, carries leading batch axes, flows gradients, and is
rejected together with edges, without either, and by the chunked entry
points; and a transformer block forwards a plan to either mixer.
"""

from __future__ import annotations

import pytest
import torch

from experiments.stanford_2d3ds.model import GmSegformerBlock
from gm_attention.discretization.stencil import build_tensor_grid_discretization, build_tensor_grid_stencil
from gm_attention.nn.attention import SO2MHSA, SO2Attention
from gm_attention.nn.blocks import SO2TransformerBlock
from gm_attention.nn.convolution import SO2Convolution
from gm_attention.nn.stencil_attention import stencil_attention, stencil_mhsa
from gm_attention.nn.stencil_convolution import stencil_convolution
from gm_attention.nn.stencil_plan import build_stencil_plan, build_tensor_grid_geometry, split_edges_and_plan
from gm_attention.representations.field_type import SO2FeatureField, SO2FieldType

SHAPE = (10, 20)
CUTOFF = 0.6
TYPE_IN = SO2FieldType({0: 3, 1: 2, 2: 1})
TYPE_ATTN = SO2FieldType({0: 2, 1: 1, 2: 1})
TYPE_OUT = SO2FieldType({0: 2, 1: 2, 2: 1})
BLOCK_TYPE = SO2FieldType({0: 4, 1: 2, 2: 2})
ATOL = {torch.float64: 1e-10, torch.float32: 2e-4}


def _geometry(dtype: torch.dtype):
    discretization = build_tensor_grid_discretization("legendre-gauss", *SHAPE, coverage_radius=CUTOFF, dtype=dtype)
    stencil = build_tensor_grid_stencil(discretization, CUTOFF, n_latitude=SHAPE[0], n_longitude=SHAPE[1])
    return discretization, build_stencil_plan(stencil)


def _field(field_type: SO2FieldType, batch_shape: tuple[int, ...], dtype: torch.dtype, seed: int) -> SO2FeatureField:
    generator = torch.Generator().manual_seed(seed)
    return SO2FeatureField.from_flat(field_type, torch.randn(*batch_shape, field_type.dimension, dtype=dtype, generator=generator))


def _assert_fields_close(a: SO2FeatureField, b: SO2FeatureField, atol: float) -> None:
    assert a.field_type == b.field_type and a.batch_shape == b.batch_shape
    for frequency in a.field_type.frequencies:
        assert torch.allclose(a[frequency], b[frequency], atol=atol, rtol=0.0), (
            f"frequency {frequency}: max |diff| {float((a[frequency] - b[frequency]).abs().max()):.3e}"
        )


def _assert_fields_equal(a: SO2FeatureField, b: SO2FeatureField) -> None:
    for frequency in a.field_type.frequencies:
        assert torch.equal(a[frequency], b[frequency]), f"frequency {frequency} differs"


def _layers(dtype: torch.dtype) -> dict[str, tuple[torch.nn.Module, SO2FieldType, object]]:
    torch.manual_seed(1)
    return {
        "attention": (
            SO2Attention(TYPE_IN, TYPE_ATTN, TYPE_OUT, cutoff=CUTOFF, window="quartic", self_interaction="linear", dtype=dtype),
            TYPE_IN, stencil_attention,
        ),
        "mhsa": (SO2MHSA(BLOCK_TYPE, num_heads=2, cutoff=CUTOFF, dtype=dtype), BLOCK_TYPE, stencil_mhsa),
        "convolution": (
            SO2Convolution(TYPE_IN, TYPE_OUT, cutoff=CUTOFF, window="quartic", self_interaction="linear", normalize=True, dtype=dtype),
            TYPE_IN, stencil_convolution,
        ),
        "depthwise": (SO2Convolution(TYPE_IN, TYPE_IN, cutoff=CUTOFF, depthwise=True, normalize=True, dtype=dtype), TYPE_IN, stencil_convolution),
    }


@pytest.fixture(params=[torch.float64, torch.float32])
def dtype(request: pytest.FixtureRequest) -> torch.dtype:
    return request.param


@pytest.mark.parametrize("name", ["attention", "mhsa", "convolution", "depthwise"])
def test_forward_with_a_plan_is_the_stencil_function_and_matches_the_graph_path(dtype: torch.dtype, name: str) -> None:
    discretization, plan = _geometry(dtype)
    module, type_in, function = _layers(dtype)[name]
    x = _field(type_in, (plan.n_nodes,), dtype, seed=2)
    via_keyword = module(x, plan=plan)
    _assert_fields_equal(via_keyword, function(module, x, plan))
    _assert_fields_equal(via_keyword, module(x, plan))  # the plan in the edges position
    _assert_fields_close(via_keyword, module(x, discretization), ATOL[dtype])


@pytest.mark.parametrize("name", ["attention", "mhsa", "convolution", "depthwise"])
def test_forward_with_a_plan_carries_leading_batch_axes(name: str) -> None:
    discretization, plan = _geometry(torch.float64)
    module, type_in, _ = _layers(torch.float64)[name]
    x = _field(type_in, (2, 3, plan.n_nodes), torch.float64, seed=3)
    out = module(x, plan=plan)
    assert out.batch_shape == (2, 3, plan.n_nodes)
    for i in range(2):
        for j in range(3):
            sample = SO2FeatureField(type_in, {f: t[i, j] for f, t in x.coefficients.items()})
            expected = module(sample, discretization)
            for frequency in out.field_type.frequencies:
                assert torch.allclose(out[frequency][i, j], expected[frequency], atol=1e-10, rtol=0.0)


def test_forward_with_a_plan_flows_gradients_like_the_graph_path() -> None:
    discretization, plan = _geometry(torch.float64)
    module, type_in, _ = _layers(torch.float64)["mhsa"]
    flat = _field(type_in, (plan.n_nodes,), torch.float64, seed=4).to_flat()
    probe = torch.randn_like(flat)
    parameters = list(module.parameters())

    def run(target) -> list[torch.Tensor]:
        leaf = flat.clone().requires_grad_(True)
        loss = (module(SO2FeatureField.from_flat(type_in, leaf), target).to_flat() * probe).sum()
        return list(torch.autograd.grad(loss, [leaf, *parameters]))

    for a, b in zip(run(plan), run(discretization), strict=True):
        assert torch.allclose(a, b, atol=1e-9, rtol=0.0)


def test_forward_rejects_both_arguments_neither_and_a_plan_of_the_wrong_cutoff() -> None:
    discretization, plan = _geometry(torch.float64)
    layers = _layers(torch.float64)
    for name in ("attention", "mhsa", "convolution"):
        module, type_in, _ = layers[name]
        x = _field(type_in, (plan.n_nodes,), torch.float64, seed=5)
        with pytest.raises(TypeError, match="not both"):
            module(x, discretization, plan=plan)
        with pytest.raises(TypeError, match="once"):
            module(x, plan, plan=plan)
        with pytest.raises(TypeError, match="needs edges"):
            module(x)
        with pytest.raises(TypeError, match="StencilPlan"):
            module(x, plan=discretization)  # type: ignore[arg-type]
    other = SO2Attention(TYPE_IN, TYPE_ATTN, TYPE_OUT, cutoff=0.5, dtype=torch.float64)
    with pytest.raises(ValueError, match="cutoff"):
        other(_field(TYPE_IN, (plan.n_nodes,), torch.float64, seed=0), plan=plan)
    assert split_edges_and_plan(discretization, None, CUTOFF) == (discretization, None)
    assert split_edges_and_plan(plan, None, CUTOFF) == (None, plan)


def test_chunked_evaluation_rejects_a_plan() -> None:
    _, plan = _geometry(torch.float64)
    layers = _layers(torch.float64)
    for name in ("attention", "mhsa", "convolution"):
        module, type_in, _ = layers[name]
        x = _field(type_in, (plan.n_nodes,), torch.float64, seed=6)
        with pytest.raises(TypeError, match="chunked"):
            module.forward_chunked(x, plan, chunk_size=16)  # type: ignore[arg-type]


@pytest.mark.parametrize("mixer", ["attention", "convolution"])
def test_transformer_block_forwards_a_plan_to_its_mixer(dtype: torch.dtype, mixer: str) -> None:
    discretization, plan = _geometry(dtype)
    torch.manual_seed(7)
    block = SO2TransformerBlock(BLOCK_TYPE, num_heads=2, cutoff=CUTOFF, window="quartic", mixer=mixer, dtype=dtype).eval()
    x = _field(BLOCK_TYPE, (plan.n_nodes,), dtype, seed=8)
    _assert_fields_close(block(x, plan), block(x, discretization), ATOL[dtype])
    with pytest.raises(TypeError, match="chunked"):
        block.forward_chunked(x, plan, chunk_size=16)


# --------------------------------------------------------------------------
# Plans at several cutoffs: each layer resolves its own


@pytest.mark.parametrize("name", ["attention", "mhsa", "convolution", "depthwise"])
def test_forward_resolves_a_plans_container_at_the_layers_cutoff(dtype: torch.dtype, name: str) -> None:
    discretization, plans = build_tensor_grid_geometry("legendre-gauss", *SHAPE, [CUTOFF, 0.4], dtype=dtype)
    module, type_in, _ = _layers(dtype)[name]
    x = _field(type_in, (plans.n_nodes,), dtype, seed=40)
    via_container = module(x, plans)
    _assert_fields_equal(via_container, module(x, plan=plans.for_cutoff(CUTOFF)))
    _assert_fields_equal(via_container, module(x, plan=plans))
    _assert_fields_close(via_container, module(x, discretization), ATOL[dtype])
    with pytest.raises(TypeError, match="chunked"):
        module.forward_chunked(x, plans, chunk_size=16)  # type: ignore[arg-type]


def test_forward_with_a_container_lacking_the_layers_cutoff_raises() -> None:
    _, plans = build_tensor_grid_geometry("legendre-gauss", *SHAPE, [0.4])
    module, type_in, _ = _layers(torch.float64)["attention"]
    with pytest.raises(ValueError, match="No stencil plan at cutoff 0.6"):
        module(_field(type_in, (plans.n_nodes,), torch.float64, seed=41), plans)


def test_segmentation_block_with_two_cutoffs_resolves_both_from_one_container(dtype: torch.dtype) -> None:
    """The attention at one radius and the depthwise mix at another, each
    finding its plan in the container the block forwards to both."""
    mix_cutoff = 0.4
    discretization, plans = build_tensor_grid_geometry("legendre-gauss", *SHAPE, [CUTOFF, mix_cutoff], dtype=dtype)
    torch.manual_seed(42)
    block = GmSegformerBlock(
        BLOCK_TYPE, num_heads=2, attention_cutoff=CUTOFF, mix_cutoff=mix_cutoff, window="hard",
        nonlinearity="scalar_gate_double_sigmoid", drop_path=0.0, n_radial_basis=4, radial_hidden_features=(8,),
        radial_basis="gaussian", bias=True, ffn_expansion=2, normalized_mix=True, dtype=dtype,
    ).eval()
    x = _field(BLOCK_TYPE, (plans.n_nodes,), dtype, seed=43)
    _assert_fields_close(block(x, plans), block(x, discretization), ATOL[dtype])
