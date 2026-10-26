"""GmAttentionSingleLevel on a Gauss-Legendre grid: the stencil path through the model.

``build_tensor_grid_geometry`` builds the grid's discretization and its plans
at the model's two cutoffs side by side; the model takes either. The checks:
the forward agrees between the two paths in float64 and float32, with the
graph path chunked as the released configuration runs it (the plans run
whole), and the gradients with respect to the input and every parameter
agree.
"""

from __future__ import annotations

import pytest
import torch

from experiments.shallow_water.models import (
    CONDITIONED_FIELD_TYPE,
    GmAttentionSingleLevel,
)
from gm_attention.nn.stencil_plan import build_tensor_grid_geometry
from gm_attention.representations.field_type import SO2FeatureField

SHAPE = (8, 16)
DEEP, CONV = 0.55, 0.4
ATOL = {torch.float64: 1e-10, torch.float32: 2e-4}


def _model(dtype: torch.dtype, **overrides) -> GmAttentionSingleLevel:
    torch.manual_seed(51)
    kwargs: dict = {
        "base_field_type": {0: 1, 1: 1, 2: 1}, "multiplicity": 4, "num_layers": 2,
        "num_heads": 2, "deep_cutoff": DEEP, "conv_cutoff": CONV, "window": "hard",
        "nonlinearity": "scalar_gate_double_sigmoid", "self_interaction": "linear",
        "dtype": dtype,
    }
    kwargs.update(overrides)
    return GmAttentionSingleLevel(**kwargs)


def _conditioned(n: int, dtype: torch.dtype, seed: int = 52) -> SO2FeatureField:
    generator = torch.Generator().manual_seed(seed)
    return SO2FeatureField(
        CONDITIONED_FIELD_TYPE,
        {
            0: torch.randn(n, 2, dtype=dtype, generator=generator),
            1: torch.randn(n, 1, 2, dtype=dtype, generator=generator),
        },
    )


@pytest.fixture(params=[torch.float64, torch.float32])
def dtype(request: pytest.FixtureRequest) -> torch.dtype:
    return request.param


@pytest.mark.parametrize("chunk_size", [None, 16])
def test_the_model_agrees_between_the_graph_path_and_the_stencil_plans(dtype: torch.dtype, chunk_size: int | None) -> None:
    discretization, plans = build_tensor_grid_geometry("legendre-gauss", *SHAPE, [DEEP, CONV], dtype=dtype)
    assert plans.cutoffs == (CONV, DEEP)
    model = _model(dtype, chunk_size=chunk_size).eval()
    x = _conditioned(discretization.n_nodes, dtype)
    expected = model(x, discretization)  # chunked when chunk_size is set
    actual = model(x, plans)  # whole: the stencil path does not chunk
    for frequency in expected.field_type.frequencies:
        assert torch.allclose(actual[frequency], expected[frequency], atol=ATOL[dtype], rtol=0.0), (
            f"frequency {frequency}: max |diff| {float((actual[frequency] - expected[frequency]).abs().max()):.3e}"
        )


def test_the_models_gradients_agree_between_the_two_paths() -> None:
    discretization, plans = build_tensor_grid_geometry("legendre-gauss", *SHAPE, [DEEP, CONV])
    model = _model(torch.float64)
    flat = _conditioned(discretization.n_nodes, torch.float64).to_flat()
    probe = torch.randn(discretization.n_nodes, model.output_field_type.dimension, dtype=torch.float64,
                        generator=torch.Generator().manual_seed(53))
    parameters = list(model.parameters())

    def run(geometry) -> list[torch.Tensor]:
        leaf = flat.clone().requires_grad_(True)
        out = model(SO2FeatureField.from_flat(CONDITIONED_FIELD_TYPE, leaf), geometry)
        return list(torch.autograd.grad((out.to_flat() * probe).sum(), [leaf, *parameters]))

    for name, a, b in zip(["input", *(n for n, _ in model.named_parameters())], run(plans), run(discretization), strict=True):
        assert torch.allclose(a, b, atol=1e-9, rtol=0.0), f"{name}: max |diff| {float((a - b).abs().max()):.3e}"


def test_plans_lacking_a_cutoff_of_the_model_are_refused() -> None:
    _, plans = build_tensor_grid_geometry("legendre-gauss", *SHAPE, [DEEP])
    model = _model(torch.float64)
    with pytest.raises(ValueError, match="No stencil plan at cutoff 0.4"):
        model(_conditioned(plans.n_nodes, torch.float64), plans)
