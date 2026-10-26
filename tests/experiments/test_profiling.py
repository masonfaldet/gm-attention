"""Tests for experiments.profiling: the path switches, their restoration and the timing table."""

from __future__ import annotations

import torch

from experiments.profiling import (
    MULTI_GRID_PATHS,
    SINGLE_GRID_PATHS,
    PathConfig,
    format_table,
    profile_paths,
    select_paths,
    switched,
)
from gm_attention.nn import fused_attention, fused_factored, fused_pairs, kernels
from gm_attention.nn.convolution import SO2Convolution
from gm_attention.nn.stencil_plan import build_tensor_grid_geometry
from gm_attention.representations.field_type import SO2FeatureField, SO2FieldType

FIELD_TYPE = SO2FieldType({0: 2, 1: 1, 2: 1})
CUTOFF = 0.6


def test_switched_sets_the_modes_for_the_configuration_and_restores_them() -> None:
    saved = (fused_pairs.MODE, fused_factored.MODE, fused_attention.MODE, kernels.FACTORED)
    cpu = torch.device("cpu")
    with switched(PathConfig("p", fused="pairs", fused_attention=True), cpu) as counts:
        assert (fused_pairs.MODE, fused_factored.MODE, fused_attention.MODE, kernels.FACTORED) == ("cpu", "0", "cpu", False)
        assert set(counts) == {
            "fused_pairs.forward_general", "fused_factored.forward", "fused_factored.forward_aggregated",
            "fused_attention.softmax", "fused_attention.softmax_head", "fused_attention.forward_depthwise",
        }
    with switched(PathConfig("f", fused="factored"), torch.device("cuda")):
        assert (fused_pairs.MODE, fused_factored.MODE, fused_attention.MODE, kernels.FACTORED) == ("0", "1", "0", True)
    with switched(PathConfig("n"), cpu):
        assert (fused_pairs.MODE, fused_factored.MODE, fused_attention.MODE, kernels.FACTORED) == ("0", "0", "0", False)
    assert (fused_pairs.MODE, fused_factored.MODE, fused_attention.MODE, kernels.FACTORED) == saved


def test_switched_counts_the_calls_into_the_fused_kernels() -> None:
    """The counters wrap the entry points: a layer that enters one is counted, native code is not."""
    discretization, _ = build_tensor_grid_geometry("legendre-gauss", 8, 16, [CUTOFF])
    torch.manual_seed(0)
    module = SO2Convolution(FIELD_TYPE, FIELD_TYPE, cutoff=CUTOFF, depthwise=True, dtype=torch.float64)
    x = SO2FeatureField.from_flat(FIELD_TYPE, torch.randn(discretization.n_nodes, FIELD_TYPE.dimension, dtype=torch.float64))
    with switched(PathConfig("native"), torch.device("cpu")) as counts:
        module(x, discretization)
    assert not any(counts.values())
    with switched(PathConfig("fused", fused_attention=True), torch.device("cpu")) as counts:
        module(x, discretization)
    assert counts["fused_attention.forward_depthwise"] == 1


def test_profile_paths_times_every_path_and_compares_against_the_first() -> None:
    discretization, plans = build_tensor_grid_geometry("legendre-gauss", 8, 16, [CUTOFF])
    torch.manual_seed(1)
    module = SO2Convolution(FIELD_TYPE, FIELD_TYPE, cutoff=CUTOFF, dtype=torch.float64)
    x = SO2FeatureField.from_flat(FIELD_TYPE, torch.randn(discretization.n_nodes, FIELD_TYPE.dimension, dtype=torch.float64))
    paths = select_paths(SINGLE_GRID_PATHS, ["native", "stencil"])
    results = profile_paths(
        paths, lambda config: (lambda: module(x, plans if config.stencil else discretization)),
        list(module.parameters()), device=torch.device("cpu"), repeats=1, warmup=0,
    )
    assert [r.config.name for r in results] == ["native", "stencil"]
    assert all(r.error is None and r.forward_min > 0 and r.train_min > 0 for r in results)
    assert results[0].max_difference == 0.0 and results[1].max_difference < 1e-10
    assert results[0].fused_calls == {} and results[1].fused_calls == {}
    table = format_table(results, torch.device("cpu"))
    assert "native" in table and "stencil" in table and "none" in table

    def failing(config: PathConfig):
        def run():
            raise RuntimeError("no such kernel here")
        return run

    broken = profile_paths([PathConfig("broken")], failing, [], device=torch.device("cpu"), repeats=1, warmup=0)
    assert broken[0].error is not None and "no such kernel" in broken[0].error
    assert "unavailable" in format_table(broken, torch.device("cpu"))


def test_path_selection_and_the_multi_grid_set() -> None:
    assert [p.name for p in MULTI_GRID_PATHS] == [
        "native", "fused-pairs", "fused-factored", "stencil", "stencil+fused-pairs", "stencil+fused-factored",
    ]
    assert [p.name for p in select_paths(MULTI_GRID_PATHS, ["stencil", "native"])] == ["stencil", "native"]
    assert select_paths(SINGLE_GRID_PATHS, None) == list(SINGLE_GRID_PATHS)
    assert PathConfig("x", stencil=True, fused="factored").description == "stencil plans, fused factored"
