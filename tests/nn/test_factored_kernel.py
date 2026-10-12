"""The factored evaluation of the steerable kernel (the per-edge readout
regrouped so that the channel mixing happens on nodes): the same function
as the per-edge readout, on every block kind, through the layers and the
model."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize

from gm_attention.nn import kernels
from gm_attention.nn._scatter import gather_to_edges
from gm_attention.nn.kernels import SO2SteerableKernel
from gm_attention.representations.field_type import SO2FieldType

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_kernels import _random_field, _synthetic_edges


@pytest.fixture
def factored(monkeypatch: pytest.MonkeyPatch):
    """The switch on for one test, whatever the environment says."""
    monkeypatch.setattr(kernels, "FACTORED", True)
    yield
    monkeypatch.setattr(kernels, "FACTORED", False)


def _edges(n_edges: int = 400, max_frequency: int = 4, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return _synthetic_edges(
        torch.rand(n_edges, dtype=torch.float64, generator=g) * 0.3,
        torch.rand(n_edges, dtype=torch.float64, generator=g) * 6.28,
        torch.rand(n_edges, dtype=torch.float64, generator=g) * 6.28,
        self_loop=torch.rand(n_edges, generator=g) < 0.1,
        max_frequency=max_frequency,
    )


@pytest.mark.parametrize(
    ("field_in", "field_out", "bias"),
    [
        ({0: 4, 1: 3, 2: 2}, {0: 3, 1: 2, 2: 2}, True),  # every block kind: real, 0->m, n->0, linear and antilinear
        ({0: 4, 1: 3, 2: 2}, {0: 3, 1: 2, 2: 2}, False),  # no bias: h~ = (1, h) rides on zero columns
        ({0: 4}, {0: 5, 1: 3, 2: 2}, True),  # the encoder's shape: scalars in
        ({0: 4, 1: 3, 2: 2}, {0: 3}, True),  # the decoder's shape: scalars out
        ({1: 3}, {2: 2}, True),  # no scalar rows at all: no real head
    ],
)
def test_the_factored_kernel_is_the_per_edge_readout_regrouped(field_in: dict, field_out: dict, bias: bool) -> None:
    fin, fout = SO2FieldType(field_in), SO2FieldType(field_out)
    kernel = SO2SteerableKernel(fin, fout, cutoff=0.3, n_radial_basis=6, radial_hidden_features=[16], bias=bias, dtype=torch.float64)
    edges = _edges()
    x = _random_field(fin, (2,), generator=torch.Generator().manual_seed(3))
    reference = kernel(gather_to_edges(x, edges.source_index), edges)
    factored = kernel.forward_factored(x, edges)
    assert factored.field_type == fout and factored.batch_shape == reference.batch_shape
    for frequency in fout.frequencies:
        torch.testing.assert_close(factored[frequency], reference[frequency], rtol=1e-12, atol=1e-12)
    # the node-side mixes can be handed in, and every block is present once
    products = kernel.node_products(x)
    assert set(products) == set(kernel._blocks()) and len(kernel._blocks()) == len(kernel._coefficient_order) + int(kernel.real_head is not None)
    again = kernel.forward_factored(x, edges, products)
    for frequency in fout.frequencies:
        torch.testing.assert_close(again[frequency], reference[frequency], rtol=1e-12, atol=1e-12)


def test_the_stacked_weights_follow_a_parameter_update() -> None:
    """Without gradients the node products' stacked block weights are cached
    against the heads' parameter versions, and an in-place update (an
    optimizer step) rebuilds them; under gradients they are rebuilt every
    call, into the forward's own graph."""
    fin, fout = SO2FieldType({0: 2, 1: 3}), SO2FieldType({0: 2, 1: 2})
    kernel = SO2SteerableKernel(fin, fout, cutoff=0.3, n_radial_basis=5, radial_hidden_features=[8], bias=True, dtype=torch.float64)
    edges = _edges(n_edges=50)
    x = _random_field(fin, (2,), generator=torch.Generator().manual_seed(9))
    with torch.no_grad():
        first = kernel._stacked_block_weights(torch.device("cpu"))
        assert kernel._stacked_block_weights(torch.device("cpu")) is first  # cached
        assert sum(len(blocks) for _, blocks, _ in first) == len(kernel._blocks())
    with torch.enable_grad():
        tracked = kernel._stacked_block_weights(torch.device("cpu"))
        assert tracked is not first and tracked[0][2].requires_grad  # rebuilt into this forward's graph
        assert kernel._stacked_block_weights(torch.device("cpu")) is not tracked  # and never reused
    with torch.no_grad():
        assert kernel._stacked_block_weights(torch.device("cpu")) is first  # the cache survives the tracked calls
        kernel.complex_head.weight.mul_(1.5)
        assert kernel._stacked_block_weights(torch.device("cpu")) is not first  # the version moved
    reference = kernel(gather_to_edges(x, edges.source_index), edges)
    factored = kernel.forward_factored(x, edges)
    for frequency in fout.frequencies:
        torch.testing.assert_close(factored[frequency], reference[frequency], rtol=1e-12, atol=1e-12)


def test_the_block_weights_reproduce_the_readout() -> None:
    """``sum_t h~_t W_t`` on an edge is the readout's coefficient block there."""
    fin, fout = SO2FieldType({0: 2, 1: 3}), SO2FieldType({0: 2, 1: 2})
    kernel = SO2SteerableKernel(fin, fout, cutoff=0.3, n_radial_basis=5, radial_hidden_features=[8], bias=True, dtype=torch.float64)
    edges = _edges(n_edges=7)
    h, _ = kernel._edge_factors(edges)  # (7, 9)
    hidden = h[:, 1:]
    coefficients = torch.nn.functional.linear(hidden, kernel.complex_head.weight, kernel.complex_head.bias)  # the readout
    block = (1, 1, "linear")
    rows = kernel._pair_ranges[block]
    b_m, a_n = fout.multiplicity(1), fin.multiplicity(1)
    from gm_attention.representations import so2

    readout_block = so2.as_complex(coefficients[:, rows].reshape(7, b_m, a_n, 2).contiguous())
    weights = kernel._block_weights(block)  # (b_m, a_n, T + 1)
    regrouped = torch.einsum("et,bat->eba", h.to(weights.dtype), weights)
    torch.testing.assert_close(regrouped, readout_block, rtol=1e-12, atol=1e-12)


# The zero-shot bundle's gm_l5_100 primary already selects the single-level L5 arm
# (its sl_attention model group); only the tiny-run overrides remain.
TINY = [
    "++model.subdivisions=2", "grid.subdivisions=2", "device=cpu", "training.batch_size=2",
    "data.nlat=32", "data.nlon=64", "data.solver_substeps=1", "data.num_train_examples=2",
    "data.num_val_examples=1", "data.normalization_examples=1",
]


def _tiny_model():
    from experiments.shallow_water_paper.train import build_scalar_experiment

    with initialize(version_base=None, config_path="../../SWE_256x512_ZeroShot/configs"):
        cfg = compose(config_name="gm_l5_100", overrides=TINY)
    torch.manual_seed(0)
    experiment = build_scalar_experiment(cfg, torch.device("cpu"))
    experiment.model.eval()
    return experiment


def _run(experiment, x_nodes: torch.Tensor, chunk_size: int | None = None):
    from experiments.shallow_water_paper.train import CONDITIONED_FIELD_TYPE
    from gm_attention.representations.field_type import SO2FeatureField

    model = experiment.model
    if chunk_size is not None:
        model.chunk_size = chunk_size
    field = SO2FeatureField(CONDITIONED_FIELD_TYPE, {0: x_nodes})
    with torch.no_grad():
        return model(field, experiment.geometry)[0]


def test_the_model_agrees_through_both_paths_and_across_chunks(factored) -> None:
    """The whole arm (encoder convolution, attention blocks, decoder) under the
    switch against the per-edge path, unchunked and chunked. The arm is the
    zero-shot bundle's single-level L5 model (``gm_l5_100``), built here on a
    level-2 icosphere."""
    experiment = _tiny_model()
    n = experiment.n_nodes
    x = torch.randn(n, 4, generator=torch.Generator().manual_seed(5))
    original = experiment.model.chunk_size
    kernels.FACTORED = False
    reference = _run(experiment, x)
    kernels.FACTORED = True
    ours = _run(experiment, x)
    torch.testing.assert_close(ours, reference, rtol=1e-4, atol=1e-5)
    chunked = _run(experiment, x, chunk_size=max(1, n // 5))
    torch.testing.assert_close(chunked, reference, rtol=1e-4, atol=1e-5)
    experiment.model.chunk_size = original
