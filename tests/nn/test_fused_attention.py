"""The scores-and-softmax op and the depthwise mix (``nn/fused/attention.h``)
through their CPU driver: the layout plan, the same attention weights,
values and gradients as the eager paths through the multi-head layer
(batched and per head, either window, the general and the factored value
order) and the depthwise convolution, through the segmentation model's
training step, and through the layers' chunked evaluation, whose chunk
geometries are views of the parent's."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import pytest
import torch

from gm_attention.discretization.edge_geometry import compute_edge_geometry
from gm_attention.discretization.frames import build_tangent_frames
from gm_attention.discretization.grids.icosphere import build_icosphere_grid
from gm_attention.nn import fused_attention, kernels
from gm_attention.nn.attention import (
    SO2MHSA,
    SO2Attention,
    _softmax_weights,
    _stacked_projection,
)
from gm_attention.nn.convolution import SO2Convolution
from gm_attention.representations.field_type import SO2FeatureField, SO2FieldType

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "stanford_2d3ds"))

FIELD_TYPE = SO2FieldType({0: 6, 1: 4, 2: 2})
CUTOFF = 0.55


@pytest.fixture(scope="module")
def _built_extension() -> None:
    """The CPU driver, built on first use; skip if no compiler is at hand."""
    try:
        fused_attention.extension(cuda=False)
    except Exception as error:  # noqa: BLE001 - any build failure means "not here"
        pytest.skip(f"fused attention extension did not build here: {error}")


@pytest.fixture
def fused_cpu(monkeypatch: pytest.MonkeyPatch, _built_extension: None) -> list[str]:
    """The fused ops on, for one test; the list records the fused calls by name."""
    calls: list[str] = []
    originals = {name: getattr(fused_attention, name) for name in ("forward_depthwise", "softmax", "softmax_head")}

    def counting(name: str):
        def wrapped(*args, **kwargs):
            calls.append(name)
            return originals[name](*args, **kwargs)
        return wrapped

    for name in originals:
        monkeypatch.setattr(fused_attention, name, counting(name))
    monkeypatch.setattr(fused_attention, "MODE", "cpu")
    yield calls
    monkeypatch.setattr(fused_attention, "MODE", "0")


def _edges(radius: float = CUTOFF):
    grid = build_icosphere_grid(subdivisions=2, dtype=torch.float64)  # 162 nodes
    frames = build_tangent_frames(grid.nodes)
    return compute_edge_geometry(grid, frames, radius, max_frequency=2 * FIELD_TYPE.max_frequency)


def _steerable_mhsa(window: str, heads: int = 2, seed: int = 0) -> SO2MHSA:
    torch.manual_seed(seed)
    return SO2MHSA(FIELD_TYPE, heads, cutoff=CUTOFF, window=window, batched=True, value_kind="steerable",  # type: ignore[arg-type]
                   n_radial_basis=5, radial_hidden_features=(8,), dtype=torch.float64)


def _leaf(n_nodes: int, seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(n_nodes, FIELD_TYPE.dimension, dtype=torch.float64, generator=generator).requires_grad_()


def _run(module: SO2MHSA, flat: torch.Tensor, edges, probe: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    module.zero_grad(set_to_none=True)
    flat.grad = None
    y = module(SO2FeatureField.from_flat(FIELD_TYPE, flat), edges).to_flat()
    (y * probe).sum().backward()
    grads = {name: p.grad.clone() for name, p in module.named_parameters() if p.grad is not None}
    return y.detach(), grads, flat.grad.clone()


def test_the_plan_lays_out_heads_and_frequencies() -> None:
    p = fused_attention.plan(_steerable_mhsa("hard"))
    # per head {0: 3, 1: 2, 2: 1}; flat blocks: 6 scalars, 4 x 2, 2 x 2; complex coefficients 4 a_n per frequency
    assert p.rows == ((0, 3, 0, 0, 1), (1, 2, 6, 0, 2), (2, 1, 14, 8, 2))
    assert (p.heads, p.width, p.a_0, p.complex_width, p.max_order) == (2, 18, 3, 12, 4)


@pytest.mark.parametrize("normalized_mix", [False, True])
def test_the_model_step_matches_through_the_fused_kernel(normalized_mix: bool, fused_cpu: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """The segmentation model's training step (icosphere stages, batched heads,
    steerable values, the depthwise mix normalized or not): the same loss and
    parameter gradients to 1e-11 with the scores-and-softmax op and the fused
    mix on and off."""
    from test_batching import CPU, IMG, _hierarchy, _images, _model

    from experiments.stanford_2d3ds.train import SegmentationScorer, gm_step

    hierarchy = _hierarchy()
    scorer = SegmentationScorer.build(IMG, grid="equiangular", label_smoothing=0.0, device=CPU)
    step = gm_step(hierarchy, IMG, scorer)
    images, targets = _images(2, seed=3)
    results = {}
    for mode in ("0", "cpu"):
        monkeypatch.setattr(fused_attention, "MODE", mode)
        model = _model(value_kind="steerable", stem_kind="pointwise", drop_path_rate=0.0, decoder_dropout=0.0,
                       normalized_mix=normalized_mix).train()
        output = step(model, (images, targets), CPU)
        output.loss.backward()
        results[mode] = (float(output.loss), {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None})
    # per block, the softmax op and the mix once in the forward (the blocks run directly, nothing is recomputed), in the "cpu" pass only
    blocks = sum(model.depths)
    assert fused_cpu.count("softmax") == blocks and fused_cpu.count("forward_depthwise") == blocks
    assert results["cpu"][0] == pytest.approx(results["0"][0], rel=1e-12)
    assert set(results["cpu"][1]) == set(results["0"][1])
    for name, expected in results["0"][1].items():
        torch.testing.assert_close(results["cpu"][1][name], expected, rtol=1e-11, atol=1e-11, msg=name)


@pytest.mark.parametrize("mode", ["0", "cpu"])
def test_the_segmentation_step_matches_under_the_factored_switches(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The segmentation model as the reported arm runs it (steerable values,
    batched heads, IcoEncode merges, the normalized readout, two panoramas
    as one union graph) under GM_FACTORED_KERNEL with the eager factored
    order or the fused kernel's CPU driver: the same loss and parameter
    gradients as the general kernels to 1e-11; every steerable kernel's
    weighted sum went through the aggregated op."""
    from test_batching import CPU, IMG, _hierarchy, _images, _model

    from experiments.stanford_2d3ds.train import SegmentationScorer, gm_step
    from gm_attention.nn import fused_factored

    if mode == "cpu":
        try:
            fused_factored.extension(cuda=False)
        except Exception as error:  # noqa: BLE001
            pytest.skip(f"fused factored extension did not build here: {error}")
    monkeypatch.setattr(fused_attention, "MODE", "0")
    hierarchy = _hierarchy()
    scorer = SegmentationScorer.build(IMG, grid="equiangular", label_smoothing=0.0, device=CPU)
    step = gm_step(hierarchy, IMG, scorer)
    images, targets = _images(2, seed=3)
    settings = {"value_kind": "steerable", "stem_kind": "pointwise", "normalized_readout": True, "drop_path_rate": 0.0,
                "decoder_dropout": 0.0}
    ops: list[str] = []
    original_apply = fused_factored.FactoredAggregate.apply
    monkeypatch.setattr(fused_factored.FactoredAggregate, "apply", lambda *a, **k: (ops.append("aggregate"), original_apply(*a, **k))[1])
    results = {}
    for factored in (False, True):
        monkeypatch.setattr(kernels, "FACTORED", factored)
        monkeypatch.setattr(fused_factored, "MODE", mode if factored else "0")
        model = _model(**settings).train()
        output = step(model, (images, targets), CPU)
        output.loss.backward()
        results[factored] = (float(output.loss), {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None})
    assert (len(ops) > 0) == (mode == "cpu")
    assert results[True][0] == pytest.approx(results[False][0], rel=1e-12)
    assert set(results[True][1]) == set(results[False][1])
    for name, expected in results[False][1].items():
        torch.testing.assert_close(results[True][1][name], expected, rtol=1e-11, atol=1e-11, msg=name)


def _conv(window: str, normalize: bool, seed: int = 0) -> SO2Convolution:
    torch.manual_seed(seed)
    return SO2Convolution(FIELD_TYPE, FIELD_TYPE, cutoff=CUTOFF, window=window, depthwise=True, normalize=normalize,  # type: ignore[arg-type]
                          n_radial_basis=5, radial_hidden_features=(8,), dtype=torch.float64)


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("window", ["hard", "quartic"])
def test_the_driver_matches_the_eager_depthwise_convolution(window: str, normalize: bool, fused_cpu: list[str],
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    """The depthwise mix through the fused walk with the weights given:
    outputs, the input gradient and every parameter gradient, to 1e-12."""
    edges = _edges()
    n_nodes = edges.row_ptr.shape[0] - 1
    module = _conv(window, normalize)
    flat = _leaf(n_nodes, seed=5)
    probe = torch.randn(n_nodes, FIELD_TYPE.dimension, dtype=torch.float64, generator=torch.Generator().manual_seed(6))
    monkeypatch.setattr(fused_attention, "MODE", "0")
    y_eager, grads_eager, dx_eager = _run(module, flat, edges, probe)  # type: ignore[arg-type]
    monkeypatch.setattr(fused_attention, "MODE", "cpu")
    y_fused, grads_fused, dx_fused = _run(module, flat, edges, probe)  # type: ignore[arg-type]
    assert fused_cpu == ["forward_depthwise"]
    torch.testing.assert_close(y_fused, y_eager, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(dx_fused, dx_eager, rtol=1e-12, atol=1e-12)
    assert set(grads_fused) == set(grads_eager)
    for name, expected in grads_eager.items():
        torch.testing.assert_close(grads_fused[name], expected, rtol=1e-12, atol=1e-12, msg=name)


# ---- The scores-and-softmax op: steerable values, whose messages come from the value kernels. ----


def _small_edges(radius: float = 0.9):
    grid = build_icosphere_grid(subdivisions=1, dtype=torch.float64)  # 42 nodes
    frames = build_tangent_frames(grid.nodes)
    return compute_edge_geometry(grid, frames, radius, max_frequency=2 * FIELD_TYPE.max_frequency)


@pytest.mark.parametrize("window", ["hard", "quartic"])
def test_the_softmax_op_matches_the_eager_weights(window: str, fused_cpu: list[str]) -> None:
    """alpha from the op equals the eager scores-and-softmax to 1e-12, and every
    target's weights sum to one per head."""
    edges = _edges()
    n_nodes = edges.row_ptr.shape[0] - 1
    module = _steerable_mhsa(window)
    heads = [cast(SO2Attention, head) for head in module.heads]
    x = SO2FeatureField.from_flat(FIELD_TYPE, _leaf(n_nodes).detach())
    with torch.no_grad():
        queries = _stacked_projection([head.query for head in heads], x)
        keys = _stacked_projection([head.key for head in heads], x)
        eager = _softmax_weights(heads, queries, keys, edges, module.num_heads)
        assert fused_attention.softmax_enabled(module, x, edges)
        fused = fused_attention.softmax(module, queries.to_flat(), keys.to_flat(), edges)
    assert fused_cpu == ["softmax"]
    torch.testing.assert_close(fused, eager, rtol=1e-12, atol=1e-12)
    sums = torch.zeros(n_nodes, module.num_heads, dtype=torch.float64).index_add_(0, edges.target_index, fused)
    torch.testing.assert_close(sums, torch.ones_like(sums), rtol=1e-12, atol=1e-12)


def test_the_softmax_op_passes_gradcheck(_built_extension: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """``Softmax.apply`` against finite differences on a small neighbourhood graph, float64."""
    monkeypatch.setattr(fused_attention, "MODE", "cpu")
    edges = _small_edges()
    n_nodes = edges.row_ptr.shape[0] - 1
    module = _steerable_mhsa("quartic")
    p = fused_attention.plan(module)
    g = fused_attention.geometry(edges, module.cutoff, module.heads[0].window, torch.float64)
    generator = torch.Generator().manual_seed(5)
    q = torch.randn(n_nodes, FIELD_TYPE.dimension, dtype=torch.float64, generator=generator).requires_grad_()
    k = torch.randn(n_nodes, FIELD_TYPE.dimension, dtype=torch.float64, generator=generator).requires_grad_()
    inputs = (q, k, g.transport, g.bias, g.inside, fused_attention.descriptors(module, p, q.device), g.row_ptr, g.source,
              p.heads, float(module.heads[0]._score_scale), max(g.max_degree, 1))
    assert torch.autograd.gradcheck(fused_attention.Softmax.apply, inputs, eps=1e-6, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize("window", ["hard", "quartic"])
@pytest.mark.parametrize("factored", [False, True])
def test_the_batched_steerable_attention_matches_through_the_softmax_op(
    window: str, factored: bool, fused_cpu: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The batched multi-head layer with steerable values (the general and the
    factored value order): outputs, the input gradient and every parameter
    gradient agree to 1e-12 with the op on and off."""
    monkeypatch.setattr(kernels, "FACTORED", factored)
    edges = _edges()
    n_nodes = edges.row_ptr.shape[0] - 1
    module = _steerable_mhsa(window)
    flat = _leaf(n_nodes)
    probe = torch.randn(n_nodes, FIELD_TYPE.dimension, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    monkeypatch.setattr(fused_attention, "MODE", "0")
    y_eager, grads_eager, dx_eager = _run(module, flat, edges, probe)
    monkeypatch.setattr(fused_attention, "MODE", "cpu")
    y_fused, grads_fused, dx_fused = _run(module, flat, edges, probe)
    assert fused_cpu == ["softmax"]  # the op ran once; the value messages come from the eager value kernels
    torch.testing.assert_close(y_fused, y_eager, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(dx_fused, dx_eager, rtol=1e-12, atol=1e-12)
    assert set(grads_fused) == set(grads_eager)
    for name, expected in grads_eager.items():
        torch.testing.assert_close(grads_fused[name], expected, rtol=1e-12, atol=1e-12, msg=name)


def test_the_chunked_steerable_evaluation_matches_through_the_softmax_op(fused_cpu: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    edges = _edges()
    n_nodes = edges.row_ptr.shape[0] - 1
    module = _steerable_mhsa("hard").eval()
    x = SO2FeatureField.from_flat(FIELD_TYPE, _leaf(n_nodes).detach())
    with torch.no_grad():
        monkeypatch.setattr(fused_attention, "MODE", "0")
        whole = module(x, edges).to_flat()
        monkeypatch.setattr(fused_attention, "MODE", "cpu")
        chunked = module.forward_chunked(x, edges, chunk_size=37).to_flat()
    # the shipped SO2MHSA chunks a batched layer per head (no batched chunk path): one per-head op per head and chunk
    assert fused_cpu.count("softmax_head") == module.num_heads * -(-n_nodes // 37)
    torch.testing.assert_close(chunked, whole, rtol=1e-12, atol=1e-12)


def test_softmax_enabled_refuses_what_the_op_does_not_cover(monkeypatch: pytest.MonkeyPatch) -> None:
    edges = _edges()
    module = _steerable_mhsa("hard")
    x = SO2FeatureField.from_flat(FIELD_TYPE, _leaf(edges.row_ptr.shape[0] - 1).detach())
    monkeypatch.setattr(fused_attention, "MODE", "0")
    assert not fused_attention.softmax_enabled(module, x, edges)
    monkeypatch.setattr(fused_attention, "MODE", "1")
    assert not fused_attention.softmax_enabled(module, x, edges)  # "1" is CUDA tensors only
    monkeypatch.setattr(fused_attention, "MODE", "cpu")
    assert fused_attention.softmax_enabled(module, x, edges)
    monkeypatch.setattr(fused_attention, "MAX_SLOTS", 1)
    assert not fused_attention.softmax_enabled(module, x, edges)


@pytest.mark.parametrize("window", ["hard", "quartic"])
@pytest.mark.parametrize("factored", [False, True])
def test_the_sequential_steerable_attention_matches_through_the_softmax_op(
    window: str, factored: bool, fused_cpu: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The per-head path (``batched=False``, the SWE arms' attention) with
    steerable values: one op per head, outputs and every gradient to 1e-12."""
    monkeypatch.setattr(kernels, "FACTORED", factored)
    edges = _edges()
    n_nodes = edges.row_ptr.shape[0] - 1
    torch.manual_seed(0)
    module = SO2MHSA(FIELD_TYPE, 2, cutoff=CUTOFF, window=window, batched=False, value_kind="steerable",  # type: ignore[arg-type]
                     n_radial_basis=5, radial_hidden_features=(8,), dtype=torch.float64)
    flat = _leaf(n_nodes)
    probe = torch.randn(n_nodes, FIELD_TYPE.dimension, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    monkeypatch.setattr(fused_attention, "MODE", "0")
    y_eager, grads_eager, dx_eager = _run(module, flat, edges, probe)
    monkeypatch.setattr(fused_attention, "MODE", "cpu")
    y_fused, grads_fused, dx_fused = _run(module, flat, edges, probe)
    assert fused_cpu == ["softmax_head", "softmax_head"]
    torch.testing.assert_close(y_fused, y_eager, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(dx_fused, dx_eager, rtol=1e-12, atol=1e-12)
    assert set(grads_fused) == set(grads_eager)
    for name, expected in grads_eager.items():
        torch.testing.assert_close(grads_fused[name], expected, rtol=1e-12, atol=1e-12, msg=name)


def test_the_sequential_chunked_evaluation_matches_through_the_softmax_op(fused_cpu: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    edges = _edges()
    n_nodes = edges.row_ptr.shape[0] - 1
    torch.manual_seed(0)
    module = SO2MHSA(FIELD_TYPE, 2, cutoff=CUTOFF, window="hard", batched=False, value_kind="steerable",  # type: ignore[arg-type]
                     n_radial_basis=5, radial_hidden_features=(8,), dtype=torch.float64).eval()
    x = SO2FeatureField.from_flat(FIELD_TYPE, _leaf(n_nodes).detach())
    with torch.no_grad():
        monkeypatch.setattr(fused_attention, "MODE", "0")
        whole = module(x, edges).to_flat()
        monkeypatch.setattr(fused_attention, "MODE", "cpu")
        chunked = module.forward_chunked(x, edges, chunk_size=37).to_flat()
    assert fused_cpu.count("softmax_head") == 2 * -(-n_nodes // 37)  # one op per head per chunk
    torch.testing.assert_close(chunked, whole, rtol=1e-12, atol=1e-12)


def test_chunk_geometries_are_views_of_the_parent_and_nothing_accumulates(fused_cpu: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """Chunked forwards register their chunks; each chunk's geometry is a
    slice of the parent's cached tables (no copy), and neither the geometry
    cache nor the chunk registry grows across forwards (an entry per chunk
    per forward once filled a card)."""
    edges = _edges()
    n_nodes = edges.row_ptr.shape[0] - 1
    module = _steerable_mhsa("quartic").eval()
    x = SO2FeatureField.from_flat(FIELD_TYPE, _leaf(n_nodes).detach())
    fused_attention._GEOMETRIES.clear()
    fused_attention._CHUNKS.clear()
    with torch.no_grad():
        whole = module(x, edges).to_flat()
        for _ in range(4):
            chunked = module.forward_chunked(x, edges, chunk_size=37).to_flat()
        torch.testing.assert_close(chunked, whole, rtol=1e-12, atol=1e-12)
        parent = fused_attention.geometry(edges, module.cutoff, module.heads[0].window, torch.float64)
        assert len(fused_attention._GEOMETRIES) == 1  # the parent's entry alone; chunks are never cached
        fused_attention._purge()
        assert len(fused_attention._CHUNKS) == 0  # every chunk has died with its forward
        # a live chunk's geometry: views into the parent's tables, its own row_ptr, the parent's masses
        first, last = int(edges.row_ptr[37]), int(edges.row_ptr[74])
        chunk = edges.select(torch.arange(first, last))
        fused_attention.register_chunk(chunk, edges, first, last)
        g = fused_attention.geometry(chunk, module.cutoff, module.heads[0].window, torch.float64)
        assert g.base is parent and g.span == (first, last) and g.transport.data_ptr() == parent.transport[first:last].data_ptr()
        assert g.row_ptr is not parent.row_ptr and g.max_degree == int((chunk.row_ptr[1:] - chunk.row_ptr[:-1]).max())
        torch.testing.assert_close(g.aggregation_weights(True, 1e-6), parent.aggregation_weights(True, 1e-6)[first:last])
        assert len(fused_attention._GEOMETRIES) == 1


def test_the_softmax_op_takes_its_shift_over_the_window_alone(fused_cpu: list[str]) -> None:
    """The eager path's float32 boundary case through the driver: an edge at
    the cutoff (quartic window 0) whose score beats the self-loop's by more
    than -log(tiny) ~ 87 must not set the per-target shift, or every
    in-window weight underflows to 0/0."""
    from gm_attention.discretization.edge_geometry import EdgeGeometry
    from gm_attention.representations import so2

    field_type = SO2FieldType({0: 1})
    attn = SO2Attention(field_type, field_type, field_type, cutoff=1.0, window="quartic", bias=False)
    with torch.no_grad():
        attn.query.weight["0"].fill_(1.0)
        attn.key.weight["0"].fill_(1.0)
    target, source = torch.tensor([0, 0, 1]), torch.tensor([0, 1, 1])
    distances, zeros = torch.tensor([0.0, 1.0, 0.0]), torch.zeros(3)
    edges = EdgeGeometry(
        edge_index=torch.stack((target, source)), row_ptr=torch.tensor([0, 2, 3]),
        log_coordinates=torch.stack((distances, zeros), dim=-1), distances=distances, polar_angles=zeros,
        transport_angles=zeros, tangent_quadrature_weights=torch.ones(3),
        angular_phases=so2.phase_powers(zeros, 2), transport_phases=so2.phase_powers(zeros, 2),
    )
    x = SO2FeatureField(field_type, {0: torch.tensor([[15.0], [30.0]], requires_grad=True)})  # scores 225 and 450
    output = attn(x, edges).to_flat()
    assert fused_cpu == ["softmax_head"]
    assert torch.isfinite(output).all()
    assert torch.allclose(output[0], torch.tensor([15.0]) * attn.value(  # the self-loop's message alone
        SO2FeatureField(field_type, {0: torch.tensor([[1.0]])}), _self_loop(edges)).to_flat()[0])
    output.sum().backward()
    assert torch.isfinite(x[0].grad).all()


def _self_loop(edges):
    """The first edge of ``edges`` (node 0's self-loop) alone."""
    return edges.select(torch.tensor([0]), n_nodes=1)
