"""The fused factored kernel (``nn/fused/factored.h``) through its CPU
driver: the plan and packing, the source grouping, and the same values as
the eager factored path on every block kind, through the layers and the
model, chunked or not."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize

from gm_attention.nn import fused_factored, kernels
from gm_attention.nn._scatter import gather_to_edges
from gm_attention.nn.kernels import SO2SteerableKernel
from gm_attention.representations.field_type import SO2FeatureField, SO2FieldType

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_kernels import _random_field, _synthetic_edges


@pytest.fixture(scope="module")
def _built_extension() -> None:
    """The CPU driver, built on first use; skip if no compiler is at hand."""
    try:
        fused_factored.extension(cuda=False)
    except Exception as error:  # noqa: BLE001 - any build failure means "not here"
        pytest.skip(f"fused factored extension did not build here: {error}")


def _count_ops(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every ``Factored.apply`` / ``FactoredAggregate.apply`` by name,
    so a test that switches the driver on can assert that it ran."""
    calls: list[str] = []
    for name in ("Factored", "FactoredAggregate"):
        cls = getattr(fused_factored, name)
        original = cls.apply

        def wrapped(*args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(cls, "apply", wrapped)
    return calls


@pytest.fixture
def fused_cpu(monkeypatch: pytest.MonkeyPatch, _built_extension: None) -> list[str]:
    """The eager factored switch and the fused driver on, for one test; the
    list records the driver's ops by name."""
    monkeypatch.setattr(kernels, "FACTORED", True)
    monkeypatch.setattr(fused_factored, "MODE", "cpu")
    calls = _count_ops(monkeypatch)
    yield calls
    monkeypatch.setattr(fused_factored, "MODE", "0")
    monkeypatch.setattr(kernels, "FACTORED", False)


def _edges(n_edges: int = 600, max_frequency: int = 4, seed: int = 0, dtype: torch.dtype = torch.float64):
    g = torch.Generator().manual_seed(seed)
    edges = _synthetic_edges(
        torch.rand(n_edges, dtype=torch.float64, generator=g) * 0.3,
        torch.rand(n_edges, dtype=torch.float64, generator=g) * 6.28,
        torch.rand(n_edges, dtype=torch.float64, generator=g) * 6.28,
        self_loop=torch.rand(n_edges, generator=g) < 0.1,
        max_frequency=max_frequency,
    )
    return edges if dtype == torch.float64 else edges.to(dtype=dtype)


def test_the_plan_describes_every_block_and_packs_the_mixes_in_its_order() -> None:
    fin, fout = SO2FieldType({0: 4, 1: 3, 2: 2}), SO2FieldType({0: 3, 1: 2, 2: 2})
    kernel = SO2SteerableKernel(fin, fout, cutoff=0.3, n_radial_basis=6, radial_hidden_features=[16], bias=True, dtype=torch.float64)
    p = fused_factored.plan(kernel)
    assert sorted(b.block for b in p.blocks) == sorted(kernel._blocks()) and p.d_out == fout.dimension == 3 + 4 + 4
    assert [b.out_offset for b in p.blocks] == sorted(b.out_offset for b in p.blocks)  # grouped by output frequency
    real = p.blocks[0]
    assert real.kind == 0 and real.b_m == 3 and real.out_offset == 0 and p.wr == 3
    by_block = {b.block: b for b in p.blocks}
    slices = fout.slices()
    assert (by_block[(0, 1, "linear")].angular, by_block[(0, 1, "linear")].transport) == (1, 0)  # e^{i m phi}, no transport
    assert (by_block[(1, 0, "linear")].angular, by_block[(1, 0, "linear")].transport, by_block[(1, 0, "linear")].take_real) == (1, -1, 1)
    assert (by_block[(2, 1, "linear")].angular, by_block[(2, 1, "linear")].transport) == (-1, 2)  # e^{i (m - n) phi} e^{i n gamma}
    assert (by_block[(2, 1, "antilinear")].angular, by_block[(2, 1, "antilinear")].transport) == (3, -2)  # e^{i (m + n) phi} e^{-i n gamma}
    assert by_block[(2, 1, "linear")].out_offset == slices[1].start and by_block[(1, 2, "linear")].out_offset == slices[2].start
    assert by_block[(1, 1, "linear")].power == kernel._pair_powers[(1, 1, "linear")]
    offsets = [b.z_offset for b in p.blocks if b.kind == 1]
    widths = [b.b_m for b in p.blocks if b.kind == 1]
    assert offsets == [sum(widths[:i]) for i in range(len(widths))] and p.wc == sum(widths)
    x = _random_field(fin, (5,), generator=torch.Generator().manual_seed(2))
    products = kernel.node_products(x)
    zc, zr = fused_factored.pack(kernel, products)
    assert zc.shape == (5, 17, p.wc, 2) and zr.shape == (5, 17, p.wr) and zc.dtype == torch.float64
    assert fused_factored.pack(kernel, products)[0] is zc and fused_factored.PACKED in products  # packed once, remembered on the dict
    block = by_block[(2, 1, "antilinear")]
    packed = torch.view_as_complex(zc[:, :, block.z_offset:block.z_offset + block.b_m].contiguous())
    torch.testing.assert_close(packed, products[(2, 1, "antilinear")])
    torch.testing.assert_close(zr, products[(0, 0, "real")])
    rows = fused_factored.descriptors(kernel, torch.device("cpu"))
    assert rows.shape == (len(p.blocks), 8) and rows.dtype == torch.int32 and rows[0].tolist() == real.row()


def test_source_groups_partition_the_edges_by_source_and_are_cached() -> None:
    source = torch.tensor([3, 1, 3, 0, 1, 3])
    order, ptr, groups = fused_factored.source_groups(source)
    assert groups.tolist() == [0, 1, 3] and ptr.tolist() == [0, 1, 3, 6]
    assert source[order].tolist() == [0, 1, 1, 3, 3, 3] and order.tolist() == [3, 1, 4, 0, 2, 5]  # stable within a source
    key = ("test", 0, 6)
    first = fused_factored.source_groups(source, key)
    assert fused_factored.source_groups(torch.zeros(6, dtype=torch.long), key) is first  # served from the cache under the key
    del fused_factored._GROUPS[key]
    # tied to an owner: the same live tensor hits, another tensor or a freed one misses
    owner = torch.arange(6)
    held = fused_factored.source_groups(source, key, owner=owner)
    assert fused_factored.source_groups(torch.zeros(6, dtype=torch.long), key, owner=owner) is held
    assert fused_factored.source_groups(source, key, owner=torch.arange(6)) is not held  # a different geometry under a recurring key
    replaced = fused_factored.source_groups(source, key, owner=owner)
    del owner
    assert fused_factored.source_groups(source, key, owner=torch.arange(6)) is not replaced  # the owner died: recomputed
    assert fused_factored.source_groups(source, key) is not replaced  # an ownerless call never takes an owned entry
    del fused_factored._GROUPS[key]


@pytest.mark.parametrize(
    ("field_in", "field_out", "bias"),
    [
        ({0: 16, 1: 16, 2: 16}, {0: 8, 1: 8, 2: 8}, True),  # the SL arm's value kernel
        ({0: 4, 1: 3, 2: 2}, {0: 3, 1: 2, 2: 2}, False),
        ({0: 4}, {0: 5, 1: 3, 2: 2}, True),
        ({0: 4, 1: 3, 2: 2}, {0: 3}, True),
        ({1: 3}, {2: 2}, True),
        # the wide tier: the segmentation arms' coarse merges and readouts (blocks over 16 wide, mixes over 256)
        ({0: 12, 1: 4, 2: 4}, {0: 18, 1: 6, 2: 6}, True),  # a stage-3 merge: b_m 18, 2wc + wr 210
        ({0: 18, 1: 6, 2: 6}, {0: 24, 1: 8, 2: 8}, True),  # a stage-4 merge: b_m 24, 2wc + wr 280
        ({0: 24, 1: 8, 2: 8}, {0: 32}, True),  # a readout: b_m 32 on the scalar slot
    ],
)
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_the_driver_is_the_per_edge_readout(field_in: dict, field_out: dict, bias: bool, dtype: torch.dtype, _built_extension: None,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    fin, fout = SO2FieldType(field_in), SO2FieldType(field_out)
    kernel = SO2SteerableKernel(fin, fout, cutoff=0.3, n_radial_basis=6, radial_hidden_features=[16], bias=bias, dtype=dtype)
    edges = _edges(dtype=dtype)
    x = _random_field(fin, (2,), generator=torch.Generator().manual_seed(3), dtype=dtype)
    monkeypatch.setattr(fused_factored, "MODE", "cpu")
    with torch.no_grad():
        reference = kernel(gather_to_edges(x, edges.source_index), edges)
        assert fused_factored.enabled(kernel, x, edges)
        fused = kernel.forward_factored(x, edges)
        explicit = fused_factored.forward(kernel, x, edges, kernel.node_products(x), fused_factored.source_groups(edges.source_index))
    tolerance = {"rtol": 1e-12, "atol": 1e-12} if dtype == torch.float64 else {"rtol": 1e-4, "atol": 1e-5}
    for frequency in fout.frequencies:
        torch.testing.assert_close(fused[frequency], reference[frequency], **tolerance)
        torch.testing.assert_close(explicit[frequency], reference[frequency], **tolerance)
    with torch.enable_grad():
        assert fused_factored.enabled(kernel, x, edges)  # forward and backward: the path serves training too


@pytest.mark.parametrize(
    ("field_in", "field_out", "bias"),
    [
        ({0: 16, 1: 16, 2: 16}, {0: 8, 1: 8, 2: 8}, True),
        ({0: 4, 1: 3, 2: 2}, {0: 3, 1: 2, 2: 2}, False),
        ({0: 4}, {0: 5, 1: 3, 2: 2}, True),
        ({0: 4, 1: 3, 2: 2}, {0: 3}, True),
        ({1: 3}, {2: 2}, True),
        ({0: 12, 1: 4, 2: 4}, {0: 18, 1: 6, 2: 6}, True),  # the wide tier, as in the forward test
        ({0: 18, 1: 6, 2: 6}, {0: 24, 1: 8, 2: 8}, True),
        ({0: 24, 1: 8, 2: 8}, {0: 32}, True),
    ],
)
def test_the_driver_backward_is_the_eager_paths_gradient(field_in: dict, field_out: dict, bias: bool, _built_extension: None,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    """The gradients of a random linear functional of the messages, for the
    input field and the heads' weights, through the fused driver's backward
    and through autograd of the eager factored path (and of the per-edge
    readout), float64."""
    fin, fout = SO2FieldType(field_in), SO2FieldType(field_out)
    kernel = SO2SteerableKernel(fin, fout, cutoff=0.3, n_radial_basis=6, radial_hidden_features=[16], bias=bias, dtype=torch.float64)
    edges = _edges(n_edges=300)
    g = torch.Generator().manual_seed(11)
    base = _random_field(fin, (2,), generator=g)
    weights = {f: torch.randn_like(v) for f, v in _random_field(fout, (300,), generator=g).coefficients.items()}

    def gradients(run):
        x = SO2FeatureField(fin, {f: v.clone().requires_grad_(True) for f, v in base.coefficients.items()})
        kernel.zero_grad(set_to_none=True)
        out = run(x)
        loss = sum((out[f] * weights[f]).sum() for f in fout.frequencies)
        loss.backward()
        return {f: x[f].grad.clone() for f in fin.frequencies}, {n: p.grad.clone() for n, p in kernel.named_parameters() if p.grad is not None}

    monkeypatch.setattr(fused_factored, "MODE", "cpu")
    with torch.enable_grad():
        fused_x, fused_p = gradients(lambda x: kernel.forward_factored(x, edges))
    monkeypatch.setattr(fused_factored, "MODE", "0")
    with torch.enable_grad():
        eager_x, eager_p = gradients(lambda x: kernel.forward_factored(x, edges))
        readout_x, readout_p = gradients(lambda x: kernel(gather_to_edges(x, edges.source_index), edges))
    assert set(fused_p) == set(eager_p) == set(readout_p) and fused_p  # every parameter the readout trains gets a gradient
    for f in fin.frequencies:
        torch.testing.assert_close(fused_x[f], eager_x[f], rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(fused_x[f], readout_x[f], rtol=1e-10, atol=1e-10)
    for n in fused_p:
        torch.testing.assert_close(fused_p[n], eager_p[n], rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(fused_p[n], readout_p[n], rtol=1e-10, atol=1e-10)


def test_the_bounds_admit_the_wide_tier_and_refuse_beyond_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """The plan records the widest block; ``enabled`` admits multiplicities up
    to MAX_BM and packed widths up to MAX_C, and refuses the next size."""
    monkeypatch.setattr(fused_factored, "MODE", "cpu")
    edges = _edges(n_edges=40)

    def admitted(field_in: dict, field_out: dict) -> tuple[bool, int, int]:
        kernel = SO2SteerableKernel(SO2FieldType(field_in), SO2FieldType(field_out), cutoff=0.3, n_radial_basis=6,
                                    radial_hidden_features=[16], bias=True, dtype=torch.float64)
        x = _random_field(SO2FieldType(field_in), (2,), generator=torch.Generator().manual_seed(0))
        p = fused_factored.plan(kernel)
        return fused_factored.enabled(kernel, x, edges), p.max_bm, 2 * p.wc + p.wr

    assert admitted({0: 18, 1: 6, 2: 6}, {0: 24, 1: 8, 2: 8}) == (True, 24, 280)  # over the narrow tier on both counts
    at_cap = admitted({0: 4, 1: 2, 2: 2}, {0: fused_factored.MAX_BM})
    assert at_cap[0] is True and at_cap[1] == fused_factored.MAX_BM
    assert admitted({0: 4, 1: 2, 2: 2}, {0: fused_factored.MAX_BM + 1})[0] is False
    at_c = admitted({0: 24, 1: 8, 2: 8}, {0: 32, 1: 8, 2: 8})  # the packed width exactly at MAX_C
    assert at_c == (True, 32, fused_factored.MAX_C)
    over_c = admitted({0: 24, 1: 8, 2: 8}, {0: 32, 1: 10, 2: 10})  # blocks within MAX_BM, packed width past MAX_C
    assert over_c[1] <= fused_factored.MAX_BM and over_c[2] > fused_factored.MAX_C and over_c[0] is False


def _gradcheck_inputs(edges, n_nodes_in: int = 2):
    """A small kernel of every block kind on ``edges``, its extension inputs
    with the differentiable ones as leaves, float64."""
    fin, fout = SO2FieldType({0: 2, 1: 2, 2: 1}), SO2FieldType({0: 2, 1: 1, 2: 1})
    kernel = SO2SteerableKernel(fin, fout, cutoff=0.3, n_radial_basis=4, radial_hidden_features=[5], bias=True, dtype=torch.float64)
    x = _random_field(fin, (n_nodes_in,), generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        i = fused_factored.inputs(kernel, x, edges, None)
    order, group_ptr, group_source = fused_factored.source_groups(i.source)
    leaves = (i.h.clone().requires_grad_(True), i.zc.clone().requires_grad_(True), i.zr.clone().requires_grad_(True))
    return kernel, i, leaves, (order, group_ptr, group_source)


def test_the_extension_passes_gradcheck(_built_extension: None) -> None:
    """``Factored.apply`` against finite differences on small random inputs
    of every block kind, float64."""
    edges = _edges(n_edges=12)
    _, i, (h, zc, zr), (order, group_ptr, group_source) = _gradcheck_inputs(edges)
    assert i.h.shape[1] + 1 == i.zc.shape[1]  # the trunk's own features; the constant one is the kernel's
    inputs = (h, zc, zr, i.angular, i.transport, i.rho, i.desc, i.source, order, group_ptr, group_source, i.plan.d_out, i.plan.max_bm)
    assert torch.autograd.gradcheck(fused_factored.Factored.apply, inputs, eps=1e-6, atol=1e-7, rtol=1e-6)


def test_the_aggregated_op_passes_gradcheck(_built_extension: None) -> None:
    """``FactoredAggregate.apply`` (the rows summed per target under weights,
    the backward through the target index) against finite differences,
    the weights a leaf too, on a small neighbourhood graph, float64."""
    from test_fused_attention import _small_edges

    edges = _small_edges()
    n_nodes = edges.row_ptr.shape[0] - 1
    _, i, (h, zc, zr), (order, group_ptr, group_source) = _gradcheck_inputs(edges, n_nodes)
    weights = torch.rand(edges.n_edges, dtype=torch.float64, generator=torch.Generator().manual_seed(8)).requires_grad_(True)
    inputs = (h, zc, zr, weights, i.angular, i.transport, i.rho, i.desc, i.source, edges.target_index.contiguous(),
              edges.row_ptr.contiguous(), order, group_ptr, group_source, i.plan.d_out, i.plan.max_bm)
    assert torch.autograd.gradcheck(fused_factored.FactoredAggregate.apply, inputs, eps=1e-6, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize(
    ("field_in", "field_out", "bias"),
    [
        ({0: 16, 1: 16, 2: 16}, {0: 8, 1: 8, 2: 8}, True),  # the SL arm's value kernel
        ({0: 4, 1: 3, 2: 2}, {0: 3, 1: 2, 2: 2}, False),
        ({0: 4}, {0: 5, 1: 3, 2: 2}, True),
        ({0: 4, 1: 3, 2: 2}, {0: 3}, True),
        ({1: 3}, {2: 2}, True),
        ({0: 18, 1: 6, 2: 6}, {0: 24, 1: 8, 2: 8}, True),  # the wide tier
    ],
)
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_the_aggregated_driver_is_the_weighted_segment_sum(field_in: dict, field_out: dict, bias: bool, dtype: torch.dtype,
                                                            _built_extension: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """``forward_factored_aggregated`` through the driver against the eager
    form (the readout's messages weighted and segment-summed): the node
    field, the input gradient, every parameter gradient and the weights'
    gradient, on a neighbourhood graph where every node is a target."""
    from test_fused_attention import _edges as _graph_edges

    from gm_attention.nn._scatter import scatter_add_to_nodes

    fin, fout = SO2FieldType(field_in), SO2FieldType(field_out)
    torch.manual_seed(7)
    kernel = SO2SteerableKernel(fin, fout, cutoff=0.55, n_radial_basis=6, radial_hidden_features=[16], bias=bias, dtype=dtype)
    edges = _graph_edges()
    if dtype != torch.float64:
        edges = edges.to(dtype=dtype)
    n_nodes = edges.row_ptr.shape[0] - 1
    g = torch.Generator().manual_seed(3)
    base = _random_field(fin, (n_nodes,), generator=g, dtype=dtype)
    base_weights = torch.rand(edges.n_edges, dtype=dtype, generator=g)
    probe = {f: torch.randn_like(v) for f, v in _random_field(fout, (n_nodes,), generator=g, dtype=dtype).coefficients.items()}

    def run(fused: bool):
        monkeypatch.setattr(fused_factored, "MODE", "cpu" if fused else "0")
        x = SO2FeatureField(fin, {f: v.clone().requires_grad_(True) for f, v in base.coefficients.items()})
        weights = base_weights.clone().requires_grad_(True)
        kernel.zero_grad(set_to_none=True)
        if fused:
            assert fused_factored.enabled(kernel, x, edges)
            out = kernel.forward_factored_aggregated(x, edges, weights)
        else:
            messages = kernel(gather_to_edges(x, edges.source_index), edges)
            out = SO2FeatureField(fout, {f: scatter_add_to_nodes(weights.reshape(-1, *([1] * (messages[f].ndim - 1))) * messages[f],
                                                                 edges.row_ptr) for f in fout.frequencies})
        sum((out[f] * probe[f]).sum() for f in fout.frequencies).backward()
        return ({f: out[f].detach() for f in fout.frequencies}, {f: x[f].grad.clone() for f in fin.frequencies},
                {n: p.grad.clone() for n, p in kernel.named_parameters() if p.grad is not None}, weights.grad.clone())

    # float32: the two orders sum the same terms differently; the noise is ~1e-6 of a tensor's scale
    # (gradients reach 160), and an entry left small by cancellation carries the large terms' noise
    tolerance = {"rtol": 1e-11, "atol": 1e-11} if dtype == torch.float64 else {"rtol": 1e-4, "atol": 1e-3}
    ref_out, ref_dx, ref_dp, ref_dw = run(fused=False)
    out, dx, dp, dw = run(fused=True)
    for f in fout.frequencies:
        torch.testing.assert_close(out[f], ref_out[f], **tolerance)
    for f in fin.frequencies:
        torch.testing.assert_close(dx[f], ref_dx[f], **tolerance)
    assert set(dp) == set(ref_dp) and dp
    for n in dp:
        torch.testing.assert_close(dp[n], ref_dp[n], **tolerance, msg=n)
    torch.testing.assert_close(dw, ref_dw, **tolerance)


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
    with torch.no_grad():
        return model(SO2FeatureField(CONDITIONED_FIELD_TYPE, {0: x_nodes}), experiment.geometry)[0]


@pytest.mark.parametrize("mode", ["0", "cpu"])
def test_the_model_trains_through_the_factored_paths(mode: str, _built_extension: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """A training step of the whole tiny arm, two samples' losses accumulated
    before the comparison as the study's training does, in chunks under
    rematerialization: the parameter gradients through the eager factored
    path (``mode`` 0) and the fused driver (``cpu``) against the per-edge
    readout's. The eager path rebuilds its stacked weights under gradients,
    so the second sample's forward never meets a consumed graph."""
    from gm_attention.nn import attention
    from gm_attention.representations.field_type import SO2FeatureField as Field

    experiment = _tiny_model()
    from experiments.shallow_water_paper.train import CONDITIONED_FIELD_TYPE

    model = experiment.model
    model.train()
    n = experiment.n_nodes
    g = torch.Generator().manual_seed(21)
    samples = [torch.randn(n, 4, generator=g) for _ in range(2)]
    targets = [torch.randn(n, 3, generator=g) for _ in range(2)]
    original = model.chunk_size
    model.chunk_size = max(1, n // 4)
    monkeypatch.setattr(attention, "REMATERIALIZE_CHUNKS", True)

    def gradients():
        model.zero_grad(set_to_none=True)
        for x, y in zip(samples, targets, strict=True):
            out = model(Field(CONDITIONED_FIELD_TYPE, {0: x}), experiment.geometry)[0]
            ((out - y) ** 2).mean().backward()
        return {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}

    calls = _count_ops(monkeypatch)
    try:
        monkeypatch.setattr(kernels, "FACTORED", False)
        monkeypatch.setattr(fused_factored, "MODE", "0")
        reference = gradients()
        monkeypatch.setattr(kernels, "FACTORED", True)
        monkeypatch.setattr(fused_factored, "MODE", mode)
        ours = gradients()
    finally:
        model.chunk_size = original
        model.eval()
    # every layer's weighted sum through the aggregated op, in the forward and the rematerialized recompute
    assert (set(calls) == {"FactoredAggregate"} and len(calls) > 0) if mode == "cpu" else calls == []
    assert set(ours) == set(reference) and len(ours) > 0
    for name in reference:
        torch.testing.assert_close(ours[name], reference[name], rtol=1e-4, atol=1e-6, msg=name)


def test_the_model_agrees_through_the_fused_driver_chunked_or_not(fused_cpu: list[str]) -> None:
    """The whole arm (encoder convolution, attention blocks, decoder) through
    the fused driver against the per-edge path; the chunked loop groups each
    chunk's edges by source and caches the grouping against the geometry."""
    experiment = _tiny_model()
    n = experiment.n_nodes
    x = torch.randn(n, 4, generator=torch.Generator().manual_seed(5))
    original = experiment.model.chunk_size
    kernels.FACTORED = False
    fused_factored.MODE = "0"
    reference = _run(experiment, x)
    kernels.FACTORED = True
    fused_factored.MODE = "cpu"
    fused_factored._GROUPS.clear()
    whole = _run(experiment, x)
    assert fused_cpu and set(fused_cpu) == {"FactoredAggregate"}  # the driver's aggregated op, nothing else
    torch.testing.assert_close(whole, reference, rtol=1e-4, atol=1e-5)
    chunked = _run(experiment, x, chunk_size=max(1, n // 5))
    torch.testing.assert_close(chunked, reference, rtol=1e-4, atol=1e-5)
    cached = len(fused_factored._GROUPS)
    assert cached > 0
    again = _run(experiment, x, chunk_size=max(1, n // 5))
    assert len(fused_factored._GROUPS) == cached  # the second forward hit the cache for every chunk
    torch.testing.assert_close(again, chunked)
    experiment.model.chunk_size = original



def test_the_batched_heads_take_the_factored_path_for_steerable_values(fused_cpu: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """SO2MHSA's batched path with steerable values (the segmentation arms'
    setting) under the factored switch and the fused CPU driver equals the
    general path: outputs, the input gradient and every parameter gradient
    to 1e-12; the general per-edge readout is not built."""
    from test_fused_attention import _edges

    from gm_attention.nn import attention
    from gm_attention.nn.attention import SO2MHSA

    field_type = SO2FieldType({0: 6, 1: 4, 2: 2})
    edges = _edges()
    n_nodes = edges.row_ptr.shape[0] - 1
    torch.manual_seed(0)
    module = SO2MHSA(field_type, 2, cutoff=0.55, batched=True, value_kind="steerable", n_radial_basis=5,
                     radial_hidden_features=(8,), dtype=torch.float64)
    generator = torch.Generator().manual_seed(1)
    flat = torch.randn(n_nodes, field_type.dimension, dtype=torch.float64, generator=generator).requires_grad_()
    probe = torch.randn(n_nodes, field_type.dimension, dtype=torch.float64, generator=generator)

    def run() -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        module.zero_grad(set_to_none=True)
        flat.grad = None
        y = module(SO2FeatureField.from_flat(field_type, flat), edges).to_flat()
        (y * probe).sum().backward()
        return y.detach(), {n: p.grad.clone() for n, p in module.named_parameters() if p.grad is not None}, flat.grad.clone()

    calls: list[str] = []
    original = attention._stacked_kernels
    monkeypatch.setattr(attention, "_stacked_kernels", lambda *a, **k: (calls.append("general"), original(*a, **k))[1])
    monkeypatch.setattr(kernels, "FACTORED", False)
    y_general, grads_general, dx_general = run()
    assert calls == ["general"]
    monkeypatch.setattr(kernels, "FACTORED", True)
    y_factored, grads_factored, dx_factored = run()
    assert calls == ["general"]  # the factored pass never built the per-edge readout
    assert fused_cpu == ["FactoredAggregate"] * 2  # one aggregated op per head, through the driver
    torch.testing.assert_close(y_factored, y_general, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(dx_factored, dx_general, rtol=1e-12, atol=1e-12)
    for name, expected in grads_general.items():
        torch.testing.assert_close(grads_factored[name], expected, rtol=1e-12, atol=1e-12, msg=name)


# ---- The layers the arms run, through the driver's aggregated op, float64. ----


def _graph():
    from test_fused_attention import _edges

    return _edges()


def _compare_runs(run, monkeypatch: pytest.MonkeyPatch, calls: list[str], expected_calls: list[str]) -> None:
    """``run()`` -> (output, input gradient, parameter gradients) under the
    general kernels and under the factored switch with the fused driver:
    equal to 1e-12, and the driver's ops as expected."""
    monkeypatch.setattr(kernels, "FACTORED", False)
    monkeypatch.setattr(fused_factored, "MODE", "0")
    y_ref, dx_ref, grads_ref = run()
    assert calls == []
    monkeypatch.setattr(kernels, "FACTORED", True)
    monkeypatch.setattr(fused_factored, "MODE", "cpu")
    y, dx, grads = run()
    assert calls == expected_calls
    torch.testing.assert_close(y, y_ref, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(dx, dx_ref, rtol=1e-12, atol=1e-12)
    assert set(grads) == set(grads_ref) and grads
    for name, expected in grads_ref.items():
        torch.testing.assert_close(grads[name], expected, rtol=1e-12, atol=1e-12, msg=name)


def _module_run(module, field_type: SO2FieldType, edges, chunk_size: int | None = None):
    n_nodes = edges.row_ptr.shape[0] - 1
    generator = torch.Generator().manual_seed(1)
    flat = torch.randn(n_nodes, field_type.dimension, dtype=torch.float64, generator=generator).requires_grad_()
    probe = torch.randn(n_nodes, module.field_type_out.dimension if hasattr(module, "field_type_out") else field_type.dimension,
                        dtype=torch.float64, generator=generator)

    def run():
        module.zero_grad(set_to_none=True)
        flat.grad = None
        x = SO2FeatureField.from_flat(field_type, flat)
        y = (module(x, edges) if chunk_size is None else module.forward_chunked(x, edges, chunk_size=chunk_size)).to_flat()
        (y * probe).sum().backward()
        return y.detach(), flat.grad.clone(), {n: p.grad.clone() for n, p in module.named_parameters() if p.grad is not None}

    return run


@pytest.mark.parametrize("chunk_size", [None, 37])
def test_the_sequential_heads_take_the_aggregated_op(chunk_size: int | None, fused_cpu: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """SO2Attention._attend (batched=False, the SWE arms' attention), whole or
    in target-range chunks under rematerialization: the aggregated op once
    per head per chunk, outputs and every gradient to 1e-12."""
    from gm_attention.nn import attention
    from gm_attention.nn.attention import SO2MHSA

    field_type = SO2FieldType({0: 6, 1: 4, 2: 2})
    edges = _graph()
    n_nodes = edges.row_ptr.shape[0] - 1
    torch.manual_seed(0)
    module = SO2MHSA(field_type, 2, cutoff=0.55, batched=False, value_kind="steerable", n_radial_basis=5,
                     radial_hidden_features=(8,), dtype=torch.float64).train()
    monkeypatch.setattr(attention, "REMATERIALIZE_CHUNKS", True)
    chunks = 1 if chunk_size is None else -(-n_nodes // chunk_size)
    recompute = 1 if chunk_size is None else 2  # each chunk's forward runs again in the backward
    _compare_runs(_module_run(module, field_type, edges, chunk_size), monkeypatch, fused_cpu,
                  ["FactoredAggregate"] * (2 * chunks * recompute))


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("chunk_size", [None, 37])
def test_the_convolution_takes_the_aggregated_op(normalize: bool, chunk_size: int | None, fused_cpu: list[str],
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """SO2Convolution._aggregate with a steerable kernel, the window mass
    normalization on or off, a linear self-interaction outside the sum,
    whole or chunked: the aggregated op once per chunk, everything to 1e-12."""
    from gm_attention.nn import attention
    from gm_attention.nn.convolution import SO2Convolution

    fin, fout = SO2FieldType({0: 6, 1: 4, 2: 2}), SO2FieldType({0: 3, 1: 2, 2: 2})
    edges = _graph()
    n_nodes = edges.row_ptr.shape[0] - 1
    torch.manual_seed(0)
    module = SO2Convolution(fin, fout, cutoff=0.55, window="quartic", self_interaction="linear", normalize=normalize,
                            n_radial_basis=5, radial_hidden_features=(8,), dtype=torch.float64).train()
    monkeypatch.setattr(attention, "REMATERIALIZE_CHUNKS", True)
    chunks = 1 if chunk_size is None else -(-n_nodes // chunk_size)
    recompute = 1 if chunk_size is None else 2
    _compare_runs(_module_run(module, fin, edges, chunk_size), monkeypatch, fused_cpu, ["FactoredAggregate"] * (chunks * recompute))


def test_the_batched_chunked_evaluation_takes_the_aggregated_op_and_caches_its_groups(fused_cpu: list[str]) -> None:
    """SO2MHSA.forward_chunked on a batched layer (it chunks per head through
    SO2Attention.forward_chunked) with steerable values: each chunk's aggregated op equals the whole
    forward, and the chunks' source groupings are cached under the parent
    geometry's key, so repeated forwards neither recompute them nor grow
    the cache."""
    from gm_attention.nn.attention import SO2MHSA

    field_type = SO2FieldType({0: 6, 1: 4, 2: 2})
    edges = _graph()
    n_nodes = edges.row_ptr.shape[0] - 1
    torch.manual_seed(0)
    module = SO2MHSA(field_type, 2, cutoff=0.55, batched=True, value_kind="steerable", n_radial_basis=5,
                     radial_hidden_features=(8,), dtype=torch.float64).eval()
    x = SO2FeatureField.from_flat(field_type, torch.randn(n_nodes, field_type.dimension, dtype=torch.float64,
                                                          generator=torch.Generator().manual_seed(1)))
    fused_factored._GROUPS.clear()
    with torch.no_grad():
        whole = module(x, edges).to_flat()
        assert fused_cpu == ["FactoredAggregate"] * 2
        cached = len(fused_factored._GROUPS)
        chunked = module.forward_chunked(x, edges, chunk_size=37).to_flat()
        chunks = -(-n_nodes // 37)
        assert fused_cpu == ["FactoredAggregate"] * (2 + 2 * chunks)
        grown = len(fused_factored._GROUPS)
        assert grown == cached + chunks  # one grouping per chunk range, keyed on the parent
        again = module.forward_chunked(x, edges, chunk_size=37).to_flat()
        assert len(fused_factored._GROUPS) == grown  # served from the cache: nothing added
    torch.testing.assert_close(chunked, whole, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(again, chunked, rtol=0, atol=0)


@pytest.mark.parametrize("batched", [False, True])
def test_the_launch_setting_assembles_both_ops(batched: bool, fused_cpu: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """The launch setting (GM_FACTORED_KERNEL, GM_FUSED_FACTORED and
    GM_FUSED_ATTENTION together): the scores-and-softmax op's weights feed
    the aggregated op, whose weight gradient feeds the softmax op's
    backward; outputs and every gradient equal the fully eager path's to
    1e-12, with one softmax op and one aggregated op per head."""
    from gm_attention.nn import fused_attention
    from gm_attention.nn.attention import SO2MHSA

    try:
        fused_attention.extension(cuda=False)
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"fused attention extension did not build here: {error}")
    field_type = SO2FieldType({0: 6, 1: 4, 2: 2})
    edges = _graph()
    torch.manual_seed(0)
    module = SO2MHSA(field_type, 2, cutoff=0.55, window="quartic", batched=batched, value_kind="steerable", n_radial_basis=5,
                     radial_hidden_features=(8,), dtype=torch.float64).train()
    softmax_calls: list[str] = []
    for name in ("softmax", "softmax_head"):
        original = getattr(fused_attention, name)
        monkeypatch.setattr(fused_attention, name, lambda *a, _n=name, _o=original, **k: (softmax_calls.append(_n), _o(*a, **k))[1])
    run = _module_run(module, field_type, edges)
    monkeypatch.setattr(fused_attention, "MODE", "0")
    monkeypatch.setattr(kernels, "FACTORED", False)
    monkeypatch.setattr(fused_factored, "MODE", "0")
    y_ref, dx_ref, grads_ref = run()
    assert fused_cpu == [] and softmax_calls == []
    monkeypatch.setattr(fused_attention, "MODE", "cpu")
    monkeypatch.setattr(kernels, "FACTORED", True)
    monkeypatch.setattr(fused_factored, "MODE", "cpu")
    y, dx, grads = run()
    monkeypatch.setattr(fused_attention, "MODE", "0")
    assert fused_cpu == ["FactoredAggregate"] * 2 and softmax_calls == (["softmax"] if batched else ["softmax_head"] * 2)
    torch.testing.assert_close(y, y_ref, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(dx, dx_ref, rtol=1e-12, atol=1e-12)
    for name, expected in grads_ref.items():
        torch.testing.assert_close(grads[name], expected, rtol=1e-12, atol=1e-12, msg=name)


def test_unchunked_layers_take_their_source_grouping_from_the_cache(fused_cpu: list[str]) -> None:
    """A convolution and a per-head attention layer called on the whole edge
    set (no chunk, no grouping of their own) key the grouping on the
    geometry: the second forward adds nothing to the cache, so no forward
    re-sorts the edges or synchronizes on their unique sources."""
    from gm_attention.nn.attention import SO2MHSA
    from gm_attention.nn.convolution import SO2Convolution

    fin, fout = SO2FieldType({0: 6, 1: 4, 2: 2}), SO2FieldType({0: 3, 1: 2, 2: 2})
    edges = _graph()
    n_nodes = edges.row_ptr.shape[0] - 1
    torch.manual_seed(0)
    conv = SO2Convolution(fin, fout, cutoff=0.55, n_radial_basis=5, radial_hidden_features=(8,), dtype=torch.float64).eval()
    heads = SO2MHSA(fin, 2, cutoff=0.55, batched=False, value_kind="steerable", n_radial_basis=5, radial_hidden_features=(8,),
                    dtype=torch.float64).eval()
    x = SO2FeatureField.from_flat(fin, torch.randn(n_nodes, fin.dimension, dtype=torch.float64, generator=torch.Generator().manual_seed(1)))
    fused_factored._GROUPS.clear()
    with torch.no_grad():
        first = (conv(x, edges).to_flat(), heads(x, edges).to_flat())
        assert fused_cpu == ["FactoredAggregate"] * 3
        cached = len(fused_factored._GROUPS)
        assert cached == 1  # one grouping of the one edge set, shared by every layer
        second = (conv(x, edges).to_flat(), heads(x, edges).to_flat())
        assert len(fused_factored._GROUPS) == cached
    for a, b in zip(first, second, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
