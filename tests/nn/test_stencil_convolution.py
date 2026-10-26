"""Tests for gm_attention.nn.stencil_convolution: the convolution and the depthwise mix on the fixed grid.

``SO2Convolution.forward`` is the oracle. Every test builds one
Gauss-Legendre discretization in the coordinate gauge and compares the
stencil path with the graph path to summation order: the general and the
depthwise kernel, both windows, the pointwise self-interaction and the
per-target normalization, in float64 and float32, with and without leading
batch axes, gradients with respect to the input and every parameter, and
the two consumers of the depthwise kernel, ``SO2TransformerBlock`` with its
convolution mixer and the segmentation model's block with its depthwise
mix, each with the stencil path swapped in for the graph path. The
per-copy depthwise tables are checked against the kernel's own forward on
the basis vectors of its type. Longitude shifts commute exactly, and the
dense stage traces under ``torch.compile(fullgraph=True)``.
"""

from __future__ import annotations

import itertools

import pytest
import torch
from torch import nn

from experiments.stanford_2d3ds.model import GmSegformerBlock
from gm_attention.discretization.stencil import (
    TensorGridStencil,
    build_tensor_grid_discretization,
    build_tensor_grid_stencil,
)
from gm_attention.nn.blocks import SO2TransformerBlock
from gm_attention.nn.convolution import SO2Convolution
from gm_attention.nn.depthwise import SO2DepthwiseKernel
from gm_attention.nn.stencil_convolution import (
    aggregate,
    convolution_tables,
    depthwise_kernel_matrices,
    stencil_convolution,
)
from gm_attention.nn.stencil_plan import StencilPlan, build_stencil_plan
from gm_attention.representations.field_type import SO2FeatureField, SO2FieldType

SHAPE = (10, 20)
CUTOFF = 0.6
TYPE_IN = SO2FieldType({0: 3, 1: 2, 2: 1})
TYPE_OUT = SO2FieldType({0: 2, 1: 2, 2: 1})
#: (field_type_in, field_type_out) pairs for the general kernel.
PAIRS = {
    "all-frequencies": (TYPE_IN, TYPE_OUT),
    "no-frequency-0-out": (SO2FieldType({0: 2, 1: 1}), SO2FieldType({1: 2, 2: 1})),
    "no-frequency-0-in": (SO2FieldType({1: 1, 2: 1}), SO2FieldType({0: 1, 1: 1})),
    "skipped-frequency": (SO2FieldType({0: 1, 2: 1}), SO2FieldType({1: 1, 2: 1})),
}
#: Types the depthwise kernel is tested on: with and without scalar copies, a single frequency.
DEPTHWISE_TYPES = {
    "all-frequencies": TYPE_IN,
    "no-frequency-0": SO2FieldType({1: 2, 2: 1}),
    "scalars-only": SO2FieldType({0: 3}),
}
ATOL = {torch.float64: 1e-10, torch.float32: 2e-4}


def _geometry(dtype: torch.dtype, shape: tuple[int, int] = SHAPE, cutoff: float = CUTOFF, max_buckets: int = 8,
              coverage: float | None = None):
    discretization = build_tensor_grid_discretization(
        "legendre-gauss", *shape, coverage_radius=cutoff if coverage is None else coverage, dtype=dtype
    )
    stencil = build_tensor_grid_stencil(discretization, cutoff, n_latitude=shape[0], n_longitude=shape[1])
    return discretization, build_stencil_plan(stencil, max_buckets=max_buckets)


def _field(field_type: SO2FieldType, batch_shape: tuple[int, ...], dtype: torch.dtype, seed: int) -> SO2FeatureField:
    generator = torch.Generator().manual_seed(seed)
    flat = torch.randn(*batch_shape, field_type.dimension, dtype=dtype, generator=generator)
    return SO2FeatureField.from_flat(field_type, flat)


def _convolution(dtype: torch.dtype, seed: int, *, depthwise: bool, pair: str = "all-frequencies",
                 cutoff: float = CUTOFF, **kwargs) -> SO2Convolution:
    torch.manual_seed(seed)
    type_in, type_out = PAIRS[pair]
    if depthwise:
        type_out = type_in
    return SO2Convolution(type_in, type_out, cutoff=cutoff, depthwise=depthwise, dtype=dtype, **kwargs)


def _assert_fields_close(a: SO2FeatureField, b: SO2FeatureField, atol: float) -> None:
    assert a.field_type == b.field_type
    assert a.batch_shape == b.batch_shape
    for frequency in a.field_type.frequencies:
        assert torch.allclose(a[frequency], b[frequency], atol=atol, rtol=0.0), (
            f"frequency {frequency}: max |diff| {float((a[frequency] - b[frequency]).abs().max()):.3e}"
        )


def _tiled(x: SO2FeatureField, count: int) -> SO2FeatureField:
    return SO2FeatureField(x.field_type, {f: t.reshape(count * x.batch_shape[-1], *t.shape[len(x.batch_shape):]) for f, t in x.coefficients.items()})


def _unfolded(y: SO2FeatureField, batch_shape: tuple[int, ...]) -> SO2FeatureField:
    return SO2FeatureField(y.field_type, {f: t.reshape(*batch_shape, *t.shape[1:]) for f, t in y.coefficients.items()})


def _rolled(field: SO2FeatureField, shift: int, shape: tuple[int, int]) -> SO2FeatureField:
    nlat, nlon = shape
    return SO2FeatureField(
        field.field_type,
        {f: t.reshape(nlat, nlon, *t.shape[1:]).roll(shift, dims=1).reshape(t.shape) for f, t in field.coefficients.items()},
    )


def _block_diagonal(tables: dict[int, torch.Tensor], field_type: SO2FieldType, n_slots: int) -> torch.Tensor:
    """The depthwise tables as one ``(n_slots, d, d)`` matrix per slot on the packed layout."""
    dimension = field_type.dimension
    full = torch.zeros(n_slots, dimension, dimension, dtype=next(iter(tables.values())).dtype)
    slices = field_type.slices()
    for frequency, multiplicity in field_type.items():
        block = slices[frequency]
        if frequency == 0:
            full[:, block, block] = torch.diag_embed(tables[0])
            continue
        for copy in range(multiplicity):
            rows = slice(block.start + 2 * copy, block.start + 2 * copy + 2)
            full[:, rows, rows] = tables[frequency][:, copy]
    return full


def _matrices_from_basis_vectors(kernel: SO2DepthwiseKernel, stencil: TensorGridStencil) -> torch.Tensor:
    d = kernel.field_type.dimension
    basis = torch.eye(d, dtype=stencil.dtype).repeat(stencil.n_slots, 1)
    messages = kernel(SO2FeatureField.from_flat(kernel.field_type, basis), stencil.repeated(d)).to_flat()
    return messages.reshape(stencil.n_slots, d, d).transpose(1, 2)  # (n_slots, d_out, d_in)


class _StencilMixer(nn.Module):
    """A convolution layer evaluated through the stencil path under the ``(x, edges)`` call of the graph path."""

    def __init__(self, convolution: SO2Convolution, plan: StencilPlan) -> None:
        super().__init__()
        self.convolution = convolution
        self.plan = plan

    def forward(self, x: SO2FeatureField, edges: object) -> SO2FeatureField:
        return stencil_convolution(self.convolution, x, self.plan)


@pytest.fixture(params=[torch.float64, torch.float32])
def dtype(request: pytest.FixtureRequest) -> torch.dtype:
    return request.param


# --------------------------------------------------------------------------
# The depthwise tables


@pytest.mark.parametrize("name", list(DEPTHWISE_TYPES))
def test_depthwise_matrices_match_the_kernel_on_basis_vectors(dtype: torch.dtype, name: str) -> None:
    field_type = DEPTHWISE_TYPES[name]
    _, plan = _geometry(dtype)
    torch.manual_seed(1)
    kernel = SO2DepthwiseKernel(field_type, cutoff=CUTOFF, dtype=dtype)
    tables = depthwise_kernel_matrices(kernel, plan.stencil)
    assert set(tables) == set(field_type.frequencies)
    for frequency, multiplicity in field_type.items():
        expected_shape = (plan.n_slots, multiplicity) if frequency == 0 else (plan.n_slots, multiplicity, 2, 2)
        assert tables[frequency].shape == expected_shape
    actual = _block_diagonal(tables, field_type, plan.n_slots)
    expected = _matrices_from_basis_vectors(kernel, plan.stencil)
    assert torch.allclose(actual, expected, atol=ATOL[dtype], rtol=0.0), f"max |diff| {float((actual - expected).abs().max()):.3e}"


def test_depthwise_matrices_differentiate_through_the_kernel_parameters() -> None:
    _, plan = _geometry(torch.float64)
    torch.manual_seed(2)
    kernel = SO2DepthwiseKernel(TYPE_IN, cutoff=CUTOFF, dtype=torch.float64)
    probe = torch.randn(plan.n_slots, TYPE_IN.dimension, TYPE_IN.dimension, dtype=torch.float64,
                        generator=torch.Generator().manual_seed(3))
    parameters = list(kernel.parameters())
    expected = torch.autograd.grad((_matrices_from_basis_vectors(kernel, plan.stencil) * probe).sum(), parameters)
    actual = torch.autograd.grad(
        (_block_diagonal(depthwise_kernel_matrices(kernel, plan.stencil), TYPE_IN, plan.n_slots) * probe).sum(), parameters
    )
    for name, a, b in zip([n for n, _ in kernel.named_parameters()], actual, expected, strict=True):
        assert torch.allclose(a, b, atol=1e-9, rtol=0.0), name


def test_depthwise_matrices_evaluate_the_radial_mlp_once_per_slot() -> None:
    _, plan = _geometry(torch.float64)
    kernel = SO2DepthwiseKernel(TYPE_IN, cutoff=CUTOFF, dtype=torch.float64)
    rows: list[int] = []
    handle = kernel.radial_basis.register_forward_hook(lambda module, inputs, output: rows.append(int(inputs[0].numel())))
    try:
        depthwise_kernel_matrices(kernel, plan.stencil)
    finally:
        handle.remove()
    assert rows == [plan.n_slots]


def test_convolution_tables_fold_the_window_and_the_quadrature_weight() -> None:
    _, plan = _geometry(torch.float64)
    geometry = plan.stencil.geometry
    weight = (1.0 - (geometry.distances / CUTOFF) ** 2) ** 2 * geometry.tangent_quadrature_weights
    general = _convolution(torch.float64, seed=4, depthwise=False, window="quartic", normalize=True)
    tables = convolution_tables(general, plan)
    assert tables.matrices is not None and tables.denominator is not None
    assert tables.depthwise_scalar is None and tables.depthwise_pairs is None
    assert tables.matrices.shape == (plan.n_slots + 1, TYPE_IN.dimension, TYPE_OUT.dimension)
    assert bool((tables.matrices[-1] == 0.0).all())
    mass = torch.zeros(plan.n_latitude, dtype=torch.float64)
    for row in range(plan.n_latitude):
        mass[row] = weight[plan.stencil.slots(row)].sum()
    assert torch.allclose(tables.denominator, mass.clamp_min(general.eps), atol=1e-14, rtol=0.0)
    depthwise = _convolution(torch.float64, seed=5, depthwise=True, window="quartic")
    tables = convolution_tables(depthwise, plan)
    assert tables.matrices is None and tables.denominator is None
    assert tables.depthwise_scalar is not None and tables.depthwise_pairs is not None
    raw = depthwise_kernel_matrices(depthwise.kernel, plan.stencil)
    assert torch.allclose(tables.depthwise_scalar[:-1], raw[0] * weight.unsqueeze(-1), atol=1e-14, rtol=0.0)
    pairs = torch.cat([raw[1], raw[2]], dim=1) * weight.reshape(-1, 1, 1, 1)  # frequencies ascending
    assert tables.depthwise_pairs.shape == (plan.n_slots + 1, 3, 2, 2)
    assert torch.allclose(tables.depthwise_pairs[:-1], pairs, atol=1e-14, rtol=0.0)
    assert bool((tables.depthwise_scalar[-1] == 0.0).all()) and bool((tables.depthwise_pairs[-1] == 0.0).all())


# --------------------------------------------------------------------------
# The convolution against the graph path


@pytest.mark.parametrize(
    "depthwise, window, self_interaction, normalize",
    list(itertools.product([False, True], ["hard", "quartic"], ["none", "linear"], [False, True])),
)
def test_convolution_matches_the_graph_path(dtype: torch.dtype, depthwise: bool, window: str, self_interaction: str, normalize: bool) -> None:
    discretization, plan = _geometry(dtype)
    module = _convolution(dtype, seed=6, depthwise=depthwise, window=window, self_interaction=self_interaction, normalize=normalize)
    x = _field(TYPE_IN, (plan.n_nodes,), dtype, seed=7)
    _assert_fields_close(stencil_convolution(module, x, plan), module(x, discretization), ATOL[dtype])


@pytest.mark.parametrize("pair", list(PAIRS))
def test_general_convolution_matches_the_graph_path_across_type_pairs(dtype: torch.dtype, pair: str) -> None:
    discretization, plan = _geometry(dtype)
    module = _convolution(dtype, seed=8, depthwise=False, pair=pair, window="quartic")
    x = _field(PAIRS[pair][0], (plan.n_nodes,), dtype, seed=9)
    _assert_fields_close(stencil_convolution(module, x, plan), module(x, discretization), ATOL[dtype])


@pytest.mark.parametrize("name", list(DEPTHWISE_TYPES))
def test_depthwise_convolution_matches_the_graph_path_across_types(dtype: torch.dtype, name: str) -> None:
    field_type = DEPTHWISE_TYPES[name]
    discretization, plan = _geometry(dtype)
    torch.manual_seed(10)
    module = SO2Convolution(field_type, field_type, cutoff=CUTOFF, depthwise=True, normalize=True, dtype=dtype)
    x = _field(field_type, (plan.n_nodes,), dtype, seed=11)
    _assert_fields_close(stencil_convolution(module, x, plan), module(x, discretization), ATOL[dtype])


@pytest.mark.parametrize("max_buckets", [1, 3, 100])
def test_convolution_matches_the_graph_path_for_every_bucket_count(max_buckets: int) -> None:
    discretization, plan = _geometry(torch.float64, max_buckets=max_buckets)
    module = _convolution(torch.float64, seed=12, depthwise=False, window="quartic", self_interaction="linear")
    x = _field(TYPE_IN, (plan.n_nodes,), torch.float64, seed=13)
    _assert_fields_close(stencil_convolution(module, x, plan), module(x, discretization), 1e-10)


@pytest.mark.parametrize("depthwise", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_convolution_gradients_match_the_graph_path(depthwise: bool, normalize: bool) -> None:
    discretization, plan = _geometry(torch.float64)
    module = _convolution(torch.float64, seed=14, depthwise=depthwise, window="quartic", self_interaction="linear", normalize=normalize)
    flat = _field(TYPE_IN, (plan.n_nodes,), torch.float64, seed=15).to_flat()
    probe = torch.randn(plan.n_nodes, module.field_type_out.dimension, dtype=torch.float64, generator=torch.Generator().manual_seed(16))
    parameters = list(module.parameters())

    def run(path) -> list[torch.Tensor]:
        leaf = flat.clone().requires_grad_(True)
        loss = (path(SO2FeatureField.from_flat(TYPE_IN, leaf)).to_flat() * probe).sum()
        return list(torch.autograd.grad(loss, [leaf, *parameters]))

    expected = run(lambda x: module(x, discretization))
    actual = run(lambda x: stencil_convolution(module, x, plan))
    for name, a, b in zip(["input", *(n for n, _ in module.named_parameters())], actual, expected, strict=True):
        assert torch.allclose(a, b, atol=1e-9, rtol=0.0), f"{name}: max |diff| {float((a - b).abs().max()):.3e}"


@pytest.mark.parametrize("shape", [(7, 13), (1, 8), (4, 2), (2, 1)])
@pytest.mark.parametrize("depthwise", [False, True])
def test_convolution_matches_the_graph_path_on_odd_and_degenerate_shapes(shape: tuple[int, int], depthwise: bool) -> None:
    discretization, plan = _geometry(torch.float64, shape=shape, cutoff=1.2)
    module = _convolution(torch.float64, seed=17, depthwise=depthwise, cutoff=1.2, window="quartic", normalize=True)
    x = _field(TYPE_IN, (plan.n_nodes,), torch.float64, seed=18)
    _assert_fields_close(stencil_convolution(module, x, plan), module(x, discretization), 1e-10)


@pytest.mark.parametrize("depthwise", [False, True])
def test_convolution_with_leading_batch_axes_matches_the_tiled_graph(dtype: torch.dtype, depthwise: bool) -> None:
    batch = (2, 3)
    discretization, plan = _geometry(dtype, max_buckets=3)
    module = _convolution(dtype, seed=19, depthwise=depthwise, self_interaction="linear", normalize=True)
    x = _field(TYPE_IN, (*batch, plan.n_nodes), dtype, seed=20)
    expected = _unfolded(module(_tiled(x, 6), discretization.tile(6)), (*batch, plan.n_nodes))
    _assert_fields_close(stencil_convolution(module, x, plan), expected, ATOL[dtype])


@pytest.mark.parametrize("depthwise", [False, True])
def test_convolution_is_exactly_equivariant_to_longitude_shifts(depthwise: bool) -> None:
    _, plan = _geometry(torch.float64)
    module = _convolution(torch.float64, seed=21, depthwise=depthwise, self_interaction="linear", normalize=True)
    x = _field(TYPE_IN, (plan.n_nodes,), torch.float64, seed=22)
    shifted = stencil_convolution(module, _rolled(x, 7, SHAPE), plan)
    expected = _rolled(stencil_convolution(module, x, plan), 7, SHAPE)
    for frequency in TYPE_IN.frequencies:
        assert torch.equal(shifted[frequency], expected[frequency])


@pytest.mark.parametrize("depthwise", [False, True])
def test_aggregate_compiles_as_one_graph(dtype: torch.dtype, depthwise: bool) -> None:
    _, plan = _geometry(dtype)
    module = _convolution(dtype, seed=23, depthwise=depthwise, window="quartic", normalize=True)
    x = _field(TYPE_IN, (2, plan.n_nodes), dtype, seed=24)
    tables = convolution_tables(module, plan)
    source = x.to_flat().reshape(2, plan.n_nodes, -1)
    torch._dynamo.reset()
    compiled = torch.compile(aggregate, backend="aot_eager", fullgraph=True)
    assert torch.allclose(compiled(plan, source, tables, TYPE_IN), aggregate(plan, source, tables, TYPE_IN), atol=ATOL[dtype], rtol=0.0)


def test_convolution_rejects_mismatches() -> None:
    _, plan = _geometry(torch.float64)
    module = _convolution(torch.float64, seed=25, depthwise=False)
    with pytest.raises(ValueError, match="type"):
        stencil_convolution(module, _field(TYPE_OUT, (plan.n_nodes,), torch.float64, seed=0), plan)
    with pytest.raises(ValueError, match="n_nodes"):
        stencil_convolution(module, _field(TYPE_IN, (plan.n_nodes + 1,), torch.float64, seed=0), plan)
    with pytest.raises(ValueError, match="dtype|float"):
        stencil_convolution(module, _field(TYPE_IN, (plan.n_nodes,), torch.float64, seed=0), plan.to(dtype=torch.float32))
    other = SO2Convolution(TYPE_IN, TYPE_OUT, cutoff=0.5, dtype=torch.float64)
    with pytest.raises(ValueError, match="cutoff"):
        stencil_convolution(other, _field(TYPE_IN, (plan.n_nodes,), torch.float64, seed=0), plan)


# --------------------------------------------------------------------------
# The consumers: a transformer block's convolution mixer and the segmentation block's depthwise mix


def test_transformer_block_with_its_convolution_mixer_on_the_stencil_matches_the_graph_path(dtype: torch.dtype) -> None:
    field_type = SO2FieldType({0: 4, 1: 2, 2: 2})
    discretization, plan = _geometry(dtype)
    torch.manual_seed(26)
    block = SO2TransformerBlock(field_type, num_heads=2, cutoff=CUTOFF, window="quartic", mixer="convolution", dtype=dtype).eval()
    x = _field(field_type, (plan.n_nodes,), dtype, seed=27)
    expected = block(x, discretization)
    block.attention = _StencilMixer(block.attention, plan)
    _assert_fields_close(block(x, discretization), expected, ATOL[dtype])


def test_segmentation_block_with_its_depthwise_mix_on_the_stencil_matches_the_graph_path(dtype: torch.dtype) -> None:
    """The segmentation model's block: attention on the graph at one cutoff, the normalized depthwise mix
    of the widened hidden type on the stencil at a smaller one."""
    field_type = SO2FieldType({0: 2, 1: 2, 2: 2})
    mix_cutoff = 0.4
    discretization, plan = _geometry(dtype, cutoff=mix_cutoff, coverage=CUTOFF)
    torch.manual_seed(28)
    block = GmSegformerBlock(
        field_type, num_heads=2, attention_cutoff=CUTOFF, mix_cutoff=mix_cutoff, window="hard",
        nonlinearity="scalar_gate_double_sigmoid", drop_path=0.0, n_radial_basis=4, radial_hidden_features=(8,),
        radial_basis="gaussian", bias=True, ffn_expansion=2, normalized_mix=True, dtype=dtype,
    ).eval()
    assert block.mix.depthwise and block.mix.normalize
    x = _field(field_type, (plan.n_nodes,), dtype, seed=29)
    expected = block(x, discretization)
    block.mix = _StencilMixer(block.mix, plan)
    _assert_fields_close(block(x, discretization), expected, ATOL[dtype])
