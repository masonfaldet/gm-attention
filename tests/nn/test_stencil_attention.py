"""Tests for gm_attention.nn.stencil_attention: the vectorized dense attention on the fixed grid.

The graph path is the oracle and the per-row reference
``gm_attention.nn.fixed_grid`` the second witness. Every test builds one
Gauss-Legendre discretization in the coordinate gauge, runs ``SO2Attention``
(or ``SO2MHSA``) on it through ``forward`` and through a stencil plan, and
demands agreement to summation order: values in float64 and float32,
gradients with respect to the input and every parameter, both windows, the
pointwise self-interaction, field types with a frequency on one side only or
without frequency 0, degenerate grid shapes, leading batch axes against the
tiled graph, every bucket count from one (plain padding) to one bucket per
degree, a cutoff on a lattice shell, and the heads of an MHSA. The
equivariance to longitude shifts is exact to the bit, and the dense stage
traces under ``torch.compile(fullgraph=True)`` without a graph break.
"""

from __future__ import annotations

import pytest
import torch

from gm_attention.discretization.stencil import (
    build_tensor_grid_discretization,
    build_tensor_grid_stencil,
)
from gm_attention.nn.attention import SO2MHSA, SO2Attention
from gm_attention.nn.fixed_grid import attention_fixed_grid, mhsa_fixed_grid, steerable_kernel_matrices
from gm_attention.nn.stencil_attention import (
    attend,
    attention_tables,
    stencil_attention,
    stencil_mhsa,
)
from gm_attention.nn.stencil_plan import StencilPlan, build_stencil_plan
from gm_attention.representations.field_type import SO2FeatureField, SO2FieldType

SHAPE = (10, 20)
CUTOFF = 0.6
TYPE_IN = SO2FieldType({0: 3, 1: 2, 2: 1})
TYPE_ATTN = SO2FieldType({0: 2, 1: 1, 2: 1})
TYPE_OUT = SO2FieldType({0: 2, 1: 2, 2: 1})
#: (field_type_in, field_type_attn, field_type_out): the frequency sets differ per side.
TRIPLES = {
    "all-frequencies": (TYPE_IN, TYPE_ATTN, TYPE_OUT),
    "no-frequency-0-on-attn-and-out": (TYPE_IN, SO2FieldType({1: 1, 2: 1}), SO2FieldType({1: 1, 2: 1})),
    "one-sided-frequencies": (SO2FieldType({0: 2, 1: 1}), SO2FieldType({0: 1, 2: 1}), SO2FieldType({1: 1, 2: 1})),
    "scalar-attention-and-output": (SO2FieldType({0: 3, 1: 1}), SO2FieldType({0: 2}), SO2FieldType({0: 3})),
}
ATOL = {torch.float64: 1e-10, torch.float32: 2e-4}
MHSA_TYPE = SO2FieldType({0: 4, 1: 2, 2: 6})  # heterogeneous multiplicities


def _geometry(dtype: torch.dtype, shape: tuple[int, int] = SHAPE, cutoff: float = CUTOFF, max_buckets: int = 8):
    discretization = build_tensor_grid_discretization("legendre-gauss", *shape, coverage_radius=cutoff, dtype=dtype)
    stencil = build_tensor_grid_stencil(discretization, cutoff, n_latitude=shape[0], n_longitude=shape[1])
    return discretization, build_stencil_plan(stencil, max_buckets=max_buckets)


def _field(field_type: SO2FieldType, batch_shape: tuple[int, ...], dtype: torch.dtype, seed: int) -> SO2FeatureField:
    generator = torch.Generator().manual_seed(seed)
    flat = torch.randn(*batch_shape, field_type.dimension, dtype=dtype, generator=generator)
    return SO2FeatureField.from_flat(field_type, flat)


def _attention(dtype: torch.dtype, seed: int, triple: str = "all-frequencies", cutoff: float = CUTOFF, **kwargs) -> SO2Attention:
    torch.manual_seed(seed)
    type_in, type_attn, type_out = TRIPLES[triple]
    return SO2Attention(type_in, type_attn, type_out, cutoff=cutoff, dtype=dtype, **kwargs)


def _mhsa(dtype: torch.dtype, seed: int, **kwargs) -> SO2MHSA:
    torch.manual_seed(seed)
    return SO2MHSA(MHSA_TYPE, num_heads=2, cutoff=CUTOFF, dtype=dtype, **kwargs)


def _assert_fields_close(a: SO2FeatureField, b: SO2FeatureField, atol: float) -> None:
    assert a.field_type == b.field_type
    assert a.batch_shape == b.batch_shape
    for frequency in a.field_type.frequencies:
        assert torch.allclose(a[frequency], b[frequency], atol=atol, rtol=0.0), (
            f"frequency {frequency}: max |diff| {float((a[frequency] - b[frequency]).abs().max()):.3e}"
        )


def _assert_fields_equal(a: SO2FeatureField, b: SO2FeatureField) -> None:
    for frequency in a.field_type.frequencies:
        assert torch.equal(a[frequency], b[frequency]), f"frequency {frequency} differs"


def _tiled(x: SO2FeatureField, count: int) -> SO2FeatureField:
    """Leading axes folded into the node axis: the disjoint union the tiled graph expects."""
    return SO2FeatureField(x.field_type, {f: t.reshape(count * x.batch_shape[-1], *t.shape[len(x.batch_shape):]) for f, t in x.coefficients.items()})


def _unfolded(y: SO2FeatureField, batch_shape: tuple[int, ...]) -> SO2FeatureField:
    return SO2FeatureField(y.field_type, {f: t.reshape(*batch_shape, *t.shape[1:]) for f, t in y.coefficients.items()})


def _rolled(field: SO2FeatureField, shift: int, shape: tuple[int, int]) -> SO2FeatureField:
    nlat, nlon = shape
    return SO2FeatureField(
        field.field_type,
        {f: t.reshape(nlat, nlon, *t.shape[1:]).roll(shift, dims=1).reshape(t.shape) for f, t in field.coefficients.items()},
    )


def _gradients(path, flat: torch.Tensor, field_type: SO2FieldType, probe: torch.Tensor, parameters) -> list[torch.Tensor]:
    leaf = flat.clone().requires_grad_(True)
    loss = (path(SO2FeatureField.from_flat(field_type, leaf)).to_flat() * probe).sum()
    return list(torch.autograd.grad(loss, [leaf, *parameters]))


@pytest.fixture(params=[torch.float64, torch.float32])
def dtype(request: pytest.FixtureRequest) -> torch.dtype:
    return request.param


# --------------------------------------------------------------------------
# The tables


def test_tables_hold_the_transposed_slot_matrices_the_window_bias_and_the_transport() -> None:
    _, plan = _geometry(torch.float64)
    module = _attention(torch.float64, seed=1, window="quartic")
    tables = attention_tables([module], plan)
    n_slots = plan.n_slots
    assert tables.matrices.shape == (1, n_slots + 1, TYPE_IN.dimension, TYPE_OUT.dimension)
    expected = steerable_kernel_matrices(module.value, plan.stencil).transpose(-1, -2)
    assert torch.allclose(tables.matrices[0, :-1], expected, atol=1e-13, rtol=0.0)  # the gather assembly, to rounding
    assert bool((tables.matrices[0, -1] == 0.0).all())
    geometry = plan.stencil.geometry
    window = (1.0 - (geometry.distances / CUTOFF) ** 2) ** 2
    inside = window > 0.0
    expected = torch.log(window[inside]) + torch.log(geometry.tangent_quadrature_weights[inside])
    assert torch.allclose(tables.bias[:-1][inside], expected, atol=1e-12, rtol=0.0)
    assert bool(torch.isneginf(tables.bias[:-1][~inside]).all()) and bool(torch.isneginf(tables.bias[-1]))
    assert tables.cosines.shape == (2, n_slots + 1)  # frequencies 1 and 2 of the attention type
    for i, frequency in enumerate((1, 2)):
        phase = geometry.transport_harmonic(frequency)
        assert torch.allclose(tables.cosines[i, :-1], phase.real) and torch.allclose(tables.sines[i, :-1], phase.imag)
    assert float(tables.cosines[0, -1]) == 1.0 and float(tables.sines[0, -1]) == 0.0


def test_tables_reject_heads_that_disagree_or_a_foreign_cutoff() -> None:
    _, plan = _geometry(torch.float64)
    first = _attention(torch.float64, seed=1)
    other = _attention(torch.float64, seed=2, window="quartic")
    with pytest.raises(ValueError, match="share"):
        attention_tables([first, other], plan)
    with pytest.raises(ValueError, match="cutoff"):
        attention_tables([SO2Attention(TYPE_IN, TYPE_ATTN, TYPE_OUT, cutoff=0.5, dtype=torch.float64)], plan)
    with pytest.raises(ValueError, match="at least one"):
        attention_tables([], plan)


# --------------------------------------------------------------------------
# Attention against the graph path and the reference


@pytest.mark.parametrize("triple", list(TRIPLES))
@pytest.mark.parametrize("window", ["hard", "quartic"])
@pytest.mark.parametrize("self_interaction", ["none", "linear"])
def test_attention_matches_the_graph_path(dtype: torch.dtype, triple: str, window: str, self_interaction: str) -> None:
    discretization, plan = _geometry(dtype)
    module = _attention(dtype, seed=4, triple=triple, window=window, self_interaction=self_interaction)
    x = _field(TRIPLES[triple][0], (plan.n_nodes,), dtype, seed=5)
    _assert_fields_close(stencil_attention(module, x, plan), module(x, discretization), ATOL[dtype])


@pytest.mark.parametrize("max_buckets", [1, 3, 8, 100])
def test_attention_matches_the_reference_for_every_bucket_count(dtype: torch.dtype, max_buckets: int) -> None:
    """From plain padding to one bucket per distinct degree: the same values as the per-row reference."""
    _, plan = _geometry(dtype, max_buckets=max_buckets)
    module = _attention(dtype, seed=6, window="quartic", self_interaction="linear")
    x = _field(TYPE_IN, (plan.n_nodes,), dtype, seed=7)
    _assert_fields_close(stencil_attention(module, x, plan), attention_fixed_grid(module, x, plan.stencil), ATOL[dtype])


@pytest.mark.parametrize("triple", ["all-frequencies", "one-sided-frequencies"])
def test_attention_gradients_match_the_graph_path(triple: str) -> None:
    discretization, plan = _geometry(torch.float64)
    module = _attention(torch.float64, seed=8, triple=triple, window="quartic", self_interaction="linear")
    type_in, _, type_out = TRIPLES[triple]
    flat = _field(type_in, (plan.n_nodes,), torch.float64, seed=9).to_flat()
    probe = torch.randn(plan.n_nodes, type_out.dimension, dtype=torch.float64, generator=torch.Generator().manual_seed(10))
    parameters = list(module.parameters())
    expected = _gradients(lambda x: module(x, discretization), flat, type_in, probe, parameters)
    actual = _gradients(lambda x: stencil_attention(module, x, plan), flat, type_in, probe, parameters)
    for name, a, b in zip(["input", *(n for n, _ in module.named_parameters())], actual, expected, strict=True):
        assert torch.allclose(a, b, atol=1e-9, rtol=0.0), f"{name}: max |diff| {float((a - b).abs().max()):.3e}"


@pytest.mark.parametrize("shape", [(7, 13), (1, 8), (4, 2), (2, 1)])
def test_attention_matches_the_graph_path_on_odd_and_degenerate_shapes(shape: tuple[int, int]) -> None:
    """Odd longitudes, a single row, two longitudes, a single longitude."""
    discretization, plan = _geometry(torch.float64, shape=shape, cutoff=1.2)
    module = _attention(torch.float64, seed=11, cutoff=1.2, window="quartic", self_interaction="linear")
    x = _field(TYPE_IN, (plan.n_nodes,), torch.float64, seed=12)
    _assert_fields_close(stencil_attention(module, x, plan), module(x, discretization), 1e-10)


def test_attention_with_leading_batch_axes_matches_the_tiled_graph(dtype: torch.dtype) -> None:
    """Independent samples along two leading axes against the graph path on
    the disjoint union of as many copies of the grid, self-interaction on."""
    batch = (2, 3)
    discretization, plan = _geometry(dtype, max_buckets=3)
    module = _attention(dtype, seed=13, self_interaction="linear")
    x = _field(TYPE_IN, (*batch, plan.n_nodes), dtype, seed=14)
    expected = _unfolded(module(_tiled(x, 6), discretization.tile(6)), (*batch, plan.n_nodes))
    _assert_fields_close(stencil_attention(module, x, plan), expected, ATOL[dtype])
    _assert_fields_close(stencil_attention(module, x, plan), attention_fixed_grid(module, x, plan.stencil), ATOL[dtype])


def test_attention_on_a_plan_moved_to_float32_matches_the_float32_graph_path() -> None:
    discretization64, plan64 = _geometry(torch.float64)
    plan, discretization = plan64.to(dtype=torch.float32), discretization64.to(dtype=torch.float32)
    module = _attention(torch.float32, seed=15, window="quartic")
    x = _field(TYPE_IN, (plan.n_nodes,), torch.float32, seed=16)
    _assert_fields_close(stencil_attention(module, x, plan), module(x, discretization), ATOL[torch.float32])


def test_attention_with_the_cutoff_on_a_lattice_shell_matches_the_graph_path() -> None:
    """The quartic window vanishes on the shell exactly at the cutoff, which
    the whole-shell rule keeps in the edge set: those slots are -inf in the
    bias, as the graph path masks them."""
    nlat, nlon = SHAPE
    grid = build_tensor_grid_discretization("legendre-gauss", nlat, nlon, coverage_radius=0.3).grid
    row = nlat // 2
    nodes = grid.exact_nodes[[row * nlon, row * nlon + 1]]
    cutoff = float(torch.arccos((nodes[0] * nodes[1]).sum().clamp(-1.0, 1.0)))
    discretization, plan = _geometry(torch.float64, cutoff=cutoff)
    module = _attention(torch.float64, seed=17, cutoff=cutoff, window="quartic")
    assert bool(torch.isneginf(attention_tables([module], plan).bias[:-1]).any())  # a slot on the shell
    x = _field(TYPE_IN, (plan.n_nodes,), torch.float64, seed=18)
    _assert_fields_close(stencil_attention(module, x, plan), module(x, discretization), 1e-10)


def test_attention_is_exactly_equivariant_to_longitude_shifts() -> None:
    """Rotating the input by k longitudes rotates the output by k longitudes
    to the bit: a shift permutes the rows of every bucket's operands within
    one batch entry, so each output element sees the same operands in the
    same order."""
    _, plan = _geometry(torch.float64)
    module = _attention(torch.float64, seed=19, self_interaction="linear")
    x = _field(TYPE_IN, (plan.n_nodes,), torch.float64, seed=20)
    _assert_fields_equal(stencil_attention(module, _rolled(x, 7, SHAPE), plan), _rolled(stencil_attention(module, x, plan), 7, SHAPE))


def test_attend_compiles_as_one_graph(dtype: torch.dtype) -> None:
    """The dense stage under ``torch.compile(fullgraph=True)``: no data-dependent
    shape and no graph break, the same values as eager."""
    _, plan = _geometry(dtype)
    module = _attention(dtype, seed=21, window="quartic")
    x = _field(TYPE_IN, (plan.n_nodes,), dtype, seed=22)
    tables = attention_tables([module], plan)
    queries = module.query(x).to_flat().reshape(1, 1, plan.n_nodes, -1)
    keys = module.key(x).to_flat().reshape(1, 1, plan.n_nodes, -1)
    source = x.to_flat().reshape(1, plan.n_nodes, -1)
    arguments = (plan, queries, keys, source, tables, module.field_type_attn, module._score_scale)
    torch._dynamo.reset()
    compiled = torch.compile(attend, backend="aot_eager", fullgraph=True)
    assert torch.allclose(compiled(*arguments), attend(*arguments), atol=ATOL[dtype], rtol=0.0)


def test_attention_rejects_mismatches() -> None:
    _, plan = _geometry(torch.float64)
    module = _attention(torch.float64, seed=23)
    with pytest.raises(ValueError, match="type"):
        stencil_attention(module, _field(TYPE_OUT, (plan.n_nodes,), torch.float64, seed=0), plan)
    with pytest.raises(ValueError, match="n_nodes"):
        stencil_attention(module, _field(TYPE_IN, (plan.n_nodes - 1,), torch.float64, seed=0), plan)
    with pytest.raises(ValueError, match="dtype|float"):
        stencil_attention(module, _field(TYPE_IN, (plan.n_nodes,), torch.float64, seed=0), plan.to(dtype=torch.float32))
    other = SO2Attention(TYPE_IN, TYPE_ATTN, TYPE_OUT, cutoff=0.5, dtype=torch.float64)
    with pytest.raises(ValueError, match="cutoff"):
        stencil_attention(other, _field(TYPE_IN, (plan.n_nodes,), torch.float64, seed=0), plan)


# --------------------------------------------------------------------------
# Multi-head attention


@pytest.mark.parametrize("batched", [False, True])
def test_mhsa_matches_the_graph_path(dtype: torch.dtype, batched: bool) -> None:
    discretization, plan = _geometry(dtype)
    module = _mhsa(dtype, seed=24, batched=batched)
    x = _field(MHSA_TYPE, (plan.n_nodes,), dtype, seed=25)
    _assert_fields_close(stencil_mhsa(module, x, plan), module(x, discretization), ATOL[dtype])


@pytest.mark.parametrize("max_buckets", [1, 4, 100])
def test_mhsa_matches_the_reference(dtype: torch.dtype, max_buckets: int) -> None:
    _, plan = _geometry(dtype, max_buckets=max_buckets)
    module = _mhsa(dtype, seed=26, window="quartic")
    x = _field(MHSA_TYPE, (plan.n_nodes,), dtype, seed=27)
    _assert_fields_close(stencil_mhsa(module, x, plan), mhsa_fixed_grid(module, x, plan.stencil), ATOL[dtype])


def test_mhsa_gradients_match_the_graph_path() -> None:
    discretization, plan = _geometry(torch.float64)
    module = _mhsa(torch.float64, seed=28)
    flat = _field(MHSA_TYPE, (plan.n_nodes,), torch.float64, seed=29).to_flat()
    probe = torch.randn_like(flat)
    parameters = list(module.parameters())
    expected = _gradients(lambda x: module(x, discretization), flat, MHSA_TYPE, probe, parameters)
    actual = _gradients(lambda x: stencil_mhsa(module, x, plan), flat, MHSA_TYPE, probe, parameters)
    for name, a, b in zip(["input", *(n for n, _ in module.named_parameters())], actual, expected, strict=True):
        assert torch.allclose(a, b, atol=1e-9, rtol=0.0), f"{name}: max |diff| {float((a - b).abs().max()):.3e}"


def test_mhsa_with_leading_batch_axes_matches_the_tiled_graph(dtype: torch.dtype) -> None:
    batch = (3, 2)
    discretization, plan = _geometry(dtype, max_buckets=2)
    module = _mhsa(dtype, seed=30)
    x = _field(MHSA_TYPE, (*batch, plan.n_nodes), dtype, seed=31)
    expected = _unfolded(module(_tiled(x, 6), discretization.tile(6)), (*batch, plan.n_nodes))
    _assert_fields_close(stencil_mhsa(module, x, plan), expected, ATOL[dtype])


def test_mhsa_is_exactly_equivariant_to_longitude_shifts() -> None:
    _, plan = _geometry(torch.float64)
    module = _mhsa(torch.float64, seed=32)
    x = _field(MHSA_TYPE, (plan.n_nodes,), torch.float64, seed=33)
    _assert_fields_equal(stencil_mhsa(module, _rolled(x, 3, SHAPE), plan), _rolled(stencil_mhsa(module, x, plan), 3, SHAPE))


def test_mhsa_rejects_mismatches() -> None:
    _, plan = _geometry(torch.float64)
    module = _mhsa(torch.float64, seed=34)
    with pytest.raises(ValueError, match="type"):
        stencil_mhsa(module, _field(TYPE_IN, (plan.n_nodes,), torch.float64, seed=0), plan)
    other = SO2MHSA(MHSA_TYPE, num_heads=2, cutoff=0.5, dtype=torch.float64)
    with pytest.raises(ValueError, match="cutoff"):
        stencil_mhsa(other, _field(MHSA_TYPE, (plan.n_nodes,), torch.float64, seed=0), plan)
    assert isinstance(plan, StencilPlan)
