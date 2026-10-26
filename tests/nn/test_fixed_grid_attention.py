"""Tests for gm_attention.nn.fixed_grid: the dense stencil evaluation of the attention.

The graph path is the oracle. Every test builds one Gauss-Legendre
discretization in the coordinate gauge, runs ``SO2Attention`` (or
``SO2MHSA``) on it through ``forward`` and through the stencil, and demands
agreement to summation order: values, gradients with respect to the input
and every parameter, both windows, the pointwise self-interaction, field
types with a frequency on one side only or without frequency 0, degenerate
grid shapes, a leading batch axis against the tiled graph, a stencil moved
to float32, a cutoff on a lattice shell (slots with a zero window), and the
heads of an MHSA. The slot matrices are checked against the kernel's own
forward on the basis vectors of its input type. The one property the
stencil path has beyond the graph path, exact equivariance to the grid's own
rotations, is checked as a longitude shift. ``test_fixed_grid_oracle.py``
checks the same function against an oracle written from the definition.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from gm_attention.discretization.stencil import (
    TensorGridStencil,
    build_tensor_grid_discretization,
    build_tensor_grid_stencil,
)
from gm_attention.nn.attention import SO2MHSA, SO2Attention
from gm_attention.nn.fixed_grid import (
    attention_fixed_grid,
    mhsa_fixed_grid,
    steerable_kernel_matrices,
)
from gm_attention.nn.kernels import SO2SteerableKernel
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
#: (field_type_in, field_type_out) pairs for the value kernel alone.
KERNEL_PAIRS = {
    "all-frequencies": (TYPE_IN, TYPE_OUT),
    "no-frequency-0-out": (SO2FieldType({0: 2, 1: 1}), SO2FieldType({1: 2, 2: 1})),
    "no-frequency-0-in": (SO2FieldType({1: 1, 2: 1}), SO2FieldType({0: 1, 1: 1})),
    "scalar-out": (SO2FieldType({0: 2, 1: 1}), SO2FieldType({0: 3})),
    "skipped-frequency": (SO2FieldType({0: 1, 2: 1}), SO2FieldType({1: 1, 2: 1})),  # no frequency 1 on the input
}
ATOL = {torch.float64: 1e-10, torch.float32: 2e-4}


def _geometry(dtype: torch.dtype, shape: tuple[int, int] = SHAPE, cutoff: float = CUTOFF):
    discretization = build_tensor_grid_discretization("legendre-gauss", *shape, coverage_radius=cutoff, dtype=dtype)
    stencil = build_tensor_grid_stencil(discretization, cutoff, n_latitude=shape[0], n_longitude=shape[1])
    return discretization, stencil


def _field(field_type: SO2FieldType, batch_shape: tuple[int, ...], dtype: torch.dtype, seed: int) -> SO2FeatureField:
    generator = torch.Generator().manual_seed(seed)
    flat = torch.randn(*batch_shape, field_type.dimension, dtype=dtype, generator=generator)
    return SO2FeatureField.from_flat(field_type, flat)


def _attention(dtype: torch.dtype, seed: int, triple: str = "all-frequencies", cutoff: float = CUTOFF, **kwargs) -> SO2Attention:
    torch.manual_seed(seed)
    type_in, type_attn, type_out = TRIPLES[triple]
    return SO2Attention(type_in, type_attn, type_out, cutoff=cutoff, dtype=dtype, **kwargs)


def _assert_fields_close(a: SO2FeatureField, b: SO2FeatureField, atol: float) -> None:
    assert a.field_type == b.field_type
    assert a.batch_shape == b.batch_shape
    for frequency in a.field_type.frequencies:
        assert torch.allclose(a[frequency], b[frequency], atol=atol, rtol=0.0), (
            f"frequency {frequency}: max |diff| {float((a[frequency] - b[frequency]).abs().max()):.3e}"
        )


def _matrices_from_basis_vectors(kernel: SO2SteerableKernel, stencil: TensorGridStencil) -> torch.Tensor:
    """The slot matrices the long way: the kernel's forward on the ``d_in``
    basis vectors at every slot (the kernel is linear in its input)."""
    d_in = kernel.field_type_in.dimension
    reference = kernel.radial_basis.centers
    basis = torch.eye(d_in, dtype=reference.dtype, device=reference.device).repeat(stencil.n_slots, 1)
    messages = kernel(SO2FeatureField.from_flat(kernel.field_type_in, basis), stencil.repeated(d_in)).to_flat()
    return messages.reshape(stencil.n_slots, d_in, -1).transpose(1, 2)  # (n_slots, d_out, d_in)


@pytest.fixture(params=[torch.float64, torch.float32])
def dtype(request: pytest.FixtureRequest) -> torch.dtype:
    return request.param


# --------------------------------------------------------------------------
# The slot matrices


@pytest.mark.parametrize("pair", list(KERNEL_PAIRS))
def test_slot_matrices_match_the_kernel_on_basis_vectors(dtype: torch.dtype, pair: str) -> None:
    """The block assembly against the kernel's forward on every basis vector."""
    type_in, type_out = KERNEL_PAIRS[pair]
    _, stencil = _geometry(dtype)
    torch.manual_seed(1)
    kernel = SO2SteerableKernel(type_in, type_out, cutoff=CUTOFF, dtype=dtype)
    matrices = steerable_kernel_matrices(kernel, stencil)
    assert matrices.shape == (stencil.n_slots, type_out.dimension, type_in.dimension)
    expected = _matrices_from_basis_vectors(kernel, stencil)
    assert torch.allclose(matrices, expected, atol=ATOL[dtype], rtol=0.0), (
        f"max |diff| {float((matrices - expected).abs().max()):.3e}"
    )


def test_slot_matrices_reproduce_the_kernel_on_random_sources(dtype: torch.dtype) -> None:
    """``matrices[s] @ x`` is the kernel's message on an edge of slot ``s``."""
    _, stencil = _geometry(dtype)
    torch.manual_seed(1)
    kernel = SO2SteerableKernel(TYPE_IN, TYPE_OUT, cutoff=CUTOFF, dtype=dtype)
    matrices = steerable_kernel_matrices(kernel, stencil)
    x = _field(TYPE_IN, (stencil.n_slots,), dtype, seed=2)
    expected = kernel(x, stencil.geometry).to_flat()  # (n_slots, d_out)
    actual = torch.einsum("sod,sd->so", matrices, x.to_flat())
    assert torch.allclose(actual, expected, atol=ATOL[dtype], rtol=0.0)


def test_slot_matrices_differentiate_through_the_kernel_parameters() -> None:
    _, stencil = _geometry(torch.float64)
    torch.manual_seed(3)
    kernel = SO2SteerableKernel(TYPE_IN, TYPE_OUT, cutoff=CUTOFF, dtype=torch.float64)
    probe = torch.randn(stencil.n_slots, TYPE_OUT.dimension, TYPE_IN.dimension, dtype=torch.float64,
                        generator=torch.Generator().manual_seed(4))
    parameters = list(kernel.parameters())
    expected = torch.autograd.grad((_matrices_from_basis_vectors(kernel, stencil) * probe).sum(), parameters)
    actual = torch.autograd.grad((steerable_kernel_matrices(kernel, stencil) * probe).sum(), parameters)
    for name, a, b in zip([n for n, _ in kernel.named_parameters()], actual, expected, strict=True):
        assert torch.allclose(a, b, atol=1e-9, rtol=0.0), name


def test_slot_matrices_evaluate_the_radial_mlp_once_per_slot() -> None:
    _, stencil = _geometry(torch.float64)
    kernel = SO2SteerableKernel(TYPE_IN, TYPE_OUT, cutoff=CUTOFF, dtype=torch.float64)
    rows: list[int] = []
    handle = kernel.radial_basis.register_forward_hook(lambda module, inputs, output: rows.append(int(inputs[0].numel())))
    try:
        steerable_kernel_matrices(kernel, stencil)
    finally:
        handle.remove()
    assert rows == [stencil.n_slots]


# --------------------------------------------------------------------------
# Attention against the graph path


@pytest.mark.parametrize("triple", list(TRIPLES))
@pytest.mark.parametrize("window", ["hard", "quartic"])
@pytest.mark.parametrize("self_interaction", ["none", "linear"])
def test_attention_matches_the_graph_path(dtype: torch.dtype, triple: str, window: str, self_interaction: str) -> None:
    discretization, stencil = _geometry(dtype)
    module = _attention(dtype, seed=4, triple=triple, window=window, self_interaction=self_interaction)
    x = _field(TRIPLES[triple][0], (stencil.n_nodes,), dtype, seed=5)
    expected = module(x, discretization)
    actual = attention_fixed_grid(module, x, stencil)
    _assert_fields_close(actual, expected, ATOL[dtype])


@pytest.mark.parametrize("triple", ["all-frequencies", "one-sided-frequencies"])
def test_attention_gradients_match_the_graph_path(triple: str) -> None:
    discretization, stencil = _geometry(torch.float64)
    module = _attention(torch.float64, seed=6, triple=triple, window="quartic", self_interaction="linear")
    type_in, _, type_out = TRIPLES[triple]
    flat = _field(type_in, (stencil.n_nodes,), torch.float64, seed=7).to_flat()
    probe = torch.randn(stencil.n_nodes, type_out.dimension, dtype=torch.float64, generator=torch.Generator().manual_seed(8))
    parameters = list(module.parameters())

    def run(path) -> list[torch.Tensor]:
        leaf = flat.clone().requires_grad_(True)
        x = SO2FeatureField.from_flat(type_in, leaf)
        loss = (path(x).to_flat() * probe).sum()
        return list(torch.autograd.grad(loss, [leaf, *parameters]))

    expected = run(lambda x: module(x, discretization))
    actual = run(lambda x: attention_fixed_grid(module, x, stencil))
    for name, a, b in zip(["input", *(n for n, _ in module.named_parameters())], actual, expected, strict=True):
        assert torch.allclose(a, b, atol=1e-9, rtol=0.0), f"{name}: max |diff| {float((a - b).abs().max()):.3e}"


@pytest.mark.parametrize("shape", [(7, 13), (1, 8), (4, 2), (2, 1)])
def test_attention_matches_the_graph_path_on_odd_and_degenerate_shapes(shape: tuple[int, int]) -> None:
    """Odd longitudes, a single row, two longitudes, a single longitude."""
    discretization, stencil = _geometry(torch.float64, shape=shape, cutoff=1.2)
    module = _attention(torch.float64, seed=20, cutoff=1.2, window="quartic", self_interaction="linear")
    x = _field(TYPE_IN, (stencil.n_nodes,), torch.float64, seed=21)
    _assert_fields_close(attention_fixed_grid(module, x, stencil), module(x, discretization), 1e-10)


def test_attention_with_a_leading_batch_axis_matches_the_tiled_graph() -> None:
    """Independent samples along two leading axes against the graph path on
    the disjoint union of as many copies of the grid, self-interaction on."""
    batch = (2, 3)
    count = 6
    discretization, stencil = _geometry(torch.float64)
    module = _attention(torch.float64, seed=9, self_interaction="linear")
    x = _field(TYPE_IN, (*batch, stencil.n_nodes), torch.float64, seed=10)
    tiled = SO2FeatureField(TYPE_IN, {f: t.reshape(count * stencil.n_nodes, *t.shape[3:]) for f, t in x.coefficients.items()})
    expected = module(tiled, discretization.tile(count))
    actual = attention_fixed_grid(module, x, stencil)
    expected_batched = SO2FeatureField(
        TYPE_OUT, {f: t.reshape(*batch, stencil.n_nodes, *t.shape[1:]) for f, t in expected.coefficients.items()}
    )
    _assert_fields_close(actual, expected_batched, 1e-10)


def test_attention_on_a_stencil_moved_to_float32_matches_the_float32_graph_path() -> None:
    nlat, nlon = SHAPE
    discretization64, stencil64 = _geometry(torch.float64)
    stencil = stencil64.to(dtype=torch.float32)
    discretization = discretization64.to(dtype=torch.float32)
    module = _attention(torch.float32, seed=22, window="quartic")
    x = _field(TYPE_IN, (nlat * nlon,), torch.float32, seed=23)
    _assert_fields_close(attention_fixed_grid(module, x, stencil), module(x, discretization), ATOL[torch.float32])


def test_attention_with_the_cutoff_on_a_lattice_shell_matches_the_graph_path() -> None:
    """The quartic window vanishes on the shell exactly at the cutoff, which
    the whole-shell rule keeps in the edge set: the stencil path masks those
    slots as the graph path does."""
    nlat, nlon = SHAPE
    grid = build_tensor_grid_discretization("legendre-gauss", nlat, nlon, coverage_radius=0.3).grid
    row = nlat // 2
    nodes = grid.exact_nodes[[row * nlon, row * nlon + 1]]  # two longitude neighbours on one row
    cutoff = float(torch.arccos((nodes[0] * nodes[1]).sum().clamp(-1.0, 1.0)))
    discretization, stencil = _geometry(torch.float64, cutoff=cutoff)
    module = _attention(torch.float64, seed=24, cutoff=cutoff, window="quartic")
    window = (1.0 - (stencil.geometry.distances / cutoff) ** 2) ** 2
    assert float(window.min()) < 1e-20  # a slot on the shell
    x = _field(TYPE_IN, (nlat * nlon,), torch.float64, seed=25)
    _assert_fields_close(attention_fixed_grid(module, x, stencil), module(x, discretization), 1e-10)


def test_attention_is_equivariant_to_longitude_shifts() -> None:
    """The one symmetry the grid has: rotating the input by k longitudes
    rotates the output by k longitudes, exactly, since every target row sees
    the same slots and the coordinate gauge rotates with the grid."""
    nlat, nlon = SHAPE
    _, stencil = _geometry(torch.float64)
    module = _attention(torch.float64, seed=11, self_interaction="linear")
    x = _field(TYPE_IN, (stencil.n_nodes,), torch.float64, seed=12)
    shift = 7

    def roll(field: SO2FeatureField) -> SO2FeatureField:
        return SO2FeatureField(
            field.field_type,
            {
                f: t.reshape(nlat, nlon, *t.shape[1:]).roll(shift, dims=1).reshape(t.shape)
                for f, t in field.coefficients.items()
            },
        )

    _assert_fields_close(attention_fixed_grid(module, roll(x), stencil), roll(attention_fixed_grid(module, x, stencil)), 1e-11)


def test_attention_is_invariant_to_a_constant_rescaling_of_the_tangent_weights() -> None:
    """The softmax normalizes per target, so scaling every slot's tangent
    quadrature weight by one constant changes nothing: the stencil path's
    weights are a partition of unity like the graph path's."""
    _, stencil = _geometry(torch.float64)
    module = _attention(torch.float64, seed=13)
    x = _field(TYPE_IN, (stencil.n_nodes,), torch.float64, seed=14)
    reference = attention_fixed_grid(module, x, stencil)
    rescaled = stencil.to()  # a fresh stencil object
    rescaled.geometry = dataclasses.replace(
        rescaled.geometry, tangent_quadrature_weights=rescaled.geometry.tangent_quadrature_weights * 3.0
    )
    _assert_fields_close(attention_fixed_grid(module, x, rescaled), reference, 1e-11)


def test_attention_rejects_mismatches() -> None:
    _, stencil = _geometry(torch.float64)
    module = _attention(torch.float64, seed=15)
    with pytest.raises(ValueError, match="type"):
        attention_fixed_grid(module, _field(TYPE_OUT, (stencil.n_nodes,), torch.float64, seed=0), stencil)
    with pytest.raises(ValueError, match="n_nodes"):
        attention_fixed_grid(module, _field(TYPE_IN, (stencil.n_nodes - 1,), torch.float64, seed=0), stencil)
    with pytest.raises(ValueError, match="dtype|float"):
        attention_fixed_grid(module, _field(TYPE_IN, (stencil.n_nodes,), torch.float64, seed=0), stencil.to(dtype=torch.float32))
    other = SO2Attention(TYPE_IN, TYPE_ATTN, TYPE_OUT, cutoff=0.5, dtype=torch.float64)
    with pytest.raises(ValueError, match="cutoff"):
        attention_fixed_grid(other, _field(TYPE_IN, (stencil.n_nodes,), torch.float64, seed=0), stencil)


# --------------------------------------------------------------------------
# Multi-head attention


@pytest.mark.parametrize("batched", [False, True])
def test_mhsa_matches_the_graph_path(dtype: torch.dtype, batched: bool) -> None:
    field_type = SO2FieldType({0: 4, 1: 2, 2: 6})  # heterogeneous multiplicities
    discretization, stencil = _geometry(dtype)
    torch.manual_seed(16)
    module = SO2MHSA(field_type, num_heads=2, cutoff=CUTOFF, batched=batched, dtype=dtype)
    x = _field(field_type, (stencil.n_nodes,), dtype, seed=17)
    expected = module(x, discretization)
    actual = mhsa_fixed_grid(module, x, stencil)
    _assert_fields_close(actual, expected, ATOL[dtype])


def test_mhsa_gradients_match_the_graph_path() -> None:
    field_type = SO2FieldType({0: 4, 1: 2, 2: 2})
    discretization, stencil = _geometry(torch.float64)
    torch.manual_seed(18)
    module = SO2MHSA(field_type, num_heads=2, cutoff=CUTOFF, dtype=torch.float64)
    flat = _field(field_type, (stencil.n_nodes,), torch.float64, seed=19).to_flat()
    probe = torch.randn_like(flat)
    parameters = list(module.parameters())

    def run(path) -> list[torch.Tensor]:
        leaf = flat.clone().requires_grad_(True)
        loss = (path(SO2FeatureField.from_flat(field_type, leaf)).to_flat() * probe).sum()
        return list(torch.autograd.grad(loss, [leaf, *parameters]))

    for a, b in zip(run(lambda x: mhsa_fixed_grid(module, x, stencil)), run(lambda x: module(x, discretization)), strict=True):
        assert torch.allclose(a, b, atol=1e-9, rtol=0.0)


def test_mhsa_rejects_a_mismatched_field_type() -> None:
    _, stencil = _geometry(torch.float64)
    module = SO2MHSA(SO2FieldType({0: 4, 1: 2}), num_heads=2, cutoff=CUTOFF, dtype=torch.float64)
    with pytest.raises(ValueError, match="type"):
        mhsa_fixed_grid(module, _field(TYPE_IN, (stencil.n_nodes,), torch.float64, seed=0), stencil)
