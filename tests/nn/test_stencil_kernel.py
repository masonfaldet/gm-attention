"""Tests for gm_attention.nn.stencil_kernel: the gather assembly of the slot matrices.

``steerable_kernel_matrices`` of the reference module is the oracle. The
assembly is checked against it on every shape of type pair (both frequency
sets, a frequency on one side only, scalars only, the released head type) in
float64 and float32, in both layouts, with the phantom row and the per-slot
factor, and in its gradients with respect to every kernel parameter. The
agreement is to rounding: one product is associated the other way round.
"""

from __future__ import annotations

import pytest
import torch

from gm_attention.discretization.stencil import build_tensor_grid_discretization, build_tensor_grid_stencil
from gm_attention.nn.fixed_grid import steerable_kernel_matrices
from gm_attention.nn.kernels import SO2SteerableKernel
from gm_attention.nn.stencil_kernel import _ASSEMBLIES, _assembly, steerable_slot_matrices
from gm_attention.representations.field_type import SO2FieldType

SHAPE = (10, 20)
CUTOFF = 0.6
PAIRS = {
    "all-frequencies": (SO2FieldType({0: 3, 1: 2, 2: 1}), SO2FieldType({0: 2, 1: 2, 2: 1})),
    "no-frequency-0-out": (SO2FieldType({0: 2, 1: 1}), SO2FieldType({1: 2, 2: 1})),
    "no-frequency-0-in": (SO2FieldType({1: 1, 2: 1}), SO2FieldType({0: 1, 1: 1})),
    "scalar-out": (SO2FieldType({0: 2, 1: 1}), SO2FieldType({0: 3})),
    "scalars-only": (SO2FieldType({0: 2}), SO2FieldType({0: 3})),
    "skipped-frequency": (SO2FieldType({0: 1, 2: 1}), SO2FieldType({1: 1, 2: 1})),
    "released-head": (SO2FieldType({0: 16, 1: 16, 2: 16}), SO2FieldType({0: 8, 1: 8, 2: 8})),
}
ATOL = {torch.float64: 1e-13, torch.float32: 1e-5}


def _stencil(dtype: torch.dtype):
    discretization = build_tensor_grid_discretization("legendre-gauss", *SHAPE, coverage_radius=CUTOFF, dtype=dtype)
    return build_tensor_grid_stencil(discretization, CUTOFF, n_latitude=SHAPE[0], n_longitude=SHAPE[1])


def _kernel(pair: str, dtype: torch.dtype, seed: int = 1) -> SO2SteerableKernel:
    torch.manual_seed(seed)
    type_in, type_out = PAIRS[pair]
    return SO2SteerableKernel(type_in, type_out, cutoff=CUTOFF, dtype=dtype)


@pytest.fixture(params=[torch.float64, torch.float32])
def dtype(request: pytest.FixtureRequest) -> torch.dtype:
    return request.param


@pytest.mark.parametrize("pair", list(PAIRS))
def test_slot_matrices_match_the_reference(dtype: torch.dtype, pair: str) -> None:
    stencil = _stencil(dtype)
    kernel = _kernel(pair, dtype)
    expected = steerable_kernel_matrices(kernel, stencil)
    actual = steerable_slot_matrices(kernel, stencil)
    assert actual.shape == expected.shape == (stencil.n_slots, PAIRS[pair][1].dimension, PAIRS[pair][0].dimension)
    assert torch.allclose(actual, expected, atol=ATOL[dtype], rtol=0.0), f"max |diff| {float((actual - expected).abs().max()):.3e}"


def test_transpose_pad_and_scale_are_the_matrices_rearranged() -> None:
    stencil = _stencil(torch.float64)
    kernel = _kernel("all-frequencies", torch.float64)
    plain = steerable_slot_matrices(kernel, stencil)
    transposed = steerable_slot_matrices(kernel, stencil, transpose=True)
    assert torch.equal(transposed, plain.transpose(-1, -2))
    padded = steerable_slot_matrices(kernel, stencil, transpose=True, pad=True)
    assert padded.shape == (stencil.n_slots + 1, *transposed.shape[1:])
    assert torch.equal(padded[:-1], transposed) and bool((padded[-1] == 0.0).all())
    weight = torch.rand(stencil.n_slots, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    scaled = steerable_slot_matrices(kernel, stencil, pad=True, scale=weight)
    assert torch.allclose(scaled[:-1], plain * weight.reshape(-1, 1, 1), atol=1e-15, rtol=0.0)
    assert bool((scaled[-1] == 0.0).all())
    with pytest.raises(ValueError, match="scale"):
        steerable_slot_matrices(kernel, stencil, scale=weight[:-1])


@pytest.mark.parametrize("pair", ["all-frequencies", "no-frequency-0-in", "scalar-out"])
def test_slot_matrices_differentiate_as_the_reference(pair: str) -> None:
    stencil = _stencil(torch.float64)
    kernel = _kernel(pair, torch.float64, seed=3)
    type_in, type_out = PAIRS[pair]
    probe = torch.randn(stencil.n_slots, type_out.dimension, type_in.dimension, dtype=torch.float64,
                        generator=torch.Generator().manual_seed(4))
    parameters = list(kernel.parameters())
    expected = torch.autograd.grad((steerable_kernel_matrices(kernel, stencil) * probe).sum(), parameters)
    actual = torch.autograd.grad((steerable_slot_matrices(kernel, stencil) * probe).sum(), parameters)
    for name, a, b in zip([n for n, _ in kernel.named_parameters()], actual, expected, strict=True):
        assert torch.allclose(a, b, atol=1e-12, rtol=0.0), f"{name}: max |diff| {float((a - b).abs().max()):.3e}"


def test_slot_matrices_evaluate_the_radial_mlp_once_per_slot() -> None:
    stencil = _stencil(torch.float64)
    kernel = _kernel("all-frequencies", torch.float64)
    rows: list[int] = []
    handle = kernel.radial_basis.register_forward_hook(lambda module, inputs, output: rows.append(int(inputs[0].numel())))
    try:
        steerable_slot_matrices(kernel, stencil)
    finally:
        handle.remove()
    assert rows == [stencil.n_slots]


def test_the_assembly_is_cached_per_type_pair_and_reads_every_entry_once() -> None:
    kernel = _kernel("all-frequencies", torch.float64)
    _ASSEMBLIES.clear()
    assembly = _assembly(kernel, transpose=False)
    assert _assembly(_kernel("all-frequencies", torch.float32), transpose=False) is assembly
    assert len(_ASSEMBLIES) == 1
    type_in, type_out = PAIRS["all-frequencies"]
    assert assembly.first.shape == assembly.second.shape == (type_out.dimension * type_in.dimension,)
    # Every term lies in the pool, the zero column is the last one, and the signs are +-1.
    assert int(assembly.first.max()) < assembly.pool_width and int(assembly.second.max()) < assembly.pool_width
    assert bool(((assembly.signs == 1.0) | (assembly.signs == -1.0)).all())
    assert float(assembly.signs[-1]) == 1.0
    # A two-term block (rho_1 -> rho_1 here) negates Im P and Re Q and nothing else.
    assert int((assembly.signs < 0).sum()) == sum(
        2 * type_in.multiplicity(n) * type_out.multiplicity(m)
        for n in type_in.frequencies for m in type_out.frequencies if n >= 1 and m >= 1
    )
