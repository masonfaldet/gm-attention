"""The blocked-backward linear map is the plain one: bitwise in forward, to
summation-order precision in every gradient, on small graphs with the
blocked path forced, and through the kernels that use it."""

from __future__ import annotations

import pytest
import torch

from gm_attention.discretization.discretization import SphericalDiscretization
from gm_attention.discretization.frames import build_tangent_frames
from gm_attention.discretization.grids.icosphere import build_icosphere_grid
from gm_attention.nn import kernels
from gm_attention.nn._scatter import gather_to_edges
from gm_attention.nn.attention import SO2MHSA
from gm_attention.nn.convolution import SO2Convolution
from gm_attention.nn.kernels import SO2SteerableKernel, split_k_linear
from gm_attention.representations.field_type import SO2FeatureField, SO2FieldType


def _plain(x, weight, bias):
    out = torch.matmul(x, weight.transpose(-1, -2))
    return out if bias is None else out + bias.unsqueeze(-2)


@pytest.mark.parametrize("lead", [(), (3,)], ids=["plain-weight", "stacked-weight"])
@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no-bias"])
@pytest.mark.parametrize("n", [5, 16, 37, 100], ids=["under-one-block", "one-block", "blocks-plus-remainder", "many-blocks"])
def test_values_bitwise_and_gradients_to_precision(monkeypatch: pytest.MonkeyPatch, lead, bias, n) -> None:
    monkeypatch.setattr(kernels, "SPLIT_K_ROWS", 8)
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(*lead, n, 6, dtype=torch.float64, generator=generator, requires_grad=True)
    weight = torch.randn(*lead, 4, 6, dtype=torch.float64, generator=generator, requires_grad=True)
    b = torch.randn(*lead, 4, dtype=torch.float64, generator=generator, requires_grad=True) if bias else None
    leaves = [t for t in (x, weight, b) if t is not None]

    reference = _plain(x, weight, b)
    blocked = split_k_linear(x, weight, b)
    if not lead:
        assert torch.equal(blocked, torch.nn.functional.linear(x, weight, b)), "forward is the plain linear map"
    torch.testing.assert_close(blocked, reference, atol=1e-13, rtol=0.0)

    grads_ref = torch.autograd.grad((reference ** 2).sum(), leaves)
    grads_blk = torch.autograd.grad((blocked ** 2).sum(), leaves)
    for g_ref, g_blk in zip(grads_ref, grads_blk, strict=True):
        torch.testing.assert_close(g_blk, g_ref, atol=1e-12, rtol=0.0)


def test_the_switch_falls_back_to_plain_autograd(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kernels, "SPLIT_K_BACKWARD", False)
    x = torch.randn(20, 3, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    out = split_k_linear(x, weight, None)
    assert out.grad_fn is not None and "SplitK" not in type(out.grad_fn).__name__


def _small_graph():
    grid = build_icosphere_grid(subdivisions=2, dtype=torch.float64)
    return SphericalDiscretization.from_grid(grid, coverage_radius=0.6, frames=build_tangent_frames(grid.nodes))


def _grads(module, x, discretization, *, chunked: bool = False):
    module.zero_grad()
    out = module.forward_chunked(x, discretization, chunk_size=11) if chunked else module(x, discretization)
    (out.to_flat() ** 2).sum().backward()
    return out, {name: p.grad.clone() for name, p in module.named_parameters()}


@pytest.mark.parametrize("rows", [37, 1000], ids=["blocked", "single-block"])
def test_kernel_convolution_and_batched_attention_agree_with_plain_autograd(monkeypatch: pytest.MonkeyPatch, rows: int) -> None:
    """Every gradient of the layers that use the map, blocked against plain,
    at a block size that splits these few-thousand-edge graphs."""
    field_type = SO2FieldType({0: 4, 1: 4, 2: 4})
    discretization = _small_graph()
    torch.manual_seed(2)
    x = SO2FeatureField.from_flat(field_type, torch.randn(discretization.n_nodes, field_type.dimension, dtype=torch.float64))
    modules = {
        "convolution": SO2Convolution(field_type, field_type, cutoff=0.6, dtype=torch.float64),
        "depthwise": SO2Convolution(field_type, field_type, cutoff=0.6, depthwise=True, dtype=torch.float64),
        "attention": SO2MHSA(field_type, 2, cutoff=0.6, dtype=torch.float64),
        "attention-batched": SO2MHSA(field_type, 2, cutoff=0.6, batched=True, dtype=torch.float64),
    }
    for name, module in modules.items():
        monkeypatch.setattr(kernels, "SPLIT_K_BACKWARD", False)
        out_plain, grads_plain = _grads(module, x, discretization)
        monkeypatch.setattr(kernels, "SPLIT_K_BACKWARD", True)
        monkeypatch.setattr(kernels, "SPLIT_K_ROWS", rows)
        out_blocked, grads_blocked = _grads(module, x, discretization, chunked=name.startswith("attention"))
        for f in field_type.frequencies:
            torch.testing.assert_close(out_blocked[f], out_plain[f], atol=1e-12, rtol=0.0)
        for key in grads_plain:
            torch.testing.assert_close(grads_blocked[key], grads_plain[key], atol=1e-11, rtol=0.0), (name, key)


def test_the_bare_kernel_matches_itself_across_the_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    field_type = SO2FieldType({0: 3, 1: 2, 2: 2})
    torch.manual_seed(1)
    kernel = SO2SteerableKernel(field_type, field_type, cutoff=0.6, dtype=torch.float64)
    discretization = _small_graph()
    edges = discretization.edges_within(0.6)
    x = SO2FeatureField.from_flat(field_type, torch.randn(discretization.n_nodes, field_type.dimension, dtype=torch.float64))
    source = gather_to_edges(x, edges.source_index)
    monkeypatch.setattr(kernels, "SPLIT_K_ROWS", 29)
    blocked = kernel(source, edges)
    monkeypatch.setattr(kernels, "SPLIT_K_BACKWARD", False)
    plain = kernel(source, edges)
    for f in field_type.frequencies:
        assert torch.equal(blocked[f], plain[f])
