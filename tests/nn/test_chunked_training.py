"""Chunked training: per-chunk checkpointing must not change the gradient.

The chunked paths are exact partitions of the per-target computation;
checkpointing changes when intermediates exist, never what they are. So
the parameter gradients from a chunked, checkpointed training-mode
forward must equal the unchunked reference's to floating-point precision
-- for both the attention and the convolution.
"""

from __future__ import annotations

import math

import torch

from gm_attention.discretization.discretization import SphericalDiscretization
from gm_attention.discretization.frames import build_tangent_frames
from gm_attention.discretization.grids.icosphere import build_icosphere_grid
from gm_attention.nn.attention import SO2MHSA
from gm_attention.nn.convolution import SO2Convolution
from gm_attention.representations.field_type import SO2FeatureField, SO2FieldType


def _discretization(cutoff: float) -> SphericalDiscretization:
    grid = build_icosphere_grid(subdivisions=2, dtype=torch.float64)
    frames = build_tangent_frames(grid.nodes)
    discretization = SphericalDiscretization.from_grid(
        grid, coverage_radius=cutoff, frames=frames
    )
    discretization.prepare_windows([cutoff])
    return discretization


def _grads(module: torch.nn.Module) -> list[torch.Tensor]:
    return [p.grad.detach().clone() for p in module.parameters() if p.grad is not None]


def _assert_same_grads(module, run_reference, run_chunked) -> None:
    module.train()
    module.zero_grad(set_to_none=True)
    run_reference().to_flat().pow(2).sum().backward()
    reference = _grads(module)

    module.zero_grad(set_to_none=True)
    run_chunked().to_flat().pow(2).sum().backward()
    chunked = _grads(module)

    assert len(reference) == len(chunked) > 0
    for a, b in zip(reference, chunked, strict=True):
        assert torch.allclose(a, b, atol=1e-12, rtol=1e-10)


def test_chunked_training_attention_gradients_match_the_reference() -> None:
    cutoff = 0.6 * math.pi
    field_type = SO2FieldType({0: 2, 1: 2, 2: 2})
    torch.manual_seed(3)
    attention = SO2MHSA(field_type, 2, cutoff=cutoff, dtype=torch.float64)
    discretization = _discretization(cutoff)
    generator = torch.Generator().manual_seed(4)
    x = SO2FeatureField.from_flat(
        field_type,
        torch.randn(discretization.grid.n_nodes, field_type.dimension,
                    dtype=torch.float64, generator=generator),
    )

    _assert_same_grads(
        attention,
        lambda: attention(x, discretization),
        lambda: attention.forward_chunked(x, discretization, chunk_size=37),
    )


def test_chunked_training_convolution_gradients_match_the_reference() -> None:
    cutoff = 0.5 * math.pi
    torch.manual_seed(5)
    conv = SO2Convolution(
        SO2FieldType({0: 2, 1: 1}), SO2FieldType({0: 1, 1: 2, 2: 1}),
        cutoff=cutoff, self_interaction="linear", dtype=torch.float64,
    )
    discretization = _discretization(cutoff)
    generator = torch.Generator().manual_seed(6)
    x = SO2FeatureField.from_flat(
        conv.field_type_in,
        torch.randn(discretization.grid.n_nodes, conv.field_type_in.dimension,
                    dtype=torch.float64, generator=generator),
    )

    _assert_same_grads(
        conv,
        lambda: conv(x, discretization),
        lambda: conv.forward_chunked(x, discretization, chunk_size=29),
    )


def test_chunked_eval_path_is_untouched_by_the_training_hook() -> None:
    """Outside training the closure runs directly -- the inference
    contract (allclose at 1e-13 to forward) holds unchanged."""
    cutoff = 0.6 * math.pi
    field_type = SO2FieldType({0: 2, 1: 2})
    torch.manual_seed(7)
    attention = SO2MHSA(field_type, 2, cutoff=cutoff, dtype=torch.float64).eval()
    discretization = _discretization(cutoff)
    x = SO2FeatureField.from_flat(
        field_type, torch.randn(discretization.grid.n_nodes, field_type.dimension, dtype=torch.float64)
    )
    with torch.no_grad():
        full = attention(x, discretization).to_flat()
        chunked = attention.forward_chunked(x, discretization, chunk_size=23).to_flat()
    assert torch.allclose(full, chunked, atol=1e-13)
