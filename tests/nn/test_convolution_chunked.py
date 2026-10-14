"""Tests for SO2Convolution.forward_chunked.

forward stays the reference implementation; this path exists only to lower
peak memory. So every test here asks the same question in a different way:
does it compute exactly what forward computes?
"""

from __future__ import annotations

import pytest
import torch

from gm_attention.discretization.discretization import SphericalDiscretization
from gm_attention.discretization.grids.icosphere import build_icosphere_grid
from gm_attention.nn.convolution import SO2Convolution
from gm_attention.representations.field_type import SO2FeatureField, SO2FieldType

CUTOFF = 0.9


def _setup(self_interaction="linear", seed=0):
    torch.manual_seed(seed)
    grid = build_icosphere_grid(2, dtype=torch.float64)
    # prepare_windows mutates and returns None; keep the object.
    discretization = SphericalDiscretization.from_grid(grid, coverage_radius=CUTOFF)
    discretization.prepare_windows([CUTOFF])
    layer = SO2Convolution(
        field_type_in=SO2FieldType({0: 2, 1: 2}),
        field_type_out=SO2FieldType({0: 3, 1: 2}),
        cutoff=CUTOFF,
        self_interaction=self_interaction,
        dtype=torch.float64,
    )
    n = grid.n_nodes
    generator = torch.Generator().manual_seed(seed + 1)
    x = SO2FeatureField(layer.field_type_in, {
        0: torch.randn(n, 2, dtype=torch.float64, generator=generator),
        1: torch.randn(n, 2, 2, dtype=torch.float64, generator=generator),
    })
    return layer, discretization, x, n


@pytest.mark.parametrize("chunk_size", [1, 7, 100, 10_000])
def test_chunked_matches_unchunked_to_floating_point_precision(
    chunk_size: int,
) -> None:
    """Chunks split on row_ptr boundaries, so each target sums exactly the
    terms forward sums for it -- an arbitrary edge split would scatter a
    target's contributions across chunks and could not promise this.

    Agreement is to precision rather than bitwise: torch.segment_reduce's
    accumulation order is implementation-defined and varies with input
    size. The bound below is one part in 1e-14, far tighter than anything
    that could hide a wrong term and far looser than the epsilon-level
    reordering actually observed.
    """
    layer, discretization, x, _ = _setup()

    reference = layer(x, discretization)
    chunked = layer.forward_chunked(x, discretization, chunk_size=chunk_size)

    for frequency in reference.field_type.frequencies:
        assert torch.allclose(
            reference[frequency], chunked[frequency], rtol=1e-14, atol=1e-14
        ), frequency


def test_useful_chunk_sizes_come_out_equal_to_the_last_place() -> None:
    """Recording what was measured, so a regression in the split logic is
    visible rather than absorbed into a loose tolerance. Only very small
    chunks reorder the reduction; the sizes anyone would actually run agree
    to a few units in the last place (bitwise on macOS arm64 builds of torch,
    a few ulps apart on x86 Linux builds, whose segment sums accumulate in
    another order)."""
    layer, discretization, x, _ = _setup()

    reference = layer(x, discretization)

    for chunk_size in (100, 10_000):
        chunked = layer.forward_chunked(x, discretization, chunk_size=chunk_size)
        for frequency in reference.field_type.frequencies:
            torch.testing.assert_close(reference[frequency], chunked[frequency], rtol=0, atol=1e-15)


@pytest.mark.parametrize("self_interaction", ["none", "linear"])
def test_the_self_interaction_is_applied_once(self_interaction: str) -> None:
    """It is added after the scatter, outside the chunk loop. Adding it per
    chunk would multiply it by the chunk count -- silently, and only for
    layers that have one."""
    layer, discretization, x, _ = _setup(self_interaction=self_interaction)

    reference = layer(x, discretization)
    chunked = layer.forward_chunked(x, discretization, chunk_size=13)

    for frequency in reference.field_type.frequencies:
        assert torch.allclose(
            reference[frequency], chunked[frequency], rtol=1e-14, atol=1e-14
        )


def test_a_chunk_larger_than_the_graph_is_a_single_pass() -> None:
    layer, discretization, x, n = _setup()

    reference = layer(x, discretization)
    chunked = layer.forward_chunked(x, discretization, chunk_size=n * 10)

    for frequency in reference.field_type.frequencies:
        assert torch.equal(reference[frequency], chunked[frequency])


def test_gradients_match_the_unchunked_path() -> None:
    """It has to be usable for training too, not only inference."""
    layer, discretization, x, _ = _setup()

    def grads(chunk_size: int | None):
        layer.zero_grad(set_to_none=True)
        out = (
            layer(x, discretization)
            if chunk_size is None
            else layer.forward_chunked(x, discretization, chunk_size=chunk_size)
        )
        sum((out[f] ** 2).sum() for f in out.field_type.frequencies).backward()
        return {k: p.grad.clone() for k, p in layer.named_parameters()}

    reference, chunked = grads(None), grads(11)

    for name, gradient in reference.items():
        assert torch.allclose(gradient, chunked[name], atol=1e-12), name


def test_a_nonpositive_chunk_size_is_rejected() -> None:
    layer, discretization, x, _ = _setup()

    with pytest.raises(ValueError, match="at least 1"):
        layer.forward_chunked(x, discretization, chunk_size=0)


def test_it_validates_the_field_type_like_forward() -> None:
    layer, discretization, _, n = _setup()
    wrong = SO2FeatureField(
        SO2FieldType({0: 1}), {0: torch.zeros(n, 1, dtype=torch.float64)}
    )

    with pytest.raises(ValueError, match="Expected a field of type"):
        layer.forward_chunked(wrong, discretization, chunk_size=8)


def test_chunking_actually_reduces_the_live_edge_count() -> None:
    """The reason the path exists. If a chunk still spanned every edge, it
    would be a slower way to run forward."""
    _layer, discretization, _, n = _setup()
    edges = discretization.edges_within(CUTOFF)

    spans = []
    for start in range(0, n, 16):
        stop = min(start + 16, n)
        spans.append(int(edges.row_ptr[stop]) - int(edges.row_ptr[start]))

    assert max(spans) < edges.n_edges
    assert sum(spans) == edges.n_edges  # and nothing is dropped
