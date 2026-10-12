"""Tests for SO2Attention.forward_chunked.

forward remains the reference; this path exists only to lower peak memory
when evaluating a fitted model on a finer mesh. The softmax makes it more
delicate than the convolution's: a target's neighbors must stay together,
or each fragment is normalized separately and the weights no longer sum to
one over the neighborhood.
"""

from __future__ import annotations

import pytest
import torch

from gm_attention.discretization.discretization import SphericalDiscretization
from gm_attention.discretization.grids.icosphere import build_icosphere_grid
from gm_attention.nn.attention import SO2Attention
from gm_attention.representations.field_type import SO2FeatureField, SO2FieldType

CUTOFF = 0.9


def _setup(window="quartic", self_interaction="linear", seed=0):
    torch.manual_seed(seed)
    grid = build_icosphere_grid(2, dtype=torch.float64)
    discretization = SphericalDiscretization.from_grid(grid, coverage_radius=CUTOFF)
    discretization.prepare_windows([CUTOFF])
    layer = SO2Attention(
        field_type_in=SO2FieldType({0: 2, 1: 2}),
        field_type_attn=SO2FieldType({0: 2, 1: 1}),
        field_type_out=SO2FieldType({0: 3, 1: 2}),
        cutoff=CUTOFF,
        window=window,
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
@pytest.mark.parametrize("window", ["hard", "quartic"])
def test_chunked_matches_unchunked(chunk_size: int, window: str) -> None:
    """Both windows: the log-space stabilization differs between them, and
    the quartic one is where a fragmented softmax would show up worst."""
    layer, discretization, x, _ = _setup(window=window)

    reference = layer(x, discretization)
    chunked = layer.forward_chunked(x, discretization, chunk_size=chunk_size)

    for frequency in reference.field_type.frequencies:
        assert torch.allclose(
            reference[frequency], chunked[frequency], rtol=1e-13, atol=1e-13
        ), frequency


def test_the_softmax_stays_a_partition_of_unity_within_a_chunk() -> None:
    """The property that makes node-range chunking valid at all.

    Splitting a target's neighbors across chunks would normalize each
    fragment separately, so its weights would sum to the number of chunks it
    was divided into rather than to one. A single node whose neighborhood
    straddled a boundary would then be silently overweighted.
    """
    _layer, discretization, _x, n = _setup()
    edges = discretization.edges_within(CUTOFF)

    for chunk_size in (1, 13, 64):
        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            first, last = int(edges.row_ptr[start]), int(edges.row_ptr[stop])
            targets = edges.target_index[first:last]
            # Every edge in the slice targets a node inside the range, and
            # every node in the range has all of its edges here.
            assert int(targets.min()) >= start
            assert int(targets.max()) < stop


@pytest.mark.parametrize("self_interaction", ["none", "linear"])
def test_the_self_interaction_is_applied_once(self_interaction: str) -> None:
    """It sits outside the attention normalization, so it must be added
    after the chunk loop rather than once per chunk."""
    layer, discretization, x, _ = _setup(self_interaction=self_interaction)

    reference = layer(x, discretization)
    chunked = layer.forward_chunked(x, discretization, chunk_size=11)

    for frequency in reference.field_type.frequencies:
        assert torch.allclose(
            reference[frequency], chunked[frequency], rtol=1e-13, atol=1e-13
        )


def test_it_is_gauge_equivariant_like_the_unchunked_path() -> None:
    """Chunking is a memory strategy; it must not touch the symmetry the
    whole architecture rests on."""
    import math

    from gm_attention.discretization.frames import rotate_tangent_frames

    layer, discretization, x, n = _setup()
    grid = discretization.grid

    generator = torch.Generator().manual_seed(5)
    angles = torch.rand(n, dtype=torch.float64, generator=generator) * 2 * math.pi
    regauged = SphericalDiscretization.from_grid(
        grid, coverage_radius=CUTOFF,
        frames=rotate_tangent_frames(discretization.frames, angles),
    )
    regauged.prepare_windows([CUTOFF])

    plain = layer.forward_chunked(x, discretization, chunk_size=9)
    rotated = layer.forward_chunked(
        x.gauge_transform(angles), regauged, chunk_size=9
    )
    expected = plain.gauge_transform(angles)

    for frequency in expected.field_type.frequencies:
        assert torch.allclose(rotated[frequency], expected[frequency], atol=1e-10)


def test_a_nonpositive_chunk_size_is_rejected() -> None:
    layer, discretization, x, _ = _setup()

    with pytest.raises(ValueError, match="at least 1"):
        layer.forward_chunked(x, discretization, chunk_size=0)
