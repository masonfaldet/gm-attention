"""Chunked evaluation must compute what unchunked evaluation computes.

The chunked path exists so a fitted model can be run on a finer mesh than
it trained on (experiments/shallow_water_paper/transfer.py, behind
SWE_256x512_ZeroShot/evaluate.py). Every number that experiment produces
is only worth as much as this equivalence, so it is checked on a real
fitted chain -- a depth-1 arm, its own injected encode radius, the L5/L4
geometry -- rather than on a toy graph. Small-graph coverage lives in
tests/nn/test_convolution_chunked.py and test_attention_chunked.py; what
those cannot show is that the wiring holds together at the size that
matters.

The arm is the shipped depth-1 ablation arm, d1_attention_full (the
SWE_256x512_Ablation primary gm_attention_full), standing in for the
development repository's gm_hard_ratio43, which this release does not
ship. The equivalence does not depend on the hidden spectrum, and both
arms share the deep and decode-refine radii, so the geometry below is the
same one.

Marked slow: six full L5 forward passes in float64 is minutes on CPU.
"""

from __future__ import annotations

import hydra
import pytest
import torch
from hydra import compose, initialize

from gm_attention.discretization.grids.icosphere import (
    build_icosphere_multiresolution,
    build_icosphere_refinement,
)
from gm_attention.representations.field_type import SO2FeatureField

#: Worst relative disagreement measured at L5 in float64 was 7.06e-16, about
#: three epsilons, from segment_reduce reordering its accumulation with input
#: size. This bound sits two orders above that -- loose enough not to be
#: flaky, and still far tighter than any wrong term could hide beneath.
TOLERANCE = 1e-13


def _fitted_model_and_geometry(dtype=torch.float64):
    """The shipped depth-1 arm, at the mesh and radii it is fitted for.

    The bundle primary already selects the d1_attention_full model group,
    so no model override is needed.
    """
    with initialize(
        version_base=None, config_path="../../../SWE_256x512_Ablation/configs"
    ):
        architecture = compose(config_name="gm_attention_full").model.architecture

    _, _, fitted = build_icosphere_refinement(4, dtype=dtype)
    torch.manual_seed(0)
    model = hydra.utils.instantiate(
        architecture, encode_cutoff=fitted.coarse_edge_length, dtype=dtype
    ).eval()
    geometry = build_icosphere_multiresolution(
        5, depth=1,
        coarsest_cutoff=float(architecture.deep_cutoff),
        local_cutoffs={0: float(architecture.decode_refine_cutoff)},
        down_cutoffs={0: float(fitted.coarse_edge_length)},
        dtype=dtype,
    )
    return model, geometry


def _input(model, geometry, dtype=torch.float64) -> SO2FeatureField:
    n = geometry.finest.n_nodes
    generator = torch.Generator().manual_seed(3)
    return SO2FeatureField(model.input_field_type, {
        0: torch.randn(n, 2, dtype=dtype, generator=generator),
        1: torch.randn(n, 1, 2, dtype=dtype, generator=generator),
    })


@pytest.mark.slow
@pytest.mark.parametrize("chunk_size", [97, 640, 1280])
def test_chunked_matches_unchunked_on_the_fitted_l5_chain(
    chunk_size: int,
) -> None:
    """One model instance, the flag toggled between passes.

    Not two constructions with matching seeds: the same object, so identical
    weights are a fact rather than something this test has to establish, and
    a seeding difference cannot masquerade as agreement.

    97 is prime and divides neither 10,242 nor 2,562, so every chunk
    boundary falls mid-structure -- the case most likely to expose an
    off-by-one in the row_ptr slicing. 640 and 1280 are the sizes actually
    used on the cluster.
    """
    model, geometry = _fitted_model_and_geometry()
    x = _input(model, geometry)

    model.evaluation_chunk_size = None
    with torch.no_grad():
        reference = model(x, geometry)

    model.evaluation_chunk_size = chunk_size
    with torch.no_grad():
        chunked = model(x, geometry)

    for frequency in reference.field_type.frequencies:
        difference = (reference[frequency] - chunked[frequency]).abs().max()
        scale = reference[frequency].abs().max().clamp_min(1e-30)
        assert difference / scale < TOLERANCE, (
            f"frequency {frequency}: relative difference "
            f"{float(difference / scale):.3e} at chunk_size={chunk_size}"
        )


@pytest.mark.slow
def test_a_single_chunk_is_exactly_the_unchunked_result() -> None:
    """A chunk size past the node count means one pass over the whole edge
    set, which must reproduce forward bit for bit -- there is no reordering
    to excuse a difference, so this is the one case tolerance would hide."""
    model, geometry = _fitted_model_and_geometry()
    x = _input(model, geometry)

    model.evaluation_chunk_size = None
    with torch.no_grad():
        reference = model(x, geometry)

    model.evaluation_chunk_size = geometry.finest.n_nodes * 2
    with torch.no_grad():
        single = model(x, geometry)

    for frequency in reference.field_type.frequencies:
        assert torch.equal(reference[frequency], single[frequency]), frequency


def test_the_chosen_chunk_sizes_actually_split_both_stages() -> None:
    """Guards against the equivalence passing vacuously.

    The flag reaches two places -- the transformer blocks on the coarse
    level and the decoder's convolution on the fine one. A chunk size larger
    than either node count would leave that stage whole, and the test above
    would then be comparing forward against itself.
    """
    _, geometry = _fitted_model_and_geometry(dtype=torch.float32)

    assert geometry.coarsest.n_nodes == 2_562
    assert geometry.finest.n_nodes == 10_242

    for chunk_size in (97, 640, 1280):
        assert chunk_size < geometry.coarsest.n_nodes, "blocks not split"
        assert chunk_size < geometry.finest.n_nodes, "decoder not split"


def test_the_geometry_is_the_fitted_one() -> None:
    """The equivalence is only interesting on the real operator. If the
    radii drifted, this would be testing a different model than the one the
    zero-shot transfer rebuilds from a run's own cutoffs."""
    _, geometry = _fitted_model_and_geometry(dtype=torch.float32)
    _, _, fitted = build_icosphere_refinement(4, dtype=torch.float32)

    realized = float(geometry.hops[0].down_edges.distances.max())

    assert realized == pytest.approx(float(fitted.coarse_edge_length), rel=1e-5)
    assert geometry.coarsest.n_edges == 124_962
    assert geometry.finest.n_edges == 510_102
