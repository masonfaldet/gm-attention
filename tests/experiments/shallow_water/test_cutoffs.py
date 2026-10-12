"""Tests for experiments.shallow_water.cutoffs.

``cutoffs.py`` transcribes formulas that live inside the pinned
``torch_harmonics`` implementation, and a transcription can drift from its
source. So the tests that matter here do **not** compare the helpers
against hard-coded numbers alone -- that would only prove the helper agrees
with itself. They construct a real ``SphericalTransformer`` and recover the
radius each submodule *actually uses* from the neighborhood it built, then
check the helper reproduces it.

The radius is recoverable because these modules consume ``theta_cutoff``
into a sparsity pattern rather than storing it: the largest geodesic
distance between an output point and any input point it is connected to is
the effective support radius, up to the grid's own discreteness.
"""

from __future__ import annotations

import math

import pytest
import torch

from experiments.shallow_water.cutoffs import s2_attention_cutoff, s2_disco_cutoff
from experiments.shallow_water.models import build_reference_transformer
from gm_attention.discretization.grids.torch_harmonics import (
    build_torch_harmonics_grid,
)

# ==========================================================================
# The specified numeric values (spec 10.6)
# ==========================================================================


@pytest.mark.parametrize(
    ("latent_nlat", "expected"),
    [
        (64, 0.19693931676727955),
        (46, 0.27571504347419135),
        (23, 0.56396258892448240),
    ],
)
def test_attention_cutoff_matches_the_specified_values(
    latent_nlat: int, expected: float
) -> None:
    assert s2_attention_cutoff(latent_nlat) == pytest.approx(expected, abs=1e-15)


@pytest.mark.parametrize(
    ("reference_nlat", "expected"),
    [
        (128, 0.049473900056532176),  # R0 encoder (input rows)
        (92, 0.069045992386588860),  # R1/R2/R3 encoder (input rows)
        (64, 0.099733100113961690),  # R0 decoder (latent rows)
        (46, 0.139626340159546360),  # R2/R3 decoder (latent rows)
        (23, 0.285599332144526650),  # R1 decoder (latent rows)
    ],
)
def test_disco_cutoff_matches_the_specified_values(
    reference_nlat: int, expected: float
) -> None:
    assert s2_disco_cutoff(reference_nlat) == pytest.approx(expected, abs=1e-15)


def test_the_r2_headline_deep_cutoff_is_the_matched_value() -> None:
    """The one number both families must share in R2."""
    assert s2_attention_cutoff(46) == pytest.approx(
        0.27571504347419135, abs=1e-15
    )


def test_the_paper_form_differs_from_the_implementation_form() -> None:
    """Recorded for provenance only. The paper writes the attention radius
    as ``7 sqrt(pi) / H`` where the implementation uses ``H - 1``; the
    helper must follow the implementation. If these ever coincided,
    primary runs would be matching a formula nothing actually executes."""
    paper_form = 7.0 * math.sqrt(math.pi) / 46

    assert s2_attention_cutoff(46) != paper_form
    assert paper_form == pytest.approx(0.26972123818127420, abs=1e-15)


@pytest.mark.parametrize("bad", [1, 0, -4])
def test_cutoffs_reject_degenerate_row_counts(bad: int) -> None:
    with pytest.raises(ValueError):
        s2_attention_cutoff(bad)
    with pytest.raises(ValueError):
        s2_disco_cutoff(bad)


# ==========================================================================
# Against the real constructed modules -- the check that can catch drift
# ==========================================================================


def _grid_points(grid: str, nlat: int, nlon: int) -> torch.Tensor:
    """Unit-sphere points of the model's internal grid, row-major.

    Built from the project's own quadrature-rule builder rather than
    assumed equally spaced: the pinned model's default internal grid is
    ``legendre-gauss``, whose latitudes are Gauss-Legendre nodes and do not
    include the poles.
    """
    return build_torch_harmonics_grid(grid, nlat, nlon, dtype=torch.float64).nodes


def _effective_attention_radius(
    attention: torch.nn.Module, grid: str = "legendre-gauss"
) -> float:
    """Largest geodesic distance any attention edge actually spans."""
    source = _grid_points(grid, attention.nlat_in, attention.nlon_in)
    target = _grid_points(grid, attention.nlat_out, attention.nlon_out)

    # psi_row_idx indexes output *latitudes* (the module exploits
    # longitudinal symmetry), psi_col_idx indexes flattened input points.
    row = attention.psi_row_idx.to(torch.long)
    col = attention.psi_col_idx.to(torch.long)
    target_points = target[row * attention.nlon_out]  # longitude 0 of that row
    source_points = source[col]

    cosine = (target_points * source_points).sum(dim=-1).clamp(-1.0, 1.0)
    return float(torch.arccos(cosine).max())


def test_every_attention_block_uses_the_helpers_radius() -> None:
    """Constructed model, every block -- not one spot-check, since a
    per-layer drop_path schedule already varies between blocks."""
    nlat, nlon, scale = 92, 184, 2
    model = build_reference_transformer(
        nlat=nlat, nlon=nlon, embed_dim=8, num_layers=4, scale_factor=scale
    )
    # The latent row count the model actually downsampled to, 46 here.
    latent_nlat = model.h
    expected = s2_attention_cutoff(latent_nlat)

    assert len(model.blocks) == 4
    for index, block in enumerate(model.blocks):
        radius = _effective_attention_radius(block.self_attn)
        # A 2% band, not an exact bound: the module exploits longitudinal
        # symmetry, so the output point a row's neighborhood is centred on
        # is recovered here rather than read off, and that reconstruction
        # is good to well under a grid cell. The band is still far tighter
        # than any formula-level error would be -- the wrong reference row
        # count gives 0.564 or 0.197 against this arm's 0.276, i.e. 40%+.
        assert radius == pytest.approx(expected, rel=0.02), (
            f"block {index}: realized radius {radius} is not the helper's "
            f"{expected}"
        )


def test_the_attention_radius_tracks_the_latent_not_the_input_rows() -> None:
    """The distinction the helper exists to preserve: two models with the
    same input grid but different scale factors must get different
    attention radii."""
    coarse = build_reference_transformer(
        nlat=92, nlon=184, embed_dim=8, num_layers=1, scale_factor=4
    )
    fine = build_reference_transformer(
        nlat=92, nlon=184, embed_dim=8, num_layers=1, scale_factor=2
    )

    coarse_radius = _effective_attention_radius(coarse.blocks[0].self_attn)
    fine_radius = _effective_attention_radius(fine.blocks[0].self_attn)

    assert coarse_radius > fine_radius
    assert coarse_radius == pytest.approx(s2_attention_cutoff(23), rel=0.02)
    assert fine_radius == pytest.approx(s2_attention_cutoff(46), rel=0.02)


def test_an_explicit_theta_cutoff_overrides_the_derived_one() -> None:
    """R2 pins the deep radius to a shared numeric value, so the override
    path must actually reach the attention blocks."""
    requested = 0.27571504347419135
    model = build_reference_transformer(
        nlat=92, nlon=184, embed_dim=8, num_layers=1, scale_factor=4,
        theta_cutoff=requested,
    )

    radius = _effective_attention_radius(model.blocks[0].self_attn)

    assert radius == pytest.approx(requested, rel=0.02)
    # Without the override this arm would use the much wider 23-row radius.
    assert radius < s2_attention_cutoff(23) - 1e-3
