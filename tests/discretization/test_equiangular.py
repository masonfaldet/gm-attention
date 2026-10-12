"""Tests for a two-level equiangular chain and its cross-grid GM resamplers.

A coarse equiangular grid is not a prefix of the fine one -- the two share
no indexing at all -- so the transfers between them are radius-based
bipartite edge sets, and the resamplers that consume them (``EquiEncode``
downward, the normalized interpolation of ``IcoLowPassRestriction`` followed
by a fine-grid convolution upward) must be genuinely gauge-covariant across
two grids that cannot be regauged together. That is a strictly stronger
claim than the icosphere case, where the coarse level is an exact prefix of
the fine one and the two therefore cannot be regauged independently.

So the equivariance tests here regauge the fine and coarse grids with
*separate* random angles. A test that rotated one shared frame array would
pass even for an implementation that ignored transport entirely.

The chain is built by ``build_grid_hierarchy`` with two equiangular levels,
the same builder the spherical SegFormer's grid chain comes from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
import torch

from gm_attention.discretization.discretization import SphericalDiscretization
from gm_attention.discretization.edge_geometry import (
    EdgeGeometry,
    compute_bipartite_edge_geometry,
)
from gm_attention.discretization.frames import (
    build_tangent_frames,
    rotate_tangent_frames,
)
from gm_attention.discretization.grids.equiangular import build_grid_hierarchy
from gm_attention.discretization.grids.torch_harmonics import (
    build_torch_harmonics_grid,
)
from gm_attention.nn.blocks import EquiEncode, IcoLowPassRestriction
from gm_attention.nn.convolution import SO2Convolution
from gm_attention.representations.field_type import SO2FeatureField, SO2FieldType
from torch import Tensor

FINE = (24, 48)
COARSE = (12, 24)
CUTOFFS = {
    "restrict_cutoff": 0.35,
    "interpolate_cutoff": 0.45,
    "fine_cutoff": 0.3,
    "coarse_cutoff": 0.6,
}


@dataclass(frozen=True, slots=True)
class _TwoLevelGeometry:
    """A fine/coarse pair of equiangular levels with both transfer directions.

    ``down_edges.source_index`` indexes fine nodes and ``target_index``
    coarse ones; ``up_edges`` is the reverse, so their ``row_ptr`` are sized
    ``n_coarse + 1`` and ``n_fine + 1`` respectively.
    """

    fine: SphericalDiscretization
    coarse: SphericalDiscretization
    down_edges: EdgeGeometry
    up_edges: EdgeGeometry


def _build(
    fine_shape: tuple[int, int],
    coarse_shape: tuple[int, int],
    *,
    restrict_cutoff: float,
    interpolate_cutoff: float,
    fine_cutoff: float,
    coarse_cutoff: float,
    fine_frames: Tensor | None = None,
    coarse_frames: Tensor | None = None,
) -> _TwoLevelGeometry:
    """Both grids, their frames and both transfer edge sets, through
    ``build_grid_hierarchy`` with two equiangular levels."""
    fine_grid = build_torch_harmonics_grid("equiangular", *fine_shape, dtype=torch.float64)
    coarse_grid = build_torch_harmonics_grid("equiangular", *coarse_shape, dtype=torch.float64)
    hierarchy = build_grid_hierarchy(
        [fine_grid, coarse_grid],
        ["equiangular", "equiangular"],
        [fine_shape, coarse_shape],
        restrict_cutoffs=[restrict_cutoff],
        interpolate_cutoffs=[interpolate_cutoff],
        local_cutoffs=[[fine_cutoff], [coarse_cutoff]],
        frames=[fine_frames, coarse_frames],
    )
    fine, coarse = hierarchy.levels
    assert fine.discretization is not None and coarse.discretization is not None
    return _TwoLevelGeometry(
        fine=fine.discretization,
        coarse=coarse.discretization,
        down_edges=hierarchy.down_edges[0],
        up_edges=hierarchy.up_edges[0],
    )


def _geometry(**overrides: object) -> _TwoLevelGeometry:
    return _build(FINE, COARSE, **{**CUTOFFS, **overrides})  # type: ignore[arg-type]


def _decoder(
    field_type_in: SO2FieldType,
    field_type_out: SO2FieldType,
    *,
    interpolate_cutoff: float,
    fine_cutoff: float,
) -> tuple[IcoLowPassRestriction, SO2Convolution]:
    """The coarse -> fine refinement as the two shipped operators it is made
    of: normalized transport-aware interpolation onto the fine grid at the
    input type, then a learned fine-grid convolution performing the type
    change."""
    interpolation = IcoLowPassRestriction(
        field_type_in, cutoff=interpolate_cutoff, window="quartic"
    )
    conv = SO2Convolution(
        field_type_in, field_type_out, cutoff=fine_cutoff, window="quartic",
        dtype=torch.float64,
    )
    return interpolation, conv


def _random_field(
    field_type: SO2FieldType, n_nodes: int, seed: int = 0
) -> SO2FeatureField:
    generator = torch.Generator().manual_seed(seed)
    return SO2FeatureField(
        field_type,
        {
            frequency: (
                torch.randn(n_nodes, m, dtype=torch.float64, generator=generator)
                if frequency == 0
                else torch.randn(
                    n_nodes, m, 2, dtype=torch.float64, generator=generator
                )
            )
            for frequency, m in field_type.items()
        },
    )


# ==========================================================================
# Bipartite edge geometry
# ==========================================================================


def test_bipartite_edges_index_each_grid_in_its_own_numbering() -> None:
    """The property that makes these edges usable like any same-grid ones:
    x[edges.source_index] must index the source grid directly."""
    geometry = _geometry()
    n_fine = geometry.fine.grid.n_nodes
    n_coarse = geometry.coarse.grid.n_nodes

    assert geometry.down_edges.row_ptr.shape == (n_coarse + 1,)
    assert int(geometry.down_edges.source_index.max()) < n_fine
    assert int(geometry.down_edges.target_index.max()) < n_coarse

    assert geometry.up_edges.row_ptr.shape == (n_fine + 1,)
    assert int(geometry.up_edges.source_index.max()) < n_coarse
    assert int(geometry.up_edges.target_index.max()) < n_fine


def test_every_target_has_support() -> None:
    """An uncovered target aggregates to zero, which downstream is
    indistinguishable from a genuine zero -- so it is an error, not a
    silently empty row."""
    geometry = _geometry()

    for edges, n_targets in (
        (geometry.down_edges, geometry.coarse.grid.n_nodes),
        (geometry.up_edges, geometry.fine.grid.n_nodes),
    ):
        counts = edges.row_ptr[1:] - edges.row_ptr[:-1]
        assert counts.shape == (n_targets,)
        assert int(counts.min()) > 0


def test_a_cutoff_too_small_to_cover_every_target_is_rejected() -> None:
    fine = build_torch_harmonics_grid("equiangular", *FINE, dtype=torch.float64)
    coarse = build_torch_harmonics_grid("equiangular", *COARSE, dtype=torch.float64)

    with pytest.raises(ValueError, match="no source within"):
        compute_bipartite_edge_geometry(
            fine.nodes,
            build_tangent_frames(fine.nodes),
            fine.quadrature_weights,
            coarse.nodes,
            build_tangent_frames(coarse.nodes),
            cutoff=1e-6,
            max_frequency=4,
        )


def test_quadrature_weights_come_from_the_source_side() -> None:
    """The source grid is the one being integrated over."""
    geometry = _geometry()
    edges = geometry.down_edges

    assert edges.tangent_quadrature_weights.shape == (edges.n_edges,)
    assert bool((edges.tangent_quadrature_weights > 0).all())


def test_the_coarse_grid_must_be_strictly_coarser() -> None:
    """``build_grid_hierarchy`` rejects a level that is not strictly coarser
    than its predecessor."""
    with pytest.raises(ValueError, match="strictly coarser"):
        _build((24, 48), (24, 48), **CUTOFFS)  # type: ignore[arg-type]


# ==========================================================================
# EquiEncode / normalized interpolation + fine-grid convolution
# ==========================================================================


def test_the_chain_maps_fine_to_coarse_and_back() -> None:
    geometry = _geometry()
    field_type_in = SO2FieldType({0: 2, 1: 1})
    hidden = SO2FieldType({0: 4, 1: 4})

    encode = EquiEncode(field_type_in, hidden, cutoff=0.35, dtype=torch.float64)
    interpolation, conv = _decoder(
        hidden, SO2FieldType({0: 1, 1: 1}),
        interpolate_cutoff=0.45, fine_cutoff=0.3,
    )

    x = _random_field(field_type_in, geometry.fine.grid.n_nodes)
    coarse = encode(x, geometry.down_edges)
    fine = conv(interpolation(coarse, geometry.up_edges), geometry.fine)

    assert coarse.batch_shape == (geometry.coarse.grid.n_nodes,)
    assert fine.batch_shape == (geometry.fine.grid.n_nodes,)
    assert fine.field_type == SO2FieldType({0: 1, 1: 1})


def test_the_chain_is_equivariant_under_independent_regauging() -> None:
    """Fine and coarse frames are rotated by *separate* random angles. On an
    icosphere the coarse level is a prefix of the fine one, so this
    situation cannot even arise; here it can, and transport is what makes
    the two sides agree."""
    torch.manual_seed(4)
    field_type_in = SO2FieldType({0: 2, 1: 1})
    hidden = SO2FieldType({0: 4, 1: 4, 2: 4})
    encode = EquiEncode(field_type_in, hidden, cutoff=0.35, dtype=torch.float64)
    interpolation, conv = _decoder(
        hidden, SO2FieldType({0: 1, 1: 1}),
        interpolate_cutoff=0.45, fine_cutoff=0.3,
    )

    def decode(
        coarse: SO2FeatureField, up_edges: EdgeGeometry, fine: SphericalDiscretization
    ) -> SO2FeatureField:
        return conv(interpolation(coarse, up_edges), fine)

    geometry = _geometry()
    n_fine = geometry.fine.grid.n_nodes
    n_coarse = geometry.coarse.grid.n_nodes
    x = _random_field(field_type_in, n_fine, seed=5)

    original = decode(
        encode(x, geometry.down_edges), geometry.up_edges, geometry.fine
    )

    generator = torch.Generator().manual_seed(6)
    fine_angles = torch.rand(n_fine, dtype=torch.float64, generator=generator) * (
        2 * math.pi
    )
    coarse_angles = torch.rand(
        n_coarse, dtype=torch.float64, generator=generator
    ) * (2 * math.pi)

    fine_grid = build_torch_harmonics_grid("equiangular", *FINE, dtype=torch.float64)
    coarse_grid = build_torch_harmonics_grid("equiangular", *COARSE, dtype=torch.float64)
    regauged_geometry = _geometry(
        fine_frames=rotate_tangent_frames(
            build_tangent_frames(fine_grid.nodes), fine_angles
        ),
        coarse_frames=rotate_tangent_frames(
            build_tangent_frames(coarse_grid.nodes), coarse_angles
        ),
    )

    regauged = decode(
        encode(x.gauge_transform(fine_angles), regauged_geometry.down_edges),
        regauged_geometry.up_edges,
        regauged_geometry.fine,
    )
    expected = original.gauge_transform(fine_angles)

    for frequency in original.field_type.frequencies:
        assert torch.allclose(regauged[frequency], expected[frequency], atol=1e-10)


def test_normalized_interpolation_preserves_a_scalar_constant() -> None:
    """The defining property of the normalized path: one invariant scalar
    denominator per target, so a constant survives resampling exactly."""
    geometry = _geometry()
    hidden = SO2FieldType({0: 3, 1: 2})
    interpolation, _ = _decoder(
        hidden, SO2FieldType({0: 1, 1: 1}),
        interpolate_cutoff=0.45, fine_cutoff=0.3,
    )
    n_coarse = geometry.coarse.grid.n_nodes

    constant = SO2FeatureField(
        hidden,
        {
            0: torch.ones(n_coarse, 3, dtype=torch.float64),
            1: torch.zeros(n_coarse, 2, 2, dtype=torch.float64),
        },
    )

    interpolated = interpolation(constant, geometry.up_edges)

    assert torch.allclose(
        interpolated[0], torch.ones_like(interpolated[0]), atol=1e-12
    )


def test_equi_encode_rejects_an_unknown_mode() -> None:
    with pytest.raises(ValueError, match="learned_integral"):
        EquiEncode(
            SO2FieldType({0: 1}), SO2FieldType({0: 2}),
            cutoff=0.35, mode="disco",  # type: ignore[arg-type]
        )
