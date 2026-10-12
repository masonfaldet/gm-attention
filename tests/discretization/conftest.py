from __future__ import annotations

import math

import pytest
import torch

from gm_attention.discretization.discretization import SphericalDiscretization
from gm_attention.discretization.frames import build_tangent_frames
from gm_attention.discretization.grid import SphericalGrid
from gm_attention.discretization.grids.icosphere import build_icosphere_grid

# Absolute tolerances used throughout the discretization test suite, keyed
# by dtype -- matches the tolerance convention used inside the library
# itself (gm_attention.discretization.*._validate*).
ATOL = {torch.float64: 1e-10, torch.float32: 1e-5}


@pytest.fixture(params=[torch.float64, torch.float32])
def dtype(request: pytest.FixtureRequest) -> torch.dtype:
    """Both floating-point dtypes the library supports."""
    return request.param


@pytest.fixture
def small_grid() -> SphericalGrid:
    """A small, deterministic icosphere grid (42 nodes) for cheap tests."""
    return build_icosphere_grid(subdivisions=1, dtype=torch.float64)


@pytest.fixture
def small_frames(small_grid: SphericalGrid) -> torch.Tensor:
    return build_tangent_frames(small_grid.nodes)


@pytest.fixture
def small_discretization(
    small_grid: SphericalGrid,
) -> SphericalDiscretization:
    """A discretization over ``small_grid`` with a generous coverage radius."""
    return SphericalDiscretization.from_grid(
        small_grid,
        coverage_radius=0.9 * math.pi,
    )
