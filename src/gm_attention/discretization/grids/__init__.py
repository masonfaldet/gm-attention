from __future__ import annotations

from gm_attention.discretization.grids.icosphere import build_icosphere_grid
from gm_attention.discretization.grids.torch_harmonics import (
    build_torch_harmonics_grid,
)

__all__ = [
    "build_icosphere_grid",
    "build_torch_harmonics_grid",
]
