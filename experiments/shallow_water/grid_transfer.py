"""``GridSpec``: a target grid for zero-shot evaluation, parsed from
``icosphere:<level>`` or ``<family>:<nlat>x<nlon>`` and built as a
``SphericalGrid`` on the icosphere or one of torch-harmonics' tensor-product
families."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from gm_attention.discretization.grid import SphericalGrid
from gm_attention.discretization.grids.icosphere import build_icosphere_grid
from gm_attention.discretization.grids.torch_harmonics import build_torch_harmonics_grid

TENSOR_FAMILIES = ("equiangular", "legendre-gauss", "lobatto")


@dataclass(frozen=True)
class GridSpec:
    """One target grid: ``icosphere:<level>`` or ``<family>:<nlat>x<nlon>``."""

    family: str
    level: int | None = None
    nlat: int | None = None
    nlon: int | None = None

    @classmethod
    def parse(cls, text: str) -> GridSpec:
        family, separator, parameters = text.partition(":")
        if not separator:
            raise ValueError(f"grid spec {text!r} needs a ':' -- icosphere:5 or equiangular:64x128.")
        if family == "icosphere":
            return cls(family="icosphere", level=int(parameters))
        if family in TENSOR_FAMILIES:
            nlat, x, nlon = parameters.partition("x")
            if not x:
                raise ValueError(f"{family} needs <nlat>x<nlon>, got {parameters!r}.")
            return cls(family=family, nlat=int(nlat), nlon=int(nlon))
        raise ValueError(
            f"unknown grid family {family!r}; expected icosphere or one of {TENSOR_FAMILIES}."
        )

    @property
    def label(self) -> str:
        if self.family == "icosphere":
            return f"icosphere:{self.level}"
        return f"{self.family}:{self.nlat}x{self.nlon}"

    def build(self, device: torch.device) -> SphericalGrid:
        if self.family == "icosphere":
            assert self.level is not None
            return build_icosphere_grid(self.level, dtype=torch.float32, device=device)
        assert self.nlat is not None and self.nlon is not None
        return build_torch_harmonics_grid(
            self.family, self.nlat, self.nlon, dtype=torch.float32, device=device
        )
