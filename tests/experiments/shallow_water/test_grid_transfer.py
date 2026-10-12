"""Tests for ``GridSpec``, the target-grid spec of the zero-shot transfer
evaluation: parsing and building every grid family."""

from __future__ import annotations

import math

import pytest
import torch

from experiments.shallow_water.grid_transfer import GridSpec


def test_grid_specs_parse_and_build_every_family() -> None:
    icosphere = GridSpec.parse("icosphere:2")
    assert icosphere.label == "icosphere:2"
    assert icosphere.build(torch.device("cpu")).n_nodes == 162

    for family in ("equiangular", "legendre-gauss", "lobatto"):
        spec = GridSpec.parse(f"{family}:8x16")
        grid = spec.build(torch.device("cpu"))
        assert spec.label == f"{family}:8x16"
        assert grid.n_nodes == 128
        assert grid.nodes.dtype == torch.float32
        assert grid.quadrature_weights.sum() == pytest.approx(4 * math.pi, rel=1e-5)

    with pytest.raises(ValueError, match="family"):
        GridSpec.parse("healpix:4")
    with pytest.raises(ValueError, match="nlat"):
        GridSpec.parse("equiangular:64")
