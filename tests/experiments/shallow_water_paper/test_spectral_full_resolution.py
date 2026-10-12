"""Independent harmonic checks at the benchmark's actual sampling resolution."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.special import sph_harm_y

from experiments.shallow_water.grid_transfer import GridSpec
from experiments.shallow_water_paper.spectral import FOUR_PI, SpectralProjector

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def projector() -> SpectralProjector:
    return SpectralProjector(42, torch.device("cpu"))


@pytest.mark.parametrize("grid_text", ["icosphere:5", "equiangular:128x256"])
def test_independent_harmonics_and_aliasing_at_benchmark_resolution(projector, grid_text: str) -> None:
    grid = GridSpec.parse(grid_text).build(torch.device("cpu"))
    fit = projector.fit(grid)
    xyz = grid.nodes.double().numpy()
    theta = np.arccos(np.clip(xyz[:, 2] / np.linalg.norm(xyz, axis=1), -1, 1))
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    assert fit.condition < 2.1
    for degree, order in ((0, 0), (17, 5), (42, 19), (43, 0), (60, 13)):
        samples = torch.from_numpy(np.asarray(sph_harm_y(degree, order, theta, phi).real.copy())).reshape(-1, 1)
        coefficients = fit.project(samples)
        energy = fit.energy_by_degree(coefficients).squeeze(-1)
        exact = (1.0 if order == 0 else 0.5) / FOUR_PI
        if degree <= 42:
            assert float(energy[degree]) == pytest.approx(exact, rel=2e-6)
            assert float(energy.sum() - energy[degree]) < 1e-10 * exact
            # A negative native-continuum gap does NOT mean negative
            # high-frequency energy: this is a pure in-band harmonic.
            if grid_text == "icosphere:5" and degree == 42:
                assert float(fit.grid_mean_square(samples)) < float(energy.sum())
        elif grid_text == "equiangular:128x256":
            assert float(energy.sum()) < 1e-9 * exact
        else:
            assert 1e-6 < float(energy.sum()) / exact < 1e-3
        residual = samples - fit.basis @ coefficients
        # Pythagoras holds under the SAME grid norm, not after substituting
        # the exact continuum norm for just the fitted term.
        torch.testing.assert_close(
            fit.grid_mean_square(samples),
            fit.grid_mean_square(fit.basis @ coefficients) + fit.grid_mean_square(residual),
            rtol=1e-10, atol=1e-12,
        )
