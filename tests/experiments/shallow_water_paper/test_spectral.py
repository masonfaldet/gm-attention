"""Tests for the spectral yardstick: conventions, projector, coverage, pooling."""

from __future__ import annotations

import math

import pytest
import torch
from torch_harmonics import InverseRealSHT
from torch_harmonics.examples.losses import get_quadrature_weights

from experiments.shallow_water.grid_transfer import GridSpec
from experiments.shallow_water_paper.data import ScalarNormalization
from experiments.shallow_water_paper.spectral import (
    FOUR_PI,
    SQRT_FOUR_PI,
    SpectralErrorAccumulator,
    SpectralProjector,
    full_bandwidth,
    spectral_energy,
    spectral_energy_total,
    target_band_coverage,
)

CPU = torch.device("cpu")


def _random_state(lmax: int, seed: int = 0, channels: int = 3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    state = torch.randn(channels, lmax, lmax, dtype=torch.complex128, generator=generator)
    state[..., 0] = state[..., 0].real.to(torch.complex128)
    degree = torch.arange(lmax)[:, None]
    order = torch.arange(lmax)[None, :]
    return state * (order <= degree)


def test_full_bandwidth_follows_the_solvers_truncation() -> None:
    assert full_bandwidth(128) == 42 and full_bandwidth(256) == 85


def test_the_coefficient_convention_is_parseval_with_the_m_weights() -> None:
    """The solver's coefficients satisfy mean(f^2) = sum w_m |c|^2 / 4 pi
    with w_0 = 1, w_m = 2, degree by degree; a constant carries sqrt(4 pi)."""
    nlat, nlon, lmax = 64, 128, 22
    isht = InverseRealSHT(nlat, nlon, lmax=lmax, mmax=lmax, grid="legendre-gauss", csphase=False).double()
    weights = get_quadrature_weights(nlat, nlon, grid="legendre-gauss", tile=True, normalized=True).double()
    state = _random_state(lmax, channels=1)
    energies = spectral_energy(state, None, lmax - 1)  # (lmax, 1)
    grid_ms = float((weights * isht(state)[0].square()).sum())
    # get_quadrature_weights builds its weights in float32; 1e-6 is that precision.
    assert float(energies.sum()) == pytest.approx(grid_ms, rel=1e-6)
    for degree in (0, 5, 21):
        alone = torch.zeros_like(state)
        alone[:, degree] = state[:, degree]
        assert float(energies[degree, 0]) == pytest.approx(float((weights * isht(alone)[0].square()).sum()), rel=1e-6)
    constant = torch.zeros(1, lmax, lmax, dtype=torch.complex128)
    constant[0, 0, 0] = SQRT_FOUR_PI
    assert torch.allclose(isht(constant), torch.ones(nlat, nlon, dtype=torch.float64))


def test_normalization_removes_the_mean_and_scales_each_channel() -> None:
    state = _random_state(12)
    normalization = ScalarNormalization(
        mean=torch.tensor([3.0, 0.0, -1.0], dtype=torch.float64),
        std=torch.tensor([2.0, 4.0, 0.5], dtype=torch.float64), num_examples=1, calibration_seed=0,
    )
    raw = spectral_energy(state, None, 11)
    scaled = spectral_energy(state, normalization, 11)
    # Above degree 0 only the channel scale acts.
    torch.testing.assert_close(scaled[1:], raw[1:] / normalization.std.square()[None, :])
    # At degree 0 a state whose channel means equal the normalization means has no energy left.
    exact = state.clone()
    exact[:, 0, 0] = (normalization.mean * SQRT_FOUR_PI).to(torch.complex128)
    assert torch.allclose(spectral_energy(exact, normalization, 11)[0], torch.zeros(3, dtype=torch.float64))


def test_the_exact_gram_is_diagonal_and_degrees_partition_the_energy() -> None:
    projector = SpectralProjector(8, CPU)
    off = projector.gram - torch.diag(projector.diagonal)
    assert float(off.abs().max() / projector.diagonal.max()) < 1e-10
    assert projector.degrees.shape == (81,) and int(projector.degrees.max()) == 8
    projection = projector.fit(GridSpec.parse("legendre-gauss:16x32").build(CPU))
    coefficients = torch.randn(81, 3, dtype=torch.float64)
    by_degree = projection.energy_by_degree(coefficients)
    assert by_degree.shape == (9, 3)
    torch.testing.assert_close(by_degree.sum(0), projection.mean_square(coefficients))


def test_fit_refuses_a_grid_that_does_not_resolve_the_band() -> None:
    with pytest.raises(ValueError, match="does not resolve degree 20"):
        SpectralProjector(20, CPU).fit(GridSpec.parse("icosphere:2").build(CPU))  # 162 nodes < 441
    projection = SpectralProjector(12, CPU).fit(GridSpec.parse("icosphere:3").build(CPU))  # 642 >= 169
    assert projection.condition < 5


def test_content_above_the_band_aliases_on_an_icosphere_but_not_on_an_exact_grid() -> None:
    """Mixed low + high degree samples: with exact quadrature the retained
    coefficients are the low ones exactly; on an icosphere the high content
    leaks into them, and the leak is measured rather than assumed away."""
    low, high = SpectralProjector(6, CPU), SpectralProjector(14, CPU)
    generator = torch.Generator().manual_seed(1)
    c_low = torch.randn(low.n_basis, 1, dtype=torch.float64, generator=generator)
    c_high = torch.zeros(high.n_basis, 1, dtype=torch.float64)
    mask = high.degrees >= 10
    c_high[mask] = torch.randn(int(mask.sum()), 1, dtype=torch.float64, generator=generator)

    def leak(grid_text: str) -> float:
        grid = GridSpec.parse(grid_text).build(CPU)
        samples = low.fit(grid).basis @ c_low + high.fit(grid).basis @ c_high
        recovered = low.fit(grid).project(samples)
        return float((recovered - c_low).norm() / c_low.norm())

    # Grids are built in float32 (the models' dtype), so "exact" means float32 precision.
    gauss = leak("legendre-gauss:24x48")  # 24 rows integrate degree-47 products exactly
    assert gauss < 1e-5
    icosphere = leak("icosphere:3")
    assert 10 * gauss < icosphere < 0.1, f"aliasing leak on L3 should be small but real, got {icosphere} vs {gauss} exact"


def test_fitting_a_wider_band_and_truncating_matches_only_with_exact_quadrature() -> None:
    narrow, wide = SpectralProjector(6, CPU), SpectralProjector(12, CPU)
    generator = torch.Generator().manual_seed(2)
    c = torch.randn(wide.n_basis, 1, dtype=torch.float64, generator=generator)

    def nesting_gap(grid_text: str) -> float:
        grid = GridSpec.parse(grid_text).build(CPU)
        samples = wide.fit(grid).basis @ c
        direct = narrow.fit(grid).project(samples)
        via_wide = wide.fit(grid).project(samples)[wide.degrees <= 6]
        return float((direct - via_wide).norm() / via_wide.norm())

    gauss = nesting_gap("legendre-gauss:24x48")
    assert gauss < 1e-5
    assert 10 * gauss < nesting_gap("icosphere:3") < 0.1


def test_coverage_is_one_at_the_full_band_and_monotone_below_it() -> None:
    lmax = 22
    targets = [_random_state(lmax, seed=s) for s in range(3)]
    normalization = ScalarNormalization(
        mean=torch.tensor([50.0, 0.0, 0.0], dtype=torch.float64),
        std=torch.tensor([3.0, 1.0, 1.0], dtype=torch.float64), num_examples=1, calibration_seed=0,
    )
    full = target_band_coverage(targets, normalization, lmax - 1)
    assert full == pytest.approx(1.0)
    values = [target_band_coverage(targets, normalization, L) for L in (0, 5, 10, 15, 21)]
    assert all(0.0 <= a <= b <= 1.0 + 1e-12 for a, b in zip(values, values[1:]))
    # A constant offset in a channel changes nothing once the mean is removed.
    shifted = [t.clone() for t in targets]
    for t in shifted:
        t[0, 0, 0] += 1000.0 * SQRT_FOUR_PI
    shifted_norm = ScalarNormalization(mean=normalization.mean + torch.tensor([1000.0, 0.0, 0.0], dtype=torch.float64), std=normalization.std, num_examples=1, calibration_seed=0)
    assert target_band_coverage(shifted, shifted_norm, 10) == pytest.approx(values[2])
    assert float(spectral_energy_total(targets[0], normalization).sum()) > 0


def test_pooled_rmse_is_the_root_of_the_pooled_mean_square() -> None:
    projector = SpectralProjector(4, CPU)
    grid = GridSpec.parse("legendre-gauss:8x16").build(CPU)
    projection = projector.fit(grid)
    accumulator = SpectralErrorAccumulator(4)
    target = torch.zeros(grid.n_nodes, 3, dtype=torch.float64)
    errors = (1.0, 3.0)  # constant errors of 1 and 3 in every channel
    for e in errors:
        accumulator.add(projection, target + e, target)
    report = accumulator.report()
    pooled = math.sqrt(sum(e * e for e in errors) / len(errors))  # sqrt(5) = 2.236, not the mean 2
    assert report["geopotential_rmse"] == pytest.approx(pooled, rel=1e-6)
    assert report["paper_squared_l2"] == pytest.approx(pooled**2, rel=1e-6)
    # A constant field is pure degree 0, so the band recovers it entirely.
    assert report["geopotential_rmse_L4"] == pytest.approx(pooled, rel=1e-6)
    D = report["D"]
    assert isinstance(D, list) and len(D) == 5
    assert sum(D) == pytest.approx(report["paper_squared_l2_L4"], rel=1e-12)
    assert D[0] == pytest.approx(pooled**2, rel=1e-6) and max(D[1:]) < 1e-10 * pooled**2
    assert FOUR_PI > 12


def test_solver_imaginary_zonal_coefficients_carry_no_real_field_energy() -> None:
    """The solver's random initial conditions are complex even at m=0;
    irfft discards that imaginary part, so Parseval must discard it too."""
    from gm_attention.data.shallow_water import ShallowWaterDataset

    source = ShallowWaterDataset(
        base_seed=3000, split="test",
        nlat=32, nlon=64, lead_time_seconds=300, solver_substeps=2, num_examples=1,
    )
    state = source.spectral_pair(0)[1].to(torch.complex128)
    assert bool((state[..., 0].imag != 0).any())
    synthesis = InverseRealSHT(32, 64, lmax=11, mmax=11, grid="legendre-gauss", csphase=False).double()
    grid = GridSpec.parse("legendre-gauss:32x64").build(CPU)
    normalization = ScalarNormalization(
        mean=torch.tensor([98000.0, 0.0, 0.0], dtype=torch.float64),
        std=torch.tensor([100.0, 1e-6, 1e-6], dtype=torch.float64), num_examples=1, calibration_seed=0,
    )
    field = normalization.normalize_channel_first(synthesis(state)).movedim(0, -1).reshape(-1, 3)
    expected = (grid.quadrature_weights.double()[:, None] * field.square()).sum(0) / FOUR_PI
    torch.testing.assert_close(spectral_energy_total(state, normalization), expected, rtol=1e-6, atol=1e-10)
    changed = state.clone()
    changed[..., 0] += 1000j
    torch.testing.assert_close(spectral_energy(changed, normalization, 10), spectral_energy(state, normalization, 10))
    assert target_band_coverage([changed], normalization, 4) == pytest.approx(target_band_coverage([state], normalization, 4))


@pytest.mark.parametrize("spec", ["legendre-gauss:8x16", "icosphere:2"])
def test_the_fit_residual_completes_the_grid_norm_exactly(spec: str) -> None:
    """``e^T W e = a^T H a + r^T W r`` on any grid; on a Gauss grid whose
    quadrature is exact for the products, the residual is the out-of-band
    energy itself."""
    projector = SpectralProjector(4, CPU)
    grid = GridSpec.parse(spec).build(CPU)
    projection = projector.fit(grid)
    generator = torch.Generator().manual_seed(1)
    in_band = torch.randn(projection.basis.shape[1], 3, dtype=torch.float64, generator=generator)
    from experiments.shallow_water.grid_transfer import (
        GridSpec as _G,  # noqa: F401  (grid nodes only)
    )
    # an out-of-band field: degree 6, sampled on this grid through the projector's basis at bandwidth 6
    wide = SpectralProjector(6, CPU).fit(grid)
    out_coefficients = torch.zeros(wide.basis.shape[1], 3, dtype=torch.float64)
    out_coefficients[wide.projector.degrees == 6] = torch.randn(int((wide.projector.degrees == 6).sum()), 3, dtype=torch.float64, generator=generator)
    field = projection.basis @ in_band + wide.basis @ out_coefficients
    accumulator = SpectralErrorAccumulator(4)
    accumulator.add(projection, field, torch.zeros_like(field))
    report = accumulator.report()
    coefficients = projection.project(field)
    fitted_grid = projection.grid_mean_square(projection.basis @ coefficients)
    residual = projection.residual_mean_square(field, coefficients)
    grid_square = projection.grid_mean_square(field)
    torch.testing.assert_close(fitted_grid + residual, grid_square, rtol=1e-10, atol=0)
    assert report["fit_residual_fraction_L4"] == pytest.approx(float(residual.mean() / grid_square.mean()), rel=1e-12)
    assert report["paper_squared_l2_residual_L4"] == pytest.approx(float(residual.mean()), rel=1e-12)
    assert 0.0 < report["fit_residual_fraction_L4"] < 1.0
    if spec.startswith("legendre-gauss"):
        # exact quadrature through degree 15 makes the degree-6 field W-orthogonal to the band
        out_energy = wide.mean_square(out_coefficients)
        torch.testing.assert_close(residual, out_energy, rtol=1e-6, atol=0)  # float32 node coordinates
        torch.testing.assert_close(coefficients, in_band, rtol=1e-5, atol=1e-7)  # float32 node coordinates


def test_the_accumulator_reports_the_papers_l1_on_the_grid_and_in_band() -> None:
    projector = SpectralProjector(4, CPU)
    grid = GridSpec.parse("legendre-gauss:8x16").build(CPU)
    projection = projector.fit(grid)
    accumulator = SpectralErrorAccumulator(4)
    target = torch.zeros(grid.n_nodes, 3, dtype=torch.float64)
    for e in (1.0, -3.0):  # constant errors: |e| integrates to |e|, and a constant is pure degree 0
        accumulator.add(projection, target + e, target)
    report = accumulator.report()
    assert report["paper_l1"] == pytest.approx(2.0, rel=1e-6)  # mean over examples of |1| and |-3|
    assert report["paper_l1_L4"] == pytest.approx(2.0, rel=1e-6)
    assert report["paper_squared_l2"] == pytest.approx(5.0, rel=1e-6)


def test_the_projector_can_live_on_another_device_than_the_samples() -> None:
    projector = SpectralProjector(4, CPU)
    grid = GridSpec.parse("icosphere:2").build(CPU)
    projection = projector.fit(grid)
    assert projection.device == CPU
    field = torch.randn(grid.n_nodes, 3, dtype=torch.float32)
    coefficients = projection.project(field)
    assert coefficients.device == CPU and coefficients.dtype == torch.float64
    torch.testing.assert_close(projection.band_field(coefficients).shape, field.shape)


def test_the_accumulator_without_a_projection_keeps_the_grids_own_errors_only() -> None:
    grid = GridSpec.parse("icosphere:1").build(CPU)  # 42 nodes: below a bandwidth-6 band's 49 coefficients
    accumulator = SpectralErrorAccumulator(6, quadrature_weights=grid.quadrature_weights)
    target = torch.zeros(grid.n_nodes, 3, dtype=torch.float64)
    for e in (1.0, -3.0):
        accumulator.add(None, target + e, target)
    report = accumulator.report()
    assert report["paper_squared_l2"] == pytest.approx(5.0, rel=1e-6) and report["paper_l1"] == pytest.approx(2.0, rel=1e-6)
    assert report["geopotential_rmse"] == pytest.approx(5.0 ** 0.5, rel=1e-6)
    assert report["D"] == [] and not any(k.endswith("_L6") for k in report)
    with pytest.raises(ValueError, match="quadrature_weights"):
        SpectralErrorAccumulator(6).add(None, target + 1.0, target)
