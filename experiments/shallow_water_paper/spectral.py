"""Spectral errors of the three-scalar state on any grid.

:class:`SpectralProjector` fits grid samples to the real spherical harmonics
of degree ``0 .. L`` inclusive by least squares in the grid's quadrature
inner product and measures the fit in the exact :math:`L^2(S^2)` norm, split
by degree (:meth:`GridProjection.energy_by_degree`). :func:`spectral_energy`
gives a solver state's per-degree energy from its own coefficients and
:func:`target_band_coverage` the fraction of it below ``L``.
:class:`SpectralErrorAccumulator` pools squared errors over an evaluation set
before any root is taken.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor
from torch_harmonics.legendre import legpoly

from experiments.shallow_water_paper.data import CHANNELS, ScalarNormalization
from gm_attention.data.shallow_water import spherical_coordinates
from gm_attention.discretization.grid import SphericalGrid
from gm_attention.discretization.grids.torch_harmonics import build_torch_harmonics_grid

FOUR_PI = 4.0 * math.pi
SQRT_FOUR_PI = math.sqrt(FOUR_PI)


def full_bandwidth(nlat: int) -> int:
    """The highest degree a solver at ``nlat`` rows carries: its truncation
    is ``ceil(nlat / 3)`` coefficients, degrees ``0 .. ceil(nlat/3) - 1``.
    128 rows give 42, 256 rows give 85."""
    return math.ceil(nlat / 3) - 1


def real_harmonic_basis(nodes: Tensor, bandwidth: int) -> tuple[Tensor, Tensor]:
    """``(n_nodes, (L+1)^2)`` float64 values of a real spherical-harmonic
    basis of degrees ``0 .. L`` at ``nodes``, and each column's degree.

    ``P_l^m(cos theta) cos(m phi)`` and ``P_l^m(cos theta) sin(m phi)`` from
    torch-harmonics' Legendre table. Orthogonal on the sphere, so the exact
    Gram is diagonal; not orthonormal, which the Gram absorbs.
    """
    theta, phi = spherical_coordinates(nodes.to(torch.float64))
    table = legpoly(
        bandwidth + 1, bandwidth + 1, torch.cos(theta),
        norm="ortho", inverse=True, csphase=False,
    ).to(device=nodes.device, dtype=torch.float64)  # (mmax, lmax, n_nodes)
    columns, degrees = [], []
    for order in range(bandwidth + 1):
        cosine, sine = torch.cos(order * phi), torch.sin(order * phi)
        for degree in range(order, bandwidth + 1):
            columns.append(table[order, degree] * cosine)
            degrees.append(degree)
            if order > 0:
                columns.append(table[order, degree] * sine)
                degrees.append(degree)
    return torch.stack(columns, dim=1), torch.tensor(degrees, device=nodes.device)


class SpectralProjector:
    """Least-squares projection of grid samples onto the real spherical
    harmonics of degree ``0 .. bandwidth``, in the grid's quadrature inner
    product, measured in the exact :math:`L^2(S^2)` norm (Gram integrated
    on a Gauss grid, hence diagonal). The norm is a property of the fitted
    function; the fit depends on the grid: where the quadrature is not exact
    for the band (an icosphere), content above ``bandwidth`` aliases into
    the retained coefficients.

    ``max_condition`` bounds the condition number of the grid's normal
    matrix for :meth:`fit` to accept it (near 2 for a resolvable band on a
    well-spread grid); an underdetermined or badly conditioned fit is
    refused.
    """

    def __init__(self, bandwidth: int, device: torch.device, *, max_condition: float = 50.0) -> None:
        if bandwidth < 0:
            raise ValueError(f"bandwidth must be nonnegative, got {bandwidth}.")
        self.bandwidth = int(bandwidth)
        self.device = device
        self.max_condition = float(max_condition)
        # L+2 Gauss rows integrate degree 2L+3 in cos(theta) exactly and 2L+4
        # longitudes every product of orders <= L: the Gram is exact, hence diagonal.
        reference = build_torch_harmonics_grid(
            "legendre-gauss", bandwidth + 2, 2 * bandwidth + 4, dtype=torch.float64, device=device
        )
        basis, self.degrees = real_harmonic_basis(reference.nodes, bandwidth)
        self.gram = basis.T @ (reference.quadrature_weights[:, None] * basis)  # (K, K)
        self.diagonal = self.gram.diagonal().clone()
        off = (self.gram - torch.diag(self.diagonal)).abs().max() / self.diagonal.max()
        if float(off) > 1e-8:
            raise RuntimeError(f"real harmonic basis is not orthogonal on the reference grid: {float(off):.2e}")

    @property
    def n_basis(self) -> int:
        return (self.bandwidth + 1) ** 2

    def basis(self, nodes: Tensor) -> Tensor:
        return real_harmonic_basis(nodes, self.bandwidth)[0]

    def fit(self, grid: SphericalGrid) -> GridProjection:
        """Bind the projector to a grid, refusing one that does not resolve
        the band."""
        if grid.n_nodes < self.n_basis:
            raise ValueError(
                f"{grid.n_nodes} nodes cannot determine {self.n_basis} coefficients: the "
                f"grid does not resolve degree {self.bandwidth}."
            )
        projection = GridProjection(self, grid)
        if projection.condition > self.max_condition:
            raise ValueError(
                f"fit of degree {self.bandwidth} on this grid has condition number "
                f"{projection.condition:.1f} (limit {self.max_condition:.0f}): the grid does "
                "not resolve the band well enough for the coefficients to be trusted."
            )
        return projection


class GridProjection:
    """:class:`SpectralProjector` bound to one grid's nodes and weights."""

    def __init__(self, projector: SpectralProjector, grid: SphericalGrid) -> None:
        self.projector = projector
        # The fit lives on the projector's device, which need not be the model's.
        self.basis = projector.basis(grid.nodes.to(projector.device))  # (n, K)
        self.weights = grid.quadrature_weights.to(device=projector.device, dtype=torch.float64)  # (n,), sums to 4 pi
        weighted = self.weights[:, None] * self.basis
        normal = self.basis.T @ weighted  # (K, K)
        eigenvalues = torch.linalg.eigvalsh(normal)
        self.condition = float(eigenvalues.max() / eigenvalues.min().clamp_min(1e-300))
        self._factor = torch.linalg.cholesky(normal)
        self._weighted_basis_t = weighted.T  # (K, n)

    @property
    def device(self) -> torch.device:
        return self.weights.device

    def project(self, field: Tensor) -> Tensor:
        """``(n, C)`` samples -> ``(K, C)`` least-squares coefficients."""
        return torch.cholesky_solve(self._weighted_basis_t @ field.to(self.device, torch.float64), self._factor)

    def band_field(self, coefficients: Tensor) -> Tensor:
        """``(K, C)`` -> ``(n, C)``: the fitted band-limited field back on
        this grid's nodes."""
        return self.basis @ coefficients

    def grid_l1(self, field: Tensor) -> Tensor:
        """``(n, C)`` -> ``(C,)``: the quadrature-weighted mean absolute
        value of the samples, the paper's L1 per channel."""
        return (self.weights[:, None] * field.to(self.device, torch.float64).abs()).sum(dim=0) / FOUR_PI

    def band_l1(self, coefficients: Tensor) -> Tensor:
        """``(K, C)`` -> ``(C,)``: the same L1 of the fitted band-limited
        field, under this grid's quadrature (exact for the fit's square,
        an estimate for its absolute value)."""
        return self.grid_l1(self.band_field(coefficients))

    def energy_by_degree(self, coefficients: Tensor) -> Tensor:
        """``(K, C)`` -> ``(L+1, C)``: the exact sphere-average of the square
        of the band-limited field, split by degree."""
        per_column = self.projector.diagonal[:, None] * coefficients.square() / FOUR_PI  # (K, C)
        out = torch.zeros(self.projector.bandwidth + 1, coefficients.shape[1], dtype=per_column.dtype, device=per_column.device)
        return out.index_add_(0, self.projector.degrees, per_column)

    def mean_square(self, coefficients: Tensor) -> Tensor:
        """``(K, C)`` -> ``(C,)``: the exact sphere-average of the square."""
        return self.energy_by_degree(coefficients).sum(dim=0)

    def grid_mean_square(self, field: Tensor) -> Tensor:
        """``(n, C)`` -> ``(C,)``: the same average of the raw samples under
        this grid's own quadrature."""
        return (self.weights[:, None] * field.to(self.device, torch.float64).square()).sum(dim=0) / FOUR_PI

    def residual_mean_square(self, field: Tensor, coefficients: Tensor) -> Tensor:
        """``(n, C)`` samples and their ``(K, C)`` fit -> ``(C,)``: the grid
        mean square of what the band fit leaves behind on this grid.

        With ``H = A^T W A`` the grid's own Gram, the fit is W-orthogonal:
        ``e^T W e = a^T H a + r^T W r`` exactly, whatever the grid. So this
        is the part of the native mean square that no field of degree
        ``<= L`` can represent on these nodes: a valid discrete
        decomposition, unlike native minus continuum band energy, which
        also carries ``a^T (H - G) a`` of either sign. Out-of-band content
        aliased into the fit stays in ``a^T H a`` and is not separable here.
        """
        residual = field.to(self.device, torch.float64) - self.basis @ coefficients
        return (self.weights[:, None] * residual.square()).sum(dim=0) / FOUR_PI


def spectral_energy(
    state: Tensor, normalization: ScalarNormalization | None, bandwidth: int
) -> Tensor:
    """``(3, lmax, mmax)`` complex solver coefficients -> ``(bandwidth+1, 3)``
    per-degree mean-square energies of the z-scored real channels:
    :math:`\\overline{f^2} = \\sum_{\\ell} \\sum_{m \\le \\ell} w_m
    |c_{\\ell m}|^2 / 4\\pi` with :math:`w_0 = 1`, :math:`w_{m>0} = 2`, and
    a constant field carrying :math:`c_{00} = \\sqrt{4\\pi}`. The imaginary
    part at order zero is discarded, as in the inverse real transform.

    The channel's mean is removed at ``(0, 0)`` and every coefficient
    divided by the channel's standard deviation, so the energies are those
    of the fields the loss sees. Degrees above ``bandwidth`` fall in no bin;
    :func:`target_band_coverage` gives the fraction they carry.
    """
    channels, lmax, mmax = state.shape
    degree = torch.arange(lmax, device=state.device)[:, None]
    order = torch.arange(mmax, device=state.device)[None, :]
    coefficients = state.to(torch.complex128) * (order <= degree)
    # The imaginary zonal (m=0) coefficients are discarded by the inverse
    # real transform and are not energy of the real field.
    coefficients[..., 0] = coefficients[..., 0].real.to(torch.complex128)
    if normalization is not None:
        mean = normalization.mean.to(device=state.device, dtype=torch.float64)
        std = normalization.std.to(device=state.device, dtype=torch.float64)
        coefficients = coefficients.clone()
        coefficients[:, 0, 0] = coefficients[:, 0, 0] - mean.to(torch.complex128) * SQRT_FOUR_PI
        coefficients = coefficients / std[:, None, None]
    weight = torch.full((mmax,), 2.0, dtype=torch.float64, device=state.device)
    weight[0] = 1.0
    by_degree = (coefficients.abs().square() * weight).sum(dim=-1) / FOUR_PI  # (3, lmax)
    top = min(bandwidth + 1, lmax)
    out = torch.zeros(bandwidth + 1, channels, dtype=torch.float64, device=state.device)
    out[:top] = by_degree[:, :top].T
    return out


def spectral_energy_total(state: Tensor, normalization: ScalarNormalization | None) -> Tensor:
    """``(3,)``: the whole mean-square energy of the z-scored channels."""
    lmax = state.shape[1]
    return spectral_energy(state, normalization, lmax - 1).sum(dim=0)


def target_band_coverage(
    targets: Sequence[Tensor], normalization: ScalarNormalization | None, bandwidth: int
) -> float:
    """The fraction of the targets' energy the band ``0 .. bandwidth`` holds,
    channels weighted equally as the loss weights them, pooled over the
    targets. One number: it depends on the targets and the band, not on
    any model's grid."""
    within = sum(spectral_energy(t, normalization, bandwidth).sum() for t in targets)
    total = sum(spectral_energy_total(t, normalization).sum() for t in targets)
    return float(within / total) if float(total) > 0 else float("nan")


@dataclass
class SpectralErrorAccumulator:
    """Pooled errors over an evaluation set: squared errors are summed
    across examples and rooted once, at report time, so a reported RMSE is
    that of the pooled error, not a mean of per-example RMSEs. The paper's
    L1/L2 (per-sample integrals averaged over samples) are kept by
    :class:`~experiments.shallow_water_paper.metrics.ScalarStateMetrics`."""

    bandwidth: int
    count: int = 0
    grid_square: Tensor = field(default_factory=lambda: torch.zeros(len(CHANNELS), dtype=torch.float64))
    residual_square: Tensor = field(default_factory=lambda: torch.zeros(len(CHANNELS), dtype=torch.float64))
    grid_abs: Tensor = field(default_factory=lambda: torch.zeros(len(CHANNELS), dtype=torch.float64))
    band_abs: Tensor = field(default_factory=lambda: torch.zeros(len(CHANNELS), dtype=torch.float64))
    band_by_degree: Tensor | None = None
    #: The grid's quadrature weights (summing to 4 pi), used when examples are added without a projection.
    quadrature_weights: Tensor | None = None

    def _grid_weights(self, like: Tensor) -> Tensor:
        if self.quadrature_weights is None:
            raise ValueError("adding without a projection needs the accumulator's quadrature_weights")
        return self.quadrature_weights.to(like.device, torch.float64)

    def add(self, projection: GridProjection | None, prediction: Tensor, target: Tensor) -> None:
        """Accumulate one example. Without a ``projection`` (a grid with
        fewer points than the band has coefficients) only the grid's own
        quantities are kept and the report carries no band-limited keys."""
        if projection is None:
            difference = (prediction - target).to(torch.float64)
            weights = self._grid_weights(prediction)
            self.grid_square = self.grid_square.to(difference.device) + (weights[:, None] * difference.square()).sum(0) / FOUR_PI
            self.grid_abs = self.grid_abs.to(difference.device) + (weights[:, None] * difference.abs()).sum(0) / FOUR_PI
            self.count += 1
            return
        difference = (prediction - target).to(projection.device, torch.float64)
        self.grid_square = self.grid_square.to(difference.device) + projection.grid_mean_square(difference)
        coefficients = projection.project(difference)
        by_degree = projection.energy_by_degree(coefficients)  # (L+1, 3)
        self.band_by_degree = by_degree if self.band_by_degree is None else self.band_by_degree + by_degree
        self.residual_square = self.residual_square.to(difference.device) + projection.residual_mean_square(difference, coefficients)
        self.grid_abs = self.grid_abs.to(difference.device) + projection.grid_l1(difference)
        self.band_abs = self.band_abs.to(difference.device) + projection.band_l1(coefficients)
        self.count += 1

    def report(self) -> dict[str, float | list[float]]:
        """Pooled metrics: ``paper_squared_l2`` (grid MSE, mean over channels)
        and the channel RMSEs; their band-limited counterparts under
        ``_L<B>``; ``D`` as ``D(l)`` for ``l = 0 .. B`` averaged equally
        over channels, with ``sum(D) == paper_squared_l2_L<B>``; and the
        fit's residual on the native grid, ``paper_squared_l2_residual_L<B>``
        (mean over channels) with ``fit_residual_fraction_L<B>`` its share
        of ``paper_squared_l2`` (see :meth:`GridProjection.residual_mean_square`);
        and the paper's L1 (quadrature-weighted mean absolute error, mean
        over channels and examples) as ``paper_l1`` on the grid and
        ``paper_l1_L<B>`` for the fitted band-limited error."""
        if self.count == 0:
            raise ValueError("nothing accumulated")
        suffix = f"_L{self.bandwidth}"
        grid_mse = self.grid_square / self.count  # (3,)
        if self.band_by_degree is None:  # accumulated without a projection
            out: dict[str, float | list[float]] = {"paper_squared_l2": float(grid_mse.mean())}
            for channel, name in enumerate(CHANNELS):
                out[f"{name}_rmse"] = float(grid_mse[channel].sqrt())
            out["paper_l1"] = float((self.grid_abs / self.count).mean())
            out["D"] = []
            return out
        band = self.band_by_degree / self.count  # (L+1, 3)
        band_mse = band.sum(dim=0)  # (3,)
        out = {"paper_squared_l2": float(grid_mse.mean()), f"paper_squared_l2{suffix}": float(band_mse.mean())}
        for channel, name in enumerate(CHANNELS):
            out[f"{name}_rmse"] = float(grid_mse[channel].sqrt())
            out[f"{name}_rmse{suffix}"] = float(band_mse[channel].sqrt())
        residual = self.residual_square / self.count  # (3,)
        out[f"paper_squared_l2_residual{suffix}"] = float(residual.mean())
        out[f"fit_residual_fraction{suffix}"] = float(residual.mean() / grid_mse.mean())
        out["paper_l1"] = float((self.grid_abs / self.count).mean())
        out[f"paper_l1{suffix}"] = float((self.band_abs / self.count).mean())
        out["D"] = [float(v) for v in band.mean(dim=1)]
        return out
