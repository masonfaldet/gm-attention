"""Shallow-water data as a three-scalar state.

The state is the three scalars of ``torch_harmonics``' ``PdeDataset``: the
solver's spectral state ``(geopotential, vorticity, divergence)`` passed
through the scalar inverse transform channel by channel, then z-scored per
channel. :class:`ScalarShallowWaterDataset` synthesizes that state at a
grid's nodes or on an equiangular view; the seeded spectral states and their
cache come from :class:`~gm_attention.data.shallow_water.ShallowWaterDataset`
and its :meth:`~gm_attention.data.shallow_water.ShallowWaterDataset.spectral_pair`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch import Tensor
from torch_harmonics import InverseRealSHT
from torch_harmonics.examples.losses import get_quadrature_weights

from gm_attention.data.shallow_water import (
    SOLVER_OMEGA,
    ArbitraryPointSHT,
    SamplePolicy,
    ShallowWaterDataset,
    Split,
)

#: Channel order of every ``(n_nodes, 3)`` tensor: the solver's own spectral state order.
CHANNELS: tuple[str, str, str] = ("geopotential", "vorticity", "divergence")


class ScalarState(NamedTuple):
    """One ``(input, target)`` pair at the nodes, point-major ``(n_nodes, 3)``
    in :data:`CHANNELS` order."""

    input: Tensor  # (n_nodes, 3) -- state at time t
    target: Tensor  # (n_nodes, 3) -- state at t + lead_time


@dataclass(frozen=True)
class ScalarNormalization:
    """Per-channel z-scores: three independent (mean, std) pairs, one per scalar channel."""

    mean: Tensor  # (3,)
    std: Tensor  # (3,)
    num_examples: int
    calibration_seed: int

    def normalize(self, field: Tensor) -> Tensor:
        """``(..., 3)`` raw -> standardized."""
        mean = self.mean.to(device=field.device, dtype=field.dtype)
        std = self.std.to(device=field.device, dtype=field.dtype)
        return (field - mean) / std

    def denormalize(self, field: Tensor) -> Tensor:
        """Inverse of :meth:`normalize`."""
        mean = self.mean.to(device=field.device, dtype=field.dtype)
        std = self.std.to(device=field.device, dtype=field.dtype)
        return field * std + mean

    def normalize_channel_first(self, field: Tensor) -> Tensor:
        """``(..., 3, nlat, nlon)`` raw -> standardized: the reference model's
        layout, channels ahead of the two grid axes."""
        mean = self.mean.to(device=field.device, dtype=field.dtype).view(-1, 1, 1)
        std = self.std.to(device=field.device, dtype=field.dtype).view(-1, 1, 1)
        return (field - mean) / std

    def apply(self, state: ScalarState, *, channel_first: bool = False) -> ScalarState:
        normalize = self.normalize_channel_first if channel_first else self.normalize
        return ScalarState(normalize(state.input), normalize(state.target))

    def state_dict(self) -> dict[str, Any]:
        """Plain Python values, for the run's protocol file."""
        return {
            "channels": list(CHANNELS),
            "mean": [float(value) for value in self.mean],
            "std": [float(value) for value in self.std],
            "num_examples": int(self.num_examples),
            "calibration_seed": int(self.calibration_seed),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> ScalarNormalization:
        return cls(
            mean=torch.as_tensor(state["mean"], dtype=torch.float64),
            std=torch.as_tensor(state["std"], dtype=torch.float64),
            num_examples=int(state["num_examples"]),
            calibration_seed=int(state["calibration_seed"]),
        )


class ScalarShallowWaterDataset(torch.utils.data.Dataset[ScalarState]):
    """The paper's three-scalar state at an icosphere's nodes.

    Parameters mirror :class:`~gm_attention.data.shallow_water.ShallowWaterDataset`
    (which does the solving): the split's ``base_seed``, the sample policy,
    the solver's resolution and timestepping, ``num_examples``, and the
    optional spectral cache. ``nodes`` are the points to synthesize at.
    """

    def __init__(
        self,
        nodes: Tensor | None = None,
        *,
        view: tuple[int, int] | None = None,
        view_grid: str = "equiangular",
        base_seed: int,
        split: Split,
        normalization: ScalarNormalization | None = None,
        sample_policy: SamplePolicy = "fixed_index",
        nlat: int = 256,
        nlon: int = 512,
        lead_time_seconds: float = 3600.0,
        solver_substeps: int = 24,
        mach: float = 0.2,
        omega: float = SOLVER_OMEGA,
        num_examples: int = 256,
        spectral_cache_dir: str | Path | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        # Point-major ``(n_nodes, 3)`` at arbitrary ``nodes``, or channel-first
        # ``(3, nlat, nlon)`` on an equiangular ``view`` (the ``spec2grid`` layout).
        if (nodes is None) == (view is None):
            raise ValueError("give exactly one of nodes (point-major) or view (equiangular).")
        if nodes is not None and (nodes.ndim != 2 or nodes.shape[-1] != 3):
            raise ValueError(f"nodes must have shape (n_nodes, 3), got {tuple(nodes.shape)}.")
        self.view = (int(view[0]), int(view[1])) if view is not None else None
        #: The view's grid family, ``equiangular`` or ``legendre-gauss``.
        self.view_grid = str(view_grid)
        self.channel_first = view is not None

        self.source = ShallowWaterDataset(
            base_seed=base_seed,
            split=split,
            sample_policy=sample_policy,
            nlat=nlat,
            nlon=nlon,
            lead_time_seconds=lead_time_seconds,
            solver_substeps=solver_substeps,
            mach=mach,
            omega=omega,
            num_examples=num_examples,
            spectral_cache_dir=spectral_cache_dir,
            dtype=dtype,
            device=device if device is not None else (nodes.device if nodes is not None else "cpu"),
        )
        self.synthesis: ArbitraryPointSHT | InverseRealSHT
        if nodes is not None:
            self.synthesis = ArbitraryPointSHT(
                nodes.to(device=self.source.device, dtype=torch.float64),
                lmax=self.source.lmax,
                mmax=self.source.mmax,
                nlon=nlon,
                table_dtype=dtype,
            )
        else:
            assert self.view is not None
            # The solver's truncation synthesized on the view's grid: spec2grid
            # at the solver's resolution, sparser sampling of it on a coarser view.
            self.synthesis = InverseRealSHT(
                self.view[0], self.view[1],
                lmax=self.source.lmax, mmax=self.source.mmax,
                grid=self.view_grid, csphase=False,
            ).to(device=self.source.device, dtype=dtype)
        self.normalization = normalization
        self.num_examples = int(num_examples)
        # Fixed-index splits are cached: solved once per process.
        self._cache: dict[int, ScalarState] | None = (
            {} if sample_policy == "fixed_index" else None
        )

    @property
    def device(self) -> torch.device:
        return self.source.device

    def set_epoch(self, epoch: int) -> None:
        """Advance the epoch an ``epoch_index`` split derives its samples from."""
        self.source.set_epoch(epoch)

    def sample_seed(self, index: int) -> int:
        return self.source.sample_seed(index)

    def __len__(self) -> int:
        return self.num_examples

    def _synthesize(self, spectral: Tensor) -> Tensor:
        """``(3, lmax, mmax)`` complex -> ``(n_nodes, 3)`` real at the nodes,
        or ``(3, nlat, nlon)`` on the view: all three channels through the
        scalar transform -- exactly ``spec2grid`` on the whole state."""
        synthesized = self.synthesis(spectral)
        return synthesized if self.channel_first else synthesized.movedim(0, -1)

    def __getitem__(self, index: int) -> ScalarState:
        if not 0 <= index < self.num_examples:
            raise IndexError(f"index {index} out of range for a {self.num_examples}-example dataset.")

        if self._cache is not None and index in self._cache:
            cached = self._cache[index]
            return ScalarState(cached.input.clone(), cached.target.clone())

        with torch.inference_mode():
            input_spectral, target_spectral = self.source.spectral_pair(index)
            state = ScalarState(self._synthesize(input_spectral), self._synthesize(target_spectral))
            if self.normalization is not None:
                state = self.normalization.apply(state, channel_first=self.channel_first)
        # Cloned outside inference_mode so autograd may consume them.
        state = ScalarState(state.input.clone(), state.target.clone())

        if self._cache is not None:
            self._cache[index] = ScalarState(state.input.clone(), state.target.clone())
        return state


def calibrate_scalar_normalization(
    *,
    seed: int,
    num_examples: int = 128,
    nlat: int = 256,
    nlon: int = 512,
    lead_time_seconds: float = 3600.0,
    solver_substeps: int = 24,
    mach: float = 0.2,
    omega: float = SOLVER_OMEGA,
    spectral_cache_dir: str | Path | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> ScalarNormalization:
    """Per-channel mean and std of the input state over a fixed calibration set.

    Area-weighted on the solver's own equiangular grid, independent of the
    grid that later consumes them. Accumulated in float64 about a shifted
    origin: geopotential has a large mean and a small spread, and
    ``E[X^2] - E[X]^2`` about zero would cancel most of its digits.
    """
    if num_examples < 1:
        raise ValueError(f"num_examples must be at least 1, got {num_examples}.")

    source = ShallowWaterDataset(
        base_seed=seed,
        split="normalization",
        nlat=nlat,
        nlon=nlon,
        lead_time_seconds=lead_time_seconds,
        solver_substeps=solver_substeps,
        mach=mach,
        omega=omega,
        num_examples=num_examples,
        spectral_cache_dir=spectral_cache_dir,
        dtype=dtype,
        device=device,
    )
    weights = get_quadrature_weights(nlat, nlon, grid="equiangular", tile=True, normalized=True).to(
        device=source.device, dtype=torch.float64
    )  # (nlat, nlon), sums to 1

    shift: Tensor | None = None
    total = torch.zeros(3, device=source.device, dtype=torch.float64)
    total_square = torch.zeros(3, device=source.device, dtype=torch.float64)
    with torch.inference_mode():
        for index in range(num_examples):
            input_spectral, _ = source.spectral_pair(index)
            field = source.solver.spec2grid(input_spectral).to(torch.float64)  # (3, nlat, nlon)
            if shift is None:
                shift = (weights * field).sum(dim=(-2, -1))
            centered = field - shift[:, None, None]
            total = total + (weights * centered).sum(dim=(-2, -1))
            total_square = total_square + (weights * centered.square()).sum(dim=(-2, -1))

    assert shift is not None  # num_examples >= 1, checked above
    mean_offset = total / num_examples
    variance = total_square / num_examples - mean_offset.square()
    return ScalarNormalization(
        mean=(shift + mean_offset).cpu().clone(),
        std=variance.clamp_min(0.0).sqrt().cpu().clone(),
        num_examples=int(num_examples),
        calibration_seed=int(seed),
    )
