"""Shallow-water-equations data generated on the fly by the torch-harmonics
solver and synthesized directly at arbitrary point sets.

``torch_harmonics.examples.shallow_water_equations.ShallowWaterSolver``
represents the solution as spherical-harmonic coefficients and synthesizes
them onto a fixed ``(nlat, nlon)`` grid through ``RealSHT``'s inverse.
``ArbitraryPointSHT`` is the arbitrary-point generalization of
``InverseRealSHT`` built from the same ``torch_harmonics.legendre.legpoly``
primitive: the identical Legendre contraction and longitude Fourier
synthesis, evaluated at given colatitude/longitude pairs such as an
icosphere's nodes, so the data are an exact sample of the solver's spectral
solution. ``ShallowWaterDataset`` holds the solver, the seeding that makes
an index name one fixed trajectory, and the on-disk cache of spectral
states, and hands out the solver's spectral states
(:meth:`ShallowWaterDataset.spectral_pair`); synthesis into a model's state
representation is the experiment's job.
"""

from __future__ import annotations

import logging
import math
import os
import pickle
from pathlib import Path
from typing import Literal

import torch
import torch_harmonics
from torch import Tensor, nn
from torch_harmonics.examples.shallow_water_equations import ShallowWaterSolver
from torch_harmonics.legendre import legpoly

from gm_attention.training.reproducibility import derive_data_seed, forked_torch_seed

logger = logging.getLogger(__name__)

Split = Literal["train", "val", "test", "normalization"]
SamplePolicy = Literal["fixed_index", "epoch_index"]


#: Earth's rotation axis in the ambient frame, +z: `coriolis_channel` is the node's z coordinate.
CORIOLIS_AXIS: tuple[float, float, float] = (0.0, 0.0, 1.0)

#: Earth's sidereal rotation rate, rad/s; ``ShallowWaterSolver``'s default, passed explicitly.
SOLVER_OMEGA: float = 7.292e-05


def coriolis_channel(
    nodes: Tensor, axis: tuple[float, float, float] = CORIOLIS_AXIS
) -> Tensor:
    """The dimensionless Coriolis conditioning channel ``c(p) = axis . p``.

    The Coriolis parameter is ``f(p) = 2 Omega . p``; this returns
    ``c = f / (2 |Omega|) = axis_hat . p``, which for the +z axis is
    ``p_z = sin(latitude)``. Dimensionless so that it is of order one next
    to z-scored state channels (``f`` itself is about ``1e-4 s^-1``); the
    rotation rate is recorded separately. A conditioning channel, not
    state: never z-scored, never a target, never scored. A ``rho_0``
    (invariant scalar) quantity, the dot product of two ambient vectors, so
    it is unchanged by a change of tangent-frame gauge.

    Parameters
    ----------
    nodes
        Unit-sphere points, shape ``(..., 3)``.
    axis
        Unit rotation axis in ambient coordinates. Defaults to
        :data:`CORIOLIS_AXIS`.

    Returns
    -------
    Tensor
        Shape ``nodes.shape[:-1]``, values in ``[-1, 1]``.
    """
    if nodes.ndim < 1 or nodes.shape[-1] != 3:
        raise ValueError(
            f"nodes must have shape (..., 3), got {tuple(nodes.shape)}."
        )

    axis_tensor = torch.tensor(axis, dtype=nodes.dtype, device=nodes.device)
    norm = torch.linalg.vector_norm(axis_tensor)

    if not bool(norm > 0):
        raise ValueError(f"axis must be nonzero, got {axis!r}.")

    return nodes @ (axis_tensor / norm)


def spherical_coordinates(nodes: Tensor) -> tuple[Tensor, Tensor]:
    """Colatitude and longitude for unit-sphere points.

    The convention ``x = sin(theta)cos(phi), y = sin(theta)sin(phi),
    z = cos(theta)`` of the grid builders
    (``discretization.grids.torch_harmonics.build_torch_harmonics_grid``).

    Parameters
    ----------
    nodes
        Unit-sphere points, shape ``(*, 3)``.

    Returns
    -------
    theta, phi
        Colatitude in ``[0, pi]`` and longitude in ``[0, 2*pi)``, each shape
        ``(*,)``.
    """
    x, y, z = nodes[..., 0], nodes[..., 1], nodes[..., 2]
    theta = torch.arccos(z.clamp(-1.0, 1.0))
    phi = torch.atan2(y, x) % (2.0 * math.pi)
    return theta, phi


def _real_fourier_series(coefficients: Tensor, phi: Tensor, nlon: int) -> Tensor:
    """Evaluate a real-signal Fourier series at arbitrary longitudes.

    Equals ``torch.fft.irfft(coefficients, n=nlon, dim=-1, norm="forward")``
    when ``phi`` is the grid ``2*pi*j/nlon``, and evaluates the same
    trigonometric polynomial off the grid. As in ``InverseRealSHT``, the
    imaginary parts of the ``m = 0`` and (for even ``nlon``) Nyquist
    coefficients are zeroed: ``sin(0) = 0`` enforces the former anyway, but
    the Nyquist mode's sine vanishes only on the original grid.

    Parameters
    ----------
    coefficients
        Complex, shape ``(..., mmax)``, ``mmax = nlon // 2 + 1``.
    phi
        Real, broadcastable against ``coefficients.shape[:-1]``.
    nlon
        The ``nlon`` this ``mmax`` was derived from.

    Returns
    -------
    Tensor
        Real, shape matching the broadcast of ``coefficients.shape[:-1]``
        and ``phi.shape``.
    """
    mmax = coefficients.shape[-1]

    real = coefficients.real
    imag = coefficients.imag.clone()
    imag[..., 0] = 0.0

    weight = torch.full((mmax,), 2.0, dtype=phi.dtype, device=coefficients.device)
    weight[0] = 1.0
    if nlon % 2 == 0 and nlon // 2 < mmax:
        weight[nlon // 2] = 1.0
        imag[..., nlon // 2] = 0.0

    m = torch.arange(mmax, device=coefficients.device, dtype=phi.dtype)
    angle = phi.unsqueeze(-1) * m  # (..., mmax)

    return (weight * (real * torch.cos(angle) - imag * torch.sin(angle))).sum(dim=-1)


class ArbitraryPointSHT(nn.Module):
    """Synthesizes a scalar spherical-harmonic expansion at arbitrary points.

    The arbitrary-point generalization of ``torch_harmonics.InverseRealSHT``.

    Parameters
    ----------
    nodes
        Unit-sphere points to synthesize at, shape ``(n_points, 3)``.
    lmax, mmax
        Must match the truncation of the coefficients this is called on
        (e.g. a ``ShallowWaterSolver``'s own ``.lmax``/``.mmax``).
    nlon
        The ``nlon`` the solver's ``mmax`` was derived from
        (``mmax = nlon // 2 + 1``), used only for the Nyquist handling of
        the longitude synthesis.
    table_dtype
        Dtype the precomputed ``(mmax, lmax, n_points)`` Legendre table is
        stored at; :meth:`forward` casts it to the input's dtype, so storing
        it at the training dtype makes that cast a no-op.
    """

    pct: Tensor
    phi: Tensor

    def __init__(
        self,
        nodes: Tensor,
        lmax: int,
        mmax: int,
        nlon: int,
        *,
        table_dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()

        if not isinstance(nodes, Tensor) or nodes.ndim != 2 or nodes.shape[-1] != 3:
            raise ValueError(
                f"nodes must have shape (n_points, 3), got {tuple(getattr(nodes, 'shape', ()))}."
            )

        self.nlon = nlon
        theta, phi = spherical_coordinates(nodes)

        # Computed in float64 (the legpoly recursion is ill-conditioned at
        # high degree), then stored at table_dtype.
        pct = legpoly(mmax, lmax, torch.cos(theta).double(), norm="ortho", inverse=True, csphase=False)
        self.register_buffer("pct", pct.to(table_dtype))  # (mmax, lmax, n_points)
        self.register_buffer("phi", phi.to(table_dtype))  # (n_points,)

    def forward(self, coefficients: Tensor) -> Tensor:
        """
        Parameters
        ----------
        coefficients
            Complex, shape ``(..., lmax, mmax)``.

        Returns
        -------
        Tensor
            Real, shape ``(..., n_points)``.
        """
        real = torch.view_as_real(coefficients)  # (..., lmax, mmax, 2)
        pct = self.pct.to(real.dtype)
        per_point = torch.einsum("...lmr,mlk->...kmr", real, pct).contiguous()
        per_point_complex = torch.view_as_complex(per_point)  # (..., n_points, mmax)
        return _real_fourier_series(per_point_complex, self.phi.to(real.dtype), self.nlon)


class ShallowWaterDataset(torch.utils.data.Dataset):
    """On-the-fly shallow-water-equations spectral pairs.

    Wraps ``torch_harmonics.examples.shallow_water_equations.ShallowWaterSolver``
    with the defaults of "Attention on the Sphere", Appendix C.2: a
    ``(256, 512)`` equiangular grid, ``mach=0.2`` random initial conditions,
    and a one-hour (``24`` substeps of ``150`` s) single-step target.

    The index identifies the sample: ``(base_seed, split, index)``, plus
    ``epoch`` under the ``epoch_index`` policy, is hashed into one sample
    seed (:func:`~gm_attention.training.reproducibility.derive_data_seed`)
    and the solver's draw is isolated inside
    :func:`~gm_attention.training.reproducibility.forked_torch_seed`, so
    ``spectral_pair(i)`` is the same trajectory on every access, in every
    run, for every architecture, independent of how much the model side
    advanced the global RNG. ``fixed_index`` (the default) repeats the set
    across epochs and is what validation and test use; ``epoch_index``
    re-derives training samples per epoch via :meth:`set_epoch`.

    Parameters
    ----------
    base_seed
        This split's data seed, independent of the model seed. Together
        with ``split`` and the sample index (and ``epoch``, under
        ``epoch_index``) it determines every sample completely.
    split
        Literal split name, mixed into the sample seed as itself.
    sample_policy
        ``"fixed_index"`` (default): index alone identifies the sample, in
        every epoch. ``"epoch_index"``: :meth:`set_epoch` additionally
        varies it. Validation and test stay ``"fixed_index"``.
    nlat, nlon
        Equiangular grid resolution the solver itself runs at.
    lead_time_seconds, solver_substeps
        The target is ``solver_substeps`` Adams-Bashforth steps of
        ``lead_time_seconds / solver_substeps`` each, applied to the input.
    mach
        Passed to ``ShallowWaterSolver.random_initial_condition``.
    num_examples
        ``len(dataset)``.
    dtype
        Floating-point dtype the solver runs at.
    device
        Device for the solver. Defaults to CPU.
    """

    def __init__(
        self,
        *,
        base_seed: int,
        split: Split,
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
        if sample_policy not in ("fixed_index", "epoch_index"):
            raise ValueError(
                f"sample_policy must be 'fixed_index' or 'epoch_index', got {sample_policy!r}."
            )
        if split not in ("train", "val", "test", "normalization"):
            raise ValueError(
                "split must be 'train', 'val', 'test' or 'normalization', got "
                f"{split!r}."
            )

        self.base_seed = int(base_seed)
        self.split: Split = split
        self.sample_policy: SamplePolicy = sample_policy
        self.num_examples = num_examples
        self.mach = mach
        self.solver_substeps = solver_substeps
        self.epoch = 0
        self.dtype = dtype

        # Optional on-disk cache of pre-synthesis spectral states, shared
        # across runs; the state depends only on the seed and the solver.
        self.spectral_cache_dir = (
            Path(spectral_cache_dir) if spectral_cache_dir is not None else None
        )

        resolved_device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        self.device = resolved_device

        lmax = math.ceil(nlat / 3)
        mmax = lmax
        dt_solver = lead_time_seconds / solver_substeps

        # omega is passed explicitly so the rotation rate is a recorded
        # property of the dataset.
        self.omega = float(omega)
        self.solver = ShallowWaterSolver(
            nlat, nlon, dt_solver, lmax=lmax, mmax=mmax, grid="equiangular",
            omega=self.omega,
        ).to(device=resolved_device, dtype=dtype)

        # Recorded for the spectral cache key.
        self.solver_nlat, self.solver_nlon = int(nlat), int(nlon)
        self.lmax, self.mmax = int(lmax), int(mmax)

    def set_epoch(self, epoch: int) -> None:
        """Advance the epoch an ``epoch_index`` policy derives sample seeds from.

        Without effect under ``fixed_index``, so a training loop may call it
        unconditionally.
        """
        self.epoch = int(epoch)

    def sample_seed(self, index: int) -> int:
        """The seed ``index`` currently resolves to."""
        epoch = self.epoch if self.sample_policy == "epoch_index" else 0
        return derive_data_seed(self.base_seed, self.split, index, epoch)

    def _spectral_cache_path(self, index: int) -> Path:
        """Where index ``index``'s spectral pair lives on disk.

        Keyed by everything that determines the trajectory: the sample seed
        (base seed, split, index and epoch) and the solver configuration
        (resolution, truncation, substeps, time step, mach, rotation rate),
        so runs with different physics cannot read each other's entries.
        """
        assert self.spectral_cache_dir is not None
        solver = (
            f"{self.solver_nlat}x{self.solver_nlon}"
            f"_l{self.lmax}m{self.mmax}"
            f"_s{self.solver_substeps}"
            f"_dt{float(self.solver.dt)!r}"
            f"_mach{float(self.mach)!r}"
            f"_omega{self.omega!r}"
            # dtype is part of the key: entries are returned as stored.
            f"_{str(self.dtype).removeprefix('torch.')}"
            # CPU and CUDA draw different random streams for the same seed,
            # and library versions determine the numerical trajectory.
            f"_{self.device.type}_torch{torch.__version__}_th{torch_harmonics.__version__}"
        )
        return self.spectral_cache_dir / solver / f"{self.sample_seed(index)}.pt"

    def spectral_pair(self, index: int) -> tuple[Tensor, Tensor]:
        """The ``(input, target)`` spectral states index ``index`` denotes,
        before any synthesis: ``(3, lmax, mmax)`` complex each, in the
        solver's (geopotential, vorticity, divergence) coefficients.

        Seed derivation and the on-disk cache live here, so two views of
        index ``index`` are two syntheses of one solve.
        """
        return self._generate_spectral_pair(index)

    def _generate_spectral_pair(self, index: int) -> tuple[Tensor, Tensor]:
        """The ``(input, target)`` spectral states index ``index`` denotes.

        Only the random draw is isolated inside the forked RNG;
        ``solver.timestep`` is deterministic given its input.
        """
        path = self._spectral_cache_path(index) if self.spectral_cache_dir else None

        if path is not None and path.is_file():
            try:
                # weights_only: entries hold nothing but tensors, and a shared
                # cache directory must not execute arbitrary pickled objects.
                cached = torch.load(path, map_location=self.device, weights_only=True)
                # Returned as stored; the key pins the solver configuration and dtype.
                return cached["input"], cached["target"]
            except (OSError, RuntimeError, EOFError, ValueError, KeyError,
                    AttributeError, TypeError, pickle.UnpicklingError) as error:
                # An unreadable entry (concurrent writers can expose a partial
                # file) is regenerated; logged so a cache that never hits is visible.
                logger.warning("unreadable spectral cache entry %s: %s", path, error)

        with forked_torch_seed(self.sample_seed(index), self.device):
            input_spec = self.solver.random_initial_condition(mach=self.mach)

        target_spec = self.solver.timestep(input_spec, self.solver_substeps)

        if path is not None:
            self._write_spectral_cache(path, input_spec, target_spec)
        return input_spec, target_spec

    def _write_spectral_cache(
        self, path: Path, input_spec: Tensor, target_spec: Tensor
    ) -> None:
        """Persist one spectral pair, atomically.

        Concurrent runs race to write the same entries; writing to a
        process-unique temporary and renaming (atomic within a filesystem)
        leaves a reader either the old file or a complete new one, and the
        racing writers agree on content.
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(f".{os.getpid()}.tmp")
            torch.save(
                {"input": input_spec.cpu(), "target": target_spec.cpu()}, temporary
            )
            os.replace(temporary, path)
        except OSError as error:
            # A full or read-only cache directory does not stop training.
            logger.warning("could not write spectral cache entry %s: %s", path, error)

    def __len__(self) -> int:
        return self.num_examples
