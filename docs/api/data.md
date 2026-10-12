# Data

The shallow-water data of the paper's experiments (the paper, Section 5.1 and
Appendix C) is not stored. Every trajectory is synthesized at run time by the
spectral solver of torch-harmonics
(`torch_harmonics.examples.shallow_water_equations.ShallowWaterSolver`) from a
fixed seed. The solver represents the state -- geopotential, vorticity,
divergence, in that order -- as spherical-harmonic coefficients;
`gm_attention.data.shallow_water` supplies the seeded, optionally cached
production of those coefficients and the primitives that evaluate them at a
model's nodes. Evaluation on a particular grid is left to the experiment, so
every model reads samples of the same spectral solution.

The module reads no environment switch. The bundles' `run.sh` scripts set the
cache directory from their own `SWE_CACHE` variable.

## Pieces

The module defines, in source order:

- `coriolis_channel` returns `c(p) = <axis, p>` for unit-sphere points `p`,
  with the rotation axis normalized and `+z` by default (`CORIOLIS_AXIS`), so
  `c = sin(latitude)`. It is the dimensionless Coriolis parameter
  `f / (2 |Omega|)`, a conditioning channel and not a state channel, and an
  invariant scalar (type `rho_0`): a change of tangent-frame gauge leaves it
  unchanged.
- `spherical_coordinates` returns colatitude `theta` in `[0, pi]` and longitude
  `phi` in `[0, 2 pi)` under `x = sin(theta) cos(phi)`, `y = sin(theta)
  sin(phi)`, `z = cos(theta)`, the convention of
  [`build_torch_harmonics_grid`][gm_attention.discretization.grids.torch_harmonics.build_torch_harmonics_grid].
- [`ArbitraryPointSHT`][gm_attention.data.shallow_water.ArbitraryPointSHT]
  generalizes torch-harmonics' `InverseRealSHT` from a grid to an arbitrary
  point set such as an icosphere's nodes. It contracts the coefficients with
  the table `legpoly` builds at the points' colatitudes (computed in float64,
  stored at `table_dtype`) and sums the longitude Fourier series at their
  longitudes; the series agrees with `irfft` on the equiangular longitudes.
  Its `lmax`, `mmax` must be those of the coefficients it is applied to.
- [`ShallowWaterDataset`][gm_attention.data.shallow_water.ShallowWaterDataset]
  wraps the solver. Its truncation is `lmax = mmax = ceil(nlat / 3)`, its
  time step is `lead_time_seconds / solver_substeps`, and its defaults are a
  256x512 equiangular grid, a one-hour lead in 24 substeps, and Mach number
  0.2. `spectral_pair(index)` returns the `(input, target)` coefficients, each
  complex of shape `(3, lmax, mmax)`; the dataset never synthesizes them.

## Seeding and cache

A sample is identified by `(base_seed, split, index)`, plus the epoch under the
`epoch_index` policy. The tuple is hashed by
[`derive_data_seed`][gm_attention.training.reproducibility.derive_data_seed],
and the solver's random initial condition is drawn inside
[`forked_torch_seed`][gm_attention.training.reproducibility.forked_torch_seed],
so `spectral_pair(i)` does not depend on how far the model side has advanced
the global RNG; the target is the deterministic `solver.timestep` of that
initial condition. `fixed_index` (the default) repeats the same samples every
epoch and is the policy of validation and test; `epoch_index` re-derives the
training samples after `set_epoch`.

With `spectral_cache_dir` set, each pair is stored under a key made of the
sample seed, the solver configuration (resolution, truncation, substeps, time
step, Mach number, rotation rate), the dtype, the device type, and the torch
and torch-harmonics versions. An unreadable entry is regenerated with a logged
warning, and a failed write is logged and does not stop the run; entries are
written through a temporary file and renamed.

## Use in the experiments

`experiments/shallow_water_paper/data.py` builds on this module.
`ScalarShallowWaterDataset` takes the spectral pair from a
`ShallowWaterDataset` and synthesizes all three channels either at a grid's
nodes with `ArbitraryPointSHT` (the icosphere models) or on an equiangular or
Legendre-Gauss view with `InverseRealSHT` (the torch-harmonics reference models), then
standardizes each channel; `calibrate_scalar_normalization` draws the
statistics from the `"normalization"` split. `train.py` appends
`coriolis_channel` of the grid's nodes to the input as a fourth channel, static
and absent from the predicted state (the paper, Section 5.1), and `spectral.py`
uses `spherical_coordinates` to tabulate the real spherical-harmonic basis of
the band-limited error (the paper, Appendix C.1).

The protocol is in the bundle READMEs: [128x256
benchmark](../experiments/swe_128x256_benchmark.md) (30-minute lead, 12
substeps, 43 retained degrees), [256x512
ablation](../experiments/swe_256x512_ablation.md) and [256x512 zero-shot
transfer](../experiments/swe_256x512_zeroshot.md) (one-hour lead, 24 substeps,
86 retained degrees). All three draw 256 fresh training samples per epoch from
seed 1000, 128 fixed validation trajectories from seed 2000, test trajectories
from seed 3000 and normalization statistics from 128 samples of seed 4000.

::: gm_attention.data.shallow_water
