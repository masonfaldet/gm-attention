"""The solver's own continuation of a held-out trajectory, for scoring autoregressive rollouts against."""

from __future__ import annotations

import torch
from torch import Tensor

from experiments.shallow_water_paper.data import ScalarShallowWaterDataset


def true_trajectory(
    dataset: ScalarShallowWaterDataset, index: int, steps: int, *, return_spectral: bool = False
) -> list[Tensor] | tuple[list[Tensor], list[Tensor]]:
    """The solver's normalized states for hours ``0 .. steps`` of trajectory
    ``index``, in the dataset's own layout: the cached (input, target) pair,
    then repeated applications of the same one-lead-time solver map to the
    latest state.

    Each application restarts the solver's Adams-Bashforth startup scheme at
    the lead-time boundary, as the training target map does; the result is
    the composition of that map, not one uninterrupted
    ``timestep(initial, steps * solver_substeps)`` solve.

    Deterministic given the sample's seed, so arms that share the data seeds
    share the whole trajectory. Point-major ``(n, 3)`` at nodes,
    channel-first ``(3, nlat, nlon)`` on a view, normalized as the dataset
    normalizes; with ``return_spectral`` the ``(3, lmax, mmax)`` spectral
    states are returned as well.
    """
    if steps < 0:
        raise ValueError(f"steps must be nonnegative, got {steps}.")
    source = dataset.source
    with torch.inference_mode():
        spectral = list(source.spectral_pair(index))
        for _ in range(steps - 1):
            spectral.append(source.solver.timestep(spectral[-1], source.solver_substeps))
        spectral = spectral[: steps + 1]
        fields = [dataset._synthesize(state) for state in spectral]
        if dataset.normalization is not None:
            normalize = (
                dataset.normalization.normalize_channel_first
                if dataset.channel_first
                else dataset.normalization.normalize
            )
            fields = [normalize(field) for field in fields]
    fields = [field.clone() for field in fields]
    if return_spectral:
        return fields, [state.clone() for state in spectral]
    return fields
