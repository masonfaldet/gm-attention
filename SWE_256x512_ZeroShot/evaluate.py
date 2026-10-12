"""Evaluate the zero-shot study: the single-level L5 arm on its own grid and
on four grids it was never trained on, beside SFNO and the S2 Transformer on
their own grids, one-step and five-step, on the test split.

The L5 arm's weights are grid-independent; for every grid in ``--grids`` its
geometry (attention neighbourhoods, tangent frames, quadrature) is rebuilt
once for that grid's nodes and shared by the seeds
(:class:`compare_spectral.GridContext`), the same 256x512 solver states are
synthesized on those nodes once, and each seed is loaded, scored and rolled
out there in turn before the next grid is built. SFNO is scored on the
256x512 grid it reads and writes, the S2 Transformer on its 128x256
Legendre-Gauss view, the same points and quadrature as the study's
Legendre-Gauss transfer row. Every arm sees the same held-out trajectories;
the band-limited errors below the solver's degree 85 put all grids on one
basis. The L4 icosphere has fewer points (2,562) than the degree-85 band has
coefficients (7,396), so its row carries native-grid errors only.

The attention geometry of the 256x512 Legendre-Gauss grid does not fit on a
GPU: ``--gm-device cpu`` keeps the transferred model and its geometry on the
host (the references untouched) and ``--append`` merges the result into an
existing JSON. ``--projector-device cpu`` keeps every grid's float64
harmonic basis off the GPU.

Usage, from the repository root::

    uv run python SWE_256x512_ZeroShot/evaluate.py --projector-device cpu \\
        --grids icosphere:4,icosphere:5,icosphere:6,legendre-gauss:128x256
    uv run python SWE_256x512_ZeroShot/evaluate.py --projector-device cpu --gm-device cpu \\
        --grids legendre-gauss:256x512 --append

The study splits over GPUs by grid and seed: ``--grids icosphere:6 --seeds 1
--output parts/l6_seed1.json`` scores one (grid, seed) pair, ``--skip-gm``
scores the references alone, and ``--merge parts/*.json --output
evaluation.json`` combines the parts after checking they share the protocol
and the trajectories (``run.sh evaluate-parallel`` and ``run.sh merge``).
Every invocation rewrites its JSON after each arm, so a killed run keeps
what it scored.

The transferred rows are named for the run they came from (``--arms
gm_l5_100=RUN`` names them ``RUN_icosphere6``), so several trainings of the
arm live in one study and one table; ``--relabel OLD=NEW`` renames a part's
rows on merge.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the repository root

from experiments.shallow_water.grid_transfer import GridSpec
from experiments.shallow_water_paper.transfer import _config_of_run
from experiments.shallow_water_paper.benchmark_evaluation import (
    discover_checkpoints,
    required_seeds,
    selected_seeds,
)
from experiments.shallow_water_paper.benchmark_evaluation import (
    main as _main,
)
from experiments.shallow_water_paper.compare_spectral import (
    ArmSource,
    GridContext,
    LazyArm,
)
from experiments.shallow_water_paper.train import SINGLE_LEVEL_KIND

GM = "gm_l5_100"
REFERENCES = {
    "sfno": "SFNO",
    "s2_transformer_sl": "S2 Transformer (local), single level, on the LG 128x256 view",
}
EXPECTED_KINDS = {GM: SINGLE_LEVEL_KIND, "sfno": "sfno", "s2_transformer_sl": "attention_on_sphere"}
GRIDS = ("icosphere:4", "icosphere:5", "icosphere:6", "legendre-gauss:128x256", "legendre-gauss:256x512")


def arm_key(grid: str, run: str = GM) -> str:
    """``icosphere:6`` -> ``gm_l5_100_icosphere6``, ``legendre-gauss:128x256`` ->
    ``gm_l5_100_lg128x256``. ``run`` is the directory the checkpoints came from
    (``--arms gm_l5_100=RUN``), so two trainings of the arm meet in one study
    under keys of their own."""
    return f"{run}_" + grid.replace("legendre-gauss", "lg").replace(":", "")


def names(grids: tuple[str, ...] = GRIDS, run: str = GM) -> dict[str, str]:
    return {arm_key(g, run): f"GM-Attention (L5-trained) on {g}" for g in grids} | REFERENCES


def check_manifest(checkpoints: dict[str, list[tuple[int, Path]]]) -> None:
    """Every discovered checkpoint must be the model kind its arm name claims."""
    for arm, seeds in checkpoints.items():
        for seed, path in seeds:
            kind = str(_config_of_run(path).model.get("kind", "?"))
            if kind != EXPECTED_KINDS[arm]:
                raise SystemExit(f"{arm} seed {seed}: {path} is a {kind!r} run, expected {EXPECTED_KINDS[arm]!r}")


def build_arms(args: argparse.Namespace, device: torch.device) -> list[ArmSource]:
    directories = dict(item.split("=", 1) for item in args.arms.split(",")) if args.arms else {a: a for a in (GM, *REFERENCES)}
    checkpoints = discover_checkpoints(Path(args.root), directories, required_seeds(args))
    check_manifest(checkpoints)
    gm_device = torch.device(args.gm_device) if args.gm_device else device
    arms: list[ArmSource] = []
    gm_seeds = [] if args.skip_gm else selected_seeds(args, checkpoints.get(GM, []))
    gm_run = directories.get(GM, GM)  # the run directory the GM rows are named for
    for grid in args.grids.split(","):
        if not gm_seeds:
            break
        context = GridContext(GridSpec.parse(grid), _config_of_run(gm_seeds[0][1]), gm_device)
        for seed, path in gm_seeds:
            arms.append(LazyArm(f"{arm_key(grid, gm_run)}/seed{seed}", path, args.chunk_size, context))
    for arm in REFERENCES:
        for seed, path in selected_seeds(args, checkpoints.get(arm, [])):
            arms.append(LazyArm(f"{arm}/seed{seed}", path, args.chunk_size))
    return arms


def extra_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--grids", default=",".join(GRIDS),
                        help="comma-separated grids the L5 arm is evaluated on (icosphere:<level> or "
                             "<equiangular|legendre-gauss|lobatto>:<nlat>x<nlon>)")
    parser.add_argument("--skip-gm", action="store_true", help="evaluate the reference arms only (a parallel part)")
    parser.add_argument("--gm-device", default=None,
                        help="where the transferred GM arm and its geometry live (default: --device); 'cpu' for "
                             "a grid whose attention geometry exceeds the GPU")


def main() -> None:
    _main(__doc__, names=names(), default_root="outputs/SWE_256x512_ZeroShot",
          default_output="outputs/SWE_256x512_ZeroShot/evaluation.json",
          build_arms=build_arms, extra_arguments=extra_arguments)


if __name__ == "__main__":
    main()
