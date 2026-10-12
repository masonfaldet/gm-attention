"""Evaluate the ablation's fitted models: one-step and five-step, every
seed, on the test split.

Every seed's ``checkpoints/best.pt`` of the four depth-1 arms is loaded on
the level-5 icosphere and scored on the same held-out trajectories through
:mod:`experiments.shallow_water_paper.compare_spectral`: the pooled
native-grid errors, the band-limited errors (every arm shares the grid, so
the table uses the native ones), and a five-step autoregressive rollout (5 h
at the one-hour lead) scored at every step against the solver's own
continuation of each trajectory. Seeds are aggregated per arm as mean and
standard deviation.

Usage, from the repository root::

    uv run python SWE_256x512_Ablation/evaluate.py                    # outputs/SWE_256x512_Ablation
    uv run python SWE_256x512_Ablation/evaluate.py --root DIR --arms \\
        gm_attention_full=RUN,gm_convolution_full=RUN,gm_attention_rho01=RUN,gm_attention_rho0=RUN
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the repository root

from experiments.shallow_water_paper.benchmark_evaluation import (
    main as _main,
)

ARMS = {
    "gm_attention_full": "GM-Attention, rho_0+rho_1+rho_2",
    "gm_convolution_full": "GM-Convolution, rho_0+rho_1+rho_2",
    "gm_attention_rho01": "GM-Attention, rho_0+rho_1",
    "gm_attention_rho0": "GM-Attention, rho_0",
}


def main() -> None:
    _main(__doc__, names=ARMS, default_root="outputs/SWE_256x512_Ablation",
          default_output="outputs/SWE_256x512_Ablation/evaluation.json")


if __name__ == "__main__":
    main()
