"""Evaluate the benchmark's fitted models: one-step and five-step, every
seed, native-grid and shared spectral losses.

For each arm (GM-Attention depth-1, SFNO, S2 Transformer) every seed's
``checkpoints/best.pt`` is loaded on its native grid and scored on the same
held-out trajectories (the test split by default) through
:mod:`experiments.shallow_water_paper.compare_spectral`: the grid's own
pooled errors, the band-limited errors below the solver's full band (degree
42 inclusive for the 128x256 solver) with their split by degree, and a
five-step autoregressive rollout (2.5 h at the 30-minute lead) scored at
every step against the solver's own continuation of each trajectory. Seeds
are aggregated per arm as mean and standard deviation.

``--zero-mean-update ARM`` adds a twin of an arm whose predictions keep the
input's global mean per channel, which the solver conserves exactly: the
twin's band-limited one-step error loses its degree-0 term and, up to the
band fit's leakage, nothing else.

Usage, from the repository root::

    uv run python SWE_128x256_Benchmark/evaluate.py                  # outputs/SWE_128x256_Benchmark
    uv run python SWE_128x256_Benchmark/evaluate.py --root DIR \\
        --arms gm_attention_depth1=RUN,sfno=RUN,s2_transformer=RUN
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the repository root

from experiments.shallow_water_paper.benchmark_evaluation import (
    display_name as _display_name,
)
from experiments.shallow_water_paper.benchmark_evaluation import (
    main as _main,
)
from experiments.shallow_water_paper.benchmark_evaluation import (
    report as _report,
)

ARMS = {
    "gm_attention_depth1": "GM-Attention, depth-1",
    "sfno": "SFNO, released config",
    "s2_transformer": "S2 Transformer (local), released config",
}


def display_name(arm: str) -> str:
    return _display_name(arm, ARMS)


def report(summary: dict[str, dict], bandwidth: int, coverage: float, split: str, num_examples: int) -> None:
    _report(summary, bandwidth, coverage, split, num_examples, ARMS)


def main() -> None:
    _main(__doc__, names=ARMS, default_root="outputs/SWE_128x256_Benchmark",
          default_output="outputs/SWE_128x256_Benchmark/evaluation.json")


if __name__ == "__main__":
    main()
