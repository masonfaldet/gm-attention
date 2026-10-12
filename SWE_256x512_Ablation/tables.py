"""The ablation's LaTeX table from ``evaluate.py``'s JSON.

One booktabs table at the text width (``booktabs`` and ``amsmath`` only):
the four depth-1 arms as rows, each named by its mixer ($SO(S^2)$-MHSA or
$SO(S^2)$-Conv) and its hidden SO(2) field type with multiplicities as
superscripts ($\\rho_0^{16} \\oplus \\rho_1^{16} \\oplus \\rho_2^{16}$), and
two column groups, the one-step error and the error after the five-step
rollout, each holding the $L_2^2$ objective and the per-channel $L_2$. Every
arm lives on the level-5 icosphere, so the errors are the grid's own
quadrature-weighted norms without a common-band projection. Cells stack the
seed sd beneath the mean; bold marks the best mean per column.

Usage::

    uv run python SWE_256x512_Ablation/tables.py [--input evaluation.json] [--output table.tex]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the repository root

from experiments.shallow_water_paper.latex_tables import (
    CHANNEL_SYMBOLS,
    CHANNELS,
    Pair,
    centered,
    check,
    inline_cell,
    stacked_cell,
    winners,
)

#: Display order (the full arm first, then its twins) and each arm's mixer and hidden SO(2) type when the JSON lacks them.
ARMS = {
    "gm_attention_full": ("attention", {0: 16, 1: 16, 2: 16}),
    "gm_convolution_full": ("convolution", {0: 16, 1: 16, 2: 16}),
    "gm_attention_rho01": ("attention", {0: 26, 1: 26}),
    "gm_attention_rho0": ("attention", {0: 72}),
}
NAMES = dict.fromkeys(ARMS)  # display order
MIXER_TEX = {"attention": r"$SO(S^2)$-MHSA", "convolution": r"$SO(S^2)$-Conv"}
ZERO_MEAN_SUFFIX = "_zero_mean"
MSE_SCALE, MSE_DIGITS = 1e3, 2
RMSE_SCALE, RMSE_DIGITS = 1e2, 2
TABCOLSEP = "3pt"


def ordered_arms(summary: dict) -> list[str]:
    arms = [a for a in summary if not a.endswith(ZERO_MEAN_SUFFIX)]
    return [a for a in NAMES if a in arms] + [a for a in arms if a not in NAMES]


def field_type_tex(hidden: dict[int, int]) -> str:
    """``{0: 16, 1: 16, 2: 16}`` -> ``$\\rho_0^{16} \\oplus \\rho_1^{16} \\oplus \\rho_2^{16}$``."""
    return "$" + r" \oplus ".join(rf"\rho_{{{k}}}^{{{hidden[k]}}}" for k in sorted(hidden)) + "$"


def architecture(summary: dict, arm: str) -> tuple[str, str]:
    """``(mixer, hidden type)`` as LaTeX, from the JSON's record of the run
    when present, else from :data:`ARMS`."""
    entry = summary[arm]
    mixer = entry.get("mixer")
    hidden = entry.get("hidden_type")
    if mixer is None or hidden is None:
        if arm not in ARMS:
            return arm.replace("_", r"\_"), "--"
        mixer, hidden = ARMS[arm]
    hidden = {int(k): int(v) for k, v in hidden.items()}
    return MIXER_TEX.get(mixer, mixer), field_type_tex(hidden)


def n_points(data: dict, arm: str) -> int:
    return int(next(r["n_nodes"] for r in data["per_seed"] if r["label"].startswith(arm + "/")))


def seed_phrase(summary: dict) -> str:
    counts = sorted({int(s["seeds"]) for s in summary.values()})
    return f"{counts[0]} seeds" if len(counts) == 1 else "the available seeds per arm (counts differ)"


def group(summary: dict, arms: list[str], pick: Callable[[dict], dict], inline: bool = False) -> list[list[str]]:
    """Per arm, the four cells of one column group (``pick`` gives the
    metrics dict: one-step ``metrics`` or a rollout step's ``rollout``);
    stacked mean-over-sd cells, or ``mean $\\pm$ sd`` on one line."""
    keys = ["paper_squared_l2", *[f"{c}_rmse" for c in CHANNELS]]

    def picker(key: str) -> Callable[[dict], Pair]:
        return lambda s: pick(s)[key]

    bold = {key: winners(summary, arms, picker(key)) for key in keys}
    cell = inline_cell if inline else stacked_cell
    return [[
        cell(pick(summary[a])[key], MSE_SCALE if key == "paper_squared_l2" else RMSE_SCALE,
             MSE_DIGITS if key == "paper_squared_l2" else RMSE_DIGITS, a in bold[key],
             seeds=int(summary[a]["seeds"]))
        for key in keys
    ] for a in arms]


def ablation_table(data: dict, *, one_step: bool = False, inline: bool = False) -> str:
    """``one_step`` drops the rollout group; ``inline`` writes ``mean $\\pm$
    sd`` on one line."""
    S = data["summary"]
    arms = ordered_arms(S)
    steps = S[arms[0]]["rollout"]
    lead_hours = float(data.get("lead_time_seconds", 3600.0)) / 3600.0
    one = group(S, arms, lambda s: s["metrics"], inline)
    groups = [(rf"One step ({lead_hours:g}\,h)", one)]
    if steps and not one_step:
        last = steps[-1]
        groups.append((rf"{last['step']} steps ({last['hours']:g}\,h)", group(S, arms, lambda s: s["rollout"][-1]["rollout"], inline)))
    n = len(groups)
    rows = [
        " & ".join([*architecture(S, a), f"{S[a]['parameters']:,}", *[c for g in groups for c in g[1][i]]]) + r" \\"
        for i, a in enumerate(arms)
    ]
    label = centered if inline else (lambda text: text)
    header = [
        "& & & " + " & ".join(rf"\multicolumn{{4}}{{c}}{{{title}}}" for title, _ in groups) + r" \\",
        " ".join(rf"\cmidrule(lr){{{4 + 4 * i}-{7 + 4 * i}}}" for i in range(n)),
        "& & & " + " & ".join(r"$10^{3}\times L_2^2$ & \multicolumn{3}{c}{$10^{2}\times L_2$ per channel}" for _ in groups) + r" \\",
        " ".join(rf"\cmidrule(lr){{{5 + 4 * i}-{7 + 4 * i}}}" for i in range(n)),
        "Mixer & Hidden type & " + label("Params") + " & "
        + " & ".join("& " + " & ".join(label(CHANNEL_SYMBOLS[c]) for c in CHANNELS) for _ in groups) + r" \\",
    ]
    nlat, nlon = data.get("solver", (256, 512))
    seeds = sorted({int(s["seeds"]) for s in S.values()})
    seed_text = f"{seeds[0]} seeds" if len(seeds) == 1 else "the seeds each row had (a single seed is marked)"
    caption = (
        rf"Depth-1 ablation on the ${nlat}\times{nlon}$ shallow-water benchmark with {data['num_examples']} "
        rf"{data['split']} trajectories. "
        + (rf"Metrics reported as mean $\pm$ sd over {seed_text}. " if inline
           else rf"Metrics reported as mean over {seed_text} with the seed sd beneath. ")
        + r"Parameters are real scalars."
    )
    if steps and not one_step:
        last = steps[-1]
        persistence = f"{steps[0]['persistence']['paper_squared_l2'][0] * MSE_SCALE:.0f} and {last['persistence']['paper_squared_l2'][0] * MSE_SCALE:.0f}"
        caption += rf" Persistence scores {persistence} in $L_2^2$ units at the two horizons."
    elif steps:
        caption += rf" Persistence scores {steps[0]['persistence']['paper_squared_l2'][0] * MSE_SCALE:.0f} in $L_2^2$ units."
    spec = "llr" + ("r" if inline else "c") * (4 * n)
    return "\n".join([
        r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", r"\label{tab:swe256-ablation}",
        r"\small", rf"\setlength{{\tabcolsep}}{{{TABCOLSEP}}}",
        rf"\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}{spec}@{{}}}}",
        r"\toprule", *header, r"\midrule", *rows, r"\bottomrule",
        r"\end{tabular*}", r"\end{table}",
    ])


def render(data: dict, *, one_step: bool = False, inline: bool = False) -> str:
    head = "\n".join([
        "% Generated by SWE_256x512_Ablation/tables.py from evaluate.py's JSON.",
        f"% split={data['split']} trajectories={data['num_examples']} steps={data['steps']}",
        "% Requires: \\usepackage{booktabs} \\usepackage{amsmath}",
    ])
    return head + "\n\n" + ablation_table(data, one_step=one_step, inline=inline) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="outputs/SWE_256x512_Ablation/evaluation.json")
    parser.add_argument("--output", default=None, help="default: <input stem>_table.tex beside the input")
    parser.add_argument("--one-step", action="store_true", help="the one-step group only, without the rollout")
    parser.add_argument("--inline", action="store_true", help="mean and sd on one line instead of stacked")
    args = parser.parse_args()
    source = Path(args.input)
    tex = render(json.loads(source.read_text()), one_step=args.one_step, inline=args.inline)
    check(tex)
    out = Path(args.output) if args.output else source.with_name(source.stem + "_table.tex")
    out.write_text(tex)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
