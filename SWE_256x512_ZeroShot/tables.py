"""The zero-shot study's LaTeX table from ``evaluate.py``'s JSON.

One booktabs table at the text width (``booktabs`` and ``amsmath`` only):
the L5-trained GM-Attention arm on each grid as one group of rows, then each
reference model on its own grid; per row the grid's point count, then two
column groups, one step and the five-step rollout, each with the $L_2^2$
objective on the native grid and after projection onto the solver's band.
The paper's $L_1$, which the evaluation also records, is not tabulated: the
absolute value of a fitted field integrated under each grid's own quadrature
is a grid-dependent estimate. Cells are mean $\\pm$ seed sd; bold marks the
best mean per column. Group titles, native marks and the benchmark
description come from the runs' recorded configs.

Usage::

    uv run python SWE_256x512_ZeroShot/tables.py [--input evaluation.json] [--output table.tex]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the repository root

from experiments.shallow_water_paper.latex_tables import (
    Pair,
    check,
    inline_cell,
    labels,
    winners,
)

GM_PREFIX = "gm_l5_"
GROUPS = {
    GM_PREFIX: "GM-Attention, trained on the {training}",
    # The ablation study's depth-1 arm on its own grid: the same model with a coarser latent under its attention blocks.
    "gm_attention_full": "GM-Attention, depth-1 on a level-4 latent",
    "sfno": "SFNO",
    "s2_transformer_sl": r"S$^2$ Transformer (local), single level",
}
#: Arms an evaluation JSON may hold that the table leaves out.
DROPPED: tuple[str, ...] = ()
ZERO_MEAN_SUFFIX = "_zero_mean"
MSE_SCALE, MSE_DIGITS = 1e3, 2
TABCOLSEP = "3pt"


def grid_name(grid: str, native: bool = False) -> str:
    family, _, size = grid.partition(":")
    if family == "icosphere":
        text = f"L{size} icosphere"
    else:
        nlat, _, nlon = size.partition("x")
        family_text = {"legendre-gauss": "Legendre--Gauss", "equiangular": "equiangular", "lobatto": "Lobatto"}.get(family, family)
        text = rf"{family_text} ${nlat}\times{nlon}$"
    return text + (" (native)" if native else "")


def gm_run(summary: dict, arm: str) -> str:
    """The training run a transferred row came from. ``evaluate.arm_key``
    names it ``<run>_<grid slug>``, so the grid the row records says where
    the run's name ends: ``gm_l5_100_icosphere6`` on ``icosphere:6`` is the
    ``gm_l5_100`` run."""
    slug = summary[arm]["grid"].replace("legendre-gauss", "lg").replace(":", "")
    return arm[: -len(slug) - 1] if arm.endswith("_" + slug) else arm


def gm_epochs(summary: dict, arms: list[str]) -> int | None:
    """The epoch of the checkpoints a GM run's rows were scored from, one
    more than the recorded index; ``None`` if its rows disagree or say
    nothing. The runs of this study have no early stopping, so this is the
    training length."""
    epochs = {c["epoch"] for a in arms for c in summary[a]["checkpoints"] if c.get("epoch") is not None}
    return epochs.pop() + 1 if len(epochs) == 1 else None


def ordered_arms(summary: dict, drop: tuple[str, ...] = ()) -> list[str]:
    """Arms in display order: zero-mean twins, the study's dropped arms and
    any named in ``drop`` left out. ``drop`` names a row (``gm_l5_icosphere4``)
    or a whole training of the arm (``gm_l5``), which takes its grids with it."""
    arms = [a for a in summary if not a.endswith(ZERO_MEAN_SUFFIX) and a not in DROPPED and a not in drop
            and gm_run(summary, a) not in drop]
    gm = [a for a in arms if a.startswith(GM_PREFIX)]
    refs = [a for a in GROUPS if a != GM_PREFIX and a in arms]
    rest = [a for a in arms if a not in gm and a not in refs]
    return gm + refs + rest


def groups(summary: dict, arms: list[str] | None = None) -> list[tuple[str, list[str]]]:
    """``(group title, arms)`` in display order: one group per GM run, each
    holding that run's grids, then the references."""
    arms = ordered_arms(summary) if arms is None else arms
    out: list[tuple[str, list[str]]] = []
    gm = [a for a in arms if a.startswith(GM_PREFIX)]
    runs = list(dict.fromkeys(gm_run(summary, a) for a in gm))
    for run in runs:
        members = [a for a in gm if gm_run(summary, a) == run]
        title = GROUPS[GM_PREFIX].format(training=grid_name(summary[members[0]]["training_grid"]))
        epochs = gm_epochs(summary, members)
        detail = f"{summary[members[0]]['parameters']:,} parameters"
        out.append((title + f" ({f'epoch {epochs}, ' if epochs is not None else ''}{detail})", members))
    for a in arms:
        if a not in gm:
            out.append((GROUPS.get(a, a.replace("_", r"\_")) + f" ({summary[a]['parameters']:,} parameters)", [a]))
    return out


def row_label(summary: dict, arm: str) -> str:
    native = summary[arm]["grid"] == summary[arm]["training_grid"]
    return grid_name(summary[arm]["grid"], native)


def cells(summary: dict, arms: list[str], pick: Callable[[dict], dict], bandwidth: int) -> dict[str, list[str]]:
    """Per arm, the two cells of one column group: native and band-limited $L_2^2$."""
    keys = [("paper_squared_l2", MSE_SCALE, MSE_DIGITS), (f"paper_squared_l2_L{bandwidth}", MSE_SCALE, MSE_DIGITS)]

    def picker(key: str) -> Callable[[dict], Pair]:
        return lambda s: pick(s)[key]

    bold = {key: winners(summary, [a for a in arms if key in pick(summary[a])], picker(key)) for key, _, _ in keys}
    return {a: [inline_cell(pick(summary[a]).get(key), scale, digits, a in bold[key], seeds=int(summary[a]["seeds"]))
                for key, scale, digits in keys] for a in arms}


def zero_shot_table(data: dict, drop: tuple[str, ...] = ()) -> str:
    S, B = data["summary"], data["bandwidth"]
    arms = ordered_arms(S, drop)
    steps = S[arms[0]]["rollout"]
    lead_hours = float(data.get("lead_time_seconds", 3600.0)) / 3600.0
    column_groups = [(rf"One step ({lead_hours:g}\,h)", cells(S, arms, lambda s: s["metrics"], B))]
    if steps:
        last = steps[-1]
        column_groups.append((rf"{last['step']} steps ({last['hours']:g}\,h)", cells(S, arms, lambda s: s["rollout"][-1]["rollout"], B)))
    n = len(column_groups)
    width = 2 + 2 * n
    rows: list[str] = []
    for title, members in groups(S, arms):
        rows.append(rf"\multicolumn{{{width}}}{{@{{}}l}}{{\emph{{{title}}}}} \\")
        for a in members:
            rows.append(" & ".join([row_label(S, a), f"{S[a]['n_nodes']:,}",
                                    *[c for _, g in column_groups for c in g[a]]]) + r" \\")
    header = [
        "& & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{title}, $10^{{3}}\times L_2^2$}}" for title, _ in column_groups) + r" \\",
        " ".join(rf"\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(n)),
        "Grid & " + labels("points", *[label for _ in column_groups for label in ("native", rf"$\ell \le {B}$")]) + r" \\",
    ]
    seeds = sorted({int(s["seeds"]) for s in S.values()})
    seed_phrase = f"{seeds[0]} seeds" if len(seeds) == 1 else "the seeds each row had (a single seed is marked)"
    nlat, nlon = data.get("solver", (256, 512))
    gm_training = grid_name(S[arms[0]]["training_grid"]) if arms[0].startswith(GM_PREFIX) else "its training grid"
    gm_titles = [title for title, members in groups(S, arms) if members[0].startswith(GM_PREFIX)]
    several = len(gm_titles) > 1
    epoch_note = (r" \emph{epoch}: the epoch a heading's checkpoints were saved at, the training's last "
                  r"where it runs to its schedule's end." if any("epoch " in t for t in gm_titles) else "")
    caption = (
        rf"Zero-shot grid transfer on the ${nlat}\times{nlon}$ shallow-water benchmark ({lead_hours:g}\,h lead), "
        rf"{data['num_examples']} {data['split']} trajectories; mean $\pm$ sd over {seed_phrase}; best mean in bold per "
        r"column. " + ("Each transferred group is one set of checkpoints" if several else
                       "The transferred rows are one set of checkpoints") + ", trained on the "
        rf"{gm_training} and applied unchanged on each grid: its weights are grid-independent, only its "
        r"neighbourhoods, frames and quadrature are rebuilt for the grid's points, and the same solver states are "
        r"synthesized there. Every other row is a model on its own grid. "
        + ("The transferred groups are trainings of that one arm under one recipe, differing in length "
           "alone. " if several else "")
        + r"\emph{points}: the grid's size." + epoch_note + r" $L_2^2$ is the "
        r"training objective, the quadrature-weighted squared error on the sphere averaged over the three z-scored "
        rf"channels and pooled over trajectories. \emph{{native}}: under each grid's own quadrature; "
        rf"$\ell \le {B}$: the continuum norm of a weighted least-squares fit to degrees $0,\ldots,{B}$, the solver's band "
        r"and the common basis across grids. The fit remains subject to each grid's sampling and aliasing, and the "
        r"native columns are not comparable across grids of different resolution."
    )
    # By grid, not by row: two runs scored on one coarse grid are two rows.
    coarse = list(dict.fromkeys(row_label(S, a) for a in arms if f"paper_squared_l2_L{B}" not in S[a]["metrics"]))
    if coarse:
        caption += (
            " " + " and ".join(coarse) + rf" ha{'s' if len(coarse) == 1 else 've'} fewer points than "
            rf"the band has coefficients ($({B}+1)^2 = {(B + 1) ** 2:,}$), so no band-limited error exists there (--)."
        )
    if steps:
        last = steps[-1]
        persistence = f"{steps[0]['persistence']['paper_squared_l2'][0] * MSE_SCALE:.0f} and {last['persistence']['paper_squared_l2'][0] * MSE_SCALE:.0f}"
        caption += (
            rf" The {last['step']}-step group is the autoregressive rollout, each step fed the model's own previous "
            r"prediction and scored against the solver's own continuation of the trajectory (the training solver map "
            r"repeated, with Adams--Bashforth history restarted at each lead time, as in the released script's "
            rf"rollout evaluation). Persistence (predicting no change) scores {persistence} in $10^{{3}}\times L_2^2$ "
            r"units at the two horizons."
        )
    spec = "lr" + "r" * (2 * n)
    return "\n".join([
        r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", r"\label{tab:swe256-zero-shot}",
        r"\small", rf"\setlength{{\tabcolsep}}{{{TABCOLSEP}}}",
        rf"\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}{spec}@{{}}}}",
        r"\toprule", *header, r"\midrule", *rows, r"\bottomrule",
        r"\end{tabular*}", r"\end{table}",
    ])


def render(data: dict, drop: tuple[str, ...] = ()) -> str:
    head = "\n".join([
        "% Generated by SWE_256x512_ZeroShot/tables.py from evaluate.py's JSON.",
        f"% split={data['split']} trajectories={data['num_examples']} bandwidth={data['bandwidth']} steps={data['steps']}",
        "% Requires: \\usepackage{booktabs} \\usepackage{amsmath}",
    ])
    return head + "\n\n" + zero_shot_table(data, drop) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="outputs/SWE_256x512_ZeroShot/evaluation.json")
    parser.add_argument("--output", default=None, help="default: <input stem>_table.tex beside the input")
    parser.add_argument("--drop", action="append", default=[], metavar="ARM",
                        help="leave this arm out of the table, or this training of the arm with every grid it was "
                             "scored on (repeatable), e.g. gm_l5_icosphere4 or gm_l5")
    args = parser.parse_args()
    source = Path(args.input)
    tex = render(json.loads(source.read_text()), tuple(args.drop))
    check(tex)
    out = Path(args.output) if args.output else source.with_name(source.stem + "_table.tex")
    out.write_text(tex)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
