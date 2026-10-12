"""LaTeX tables from ``evaluate.py``'s JSON.

Three booktabs tables (``booktabs`` and ``amsmath`` only): the benchmark
table, one step (parameters, $L_2^2$ native and band-limited, per-channel
$L_2$) and, per lead time, the autoregressive rollout beside the
teacher-forced one-step error on the native grid; the reference paper's
$L_1$/$L_2$ conventions with band-limited per-channel $L_2$; and the split
of the error by degree band with the fit residual. Errors are
quadrature-weighted norms on the sphere: $L_2^2$ is the objective (squared
norm averaged over the three z-scored channels, pooled over trajectories),
$L_2$ is per channel, pooled before the root. Cells are seed mean $\\pm$ sd,
best mean in bold; $L_2^2$ is printed as ``10^3 x`` and $L_2$ as ``10^2 x``.
Zero-mean twins in the JSON are left out.

Usage::

    uv run python SWE_128x256_Benchmark/tables.py [--input evaluation.json] [--output tables.tex]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the repository root

from experiments.shallow_water_paper.latex_tables import (  # noqa: F401  (re-exported for tests)
    CHANNEL_SYMBOLS,
    CHANNELS,
    Pair,
    centered,
    check,
    inline_cell,
    labels,
    winners,
)

#: Display order, baselines first; arms the JSON has beyond these are appended under their own key.
NAMES = {
    "s2_transformer": r"S$^2$ Transformer",
    "sfno": "SFNO",
    "gm_attention_depth1": "GM-Attention (ours)",
}
#: The S2 Transformer paper's Table 3 values for this configuration: absolute quadrature-weighted L1 and L2 of the z-scored channels.
REPORTED = {"s2_transformer": {"paper_l1": 0.063, "paper_l2": 0.079, "epochs": 50}}
BAND_EDGES = (0, 1, 4, 24)
MSE_SCALE, MSE_DIGITS = 1e3, 3
RMSE_SCALE, RMSE_DIGITS = 1e2, 2
TABCOLSEP = "4pt"

ZERO_MEAN_SUFFIX = "_zero_mean"  # suffix of the zero-mean twins evaluate.py may record


def ordered_arms(summary: dict) -> list[str]:
    """Baselines first, GM-Attention last; zero-mean twins are left out."""
    arms = [a for a in summary if not a.endswith(ZERO_MEAN_SUFFIX)]
    return [a for a in NAMES if a in arms] + [a for a in arms if a not in NAMES]


def name(arm: str) -> str:
    return NAMES.get(arm, arm.replace("_", r"\_"))


cell = inline_cell


def metric(key: str) -> Callable[[dict], Pair]:
    return lambda s: s["metrics"][key]


def step_metric(k: int, kind: str, key: str) -> Callable[[dict], Pair]:
    return lambda s: s["rollout"][k][kind][key]


def degree_bands(bandwidth: int) -> list[tuple[int, int]]:
    edges = sorted({min(e, bandwidth + 1) for e in (*BAND_EDGES, bandwidth + 1)})
    return list(zip(edges, edges[1:]))


def band_label(lo: int, hi: int) -> str:
    return f"{lo}--{hi - 1}" if hi - 1 > lo else f"{lo}"


def n_points(data: dict, arm: str) -> int:
    return int(next(r["n_nodes"] for r in data["per_seed"] if r["label"].startswith(arm + "/")))


BOLD_RULE = r"best mean in bold"


def seed_phrase(summary: dict) -> str:
    counts = sorted({int(s["seeds"]) for s in summary.values()})
    return f"{counts[0]} seeds" if len(counts) == 1 else "the available seeds per arm (counts differ)"


def grid_phrase(data: dict, arms: list[str]) -> list[str]:
    """One clause per distinct grid, naming the arms on it: ``S$^2$
    Transformer and SFNO on the 32,768-point $128\\times256$ equiangular
    grid``, ``GM-Attention (ours) on the 10,242-vertex level-5 icosphere``."""
    by_grid: dict[str, list[str]] = {}
    for a in arms:
        by_grid.setdefault(data["summary"][a]["grid"], []).append(a)
    clauses = []
    for grid, members in by_grid.items():
        who = " and ".join(name(a) for a in members)
        points = n_points(data, members[0])
        kind, _, size = grid.partition(":")
        if kind == "icosphere":
            where = f"the {points:,}-vertex level-{size} icosphere"
        elif "x" in size:
            nlat, nlon = size.split("x")
            where = rf"the {points:,}-point ${nlat}\times{nlon}$ {kind.replace('-', ' ')} grid"
        else:
            where = f"the {points:,}-point {kind.replace('_', ' ')} grid"
        clauses.append(f"{who} on {where}")
    return clauses


def table(label: str, caption: str, spec: str, header: list[str], rows: list[str]) -> str:
    return "\n".join([
        r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        r"\small", rf"\setlength{{\tabcolsep}}{{{TABCOLSEP}}}",
        rf"\begin{{tabular}}{{{spec}}}", r"\toprule", *header, r"\midrule", *rows, r"\bottomrule",
        r"\end{tabular}", r"\end{table}",
    ])


def panel(title: str, spec: str, header: list[str], rows: list[str]) -> str:
    """One booktabs ``tabular*`` at the text width, titled in its first row."""
    return "\n".join([
        rf"\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}{spec}@{{}}}}",
        r"\toprule",
        rf"\multicolumn{{{len(spec)}}}{{@{{}}l}}{{{title}}} \\",
        r"\addlinespace[2pt]",
        *header, r"\midrule", *rows, r"\bottomrule",
        r"\end{tabular*}",
    ])


def one_step_panel(data: dict, arms: list[str]) -> str:
    S, B = data["summary"], data["bandwidth"]
    band = f"paper_squared_l2_L{B}"
    bold = {
        "native": winners(S, arms, metric("paper_squared_l2")),
        "band": winners(S, arms, metric(band)),
        **{c: winners(S, arms, metric(f"{c}_rmse")) for c in CHANNELS},
    }
    rows = [" & ".join([
        name(a), f"{S[a]['parameters']:,}",
        cell(S[a]["metrics"]["paper_squared_l2"], MSE_SCALE, MSE_DIGITS, a in bold["native"]),
        cell(S[a]["metrics"][band], MSE_SCALE, MSE_DIGITS, a in bold["band"]),
        *[cell(S[a]["metrics"][f"{c}_rmse"], RMSE_SCALE, RMSE_DIGITS, a in bold[c]) for c in CHANNELS],
    ]) + r" \\" for a in arms]
    header = [
        r"& & \multicolumn{2}{c}{$10^{3}\times L_2^2$} & \multicolumn{3}{c}{$10^{2}\times L_2$ per channel, native} \\",
        r"\cmidrule(lr){3-4} \cmidrule(lr){5-7}",
        "Model & " + labels("Params", "native", rf"$\ell \le {B}$", *[CHANNEL_SYMBOLS[c] for c in CHANNELS]) + r" \\",
    ]
    return panel(rf"\textbf{{(a)}} One step ({data['lead_minutes']:g}\,min)", "lrrrrrr", header, rows)


def lead_time_panel(data: dict, arms: list[str]) -> str:
    """Lead times as rows; per model, the native-grid $L_2^2$ of the
    rollout and of the teacher-forced one step. Bold per lead time within
    each of the two kinds."""
    S = data["summary"]
    kinds = (("rollout", "rollout"), ("one_step", "teacher-forced"))
    steps = S[arms[0]]["rollout"]
    rows = []
    for k in range(1, len(steps)):
        bold = {kind: winners(S, arms, step_metric(k, kind, "paper_squared_l2")) for kind, _ in kinds}
        cells = [f"{steps[k]['hours']:.1f}"]
        for a in arms:
            cells += [cell(S[a]["rollout"][k][kind]["paper_squared_l2"], MSE_SCALE, MSE_DIGITS, a in bold[kind]) for kind, _ in kinds]
        rows.append(" & ".join(cells) + r" \\")
    header = [
        "& " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{name(a)}}}" for a in arms) + r" \\",
        " ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(arms))),
        "Lead (h) & " + " & ".join(labels(*[label for _, label in kinds]) for _ in arms) + r" \\",
    ]
    return panel(r"\textbf{(b)} Rollout and teacher-forced one step, $10^{3}\times L_2^2$, native", "l" + "rr" * len(arms), header, rows)


def benchmark_table(data: dict) -> str:
    """Panel (a) one step; (b) the rollout beside the teacher-forced
    one-step error along the trajectory, on the native grid."""
    S, B = data["summary"], data["bandwidth"]
    arms = ordered_arms(S)
    steps = S[arms[0]]["rollout"]
    panels = [one_step_panel(data, arms)]
    if len(steps) >= 2:
        panels.append(lead_time_panel(data, arms))

    grids = "; ".join(grid_phrase(data, arms))
    caption = (
        rf"The $128\times256$ shallow-water benchmark on {data['num_examples']} {data['split']} trajectories; "
        rf"mean $\pm$ sd over {seed_phrase(S)}; {BOLD_RULE}, per column in (a) and per lead time and column in "
        r"(b). $L_2^2$ is the training objective, the quadrature-weighted squared error on the sphere "
        r"averaged over the three z-scored channels and pooled over trajectories; $L_2$ is the per-channel error "
        rf"norm, pooled before the root. \emph{{native}}: each model scored on its own grid ({grids}); "
        rf"$\ell \le {B}$: the exact continuum norm of a weighted least-squares fit to degrees $0,\ldots,{B}$. "
        r"The fit remains subject to native-grid sampling and aliasing; the condition number does not bound "
        rf"aliasing. Target energy coverage is {data['coverage']:.6g}. Parameters are real scalars."
    )
    if len(steps) >= 2:
        persistence = ", ".join(f"{s['persistence']['paper_squared_l2'][0] * MSE_SCALE:.0f}" for s in steps[1:])
        caption += (
            r" (b) is on each model's native grid. \emph{Rollout}: each step is fed the model's own previous "
            r"prediction; \emph{teacher-forced}: each step is fed the solver's state at the previous lead time; "
            r"the first step of both is the native column of (a). Truth repeats the training solver map with "
            r"Adams--Bashforth history restarted at each "
            r"lead time, as the released script's own rollout evaluation does. Persistence (predicting no change) "
            rf"scores {persistence} in the same units at these lead times."
        )
    return "\n".join([
        r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", r"\label{tab:swe128-benchmark}",
        r"\small", rf"\setlength{{\tabcolsep}}{{{TABCOLSEP}}}",
        "\n\\par\\medskip\n".join(panels),
        r"\end{table}",
    ])


def conventions_table(data: dict) -> str:
    S, B = data["summary"], data["bandwidth"]
    arms = ordered_arms(S)
    bold = {
        "l1": winners(S, arms, lambda s: s["paper_l1"]), "l2": winners(S, arms, lambda s: s["paper_l2"]),
        **{c: winners(S, arms, metric(f"{c}_rmse_L{B}")) for c in CHANNELS},
    }
    rows = []
    reported = [a for a in arms if a in REPORTED]
    for a in reported:
        rows.append(" & ".join([
            rf"{name(a)}, reported", f"{REPORTED[a]['paper_l1'] * RMSE_SCALE:.1f}", f"{REPORTED[a]['paper_l2'] * RMSE_SCALE:.1f}",
            *["--" for _ in CHANNELS],
        ]) + r" \\")
    for a in arms:
        rows.append(" & ".join([
            name(a), cell(S[a]["paper_l1"], RMSE_SCALE, RMSE_DIGITS, a in bold["l1"]),
            cell(S[a]["paper_l2"], RMSE_SCALE, RMSE_DIGITS, a in bold["l2"]),
            *[cell(S[a]["metrics"][f"{c}_rmse_L{B}"], RMSE_SCALE, RMSE_DIGITS, a in bold[c]) for c in CHANNELS],
        ]) + r" \\")
    header = [
        rf"& \multicolumn{{2}}{{c}}{{$10^{{2}}\times$ paper convention}} & \multicolumn{{3}}{{c}}{{$10^{{2}}\times L_2$ per channel, $\ell \le {B}$}} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-6}",
        "Model & " + labels("$L_1$", "$L_2$", *[CHANNEL_SYMBOLS[c] for c in CHANNELS]) + r" \\",
    ]
    reported_note = ""
    if reported:
        epochs = ", ".join(f"{REPORTED[a]['epochs']}" for a in reported)
        reported_note = (
            rf" The \emph{{reported}} row gives the paper's Table~3 values as context (3,200 optimizer steps, "
            rf"equivalent to {epochs} epochs here). Our runs train for 100 epochs and differ in solver grid, "
            r"normalization, validation sampling, and Coriolis conditioning."
        )
    caption = (
        r"One-step errors in the conventions of the S$^2$ Transformer paper's Table~3 (quadrature-weighted "
        r"$L_1$ and $L_2$ norms of the z-scored error, averaged over channels per trajectory and then over "
        rf"trajectories, on each model's native grid) and per-channel $L_2$ after projection onto $\ell \le {B}$ "
        rf"(pooled before the root); mean $\pm$ sd over {seed_phrase(S)}; "
        rf"{BOLD_RULE}.{reported_note}"
    )
    return table("tab:swe128-conventions", caption, "lrrrrr", header, rows)


def spectrum_table(data: dict) -> str:
    S, B = data["summary"], data["bandwidth"]
    arms = ordered_arms(S)
    bands = degree_bands(B)
    steps = S[arms[0]]["rollout"]
    last = len(steps) - 1 if steps else None

    def fractions(D: list[float]) -> list[str]:
        total = sum(D) or 1.0
        return [f"{sum(D[lo:hi]) / total:.3f}" for lo, hi in bands]

    residual_key = f"fit_residual_fraction_L{B}"
    residual = all(residual_key in S[a]["metrics"] for a in arms)
    rows = []
    for a in arms:
        cells = [name(a)]
        if residual:
            cells.append(f"{S[a]['metrics'][residual_key][0]:.3f}")
        cells += fractions(S[a]["D"])
        if last is not None:
            cells += fractions(S[a]["rollout"][last]["D"])
        rows.append(" & ".join(cells) + r" \\")
    k = len(bands)
    offset = 3 if residual else 2
    groups = [rf"\multicolumn{{{k}}}{{c}}{{one step, by degree}}"]
    rules = [rf"\cmidrule(lr){{{offset}-{offset + k - 1}}}"]
    if last is not None:
        groups.append(rf"\multicolumn{{{k}}}{{c}}{{{steps[last]['hours']:g}\,h rollout, by degree}}")
        rules.append(rf"\cmidrule(lr){{{offset + k}-{offset + 2 * k - 1}}}")
    bands_row = labels(*[band_label(lo, hi) for lo, hi in bands])
    header = [
        ("& & " if residual else "& ") + " & ".join(groups) + r" \\",
        " ".join(rules),
        "Model & " + (labels("fit residual") + " & " if residual else "") + bands_row
        + (f" & {bands_row}" if last is not None else "") + r" \\",
    ]
    caption = (
        r"Where the fitted error sits in degree. "
        + (
            rf"\emph{{fit residual}}: the share of each model's native-grid $L_2^2$ that no field of degree "
            rf"$\le {B}$ can represent on its own nodes, an exact decomposition under the grid's quadrature "
            r"(seed mean); out-of-band content aliased into the fit is not separable from it. "
            if residual else ""
        )
        + r"The remaining columns split the fitted continuum $L_2^2$ by degree (ratios of seed-mean energies) "
        r"after one step and at the end of the rollout. Degree $0$ is the fitted global-mean error. The solver "
        r"conserves its exact channel means, but native-grid quadrature and aliasing can affect their estimates."
    )
    spec = "l" + ("r" if residual else "") + "r" * k + ("r" * k if last is not None else "")
    return table("tab:swe128-spectrum", caption, spec, header, rows)


def render(data: dict) -> str:
    """The three tables, separated by blank lines, with a header comment."""
    lead_seconds = float(data.get("lead_time_seconds", 0.0))
    if not lead_seconds and data["summary"]:
        steps = next(iter(data["summary"].values()))["rollout"]
        lead_seconds = 3600.0 * steps[0]["hours"] / steps[0]["step"] if steps else 1800.0
    data = {**data, "lead_minutes": lead_seconds / 60.0}
    head = "\n".join([
        "% Generated by SWE_128x256_Benchmark/tables.py from evaluate.py's JSON.",
        f"% split={data['split']} trajectories={data['num_examples']} bandwidth={data['bandwidth']} steps={data['steps']}",
        "% Requires: \\usepackage{booktabs} \\usepackage{amsmath}",
    ])
    tables = [benchmark_table(data), conventions_table(data), spectrum_table(data)]
    return "\n\n".join([head, *[t for t in tables if t]]) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="outputs/SWE_128x256_Benchmark/evaluation.json")
    parser.add_argument("--output", default=None, help="default: <input stem>_tables.tex beside the input")
    args = parser.parse_args()
    source = Path(args.input)
    data = json.loads(source.read_text())
    tex = render(data)
    check(tex)
    out = Path(args.output) if args.output else source.with_name(source.stem + "_tables.tex")
    out.write_text(tex)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
