"""Appendix tables of where each model's band-limited error sits in degree,
after one step and at the end of the rollout, from the evaluation JSONs of
the three studies: one table for the 128x256 configuration and one for
256x512, each model with two rows (one step, last rollout step) giving the
band-limited $L_2^2$ (x10^3) and its fraction in each degree band. Degree
0, the global mean the solver conserves, is its own band; the other bands
follow the solver's truncation. The output needs ``booktabs`` and ``amsmath``.

Usage::

    uv run python -m experiments.shallow_water_paper.degree_band_tables \\
        --easy outputs/SWE_128x256_Benchmark/evaluation.json \\
        --ablation outputs/SWE_256x512_Ablation/evaluation.json \\
        --zeroshot outputs/SWE_256x512_ZeroShot/evaluation.json \\
        --output figures/degree_band_tables.tex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.shallow_water_paper.latex_tables import check, labels

#: Display names per study, in row order; unnamed arms follow under their own key; zero-mean twins are left out.
EASY_NAMES = {
    "s2_transformer": r"S$^2$ Transformer",
    "sfno": "SFNO",
    "gm_attention_depth1": "SOS2-T D1",
}
ABLATION_NAMES = {
    "gm_attention_full": r"D1 $SO(S^2)$-MHSA, $\rho_0^{16} \oplus \rho_1^{16} \oplus \rho_2^{16}$",
    "gm_convolution_full": r"D1 $SO(S^2)$-Conv, $\rho_0^{16} \oplus \rho_1^{16} \oplus \rho_2^{16}$",
    "gm_attention_rho01": r"D1 $SO(S^2)$-MHSA, $\rho_0^{26} \oplus \rho_1^{26}$",
    "gm_attention_rho0": r"D1 $SO(S^2)$-MHSA, $\rho_0^{72}$",
}
ZEROSHOT_NAMES = {
    # Rows of the transferred arm name their training; other trainings of it are in DROPPED.
    "gm_l5_100_icosphere5": "SL (100 epochs) on the L5 icosphere (native)",
    "gm_l5_100_icosphere6": "SL (100 epochs) on the L6 icosphere",
    "gm_l5_100_lg128x256": r"SL (100 epochs) on Legendre--Gauss $128\times256$",
    # The ablation's depth-1 arm, also a row of the transfer table.
    "gm_attention_full": "D1 on the L5 icosphere (native)",
    "sfno": r"SFNO ($256\times512$, native)",
    "s2_transformer_sl": r"S$^2$ Transformer, single level ($128\times256$, native)",
}
#: Arms present in a JSON that the tables leave out; ``--drop`` leaves out more.
DROPPED = ("s2_transformer_zero_mean", "gm_l5_icosphere4", "gm_l5_lg256x512", "gm_l5_100_icosphere4",
           "gm_l5_icosphere5", "gm_l5_icosphere6", "gm_l5_lg128x256")
BAND_EDGES = {42: (0, 1, 4, 12, 24, 43), 85: (0, 1, 4, 24, 48, 86)}
TOTAL_SCALE = 1e3


def band_label(lo: int, hi: int) -> str:
    return f"{lo}" if hi - 1 == lo else f"{lo}--{hi - 1}"


def ordered(summary: dict, names: dict[str, str], drop: tuple[str, ...] = ()) -> list[str]:
    arms = [a for a in summary if not a.endswith("_zero_mean") and a not in DROPPED and a not in drop]
    return [a for a in names if a in arms] + [a for a in arms if a not in names]


def fractions(D: list[float], edges: tuple[int, ...]) -> tuple[float, list[float]]:
    total = sum(D)
    return total, [sum(D[a:b]) / total if total else 0.0 for a, b in zip(edges, edges[1:])]


def model_rows(summary: dict, arm: str, name: str, edges: tuple[int, ...], hours_label: str | None = None) -> list[str]:
    """Two rows: the one-step error and the last rollout step's."""
    s = summary[arm]
    total, fr = fractions(s["D"], edges)
    rows = [" & ".join([name, "1 step", f"{total * TOTAL_SCALE:.2f}", *[f"{x:.2f}" for x in fr]]) + r" \\"]
    if s["rollout"]:
        last = s["rollout"][-1]
        total, fr = fractions(last["D"], edges)
        rows.append(" & ".join(["", rf"{last['hours']:g}\,h", f"{total * TOTAL_SCALE:.2f}", *[f"{x:.2f}" for x in fr]]) + r" \\")
    return rows


def table(
    label: str, caption: str, bandwidth: int, groups: list[tuple[str | None, dict, dict[str, str]]],
    drop: tuple[str, ...] = (),
) -> str:
    """``groups``: ``(group title or None, summary, names)`` per study."""
    edges = BAND_EDGES[bandwidth]
    k = len(edges) - 1
    header = [
        # A group header wider than the columns it spans widens the last of them.
        r"& & & \multicolumn{" + str(k) + r"}{c}{share by degree band} \\",
        rf"\cmidrule(lr){{4-{3 + k}}}",
        "Model & " + labels("horizon", r"$10^{3}\times L_2^2$", *[band_label(a, b) for a, b in zip(edges, edges[1:])]) + r" \\",
    ]
    rows: list[str] = []
    for title, summary, names in groups:
        if title is not None:
            rows.append(rf"\multicolumn{{{3 + k}}}{{@{{}}l}}{{\emph{{{title}}}}} \\")
        for arm in ordered(summary, names, drop):
            rows.extend(model_rows(summary, arm, names.get(arm, arm.replace("_", r"\_")), edges))
    return "\n".join([
        r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        r"\small", r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}ll{'r' * (k + 1)}@{{}}}}",
        r"\toprule", *header, r"\midrule", *rows, r"\bottomrule",
        r"\end{tabular*}", r"\end{table}",
    ])


#: Appended to a caption whose groups share a model, so a shared row does not read as two measurements.
SHARED_NOTE = (
    r" Each group is one of the paper's tables, row for row, so a model both report appears in both; those "
    r"rows are one evaluation shown twice, not two."
)


def caption_for(nlat: int, nlon: int, lead_minutes: float, bandwidth: int, num_examples: int) -> str:
    return (
        rf"Where the error sits in degree on the ${nlat}\times{nlon}$ configuration ({lead_minutes:g}\,min lead, "
        rf"{num_examples} test trajectories): the band-limited $L_2^2$ (degrees $0$ to ${bandwidth}$, seed mean) after one "
        r"step and at the end of the autoregressive rollout, and the fraction of it in each spherical-harmonic degree "
        r"band (ratios of seed-mean band energies). Degree $0$ is the global mean of each channel, which the solver "
        r"conserves exactly, so all error there is drift. Every one-step error concentrates in the top band, where the "
        r"target increment's energy is; between the two horizons of one model, a rollout that drifts moves its mass to "
        r"the low bands and one that loses per-step accuracy spreads it."
    )


def render(easy: dict | None, ablation: dict | None, zeroshot: dict | None, drop: tuple[str, ...] = ()) -> str:
    tables = []
    if easy is not None:
        B = int(easy["bandwidth"]); nlat, nlon = easy.get("solver", (128, 256))
        tables.append(table(
            "tab:degree-bands-128", caption_for(nlat, nlon, float(easy.get("lead_time_seconds", 1800.0)) / 60, B, easy["num_examples"]),
            B, [(None, easy["summary"], EASY_NAMES)],
        ))
    hard = [d for d in (ablation, zeroshot) if d is not None]
    if hard:
        first = hard[0]; B = int(first["bandwidth"]); nlat, nlon = first.get("solver", (256, 512))
        for other in hard[1:]:
            for key in ("bandwidth", "num_examples", "lead_time_seconds"):
                if other.get(key) != first.get(key):
                    raise SystemExit(f"the 256x512 studies disagree on {key}: {first.get(key)!r} vs {other.get(key)!r}")
        groups: list[tuple[str | None, dict, dict[str, str]]] = []
        if ablation is not None:
            groups.append(("Depth-1 ablation", ablation["summary"], ABLATION_NAMES))
        if zeroshot is not None:
            # A model both tables report appears in both groups; SHARED_NOTE says so.
            groups.append(("Zero-shot grid transfer", zeroshot["summary"], ZEROSHOT_NAMES))
        shown = [set(ordered(summary, names, drop)) for _, summary, names in groups]
        shared = any(a & b for a, b in zip(shown, shown[1:]))
        tables.append(table(
            "tab:degree-bands-256",
            caption_for(nlat, nlon, float(first.get("lead_time_seconds", 3600.0)) / 60, B, first["num_examples"])
            + (SHARED_NOTE if shared else ""),
            B, groups, drop,
        ))
    head = "% Generated by experiments/shallow_water_paper/degree_band_tables.py from the evaluation JSONs.\n% Requires: \\usepackage{booktabs} \\usepackage{amsmath}"
    return head + "\n\n" + "\n\n".join(tables) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--easy", default=None, help="the 128x256 benchmark's evaluation JSON")
    parser.add_argument("--ablation", default=None, help="the depth-1 ablation's evaluation JSON")
    parser.add_argument("--zeroshot", default=None, help="the zero-shot study's evaluation JSON")
    parser.add_argument("--drop", action="append", default=[], metavar="ARM",
                        help="leave this arm out of the 256x512 table (repeatable), e.g. a training of the "
                             "transferred arm the study holds but does not report")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    load = lambda p: json.loads(Path(p).read_text()) if p else None
    tex = render(load(args.easy), load(args.ablation), load(args.zeroshot), tuple(args.drop))
    check(tex)
    Path(args.output).write_text(tex)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
