"""The segmentation table from ``evaluate.py``'s JSONs.

One row per arm: parameters, and the report split's cross-entropy, IoU and
mean class IoU, each the mean over seeds of the per-seed mean with the seed
standard deviation; the caption carries the paired per-image difference
between the SO(S^2) arm and the released S^2 SegFormer.

Usage, from the repository root::

    uv run python S2D3DS_128x256_Segmentation/tables.py --input outputs/S2D3DS_128x256_Segmentation/evaluation.json \\
        outputs/S2D3DS_128x256_Segmentation/evaluation_released.json ... --output ...
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the repository root

from experiments.shallow_water_paper.latex_tables import check

NAMES = {
    "sos2_segformer": r"$\mathrm{SO}(S^2)$-SegFormer (ours)",
    "sos2_segformer_scalar": r"$\mathrm{SO}(S^2)$-SegFormer, scalar-only (ablation)",
    "s2_segformer_released": r"S$^2$ SegFormer",
    "r2_segformer_released": r"R$^2$ SegFormer",
    "s2_unet_released": r"S$^2$ UNet",
}


def pm(mean: float, sd: float, bold: bool = False) -> str:
    text = rf"{mean:.3f} $\pm$ {sd:.3f}"
    return rf"{{\bfseries\boldmath {text}}}" if bold else text


def training_loss(runs: list[dict[str, Any]]) -> float | None:
    """Mean over seeds of each run's last logged training cross-entropy."""
    values: list[float] = []
    for run in runs:
        path = Path(run["run_dir"]) / "logs" / "metrics.jsonl"
        if not path.is_file():
            continue
        train = [json.loads(line) for line in path.open() if '"train"' in line]
        train = [row for row in train if row.get("split") == "train"]
        if train:
            values.append(float(train[-1]["cross_entropy"]))
    return statistics.fmean(values) if values else None


def per_image(entry: dict[str, Any], metric: str) -> dict[int, float]:
    """One arm's seed-averaged score per report image, keyed by dataset index;
    a seed with no score on an image (``null``: no valid pixel) is left out,
    and an image no seed scored is absent."""
    totals: dict[int, list[float]] = {}
    for run in entry["runs"]:
        for sample in run["samples"]:
            if sample[metric] is not None:
                totals.setdefault(int(sample["index"]), []).append(float(sample[metric]))
    return {index: statistics.fmean(values) for index, values in totals.items()}


def paired_difference(
    payload: dict[str, Any], arm: str, reference: str, metric: str
) -> tuple[float, float, int] | None:
    """``(mean, standard error, n)`` of ``arm - reference`` over the report
    images both arms scored, seed-averaged per image first."""
    if arm not in payload["arms"] or reference not in payload["arms"]:
        return None
    a, b = per_image(payload["arms"][arm], metric), per_image(payload["arms"][reference], metric)
    if set(a) != set(b):
        return None  # different image sets: no pairing
    common = sorted(a)
    if len(common) < 2:
        return None
    differences = [a[i] - b[i] for i in common]
    return statistics.fmean(differences), statistics.stdev(differences) / len(common) ** 0.5, len(common)


def render(
    payload: dict[str, Any],
    *,
    drop: tuple[str, ...] = (),
    pair: tuple[str, str] = ("sos2_segformer", "s2_segformer_released"),
) -> str:
    """The table, from one evaluation payload (several JSONs merge into one with ``merge_payloads``)."""
    names = {**payload.get("names", {}), **NAMES}
    arms = [a for a in payload["arms"] if a not in drop]
    if not arms:
        raise ValueError("no arms to render")
    metrics = ("cross_entropy", "iou", "mean_class_iou")
    rows: list[tuple[str, int, dict[str, tuple[float, float]]]] = []
    incomplete: list[str] = []
    for arm in arms:
        s = payload["arms"][arm]["summary"]
        name = names.get(arm, arm)
        if not s.get("complete", True):
            unfinished = s.get("unfinished_seeds") or [{"seed": "?", "run_epoch": max(s["epochs"]), "max_epochs": s["max_epochs"]}]
            name += " (unfinished: " + ", ".join(
                f"seed {u['seed']} at epoch {u['run_epoch']} of {u['max_epochs']}" for u in unfinished) + ")"
            incomplete.append(name)
        cells = {}
        for metric in metrics:
            means = [float(per[f"{metric}_mean"]) for per in s["per_seed"].values()]
            cells[metric] = (statistics.fmean(means), statistics.stdev(means) if len(means) > 1 else 0.0)
        rows.append((name, int(s["parameters"]), cells))
    best = {"cross_entropy": min(r[2]["cross_entropy"][0] for r in rows),
            "iou": max(r[2]["iou"][0] for r in rows), "mean_class_iou": max(r[2]["mean_class_iou"][0] for r in rows)}
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\begin{tabular}{lrccc}", r"\toprule",
        r"Model & Params & CE $\downarrow$ & IoU $\uparrow$ & mIoU $\uparrow$ \\", r"\midrule",
    ]
    for name, params, cells in rows:
        lines.append(f"{name} & {params:,} & " + " & ".join(
            pm(*cells[m], cells[m][0] == best[m]) for m in metrics) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]

    split = payload.get("split", "report")
    checkpoint = payload.get("checkpoint", "last")
    seeds = sorted({len(payload["arms"][a]["summary"]["seeds"]) for a in arms})
    n_images = int(next(iter(payload["arms"][arms[0]]["summary"]["per_seed"].values()))["num_samples"])
    model_state = (
        rf"the model after its final epoch (\texttt{{{checkpoint}.pt}})" if checkpoint == "last"
        else rf"the epoch with the lowest validation cross-entropy (\texttt{{{checkpoint}.pt}})"
    )
    if incomplete:
        model_state += (
            rf"; rows marked unfinished include seeds whose run has not reached its configured last "
            rf"epoch, scored at their \texttt{{{checkpoint}.pt}} as it stands, and are not final results"
        )
    paired_notes = []
    if pair[0] in arms and pair[1] in arms:
        parts = []
        for metric, symbol in (("iou", "IoU"), ("mean_class_iou", "mIoU"), ("cross_entropy", "CE")):
            result = paired_difference(payload, pair[0], pair[1], metric)
            if result is not None:
                mean, se, n = result
                parts.append(rf"$\Delta${symbol} $= {mean:+.3f} \pm {se:.3f}$")
        if parts:
            n = paired_difference(payload, pair[0], pair[1], "iou")[2]  # type: ignore[index]
            paired_notes.append(
                rf"Paired over the {n} report images both arms scored, {names.get(pair[0], pair[0])} minus "
                rf"{names.get(pair[1], pair[1])} (mean $\pm$ standard error): " + ", ".join(parts) + "."
            )
    caption = (
        rf"\caption{{Stanford 2D-3D-S semantic segmentation at $128\times256$, {split} split ({n_images} panoramas), "
        rf"{model_state}; mean $\pm$ standard deviation over {' / '.join(str(k) for k in seeds)} seeds. "
        r"CE is the released script's quadrature-weighted cross-entropy, IoU its class-agnostic "
        r"quadrature-weighted intersection over union, and mIoU the per-class IoU averaged over the classes "
        r"present. Best value per column in bold. "
        + " ".join(paired_notes) + "}"
    )
    lines += [caption, r"\label{tab:segmentation}", r"\end{table}", ""]
    tex = "\n".join(lines)
    check(tex)
    return tex


def merge_payloads(paths: list[Path]) -> dict[str, Any]:
    """Several evaluation JSONs as one payload; they must agree on the split and checkpoint."""
    payload: dict[str, Any] | None = None
    for path in paths:
        doc = json.loads(path.read_text())
        if payload is None:
            payload = {"split": doc["split"], "checkpoint": doc["checkpoint"], "arms": {}, "names": {}}
        elif (doc["split"], doc["checkpoint"]) != (payload["split"], payload["checkpoint"]):
            raise ValueError(f"{path} was scored on {doc['split']} / {doc['checkpoint']}, the first file on "
                             f"{payload['split']} / {payload['checkpoint']}")
        payload["arms"].update(doc["arms"])
        payload["names"].update(doc.get("names", {}))
    assert payload is not None
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", nargs="+", default=["outputs/S2D3DS_128x256_Segmentation/evaluation.json"],
                        help="evaluation JSONs (our arm and the released references may sit in separate files)")
    parser.add_argument("--output", default="outputs/S2D3DS_128x256_Segmentation/segmentation_table.tex")
    parser.add_argument("--drop", action="append", default=[], help="an arm to leave out (repeatable)")
    parser.add_argument("--pair", default="sos2_segformer,s2_segformer_released",
                        help="two arms, 'arm,reference', for the paired per-image comparison in the caption")
    args = parser.parse_args()
    payload = merge_payloads([Path(p) for p in args.input])
    first, _, second = args.pair.partition(",")
    tex = render(payload, drop=tuple(args.drop), pair=(first.strip(), second.strip()))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(tex)
    print(tex)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
