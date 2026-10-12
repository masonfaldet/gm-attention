"""What the reproducibility bundles' ``evaluate.py`` scripts share.

Each bundle names its arms and defaults; this module discovers every
seed's ``checkpoints/best.pt``, loads it on its native grid, scores all
of them on the same held-out trajectories through
:mod:`experiments.shallow_water_paper.compare_spectral` (native pooled
errors, band-limited errors with their split by degree, and a
multi-step autoregressive rollout against the solver's own continuation),
aggregates seeds per arm as mean and sd, prints a report, and writes a
JSON holding the summary and every per-seed row.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from experiments.shallow_water_paper.compare_spectral import (
    ArmResult,
    ArmSource,
    LazyArm,
    evaluate_arms,
)
from experiments.shallow_water_paper.data import CHANNELS
from experiments.shallow_water_paper.spectral import full_bandwidth
from gm_attention.discretization.edge_geometry import cutoff_rule

ZERO_MEAN_SUFFIX = "_zero_mean"


def display_name(arm: str, names: dict[str, str]) -> str:
    if arm.endswith(ZERO_MEAN_SUFFIX):
        return display_name(arm[: -len(ZERO_MEAN_SUFFIX)], names) + ", zero-mean update"
    return names.get(arm, arm)


def with_zero_mean_variants(
    checkpoints: dict[str, list[tuple[int, Path]]], zero_mean: list[str],
) -> dict[str, tuple[list[tuple[int, Path]], bool]]:
    """``arm -> (seed checkpoints, zero_mean_update)``: every discovered arm,
    each arm named in ``zero_mean`` followed by a ``<arm>_zero_mean`` twin
    that loads the same checkpoints under the constraint."""
    unknown = set(zero_mean) - set(checkpoints)
    if unknown:
        raise SystemExit(f"--zero-mean-update names arms that were not evaluated: {sorted(unknown)}")
    out: dict[str, tuple[list[tuple[int, Path]], bool]] = {}
    for arm, seeds in checkpoints.items():
        out[arm] = (seeds, False)
        if arm in zero_mean:
            out[arm + ZERO_MEAN_SUFFIX] = (seeds, True)
    return out


def discover_checkpoints(
    root: Path, arms: dict[str, str], required: set[int] | None = None,
) -> dict[str, list[tuple[int, Path]]]:
    """``arm -> [(seed, best.pt), ...]`` for every seed directory that holds
    a best checkpoint, sorted by seed. With ``required``, every arm must
    hold exactly those seeds, so a partial study cannot pass as complete."""
    found: dict[str, list[tuple[int, Path]]] = {}
    for arm, directory in arms.items():
        seeds = []
        for seed_dir in sorted((root / directory).glob("seed*")):
            checkpoint = seed_dir / "checkpoints" / "best.pt"
            if checkpoint.is_file():
                seeds.append((int(seed_dir.name.removeprefix("seed")), checkpoint))
        if not seeds:
            raise FileNotFoundError(f"no seed*/checkpoints/best.pt under {root / directory}")
        if required is not None and {seed for seed, _ in seeds} != required:
            raise SystemExit(f"{arm}: seeds {sorted(seed for seed, _ in seeds)} under {root / directory}, "
                             f"required {sorted(required)}")
        found[arm] = seeds
    return found


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def aggregate(results: Sequence[ArmResult | dict], arm: str) -> dict:
    """Mean and sd over the seeds of ``arm`` for every scalar metric, the
    mean D(l), and the per-step rollout metrics. Accepts the results or
    their serialized (``per_seed``) rows, so a merge of parts can rebuild
    the summary from the rows it combined."""
    rows = [r if isinstance(r, dict) else asdict(r) for r in results]
    seeds = [r for r in rows if r["label"].startswith(arm + "/")]
    if not seeds:
        raise ValueError(f"no results for {arm}")
    first = seeds[0]
    keys = first["metrics"].keys()
    out: dict = {
        "seeds": len(seeds), "grid": first["grid"], "parameters": first["parameters"],
        "condition": first["condition"], "n_nodes": first["n_nodes"], "mean_degree": first.get("mean_degree"),
        "training_grid": first.get("training_grid", ""), "kind": first["kind"],
        "mixer": first.get("mixer"), "hidden_type": first.get("hidden_type"),
        "checkpoints": [{"label": r["label"], "checkpoint": r.get("checkpoint"), "epoch": r.get("epoch")} for r in seeds],
        "metrics": {k: mean_sd([r["metrics"][k] for r in seeds]) for k in keys},
        "paper_l1": mean_sd([r["paper_l1"] for r in seeds]), "paper_l2": mean_sd([r["paper_l2"] for r in seeds]),
        "D": [statistics.mean(r["D"][i] for r in seeds) for i in range(len(first["D"]))],
        "rollout": [],
    }
    for k in range(len(first["rollout"])):
        step: dict[str, Any] = {"step": first["rollout"][k]["step"], "hours": first["rollout"][k]["hours"], "coverage": first["rollout"][k]["coverage"]}
        for kind in ("rollout", "one_step", "persistence"):
            step[kind] = {key: mean_sd([r["rollout"][k][kind][key] for r in seeds]) for key in first["rollout"][k][kind]}
        step["D"] = [statistics.mean(r["rollout"][k]["D"][i] for r in seeds) for i in range(len(first["rollout"][k]["D"]))]
        out["rollout"].append(step)
    return out


def fmt(pair: tuple[float, float], digits: int = 4, exp: bool = False) -> str:
    mean, sd = pair
    return f"{mean:.{digits}e} ± {sd:.1e}" if exp else f"{mean:.{digits}f} ± {sd:.{digits}f}"


def report(
    summary: dict[str, dict], bandwidth: int, coverage: float, split: str, num_examples: int,
    names: dict[str, str] | None = None,
) -> None:
    names = names or {}
    suffix = f"_L{bandwidth}"
    print(f"\n{split} split, {num_examples} trajectories, band 0..{bandwidth} inclusive, target-band coverage {coverage:.4f}")
    print("\nsingle step (pooled over trajectories; mean ± sd over seeds)")
    print(f"{'arm':44s}{'n':>3}{'params':>9} | {'native L2^2':>21} {'geo':>17} {'vort':>17} {'div':>17} | {'band L2^2':>21}")
    print("-" * 160)
    for arm, s in summary.items():
        m = s["metrics"]
        band_cell = fmt(m['paper_squared_l2' + suffix], exp=True) if 'paper_squared_l2' + suffix in m else "n/a (grid too coarse)"
        print(f"{display_name(arm, names):44s}{s['seeds']:>3}{s['parameters']:>9,} | {fmt(m['paper_squared_l2'], exp=True):>21} "
              f"{fmt(m['geopotential_rmse']):>17} {fmt(m['vorticity_rmse']):>17} {fmt(m['divergence_rmse']):>17} | "
              f"{band_cell:>21}")
    print("\npaper-convention errors (per-sample L1 / L2 averaged; mean ± sd over seeds)")
    for arm, s in summary.items():
        print(f"  {display_name(arm, names):44s} L1 {fmt(s['paper_l1'])}   L2 {fmt(s['paper_l2'])}")
    edges = sorted(set(min(e, bandwidth + 1) for e in (0, 4, 12, 24, bandwidth + 1)))
    print("\nD(l) by band as a fraction of the band-limited L2^2 (seed-mean)")
    print(f"{'arm':44s}" + "".join(f"{f'{a}-{b - 1}':>12}" for a, b in zip(edges, edges[1:])))
    for arm, s in summary.items():
        if not s["D"]:
            print(f"{display_name(arm, names):44s}{'n/a':>12}")
            continue
        total = sum(s["D"]) or 1.0
        print(f"{display_name(arm, names):44s}" + "".join(f"{sum(s['D'][a:b]) / total:>12.3f}" for a, b in zip(edges, edges[1:])))
    steps = next(iter(summary.values()))["rollout"]
    if steps:
        band = "paper_squared_l2" + suffix
        print("\nautoregressive rollout, band-limited L2^2 (pooled; mean ± sd over seeds); persistence is arm-independent")
        print(f"{'hour':>6}{'persistence':>13} | " + " | ".join(f"{display_name(arm, names)[:22]:>22}  one-step" for arm in summary))
        def cell(d: dict, key: str, width: int, exp: bool = True) -> str:
            return (fmt(d[key], exp=exp) if exp else f"{d[key][0]:.2e}").rjust(width) if key in d else "n/a".rjust(width)

        for k, step in enumerate(steps):
            line = f"{step['hours']:>6g}{step['persistence'][band][0] if band in step['persistence'] else float('nan'):>13.4e} | "
            line += " | ".join(
                cell(summary[arm]['rollout'][k]['rollout'], band, 22) + " " + cell(summary[arm]['rollout'][k]['one_step'], band, 9, exp=False)
                for arm in summary)
            print(line)
        print("coverage of the band by the hour-k target: " + " ".join(f"h{s['hours']:g}={s['coverage']:.3f}" for s in steps))
        print("\nrollout channel RMSEs at the last hour (pooled; mean over seeds): " + "; ".join(
            f"{display_name(arm, names)}: " + " ".join(f"{short} {summary[arm]['rollout'][-1]['rollout'][name + '_rmse'][0]:.3f}" for name, short in zip(CHANNELS, ("geo", "vort", "div")))
            for arm in summary))


def build_parser(doc: str | None, *, default_root: str, default_output: str, n_arms: int) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=default_root)
    parser.add_argument("--arms", default=None,
                        help=f"comma-separated arm=run_directory pairs replacing the bundle's {n_arms} arms")
    parser.add_argument("--zero-mean-update", action="append", default=[], metavar="ARM",
                        help="also evaluate this arm with its predictions constrained to keep the input's global "
                             "mean per channel, as a <ARM>_zero_mean twin (repeatable)")
    parser.add_argument("--require-seeds", default=None, metavar="0,1,2",
                        help="refuse to run unless every arm holds exactly these seeds")
    parser.add_argument("--seeds", default=None, metavar="0,2",
                        help="evaluate only these seeds of every arm (after --require-seeds is checked), for "
                             "splitting a study over several invocations; merge the parts with --merge")
    parser.add_argument("--merge", nargs="+", default=None, metavar="PART.json",
                        help="instead of evaluating, merge these evaluation JSONs (parts of one study) into --output")
    parser.add_argument("--merge-only", action="append", default=[], metavar="ARM",
                        help="while merging, keep only these arms of each part (repeatable), for borrowing one "
                             "arm's rows from another evaluation of the same protocol")
    parser.add_argument("--relabel", action="append", default=[], metavar="OLD=NEW",
                        help="while merging, rename arm OLD, and every OLD_<suffix> arm, to NEW: parts that scored "
                             "two runs of one arm under the same labels join one study (repeatable)")
    parser.add_argument("--append", action="store_true",
                        help="merge this run's arms (or --merge's parts) into an existing --output JSON of the same "
                             "protocol instead of replacing it (for evaluating a large grid in a second invocation)")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--num-examples", type=int, default=32)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--no-teacher-forcing", action="store_true",
                        help="skip the teacher-forced pass of the rollout (its columns come back empty), "
                             "halving the rollout's forwards")
    parser.add_argument("--bandwidth", type=int, default=None, help="default: the solver's full band")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--projector-device", default=None,
                        help="where the band-limited fits live (default: --device); 'cpu' keeps a large grid's "
                             "float64 basis off the GPU")
    parser.add_argument("--output", default=default_output)
    return parser


ArmBuilder = Callable[[argparse.Namespace, torch.device], Sequence[ArmSource]]


def load_bundle_arms(args: argparse.Namespace, device: torch.device, names: dict[str, str]) -> list[ArmSource]:
    """The default arms: every seed's best checkpoint of each named arm on
    its own grid, plus the zero-mean twins asked for."""
    arms = dict(item.split("=", 1) for item in args.arms.split(",")) if args.arms else {arm: arm for arm in names}
    variants = with_zero_mean_variants(discover_checkpoints(Path(args.root), arms, required_seeds(args)), args.zero_mean_update)
    return [
        LazyArm(f"{arm}/seed{seed}", path, args.chunk_size, zero_mean_update=zero_mean)
        for arm, (seeds, zero_mean) in variants.items() for seed, path in selected_seeds(args, seeds)
    ]


def required_seeds(args: argparse.Namespace) -> set[int] | None:
    return {int(s) for s in args.require_seeds.split(",")} if getattr(args, "require_seeds", None) else None


def selected_seeds(args: argparse.Namespace, seeds: list[tuple[int, Path]]) -> list[tuple[int, Path]]:
    """``seeds`` restricted to ``--seeds`` when given (all of them otherwise)."""
    chosen = getattr(args, "seeds", None)
    if not chosen:
        return seeds
    wanted = {int(s) for s in chosen.split(",")}
    return [(seed, path) for seed, path in seeds if seed in wanted]


def run(args: argparse.Namespace, names: dict[str, str], *, build_arms: ArmBuilder | None = None) -> Path:
    """Evaluate ``args`` for a bundle whose arms are the keys of ``names``
    (display names as values); returns the JSON written. ``build_arms``
    replaces the default discovery for bundles that evaluate a run on
    several grids."""
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.merge:
        rules = dict(item.split("=", 1) for item in getattr(args, "relabel", []))
        only = set(getattr(args, "merge_only", [])) or None
        document = merge_files([Path(part) for part in args.merge], rules, only)
        # --append merges the parts into the JSON already there, one
        # invocation per set of --relabel rules.
        if args.append and out.exists():
            document = merged(json.loads(out.read_text()), document)
        out.write_text(json.dumps(document, indent=2) + "\n")
        print(f"merged {len(args.merge)} parts, {len(document['summary'])} arms -> {out}")
        return out

    device = torch.device(args.device)
    loaded = build_arms(args, device) if build_arms is not None else load_bundle_arms(args, device, names)
    if not loaded:
        raise SystemExit("nothing to evaluate: no arm matched --arms / --grids / --seeds")
    bandwidth = args.bandwidth if args.bandwidth is not None else full_bandwidth(int(loaded[0].cfg.data.nlat))
    projector_device = torch.device(args.projector_device) if args.projector_device else None
    existing = json.loads(out.read_text()) if args.append and out.exists() else None

    def document_of(results: list[ArmResult], coverage: float, *, partial: bool) -> dict:
        summary = {arm: aggregate(results, arm) for arm in dict.fromkeys(r.label.split("/")[0] for r in results)}
        document = {
            "split": args.split, "num_examples": args.num_examples, "bandwidth": bandwidth, "degrees": "0..L inclusive",
            "coverage": coverage, "steps": args.steps,
            "lead_time_seconds": float(loaded[0].cfg.data.lead_time_seconds),
            "solver": [int(loaded[0].cfg.data.nlat), int(loaded[0].cfg.data.nlon)],
            "solver_bandwidth": full_bandwidth(int(loaded[0].cfg.data.nlat)),
            "rollout_truth": "repeated_solver_step_with_restarted_adams_bashforth_history",
            "environment": {"torch": torch.__version__, "device": str(device), "cutoff_rule": cutoff_rule(),
                            "cuda": torch.version.cuda if device.type == "cuda" else None},
            "partial": partial,  # True while arms remain: a killed run keeps what it scored
            "summary": summary, "per_seed": [asdict(r) for r in results],
        }
        return merged(existing, document) if existing is not None else document

    def save(results: list[ArmResult], coverage: float) -> None:
        # Written after every arm, so a kill or a crash loses at most the arm in flight.
        out.write_text(json.dumps(document_of(results, coverage, partial=True), indent=2) + "\n")

    results, coverage = evaluate_arms(
        loaded, bandwidth=bandwidth, num_examples=args.num_examples, split=args.split, device=device,
        steps=args.steps, projector_device=projector_device, teacher_forcing=not args.no_teacher_forcing,
        on_result=save,
    )
    document = document_of(results, coverage, partial=False)
    report(document["summary"], bandwidth, coverage, args.split, args.num_examples, names)
    out.write_text(json.dumps(document, indent=2) + "\n")
    print(f"\nwrote {out}")
    return out


PROTOCOL_KEYS = ("split", "num_examples", "bandwidth", "steps", "lead_time_seconds", "solver")


def merged(existing: dict, new: dict) -> dict:
    """``new``'s arms merged into ``existing`` (same labels replaced), after
    checking the two evaluations share their protocol and, through the
    persistence errors (a property of the data alone), their trajectories."""
    for key in PROTOCOL_KEYS:
        if existing.get(key) != new.get(key):
            raise SystemExit(f"cannot merge: {key} differs ({existing.get(key)!r} vs {new.get(key)!r})")
    check_same_trajectories(existing, new)
    labels = {r["label"] for r in new["per_seed"]}
    out = dict(existing)
    out["per_seed"] = [r for r in existing["per_seed"] if r["label"] not in labels] + new["per_seed"]
    # Rebuilt from the combined rows: an arm whose seeds arrived in separate
    # parts is aggregated over all of them, in order of first appearance.
    arms = list(dict.fromkeys([*existing["summary"], *new["summary"]]))
    out["summary"] = {arm: aggregate(out["per_seed"], arm) for arm in arms}
    out["environment"] = new.get("environment")
    out["partial"] = bool(existing.get("partial")) or bool(new.get("partial"))
    return out


def check_same_trajectories(a: dict, b: dict) -> None:
    """Persistence errors depend on the data only, so two parts of one study
    must agree on them. The band-limited persistence is the same continuum
    norm on every grid (to float32 synthesis precision); the native one is
    each grid's own quadrature of it and differs by that quadrature's
    accuracy, so it is the fallback, with a looser tolerance, only where no
    row has the band."""
    band = f"paper_squared_l2_L{a.get('bandwidth')}"

    def persistence(document: dict) -> tuple[list[float], bool] | None:
        rows = [row for row in document["per_seed"] if row["rollout"]]
        for row in rows:
            if band in row["rollout"][0]["persistence"]:
                return [step["persistence"][band] for step in row["rollout"]], True
        for row in rows:
            return [step["persistence"]["paper_squared_l2"] for step in row["rollout"]], False
        return None

    pa, pb = persistence(a), persistence(b)
    if pa is None or pb is None:
        return
    (va, band_a), (vb, band_b) = pa, pb
    if band_a != band_b:  # one part has only grids too coarse for the band: compare native to native
        va = [step["persistence"]["paper_squared_l2"] for row in a["per_seed"] if row["rollout"] for step in row["rollout"]][: len(vb)]
        vb = [step["persistence"]["paper_squared_l2"] for row in b["per_seed"] if row["rollout"] for step in row["rollout"]][: len(va)]
        band_a = band_b = False
    tolerance = 1e-4 if band_a else 1e-2
    for k, (x, y) in enumerate(zip(va, vb, strict=True)):
        if abs(x - y) > tolerance * max(abs(x), abs(y), 1e-12):
            raise SystemExit(f"cannot merge: the parts saw different trajectories (persistence at step {k + 1}: {x!r} vs {y!r})")


def renamed(arm: str, rules: dict[str, str]) -> str:
    """``arm`` under the first matching rule, the longest ``old`` first so
    that a rule for a prefix does not claim a longer arm name. An arm
    already named for a rule's target is left alone."""
    for new in rules.values():
        if arm == new or arm.startswith(new + "_"):
            return arm
    for old in sorted(rules, key=len, reverse=True):
        if arm == old or arm.startswith(old + "_"):
            return rules[old] + arm[len(old):]
    return arm


def relabelled(document: dict, rules: dict[str, str]) -> dict:
    """``document`` with its arms renamed by ``rules``, in the summary's keys
    and in the per-seed labels alike, so parts that scored different runs
    of one arm under the same label can join one study."""
    if not rules:
        return document
    out = dict(document)
    out["summary"] = {}
    for arm, entry in document["summary"].items():
        key = renamed(arm, rules)
        if key in out["summary"]:
            raise SystemExit(f"cannot relabel: {arm} and another arm both become {key}")
        checkpoints = [{**c, "label": renamed(c["label"].split("/")[0], rules) + "/" + c["label"].split("/", 1)[1]}
                       for c in entry.get("checkpoints", [])]
        out["summary"][key] = {**entry, "checkpoints": checkpoints} if "checkpoints" in entry else entry
    out["per_seed"] = [{**row, "label": renamed(row["label"].split("/")[0], rules) + "/" + row["label"].split("/", 1)[1]}
                       for row in document["per_seed"]]
    return out


def restricted(document: dict, arms: set[str] | None) -> dict:
    """``document`` cut down to ``arms``, by the names it uses itself (before
    any relabelling): a study borrows a row from another evaluation of the
    same protocol without the rest of it; the merge still checks that the
    two saw the same trajectories."""
    if not arms:
        return document
    missing = arms - set(document["summary"])
    if missing:
        # The filter applies to every part of one invocation.
        raise SystemExit(f"--merge-only names arms this part does not hold: {sorted(missing)}; "
                         f"it holds {sorted(document['summary'])}")
    out = dict(document)
    out["summary"] = {a: s for a, s in document["summary"].items() if a in arms}
    out["per_seed"] = [r for r in document["per_seed"] if r["label"].split("/")[0] in arms]
    return out


def merge_files(parts: list[Path], relabel: dict[str, str] | None = None, only: set[str] | None = None) -> dict:
    """The parts of one study, evaluated in separate invocations, as one
    document; arms scored in several parts take the last part's rows.
    ``only`` keeps just those arms of each part (:func:`restricted`) and
    ``relabel`` renames what is left (:func:`relabelled`)."""
    documents = [relabelled(restricted(json.loads(part.read_text()), only), relabel or {}) for part in parts]
    out = documents[0]
    for document in documents[1:]:
        out = merged(out, document)
    return out


def main(
    doc: str | None, *, names: dict[str, str], default_root: str, default_output: str,
    build_arms: ArmBuilder | None = None, extra_arguments: Callable[[argparse.ArgumentParser], None] | None = None,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_parser(doc, default_root=default_root, default_output=default_output, n_arms=len(names))
    if extra_arguments is not None:
        extra_arguments(parser)
    run(parser.parse_args(), names, build_arms=build_arms)
