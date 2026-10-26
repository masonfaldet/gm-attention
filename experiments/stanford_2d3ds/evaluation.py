"""Score fitted segmentation runs on the report split, every seed of every arm.

Each model's cross-entropy, IoU and accuracy on the held-out 2.5 % report
split are reported as a mean and standard deviation over the split's
samples, scored one panorama at a time by the released loss and metric
classes, on the model as it stands after the last epoch (the protocol of
"Attention on the Sphere", Section 4.1). The per-sample scores are kept,
so a table can show the sample spread and the seed spread; one JSON is
written per run root.

Usage, from the repository root::

    uv run python -m experiments.stanford_2d3ds.evaluation --root outputs/S2D3DS_128x256_Segmentation
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import statistics
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from experiments.stanford_2d3ds.data import (
    SegmentationSplits,
    dataset_digest,
    image_to_field,
    logits_to_image,
)
from experiments.stanford_2d3ds.model import real_parameter_count
from experiments.stanford_2d3ds.train import (
    MONITOR_METRIC,
    PROTOCOL_FILENAME,
    SegmentationExperiment,
    SegmentationScorer,
    _same_value,
    build_experiment,
    open_splits,
    resolve_device,
)
from gm_attention.discretization.edge_geometry import cutoff_rule
from gm_attention.training.checkpointing import load_checkpoint

log = logging.getLogger(__name__)

METRICS = (MONITOR_METRIC, "iou", "accuracy", "pixel_accuracy", "mean_class_iou")
CHECKPOINTS = ("last", "best")


@dataclass(frozen=True)
class SampleScore:
    """The released loss and metrics on one panorama."""

    index: int
    cross_entropy: float
    iou: float
    accuracy: float
    pixel_accuracy: float
    mean_class_iou: float
    class_iou: list[float | None]  #: per class; None where the class is absent from target and prediction


@dataclass
class RunResult:
    """One seed of one arm on the report split."""

    arm: str
    seed: int
    run_dir: str
    checkpoint: str
    epoch: int  #: the scored checkpoint's epoch (0-based)
    run_epoch: int  #: how far the run got: last.pt's epoch (0-based), the scored one if there is no last.pt
    max_epochs: int  #: the run's configured length, from its protocol
    complete: bool
    parameters: int
    samples: list[SampleScore]
    model_fingerprint: str = ""  #: SHA-256 of the protocol's resolved model config; seeds of one arm must share it

    def summary(self) -> dict[str, Any]:
        """Mean and standard deviation over the split's samples, per metric,
        over the finite values; a non-finite value (a fully ignored panorama
        has no IoU) is counted, never averaged, and an empty metric is None."""
        out: dict[str, Any] = {}
        for metric in METRICS:
            raw = [getattr(s, metric) for s in self.samples]
            values = [v for v in raw if math.isfinite(v)]
            out[f"{metric}_mean"] = statistics.fmean(values) if values else None
            out[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            out[f"{metric}_nonfinite"] = len(raw) - len(values)
        out["num_samples"] = float(len(self.samples))
        out.update({"epoch": self.epoch + 1, "run_epoch": self.run_epoch + 1, "max_epochs": self.max_epochs, "complete": self.complete})
        return out


def sample_record(sample: SampleScore) -> dict[str, Any]:
    """``sample`` as a JSON row, a non-finite metric (a panorama with no
    valid pixel has no IoU) written as ``null`` as ``class_iou`` already
    does: the JSON is strict (``allow_nan=False``)."""
    return {key: None if isinstance(value, float) and not math.isfinite(value) else value
            for key, value in asdict(sample).items()}


def read_protocol(run_dir: Path) -> dict[str, Any]:
    path = run_dir / PROTOCOL_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"{run_dir} has no {PROTOCOL_FILENAME}.")
    return dict(json.loads(path.read_text()))



def _int_keys(value: Any) -> Any:
    """Undo JSON's stringification of int dict keys (``{0: 1}`` -> ``{"0": 1}``)."""
    if isinstance(value, dict):
        return {(int(k) if isinstance(k, str) and k.lstrip("-").isdigit() else k): _int_keys(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_int_keys(v) for v in value]
    return value


def config_from_protocol(protocol: dict[str, Any], *, dataset_file: str | Path | None = None) -> DictConfig:
    """The config a run's protocol records: the resolved model and training
    settings and the data settings the run trained with. An evaluation
    builds from this, not from the ``.hydra/config.yaml`` beside it, which
    Hydra rewrites on every launch into the directory, a refused one
    included."""
    for key in ("model_config", "training_config"):
        if key not in protocol:
            raise ValueError(f"the protocol records no {key}; it predates the config record and cannot be rebuilt from.")
    data = {
        "dataset_file": str(dataset_file if dataset_file is not None else protocol["dataset_file"]),
        "grid": protocol["grid"], "img_size": list(protocol["img_size"]),
        "split_seed": protocol["split_seed"], "split_ratios": list(protocol["split_ratios"]),
        "exclude_polar_fraction": protocol.get("exclude_polar_fraction", 0.0),
        "loader_order_seed": protocol.get("loader_order_seed", 0), "num_workers": 0,
    }
    cfg = OmegaConf.create({
        "model": _int_keys(protocol["model_config"]), "training": _int_keys(protocol["training_config"]),
        "data": data, "seed": protocol.get("model_seed", 0), "device": None,
    })
    assert isinstance(cfg, DictConfig)
    return cfg


def hydra_config_drift(run_dir: Path, protocol: dict[str, Any]) -> list[str]:
    """The protocol keys on which the run directory's ``.hydra/config.yaml``
    disagrees with the protocol (a refused relaunch still rewrites that
    file); informational, the protocol is what is scored."""
    from experiments.stanford_2d3ds.train import _training_identity

    path = run_dir / ".hydra" / "config.yaml"
    if not path.is_file():
        return []
    cfg = OmegaConf.load(path)
    assert isinstance(cfg, DictConfig)
    current = {
        "model_config": OmegaConf.to_container(cfg.model, resolve=True),
        "training_config": _training_identity(cfg),
    }
    return [key for key, value in current.items() if not _same_value(protocol.get(key), json.loads(json.dumps(value)))]


def restore_experiment(
    run_dir: Path,
    *,
    checkpoint: str,
    device: torch.device,
    dataset_file: str | Path | None = None,
) -> tuple[DictConfig, SegmentationExperiment, SegmentationSplits, int]:
    """Rebuild a run's data, model and geometry from its protocol and load its weights.

    The data is verified against the protocol by identity: the file's
    SHA-256 (files of equal size and sample count can hold different
    panoramas in a different order, and the released split is a permutation
    of positions), the class labels, the split indices and the normalization
    the run trained with. The model and its geometry are built from the
    protocol's recorded configs (``config_from_protocol``), so a rewritten
    ``.hydra/config.yaml`` cannot change what is scored; a disagreement
    between the two is logged. The cutoff rule of the process must be the
    one the run trained under.
    """
    protocol = read_protocol(run_dir)
    cfg = config_from_protocol(protocol, dataset_file=dataset_file)
    drift = hydra_config_drift(run_dir, protocol)
    if drift:
        log.warning("%s: .hydra/config.yaml disagrees with the protocol on %s; the protocol is what is scored",
                    run_dir, ", ".join(drift))
    splits = open_splits(cfg)
    current: dict[str, Any] = {
        "dataset_sha256": dataset_digest(splits.dataset_file),
        "num_samples": len(splits.dataset),
        "num_classes": splits.num_classes,
        "class_labels": list(splits.class_labels),
        "split_indices": {k: list(v) for k, v in splits.indices.items()},
        "normalization": splits.normalization.state_dict(),
    }
    for key, value in current.items():
        recorded = protocol.get(key)
        if recorded is None:
            raise ValueError(
                f"{run_dir}: the run's protocol records no {key}; it predates the identity "
                "check and cannot be scored against an unverified file."
            )
        if not _same_value(recorded, json.loads(json.dumps(value))):
            raise ValueError(
                f"{run_dir}: the dataset's {key} differs from the run's protocol; refusing "
                "to score a model on data it did not train beside."
            )
    if "cutoff_rule" not in protocol:
        raise ValueError(f"{run_dir}: the run's protocol records no cutoff_rule; it cannot be scored under an unverified rule.")
    if protocol["cutoff_rule"] != cutoff_rule():
        raise ValueError(
            f"{run_dir} trained under the {protocol['cutoff_rule']!r} cutoff rule and this process "
            f"runs {cutoff_rule()!r} (GM_CUTOFF_RULE): the model would aggregate over different "
            "edges than it was trained with."
        )
    experiment = build_experiment(
        cfg, device, shape=splits.shape, num_classes=splits.num_classes, in_channels=splits.in_channels
    )
    if real_parameter_count(experiment.model) != int(protocol["real_parameter_count"]):
        raise ValueError(f"{run_dir}: the rebuilt model has {real_parameter_count(experiment.model)} parameters, "
                         f"the protocol records {protocol['real_parameter_count']}.")
    path = run_dir / "checkpoints" / f"{checkpoint}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist.")
    metadata = load_checkpoint(path, model=experiment.model, map_location=device, restore_rng_state=False)
    experiment.model.eval()
    return cfg, experiment, splits, int(metadata.epoch)


def score_split(
    experiment: SegmentationExperiment,
    splits: SegmentationSplits,
    *,
    split: str,
    device: torch.device,
    grid: str = "equiangular",
    scorer: SegmentationScorer | None = None,
    log_every: int = 8,
) -> list[SampleScore]:
    """Every sample of ``split`` through the model, one at a time."""
    scorer = scorer or SegmentationScorer.build(
        splits.shape, grid=grid, label_smoothing=0.0, device=device
    )
    dataset = splits.subset(split)
    loader: DataLoader[Any] = DataLoader(dataset, batch_size=None, shuffle=False)
    scores: list[SampleScore] = []
    started = time.time()
    with torch.no_grad():
        for position, (image, target) in enumerate(loader):
            image, target = image.to(device), target.to(device)
            if experiment.batched:
                logits = experiment.model(image.unsqueeze(0))
            else:
                assert experiment.hierarchy is not None
                logits = logits_to_image(
                    experiment.model(image_to_field(image), experiment.hierarchy), experiment.shape
                )
            output = scorer(logits, target.unsqueeze(0))
            per_class = scorer.class_iou(logits, target.unsqueeze(0))[0].tolist()
            scores.append(
                SampleScore(
                    index=int(dataset.indices[position]),
                    cross_entropy=output.metrics[MONITOR_METRIC],
                    iou=output.metrics["iou"],
                    accuracy=output.metrics["accuracy"],
                    pixel_accuracy=output.metrics["pixel_accuracy"],
                    mean_class_iou=output.metrics["mean_class_iou"],
                    class_iou=[None if math.isnan(v) else float(v) for v in per_class],
                )
            )
            if log_every and (position + 1) % log_every == 0:
                elapsed = time.time() - started
                log.info(
                    "  %d/%d samples, %.1f s elapsed, %.1f s left",
                    position + 1, len(dataset), elapsed, elapsed / (position + 1) * (len(dataset) - position - 1),
                )
    return scores



def evaluate_run(
    run_dir: Path,
    *,
    arm: str,
    seed: int,
    checkpoint: str = "last",
    split: str = "report",
    device: torch.device,
    dataset_file: str | Path | None = None,
) -> RunResult:
    cfg, experiment, splits, epoch = restore_experiment(
        run_dir, checkpoint=checkpoint, device=device, dataset_file=dataset_file
    )
    protocol = read_protocol(run_dir)
    progress = run_progress(run_dir)
    log.info("%s seed %d: %s.pt (epoch %d; the run at epoch %d of %d%s), %d parameters, %d %s samples",
             arm, seed, checkpoint, epoch + 1, progress.run_epoch + 1, progress.max_epochs,
             ", finished" if progress.complete else "", int(protocol["real_parameter_count"]),
             len(splits.indices[split]), split)
    samples = score_split(
        experiment, splits, split=split, device=device,
        grid=str(cfg.data.get("grid", "equiangular")),
    )
    return RunResult(
        arm=arm, seed=seed, run_dir=str(run_dir), checkpoint=checkpoint, epoch=epoch,
        run_epoch=progress.run_epoch if progress.run_epoch >= 0 else epoch,
        max_epochs=progress.max_epochs, complete=progress.complete,
        parameters=int(protocol["real_parameter_count"]), samples=samples,
        model_fingerprint=model_fingerprint(protocol),
    )


def model_fingerprint(protocol: dict[str, Any]) -> str:
    """SHA-256 of the protocol's resolved model config, radii, stage levels
    and cutoff rule: the identity the seeds of one arm must share."""
    identity = {key: protocol.get(key) for key in ("model", "model_config", "radii", "stage_levels", "grid_internal", "cutoff_rule")}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class RunProgress:
    """How far a run got, from its protocol and its ``last.pt``."""

    run_epoch: int  #: last.pt's epoch (0-based); -1 without a last.pt
    max_epochs: int  #: from the protocol's recorded training config
    finished: bool  #: the protocol records the training loop ending
    complete: bool  #: finished, or last.pt is the configured last epoch


def checkpoint_epoch(run_dir: Path, checkpoint: str) -> int:
    """The epoch a checkpoint was written at, without building the model."""
    payload = torch.load(run_dir / "checkpoints" / f"{checkpoint}.pt", map_location="cpu", weights_only=False)
    return int(payload["epoch"])


def run_progress(run_dir: Path) -> RunProgress:
    """Whether a run finished: its protocol records a ``finished`` event when
    the training loop ends (early stopping included), or its ``last.pt`` is
    the model after the configured last epoch. Both come from the run's own
    records; ``.hydra/config.yaml`` is not consulted. A run without
    ``last.pt`` is unfinished."""
    protocol = read_protocol(run_dir)
    max_epochs = int(protocol["training_config"]["max_epochs"])
    last = run_dir / "checkpoints" / "last.pt"
    run_epoch = checkpoint_epoch(run_dir, "last") if last.is_file() else -1
    # A finished event counts only while it describes this last.pt under this
    # length; one recorded before an extension would pass an interrupted run
    # off as complete. An entry predating these fields is taken at its word.
    finished = any(
        entry.get("event") == "finished"
        and int(entry.get("max_epochs", max_epochs)) == max_epochs
        and int(entry.get("epochs", run_epoch + 1)) == run_epoch + 1
        for entry in protocol.get("history", [])
    )
    complete = last.is_file() and (finished or run_epoch + 1 >= max_epochs)
    return RunProgress(run_epoch=run_epoch, max_epochs=max_epochs, finished=finished, complete=complete)


def discover_runs(
    root: Path, arms: Mapping[str, str], *, checkpoint: str = "last"
) -> dict[str, list[tuple[int, Path]]]:
    """``{arm: [(seed, run_dir), ...]}`` for every ``<root>/<dir>/seed<N>``
    holding ``checkpoints/<checkpoint>.pt``; a run without that checkpoint
    is skipped."""
    found: dict[str, list[tuple[int, Path]]] = {}
    for arm, directory in arms.items():
        base = root / directory
        seeds: list[tuple[int, Path]] = []
        for candidate in sorted(base.glob("seed*")):
            if not candidate.is_dir() or not candidate.name[4:].isdigit():
                continue
            if not (candidate / "checkpoints" / f"{checkpoint}.pt").is_file():
                log.warning("%s has no checkpoints/%s.pt; skipped", candidate, checkpoint)
                continue
            seeds.append((int(candidate.name[4:]), candidate))
        if seeds:
            found[arm] = seeds
    return found


def aggregate(results: Sequence[RunResult]) -> dict[str, Any]:
    """Seed mean and sd of the per-sample means, plus every seed's own summary."""
    per_seed = {str(r.seed): r.summary() for r in results}
    summary: dict[str, Any] = {"seeds": sorted(r.seed for r in results), "per_seed": per_seed}
    for metric in METRICS:
        means = [m for m in (r.summary()[f"{metric}_mean"] for r in results) if m is not None]
        stds = [r.summary()[f"{metric}_std"] for r in results]
        summary[f"{metric}_mean"] = statistics.fmean(means) if means else None
        summary[f"{metric}_seed_sd"] = statistics.stdev(means) if len(means) > 1 else 0.0
        # The sample spread: the per-sample sd, averaged over seeds.
        summary[f"{metric}_sample_sd"] = statistics.fmean(stds)
        summary[f"{metric}_nonfinite"] = sum(int(r.summary()[f"{metric}_nonfinite"]) for r in results)
    parameters = {r.parameters for r in results}
    if len(parameters) != 1:
        raise ValueError(f"the seeds of one arm have different parameter counts: {sorted(parameters)}.")
    fingerprints = {r.model_fingerprint for r in results}
    if len(fingerprints) != 1:
        raise ValueError("the seeds of one arm were trained as different models (model config, radii, "
                         "stage levels or cutoff rule differ); they cannot be one arm.")
    indices = {tuple(s.index for s in r.samples) for r in results}
    if len(indices) != 1:
        raise ValueError("the seeds of one arm scored different samples; they cannot be one arm.")
    # Per-class IoU, averaged over the samples (and seeds) in which the class
    # is present in the target or the prediction; None if never.
    num_classes = max(len(s.class_iou) for r in results for s in r.samples)
    per_class: list[list[float]] = [[] for _ in range(num_classes)]
    for r in results:
        for s in r.samples:
            for c, value in enumerate(s.class_iou):
                if value is not None:
                    per_class[c].append(value)
    summary["class_iou_mean"] = [statistics.fmean(v) if v else None for v in per_class]
    summary["class_iou_samples"] = [len(v) for v in per_class]
    summary["parameters"] = results[0].parameters
    summary["epochs"] = sorted({r.epoch + 1 for r in results})
    max_epochs = {r.max_epochs for r in results}
    if len(max_epochs) != 1:
        raise ValueError(f"the seeds of one arm have different configured lengths: {sorted(max_epochs)}.")
    summary["max_epochs"] = results[0].max_epochs
    summary["complete"] = all(r.complete for r in results)
    summary["unfinished_seeds"] = [
        {"seed": r.seed, "run_epoch": r.run_epoch + 1, "max_epochs": r.max_epochs, "epoch": r.epoch + 1}
        for r in results if not r.complete
    ]
    summary["checkpoint"] = results[0].checkpoint
    return summary


def run_evaluation(
    root: Path,
    arms: Mapping[str, str],
    *,
    checkpoint: str,
    split: str,
    device: torch.device,
    seeds: Iterable[int] | None = None,
    dataset_file: str | Path | None = None,
    allow_incomplete: bool = False,
    require_seeds: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Score every discovered run. A run whose ``last.pt`` is not the model
    after its configured last epoch is unfinished: skipped with a warning
    unless ``allow_incomplete``, in which case it is scored and the arm's
    summary says so (``complete: false``). With ``require_seeds``, an arm
    that does not end up with exactly those seeds scored is refused."""
    wanted = None if seeds is None else set(seeds)
    required = None if require_seeds is None else set(require_seeds)
    found = discover_runs(root, arms, checkpoint=checkpoint)
    if not found:
        raise SystemExit(f"no runs with checkpoints/{checkpoint}.pt under {root} for arms {sorted(arms)}.")
    payload: dict[str, Any] = {
        "split": split, "checkpoint": checkpoint, "root": str(root), "arms": {},
        "allow_incomplete": allow_incomplete,
        "environment": {"torch": torch.__version__, "device": str(device)},
    }
    for arm, runs in found.items():
        results = []
        for seed, run_dir in runs:
            if wanted is not None and seed not in wanted:
                continue
            progress = run_progress(run_dir)
            if not progress.complete and not allow_incomplete:
                log.warning("%s: %s, an unfinished run; skipped (--allow-incomplete scores its %s.pt, marked)",
                            run_dir,
                            f"last.pt is epoch {progress.run_epoch + 1} of {progress.max_epochs}"
                            if progress.run_epoch >= 0 else "no last.pt",
                            checkpoint)
                continue
            results.append(
                evaluate_run(run_dir, arm=arm, seed=seed, checkpoint=checkpoint, split=split,
                             device=device, dataset_file=dataset_file)
            )
        if not results:
            continue
        if required is not None and {r.seed for r in results} != required:
            raise ValueError(
                f"{arm}: seeds {sorted(r.seed for r in results)} scored where {sorted(required)} are required "
                "(unfinished or missing seeds; --allow-incomplete scores unfinished ones, marked)."
            )
        payload["arms"][arm] = {
            "summary": aggregate(results),
            "runs": [
                {**asdict(r), "samples": [sample_record(s) for s in r.samples], "summary": r.summary()}
                for r in results
            ],
        }
    return payload


def build_parser(doc: str | None, *, default_root: str, default_output: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=default_root, help="run root holding <arm>/seed<N>/ directories")
    parser.add_argument("--arms", default=None, help="comma-separated arm=directory pairs (default: every known arm under its own name)")
    parser.add_argument("--seeds", default=None, help="comma-separated seeds to score (default: every seed found)")
    parser.add_argument("--checkpoint", choices=CHECKPOINTS, default="last",
                        help="last.pt is the released protocol (no selection); best.pt the lowest val cross-entropy")
    parser.add_argument("--split", choices=("report", "val", "train"), default="report")
    parser.add_argument("--dataset", default=None, help="override the runs' recorded dataset path (same file, another location)")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="score runs whose last.pt precedes their configured last epoch, marked as such")
    parser.add_argument("--require-seeds", default=None, metavar="0,1,2",
                        help="refuse an arm unless exactly these seeds end up scored")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=default_output)
    return parser


def main(doc: str | None = None, *, names: Mapping[str, str], default_root: str, default_output: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = build_parser(doc, default_root=default_root, default_output=default_output)
    args = parser.parse_args()
    arms = dict(names)
    if args.arms:
        arms = {}
        for item in args.arms.split(","):
            arm, _, directory = item.partition("=")
            arms[arm.strip()] = (directory or arm).strip()
    seeds = None if args.seeds is None else [int(s) for s in args.seeds.split(",")]
    device = resolve_device(args.device)
    payload = run_evaluation(
        Path(args.root), arms, checkpoint=args.checkpoint, split=args.split,
        device=device, seeds=seeds, dataset_file=args.dataset, allow_incomplete=args.allow_incomplete,
        require_seeds=None if args.require_seeds is None else [int(s) for s in args.require_seeds.split(",")],
    )
    payload["names"] = {arm: names.get(arm, arm) for arm in payload["arms"]}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"wrote {output}")
    for arm, entry in payload["arms"].items():
        s = entry["summary"]
        print(
            f"{names.get(arm, arm):<40} loss {s['cross_entropy_mean']:.3f} +- {s['cross_entropy_sample_sd']:.3f}"
            f"  IoU {s['iou_mean']:.3f} +- {s['iou_sample_sd']:.3f}"
            f"  Acc {s['accuracy_mean']:.3f} +- {s['accuracy_sample_sd']:.3f}"
            f"  pixel acc {s['pixel_accuracy_mean']:.3f}  mIoU {s['mean_class_iou_mean']:.3f}"
            f"  (seeds {s['seeds']}, seed sd IoU {s['iou_seed_sd']:.3f})"
            + ("" if s.get("complete", True) else "  UNFINISHED: " + ", ".join(
                f"seed {u['seed']} at epoch {u['run_epoch']} of {u['max_epochs']}" for u in s["unfinished_seeds"]))
        )


if __name__ == "__main__":
    main(__doc__, names={"gm_segformer": "SOS2-SegFormer"},
         default_root="outputs/S2D3DS_128x256_Segmentation",
         default_output="outputs/S2D3DS_128x256_Segmentation/evaluation.json")
