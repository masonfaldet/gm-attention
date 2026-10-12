"""Adopt the released script's finished runs into the bundle's run layout.

The released script leaves each seed a plain state dict
(``<model>/checkpoint.pt``) and a pickle of the report split's metrics
(``output_data/metrics.pkl``). This writes beside them what ``evaluate.py``
reads: ``segmentation_protocol.json``, from the trainer's ``build_protocol``
on the bundle arm that builds the same model, with the optimizer settings
(``--learning_rate``, ``--num_epochs``, ``--batch_size``,
``--max_grad_norm``) read from the run log's ``argv`` banner and a
``finished`` history event at the released epoch count; and
``checkpoints/last.pt``, the weights loaded strictly into that model. The
pickle's numbers, when pandas can read them, are kept under
``released_report_metrics`` for the cross-check against the evaluator's
per-panorama ``iou`` and ``accuracy`` means on the report split.

Usage, from the repository root, in the reference environment::

    UV_PROJECT_ENVIRONMENT=.venv-ref uv run --no-sync python S2D3DS_128x256_Segmentation/reference/adopt.py
    ... adopt.py --model s2unet_sc2_layers4_e128   # another registry entry (its own arm)
    ... adopt.py --seeds 0 1                       # some seeds
    ... adopt.py --force                           # rewrite an adoption
    ... adopt.py --override data.img_size=[64,128] # compose overrides

Then ``S2D3DS_128x256_Segmentation/reference/run.sh evaluate --device cuda:0``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pathlib
import re
import sys
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.stanford_2d3ds.train import (
    PROTOCOL_FILENAME,
    build_experiment,
    build_protocol,
    open_splits,
)
from gm_attention.training.checkpointing import save_checkpoint
from gm_attention.training.provenance import code_revision

log = logging.getLogger("adopt")

BUNDLE = REPO / "S2D3DS_128x256_Segmentation"
#: Registry entry -> (the bundle config that builds the same model, the arm name the adopted runs are scored as).
MODELS: dict[str, tuple[str, str]] = {
    "s2nsegformer_sc2_layers4_e128": ("s2_segformer", "s2_segformer_released"),
    "s2unet_sc2_layers4_e128": ("s2_unet", "s2_unet_released"),
    "nsegformer_sc2_layers4_e128": ("r2_segformer", "r2_segformer_released"),
}
DEFAULT_MODEL = "s2nsegformer_sc2_layers4_e128"
OUTPUTS = pathlib.Path("outputs/S2D3DS_128x256_Segmentation")
LOGS = pathlib.Path("logs/S2D3DS_128x256_Segmentation")
#: The runner's banner: ``argv: ... --learning_rate 5e-4 ...``.
ARGV = re.compile(r"^argv: (.*)$")
#: Released script flag -> the bundle's training key and its parser.
FLAGS: dict[str, tuple[str, Any]] = {
    "--learning_rate": ("lr", float),
    "--num_epochs": ("max_epochs", int),
    "--batch_size": ("batch_size", int),
    "--max_grad_norm": ("grad_clip_norm", float),
}
#: The released script's own defaults.
SCRIPT_DEFAULTS: dict[str, Any] = {"lr": 5e-4, "max_epochs": 200, "batch_size": 8, "grad_clip_norm": 4.0}
TRAINED_BY = {
    "script": "torch-harmonics v0.8.0 examples/segmentation/train.py (verbatim; S2D3DS_128x256_Segmentation/reference/upstream)",
    "runner": "S2D3DS_128x256_Segmentation/reference/run_released.py",
}


def training_settings(log_text: str | None) -> tuple[dict[str, Any], str]:
    """The optimizer settings the released run used, from its log banner,
    or the script's defaults when there is no banner to read."""
    if log_text is not None:
        for line in log_text.splitlines():
            match = ARGV.match(line)
            if match:
                tokens = match.group(1).split()
                settings = dict(SCRIPT_DEFAULTS)
                for flag, (key, parse) in FLAGS.items():
                    if flag in tokens:
                        settings[key] = parse(tokens[tokens.index(flag) + 1])
                return settings, "the run log's argv banner"
    return dict(SCRIPT_DEFAULTS), "the released script's defaults (no log banner found)"


def compose_config(seed: int, settings: dict[str, Any], overrides: list[str], *, model: str = DEFAULT_MODEL) -> DictConfig:
    """The bundle arm that builds ``model`` at this seed and these settings."""
    base = [f"seed={seed}"] + [f"training.{key}={value}" for key, value in settings.items()]
    with initialize_config_dir(version_base=None, config_dir=str(BUNDLE / "configs")):
        cfg = compose(config_name=MODELS[model][0], overrides=base + list(overrides))
    assert isinstance(cfg, DictConfig)
    return cfg


def released_metrics(run_dir: pathlib.Path, model: str = DEFAULT_MODEL) -> dict[str, float] | None:
    """The released pickle's column as plain floats, or None without
    pandas (the pickle is a DataFrame) or without the file."""
    path = run_dir / model / "output_data" / "metrics.pkl"
    if not path.is_file():
        return None
    try:
        import pandas as pd  # noqa: F401
    except ImportError:
        log.warning("%s: pandas is not installed here, so the released numbers are not recorded", path)
        return None
    import pickle

    with path.open("rb") as handle:
        frame = pickle.load(handle)
    column = frame[model] if model in frame else frame.iloc[:, 0]
    return {str(index): float(value) for index, value in column.items()}


def adopt(
    run_dir: pathlib.Path,
    seed: int,
    *,
    log_text: str | None,
    overrides: list[str],
    force: bool,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Write ``segmentation_protocol.json``, ``checkpoints/last.pt`` and
    ``.hydra/config.yaml`` into ``run_dir`` from its released files."""
    source = run_dir / model / "checkpoint.pt"
    if not source.is_file():
        raise FileNotFoundError(f"{source} does not exist; the released run has not finished (or is elsewhere).")
    last = run_dir / "checkpoints" / "last.pt"
    protocol_path = run_dir / PROTOCOL_FILENAME
    if (last.exists() or protocol_path.exists()) and not force:
        raise FileExistsError(f"{run_dir} is already adopted ({last.name} or {protocol_path.name} exists); pass --force to rewrite.")

    settings, provenance = training_settings(log_text)
    cfg = compose_config(seed, settings, overrides, model=model)
    splits = open_splits(cfg)
    device = torch.device("cpu")
    experiment = build_experiment(
        cfg, device, shape=splits.shape, num_classes=splits.num_classes, in_channels=splits.in_channels
    )
    state = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or any(not isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError(f"{source} is not a plain state dict.")
    experiment.model.load_state_dict(state, strict=True)  # the key layout is the check
    experiment.model.eval()

    epochs = int(settings["max_epochs"])
    steps_per_epoch = math.ceil(splits.sizes()["train"] / int(settings["batch_size"]))
    protocol = build_protocol(cfg, splits, experiment)
    protocol["trained_by"] = dict(
        TRAINED_BY, model=model, checkpoint=str(source), settings_from=provenance, settings=dict(settings),
    )
    metrics = released_metrics(run_dir, model)
    if metrics is not None:
        protocol["released_report_metrics"] = metrics
    revision = code_revision()
    protocol["history"] = [
        {"event": "adopted", "from": str(source), "git_revision": revision},
        {"event": "finished", "epochs": epochs, "max_epochs": epochs, "git_revision": revision},
    ]
    save_checkpoint(
        last, model=experiment.model, epoch=epochs - 1, global_step=epochs * steps_per_epoch,
        metrics={}, include_rng_state=False,
    )
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    hydra_dir = run_dir / ".hydra"
    hydra_dir.mkdir(exist_ok=True)
    OmegaConf.save(cfg, hydra_dir / "config.yaml")
    return {
        "run_dir": str(run_dir), "seed": seed, "parameters": protocol["real_parameter_count"],
        "epochs": epochs, "settings": settings, "settings_from": provenance, "released_report_metrics": metrics,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), default=DEFAULT_MODEL, help="the registry entry the runs trained")
    parser.add_argument("--root", default=None, help="directory holding seed<N>/ (default: outputs/.../<arm of --model>)")
    parser.add_argument("--logs", default=str(LOGS), help="where <arm>_seed<N>.log lives, for the argv banner")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="seeds to adopt (default: every seed<N> found)")
    parser.add_argument("--override", action="append", default=[], help="a compose override, e.g. data.dataset_file=...")
    parser.add_argument("--force", action="store_true", help="rewrite an existing adoption")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    arm = MODELS[args.model][1]
    root = pathlib.Path(args.root) if args.root else OUTPUTS / arm
    if args.seeds is None:
        seeds = sorted(int(p.name[4:]) for p in root.glob("seed*") if p.is_dir() and p.name[4:].isdigit())
    else:
        seeds = list(args.seeds)
    if not seeds:
        print(f"no seed directories under {root}", file=sys.stderr)
        return 1
    for seed in seeds:
        run_dir = root / f"seed{seed}"
        log_path = pathlib.Path(args.logs) / f"{arm}_seed{seed}.log"
        text = log_path.read_text(errors="replace") if log_path.is_file() else None
        report = adopt(run_dir, seed, log_text=text, overrides=list(args.override), force=args.force, model=args.model)
        released = report["released_report_metrics"]
        summary = "" if released is None else (
            f"; released report split: IoU {released.get('mean IoU mean', float('nan')):.4f}"
            f" Acc {released.get('mean Accuracy mean', float('nan')):.4f} loss {released.get('loss mean', float('nan')):.4f}"
        )
        print(
            f"{run_dir}: adopted {report['parameters']} parameters after {report['epochs']} epochs "
            f"({report['settings']} from {report['settings_from']}){summary}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
