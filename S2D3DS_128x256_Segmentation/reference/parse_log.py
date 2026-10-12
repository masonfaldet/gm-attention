"""The released script's per-epoch stdout as ``metrics.jsonl`` rows.

The released script prints one summary per epoch and logs nothing else
locally (its Weights & Biases run is disabled). Each

    Epoch 12 summary:
    time taken: 61.02
    accumulated training loss: 0.98
    relative validation loss: 1.01
    mean IoU: 0.51
    mean Accuracy: 0.93

becomes a train row and a val row in the bundle's log format (``epoch``,
``split``, ``cross_entropy``, ``iou``, ``accuracy``) under the run
directory's ``logs/metrics.jsonl``. The script prints durations, not
timestamps, so ``time`` is reconstructed: the last summary is anchored at
the log file's modification time and earlier epochs are placed by their
durations. The epoch budget in the runner's banner (``--num_epochs``) is
written to the run directory's ``.hydra/config.yaml`` as
``training.max_epochs``. The script's per-epoch "mean IoU" pools each
validation batch of 8 before averaging (a per-batch, not per-panorama,
mean), and its "validation loss" is the released "test" split, this
bundle's ``val``.

Usage, from the repository root::

    uv run python S2D3DS_128x256_Segmentation/reference/parse_log.py            # every *_released log
    uv run python S2D3DS_128x256_Segmentation/reference/parse_log.py logs/S2D3DS_128x256_Segmentation/s2_segformer_released_seed0.log
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

LOGS = pathlib.Path("logs/S2D3DS_128x256_Segmentation")
OUTPUTS = pathlib.Path("outputs/S2D3DS_128x256_Segmentation")
EPOCH = re.compile(r"^Epoch (\d+) summary:")
BUDGET = re.compile(r"^argv: .*--num_epochs (\d+)")
FIELDS = {
    "time taken": re.compile(r"^time taken: ([-+0-9.eE]+|nan|inf)"),
    "train": re.compile(r"^accumulated training loss: ([-+0-9.eE]+|nan|inf)"),
    "val": re.compile(r"^relative validation loss: ([-+0-9.eE]+|nan|inf)"),
    "iou": re.compile(r"^mean IoU: ([-+0-9.eE]+|nan|inf)"),
    "accuracy": re.compile(r"^mean Accuracy: ([-+0-9.eE]+|nan|inf)"),
}


def parse(text: str, end_time: float | None = None) -> list[dict]:
    """The rows, with ``time`` counted back from ``end_time`` (the moment
    the last summary was written) or, without one, up from zero."""
    rows: list[dict] = []
    epoch: int | None = None
    values: dict[str, float] = {}
    clock = 0.0

    def flush() -> None:
        nonlocal clock
        if epoch is None or "val" not in values:
            return
        clock += values.get("time taken", 0.0)
        rows.append({"time": clock, "epoch": epoch, "split": "train", "cross_entropy": values.get("train"),
                     "epoch_time": values.get("time taken")})
        rows.append({"time": clock, "epoch": epoch, "split": "val", "cross_entropy": values["val"],
                     "iou": values.get("iou"), "accuracy": values.get("accuracy")})

    for line in text.splitlines():
        match = EPOCH.match(line)
        if match:
            flush()
            epoch, values = int(match.group(1)), {}
            continue
        for key, pattern in FIELDS.items():
            match = pattern.match(line)
            if match:
                values[key] = float(match.group(1))
    flush()
    if end_time is not None and rows:
        shift = end_time - rows[-1]["time"]
        for row in rows:
            row["time"] += shift
    return rows


def budget(text: str) -> int | None:
    """The epoch budget the runner's banner recorded, if any."""
    for line in text.splitlines():
        match = BUDGET.match(line)
        if match:
            return int(match.group(1))
    return None


def run_directory(log: pathlib.Path) -> pathlib.Path:
    """``<arm>_seed<N>.log`` -> ``outputs/.../<arm>/seed<N>``."""
    stem = log.stem
    arm, _, seed = stem.rpartition("_seed")
    return OUTPUTS / arm / f"seed{seed}"


def main(argv: list[str]) -> int:
    logs = [pathlib.Path(a) for a in argv] or sorted(LOGS.glob("*_released_seed*.log"))
    if not logs:
        print("no logs found", file=sys.stderr)
        return 1
    for log in logs:
        text = log.read_text(errors="replace")
        rows = parse(text, end_time=log.stat().st_mtime)
        run = run_directory(log)
        if not rows:
            # No summaries yet: a run directory written here would make the
            # launch verb refuse a rerun.
            print(f"{log}: no epoch summaries yet; nothing written")
            continue
        target = run / "logs" / "metrics.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(json.dumps(row) + "\n" for row in rows))
        epochs_planned = budget(text)
        config = run / ".hydra" / "config.yaml"
        if epochs_planned is not None and not config.exists():
            # The one key the progress script reads from a run's config.
            config.parent.mkdir(parents=True, exist_ok=True)
            config.write_text(f"training:\n  max_epochs: {epochs_planned}\n")
        epochs = len(rows) // 2
        last = rows[-1] if rows else None
        summary = "" if last is None else f", epoch {last['epoch'] + 1}: val CE {last['cross_entropy']:.3f} IoU {last['iou']:.3f}"
        print(f"{log} -> {target}: {epochs} epochs{summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
