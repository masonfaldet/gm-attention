#!/usr/bin/env bash
# Progress of a study's runs, read from each run's metrics.jsonl (written and
# closed per record by MetricLogger, so never behind the console log).
#
# Usage: progress.sh <run_root>     e.g. outputs/SWE_256x512_ZeroShot
#        progress.sh outputs/SWE_256x512_Ablation
#        progress.sh outputs/S2D3DS_128x256_Segmentation
set -euo pipefail
cd "$(dirname "$0")/.."

uv run python - "${1:?run root, e.g. outputs/SWE_256x512_ZeroShot}" <<'PY'
import json, pathlib, sys, time
from omegaconf import OmegaConf

root = pathlib.Path(sys.argv[1])
# Run directories, not metrics files: the run directory exists from startup,
# metrics.jsonl only after the first epoch.
run_dirs = sorted(d for d in root.glob("*/seed*") if d.is_dir())
if not run_dirs:
    print(f"no runs under {root} yet")
    raise SystemExit(0)

def budget(run_dir):
    """The run's epoch budget from its Hydra config; ``None`` when unknown."""
    config = run_dir / ".hydra" / "config.yaml"
    if not config.exists():
        return None
    try:
        return int(OmegaConf.load(config).training.max_epochs)
    except Exception:
        return None

# The monitored validation metric, read off the records: paper_squared_l2
# (shallow water) or cross_entropy (segmentation, iou shown beside it).
MONITORED = ("field_mse", "paper_squared_l2", "cross_entropy")

def monitored(row):
    for name in MONITORED:
        if name in row:
            return name
    raise KeyError(f"no monitored metric among {MONITORED} in {sorted(row)}")

print(f"{'arm':<28}{'seed':<6}{'epoch':>9}{'val loss':>15}"
      f"{'best':>10}{'min/ep':>9}{'eta':>9}{'val IoU':>9}")
print("-" * 95)

incomplete = 0
written = []
for run_dir in run_dirs:
    path = run_dir / "logs" / "metrics.jsonl"
    arm, seed = run_dir.parts[-2], run_dir.parts[-1]
    total = budget(run_dir)
    rows = (
        [json.loads(line) for line in path.open()] if path.is_file() else []
    )
    val = [r for r in rows if r["split"] == "val"]
    if not val:
        # Started, but no epoch has completed yet.
        epoch = f"0/{total}" if total else "starting"
        print(f"{arm:<28}{seed:<6}{epoch:>9}{'(epoch 0)':>15}")
        incomplete += 1
        continue
    written.append(path)
    scores = [r[monitored(r)] for r in val]
    done = len(val)
    # Wall-clock per epoch from the records' own timestamps.
    if len(val) > 1:
        rate = (val[-1]["time"] - val[0]["time"]) / (len(val) - 1) / 60
        eta = f"{rate * (total - done) / 60:.1f}h" if total and done < total else "--"
    else:
        rate, eta = float("nan"), "--"
    epoch = f"{done}/{total}" if total else str(done)
    if not total or done < total:
        incomplete += 1
    iou = f"{val[-1]['iou']:>9.4f}" if "iou" in val[-1] else ""
    print(f"{arm:<28}{seed:<6}{epoch:>9}{scores[-1]:>15.5f}"
          f"{min(scores):>10.5f}{rate:>9.1f}{eta:>9}{iou}")

metrics_seen = sorted({monitored(json.loads(p.open().readline())) for p in written})
if metrics_seen:
    print(f"\nval loss = {' / '.join(metrics_seen)}")
print(f"\n{len(run_dirs)} runs, {incomplete} still going", end="")
if written:
    age = (time.time() - max(p.stat().st_mtime for p in written)) / 60
    print(f"  |  last metric written {age:.1f} min ago")
    if age > 30 and incomplete:
        print("  warning: nothing written in 30+ min -- check the worker logs and the console logs")
else:
    print("  |  no epoch finished yet")
PY
