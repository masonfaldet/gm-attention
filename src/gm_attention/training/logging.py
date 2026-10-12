"""Training-run logging: metrics to disk.

``MetricLogger`` appends one JSON line per call to
``<log_dir>/metrics.jsonl`` and mirrors a human-readable line to the console.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Protocol

_console = logging.getLogger("gm_attention.training")
if not _console.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    _console.addHandler(_handler)
    _console.setLevel(logging.INFO)
    _console.propagate = False  # no duplicate line through the root logger


class Logger(Protocol):
    """Structural type every logger in this module satisfies."""

    def log(
        self,
        metrics: dict[str, float],
        *,
        step: int,
        epoch: int | None = None,
        split: str = "train",
    ) -> None:
        """Record one epoch's averaged metrics for the given split."""
        ...

    def close(self) -> None:
        """Release any resources the logger holds open."""
        ...


class MetricLogger:
    """Appends one JSON line per call to ``<log_dir>/metrics.jsonl``.

    Parameters
    ----------
    log_dir
        Directory to write ``metrics.jsonl`` into; created if needed.
    """

    def __init__(self, log_dir: str | Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.log_dir / "metrics.jsonl"

    def log(
        self,
        metrics: dict[str, float],
        *,
        step: int,
        epoch: int | None = None,
        split: str = "train",
    ) -> None:
        """Append ``metrics`` as one record, and print a summary line."""
        record = {
            "time": time.time(),
            "step": step,
            "epoch": epoch,
            "split": split,
            **metrics,
        }

        with self._path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")

        summary = ", ".join(f"{name}={value:.4g}" for name, value in metrics.items())
        _console.info("[%s] epoch=%s step=%d %s", split, epoch, step, summary)

    def close(self) -> None:
        """No open resources to release; present for the ``Logger`` protocol."""
