"""Tests for gm_attention.training.logging."""

from __future__ import annotations

import json
from pathlib import Path

from gm_attention.training.logging import MetricLogger

# ==========================================================================
# MetricLogger
# ==========================================================================


def test_log_writes_one_json_line_with_the_given_fields(tmp_path: Path) -> None:
    logger = MetricLogger(tmp_path)
    logger.log({"loss": 0.5, "accuracy": 0.9}, step=10, epoch=2, split="val")

    lines = (tmp_path / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["step"] == 10
    assert record["epoch"] == 2
    assert record["split"] == "val"
    assert record["loss"] == 0.5
    assert record["accuracy"] == 0.9
    assert "time" in record


def test_repeated_log_calls_append_rather_than_overwrite(tmp_path: Path) -> None:
    logger = MetricLogger(tmp_path)
    logger.log({"loss": 1.0}, step=0)
    logger.log({"loss": 0.5}, step=1)

    lines = (tmp_path / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["loss"] == 1.0
    assert json.loads(lines[1])["loss"] == 0.5


def test_creates_the_log_directory_if_missing(tmp_path: Path) -> None:
    log_dir = tmp_path / "nested" / "run"
    MetricLogger(log_dir)

    assert log_dir.is_dir()


def test_close_does_not_raise(tmp_path: Path) -> None:
    MetricLogger(tmp_path).close()
