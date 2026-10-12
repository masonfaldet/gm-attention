"""The benchmark evaluation's discovery and seed aggregation.

The bundle script ``SWE_128x256_Benchmark/evaluate.py`` names the arms and
wraps ``display_name`` and ``report``; discovery, the zero-mean twins and
the seed aggregation are the shared implementation in
:mod:`experiments.shallow_water_paper.benchmark_evaluation`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("swe_evaluate", ROOT / "SWE_128x256_Benchmark" / "evaluate.py")
assert spec is not None and spec.loader is not None
evaluate = importlib.util.module_from_spec(spec)
sys.modules["swe_evaluate"] = evaluate
spec.loader.exec_module(evaluate)

from experiments.shallow_water_paper.benchmark_evaluation import (
    aggregate,
    discover_checkpoints,
    with_zero_mean_variants,
)
from experiments.shallow_water_paper.compare_spectral import (
    ArmResult,
    RolloutStep,
)


def _result(label: str, value: float) -> ArmResult:
    metrics = {"paper_squared_l2": value, "geopotential_rmse": value ** 0.5, "paper_squared_l2_L4": 0.9 * value}
    step = RolloutStep(step=1, hours=0.5, rollout=dict(metrics), one_step=dict(metrics), persistence={"paper_squared_l2": 1.0, "paper_squared_l2_L4": 0.9}, D=[0.5 * value] * 5, coverage=1.0)
    return ArmResult(label=label, kind="k", grid="g", n_nodes=10, parameters=100, condition=2.0,
                     metrics=metrics, D=[value / 5] * 5, paper_l1=value, paper_l2=value, rollout=[step])


def test_aggregate_reports_mean_and_sd_over_seeds_only_for_that_arm() -> None:
    results = [_result("a/seed0", 1.0), _result("a/seed1", 3.0), _result("b/seed0", 10.0)]
    a = aggregate(results, "a")
    assert a["seeds"] == 2
    assert a["metrics"]["paper_squared_l2"] == pytest.approx((2.0, 2.0 ** 0.5))
    assert a["D"] == pytest.approx([0.4] * 5)
    assert a["rollout"][0]["rollout"]["paper_squared_l2"] == pytest.approx((2.0, 2.0 ** 0.5))
    assert a["rollout"][0]["persistence"]["paper_squared_l2"] == pytest.approx((1.0, 0.0))
    b = aggregate(results, "b")
    assert b["seeds"] == 1 and b["metrics"]["paper_squared_l2"] == pytest.approx((10.0, 0.0))
    with pytest.raises(ValueError):
        aggregate(results, "c")


def test_discovery_finds_only_seeds_with_a_best_checkpoint(tmp_path: Path) -> None:
    for seed in (0, 2):
        (tmp_path / "arm" / f"seed{seed}" / "checkpoints").mkdir(parents=True)
        (tmp_path / "arm" / f"seed{seed}" / "checkpoints" / "best.pt").write_bytes(b"")
    (tmp_path / "arm" / "seed1" / "checkpoints").mkdir(parents=True)  # no best.pt
    found = discover_checkpoints(tmp_path, {"arm": "arm"})
    assert [seed for seed, _ in found["arm"]] == [0, 2]
    with pytest.raises(FileNotFoundError):
        discover_checkpoints(tmp_path, {"missing": "nowhere"})


def test_end_to_end_on_two_tiny_seeds(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Two one-epoch seeds of a tiny GM arm laid out as the benchmark
    lays out its runs; the evaluation discovers, loads, scores and
    aggregates them and its report prints every table.

    The arm is the depth-1 GM arm (model group ``d1_attention_full``)
    composed from the 256x512 ablation's ``gm_attention_full`` primary,
    whose one-hour lead the rollout-hours assertion expects; the benchmark's
    own ``gm_attention_depth1`` is the same arm on the 128x256 data."""
    import os

    import torch
    from hydra import compose, initialize
    from omegaconf import OmegaConf

    from experiments.shallow_water_paper.compare_spectral import evaluate_arms, load_arm
    from experiments.shallow_water_paper.spectral import full_bandwidth
    from experiments.shallow_water_paper.train import main as train_main

    tiny = [
        "grid.subdivisions=2", "device=cpu", "training.batch_size=2", "training.max_epochs=1",
        "data.nlat=32", "data.nlon=64", "data.solver_substeps=1", "data.num_train_examples=4",
        "data.num_val_examples=2", "data.normalization_examples=1",
    ]
    previous = Path.cwd()
    for seed in (0, 1):
        run = tmp_path / "arm" / f"seed{seed}"
        run.mkdir(parents=True)
        with initialize(version_base=None, config_path="../SWE_256x512_Ablation/configs"):
            cfg = compose(config_name="gm_attention_full", overrides=[*tiny, f"seed={seed}"])
        (run / ".hydra").mkdir()
        OmegaConf.save(cfg, run / ".hydra" / "config.yaml")
        os.chdir(run)
        try:
            train_main(cfg)
        finally:
            os.chdir(previous)

    found = discover_checkpoints(tmp_path, {"arm": "arm"})
    device = torch.device("cpu")
    loaded = [load_arm(f"arm/seed{seed}", path, device, None) for seed, path in found["arm"]]
    bandwidth = full_bandwidth(32)
    results, coverage = evaluate_arms(loaded, bandwidth=bandwidth, num_examples=2, split="test", device=device, steps=2)
    summary = {"arm": aggregate(results, "arm")}

    assert summary["arm"]["seeds"] == 2
    assert len(summary["arm"]["D"]) == bandwidth + 1
    assert [s["hours"] for s in summary["arm"]["rollout"]] == pytest.approx([1.0, 2.0])
    mean, sd = summary["arm"]["metrics"][f"paper_squared_l2_L{bandwidth}"]
    assert mean > 0 and sd >= 0
    # the persistence baseline is arm-independent, so its seed sd vanishes
    assert summary["arm"]["rollout"][0]["persistence"]["paper_squared_l2"][1] == pytest.approx(0.0)

    evaluate.report(summary, bandwidth, coverage, "test", 2)
    out = capsys.readouterr().out
    assert "single step" in out and "autoregressive rollout" in out and "D(l) by band" in out


def test_zero_mean_twins_follow_their_arm_and_share_its_checkpoints(tmp_path: Path) -> None:
    checkpoints = {"a": [(0, tmp_path / "a0")], "b": [(1, tmp_path / "b1")]}
    out = with_zero_mean_variants(checkpoints, ["a"])
    assert list(out) == ["a", "a_zero_mean", "b"]
    assert out["a"] == (checkpoints["a"], False) and out["a_zero_mean"] == (checkpoints["a"], True)
    assert out["b"] == (checkpoints["b"], False)
    assert evaluate.display_name("gm_attention_depth1_zero_mean") == "GM-Attention, depth-1, zero-mean update"
    with pytest.raises(SystemExit):
        with_zero_mean_variants(checkpoints, ["c"])
    # aggregation keys on the label's arm prefix, so a twin never mixes with its base arm
    results = [_result("a/seed0", 1.0), _result("a_zero_mean/seed0", 0.5)]
    assert aggregate(results, "a")["metrics"]["paper_squared_l2"] == pytest.approx((1.0, 0.0))
    assert aggregate(results, "a_zero_mean")["metrics"]["paper_squared_l2"] == pytest.approx((0.5, 0.0))
