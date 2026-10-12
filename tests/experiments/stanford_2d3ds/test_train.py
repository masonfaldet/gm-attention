"""Config, wiring, a tiny end-to-end run of both arms, and the evaluation on it."""

from __future__ import annotations

import dataclasses
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from experiments.stanford_2d3ds.data import dataset_digest
from experiments.stanford_2d3ds.evaluation import (
    aggregate,
    evaluate_run,
    hydra_config_drift,
    run_evaluation,
)
from experiments.stanford_2d3ds.train import (
    MONITOR_METRIC,
    PROTOCOL_FILENAME,
    SegmentationScorer,
    build_experiment,
    build_loaders,
    open_splits,
    verify_or_save_protocol,
)
from experiments.stanford_2d3ds.train import main as train_main
from tests.support.synthetic_2d3ds import write_synthetic_dataset

ROOT = Path(__file__).resolve().parents[3]
CONFIGS = ROOT / "S2D3DS_128x256_Segmentation" / "configs"
# The shipped arms at two stages, small enough for a one-epoch run on the CPU.
TINY_SOS2 = [  # the reported model: gravity-conditioned, icosphere stages, the fusing decoder
    "model.architecture.multiplicities=[2,4]", "model.architecture.heads=[1,1]", "model.architecture.depths=[1,1]",
    "model.architecture.readout_channels=[8,16]", "model.icosphere_levels=[3,2]", "training.panoramas_per_forward=2",
]
TINY_SOS2_SCALAR = [*TINY_SOS2, "model.architecture.stage_field_types=[{0:4},{0:8}]"]  # its scalar ablation
TINY_REF = ["model.architecture.embed_dims=[4,8]", "model.architecture.heads=[1,2]", "model.architecture.depths=[1,1]"]
ARMS = {"sos2": ("sos2_segformer", TINY_SOS2), "sos2-scalar": ("sos2_segformer_scalar", TINY_SOS2_SCALAR),
        "reference": ("s2_segformer", TINY_REF)}


@pytest.fixture(scope="module")
def dataset_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_synthetic_dataset(tmp_path_factory.mktemp("data") / "synthetic.h5", num_samples=16, shape=(32, 64), num_classes=4)


def _cfg(dataset_file: Path, primary: str = "sos2_segformer", *overrides: str) -> DictConfig:
    tiny = [
        f"data.dataset_file={dataset_file}", "data.img_size=[32,64]", "data.split_ratios=[0.5,0.25,0.25]", "data.num_workers=0",
        "device=cpu", "training.batch_size=2", "training.max_epochs=1",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        return compose(config_name=primary, overrides=[*tiny, *overrides])


def test_the_config_is_the_reported_run() -> None:
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        cfg = compose(config_name="sos2_segformer")
    t = cfg.training
    assert (t.optimizer, t.weight_decay, t.grad_clip_norm, t.batch_size, t.max_epochs) == ("adamw", 0.1, 4.0, 8, 200)
    assert (t.scheduler.kind, t.scheduler.eta_min, t.early_stopping_patience, t.label_smoothing) == ("cosine", 1e-6, None, 0.0)
    assert t.lr == 3e-3 and t.panoramas_per_forward == 4  # the bundle README's settings
    a = cfg.model.architecture
    assert list(a.heads) == [1, 1, 2, 4] and list(a.depths) == [3, 4, 6, 3]
    assert (a.drop_path_rate, a.window, a.nonlinearity, a.readout) == (0.2, "hard", "scalar_gate_double_sigmoid", "convolution")
    assert list(a.multiplicities) == [2, 4, 6, 8] and list(a.radial_hidden_features) == [14]
    assert cfg.model.readout_rows == 1.5 and list(a.readout_channels) == [8, 16, 32, 32]
    assert cfg.model.grid_internal == "icosphere" and list(cfg.model.icosphere_levels) == [5, 4, 3, 2]
    assert (cfg.data.split_seed, list(cfg.data.split_ratios), list(cfg.data.img_size)) == (333, [0.95, 0.025, 0.025], [128, 256])


def test_training_never_touches_the_report_split(dataset_file: Path) -> None:
    """The loaders draw from train and val only; report is scored once, at the end."""
    cfg = _cfg(dataset_file, "sos2_segformer", *TINY_SOS2, "training.panoramas_per_forward=1")  # one panorama per forward: the accumulation is the batch
    splits = open_splits(cfg)
    train_loader, val_loader, accumulation = build_loaders(cfg, splits, batched=False)
    seen = set(train_loader.dataset.indices) | set(val_loader.dataset.indices)  # type: ignore[attr-defined]
    assert not seen & set(splits.indices["report"])
    assert not set(train_loader.dataset.indices) & set(val_loader.dataset.indices)  # type: ignore[attr-defined]
    assert accumulation == cfg.training.batch_size


def test_pixel_accuracy_ignores_masked_pixels_and_is_class_count_free() -> None:
    scorer = SegmentationScorer.build((8, 16), grid="equiangular", label_smoothing=0.0, device=torch.device("cpu"))
    target = torch.randint(0, 3, (1, 8, 16))
    perfect = torch.nn.functional.one_hot(target, 3).permute(0, 3, 1, 2).float() * 4.0
    assert scorer.pixel_accuracy(perfect, target).item() == pytest.approx(1.0)
    wrong = perfect.roll(1, dims=1)
    assert scorer.pixel_accuracy(wrong, target).item() == pytest.approx(0.0)
    masked = target.clone()
    masked[:, :4] = -100
    half_wrong = perfect.clone()
    half_wrong[:, :, :4] = wrong[:, :, :4]  # wrong only where masked
    assert scorer.pixel_accuracy(half_wrong, masked).item() == pytest.approx(1.0)
    # Padding the class axis with never-predicted classes changes the paper's
    # accuracy of an imperfect prediction (1 - 2U/C) but not the pixel accuracy.
    partly_wrong = perfect.clone()
    partly_wrong[:, :, :2] = wrong[:, :, :2]
    padded = torch.cat([partly_wrong, torch.full((1, 5, 8, 16), -9.0)], dim=1)
    assert scorer.pixel_accuracy(padded, target).item() == pytest.approx(scorer.pixel_accuracy(partly_wrong, target).item())
    narrow = scorer(partly_wrong, target).metrics["accuracy"]
    wide = scorer(padded, target).metrics["accuracy"]
    assert narrow != pytest.approx(wide)
    assert (1.0 - wide) * 8 == pytest.approx((1.0 - narrow) * 3, rel=1e-4)


def test_the_released_metrics_are_functions_of_the_correct_area_alone() -> None:
    """With no pixel ignored every misclassified pixel is one false positive
    and one false negative, so the micro IoU is P / (2 - P) and the accuracy
    1 - 2 (1 - P) / C for the correctly classified area fraction P; the
    confusion matrix reproduces the released per-class areas and gives the
    class-averaged IoU the released metrics do not."""
    from experiments.stanford_2d3ds.metrics import _get_stats_multiclass

    scorer = SegmentationScorer.build((16, 32), grid="equiangular", label_smoothing=0.0, device=torch.device("cpu"))
    generator = torch.Generator().manual_seed(0)
    num_classes = 6
    target = torch.randint(0, num_classes, (2, 16, 32), generator=generator)
    logits = torch.randn(2, num_classes, 16, 32, generator=generator)
    logits[0, :, :8] = torch.nn.functional.one_hot(target[0, :8], num_classes).permute(2, 0, 1).float() * 9  # half right
    out = scorer(logits, target).metrics
    p = scorer.pixel_accuracy(logits, target).item()
    confusion = scorer.confusion(logits, target)
    p_per_item = confusion.diagonal(dim1=-2, dim2=-1).sum(-1) / confusion.sum(dim=(-2, -1))
    # The released IoU pools the batch's areas (mean over items of tp etc.).
    P = p_per_item.mean().item()
    assert out["iou"] == pytest.approx(P / (2 - P), rel=1e-5)
    assert out["accuracy"] == pytest.approx(1 - 2 * (1 - P) / num_classes, rel=1e-5)
    assert p == pytest.approx(P, rel=1e-5)
    tp, fp, fn, _tn = _get_stats_multiclass(logits.argmax(1), target, num_classes, scorer.quadrature, -100)
    torch.testing.assert_close(confusion.diagonal(dim1=-2, dim2=-1), tp, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(confusion.sum(-2) - confusion.diagonal(dim1=-2, dim2=-1), fp, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(confusion.sum(-1) - confusion.diagonal(dim1=-2, dim2=-1), fn, rtol=1e-5, atol=1e-7)
    class_iou = scorer.class_iou(logits, target)
    torch.testing.assert_close(class_iou, tp / (tp + fp + fn), rtol=1e-5, atol=1e-7)
    assert out["mean_class_iou"] == pytest.approx(class_iou.nanmean(-1).mean().item(), rel=1e-5)
    # A class absent from target and prediction has no IoU; a panorama with
    # no valid pixel has no pixel accuracy, as it has no released IoU.
    narrow = torch.randint(0, 2, (1, 16, 32), generator=generator)
    wide = torch.nn.functional.one_hot(narrow, num_classes).permute(0, 3, 1, 2).float()
    assert torch.isnan(scorer.class_iou(wide, narrow)[0, 2:]).all() and not torch.isnan(scorer.class_iou(wide, narrow)[0, :2]).any()
    assert scorer.mean_class_iou(wide, narrow).item() == pytest.approx(1.0)
    ignored = torch.full((1, 16, 32), -100)
    assert torch.isnan(scorer.pixel_accuracy(wide, ignored)) and torch.isnan(scorer.mean_class_iou(wide, ignored))


def test_the_cli_resolves_a_relative_dataset_path_from_the_launch_directory(dataset_file: Path, tmp_path: Path) -> None:
    """hydra.main changes into the run directory before training; a relative
    data.dataset_file must still mean 'relative to where I launched'."""
    launch = tmp_path / "launch"
    launch.mkdir()
    shutil.copy(dataset_file, launch / "synthetic.h5")
    run_dir = tmp_path / "run"
    command = [
        sys.executable, "-m", "experiments.stanford_2d3ds.train",
        "-cp", str(CONFIGS), "-cn", "sos2_segformer", f"hydra.searchpath=[file://{ROOT}/configs]",
        "data.dataset_file=synthetic.h5", "data.img_size=[32,64]", "data.split_ratios=[0.5,0.25,0.25]", "data.num_workers=0",
        "device=cpu", "training.batch_size=2", "training.max_epochs=1", f"hydra.run.dir={run_dir}",
        *TINY_SOS2,
    ]
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    completed = subprocess.run(command, cwd=launch, env=env, capture_output=True, text=True, timeout=900, check=False)
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert (run_dir / "checkpoints" / "last.pt").exists()
    protocol = json.loads((run_dir / PROTOCOL_FILENAME).read_text())
    assert Path(protocol["dataset_file"]).is_absolute()
    assert Path(protocol["dataset_file"]).resolve() == (launch / "synthetic.h5").resolve()


def test_a_wrong_img_size_is_refused(dataset_file: Path) -> None:
    with pytest.raises(ValueError, match="img_size"):
        open_splits(_cfg(dataset_file, "sos2_segformer", "data.img_size=[128,256]"))


@pytest.mark.parametrize("arm", list(ARMS), ids=list(ARMS))
def test_a_tiny_run_trains_writes_its_protocol_and_evaluates(dataset_file: Path, tmp_path: Path, arm: str) -> None:
    run = tmp_path / "seed0"
    run.mkdir()
    primary, overrides = ARMS[arm]
    cfg = _cfg(dataset_file, primary, *overrides)
    previous = Path.cwd()
    os.chdir(run)
    try:
        train_main(cfg)
    finally:
        os.chdir(previous)
    (run / ".hydra").mkdir()
    OmegaConf.save(cfg, run / ".hydra" / "config.yaml")

    protocol = json.loads((run / PROTOCOL_FILENAME).read_text())
    assert protocol["split_sizes"] == {"train": 8, "val": 4, "report": 4}
    assert protocol["num_classes"] == 4 and protocol["split_seed"] == 333
    assert (run / "checkpoints" / "last.pt").exists() and (run / "checkpoints" / "best.pt").exists()
    rows = [json.loads(line) for line in (run / "logs" / "metrics.jsonl").open()]
    assert {row["split"] for row in rows} == {"train", "val"}
    assert {MONITOR_METRIC, "iou", "accuracy"} <= rows[-1].keys()

    # The evaluation reproduces the logged validation loss exactly: the last
    # checkpoint is the final model and the val loader is deterministic. The
    # GM arm validates one panorama at a time, so its IoU matches too; the
    # batched reference pools the metric over each validation batch, as the
    # released script does, which is not the per-sample mean.
    result = evaluate_run(run, arm="arm", seed=0, checkpoint="last", split="val", device=torch.device("cpu"))
    logged = [row for row in rows if row["split"] == "val"][-1]
    assert result.summary()["cross_entropy_mean"] == pytest.approx(logged[MONITOR_METRIC], rel=1e-5)
    if arm != "reference":
        assert result.summary()["iou_mean"] == pytest.approx(logged["iou"], rel=1e-5)
    else:
        assert result.summary()["iou_mean"] == pytest.approx(logged["iou"], rel=0.05)
    assert len(result.samples) == 4

    (tmp_path / "seed7" / "checkpoints").mkdir(parents=True)  # a run still in its first epoch
    payload = run_evaluation(tmp_path.parent, {"arm": tmp_path.name}, checkpoint="last", split="report", device=torch.device("cpu"))
    summary = payload["arms"]["arm"]["summary"]
    assert summary["seeds"] == [0] and summary["per_seed"]["0"]["num_samples"] == 4
    assert aggregate([result])["parameters"] == protocol["real_parameter_count"]

    # The protocol pins the data by content and the model and training by
    # their resolved configs, and remembers where it was made.
    assert protocol["dataset_sha256"] == dataset_digest(dataset_file) and len(protocol["dataset_sha256"]) == 64
    assert protocol["model_config"]["kind"] == cfg.model.kind
    assert "resume" not in protocol["training_config"] and "restart" not in protocol["training_config"]["scheduler"]
    assert protocol["training_config"]["max_epochs"] == int(cfg.training.max_epochs)
    assert isinstance(protocol["git_revision"], str)
    # The loop's end is on record, with the epochs it ran.
    assert [entry["event"] for entry in protocol["history"]] == ["finished"]
    assert protocol["history"][0]["epochs"] == int(cfg.training.max_epochs)
    # A fresh launch into a run directory is refused rather than restarting it.
    os.chdir(run)
    try:
        with pytest.raises(FileExistsError, match="already holds a run"):
            train_main(cfg)
    finally:
        os.chdir(previous)

    # A resume into this directory with different data is refused; so is one
    # with a different model or training protocol unless the change is
    # declared, and then it is written into the history. The same protocol
    # at the same revision verifies silently.
    with pytest.raises(ValueError, match="different data"):
        verify_or_save_protocol(dict(protocol, split_seed=1), run)
    with pytest.raises(ValueError, match="different data"):
        verify_or_save_protocol(dict(protocol, dataset_sha256="0" * 64), run)
    with pytest.raises(ValueError, match="different model or training"):
        verify_or_save_protocol(dict(protocol, model_seed=7), run)
    drifted = dict(protocol, model_seed=7, training_config={**protocol["training_config"], "lr": 1e-3})
    verify_or_save_protocol(drifted, run, allow_drift=True, completed_epochs=1)
    recorded = json.loads((run / PROTOCOL_FILENAME).read_text())
    assert recorded["model_seed"] == 7 and recorded["training_config"]["lr"] == 1e-3
    assert [entry["event"] for entry in recorded["history"]] == ["finished", "resume"]
    assert set(recorded["history"][1]["changed"]) == {"model_seed", "training_config"}
    assert recorded["history"][1]["changed"]["model_seed"] == {"previous": int(cfg.seed), "current": 7}
    assert recorded["history"][1]["completed_epochs"] == 1
    verify_or_save_protocol(drifted, run)  # now the saved protocol: verified, no new history
    assert json.loads((run / PROTOCOL_FILENAME).read_text())["history"] == recorded["history"]
    # Another revision advances the record once, not on every later verify.
    elsewhere = dict(drifted, git_revision="0123abcd")
    verify_or_save_protocol(elsewhere, run)
    verify_or_save_protocol(elsewhere, run)
    recorded = json.loads((run / PROTOCOL_FILENAME).read_text())
    assert recorded["git_revision"] == "0123abcd" and [e["event"] for e in recorded["history"]] == ["finished", "resume", "resume"]
    # The file's location and size are not its identity: a moved checkout resumes.
    verify_or_save_protocol(dict(drifted, git_revision="0123abcd", dataset_file="/elsewhere/synthetic.h5", dataset_size=1), run)
    (run / PROTOCOL_FILENAME).write_text(json.dumps(protocol, indent=2) + "\n")

    # Evaluation verifies the data's identity, not its shape: a protocol
    # recording another file's digest, class names or normalization is refused.
    for key, value in (
        ("dataset_sha256", "0" * 64),
        ("class_labels", [*protocol["class_labels"][1:], "other"]),
        ("normalization", {"mean": [0.0] * 3, "std": [1.0] * 3}),
    ):
        (run / PROTOCOL_FILENAME).write_text(json.dumps(dict(protocol, **{key: value})) + "\n")
        with pytest.raises(ValueError, match=f"{key} differs"):
            evaluate_run(run, arm="arm", seed=0, checkpoint="last", split="val", device=torch.device("cpu"))
    (run / PROTOCOL_FILENAME).write_text(json.dumps({k: v for k, v in protocol.items() if k != "dataset_sha256"}) + "\n")
    with pytest.raises(ValueError, match="records no dataset_sha256"):
        evaluate_run(run, arm="arm", seed=0, checkpoint="last", split="val", device=torch.device("cpu"))
    (run / PROTOCOL_FILENAME).write_text(json.dumps(protocol, indent=2) + "\n")

    # The evaluation builds from the protocol, not from .hydra/config.yaml,
    # which any relaunch rewrites (a refused one included): a rewritten copy
    # changes nothing scored, and is reported as drift.
    assert result.complete and result.epoch + 1 == result.max_epochs == int(cfg.training.max_epochs)
    rewritten = OmegaConf.load(run / ".hydra" / "config.yaml")
    rewritten.training.max_epochs = 50
    if "window" in rewritten.model.get("architecture", {}):
        rewritten.model.architecture.window = "quartic"
    OmegaConf.save(rewritten, run / ".hydra" / "config.yaml")
    assert hydra_config_drift(run, protocol) == (["model_config", "training_config"] if "window" in rewritten.model.get("architecture", {}) else ["training_config"])
    again = evaluate_run(run, arm="arm", seed=0, checkpoint="last", split="val", device=torch.device("cpu"))
    assert again.complete and again.max_epochs == int(cfg.training.max_epochs)
    assert again.summary()["cross_entropy_mean"] == pytest.approx(result.summary()["cross_entropy_mean"], rel=1e-6)
    OmegaConf.save(cfg, run / ".hydra" / "config.yaml")

    # An unfinished run is skipped unless asked for, and then marked seed by
    # seed: both seeds are configured for 50 epochs; seed 0's protocol
    # records the loop's end (early stopping would too), seed 8's does not.
    # A seed without last.pt is unfinished whichever checkpoint is scored.
    unfinished = tmp_path / "seed8"
    shutil.copytree(run, unfinished)
    for directory, finished in ((run, True), (unfinished, False)):
        record = json.loads((directory / PROTOCOL_FILENAME).read_text())
        record["training_config"]["max_epochs"] = 50
        record["history"] = [e for e in record["history"] if finished or e["event"] != "finished"]
        (directory / PROTOCOL_FILENAME).write_text(json.dumps(record, indent=2) + "\n")
    headless = tmp_path / "seed9"
    shutil.copytree(run, headless)
    (headless / "checkpoints" / "last.pt").unlink()
    payload = run_evaluation(tmp_path.parent, {"arm": tmp_path.name}, checkpoint="last", split="report", device=torch.device("cpu"))
    assert payload["arms"]["arm"]["summary"]["seeds"] == [0] and payload["arms"]["arm"]["summary"]["complete"]
    best = run_evaluation(tmp_path.parent, {"arm": tmp_path.name}, checkpoint="best", split="report", device=torch.device("cpu"))
    assert best["arms"]["arm"]["summary"]["seeds"] == [0] and best["arms"]["arm"]["summary"]["complete"]
    payload = run_evaluation(tmp_path.parent, {"arm": tmp_path.name}, checkpoint="best", split="report",
                             device=torch.device("cpu"), allow_incomplete=True)
    summary = payload["arms"]["arm"]["summary"]
    assert summary["seeds"] == [0, 8, 9] and not summary["complete"] and summary["max_epochs"] == 50
    assert summary["unfinished_seeds"] == [
        {"seed": 8, "run_epoch": 1, "max_epochs": 50, "epoch": 1},
        {"seed": 9, "run_epoch": 1, "max_epochs": 50, "epoch": 1},
    ]
    marked = next(r for r in payload["arms"]["arm"]["runs"] if r["seed"] == 8)
    assert not marked["complete"] and marked["run_epoch"] == 0
    (run / PROTOCOL_FILENAME).write_text(json.dumps(protocol, indent=2) + "\n")
    # The bundle's seed set is a requirement, not a hope.
    with pytest.raises(ValueError, match="are required"):
        run_evaluation(tmp_path.parent, {"arm": tmp_path.name}, checkpoint="last", split="report",
                       device=torch.device("cpu"), require_seeds=[0, 1])
    with pytest.raises(ValueError, match="different parameter counts"):
        aggregate([result, dataclasses.replace(result, seed=1, parameters=result.parameters + 1)])
    with pytest.raises(ValueError, match="different samples"):
        aggregate([result, dataclasses.replace(result, seed=1, samples=result.samples[1:])])
    with pytest.raises(ValueError, match="different configured lengths"):
        aggregate([result, dataclasses.replace(result, seed=1, max_epochs=result.max_epochs + 1)])
    # A non-finite metric is counted, not averaged, and never reaches the JSON.
    poisoned = dataclasses.replace(result, samples=[dataclasses.replace(result.samples[0], pixel_accuracy=float("nan")), *result.samples[1:]])
    assert poisoned.summary()["pixel_accuracy_nonfinite"] == 1 and math.isfinite(poisoned.summary()["pixel_accuracy_mean"])
    assert aggregate([poisoned])["pixel_accuracy_nonfinite"] == 1


def test_the_arm_builds_on_icosphere_levels_with_the_chains_radii(dataset_file: Path) -> None:
    cfg = _cfg(dataset_file, "sos2_segformer", *TINY_SOS2)
    splits = open_splits(cfg)
    experiment = build_experiment(cfg, torch.device("cpu"), shape=splits.shape, num_classes=splits.num_classes)
    assert experiment.hierarchy is not None and experiment.radii is not None
    assert [level.describe() for level in experiment.hierarchy.levels] == ["32x64 equiangular", "icosphere L3", "icosphere L2"]
    assert experiment.model.attention_cutoffs == experiment.radii.attention
    assert experiment.model.merge_cutoffs == experiment.radii.merge
    assert experiment.model.readout_cutoffs == experiment.radii.readout
    assert cfg.model.grid_internal == "icosphere" and cfg.model.interpolate_spacings == 1.5


def test_the_fusion_arm_builds_its_lateral_transfers_on_icosphere_levels(dataset_file: Path) -> None:
    """The fusing decoder's top-down transfer runs between icosphere levels,
    at the coarser level's interpolation radius, and reaches every node of
    the finer one."""
    cfg = _cfg(dataset_file, "sos2_segformer", *TINY_SOS2)
    splits = open_splits(cfg)
    experiment = build_experiment(cfg, torch.device("cpu"), shape=splits.shape, num_classes=splits.num_classes)
    hierarchy = experiment.hierarchy
    assert hierarchy is not None and experiment.radii is not None
    assert [level.describe() for level in hierarchy.levels] == ["32x64 equiangular", "icosphere L3", "icosphere L2"]
    assert len(hierarchy.lateral_edges) == 1
    lateral = hierarchy.lateral_edges[0]
    fine = hierarchy.levels[1]
    assert torch.bincount(lateral.target_index, minlength=fine.n_nodes).min() > 0
    assert lateral.distances.max() <= experiment.radii.interpolate[1] + 1e-6
    assert experiment.model.decoder == "fusion"


def test_code_revision_reads_the_archive_marker_outside_a_repository(tmp_path: Path) -> None:
    """A tree copied without its .git but with a GIT_REVISION file: the
    protocol then names the commit rather than 'unknown'."""
    from experiments.stanford_2d3ds.train import code_revision

    assert code_revision(tmp_path) == "unknown"
    (tmp_path / "GIT_REVISION").write_text("0123abcd" * 5 + "\n")
    assert code_revision(tmp_path) == "0123abcd" * 5 + "+archive"
