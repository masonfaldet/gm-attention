"""The segmentation bundle composes to the released protocol and renders its table.

The bundle ships the five reported arms (README, "Arms"): the SO(S^2)-SegFormer,
its scalar-only ablation and the three released baselines. Run from the
repository root: the primaries' ``hydra.searchpath`` resolves ``file://configs``
against the working directory.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "S2D3DS_128x256_Segmentation"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tables = _load("s2d3ds_tables", BUNDLE / "tables.py")


def _compose(arm: str, *overrides: str, with_hydra: bool = False):
    with initialize_config_dir(version_base=None, config_dir=str(BUNDLE / "configs")):
        return compose(config_name=arm, overrides=list(overrides), return_hydra_config=with_hydra)


@pytest.mark.parametrize("arm", ["sos2_segformer", "sos2_segformer_scalar", "s2_segformer", "s2_unet", "r2_segformer"])
def test_every_arm_composes_the_released_protocol(arm: str) -> None:
    cfg = _compose(arm, "seed=2", with_hydra=True)
    d, t = cfg.data, cfg.training
    assert (d.split_seed, list(d.split_ratios), list(d.img_size), d.grid) == (333, [0.95, 0.025, 0.025], [128, 256], "equiangular")
    assert d.dataset_file.endswith("stanford_2d3ds_dataset_ds16.h5")
    assert (t.optimizer, t.weight_decay, t.grad_clip_norm, t.batch_size, t.max_epochs, t.scheduler.kind) == ("adamw", 0.1, 4.0, 8, 200, "cosine")
    assert t.early_stopping_patience is None
    assert cfg.hydra.run.dir == f"outputs/S2D3DS_128x256_Segmentation/{arm}/seed2"


def test_the_sos2_arm_is_the_matched_configuration() -> None:
    """The reported SO(S^2)-SegFormer against the released S^2 SegFormer it is
    matched to: the same depths and scale factor, four stages on icosphere
    levels 5, 4, 3, 2 with types 6 + 2 + 2, 12 + 4 + 4, 18 + 6 + 6, 24 + 8 + 8
    (README, "Arms"), at the reported rate with four panoramas per forward."""
    cfg, s2 = _compose("sos2_segformer"), _compose("s2_segformer").model.architecture
    m = cfg.model
    a = OmegaConf.to_container(m.architecture)
    assert a["_target_"].endswith("GmSegformer") and m.kind == "gm_segformer"
    assert a["depths"] == list(s2.depths) == [3, 4, 6, 3] and m.scale_factor == s2.scale_factor == 2
    assert a["base_field_type"] == {0: 3, 1: 1, 2: 1} and a["multiplicities"] == [2, 4, 6, 8]
    assert a["final_stage_scalars"] is None and "stage_field_types" not in a and "gravity_input" not in a
    assert a["heads"] == [1, 1, 2, 4] and a["radial_hidden_features"] == [14] and a["value_kind"] == "steerable"
    assert a["drop_path_rate"] == 0.2 and a["gradient_checkpointing"] is False and a["attention_batched"] is True
    assert a["readout"] == "convolution" and a["readout_channels"] == [8, 16, 32, 32]
    assert (a["gravity_conditioning"], a["gravity_projection"], a["gravity_readout"], a["gravity_readout_expansion"]) == (True, True, True, 2)
    assert (a["normalized_mix"], a["normalized_readout"], a["head_hidden"], a["decoder"], a["vertical_conditioning"]) == (True, True, 64, "fusion", True)
    assert (m.grid_internal, list(m.icosphere_levels), m.interpolate_spacings, m.readout_rows) == ("icosphere", [5, 4, 3, 2], 1.5, 1.5)
    assert (cfg.training.lr, cfg.training.panoramas_per_forward, cfg.data.num_workers) == (3.0e-3, 4, 2)


def test_the_reference_arm_is_the_registry_entry() -> None:
    a = _compose("s2_segformer").model.architecture
    assert a._target_.endswith("SphericalSegformer")
    assert list(a.embed_dims) == [16, 32, 64, 128] and list(a.kernel_shape) == [5, 4]
    assert (a.filter_basis_type, a.attention_mode, a.grid_internal, a.bias) == ("piecewise linear", "neighborhood", "equiangular", False)


def test_the_scalar_ablation_differs_from_the_sos2_arm_only_in_its_declared_switches() -> None:
    """Every stage scalar (14 / 26 / 38 / 48 rho_0, README) at the ablation's
    depths [3, 4, 7, 3] and drop-path 0.1, the per-stage gravity conditioning
    and projection off (they need directional channels) and the gravity field
    as input channels instead; nothing else changed. ``multiplicities`` differs
    too, but ``stage_field_types`` replaces the ladder it would scale (only its
    length counts, experiments/stanford_2d3ds/model.py), so it is inert here."""
    sos2, scalar = _compose("sos2_segformer"), _compose("sos2_segformer_scalar")
    a, b = OmegaConf.to_container(sos2.model.architecture), OmegaConf.to_container(scalar.model.architecture)
    changed = ("depths", "drop_path_rate", "multiplicities", "stage_field_types", "gravity_conditioning", "gravity_projection", "gravity_input")
    assert {k: v for k, v in b.items() if k not in changed} == {k: v for k, v in a.items() if k not in changed}
    assert b["stage_field_types"] == [{0: 14}, {0: 26}, {0: 38}, {0: 48}] and b["final_stage_scalars"] is None
    assert (b["depths"], b["drop_path_rate"], b["multiplicities"]) == ([3, 4, 7, 3], 0.1, [2, 4, 6, 6])
    assert (b["gravity_conditioning"], b["gravity_projection"], b["gravity_input"]) == (False, False, True)
    assert (b["gravity_readout"], b["vertical_conditioning"], b["value_kind"]) == (True, True, "steerable")
    # the same grid chain and readout rows; the ablation's own rate and two panoramas per forward (README)
    def top(cfg) -> dict:
        return {k: v for k, v in OmegaConf.to_container(cfg.model).items() if k != "architecture"}

    assert top(scalar) == top(sos2)
    assert (scalar.training.lr, scalar.training.panoramas_per_forward, scalar.data.num_workers) == (1.5e-3, 2, 2)


def _arm(tmp_path: Path, name: str, *, iou: float, ce: float, params: int) -> dict:
    """One arm's entry as the evaluator writes it (``experiments.stanford_2d3ds.evaluation.aggregate``):
    the seed-level summary the table reads under ``per_seed`` and the per-sample scores the pairing reads."""
    samples = [
        {"index": i, "cross_entropy": 0.6 + 0.01 * i, "iou": iou + 0.005 * i, "accuracy": 0.97, "pixel_accuracy": 0.9,
         "mean_class_iou": 0.5, "class_iou": [0.5] * 14}
        for i in range(40)
    ]
    means = {"cross_entropy": ce, "iou": iou, "accuracy": 0.972, "pixel_accuracy": 0.9, "mean_class_iou": iou - 0.1}
    spreads = {"cross_entropy": 0.3, "iou": 0.1, "accuracy": 0.01, "pixel_accuracy": 0.02, "mean_class_iou": 0.05}
    seed = {**{f"{m}_mean": means[m] for m in means}, **{f"{m}_std": spreads[m] for m in means},
            **{f"{m}_nonfinite": 0 for m in means},
            "num_samples": 40.0, "epoch": 200, "run_epoch": 200, "max_epochs": 200, "complete": True}
    summary = {
        "seeds": [0], "per_seed": {"0": seed},
        **{f"{m}_mean": means[m] for m in means}, **{f"{m}_seed_sd": 0.0 for m in means},
        **{f"{m}_sample_sd": spreads[m] for m in means}, **{f"{m}_nonfinite": 0 for m in means},
        "class_iou_mean": [0.5] * 14, "class_iou_samples": [40] * 14,
        "parameters": params, "epochs": [200], "max_epochs": 200, "complete": True, "unfinished_seeds": [], "checkpoint": "last",
    }
    return {"summary": summary, "runs": [{"run_dir": str(tmp_path / name / "seed0"), "samples": samples}]}


def _payload(tmp_path: Path, *, with_reference: bool = False) -> dict:
    arms = {"sos2_segformer": _arm(tmp_path, "sos2_segformer", iou=0.68, ce=0.61, params=593518)}
    if with_reference:
        arms["s2_segformer_released"] = _arm(tmp_path, "s2_segformer_released", iou=0.66, ce=0.59, params=609246)
    return {
        "split": "report", "checkpoint": "last", "arms": arms,
        "names": {"sos2_segformer": "SO(S^2)-SegFormer (ours)"},  # the evaluator's plain names (evaluate.py ARMS)
    }


def test_the_table_renders_one_row_per_arm_with_the_best_per_column_in_bold(tmp_path: Path) -> None:
    tex = tables.render(_payload(tmp_path, with_reference=True))
    assert tex.count(r"\\") == 3  # the header and one row per arm
    assert r"\begin{tabular}{lrccc}" in tex and r"Model & Params & CE $\downarrow$ & IoU $\uparrow$ & mIoU $\uparrow$ \\" in tex
    assert r"$\mathrm{SO}(S^2)$-SegFormer (ours) & 593,518 & 0.610 $\pm$ 0.000 & {\bfseries\boldmath 0.680 $\pm$ 0.000} & {\bfseries\boldmath 0.580 $\pm$ 0.000} \\" in tex
    assert r"S$^2$ SegFormer & 609,246 & {\bfseries\boldmath 0.590 $\pm$ 0.000} & 0.660 $\pm$ 0.000 & 0.560 $\pm$ 0.000 \\" in tex
    assert "SO(S^2)-SegFormer (ours) &" not in tex  # the module's LaTeX names take precedence over the evaluator's
    assert "report split (40 panoramas)" in tex and r"(\texttt{last.pt})" in tex and "over 1 seeds" in tex
    assert "Best value per column in bold." in tex and r"\label{tab:segmentation}" in tex
    assert "Paired" not in tables.render(_payload(tmp_path))
    dropped = tables.render(_payload(tmp_path, with_reference=True), drop=("sos2_segformer",))
    assert r"\mathrm{SO}" not in dropped and "Paired" not in dropped
    with pytest.raises(ValueError, match="no arms"):
        tables.render(_payload(tmp_path), drop=("sos2_segformer",))


def test_an_unfinished_run_is_marked_in_its_row_and_the_caption(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    payload["arms"]["sos2_segformer"]["summary"].update({
        "epochs": [37, 200], "complete": False,
        "unfinished_seeds": [{"seed": 1, "run_epoch": 37, "max_epochs": 200, "epoch": 37}],
    })
    tex = tables.render(payload)
    assert "(unfinished: seed 1 at epoch 37 of 200)" in tex and "not final results" in tex
    assert "unfinished" not in tables.render(_payload(tmp_path))


def test_the_paired_difference_needs_the_same_images(tmp_path: Path) -> None:
    payload = _payload(tmp_path, with_reference=True)
    assert tables.paired_difference(payload, "sos2_segformer", "s2_segformer_released", "iou") is not None
    payload["arms"]["s2_segformer_released"]["runs"][0]["samples"].pop()
    assert tables.paired_difference(payload, "sos2_segformer", "s2_segformer_released", "iou") is None
    assert "Paired over" not in tables.render(payload)


def test_the_caption_carries_the_paired_difference_when_both_arms_are_present(tmp_path: Path) -> None:
    payload = _payload(tmp_path, with_reference=True)
    mean, se, n = tables.paired_difference(payload, "sos2_segformer", "s2_segformer_released", "iou")
    assert (mean, se, n) == (pytest.approx(0.02), pytest.approx(0.0), 40)
    tex = tables.render(payload)
    assert "Paired over the 40 report images" in tex and r"$\Delta$IoU $= +0.020 \pm 0.000$" in tex
    assert r"$\mathrm{SO}(S^2)$-SegFormer (ours) minus S$^2$ SegFormer (mean $\pm$ standard error)" in tex


def test_resolve_device_makes_an_indexed_cuda_device_current_and_leaves_a_bare_one_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """torch.cuda.set_device rejects a device without an index; the bare
    "cuda" (the default and cfg.device=null) already means the current one."""
    import torch

    from experiments.stanford_2d3ds.train import resolve_device

    calls: list[torch.device] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", calls.append)
    assert resolve_device("cuda:3") == torch.device("cuda:3") and calls == [torch.device("cuda:3")]
    assert resolve_device("cuda") == torch.device("cuda") and len(calls) == 1
    assert resolve_device(None) == torch.device("cuda") and len(calls) == 1
    assert resolve_device("cpu") == torch.device("cpu") and len(calls) == 1
