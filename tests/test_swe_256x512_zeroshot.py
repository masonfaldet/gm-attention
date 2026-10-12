"""The zero-shot bundle: configs, the transferred arms end to end on a
tiny single-level run, and the grouped table."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "SWE_256x512_ZeroShot" / "configs"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tables = _load("swe_zeroshot_tables", ROOT / "SWE_256x512_ZeroShot" / "tables.py")
evaluate = _load("swe_zeroshot_evaluate", ROOT / "SWE_256x512_ZeroShot" / "evaluate.py")


def _compose(arm: str, seed: int = 1, *, with_hydra: bool = False):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        return compose(config_name=arm, overrides=[f"seed={seed}"], return_hydra_config=with_hydra)


@pytest.mark.parametrize("arm", ["gm_l5_100", "sfno", "s2_transformer_sl"])
def test_every_arm_shares_the_main_benchmarks_data(arm: str) -> None:
    cfg = _compose(arm)
    d = cfg.data
    assert (d.nlat, d.nlon, d.lead_time_seconds, d.solver_substeps) == (256, 512, 3600.0, 24)
    assert (d.train_seed, d.val_seed, d.test_seed, d.normalization_seed, d.normalization_examples) == (1000, 2000, 3000, 4000, 128)
    assert cfg.objective.residual is True and cfg.training.batch_size == 4
    assert cfg.training.grad_clip_norm == 1.0
    assert cfg.training.early_stopping_patience is None
    cfg = _compose(arm, seed=2, with_hydra=True)
    assert cfg.hydra.run.dir == f"outputs/SWE_256x512_ZeroShot/{arm}/seed2"


def test_the_arms_are_the_recorded_runs() -> None:
    gm = _compose("gm_l5_100")
    assert gm.model.kind == "gm_attention_single" and gm.grid.subdivisions == 5
    assert (gm.training.lr, gm.training.max_epochs, gm.training.scheduler.kind) == (2e-2, 100, "cosine")
    assert gm.model.architecture.chunk_size == 2048
    sfno = _compose("sfno")
    assert sfno.model.kind == "sfno" and (sfno.model.model_view.nlat, sfno.model.model_view.nlon) == (256, 512)
    assert sfno.model.architecture.embed_dim == 22 and (sfno.training.lr, sfno.training.max_epochs) == (1.6e-2, 100)
    sl = _compose("s2_transformer_sl")
    a = sl.model.architecture
    assert sl.model.kind == "attention_on_sphere" and (sl.model.model_view.nlat, sl.model.model_view.nlon) == (128, 256)
    assert sl.model.model_view.grid == "legendre-gauss" and a.grid == "legendre-gauss"  # the transfer row's grid
    assert (a.nlat, a.nlon, a.embed_dim, a.num_heads, a.scale_factor, a.normalization_layer) == (128, 256, 128, 1, 1, "instance_norm")
    assert list(a.encoder_kernel_shape) == [5, 4] and a.filter_basis_type == "piecewise linear" and a.grid_internal == "legendre-gauss"
    assert sl.training.micro_batch_size == 2 and sl.training.batch_size == 4  # two accumulated micro-batches
    assert (sl.training.lr, sl.training.max_epochs) == (6e-3, 100)  # its ladder's best rung, seed-filled


def test_arm_keys_name_the_grids_and_the_run() -> None:
    assert evaluate.arm_key("icosphere:6") == "gm_l5_100_icosphere6"
    assert evaluate.arm_key("legendre-gauss:128x256") == "gm_l5_100_lg128x256"
    assert list(evaluate.names()) == ["gm_l5_100_icosphere4", "gm_l5_100_icosphere5", "gm_l5_100_icosphere6", "gm_l5_100_lg128x256", "gm_l5_100_lg256x512", "sfno", "s2_transformer_sl"]
    assert "s2_transformer_sl" in tables.GROUPS
    # another training of the arm keeps its own rows
    assert evaluate.arm_key("icosphere:6", "gm_l5") == "gm_l5_icosphere6"
    assert list(evaluate.names(("icosphere:6",), "gm_l5"))[0] == "gm_l5_icosphere6"
    summary = {"gm_l5_icosphere6": {"grid": "icosphere:6"}, "gm_l5_100_icosphere6": {"grid": "icosphere:6"},
               "gm_l5_100_lg128x256": {"grid": "legendre-gauss:128x256"}, "sfno": {"grid": "equiangular:256x512"}}
    assert [tables.gm_run(summary, a) for a in summary] == ["gm_l5", "gm_l5_100", "gm_l5_100", "sfno"]


TINY = [  # the single-level arm of SWE_256x512_ZeroShot/configs/gm_l5_100.yaml, shrunk
    "++model.subdivisions=2", "grid.subdivisions=2", "device=cpu", "training.batch_size=2",
    "training.max_epochs=1", "data.nlat=32", "data.nlon=64", "data.solver_substeps=1", "data.num_train_examples=4",
    "data.num_val_examples=2", "data.normalization_examples=1",
]


def test_a_tiny_single_level_run_transfers_to_other_grids_end_to_end(tmp_path: Path) -> None:
    from experiments.shallow_water.grid_transfer import GridSpec
    from experiments.shallow_water_paper.benchmark_evaluation import aggregate
    from experiments.shallow_water_paper.compare_spectral import (
        evaluate_arms,
        load_arm,
    )
    from experiments.shallow_water_paper.train import main as train_main

    run = tmp_path / "gm_l5" / "seed0"
    run.mkdir(parents=True)
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        cfg = compose(config_name="gm_l5_100", overrides=TINY)
    (run / ".hydra").mkdir()
    OmegaConf.save(cfg, run / ".hydra" / "config.yaml")
    previous = Path.cwd()
    os.chdir(run)
    try:
        train_main(cfg)
    finally:
        os.chdir(previous)
    checkpoint = run / "checkpoints" / "best.pt"
    device = torch.device("cpu")

    from experiments.shallow_water_paper.transfer import _config_of_run
    from experiments.shallow_water_paper.compare_spectral import GridContext, LazyArm

    grids = ("icosphere:1", "icosphere:2", "icosphere:3", "legendre-gauss:16x32")  # L1: 42 points, below the band's 49
    # two "seeds" (the same checkpoint twice) share one context per grid
    contexts = {g: GridContext(GridSpec.parse(g), _config_of_run(checkpoint), device) for g in grids}
    lazy = [LazyArm(f"{evaluate.arm_key(g, 'gm_l5')}/seed{k}", checkpoint, None, contexts[g]) for g in grids for k in (0, 1)]
    assert all(c.users == 2 for c in contexts.values())
    assert lazy[0].cfg.model.kind == "gm_attention_single" and lazy[0].normalization is not None  # known before building
    native = load_arm("native/seed0", checkpoint, device, None)
    saved: list[int] = []
    results, coverage = evaluate_arms([*lazy, native], bandwidth=6, num_examples=2, split="test", device=device,
                                      steps=1, projector_device=device, on_result=lambda rs, _: saved.append(len(rs)))
    assert saved == list(range(1, len(lazy) + 2))  # called after every arm with the results so far
    assert all(c._built is None and c.users == 0 for c in contexts.values())  # released after their last seed
    by_label = {r.label: r for r in results}
    assert [r.grid for r in results] == ["icosphere:1"] * 2 + ["icosphere:2"] * 2 + ["icosphere:3"] * 2 + ["legendre-gauss:16x32"] * 2 + ["icosphere:2"]
    coarse = by_label["gm_l5_icosphere1/seed0"]
    assert "paper_squared_l2" in coarse.metrics and "paper_squared_l2_L6" not in coarse.metrics and coarse.D == []
    assert coarse.rollout[0].rollout and "paper_squared_l2_L6" not in coarse.rollout[0].rollout and coarse.rollout[0].D == []
    import math
    assert math.isnan(coarse.condition)
    assert all(r.training_grid == "icosphere:2" and r.checkpoint == str(checkpoint) and r.epoch == 0 for r in results)
    assert by_label["gm_l5_icosphere3/seed0"].metrics == by_label["gm_l5_icosphere3/seed1"].metrics  # same weights, shared geometry
    assert by_label["gm_l5_icosphere3/seed0"].n_nodes == 642 and by_label["gm_l5_lg16x32/seed0"].n_nodes == 512
    degrees = [r.mean_degree for r in results]
    assert all(d is not None and d >= 1 for d in degrees)  # the node itself is counted; L2 is coarser than the cutoff
    assert degrees[4] > 10 > degrees[2]  # the same angular cutoff holds far more of a finer grid's nodes
    # the transferred arm on the training grid is the native arm
    transferred, own = by_label["gm_l5_icosphere2/seed0"], by_label["native/seed0"]
    for key in ("paper_squared_l2", "paper_squared_l2_L6", "paper_l1", "paper_l1_L6"):
        assert transferred.metrics[key] == pytest.approx(own.metrics[key], rel=1e-5)
        assert transferred.rollout[0].rollout[key] == pytest.approx(own.rollout[0].rollout[key], rel=1e-5)
    assert transferred.mean_degree == pytest.approx(own.mean_degree)
    # the paper L1 in the accumulator agrees with the metric module's
    assert own.metrics["paper_l1"] == pytest.approx(own.paper_l1, rel=1e-5)
    summary = {arm: aggregate(results, arm) for arm in ("gm_l5_icosphere1", "gm_l5_icosphere2", "gm_l5_icosphere3", "gm_l5_lg16x32")}
    assert summary["gm_l5_icosphere3"]["n_nodes"] == 642 and summary["gm_l5_lg16x32"]["mean_degree"] > 1
    assert summary["gm_l5_icosphere3"]["seeds"] == 2 and summary["gm_l5_icosphere3"]["training_grid"] == "icosphere:2"
    assert summary["gm_l5_icosphere3"]["checkpoints"][0]["checkpoint"] == str(checkpoint)
    data = {"split": "test", "num_examples": 2, "bandwidth": 6, "coverage": coverage, "steps": 1,
            "lead_time_seconds": 3600.0, "solver": [32, 64], "summary": summary, "per_seed": []}
    # the bundle's arm builder honours --seeds and --skip-gm
    import argparse
    args = argparse.Namespace(root=str(tmp_path), arms="gm_l5_100=gm_l5", grids="icosphere:2,icosphere:3", seeds=None,
                              skip_gm=False, chunk_size=None, gm_device=None, require_seeds=None)
    assert [a.label for a in evaluate.build_arms(args, device)] == ["gm_l5_icosphere2/seed0", "gm_l5_icosphere3/seed0"]  # labelled by the run directory
    args.seeds = "1"
    assert evaluate.build_arms(args, device) == []  # no seed 1 under this root
    args.seeds, args.skip_gm = None, True
    assert evaluate.build_arms(args, device) == []
    tex = tables.render(data)
    tables.check(tex)
    assert "L2 icosphere (native) &" in tex and "L3 icosphere &" in tex and r"Legendre--Gauss $16\times32$ &" in tex
    l1 = next(line for line in tex.splitlines() if line.startswith("L1 icosphere &"))
    assert l1.count(" -- ") == 2 or l1.count("& --") == 2  # the band columns of the too-coarse row
    assert "L1 icosphere has fewer points than the band has coefficients ($(6+1)^2 = 49$)" in tex
    assert "trained on the L2 icosphere" in tex and r"$32\times64$ shallow-water benchmark" in tex  # from the runs, not constants


def _arm(value: float, grid: str, n_nodes: int, degree: float | None, steps: int = 5, training_grid: str | None = None,
         epoch: int | None = None, arm: str = "a") -> dict:
    metrics = {"paper_squared_l2": (value, 0.1 * value), "paper_squared_l2_L85": (0.9 * value, 0.1 * value),
               "paper_l1": (value ** 0.5, 0.01), "paper_l1_L85": (0.9 * value ** 0.5, 0.01)}
    step = {"rollout": {k: (3 * v[0], v[1]) for k, v in metrics.items()}, "one_step": dict(metrics),
            "persistence": {"paper_squared_l2": (2.0, 0.0), "paper_squared_l2_L85": (2.0, 0.0)}, "D": [0.0] * 86, "coverage": 1.0}
    checkpoints = [] if epoch is None else [{"label": f"{arm}/seed{k}", "checkpoint": "best.pt", "epoch": epoch} for k in range(3)]
    return {"seeds": 3, "grid": grid, "parameters": 491404, "condition": 2.0, "n_nodes": n_nodes, "mean_degree": degree,
            "training_grid": training_grid or "icosphere:5", "kind": "k",
            "checkpoints": checkpoints, "metrics": metrics, "paper_l1": (value, 0.0), "paper_l2": (value, 0.0), "D": [0.0] * 86,
            "rollout": [{**step, "step": k + 1, "hours": float(k + 1)} for k in range(steps)]}


def _data(steps: int = 5) -> dict:
    summary = {
        "gm_l5_icosphere4": _arm(0.0100, "icosphere:4", 2562, 49.0, steps),
        "gm_l5_icosphere5": _arm(0.0013, "icosphere:5", 10242, 190.0, steps),
        "gm_l5_icosphere6": _arm(0.0011, "icosphere:6", 40962, 760.0, steps),
        "gm_l5_lg128x256": _arm(0.0012, "legendre-gauss:128x256", 32768, 600.0, steps),
        "gm_l5_lg256x512": _arm(0.0010, "legendre-gauss:256x512", 131072, 2400.0, steps),
        "sfno": _arm(0.0009, "equiangular:256x512", 131072, None, steps, training_grid="equiangular:256x512"),
        "s2_transformer_sl": _arm(0.013, "legendre-gauss:128x256", 32768, 141.0, steps, training_grid="legendre-gauss:128x256"),
    }
    summary["gm_l5_lg256x512"]["seeds"] = 1
    for key in ("metrics",):  # the L4 row has no band-limited entries
        summary["gm_l5_icosphere4"][key] = {k: v for k, v in summary["gm_l5_icosphere4"][key].items() if "_L85" not in k}
    for step in summary["gm_l5_icosphere4"]["rollout"]:
        step["rollout"] = {k: v for k, v in step["rollout"].items() if "_L85" not in k}
    return {"split": "test", "num_examples": 64, "bandwidth": 85, "coverage": 1.0, "steps": steps,
            "lead_time_seconds": 3600.0, "solver": [256, 512], "summary": summary, "per_seed": []}


def test_the_table_groups_the_gm_rows_and_marks_native_grids() -> None:
    tex = tables.render(_data())
    tables.check(tex)
    lines = tex.splitlines()
    assert tex.count(r"\begin{table}") == 1
    titles = [line for line in lines if line.startswith(r"\multicolumn{6}{@{}l}{\emph{")]
    assert len(titles) == 3 and "GM-Attention, trained on the L5 icosphere (491,404 parameters)" in titles[0]
    assert "S$^2$ Transformer (local), single level (491,404 parameters)" in titles[2]
    assert tex.count(r"equiangular $256\times512$ (native) & 131,072 &") == 1  # SFNO
    assert r"Legendre--Gauss $128\times256$ (native) & 32,768 &" in tex  # the S2 transformer, on the transfer row's grid
    assert "nbrs" not in tex and "mean_degree" not in tex  # the neighbourhood column is not tabulated
    assert "L4 icosphere & 2,562 & " in tex
    l4 = next(line for line in lines if line.startswith("L4 icosphere &"))
    assert l4.count("& -- ") == 2 and r"\bfseries" not in l4  # its two band columns
    assert "L4 icosphere has fewer points than the band has coefficients ($(85+1)^2 = 7,396$)" in tex
    assert "L5 icosphere (native) & 10,242 & " in tex
    assert "L6 icosphere & 40,962 & " in tex
    assert r"Legendre--Gauss $256\times512$ & 131,072 & " in tex
    assert r"\multicolumn{2}{c}{One step (1\,h), $10^{3}\times L_2^2$} & \multicolumn{2}{c}{5 steps (5\,h), $10^{3}\times L_2^2$}" in tex
    c = tables.labels
    assert "Grid & " + c("points", "native", r"$\ell \le 85$", "native", r"$\ell \le 85$") + r" \\" in tex  # centred over their columns
    assert "L_1" not in tex  # no L1 columns
    sfno = next(line for line in lines if line.startswith("equiangular $256"))
    assert sfno.count(r"\bfseries") == 4  # lowest in every column of the synthetic data
    lg = next(line for line in lines if line.startswith(r"Legendre--Gauss $256"))
    assert lg.count("(1 seed)") == 4 and r"$\pm$" not in lg  # a single seed shows no sd
    assert r"\begin{tabular}[c]" not in tex  # inline cells, not stacked
    assert "a single seed is marked" in tex
    assert r"$256\times512$ shallow-water benchmark" in tex and r"\small" in tex and r"\footnotesize" not in tex
    assert "zero-mean" not in tex and "zero_mean" not in tex


def test_a_reference_row_is_borrowed_from_another_study_of_the_same_protocol(tmp_path: Path) -> None:
    """The depth-1 arm's row comes from the ablation study, not from a
    second scoring: --merge-only takes it and leaves that study's other
    arms behind, and the merge still checks the trajectories."""
    from experiments.shallow_water_paper.benchmark_evaluation import merge_files, restricted

    ablation = _document({"gm_attention_full": 4.7, "gm_convolution_full": 9.0, "gm_attention_rho0": 57.0})
    kept = restricted(ablation, {"gm_attention_full"})
    assert set(kept["summary"]) == {"gm_attention_full"}
    assert [r["label"] for r in kept["per_seed"]] == ["gm_attention_full/seed0"]
    assert set(ablation["summary"]) == {"gm_attention_full", "gm_convolution_full", "gm_attention_rho0"}  # untouched
    assert restricted(ablation, None) is ablation
    with pytest.raises(SystemExit, match="does not hold"):
        restricted(ablation, {"gm_attention_full", "not_an_arm"})

    # run.sh borrows in an --append invocation of the ablation JSON's own,
    # because the filter applies to every part of one invocation: naming an
    # arm the parts in hand do not hold is an error, not a silent empty set
    study = tmp_path / "study.json"
    borrowed = tmp_path / "ablation.json"
    study.write_text(json.dumps(_document({"gm_l5_icosphere5": 1.0, "sfno": 0.5})))
    borrowed.write_text(json.dumps(ablation))
    with pytest.raises(SystemExit, match="does not hold"):
        merge_files([study, borrowed], only={"gm_attention_full"})

    from experiments.shallow_water_paper.benchmark_evaluation import merged as merge_documents
    whole = merge_documents(merge_files([study]), merge_files([borrowed], only={"gm_attention_full"}))
    assert set(whole["summary"]) == {"gm_l5_icosphere5", "sfno", "gm_attention_full"}
    assert len(whole["per_seed"]) == 3
    assert whole["summary"]["gm_attention_full"]["metrics"]["paper_squared_l2"] == pytest.approx((4.7, 0.0))
    # a borrowed study whose trajectories differ is still refused
    other = tmp_path / "other.json"
    other.write_text(json.dumps(_document({"gm_attention_full": 4.7}, band_persistence=2.5)))
    with pytest.raises(SystemExit, match="different trajectories"):
        merge_documents(merge_files([study]), merge_files([other], only={"gm_attention_full"}))


def test_the_depth1_arm_is_a_reference_group_on_its_own_grid() -> None:
    data = _data()
    S = data["summary"]
    S["gm_attention_full"] = _arm(0.0047, "icosphere:5", 10242, 49.0, training_grid="icosphere:5")
    tex = tables.render(data, drop=("gm_l5_icosphere4", "gm_l5_lg256x512"))
    tables.check(tex)
    titles = [line for line in tex.splitlines() if line.startswith(r"\multicolumn{6}{@{}l}{\emph{")]
    # between the transferred group and the references, as GROUPS orders it
    assert len(titles) == 4
    assert "trained on the L5 icosphere" in titles[0]
    assert "GM-Attention, depth-1 on a level-4 latent (491,404 parameters)" in titles[1]
    assert "SFNO" in titles[2] and "S$^2$ Transformer" in titles[3]
    assert tables.ordered_arms(S)[:2] == ["gm_l5_icosphere4", "gm_l5_icosphere5"]
    assert tables.ordered_arms(S).index("gm_attention_full") < tables.ordered_arms(S).index("sfno")
    # its own grid, so the row is marked native and the caption says so
    assert tex.count("L5 icosphere (native) &") == 2  # the transferred arm's and the depth-1 arm's
    assert "Every other row is a model on its own grid" in tex


def test_the_manifest_check_refuses_a_run_of_the_wrong_kind(tmp_path: Path) -> None:
    from omegaconf import OmegaConf

    run = tmp_path / "gm_l5" / "seed0"
    (run / ".hydra").mkdir(parents=True)
    (run / "checkpoints").mkdir()
    (run / "checkpoints" / "best.pt").write_bytes(b"")
    OmegaConf.save(OmegaConf.create({"model": {"kind": "sfno"}}), run / ".hydra" / "config.yaml")
    with pytest.raises(SystemExit, match="expected 'gm_attention_single'"):
        evaluate.check_manifest({"gm_l5_100": [(0, run / "checkpoints" / "best.pt")]})


def _row(arm: str, value: float, seed: int, persistence: float, band_persistence: float | None) -> dict:
    p = {"paper_squared_l2": persistence}
    if band_persistence is not None:
        p["paper_squared_l2_L10"] = band_persistence
    return {"label": f"{arm}/seed{seed}", "kind": "k", "grid": "g", "n_nodes": 10, "parameters": 1, "condition": 2.0,
            "metrics": {"paper_squared_l2": value}, "D": [value], "paper_l1": value, "paper_l2": value,
            "rollout": [{"step": 1, "hours": 1.0, "coverage": 1.0, "rollout": {"paper_squared_l2": 2 * value},
                         "one_step": {}, "persistence": p, "D": [2 * value]}]}


def _document(arms: dict[str, float], persistence: float = 2.0, band_persistence: float | None = 2.0, seed: int = 0, **overrides) -> dict:
    per_seed = [_row(arm, value, seed, persistence, band_persistence) for arm, value in arms.items()]
    document = {"split": "test", "num_examples": 4, "bandwidth": 10, "steps": 1, "lead_time_seconds": 3600.0,
                "solver": [32, 64], "summary": {arm: {"metrics": {"paper_squared_l2": (v, 0.0)}} for arm, v in arms.items()},
                "per_seed": per_seed, "partial": False, "environment": {"device": "cpu"}}
    document.update(overrides)
    return document


def test_parts_of_one_study_merge_and_mismatched_parts_are_refused(tmp_path: Path) -> None:
    from experiments.shallow_water_paper.benchmark_evaluation import (
        merge_files,
        merged,
        selected_seeds,
    )

    a = _document({"gm_l5_icosphere5": 1.0, "sfno": 0.5})
    b = _document({"gm_l5_icosphere6": 0.8})
    out = merged(a, b)
    assert set(out["summary"]) == {"gm_l5_icosphere5", "sfno", "gm_l5_icosphere6"} and len(out["per_seed"]) == 3
    assert out["partial"] is False
    assert merged(a, _document({"x": 1.0}, partial=True))["partial"] is True
    with pytest.raises(SystemExit, match="different trajectories"):
        merged(a, _document({"gm_l5_icosphere6": 0.8}, band_persistence=2.5))
    # native persistence is each grid's own quadrature: a 0.1% difference is a grid, not a trajectory
    merged(a, _document({"gm_l5_icosphere6": 0.8}, persistence=2.002))
    # a part with only a too-coarse grid has no band persistence: native to native, loosely
    merged(a, _document({"gm_l5_icosphere4": 9.0}, persistence=2.002, band_persistence=None))
    with pytest.raises(SystemExit, match="different trajectories"):
        merged(a, _document({"gm_l5_icosphere4": 9.0}, persistence=2.5, band_persistence=None))
    with pytest.raises(SystemExit, match="num_examples differs"):
        merged(a, _document({"gm_l5_icosphere6": 0.8}, num_examples=8))
    parts = []
    for name, document in (("a", a), ("b", b)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(document))
        parts.append(path)
    assert set(merge_files(parts)["summary"]) == set(out["summary"])
    # one arm's seeds arriving in separate parts are aggregated over all of them
    three = merged(merged(_document({"gm_l5_icosphere6": 1.0}, seed=0), _document({"gm_l5_icosphere6": 3.0}, seed=1)),
                   _document({"gm_l5_icosphere6": 2.0}, seed=2))
    s = three["summary"]["gm_l5_icosphere6"]
    assert s["seeds"] == 3 and s["metrics"]["paper_squared_l2"] == pytest.approx((2.0, 1.0))
    assert s["rollout"][0]["rollout"]["paper_squared_l2"] == pytest.approx((4.0, 2.0)) and s["D"] == pytest.approx([2.0])
    assert list(three["summary"]) == ["gm_l5_icosphere6"] and len(three["per_seed"]) == 3

    import argparse
    seeds = [(0, Path("s0")), (1, Path("s1")), (2, Path("s2"))]
    assert selected_seeds(argparse.Namespace(seeds="0,2"), seeds) == [(0, Path("s0")), (2, Path("s2"))]
    assert selected_seeds(argparse.Namespace(seeds=None), seeds) == seeds


def test_two_trainings_of_the_arm_are_two_groups_named_by_their_epoch() -> None:
    data = _data()
    S = data["summary"]
    for grid, key, nodes, degree, value in (("icosphere:5", "icosphere5", 10242, 190.0, 0.0008),
                                            ("icosphere:6", "icosphere6", 40962, 760.0, 0.0028),
                                            ("legendre-gauss:128x256", "lg128x256", 32768, 600.0, 0.0039)):
        S[f"gm_l5_100_{key}"] = _arm(value, grid, nodes, degree, epoch=99, arm=f"gm_l5_100_{key}")
    for arm in ("gm_l5_icosphere4", "gm_l5_icosphere5", "gm_l5_icosphere6", "gm_l5_lg128x256", "gm_l5_lg256x512"):
        S[arm]["checkpoints"] = [{"label": f"{arm}/seed{k}", "checkpoint": "best.pt", "epoch": 49} for k in range(3)]
    tex = tables.render(data, drop=("gm_l5_icosphere4", "gm_l5_lg256x512"))
    tables.check(tex)
    titles = [line for line in tex.splitlines() if line.startswith(r"\multicolumn{6}{@{}l}{\emph{")]
    assert len(titles) == 4  # two GM runs, then the two references
    assert "trained on the L5 icosphere (epoch 50, 491,404 parameters)" in titles[0]
    assert "trained on the L5 icosphere (epoch 100, 491,404 parameters)" in titles[1]
    assert "SFNO" in titles[2] and "S$^2$ Transformer" in titles[3]
    # each run keeps its own rows: the native L5 row appears once per group
    assert tex.count("L5 icosphere (native) & 10,242 & ") == 2
    assert "0.80 $\\pm$" in tex and "1.30 $\\pm$" in tex  # the 100-epoch native and the 50-epoch one
    assert "differing in length alone" in tex and r"\emph{epoch}: the epoch a heading's checkpoints" in tex
    assert "Each transferred group is one set of checkpoints" in tex
    # one run alone is one group, and rows recording no epoch carry none
    single = tables.render(_data())
    assert "epoch " not in single and "differing in length alone" not in single
    assert "The transferred rows are one set of checkpoints" in single


def test_a_second_run_of_one_arm_is_relabelled_into_the_study(tmp_path: Path) -> None:
    from experiments.shallow_water_paper.benchmark_evaluation import (
        merge_files,
        relabelled,
        renamed,
    )

    rules = {"gm_l5": "gm_l5_100"}
    assert renamed("gm_l5", rules) == "gm_l5_100" and renamed("gm_l5_icosphere6", rules) == "gm_l5_100_icosphere6"
    assert renamed("sfno", rules) == "sfno" and renamed("gm_l5x", rules) == "gm_l5x"  # only whole names and _ suffixes
    # idempotent: a part evaluated after the rows carried their run's name
    # is already named that, and the same rule leaves it alone
    assert renamed("gm_l5_100_icosphere6", rules) == "gm_l5_100_icosphere6"
    # the longest rule wins, so gm_l5 does not claim what gm_l5_100 names
    both = {"gm_l5": "a", "gm_l5_100": "b"}
    assert renamed("gm_l5_100_icosphere6", both) == "b_icosphere6" and renamed("gm_l5_icosphere6", both) == "a_icosphere6"

    part = _document({"gm_l5_icosphere6": 0.8, "sfno": 0.5})
    part["summary"]["gm_l5_icosphere6"]["checkpoints"] = [{"label": "gm_l5_icosphere6/seed0", "epoch": 99}]
    out = relabelled(part, rules)
    assert set(out["summary"]) == {"gm_l5_100_icosphere6", "sfno"}
    assert [r["label"] for r in out["per_seed"]] == ["gm_l5_100_icosphere6/seed0", "sfno/seed0"]
    assert out["summary"]["gm_l5_100_icosphere6"]["checkpoints"][0]["label"] == "gm_l5_100_icosphere6/seed0"
    assert part["summary"]["gm_l5_icosphere6"]["checkpoints"][0]["label"] == "gm_l5_icosphere6/seed0"  # the part is untouched
    # a rule that would fold two arms into one is refused, not applied: a
    # part holding both runs, or two rules with one target
    with pytest.raises(SystemExit, match="both become"):
        relabelled(_document({"gm_l5_icosphere6": 0.8, "gm_l5_100_icosphere6": 0.7}), rules)
    with pytest.raises(SystemExit, match="both become"):
        relabelled(_document({"gm_l5_icosphere6": 0.8, "sfno": 0.7}), {"gm_l5_icosphere6": "x", "sfno": "x"})

    # the collision the relabel exists to prevent: without it the second
    # run's rows replace the first's, silently, under the same labels
    fifty = tmp_path / "fifty.json"
    hundred = tmp_path / "hundred.json"
    fifty.write_text(json.dumps(_document({"gm_l5_icosphere6": 1.0, "sfno": 0.5})))
    hundred.write_text(json.dumps(_document({"gm_l5_icosphere6": 2.0})))
    assert set(merge_files([fifty, hundred])["summary"]) == {"gm_l5_icosphere6", "sfno"}
    merged_rows = {r["label"]: r["metrics"]["paper_squared_l2"] for r in merge_files([fifty, hundred])["per_seed"]}
    assert merged_rows["gm_l5_icosphere6/seed0"] == 2.0  # the 50-epoch row is gone
    # relabelling both sources keeps them apart; the rules go one source at a time
    study = merge_files([fifty])
    second = merge_files([hundred], {"gm_l5": "gm_l5_100"})
    from experiments.shallow_water_paper.benchmark_evaluation import merged as merge_documents
    whole = merge_documents(study, second)
    assert set(whole["summary"]) == {"gm_l5_icosphere6", "sfno", "gm_l5_100_icosphere6"}
    rows = {r["label"]: r["metrics"]["paper_squared_l2"] for r in whole["per_seed"]}
    assert rows["gm_l5_icosphere6/seed0"] == 1.0 and rows["gm_l5_100_icosphere6/seed0"] == 2.0
    assert whole["summary"]["gm_l5_100_icosphere6"]["metrics"]["paper_squared_l2"] == pytest.approx((2.0, 0.0))


def test_one_coarse_grid_is_named_once_however_many_runs_were_scored_on_it() -> None:
    data = _data()
    S = data["summary"]
    # a second run scored on the same too-coarse grid: two rows, one grid
    S["gm_l5_100_icosphere4"] = _arm(0.0100, "icosphere:4", 2562, 49.0, epoch=99, arm="gm_l5_100_icosphere4")
    S["gm_l5_100_icosphere4"]["metrics"] = {k: v for k, v in S["gm_l5_100_icosphere4"]["metrics"].items() if "_L85" not in k}
    for step in S["gm_l5_100_icosphere4"]["rollout"]:
        step["rollout"] = {k: v for k, v in step["rollout"].items() if "_L85" not in k}
    tex = tables.render(data)
    tables.check(tex)
    assert tex.count("L4 icosphere &") == 2  # both rows are there
    assert "L4 icosphere and L4 icosphere" not in tex
    assert "L4 icosphere has fewer points than the band has coefficients" in tex


def test_arms_can_be_dropped_from_the_table() -> None:
    tex = tables.render(_data(), drop=("gm_l5_icosphere4",))
    tables.check(tex)
    assert "L4 icosphere" not in tex and "fewer points than the band" not in tex
    assert "L5 icosphere (native)" in tex and "L6 icosphere" in tex and "equiangular $256" in tex


def test_a_whole_training_of_the_arm_can_be_dropped_by_name() -> None:
    data = _data()
    S = data["summary"]
    for grid, key, nodes, value in (("icosphere:5", "icosphere5", 10242, 0.0008), ("icosphere:6", "icosphere6", 40962, 0.0028)):
        S[f"gm_l5_100_{key}"] = _arm(value, grid, nodes, 190.0, epoch=99, arm=f"gm_l5_100_{key}")
    # the run's name takes every grid it was scored on, and nothing else
    tex = tables.render(data, drop=("gm_l5",))
    tables.check(tex)
    titles = [line for line in tex.splitlines() if line.startswith(r"\multicolumn{6}{@{}l}{\emph{")]
    assert len(titles) == 3  # the surviving run, then the two references
    assert "L4 icosphere" not in tex and r"Legendre--Gauss $256\times512$" not in tex  # the dropped run's grids
    assert tex.count("L5 icosphere (native) & 10,242 & ") == 1 and "L6 icosphere & 40,962 & " in tex
    assert "equiangular $256" in tex and "S$^2$ Transformer" in tex
    # the survivor still says which training it is: that is the whole point
    assert "trained on the L5 icosphere (epoch 100, 491,404 parameters)" in tex
    assert "differing in length alone" not in tex  # only one group is shown
    assert tables.ordered_arms(S, ("gm_l5",)) == ["gm_l5_100_icosphere5", "gm_l5_100_icosphere6", "sfno", "s2_transformer_sl"]


def test_values_too_large_for_a_column_are_written_in_scientific_notation() -> None:
    from experiments.shallow_water_paper.latex_tables import (
        inline_cell,
        number,
        stacked_cell,
    )

    assert number(4717.17, 2) == "4717.17" and number(9999.9, 1) == "9999.9"
    assert number(24923.35, 2) == r"$2.5\times10^{4}$" and number(4717169.99, 2) == r"$4.7\times10^{6}$"
    assert inline_cell((4.71717, 2.966), 1e3, 2) == r"4717.17 $\pm$ 2966.00"
    assert inline_cell((4717.17, 2965.97), 1e3, 2) == r"$4.7\times10^{6}$ $\pm$ $3.0\times10^{6}$"
    assert r"\times10^{6}" in stacked_cell((4717.17, 2965.97), 1e3, 2)
