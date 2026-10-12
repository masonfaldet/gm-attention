"""The ablation bundle composes to the settings that produced the runs, and
its table generator renders one well-formed table."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "SWE_256x512_Ablation" / "configs"
ARMS = ("gm_attention_full", "gm_convolution_full", "gm_attention_rho01", "gm_attention_rho0")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tables = _load("swe_ablation_tables", ROOT / "SWE_256x512_Ablation" / "tables.py")
evaluate = _load("swe_ablation_evaluate", ROOT / "SWE_256x512_Ablation" / "evaluate.py")


def _compose(arm: str, seed: int = 1, *, with_hydra: bool = False):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        return compose(config_name=arm, overrides=[f"seed={seed}"], return_hydra_config=with_hydra)


@pytest.mark.parametrize("arm", ARMS)
def test_every_arm_shares_the_main_benchmarks_data_and_protocol(arm: str) -> None:
    cfg = _compose(arm)
    d = cfg.data
    assert (d.nlat, d.nlon, d.lead_time_seconds, d.solver_substeps) == (256, 512, 3600.0, 24)
    assert (d.train_seed, d.val_seed, d.test_seed, d.normalization_seed) == (1000, 2000, 3000, 4000)
    assert (d.normalization_examples, d.train_sample_policy, d.num_train_examples, d.num_val_examples) == (128, "epoch_index", 256, 128)
    assert cfg.grid.subdivisions == 5 and cfg.objective.residual is True
    assert cfg.model.kind == "gm_attention_depth1"
    t = cfg.training
    assert (t.batch_size, t.max_epochs, t.grad_clip_norm, t.weight_decay, t.early_stopping_patience) == (4, 100, 1.0, 0.0, None)
    assert (t.scheduler.kind, t.scheduler.factor, t.scheduler.patience) == ("plateau", 0.5, 5)
    a = cfg.model.architecture
    assert (a.nonlinearity, a.decode_self_interaction, a.num_heads, a.window) == ("scalar_gate_double_sigmoid", "linear", 2, "hard")
    assert cfg.seed == 1


def test_each_twin_changes_one_thing_at_its_own_rate() -> None:
    full = _compose("gm_attention_full")
    assert OmegaConf.to_container(full.model.architecture.base_field_type) == {0: 1, 1: 1, 2: 1}
    assert full.model.architecture.multiplicity == 16 and full.training.lr == 1.2e-2
    assert "deep_mixer" not in full.model.architecture

    conv = _compose("gm_convolution_full")
    assert conv.model.architecture.deep_mixer == "convolution"
    assert OmegaConf.to_container(conv.model.architecture.base_field_type) == {0: 1, 1: 1, 2: 1}
    assert conv.model.architecture.multiplicity == 16  # the full arm's own spectrum
    assert conv.training.lr == 6e-3  # its ladder's best rung, seed-filled

    rho01 = _compose("gm_attention_rho01")
    assert OmegaConf.to_container(rho01.model.architecture.base_field_type) == {0: 1, 1: 1}
    assert rho01.model.architecture.multiplicity == 26  # equal copies, +0.4% over the full arm
    assert rho01.training.lr == 1.6e-2  # its ladder's best rung, seed-filled

    rho0 = _compose("gm_attention_rho0")
    assert OmegaConf.to_container(rho0.model.architecture.base_field_type) == {0: 1}
    assert rho0.model.architecture.multiplicity == 72 and rho0.training.lr == 1.2e-2


def test_run_directories_are_keyed_by_arm_and_seed() -> None:
    for arm in ARMS:
        cfg = _compose(arm, seed=2, with_hydra=True)
        assert cfg.hydra.run.dir == f"outputs/SWE_256x512_Ablation/{arm}/seed2"


def test_the_evaluation_names_the_bundles_arms() -> None:
    assert tuple(evaluate.ARMS) == ARMS and tuple(tables.ARMS) == ARMS


def _arm(value: float, steps: int) -> dict:
    metrics = {"paper_squared_l2": (value, 0.1 * value), "paper_squared_l2_L85": (0.9 * value, 0.1 * value)}
    for c in ("geopotential", "vorticity", "divergence"):
        metrics[f"{c}_rmse"] = (value ** 0.5, 0.01)
        metrics[f"{c}_rmse_L85"] = (0.9 * value ** 0.5, 0.01)
    step = {"rollout": {k: (3 * v[0], v[1]) for k, v in metrics.items()}, "one_step": dict(metrics),
            "persistence": {"paper_squared_l2": (2.0, 0.0), "paper_squared_l2_L85": (2.0, 0.0)}, "D": [0.0] * 86, "coverage": 1.0}
    return {"seeds": 3, "grid": "icosphere:5", "parameters": 500000, "condition": 2.0, "metrics": metrics,
            "paper_l1": (value, 0.0), "paper_l2": (value, 0.0), "D": [0.0] * 86,
            "rollout": [{**step, "step": k + 1, "hours": float(k + 1)} for k in range(steps)]}


def _data(steps: int = 5) -> dict:
    values = {"gm_attention_full": 0.0047, "gm_convolution_full": 0.0062, "gm_attention_rho01": 0.0073, "gm_attention_rho0": 0.0575}
    return {
        "split": "test", "num_examples": 64, "bandwidth": 85, "coverage": 1.0, "steps": steps, "lead_time_seconds": 3600.0,
        "solver": [256, 512],
        "summary": {arm: _arm(v, steps) for arm, v in values.items()},
        "per_seed": [{"label": f"{arm}/seed0", "n_nodes": 10242} for arm in values],
    }


def test_the_table_has_one_step_and_five_step_groups_with_stacked_cells() -> None:
    tex = tables.render(_data())
    tables.check(tex)
    assert tex.count(r"\begin{table}") == 1
    assert r"\multicolumn{4}{c}{One step (1\,h)} & \multicolumn{4}{c}{5 steps (5\,h)}" in tex
    assert r"\ell" not in tex  # no band metric: every arm shares the grid
    assert r"Mixer & Hidden type & Params & & $\varphi$ & $\zeta$ & $\delta$ & & $\varphi$ & $\zeta$ & $\delta$ \\" in tex
    assert all(row.split(" & ")[2] == "500,000" for row in tex.splitlines() if row.startswith("$SO(S^2)$"))  # the Params column
    lines = tex.splitlines()
    rows = [line for line in lines if line.startswith("$SO(S^2)$")]
    assert [" & ".join(row.split(" & ")[:2]) for row in rows] == [
        r"$SO(S^2)$-MHSA & $\rho_{0}^{16} \oplus \rho_{1}^{16} \oplus \rho_{2}^{16}$",
        r"$SO(S^2)$-Conv & $\rho_{0}^{16} \oplus \rho_{1}^{16} \oplus \rho_{2}^{16}$",
        r"$SO(S^2)$-MHSA & $\rho_{0}^{26} \oplus \rho_{1}^{26}$",
        r"$SO(S^2)$-MHSA & $\rho_{0}^{72}$",
    ]
    full = rows[0]
    assert full.count(r"\bfseries") == 8  # best in every column of the synthetic data
    assert all(r"\bfseries" not in row for row in rows[1:])
    assert r"\begin{tabular}[c]{@{}c@{}}{\bfseries 4.70}\\[-1pt]{\scriptsize $\pm$0.47}\end{tabular}" in full
    assert (r"Depth-1 ablation on the $256\times512$ shallow-water benchmark with 64 test trajectories. Metrics reported "
            "as mean over 3 seeds with the seed sd beneath. Parameters are real scalars. "
            "Persistence scores 2000 and 2000 in $L_2^2$ units at the two horizons.") in tex


def test_the_architecture_comes_from_the_json_when_it_recorded_it() -> None:
    data = _data()
    data["summary"]["gm_attention_full"]["mixer"] = "convolution"
    data["summary"]["gm_attention_full"]["hidden_type"] = {"0": 8, "2": 4}  # JSON keys are strings
    assert tables.architecture(data["summary"], "gm_attention_full") == (r"$SO(S^2)$-Conv", r"$\rho_{0}^{8} \oplus \rho_{2}^{4}$")
    assert tables.architecture(data["summary"], "gm_attention_rho0") == (r"$SO(S^2)$-MHSA", r"$\rho_{0}^{72}$")  # the fallback
    assert tables.architecture({"unknown": {}}, "unknown") == ("unknown", "--")


def test_without_a_rollout_the_table_has_the_one_step_group_alone() -> None:
    tex = tables.render(_data(steps=0))
    tables.check(tex)
    assert "5 steps" not in tex and r"\multicolumn{4}{c}{One step (1\,h)}" in tex
    assert "rollout" not in tex.split(r"\label")[1].split(r"\small")[0] or "autoregressive" not in tex


def test_zero_mean_twins_are_left_out() -> None:
    data = _data()
    data["summary"]["gm_attention_full_zero_mean"] = _arm(0.003, 5)
    data["per_seed"].append({"label": "gm_attention_full_zero_mean/seed0", "n_nodes": 10242})
    tex = tables.render(data)
    tables.check(tex)
    assert "zero" not in tex and tables.ordered_arms(data["summary"]) == list(ARMS)


def test_a_one_step_inline_table_has_one_group_and_no_stacked_cells() -> None:
    tex = tables.render(_data(), one_step=True, inline=True)
    tables.check(tex)
    assert "5 steps" not in tex and r"\begin{tabular}[c]" not in tex
    assert r"\multicolumn{4}{c}{One step (1\,h)}" in tex and tex.count(r"\cmidrule(lr){4-7}") == 1
    c = tables.centered
    assert f"Mixer & Hidden type & {c('Params')} & & {c('$\\varphi$')} & {c('$\\zeta$')} & {c('$\\delta$')} \\\\" in tex
    full = next(line for line in tex.splitlines() if line.startswith("$SO(S^2)$-MHSA & $\\rho_{0}^{16}"))
    assert full.count(r"\bfseries\boldmath") == 4 and r"4.70 $\pm$ 0.47" in full
    assert "Persistence scores 2000 in $L_2^2$ units." in tex and "two horizons" not in tex
    assert r"mean $\pm$ sd over 3 seeds." in tex and "beneath" not in tex
    assert r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrrr@{}}" in tex
