"""The benchmark's LaTeX tables: structure, bolding, and degree bands."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("swe_tables", ROOT / "SWE_128x256_Benchmark" / "tables.py")
assert spec is not None and spec.loader is not None
tables = importlib.util.module_from_spec(spec)
sys.modules["swe_tables"] = tables
spec.loader.exec_module(tables)

B = 10


def _arm(value: float, steps: int = 2, grid: str = "equiangular:128x256") -> dict:
    metrics = {"paper_squared_l2": (value, 0.1 * value), f"paper_squared_l2_L{B}": (0.9 * value, 0.1 * value),
               f"fit_residual_fraction_L{B}": (0.1, 0.01)}
    for c in ("geopotential", "vorticity", "divergence"):
        metrics[f"{c}_rmse"] = (value ** 0.5, 0.01)
        metrics[f"{c}_rmse_L{B}"] = (0.9 * value ** 0.5, 0.01)
    step = {"rollout": dict(metrics), "one_step": dict(metrics), "persistence": {"paper_squared_l2": (2.0, 0.0), f"paper_squared_l2_L{B}": (2.0, 0.0)},
            "D": [value / (B + 1)] * (B + 1), "coverage": 1.0}
    return {"seeds": 3, "grid": grid, "parameters": 1000, "condition": 2.0, "metrics": metrics,
            "paper_l1": (value, 0.0), "paper_l2": (value, 0.0), "D": [value / (B + 1)] * (B + 1),
            "rollout": [{**step, "step": k + 1, "hours": 0.5 * (k + 1)} for k in range(steps)]}


def _data(steps: int = 2) -> dict:
    return {
        "split": "test", "num_examples": 4, "bandwidth": B, "coverage": 1.0, "steps": steps,
        "summary": {"gm_attention_depth1": _arm(2.0, steps, "icosphere:5"), "sfno": _arm(1.0, steps), "s2_transformer": _arm(5.0, steps)},
        "per_seed": [{"label": f"{a}/seed0", "n_nodes": 10} for a in ("gm_attention_depth1", "sfno", "s2_transformer")],
    }


def _panels(tex: str) -> list[str]:
    return re.findall(r"\\begin\{tabular\*\}.*?\\end\{tabular\*\}", tex, flags=re.DOTALL)


def test_the_benchmark_table_stacks_one_step_rollout_and_teacher_forcing() -> None:
    tex = tables.render(_data())
    tables.check(tex)
    assert tex.count(r"\begin{table}") == 3
    panels = _panels(tex)
    assert len(panels) == 2
    assert r"\textbf{(a)} One step (30\,min)" in panels[0]
    assert r"\textbf{(b)} Rollout and teacher-forced one step, $10^{3}\times L_2^2$, native" in panels[1]
    lines = panels[0].splitlines()
    sfno = next(line for line in lines if line.startswith("SFNO &"))
    assert sfno.count(r"\bfseries") == 5  # best in every numeric column of the synthetic data
    gm = next(line for line in lines if line.startswith("GM-Attention (ours) &"))
    assert r"\bfseries" not in gm
    order = [line.split(" &")[0] for line in lines if line.startswith(("SFNO", "GM-Attention", "S$^2$"))]
    assert order == ["S$^2$ Transformer", "SFNO", "GM-Attention (ours)"]  # baselines first, ours last
    p = panels[1]
    assert r"& \multicolumn{2}{c}{S$^2$ Transformer} & \multicolumn{2}{c}{SFNO} & \multicolumn{2}{c}{GM-Attention (ours)} \\" in p
    pair = r"\multicolumn{1}{c}{rollout} & \multicolumn{1}{c}{teacher-forced}"
    assert rf"Lead (h) & {pair} & {pair} & {pair} \\" in p
    assert "1.0 &" in p and "0.5 &" not in p  # the first step is panel (a)'s native column
    row = next(line for line in p.splitlines() if line.startswith("1.0 &"))
    assert row.count(r"\bfseries") == 2  # SFNO best in both the rollout and the teacher-forced column
    assert r"\ell" not in p  # panel (b) is native-grid only
    assert "MSE" not in tex and "RMSE" not in tex and "Points" not in tex
    assert "S$^2$ Transformer, reported & 6.3 & 7.9 & -- & -- & --" in tex
    assert r"S$^2$ Transformer and SFNO on the 10-point $128\times256$ equiangular grid; GM-Attention (ours) on the 10-vertex level-5 icosphere" in tex
    assert tables.grid_phrase({"summary": {"x": {"grid": "g"}}, "per_seed": [{"label": "x/seed0", "n_nodes": 7}]}, ["x"]) == ["x on the 7-point g grid"]
    assert "scores 2000 in the same units" in tex  # persistence in the table's own scale, first step excluded


def test_without_a_rollout_the_benchmark_table_is_panel_a_alone() -> None:
    tex = tables.render(_data(steps=0))
    tables.check(tex)
    assert tex.count(r"\begin{table}") == 3
    assert len(_panels(tex)) == 1 and "Rollout" not in tex and "rollout, by degree" not in tex


def test_bold_marks_the_single_lowest_mean() -> None:
    summary = {"a": {"m": (1.0, 0.5)}, "b": {"m": (1.4, 0.1)}, "c": {"m": (1.2, 0.1)}, "d": {"m": (3.0, 0.1)}}
    assert tables.winners(summary, list(summary), lambda s: s["m"]) == {"a"}  # b and c sit inside a's sd, still not bold
    summary["c"] = {"m": (1.0, 0.3)}
    assert tables.winners(summary, list(summary), lambda s: s["m"]) == {"a", "c"}  # an exact tie shares the bold


def test_degree_bands_isolate_the_global_mean_and_clip_to_the_bandwidth() -> None:
    assert tables.degree_bands(42) == [(0, 1), (1, 4), (4, 24), (24, 43)]
    assert tables.degree_bands(10) == [(0, 1), (1, 4), (4, 11)]
    assert tables.band_label(0, 1) == "0" and tables.band_label(24, 43) == "24--42"


def test_check_rejects_a_row_with_the_wrong_number_of_cells() -> None:
    bad = "\\begin{tabular}{lrr}\n\\toprule\nA & 1 \\\\\n\\end{tabular}"
    with pytest.raises(AssertionError):
        tables.check(bad)


def test_zero_mean_twins_an_evaluation_recorded_are_left_out() -> None:
    data = _data()
    data["summary"]["gm_attention_depth1_zero_mean"] = _arm(1.5, 2, "icosphere:5")
    data["per_seed"].append({"label": "gm_attention_depth1_zero_mean/seed0", "n_nodes": 10})
    tex = tables.render(data)
    tables.check(tex)
    assert tables.ordered_arms(data["summary"]) == ["s2_transformer", "sfno", "gm_attention_depth1"]
    assert "zero-mean" not in tex and "zero_mean" not in tex
    assert len(_panels(tex)) == 2
    assert tex.count("GM-Attention (ours) &") == 3  # panel (a), the conventions table and the spectrum table, once each


def test_table_captions_report_actual_split_seeds_and_projection_limits() -> None:
    data = _data()
    data['split'] = 'val'
    for arm in data['summary'].values():
        arm['seeds'] = 2
    tex = tables.render(data)
    tables.check(tex)
    assert '4 val trajectories' in tex and 'over 2 seeds' in tex
    assert 'test trajectories' not in tex and 'over three seeds' not in tex
    assert 'norm gap' not in tex and 'fit residual' in tex and 'not separable' in tex
    assert 'condition number does not bound aliasing' in tex
    assert 'restarted at each lead time' in tex and "released script's own rollout evaluation" in tex
    assert 'per lead time and column in (b).' in tex and 'best mean in bold' in tex and 'within one seed sd' not in tex
    assert "removes only a model's" not in tex


def test_the_spectrum_table_reports_the_fit_residual_only_when_recorded() -> None:
    tex = tables.render(_data())
    c = tables.centered
    assert f"Model & {c('fit residual')} & {c('0')} & {c('1--3')} & {c('4--10')} & {c('0')} & {c('1--3')} & {c('4--10')}" in tex
    assert "norm gap" not in tex
    assert "SFNO & 0.100 &" in tex
    data = _data()
    for arm in data["summary"].values():
        del arm["metrics"]["fit_residual_fraction_L10"]
    tex = tables.render(data)
    tables.check(tex)
    assert "fit residual" not in tex and f"Model & {c('0')} & {c('1--3')} & {c('4--10')} & {c('0')} & {c('1--3')} & {c('4--10')}" in tex


def test_column_labels_are_centred_over_their_columns() -> None:
    tex = tables.render(_data())
    tables.check(tex)
    c = tables.centered
    assert f"Model & {c('Params')} & {c('native')} & {c(r'$\ell \le 10$')} & {c(r'$\varphi$')} & {c(r'$\zeta$')} & {c(r'$\delta$')} \\\\" in tex
    assert f"Model & {c('$L_1$')} & {c('$L_2$')} & {c(r'$\varphi$')} & {c(r'$\zeta$')} & {c(r'$\delta$')} \\\\" in tex
    # every column-label row is centred; the first (l) column's own label is not
    for line in tex.splitlines():
        if line.startswith(("Model &", "Lead (h) &")):
            cells = line[: -len(r"\\")].split(" & ")[1:]
            assert all(cell.strip().startswith(r"\multicolumn{1}{c}{") for cell in cells), line
