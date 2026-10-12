"""The appendix's degree-band tables render from the evaluation JSONs."""

from __future__ import annotations

import pytest

from experiments.shallow_water_paper import degree_band_tables as dbt


def _summary(arms: dict[str, list[float]], steps: int = 2, hours: float = 1.0) -> dict:
    out = {}
    for arm, D in arms.items():
        D5 = [x * 3 for x in D]
        out[arm] = {"D": D, "rollout": [{"step": k + 1, "hours": hours * (k + 1), "D": D5} for k in range(steps)]}
    return out


def _doc(bandwidth: int, arms: dict[str, list[float]], **over) -> dict:
    d = {"bandwidth": bandwidth, "num_examples": 64, "lead_time_seconds": 3600.0 if bandwidth == 85 else 1800.0,
         "solver": [256, 512] if bandwidth == 85 else [128, 256], "summary": _summary(arms, hours=1.0 if bandwidth == 85 else 0.5)}
    d.update(over)
    return d


def test_both_tables_render_with_grouped_rows_and_fractions() -> None:
    easy = _doc(42, {"gm_attention_depth1": [1e-3] * 43, "sfno": [2e-3] * 43, "s2_transformer": [3e-3] * 43})
    ablation = _doc(85, {"gm_attention_full": [1.0] * 86, "gm_convolution_full": [1.0] * 86})
    zeroshot = _doc(85, {"gm_l5_100_icosphere5": [1.0] * 86, "gm_l5_100_icosphere4": [9.0] * 86, "sfno": [2.0] * 86, "sfno_zero_mean": [2.0] * 86})
    tex = dbt.render(easy, ablation, zeroshot)
    dbt.check(tex)
    assert tex.count(r"\begin{table}") == 2
    lines = tex.splitlines()
    # 128x256: baselines first, ours last; each model has a one-step and a 2.5 h row
    order = [line.split(" &")[0] for line in lines if line.startswith(("S$^2$", "SFNO &", "SOS2"))]
    assert order == [r"S$^2$ Transformer", "SFNO", "SOS2-T D1"]
    assert "0 & 1--3 & 4--11 & 12--23 & 24--42" in tex.replace(r"\multicolumn{1}{c}{", "").replace("}", "")
    assert "0 & 1--3 & 4--23 & 24--47 & 48--85" in tex.replace(r"\multicolumn{1}{c}{", "").replace("}", "")
    # a uniform D puts 1/43 of the energy in degree 0 and 3/43 in degrees 1-3
    row = next(line for line in lines if line.startswith("SOS2-T D1 &"))
    cells = [c.strip() for c in row[:-2].split("&")]
    assert cells[1:4] == ["1 step", "43.00", f"{1 / 43:.2f}"] and cells[4] == f"{3 / 43:.2f}"
    assert any(line.startswith(" & 1\\,h &") for line in lines)  # the rollout rows of the 256 table
    assert r"\emph{Depth-1 ablation}" in tex and r"\emph{Zero-shot grid transfer}" in tex  # the paper's two tables
    assert "icosphere4" not in tex and "zero" not in tex.lower().replace("zero-shot", "")  # dropped arms
    assert r"$SO(S^2)$-Conv" in tex and "SL (100 epochs) on the L5 icosphere (native)" in tex


def test_the_reference_models_are_named_and_each_group_is_one_papers_table() -> None:
    ablation = _doc(85, {"gm_attention_full": [1.0] * 86, "gm_convolution_full": [2.0] * 86})
    # the zero-shot study as `run.sh merge` builds it: both trainings of the
    # transferred arm, the references, and the depth-1 row borrowed from the
    # ablation it was scored in
    zeroshot = _doc(85, {"gm_l5_icosphere5": [1.0] * 86, "gm_l5_100_icosphere5": [0.5] * 86,
                         "sfno": [2.0] * 86, "s2_transformer_sl": [4.0] * 86,
                         "gm_attention_full": [1.0] * 86})
    tex = dbt.render(None, ablation, zeroshot)
    dbt.check(tex)
    # the reference model has a name, not its key
    assert r"S$^2$ Transformer, single level ($128\times256$, native) & 1 step" in tex
    assert "s2\\_transformer\\_sl" not in tex and "gm\\_attention\\_full" not in tex
    # each group is one of the paper's tables, so the depth-1 arm -- a row
    # of both -- is in both, under each table's own name for it
    assert tex.count(r"D1 $SO(S^2)$-MHSA, $\rho_0^{16} \oplus \rho_1^{16} \oplus \rho_2^{16}$ & 1 step") == 1
    assert tex.count("D1 on the L5 icosphere (native) & 1 step") == 1
    ablation_at = tex.index(r"\emph{Depth-1 ablation}")
    transfer_at = tex.index(r"\emph{Zero-shot grid transfer}")
    assert ablation_at < tex.index(r"D1 $SO(S^2)$-MHSA, $\rho_0^{16}") < transfer_at
    assert transfer_at < tex.index("D1 on the L5 icosphere (native)")
    # and the caption says the two rows are one evaluation, not two
    assert "one evaluation shown twice, not two" in tex
    # the study reports one training of the transferred arm, and the row
    # says which; the 50-epoch rows the JSON still holds are not reported
    assert "SL (100 epochs) on the L5 icosphere (native)" in tex
    assert "50 epochs" not in tex and "gm\\_l5\\_icosphere5" not in tex
    assert "gm_l5_icosphere5" in zeroshot["summary"]  # held, not reported
    # --drop leaves out more without editing DROPPED
    fewer = dbt.render(None, ablation, zeroshot, drop=("gm_l5_100_icosphere5",))
    dbt.check(fewer)
    assert "SL (100 epochs)" not in fewer and r"S$^2$ Transformer, single level" in fewer


def test_the_256_studies_must_share_their_protocol() -> None:
    ablation = _doc(85, {"gm_attention_full": [1.0] * 86})
    zeroshot = _doc(85, {"gm_l5_icosphere5": [1.0] * 86}, num_examples=32)
    with pytest.raises(SystemExit, match="num_examples"):
        dbt.render(None, ablation, zeroshot)


def test_a_single_study_renders_alone() -> None:
    tex = dbt.render(_doc(42, {"sfno": [1.0] * 43}), None, None)
    dbt.check(tex)
    assert tex.count(r"\begin{table}") == 1 and "tab:degree-bands-128" in tex
