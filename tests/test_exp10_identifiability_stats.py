# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-17
# License: see the LICENSE file in the repository root
"""Tests for src/experiments/exp10_identifiability_stats.py (analysis items
2-6 of reserse/2026-09-17_zostreni_propozice2.md section 10, left out of
exp10_identifiability_check.py itself). The regression and TOST analyses are
exercised on SYNTHETIC data with a KNOWN ground-truth result (project rule:
"k novym funkcim dopln testy, zejmena regrese a TOST na syntetickych datech
se znamym vysledkem")."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from src.experiments.exp10_identifiability_stats import (
    STATS_COLUMNS,
    TABLE_COLUMNS,
    _safe_extend,
    analyze_h1_spearman,
    analyze_h2_phi_low,
    analyze_h3_tost_high,
    analyze_quadratic_law_fraction,
    analyze_regression_slack_vs_pairs,
    build_per_dataset_table,
    compute_stats,
    dataset_level_frame,
)

_LOGGER = logging.getLogger("test_exp10_identifiability_stats")


def _make_table_row(dataset: str, alpha: float, **overrides) -> dict:
    base = {
        "dataset": dataset, "alpha": alpha, "n_samples": 100, "rho_nn": 0.2, "s_u": 0.5, "c_alpha": 0.3,
        "gamma_tilde_Y0": 1.5, "gamma_tilde_Ya": 1.6, "r_Y0": 0.1, "r_Ya": -0.2,
        "bound_v1": 0.5, "Delta_prime": 0.05, "tight_v1": 0.1, "tight_sandwich": 0.02,
        "I0_exact": 0.02, "cert_lower": -0.5, "alpha_eta": np.nan, "alpha_dagger": 5.0,
        "bound_nontrivial": True, "N_over_n_near": 100.0, "hessian_skipped": False,
        "Delta": 0.05, "pred_quadratic": 0.05, "quadratic_law_holds": True,
        "G_auc_oracle": 0.1, "G_pred": 0.05, "stratum": "low",
        "S1": 1.1, "S2": 200.0, "S3": 1.05, "S4": 2.0, "status": "ok",
    }
    base.update(overrides)
    return base


# ============================================================================
# build_per_dataset_table
# ============================================================================


def test_build_per_dataset_table_filters_and_sorts() -> None:
    df_ok = pd.DataFrame([
        _make_table_row("b", 1.0), _make_table_row("a", 1.0), _make_table_row("a", 2.0), _make_table_row("a", 0.5),
    ])
    table = build_per_dataset_table(df_ok, alpha_table=[1.0, 2.0], logger=_LOGGER)
    assert list(table.columns) == TABLE_COLUMNS
    assert table.shape[0] == 3  # alpha=0.5 row excluded (not in alpha_table)
    assert list(table["dataset"]) == ["a", "a", "b"]
    assert list(table["alpha"]) == [1.0, 2.0, 1.0]


def test_build_per_dataset_table_warns_but_does_not_fail_on_missing_alpha(caplog) -> None:
    df_ok = pd.DataFrame([_make_table_row("a", 1.0)])
    with caplog.at_level(logging.WARNING, logger=_LOGGER.name):
        table = build_per_dataset_table(df_ok, alpha_table=[1.0, 3.0], logger=_LOGGER)
    assert table.shape[0] == 1
    assert any("3.0" in r.message for r in caplog.records)


def test_build_per_dataset_table_raises_if_none_present() -> None:
    df_ok = pd.DataFrame([_make_table_row("a", 1.0)])
    with pytest.raises(ValueError):
        build_per_dataset_table(df_ok, alpha_table=[2.0], logger=_LOGGER)


# ============================================================================
# dataset_level_frame
# ============================================================================


def test_dataset_level_frame_one_row_per_dataset() -> None:
    df_ok = pd.DataFrame([_make_table_row("a", 1.0), _make_table_row("a", 2.0), _make_table_row("b", 1.0)])
    ds = dataset_level_frame(df_ok, reference_alpha=1.0)
    assert list(ds.index) == ["a", "b"]
    assert "c_1" in ds.columns  # c_alpha renamed to c_1 (reserse notation)


def test_dataset_level_frame_raises_on_missing_reference_alpha() -> None:
    df_ok = pd.DataFrame([_make_table_row("a", 2.0)])
    with pytest.raises(ValueError):
        dataset_level_frame(df_ok, reference_alpha=1.0)


def test_dataset_level_frame_raises_on_duplicate_rows() -> None:
    df_ok = pd.DataFrame([_make_table_row("a", 1.0), _make_table_row("a", 1.0)])
    with pytest.raises(ValueError):
        dataset_level_frame(df_ok, reference_alpha=1.0)


# ============================================================================
# analyze_regression_slack_vs_pairs (synthetic, KNOWN slope, via a temp E8 CSV)
# ============================================================================


def test_analyze_regression_slack_vs_pairs_recovers_known_slope(tmp_path, monkeypatch) -> None:
    """tight_a := N_over_n_near^(-2) exactly (noiseless) -> log(1/tight_a) =
    2*log(N_over_n_near), so the OLS slope must recover 2.0 exactly."""
    rng = np.random.default_rng(0)
    datasets = [f"d{i}" for i in range(8)]
    n_over_n_near = {d: float(rng.uniform(10.0, 500.0)) for d in datasets}

    df_ok_rows = [_make_table_row(d, 1.0, N_over_n_near=n_over_n_near[d]) for d in datasets]
    df_ok = pd.DataFrame(df_ok_rows)

    e8_rows = []
    for d in datasets:
        for seed in range(3):
            e8_rows.append({
                "dataset": d, "alpha": 1.0, "seed": seed,
                "tight_a": n_over_n_near[d] ** (-2.0), "p2_holds": True,
            })
    e8_path = tmp_path / "exp8_prop2_check_results.csv"
    pd.DataFrame(e8_rows).to_csv(e8_path, index=False)

    monkeypatch.setattr("src.experiments.exp10_identifiability_stats.results_csv_path", lambda name: e8_path)

    out = analyze_regression_slack_vs_pairs(df_ok, mode="smoke", n_boot=300, seed=1, ci_level=0.95, logger=_LOGGER)
    slope_row = next(r for r in out if r["label"] == "slope")
    assert slope_row["value"] == pytest.approx(2.0, abs=1e-6)
    assert slope_row["ci_low"] <= 2.0 + 1e-6 <= slope_row["ci_high"] + 1e-6 or slope_row["ci_low"] == pytest.approx(slope_row["ci_high"])
    assert slope_row["n"] == 24


def test_analyze_regression_slack_vs_pairs_missing_e8_file_raises(tmp_path, monkeypatch) -> None:
    df_ok = pd.DataFrame([_make_table_row("a", 1.0)])
    missing = tmp_path / "does_not_exist.csv"
    monkeypatch.setattr("src.experiments.exp10_identifiability_stats.results_csv_path", lambda name: missing)
    with pytest.raises(FileNotFoundError):
        analyze_regression_slack_vs_pairs(df_ok, mode="smoke", n_boot=10, seed=0, ci_level=0.95, logger=_LOGGER)


# ============================================================================
# analyze_h1_spearman / analyze_h2_phi_low / analyze_h3_tost_high (synthetic)
# ============================================================================


def test_analyze_h1_spearman_monotone_relationship_gives_rho_one() -> None:
    ds = pd.DataFrame({
        "rho_nn": [0.1, 0.2, 0.3, 0.4, 0.5],
        "c_1": [1.0, 1.0, 1.0, 1.0, 1.0],
        "gamma_tilde_Y0": [1.0, 2.0, 3.0, 4.0, 5.0],
        "I0_exact": [5.0, 4.0, 3.0, 2.0, 1.0],
        "G_auc_oracle": [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    out = analyze_h1_spearman(ds, n_perm=500, seed=0, logger=_LOGGER)
    by_label = {r["label"]: r for r in out}
    assert by_label["rho_nn_vs_G_auc_oracle"]["value"] == pytest.approx(1.0)
    assert by_label["cGammaTildeZero_vs_G_auc_oracle"]["value"] == pytest.approx(1.0)  # c_1*gammaY0 is monotone in gamma_tilde_Y0 (c_1 constant)
    assert by_label["I0exact_vs_G_auc_oracle"]["value"] == pytest.approx(-1.0)  # I0 decreasing while G_auc_oracle increasing


def test_analyze_h2_phi_low_known_ratio_with_ci() -> None:
    """Every 'low' dataset has G_pred/G_auc_oracle == 0.5 exactly -> phi=0.5,
    a degenerate (zero-width) bootstrap CI."""
    ds = pd.DataFrame({
        "stratum": ["low"] * 6 + ["high"] * 2,
        "G_pred": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0, 100.0],
        "G_auc_oracle": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 1.0, 1.0],
    }, index=[f"d{i}" for i in range(8)])
    out = analyze_h2_phi_low(ds, n_boot=300, seed=0, ci_level=0.95, logger=_LOGGER)
    assert len(out) == 1
    assert out[0]["value"] == pytest.approx(0.5)
    assert out[0]["ci_low"] == pytest.approx(0.5, abs=1e-9)
    assert out[0]["ci_high"] == pytest.approx(0.5, abs=1e-9)
    assert out[0]["n"] == 6


def test_analyze_h2_phi_low_raises_if_no_low_stratum() -> None:
    ds = pd.DataFrame({"stratum": ["high", "mid"], "G_pred": [1.0, 2.0], "G_auc_oracle": [1.0, 1.0]}, index=["a", "b"])
    with pytest.raises(ValueError):
        analyze_h2_phi_low(ds, n_boot=100, seed=0, ci_level=0.95, logger=_LOGGER)


def test_analyze_h3_tost_high_equivalence_accepted_for_small_g_pred() -> None:
    """G_pred tightly clustered around 0, well within a generous delta_eq
    (median G_auc_oracle in the 'high' stratum) - TOST must ACCEPT
    equivalence (p_tost small, i.e. reject the non-equivalence null)."""
    rng = np.random.default_rng(3)
    ds = pd.DataFrame({
        "stratum": ["high"] * 10 + ["low"] * 2,
        "G_pred": list(rng.normal(loc=0.0, scale=0.001, size=10)) + [0.5, 0.5],
        "G_auc_oracle": [0.2] * 10 + [0.9, 0.9],
    }, index=[f"d{i}" for i in range(12)])
    out = analyze_h3_tost_high(ds, logger=_LOGGER)
    assert len(out) == 1
    assert out[0]["n"] == 10
    assert out[0]["pvalue"] < 0.01  # equivalence clearly accepted: delta_eq=0.2 >> spread of G_pred


def test_analyze_h3_tost_high_equivalence_rejected_for_large_g_pred() -> None:
    """G_pred far outside a tiny delta_eq - TOST must NOT find equivalence
    (large p_tost)."""
    ds = pd.DataFrame({
        "stratum": ["high"] * 6,
        "G_pred": [0.5, 0.52, 0.49, 0.51, 0.50, 0.48],
        "G_auc_oracle": [0.001] * 6,  # delta_eq ~ 0.001, but G_pred ~ 0.5
    }, index=[f"d{i}" for i in range(6)])
    out = analyze_h3_tost_high(ds, logger=_LOGGER)
    assert out[0]["pvalue"] > 0.5


def test_analyze_h3_tost_high_raises_if_too_few_high_datasets() -> None:
    ds = pd.DataFrame({"stratum": ["low"], "G_pred": [0.1], "G_auc_oracle": [0.2]}, index=["a"])
    with pytest.raises(ValueError):
        analyze_h3_tost_high(ds, logger=_LOGGER)


# ============================================================================
# analyze_quadratic_law_fraction
# ============================================================================


def test_analyze_quadratic_law_fraction_pooled_and_per_alpha() -> None:
    df_ok = pd.DataFrame([
        _make_table_row("a", 1.0, quadratic_law_holds=True, hessian_skipped=False),
        _make_table_row("b", 1.0, quadratic_law_holds=False, hessian_skipped=False),
        _make_table_row("a", 2.0, quadratic_law_holds=True, hessian_skipped=False),
        _make_table_row("b", 2.0, quadratic_law_holds=True, hessian_skipped=False),
        _make_table_row("a", 0.0, quadratic_law_holds=True, hessian_skipped=False),  # alpha=0 excluded
        _make_table_row("c", 1.0, quadratic_law_holds=True, hessian_skipped=True),  # skipped (Hessian not computed) -> excluded
    ])
    out = analyze_quadratic_law_fraction(df_ok, tol=0.3)
    by_label = {r["label"]: r for r in out}
    assert by_label["pooled"]["value"] == pytest.approx(3.0 / 4.0)  # 3 True out of {a@1,b@1,a@2,b@2}
    assert by_label["pooled"]["n"] == 4
    assert by_label["alpha_1"]["value"] == pytest.approx(0.5)
    assert by_label["alpha_2"]["value"] == pytest.approx(1.0)


def test_analyze_quadratic_law_fraction_empty_returns_empty_list() -> None:
    df_ok = pd.DataFrame([_make_table_row("a", 0.0)])  # only alpha=0 rows
    out = analyze_quadratic_law_fraction(df_ok, tol=0.3)
    assert out == []


# ============================================================================
# _safe_extend / compute_stats resilience (smoke-like: too few datasets)
# ============================================================================


def test_safe_extend_appends_on_success_and_warns_on_failure(caplog) -> None:
    rows: list[dict] = []
    _safe_extend(rows, "ok block", lambda: [{"a": 1}], _LOGGER)
    assert rows == [{"a": 1}]

    def _boom():
        raise ValueError("synthetic failure")

    with caplog.at_level(logging.WARNING, logger=_LOGGER.name):
        _safe_extend(rows, "failing block", _boom, _LOGGER)
    assert rows == [{"a": 1}]  # unchanged - the failing block contributed nothing
    assert any("failing block" in r.message for r in caplog.records)


def test_compute_stats_schema_and_resilience_with_single_dataset(tmp_path, monkeypatch) -> None:
    """A single dataset (like --smoke with n_datasets=1) cannot support
    H1 (Spearman needs n>=3 datasets), H2/H3 (need >=1 low/high stratum
    dataset with >=2 for H3) - compute_stats must still return a
    (possibly partial) DataFrame with the correct schema, not raise."""
    df_ok = pd.DataFrame([_make_table_row("solo", 1.0, stratum="low")])
    e8_path = tmp_path / "e8.csv"
    pd.DataFrame([{"dataset": "solo", "alpha": 1.0, "seed": 0, "tight_a": 0.5, "p2_holds": True}]).to_csv(e8_path, index=False)
    monkeypatch.setattr("src.experiments.exp10_identifiability_stats.results_csv_path", lambda name: e8_path)

    cfg = {"seed": 0, "n_perm": 50, "n_boot": 50, "ci_level": 0.95, "alpha_table": [1.0]}
    exp10_cfg = {"quadratic_law_tol": 0.3}
    out = compute_stats(df_ok, mode="smoke", cfg=cfg, exp10_cfg=exp10_cfg, logger=_LOGGER)
    assert list(out.columns) == STATS_COLUMNS
    # with a single dataset: the regression needs x-variation across
    # datasets (constant here -> skipped) and H3 needs a non-empty 'high'
    # stratum (the lone dataset is 'low' -> skipped); H1 (returns NaN
    # gracefully for n<3, does not raise) and H2 (n=1 'low' dataset is
    # enough for a degenerate median+CI) and the quadratic-law fraction
    # still produce rows - the important property is that NOTHING raises.
    assert "regression_slack_vs_pairs" not in set(out["analysis"])
    assert "H3_tost_high" not in set(out["analysis"])
    assert "quadratic_law_fraction" in set(out["analysis"])
