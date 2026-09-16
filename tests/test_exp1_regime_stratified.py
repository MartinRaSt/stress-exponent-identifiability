# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-13
# License: see the LICENSE file in the repository root
"""Tests for `src/experiments/exp1_regime_stratified.py` - the pure
computational functions (regime assignment per the JSON rule, the shape of
the resulting long table) without file I/O (no reading of real CSV files -
see ~/.claude/CLAUDE.md "never read large CSV files into context")."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from src.experiments.exp1_regime_stratified import (
    METHODS,
    METRICS,
    PAIR_BASELINES,
    PRED_METHOD,
    REGIME_ORDER,
    _LONG_TABLE_COLUMNS,
    build_dataset_regime_table,
    build_regime_stratified_table,
    classify_regime,
    regime_alpha_pred,
)

_RULE = {
    "variant": "two_threshold",
    "coefficients": {"t1": -2.214482722197313, "t2": -1.6491148519193841, "a_low": 2.5, "a_mid": 0.75, "a_high": 0.0},
}


def _null_logger() -> logging.Logger:
    logger = logging.getLogger("test_exp1_regime_stratified")
    logger.addHandler(logging.NullHandler())
    return logger


def test_classify_regime_matches_thresholds_from_rule() -> None:
    """Boundary and interior values of each band - the classification must
    exactly match the t1/t2 thresholds from the JSON (no hardcoded boundaries)."""
    t1, t2 = _RULE["coefficients"]["t1"], _RULE["coefficients"]["t2"]
    nn_ratios = np.exp([t1 - 1.0, t1 - 0.01, (t1 + t2) / 2.0, t2 + 0.01, t2 + 1.0])
    regimes = classify_regime(nn_ratios, _RULE)
    assert list(regimes) == ["low_ratio", "low_ratio", "mid_ratio", "high_ratio", "high_ratio"]


def test_classify_regime_rejects_non_two_threshold_variant() -> None:
    rule = {"variant": "log_linear", "coefficients": {"a": 1.0, "b": -0.5}}
    with pytest.raises(ValueError):
        classify_regime(np.array([0.1, 0.2]), rule)


def test_classify_regime_rejects_non_positive_nn_ratio() -> None:
    with pytest.raises(ValueError):
        classify_regime(np.array([0.1, 0.0]), _RULE)


def test_regime_alpha_pred_reads_coefficients_from_rule() -> None:
    mapping = regime_alpha_pred(_RULE)
    assert mapping == {"low_ratio": 2.5, "mid_ratio": 0.75, "high_ratio": 0.0}


def test_build_dataset_regime_table_assigns_all_three_regimes() -> None:
    t1, t2 = _RULE["coefficients"]["t1"], _RULE["coefficients"]["t2"]
    props = pd.DataFrame({
        "dataset": ["a_low_ratio", "b_mid_ratio", "c_high_ratio"],
        "nn_ratio_k1": np.exp([t1 - 1.0, (t1 + t2) / 2.0, t2 + 1.0]),
    })
    out = build_dataset_regime_table(props, _RULE)
    assert set(out["regime"]) == set(REGIME_ORDER)
    assert out.loc[out["dataset"] == "a_low_ratio", "alpha_pred_regime"].item() == 2.5
    assert out.loc[out["dataset"] == "c_high_ratio", "alpha_pred_regime"].item() == 0.0


def _make_synthetic_e1(n_per_regime: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Builds a synthetic exp1-like DataFrame (all METHODS, 2 seeds) and a
    corresponding dataset_regime table with 3 regimes of `n_per_regime`
    datasets each, so that Wilcoxon (n>=5) has enough data in every regime."""
    t1, t2 = _RULE["coefficients"]["t1"], _RULE["coefficients"]["t2"]
    log_nn_by_regime = {
        "low_ratio": np.linspace(t1 - 3.0, t1 - 0.1, n_per_regime),
        "mid_ratio": np.linspace(t1 + 0.1, t2 - 0.1, n_per_regime),
        "high_ratio": np.linspace(t2 + 0.1, t2 + 3.0, n_per_regime),
    }
    rows_props = []
    rows_e1 = []
    rng = np.random.default_rng(0)
    for regime, log_nns in log_nn_by_regime.items():
        for i, log_nn in enumerate(log_nns):
            dataset = f"{regime}_{i}"
            rows_props.append({"dataset": dataset, "nn_ratio_k1": float(np.exp(log_nn))})
            for method in METHODS:
                base = 0.5 if method != "sammon_alpha_pred" else 0.55  # alpha_pred slightly better -> measured by the test
                for seed in (0, 1):
                    rows_e1.append({
                        "dataset": dataset, "method": method, "seed": seed, "status": "ok",
                        "auc_rnx": base + 0.01 * rng.normal(),
                        "stress_scale_invariant": (1.0 - base) + 0.01 * rng.normal(),
                    })
    props = pd.DataFrame(rows_props)
    e1_ok = pd.DataFrame(rows_e1)
    return e1_ok, props


def test_build_regime_stratified_table_shape_and_columns() -> None:
    e1_ok, props = _make_synthetic_e1(n_per_regime=6)
    dataset_regime_df = build_dataset_regime_table(props, _RULE)
    table = build_regime_stratified_table(e1_ok, dataset_regime_df, METHODS, METRICS, _null_logger())

    assert list(table.columns) == _LONG_TABLE_COLUMNS
    n_method_rows = len(REGIME_ORDER) * len(METRICS) * len(METHODS)
    n_pair_rows = len(REGIME_ORDER) * len(METRICS) * len(PAIR_BASELINES)
    assert table.shape[0] == n_method_rows + n_pair_rows
    assert set(table["regime"]) == set(REGIME_ORDER)
    assert set(table["row_type"]) == {"method", "paired_diff"}

    # each regime has exactly 6 datasets -> Wilcoxon (n>=5) must be computed (not NaN)
    pair_rows = table[table["row_type"] == "paired_diff"]
    assert (pair_rows["n"] >= 5).all()
    assert pair_rows["wilcoxon_pvalue"].notna().all()

    # alpha_pred was synthetically constructed with a higher 'base' (0.55
    # vs. 0.5) than both baselines -> for auc_rnx (higher=better) the
    # difference is ALWAYS positive, for stress (lower=better,
    # stress=1-base) the difference is ALWAYS negative (n_positive==0) - we
    # verify both directions so the test catches a possible sign swap of the
    # difference.
    pred_vs_alpha0 = pair_rows[pair_rows["method"] == f"{PRED_METHOD} - sammon_alpha0_smacof"]
    auc_rows = pred_vs_alpha0[pred_vs_alpha0["metric"] == "auc_rnx"]
    stress_rows = pred_vs_alpha0[pred_vs_alpha0["metric"] == "stress_scale_invariant"]
    assert (auc_rows["n_positive"] == auc_rows["n"]).all(), "alpha_pred should have higher auc_rnx than alpha0 on all synthetic datasets"
    assert (stress_rows["n_positive"] == 0).all(), "alpha_pred should have lower stress than alpha0 on all synthetic datasets"


def test_build_regime_stratified_table_wilcoxon_nan_below_min_n() -> None:
    """A regime with only 2 datasets -> the paired Wilcoxon test must be NaN (n<5), no silent fallback."""
    e1_ok, props = _make_synthetic_e1(n_per_regime=2)
    dataset_regime_df = build_dataset_regime_table(props, _RULE)
    table = build_regime_stratified_table(e1_ok, dataset_regime_df, METHODS, METRICS, _null_logger())
    pair_rows = table[table["row_type"] == "paired_diff"]
    assert pair_rows["wilcoxon_pvalue"].isna().all()


def test_build_regime_stratified_table_missing_method_column_is_dropped_silently_not_fabricated() -> None:
    """If a (metric, method) combination is missing from `wide` (the method
    has no 'ok' row at all in the given input), the method row for that
    metric is simply not generated (see `_regime_method_summary_rows`) - no
    fabricated value is filled in."""
    e1_ok, props = _make_synthetic_e1(n_per_regime=6)
    e1_ok = e1_ok[e1_ok["method"] != "umap"]  # simulating a missing method in the data
    dataset_regime_df = build_dataset_regime_table(props, _RULE)
    table = build_regime_stratified_table(e1_ok, dataset_regime_df, METHODS, METRICS, _null_logger())
    assert "umap" not in set(table.loc[table["row_type"] == "method", "method"])
