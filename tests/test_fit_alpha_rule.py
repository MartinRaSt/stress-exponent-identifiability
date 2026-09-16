# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-12
# License: see the LICENSE file in the repository root
"""K7 tests (`src/experiments/fit_alpha_rule.py`) - pure computational
functions (fitting the rule, interpolation on the grid) without file I/O."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.common.config import get_path
from src.experiments.config_experiments import load_experiments_config
from src.experiments.fit_alpha_rule import _fit_log_linear, _fit_two_threshold, _interp_auc, exclude_holdout_datasets


def test_fit_log_linear_recovers_known_coefficients() -> None:
    """Synthetic data exactly on the line alpha = a + b*ln(nn_ratio) -> OLS
    must return (a,b) (up to numerical precision). `_fit_log_linear` itself
    does NOT clip (clipping to [0,3] is only done by
    `predict_alpha_log_linear` when applying the rule) - the nn_ratio range
    is chosen here so the resulting alpha_star lies entirely within [0,3],
    avoiding an unintended dependency on clipping, which would bias the OLS fit."""
    a_true, b_true = 1.5, -0.3
    nn_ratios = np.array([0.15, 0.25, 0.4, 0.6, 0.8, 1.0])
    alpha_star = a_true + b_true * np.log(nn_ratios)
    assert np.all((alpha_star >= 0.0) & (alpha_star <= 3.0)), "the test data must lie within [0,3] (without clipping)"
    train = pd.DataFrame({"nn_ratio_k1": nn_ratios, "alpha_star": alpha_star})
    coef = _fit_log_linear(train)
    assert np.isclose(coef["a"], a_true, atol=1e-6)
    assert np.isclose(coef["b"], b_true, atol=1e-6)


def test_fit_two_threshold_separates_three_clear_regimes() -> None:
    """3 clearly separated nn_ratio clusters (small/medium/large) with a
    constant alpha_star in each cluster - the fit must find thresholds
    separating the clusters and a_low/a_mid/a_high exactly matching the
    cluster values."""
    low = np.exp(np.linspace(-4.0, -3.5, 5))
    mid = np.exp(np.linspace(-2.0, -1.5, 5))
    high = np.exp(np.linspace(0.0, 0.5, 5))
    nn_ratios = np.concatenate([low, mid, high])
    alpha_star = np.concatenate([np.full(5, 2.0), np.full(5, 1.0), np.full(5, 0.0)])
    train = pd.DataFrame({"nn_ratio_k1": nn_ratios, "alpha_star": alpha_star})

    quantile_grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    coef = _fit_two_threshold(train, quantile_grid)
    assert coef["a_low"] == 2.0
    assert coef["a_mid"] == 1.0
    assert coef["a_high"] == 0.0
    assert coef["t1"] < coef["t2"]


def test_fit_two_threshold_raises_when_no_valid_split() -> None:
    """All nn_ratio values identical -> no threshold pair splits the data
    into 3 non-empty bands -> a fail-loud ValueError."""
    train = pd.DataFrame({"nn_ratio_k1": np.full(10, 0.1), "alpha_star": np.zeros(10)})
    with pytest.raises(ValueError):
        _fit_two_threshold(train, [0.1, 0.5, 0.9])


def test_interp_auc_linear_between_grid_points() -> None:
    grids = {"ds": (np.array([0.0, 1.0, 2.0]), np.array([0.4, 0.6, 0.5]))}
    assert np.isclose(_interp_auc(grids, "ds", 0.5), 0.5)  # midpoint between 0.4 and 0.6
    assert np.isclose(_interp_auc(grids, "ds", 0.0), 0.4)
    assert np.isclose(_interp_auc(grids, "ds", 2.0), 0.5)


# --------------------------------------------------------------------------
# Q1 step 2 (reserse/2026-09-14_specifikace_rozsireni_q1.md, A.5) - hold-out
# excluded from the rule fit + a frozen JSON
# --------------------------------------------------------------------------

def test_exclude_holdout_datasets_removes_only_listed_rows() -> None:
    df = pd.DataFrame({"dataset": ["iris", "swiss_roll", "shuttle", "abalone"], "nn_ratio_k1": [0.1, 0.04, 0.03, 0.09]})
    out = exclude_holdout_datasets(df, {"shuttle", "abalone"})
    assert sorted(out["dataset"]) == ["iris", "swiss_roll"]


def test_exclude_holdout_datasets_noop_when_empty() -> None:
    df = pd.DataFrame({"dataset": ["iris", "swiss_roll"], "nn_ratio_k1": [0.1, 0.04]})
    out = exclude_holdout_datasets(df, set())
    pd.testing.assert_frame_equal(out, df)


def test_q1_holdout_datasets_never_overlap_core_exp1_datasets() -> None:
    """A.5: the hold-out candidates (sammon_alpha_pred.holdout_datasets) must
    NOT overlap with the core datasets on which the rule is fitted
    (exp1_dr_benchmark.datasets, anchor &exp1_datasets) - otherwise the rule
    fit would not be truly out-of-sample."""
    cfg = load_experiments_config()
    core = set(cfg["exp1_dr_benchmark"]["datasets"])
    holdout = set(cfg["sammon_alpha_pred"]["holdout_datasets"])
    overlap = core & holdout
    assert not overlap, f"hold-out datasets overlap with the core datasets used to fit the rule: {overlap}"
    assert holdout, "sammon_alpha_pred.holdout_datasets is empty - Q1 step 2 candidates are missing from the config."


def test_alpha_pred_rule_matches_frozen_snapshot() -> None:
    """A.5: the production results/data/alpha_pred_rule.json must have THE
    SAME coefficients as the frozen copy
    results/data/alpha_pred_rule_frozen_20260913.json (tolerance 1e-12).
    This test FAILS if the rule changes (e.g. if hold-out data accidentally
    leaked into the fit during a --full run with Q1 candidates) - the
    author must then consciously decide whether to update the frozen copy."""
    live_path = get_path("results_data_dir") / "alpha_pred_rule.json"
    frozen_path = get_path("results_data_dir") / "alpha_pred_rule_frozen_20260913.json"
    if not live_path.exists() or not frozen_path.exists():
        pytest.skip(f"missing {live_path} or {frozen_path} (first run fit_alpha_rule --full)")
    live = json.loads(live_path.read_text(encoding="utf-8"))
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    assert live["variant"] == frozen["variant"]
    for key in ("t1", "t2", "a_low", "a_mid", "a_high"):
        assert live["coefficients"][key] == pytest.approx(frozen["coefficients"][key], abs=1e-12), (
            f"coefficient '{key}' has changed relative to the frozen rule ({live['coefficients'][key]} != "
            f"{frozen['coefficients'][key]}) - verify that hold-out datasets are not included in the fit."
        )
