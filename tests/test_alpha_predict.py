# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-12
# License: see the LICENSE file in the repository root
"""Tests for K7 (`src/sammon/alpha_predict.py`) - pure alpha-prediction functions."""
from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

from src.sammon.alpha_predict import (
    ALPHA_MAX,
    ALPHA_MIN,
    load_alpha_pred_rule,
    nn_ratio_k1_from_D,
    predict_alpha,
    predict_alpha_log_linear,
    predict_alpha_two_threshold,
)


def test_nn_ratio_k1_from_D_matches_dataset_properties_definition() -> None:
    """nn_ratio_k1_from_D must give exactly the same result as the direct
    computation via K1 (`dataset_properties.nn_ratio(D, 1, median_all)`) - they
    share the same implementation (import, not a copy)."""
    from src.experiments.dataset_properties import nn_ratio

    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(50, 3))
    D = squareform(pdist(X))
    median_all = float(np.median(D[np.triu_indices(50, k=1)]))
    expected = nn_ratio(D, 1, median_all)
    assert np.isclose(nn_ratio_k1_from_D(D), expected)


def test_predict_alpha_log_linear_clips_to_bounds() -> None:
    # very negative log(nn_ratio) -> large raw -> clipped to ALPHA_MAX
    assert predict_alpha_log_linear(nn_ratio_k1=1e-6, a=0.0, b=-1.0) == ALPHA_MAX
    # very positive log(nn_ratio) -> negative raw -> clipped to ALPHA_MIN
    assert predict_alpha_log_linear(nn_ratio_k1=1e6, a=0.0, b=-1.0) == ALPHA_MIN


def test_predict_alpha_log_linear_matches_formula() -> None:
    a, b, nn = 1.0, -0.5, 0.2
    expected = float(np.clip(a + b * np.log(nn), 0.0, 3.0))
    assert np.isclose(predict_alpha_log_linear(nn, a, b), expected)


def test_predict_alpha_log_linear_rejects_nonpositive_nn_ratio() -> None:
    with pytest.raises(ValueError):
        predict_alpha_log_linear(0.0, a=1.0, b=1.0)


def test_predict_alpha_two_threshold_picks_correct_regime() -> None:
    t1, t2 = -2.0, -1.0
    # log(nn) < t1 -> a_low; nn = exp(-3) < exp(t1)
    assert predict_alpha_two_threshold(np.exp(-3.0), t1, t2, a_low=2.0, a_mid=1.0, a_high=0.0) == 2.0
    # t1 <= log(nn) < t2 -> a_mid
    assert predict_alpha_two_threshold(np.exp(-1.5), t1, t2, a_low=2.0, a_mid=1.0, a_high=0.0) == 1.0
    # log(nn) >= t2 -> a_high
    assert predict_alpha_two_threshold(np.exp(0.0), t1, t2, a_low=2.0, a_mid=1.0, a_high=0.0) == 0.0


def test_predict_alpha_two_threshold_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError):
        predict_alpha_two_threshold(0.1, t1=0.0, t2=0.0, a_low=1.0, a_mid=1.0, a_high=1.0)


def test_predict_alpha_two_threshold_clips_out_of_range_levels() -> None:
    # a_high = 5.0 is outside [0,3] -> must be clipped
    result = predict_alpha_two_threshold(np.exp(1.0), t1=-2.0, t2=-1.0, a_low=0.0, a_mid=0.0, a_high=5.0)
    assert result == ALPHA_MAX


def test_predict_alpha_dispatch_log_linear() -> None:
    rule = {"variant": "log_linear", "coefficients": {"a": 1.0, "b": -0.5}}
    nn = 0.2
    expected = predict_alpha_log_linear(nn, a=1.0, b=-0.5)
    assert predict_alpha(nn, rule) == expected


def test_predict_alpha_dispatch_two_threshold() -> None:
    rule = {"variant": "two_threshold", "coefficients": {"t1": -2.0, "t2": -1.0, "a_low": 2.0, "a_mid": 1.0, "a_high": 0.0}}
    assert predict_alpha(np.exp(-3.0), rule) == 2.0


def test_predict_alpha_rejects_unknown_variant() -> None:
    with pytest.raises(ValueError):
        predict_alpha(0.1, {"variant": "bogus", "coefficients": {}})


def test_load_alpha_pred_rule_fail_loud_on_missing_file(tmp_path) -> None:
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError):
        load_alpha_pred_rule(missing)


def test_load_alpha_pred_rule_fail_loud_on_invalid_variant(tmp_path) -> None:
    path = tmp_path / "rule.json"
    path.write_text(json.dumps({"variant": "not_a_real_variant", "coefficients": {}}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_alpha_pred_rule(path)


def test_load_alpha_pred_rule_roundtrip(tmp_path) -> None:
    path = tmp_path / "rule.json"
    rule = {"variant": "log_linear", "coefficients": {"a": 1.0, "b": -0.5}}
    path.write_text(json.dumps(rule), encoding="utf-8")
    loaded = load_alpha_pred_rule(path)
    assert loaded["variant"] == "log_linear"
    assert loaded["coefficients"]["a"] == 1.0
