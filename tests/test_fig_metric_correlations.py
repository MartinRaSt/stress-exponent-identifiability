# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-16
# License: see the LICENSE file in the repository root
"""Tests for the metric correlation analysis
(`src/figures/fig_metric_correlations.py`): orienting metrics to "more =
better", pairwise Spearman computation with missing values, and clustering
practically interchangeable metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.figures import fig_metric_correlations as fmc


def test_orient_flips_only_lower_is_better() -> None:
    df = pd.DataFrame({"good": [1.0, 2.0], "cost": [3.0, 4.0]})
    out = fmc._orient(df, ["good", "cost"], ["cost"])
    assert out["good"].tolist() == [1.0, 2.0]
    assert out["cost"].tolist() == [-3.0, -4.0]


def test_spearman_matrix_diagonal_is_one_and_symmetric() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 4.0, 6.0, 8.0], "c": [4.0, 3.0, 2.0, 1.0]})
    rho, n_used = fmc._spearman_matrix(df, ["a", "b", "c"])
    assert rho.loc["a", "a"] == pytest.approx(1.0)
    assert rho.loc["a", "b"] == pytest.approx(1.0)      # monotonic transformation
    assert rho.loc["a", "c"] == pytest.approx(-1.0)     # reversed order
    assert rho.loc["a", "b"] == pytest.approx(rho.loc["b", "a"])
    assert n_used.loc["a", "b"] == 4


def test_spearman_matrix_uses_pairwise_complete_observations() -> None:
    """Cluster-geometry metrics exist only for part of the datasets - a pair
    is computed from rows where BOTH values are present, not after dropping
    the whole row."""
    df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": [1.0, 2.0, 3.0, np.nan],
        "c": [np.nan, np.nan, 1.0, 2.0],
    })
    rho, n_used = fmc._spearman_matrix(df, ["a", "b", "c"])
    assert n_used.loc["a", "b"] == 3
    assert n_used.loc["a", "c"] == 2
    assert np.isnan(rho.loc["a", "c"])  # fewer than 3 observations -> not computed


def test_cluster_metrics_groups_redundant_pairs() -> None:
    """Two practically interchangeable metrics must fall into one cluster,
    an independent metric into another."""
    metrics = ["a", "b", "c"]
    rho = pd.DataFrame(
        [[1.00, 0.98, 0.05],
         [0.98, 1.00, 0.02],
         [0.05, 0.02, 1.00]],
        index=metrics, columns=metrics,
    )
    clusters = fmc._cluster_metrics(rho, threshold=0.9)
    assert clusters["a"] == clusters["b"]
    assert clusters["c"] != clusters["a"]


def test_long_form_has_one_row_per_pair() -> None:
    metrics = ["a", "b", "c"]
    rho = pd.DataFrame(np.eye(3), index=metrics, columns=metrics)
    n_used = pd.DataFrame(np.full((3, 3), 7), index=metrics, columns=metrics)
    out = fmc._long_form(rho, n_used, "all")
    assert len(out) == 3  # three pairs from three metrics
    assert set(out.columns) == {"family", "metric_a", "metric_b", "spearman_rho", "n_units", "abs_rho"}


def test_config_section_exists_and_is_consistent() -> None:
    """Both the metric list and lower_is_better must be in the config (fail
    loud) and lower_is_better must be a subset of the metrics."""
    cfg = fmc._config()
    assert cfg["metrics"], "fig_metric_correlations.metrics must not be empty"
    assert set(cfg["lower_is_better"]).issubset(set(cfg["metrics"]))
    assert 0.0 < float(cfg["redundancy_threshold"]) <= 1.0
