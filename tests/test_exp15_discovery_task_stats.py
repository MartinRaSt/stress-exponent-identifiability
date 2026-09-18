# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-18
# License: see the LICENSE file in the repository root
"""Tests for src/experiments/exp15_discovery_task_stats.py: the dataset-level
seed aggregation (majority tie-break), the NaN-safe Holm correction, and the
McNemar/Wilcoxon pairwise comparison on small synthetic contingency tables
with a known-sign outcome."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.experiments.exp15_discovery_task_stats import (
    PAIRWISE_COLUMNS,
    SUMMARY_COLUMNS,
    _aggregate_over_seeds,
    _holm_adjust,
    _wilcoxon_two_sided,
    build_pairwise,
    build_summary,
)
from src.sammon.discovery_task import QUESTION_NEAREST_PAIR


def _make_ok_rows(dataset: str, method: str, question: str, seed_correct: list[int], seed_soft: list[float]) -> pd.DataFrame:
    n = len(seed_correct)
    return pd.DataFrame({
        "dataset": [dataset] * n, "dr_method": [method] * n, "question": [question] * n,
        "seed": list(range(n)), "correct": seed_correct, "soft_rho": seed_soft, "status": ["ok"] * n,
    })


# ============================================================================
# _aggregate_over_seeds: majority tie-break
# ============================================================================

def test_aggregate_strict_majority_correct() -> None:
    df = _make_ok_rows("d1", "m1", QUESTION_NEAREST_PAIR, [1, 1, 1, 0, 0], [0.9] * 5)
    agg = _aggregate_over_seeds(df)
    assert agg.loc[0, "median_correct"] == 1.0
    assert agg.loc[0, "dataset_correct"] == 1.0


def test_aggregate_exact_tie_counts_as_wrong() -> None:
    """Even seed count with a 5-5 (well, e.g. 3-3) split -> median==0.5 must
    be scored as WRONG (0), never a fabricated coin flip."""
    df = _make_ok_rows("d1", "m1", QUESTION_NEAREST_PAIR, [1, 1, 1, 0, 0, 0], [0.9] * 6)
    agg = _aggregate_over_seeds(df)
    assert agg.loc[0, "median_correct"] == 0.5
    assert agg.loc[0, "dataset_correct"] == 0.0


def test_aggregate_undefined_question_stays_nan_not_wrong() -> None:
    """A dataset where the question is undefined (NaN correct, e.g.
    insufficient classes) must NOT be silently counted as 'wrong'."""
    df = _make_ok_rows("d1", "m1", QUESTION_NEAREST_PAIR, [np.nan, np.nan], [np.nan, np.nan])
    agg = _aggregate_over_seeds(df)
    assert np.isnan(agg.loc[0, "median_correct"])
    assert np.isnan(agg.loc[0, "dataset_correct"])


# ============================================================================
# build_summary
# ============================================================================

def test_build_summary_error_rate_and_schema() -> None:
    df = pd.concat([
        _make_ok_rows("d1", "m1", QUESTION_NEAREST_PAIR, [1, 1], [0.8, 0.9]),
        _make_ok_rows("d2", "m1", QUESTION_NEAREST_PAIR, [0, 0], [0.1, 0.2]),
    ])
    agg = _aggregate_over_seeds(df)
    summary = build_summary(agg)
    assert list(summary.columns) == SUMMARY_COLUMNS
    row = summary.iloc[0]
    assert row["n_datasets"] == 2
    assert row["error_rate"] == pytest.approx(0.5)  # 1 of 2 datasets wrong


def test_build_summary_excludes_undefined_datasets_from_denominator() -> None:
    df = pd.concat([
        _make_ok_rows("d1", "m1", QUESTION_NEAREST_PAIR, [1, 1], [0.8, 0.9]),
        _make_ok_rows("d2", "m1", QUESTION_NEAREST_PAIR, [np.nan, np.nan], [np.nan, np.nan]),
    ])
    agg = _aggregate_over_seeds(df)
    summary = build_summary(agg)
    assert summary.iloc[0]["n_datasets"] == 1  # d2 excluded, not counted as an error
    assert summary.iloc[0]["error_rate"] == pytest.approx(0.0)


# ============================================================================
# _holm_adjust (NaN-safe)
# ============================================================================

def test_holm_adjust_matches_hand_computation() -> None:
    # m=3, sorted p = [0.01, 0.02, 0.04] -> adjusted = [0.03, 0.04, 0.04] (running max)
    adjusted = _holm_adjust([0.02, 0.01, 0.04])
    assert adjusted[1] == pytest.approx(0.03)  # p=0.01 (smallest) * (3-0)
    assert adjusted[0] == pytest.approx(0.04)  # p=0.02 * (3-1)=0.04, running_max stays 0.04
    assert adjusted[2] == pytest.approx(0.04)  # p=0.04 * (3-2)=0.04


def test_holm_adjust_excludes_nan_from_family_size() -> None:
    """A NaN p-value must pass through as NaN and NOT shrink the effective
    family size for the OTHER (computable) tests."""
    adjusted = _holm_adjust([0.01, float("nan"), 0.02])
    assert np.isnan(adjusted[1])
    # family size for the 2 valid tests is 2, not 3
    assert adjusted[0] == pytest.approx(0.02)  # 0.01 * (2-0)
    assert adjusted[2] == pytest.approx(0.02)  # 0.02 * (2-1), running_max=0.02


def test_holm_adjust_all_nan_returns_all_nan() -> None:
    adjusted = _holm_adjust([float("nan"), float("nan")])
    assert all(np.isnan(v) for v in adjusted)


# ============================================================================
# _wilcoxon_two_sided
# ============================================================================

def test_wilcoxon_two_sided_all_zero_diff_is_nan() -> None:
    assert np.isnan(_wilcoxon_two_sided(np.zeros(5)))


def test_wilcoxon_two_sided_clear_shift_is_significant() -> None:
    rng = np.random.default_rng(0)
    diff = rng.normal(loc=5.0, scale=0.1, size=20)  # consistently positive, far from 0
    p = _wilcoxon_two_sided(diff)
    assert p < 0.01


# ============================================================================
# build_pairwise (McNemar + Wilcoxon on a small hand-built contingency table)
# ============================================================================

def _agg_two_methods(question: str, ref_correct: list[float], base_correct: list[float], ref_soft: list[float], base_soft: list[float]) -> pd.DataFrame:
    n = len(ref_correct)
    datasets = [f"d{i}" for i in range(n)]
    rows = []
    for ds, rc, bc, rs, bs in zip(datasets, ref_correct, base_correct, ref_soft, base_soft):
        rows.append({"dataset": ds, "dr_method": "ref", "question": question, "n_seeds": 1, "median_correct": rc, "median_soft_rho": rs, "dataset_correct": rc})
        rows.append({"dataset": ds, "dr_method": "base", "question": question, "n_seeds": 1, "median_correct": bc, "median_soft_rho": bs, "dataset_correct": bc})
    return pd.DataFrame(rows)


def test_build_pairwise_mcnemar_favors_reference_when_discordant_pairs_favor_it(caplog: pytest.LogCaptureFixture) -> None:
    import logging
    logger = logging.getLogger("test_exp15_stats")
    # 8 datasets: reference correct+baseline wrong in 6, reference wrong+baseline correct in 1, both correct in 1
    ref = [1, 1, 1, 1, 1, 1, 0, 1]
    base = [0, 0, 0, 0, 0, 0, 1, 1]
    soft_ref = [0.9] * 8
    soft_base = [0.5] * 8
    agg = _agg_two_methods(QUESTION_NEAREST_PAIR, ref, base, soft_ref, soft_base)
    pairwise = build_pairwise(agg, "ref", ["base"], (QUESTION_NEAREST_PAIR,), min_paired_datasets=3, mcnemar_exact_max_n=25, alpha=0.05, logger=logger)
    assert list(pairwise.columns) == PAIRWISE_COLUMNS
    row = pairwise.iloc[0]
    assert row["n_datasets_hard"] == 8
    assert row["n_discordant_a_only"] == 6
    assert row["n_discordant_b_only"] == 1
    # exact two-sided binomial for 6-vs-1 discordant pairs (n=7): 2*P(X<=1|n=7,p=0.5) = 0.125
    assert row["p_mcnemar"] == pytest.approx(0.125, abs=1e-6)


def test_build_pairwise_zero_discordant_pairs_is_nan_not_fabricated() -> None:
    import logging
    logger = logging.getLogger("test_exp15_stats")
    ref = [1, 1, 0, 0]
    base = [1, 1, 0, 0]  # perfectly concordant
    agg = _agg_two_methods(QUESTION_NEAREST_PAIR, ref, base, [0.9] * 4, [0.9] * 4)
    pairwise = build_pairwise(agg, "ref", ["base"], (QUESTION_NEAREST_PAIR,), min_paired_datasets=3, mcnemar_exact_max_n=25, alpha=0.05, logger=logger)
    row = pairwise.iloc[0]
    assert np.isnan(row["p_mcnemar"])


def test_build_pairwise_below_min_paired_datasets_skips_test(caplog: pytest.LogCaptureFixture) -> None:
    import logging
    logger = logging.getLogger("test_exp15_stats")
    ref = [1, 0]
    base = [0, 1]
    agg = _agg_two_methods(QUESTION_NEAREST_PAIR, ref, base, [0.9, 0.5], [0.5, 0.9])
    pairwise = build_pairwise(agg, "ref", ["base"], (QUESTION_NEAREST_PAIR,), min_paired_datasets=3, mcnemar_exact_max_n=25, alpha=0.05, logger=logger)
    row = pairwise.iloc[0]
    assert np.isnan(row["p_mcnemar"])
    assert np.isnan(row["p_wilcoxon_soft"])


def test_build_pairwise_missing_reference_method_yields_nan_rows_not_crash() -> None:
    import logging
    logger = logging.getLogger("test_exp15_stats")
    agg = _agg_two_methods(QUESTION_NEAREST_PAIR, [1, 0, 1], [0, 1, 0], [0.9, 0.1, 0.9], [0.1, 0.9, 0.1])
    agg = agg[agg["dr_method"] != "ref"]  # simulate reference_method absent from this subset
    pairwise = build_pairwise(agg, "ref", ["base"], (QUESTION_NEAREST_PAIR,), min_paired_datasets=1, mcnemar_exact_max_n=25, alpha=0.05, logger=logger)
    assert len(pairwise) == 1
    assert np.isnan(pairwise.iloc[0]["p_mcnemar"])
