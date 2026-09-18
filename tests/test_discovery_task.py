# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-18
# License: see the LICENSE file in the repository root
"""Tests for src/sammon/discovery_task.py (E15 core metric): the hard
argmin/argmax answers, the soft rank-correlation reuse from
src/sammon/cluster_geometry.py, tie detection, and the
insufficient/excessive-classes NaN-row contract."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

from src.sammon.cluster_geometry import (
    MAX_CLASSES_FOR_GEOMETRY,
    MIN_CLASSES_FOR_GEOMETRY,
    centroid_dist_spearman,
    class_spread_spearman,
)
from src.sammon.discovery_task import (
    DISCOVERY_QUESTIONS,
    DISCOVERY_TASK_KEYS,
    NOTE_EXCESSIVE_CLASSES,
    NOTE_INSUFFICIENT_CLASSES,
    NOTE_INSUFFICIENT_SPREAD_CANDIDATES,
    QUESTION_MOST_DISPERSED,
    QUESTION_NEAREST_PAIR,
    _pick_extremum,
    discovery_task_rows,
)


def _three_blob_dataset(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """3 Gaussian blobs at (0,0), (5,0), (0,20) with class 0 (near the
    origin) noticeably more spread out than 1/2, and centers placed so that
    0/1 are unambiguously the closest pair (distance 5) while 0/2 (20) and
    1/2 (~20.6) are both far - a dataset with an unambiguous ground truth
    for BOTH E15 questions."""
    rng = np.random.default_rng(seed)
    n_per = 30
    c0 = rng.normal(loc=(0.0, 0.0), scale=1.5, size=(n_per, 2))
    c1 = rng.normal(loc=(5.0, 0.0), scale=0.2, size=(n_per, 2))
    c2 = rng.normal(loc=(0.0, 20.0), scale=0.2, size=(n_per, 2))
    X = np.vstack([c0, c1, c2])
    y = np.array([0] * n_per + [1] * n_per + [2] * n_per)
    return X, y


def _dmat(X: np.ndarray) -> np.ndarray:
    return squareform(pdist(X, metric="euclidean"))


# ============================================================================
# _pick_extremum
# ============================================================================

def test_pick_extremum_min_picks_lowest_value() -> None:
    idx, tied = _pick_extremum(np.array([3.0, 1.0, 2.0]), 1e-9, "min")
    assert idx == 1
    assert tied is False


def test_pick_extremum_max_picks_highest_value() -> None:
    idx, tied = _pick_extremum(np.array([3.0, 1.0, 2.0]), 1e-9, "max")
    assert idx == 0
    assert tied is False


def test_pick_extremum_detects_exact_tie_deterministically() -> None:
    # two equal minimal values -> lowest INDEX wins, tie=True
    idx, tied = _pick_extremum(np.array([1.0, 1.0, 2.0]), 1e-9, "min")
    assert idx == 0
    assert tied is True


def test_pick_extremum_no_tie_outside_tolerance() -> None:
    idx, tied = _pick_extremum(np.array([1.0, 1.01, 2.0]), 1e-9, "min")
    assert idx == 0
    assert tied is False


def test_pick_extremum_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        _pick_extremum(np.array([]), 1e-9, "min")


# ============================================================================
# discovery_task_rows - well-separated ground truth
# ============================================================================

def test_discovery_task_rows_recovers_truth_when_map_equals_input() -> None:
    """D_out == D_in (a perfect map) must answer both questions correctly
    with soft_rho == 1.0 (Spearman self-correlation)."""
    X, y = _three_blob_dataset()
    D = _dmat(X)
    rows = discovery_task_rows(D, D, y, tie_relative_tolerance=1e-9)
    assert [r["question"] for r in rows] == list(DISCOVERY_QUESTIONS)
    for r in rows:
        assert r["correct"] == 1
        assert r["soft_rho"] == pytest.approx(1.0, abs=1e-9)
        assert r["note"] == ""
        assert r["n_classes"] == 3


def test_discovery_task_rows_nearest_pair_matches_expected_classes() -> None:
    """Classes 0 and 1 are the closest pair by construction (see
    _three_blob_dataset); class 2 is the outlier for BOTH questions is not
    guaranteed, but the nearest pair IS unambiguous here."""
    X, y = _three_blob_dataset()
    D = _dmat(X)
    rows = discovery_task_rows(D, D, y, tie_relative_tolerance=1e-9)
    q1 = next(r for r in rows if r["question"] == QUESTION_NEAREST_PAIR)
    assert q1["true_answer"] == "0|1"
    assert q1["n_candidates"] == 3  # C(3,2)


def test_discovery_task_rows_soft_scores_match_cluster_geometry_directly() -> None:
    """The soft score of Q1/Q2 must be EXACTLY the corresponding K2 metric
    (centroid_dist_spearman / class_spread_spearman) - no separate
    reimplementation, see the module docstring."""
    X, y = _three_blob_dataset()
    D_in = _dmat(X)
    rng = np.random.default_rng(1)
    Y = X + rng.normal(scale=0.5, size=X.shape)
    D_out = _dmat(Y)
    rows = discovery_task_rows(D_in, D_out, y, tie_relative_tolerance=1e-9)
    q1 = next(r for r in rows if r["question"] == QUESTION_NEAREST_PAIR)
    q2 = next(r for r in rows if r["question"] == QUESTION_MOST_DISPERSED)
    assert q1["soft_rho"] == pytest.approx(centroid_dist_spearman(D_in, D_out, y))
    assert q2["soft_rho"] == pytest.approx(class_spread_spearman(D_in, D_out, y))


def test_discovery_task_rows_detects_wrong_prediction() -> None:
    """A map that swaps which pair is nearest must be scored incorrect (0),
    not silently coerced to 'correct'."""
    X, y = _three_blob_dataset()
    D_in = _dmat(X)
    # construct D_out where class 0 and 2 become the (wrongly) nearest pair -
    # drag class 2's cloud from (0, 20) right on top of class 0's cloud at (0, 0)
    X_out = X.copy()
    X_out[y == 2] += np.array([0.0, -20.0])
    D_out = _dmat(X_out)
    rows = discovery_task_rows(D_in, D_out, y, tie_relative_tolerance=1e-9)
    q1 = next(r for r in rows if r["question"] == QUESTION_NEAREST_PAIR)
    assert q1["true_answer"] == "0|1"
    assert q1["pred_answer"] != q1["true_answer"]
    assert q1["correct"] == 0


# ============================================================================
# Edge cases: too few / too many classes, no labels, sparse classes
# ============================================================================

def test_discovery_task_rows_insufficient_classes_returns_nan_rows() -> None:
    X = np.random.default_rng(0).normal(size=(20, 2))
    y = np.array([0] * 10 + [1] * 10)  # only 2 classes < MIN_CLASSES_FOR_GEOMETRY
    D = _dmat(X)
    rows = discovery_task_rows(D, D, y, tie_relative_tolerance=1e-9)
    assert len(rows) == 2
    for r in rows:
        assert r["note"] == NOTE_INSUFFICIENT_CLASSES
        assert np.isnan(r["correct"])
        assert np.isnan(r["soft_rho"])
        assert r["n_classes"] == 2


def test_discovery_task_rows_no_labels_returns_nan_rows() -> None:
    X = np.random.default_rng(0).normal(size=(20, 2))
    D = _dmat(X)
    rows = discovery_task_rows(D, D, None, tie_relative_tolerance=1e-9)
    assert len(rows) == 2
    for r in rows:
        assert r["note"] == NOTE_INSUFFICIENT_CLASSES
        assert r["n_classes"] == 0


def test_discovery_task_rows_excessive_classes_returns_nan_rows() -> None:
    n = MAX_CLASSES_FOR_GEOMETRY + 5
    X = np.random.default_rng(0).normal(size=(n * 2, 2))
    y = np.repeat(np.arange(n), 2)  # each "class" has exactly 2 points, n classes
    D = _dmat(X)
    rows = discovery_task_rows(D, D, y, tie_relative_tolerance=1e-9)
    for r in rows:
        assert r["note"] == NOTE_EXCESSIVE_CLASSES
        assert np.isnan(r["correct"])


def test_discovery_task_rows_q2_undefined_when_fewer_than_2_spreadable_classes() -> None:
    """Q2 needs >=2 classes with >=2 points to have a comparison at all -
    here only 1 of 3 classes has >=2 points."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 2))
    y = np.array([0] * 10 + [1] + [2])  # class 1 and 2 are singletons
    D = _dmat(X)
    rows = discovery_task_rows(D, D, y, tie_relative_tolerance=1e-9)
    q1 = next(r for r in rows if r["question"] == QUESTION_NEAREST_PAIR)
    q2 = next(r for r in rows if r["question"] == QUESTION_MOST_DISPERSED)
    assert q1["note"] == ""  # Q1 does not need per-class point counts
    assert np.isfinite(q1["correct"])
    assert q2["note"] == NOTE_INSUFFICIENT_SPREAD_CANDIDATES
    assert np.isnan(q2["correct"])


def test_discovery_task_row_keys_are_stable() -> None:
    """Every row must expose exactly 'question' + DISCOVERY_TASK_KEYS - the
    schema the experiment script's COLUMN_KEYS is built from."""
    X, y = _three_blob_dataset()
    D = _dmat(X)
    rows = discovery_task_rows(D, D, y, tie_relative_tolerance=1e-9)
    expected = {"question"} | set(DISCOVERY_TASK_KEYS)
    for r in rows:
        assert set(r.keys()) == expected
