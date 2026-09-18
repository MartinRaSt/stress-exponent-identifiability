# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-18
# License: see the LICENSE file in the repository root
"""Tests for src/experiments/exp15_discovery_task.py: the row-granular
resume key (question folded into the RunKey method field, plain method name
carried separately in 'dr_method'), the CSV schema, and the per-task
computation (`_compute_rows_for_task`) wired against a mocked
`load_embedding` - no real dataset/exp1_dr_benchmark run needed (same
"exercise the pure/testable pieces + one fully-mocked call" style as
test_exp13_neighbor_survival.py)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist, squareform

from src.common import checkpoint
from src.experiments import exp15_discovery_task as e15
from src.sammon.discovery_task import DISCOVERY_QUESTIONS, DISCOVERY_TASK_KEYS


@pytest.fixture
def isolated_results_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirects `checkpoint.get_path` to a tmp dir - see test_checkpoint.py."""
    data_dir = tmp_path / "results_data"
    emb_dir = tmp_path / "results_data" / "embeddings"

    def _fake_get_path(key: str) -> Path:
        if key == "results_data_dir":
            return data_dir
        if key == "embeddings_dir":
            return emb_dir
        raise KeyError(key)

    monkeypatch.setattr(checkpoint, "get_path", _fake_get_path)
    return data_dir


def _dmat(X: np.ndarray) -> np.ndarray:
    return squareform(pdist(X, metric="euclidean"))


def _labeled_dataset(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_per = 15
    c0 = rng.normal(loc=(0.0, 0.0), scale=1.0, size=(n_per, 2))
    c1 = rng.normal(loc=(5.0, 0.0), scale=0.2, size=(n_per, 2))
    c2 = rng.normal(loc=(0.0, 20.0), scale=0.2, size=(n_per, 2))
    X = np.vstack([c0, c1, c2])
    y = np.array([0] * n_per + [1] * n_per + [2] * n_per)
    return X, y


# ============================================================================
# COLUMN_KEYS / composite RunKey
# ============================================================================

def test_column_keys_matches_discovery_task_schema() -> None:
    assert set(e15.COLUMN_KEYS) == {"dr_method", "question"} | set(DISCOVERY_TASK_KEYS)


def test_composite_key_folds_question_into_method() -> None:
    key = e15._composite_key("exp15_discovery_task", "iris", "pca", 3, "nearest_class_pair")
    assert key.experiment == "exp15_discovery_task"
    assert key.dataset == "iris"
    assert key.method == "pca__nearest_class_pair"
    assert key.seed == 3


# ============================================================================
# row-granular resume
# ============================================================================

def test_pending_questions_all_pending_when_csv_absent(isolated_results_dir: Path) -> None:
    pending = e15._pending_questions("exp_test", "iris", "pca", 0)
    assert pending == list(DISCOVERY_QUESTIONS)


def test_pending_questions_shrinks_after_one_question_written(isolated_results_dir: Path) -> None:
    key = e15._composite_key("exp_test", "iris", "pca", 0, DISCOVERY_QUESTIONS[0])
    checkpoint.append_result(key, {"dr_method": "pca", "question": DISCOVERY_QUESTIONS[0], "status": "ok", "error": "", "wall_time_sec": 0.1})
    pending = e15._pending_questions("exp_test", "iris", "pca", 0)
    assert pending == [DISCOVERY_QUESTIONS[1]]


def test_pending_questions_empty_after_both_written(isolated_results_dir: Path) -> None:
    for q in DISCOVERY_QUESTIONS:
        key = e15._composite_key("exp_test", "iris", "pca", 0, q)
        checkpoint.append_result(key, {"dr_method": "pca", "question": q, "status": "ok", "error": "", "wall_time_sec": 0.1})
    assert e15._pending_questions("exp_test", "iris", "pca", 0) == []


def test_composite_key_resume_does_not_confuse_methods(isolated_results_dir: Path) -> None:
    """Two DIFFERENT dr_methods sharing (dataset, seed) must be tracked independently."""
    key_a = e15._composite_key("exp_test", "iris", "pca", 0, DISCOVERY_QUESTIONS[0])
    checkpoint.append_result(key_a, {"dr_method": "pca", "question": DISCOVERY_QUESTIONS[0], "status": "ok", "error": "", "wall_time_sec": 0.1})
    assert e15._pending_questions("exp_test", "iris", "tsne_auto", 0) == list(DISCOVERY_QUESTIONS)


# ============================================================================
# _compute_rows_for_task (mocked load_embedding, no real dataset/config)
# ============================================================================

def test_compute_rows_for_task_ok_path(monkeypatch: pytest.MonkeyPatch) -> None:
    X, y = _labeled_dataset()
    D_in = _dmat(X)
    rng = np.random.default_rng(1)
    Y = X + rng.normal(scale=0.1, size=X.shape)

    monkeypatch.setattr(e15, "load_embedding", lambda key: Y)
    rows = e15._compute_rows_for_task(
        "exp1_dr_benchmark", "synthetic_ds", "pca", 0, list(DISCOVERY_QUESTIONS), D_in, y, None, 1e-9,
    )
    assert set(rows.keys()) == set(DISCOVERY_QUESTIONS)
    for q, row in rows.items():
        assert row["status"] == "ok"
        assert row["error"] == ""
        assert row["correct"] in (0, 1)


def test_compute_rows_for_task_load_error_produces_error_rows_for_all_pending() -> None:
    rows = e15._compute_rows_for_task(
        "exp1_dr_benchmark", "synthetic_ds", "pca", 0, list(DISCOVERY_QUESTIONS), None, None, "FileNotFoundError: boom", 1e-9,
    )
    assert set(rows.keys()) == set(DISCOVERY_QUESTIONS)
    for row in rows.values():
        assert row["status"] == "error"
        assert "boom" in row["error"]


def test_compute_rows_for_task_shape_mismatch_is_error_not_silent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A saved embedding with the WRONG number of points (stale cache /
    subsample-seed drift) must fail loud, never silently truncate/pad."""
    X, y = _labeled_dataset()
    D_in = _dmat(X)
    wrong_shape_Y = X[:-5]  # too few rows

    monkeypatch.setattr(e15, "load_embedding", lambda key: wrong_shape_Y)
    rows = e15._compute_rows_for_task(
        "exp1_dr_benchmark", "synthetic_ds", "pca", 0, list(DISCOVERY_QUESTIONS), D_in, y, None, 1e-9,
    )
    for row in rows.values():
        assert row["status"] == "error"
        assert "does not match" in row["error"]


def test_compute_rows_for_task_only_computes_pending_questions(monkeypatch: pytest.MonkeyPatch) -> None:
    X, y = _labeled_dataset()
    D_in = _dmat(X)
    monkeypatch.setattr(e15, "load_embedding", lambda key: X)
    only_one = [DISCOVERY_QUESTIONS[0]]
    rows = e15._compute_rows_for_task("exp1_dr_benchmark", "synthetic_ds", "pca", 0, only_one, D_in, y, None, 1e-9)
    assert set(rows.keys()) == set(only_one)


def test_compute_rows_for_task_partial_failure_marks_all_pending_as_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """If loading the (single, shared) embedding fails, BOTH pending
    questions must be marked as error - there is no such thing as computing
    Q1 successfully while Q2 fails, since both are derived from the same D_out."""
    def _raise(key):
        raise FileNotFoundError("no embedding on disk")

    monkeypatch.setattr(e15, "load_embedding", _raise)
    rows = e15._compute_rows_for_task(
        "exp1_dr_benchmark", "synthetic_ds", "pca", 0, list(DISCOVERY_QUESTIONS), np.eye(3), np.array([0, 1, 2]), None, 1e-9,
    )
    assert all(r["status"] == "error" for r in rows.values())
    assert all("no embedding on disk" in r["error"] for r in rows.values())
