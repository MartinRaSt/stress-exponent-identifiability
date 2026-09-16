# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-15
# License: see the LICENSE file in the repository root
"""K16 (2026-09-15) tests for two follow-up fixes in
`src/experiments/exp4_temporal.py`:

1) `_paired_permutation_test` - the fixed Monte-Carlo p-value
   `(1 + count) / (1 + n_permutations)` (the original `count / n_permutations`
   could return exactly 0.0, a statistically incorrect claim) - aligned with
   `stats_holdout.py::mc_sign_flip_pvalue`/`exp9_metric_fidelity_stats.py::
   sign_flip_test`. `exp4_neighbor_metrics.py` imports the same function, so
   the fix propagates there automatically (without a separate import in this test).

2) Protection against result loss: `exp4_stats.csv` (full overwrite) is only
   written at 100% coverage of the .npz pertrans cache for all 'ok' runs in
   the results CSV (`_compute_stats_coverage`/`_finalize_stats_csv`); at
   partial coverage the output goes to `exp4_stats_partial.csv` and the
   existing full `exp4_stats.csv` remains UNTOUCHED."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from src.common import checkpoint
from src.common.checkpoint import RunKey
from src.experiments import exp4_temporal as et


# ---------------------------------------------------------------------------
# 1) _paired_permutation_test - the MC p-value is never exactly 0
# ---------------------------------------------------------------------------

def test_paired_permutation_test_p_always_in_open_zero_closed_one() -> None:
    rng = np.random.default_rng(0)
    diff = rng.normal(loc=0.0, scale=1.0, size=30)
    for n_perm in (10, 100, 1000):
        _observed, p = et._paired_permutation_test(diff, n_perm, seed=1)
        assert 0.0 < p <= 1.0


def test_paired_permutation_test_extreme_effect_never_zero() -> None:
    """An extreme, unambiguous effect (all differences strongly positive)
    could previously (before K16) return exactly p=0.0 - now it must ALWAYS be > 0."""
    diff = np.full(50, 10.0)
    for n_perm in (10, 1000, 20_000):
        _observed, p = et._paired_permutation_test(diff, n_perm, seed=0)
        assert p > 0.0


def test_paired_permutation_test_p_decreases_with_more_permutations_for_strong_effect() -> None:
    """For a clearly nonzero (extreme, one-sided) effect, the smallest
    attainable MC p-value is `1/(1+n_permutations)` - it must decrease in
    the right direction as the number of permutations grows (a more precise
    estimate of a small p-value)."""
    diff = np.full(50, 10.0)
    ps = [et._paired_permutation_test(diff, n_perm, seed=0)[1] for n_perm in (10, 1_000, 100_000)]
    assert ps[0] > ps[1] > ps[2]
    # the analytic lower bound for this extreme case (count almost surely 0)
    assert ps[0] == pytest.approx(1.0 / 11.0)
    assert ps[2] == pytest.approx(1.0 / 100_001.0)


def test_paired_permutation_test_matches_mc_formula_terminology() -> None:
    """Same formula as `stats_holdout.py::mc_sign_flip_pvalue`/
    `exp9_metric_fidelity_stats.py::sign_flip_test` for the MC branch -
    manually computed count/p for a small deterministic example."""
    diff = np.array([1.0, 1.0, 1.0, -1.0])
    n_perm = 200
    seed = 7
    observed, p = et._paired_permutation_test(diff, n_perm, seed)
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(diff)))
    null_dist = (signs * diff[None, :]).mean(axis=1)
    expected_count = int(np.sum(np.abs(null_dist) >= abs(observed)))
    assert p == pytest.approx((1 + expected_count) / (1 + n_perm))


# ---------------------------------------------------------------------------
# 2) npz cache coverage before overwriting exp4_stats.csv
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated_results_dir(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Same pattern as tests/test_checkpoint.py and
    tests/test_exp4_temporal_neighbor_metrics.py - redirects
    `checkpoint.get_path` (`results_csv_path`) and `exp4_temporal.get_path`
    (`_pertrans_cache_path`) to a temporary directory."""
    data_dir = tmp_path / "results_data"
    emb_dir = data_dir / "embeddings"

    def _fake_get_path(key: str):
        if key == "results_data_dir":
            return data_dir
        if key == "embeddings_dir":
            return emb_dir
        raise KeyError(key)

    monkeypatch.setattr(checkpoint, "get_path", _fake_get_path)
    monkeypatch.setattr(et, "get_path", _fake_get_path)
    return data_dir


def _write_results_csv(data_dir, rows: list[dict]) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(data_dir / "exp4_temporal_results.csv", index=False)


def _ok_row(method: str, seed: int) -> dict:
    return {
        "experiment": "exp4_temporal", "dataset": "dsA", "method": method, "seed": seed,
        "stab": 0.1, "qual": 0.2, "status": "ok", "error": "", "wall_time_sec": 1.0,
    }


def test_compute_stats_coverage_full(isolated_results_dir) -> None:
    rows = [_ok_row("lambda0.0_alpha1.0_smacof", 0), _ok_row("lambda1.0_alpha1.0_smacof", 0)]
    _write_results_csv(isolated_results_dir, rows)
    for row in rows:
        key = RunKey("exp4_temporal", row["dataset"], row["method"], row["seed"])
        et._save_pertrans_cache(key, np.array([0.1]), np.array([0.2]))

    n_with_cache, n_ok = et._compute_stats_coverage("exp4_temporal")
    assert (n_with_cache, n_ok) == (2, 2)


def test_compute_stats_coverage_partial(isolated_results_dir) -> None:
    rows = [_ok_row("lambda0.0_alpha1.0_smacof", 0), _ok_row("lambda1.0_alpha1.0_smacof", 0)]
    _write_results_csv(isolated_results_dir, rows)
    # only the FIRST row has a cache (the second simulates a lost/missing .npz cache)
    key = RunKey("exp4_temporal", rows[0]["dataset"], rows[0]["method"], rows[0]["seed"])
    et._save_pertrans_cache(key, np.array([0.1]), np.array([0.2]))

    n_with_cache, n_ok = et._compute_stats_coverage("exp4_temporal")
    assert (n_with_cache, n_ok) == (1, 2)


def test_finalize_stats_csv_full_coverage_writes_exp4_stats(isolated_results_dir) -> None:
    rows = [_ok_row("lambda0.0_alpha1.0_smacof", 0)]
    _write_results_csv(isolated_results_dir, rows)
    key = RunKey("exp4_temporal", rows[0]["dataset"], rows[0]["method"], rows[0]["seed"])
    et._save_pertrans_cache(key, np.array([0.1]), np.array([0.2]))

    logger = logging.getLogger("test_k16_finalize_full")
    out_path = et._finalize_stats_csv("exp4_temporal", [{"dataset": "dsA", "metric": "stab", "p_value": 0.5}], logger)

    assert out_path.name == "exp4_stats.csv"
    assert out_path.exists()
    assert not (isolated_results_dir / "exp4_stats_partial.csv").exists()


def test_finalize_stats_csv_partial_coverage_does_not_overwrite_existing_full(isolated_results_dir) -> None:
    """The key protection test (K16 item 2): with incomplete coverage, only
    exp4_stats_partial.csv is written and the already existing (previously
    correctly computed) exp4_stats.csv stays UNCHANGED (both content and mtime)."""
    rows = [_ok_row("lambda0.0_alpha1.0_smacof", 0), _ok_row("lambda1.0_alpha1.0_smacof", 0)]
    _write_results_csv(isolated_results_dir, rows)
    # only one row has a cache -> 50% coverage
    key0 = RunKey("exp4_temporal", rows[0]["dataset"], rows[0]["method"], rows[0]["seed"])
    et._save_pertrans_cache(key0, np.array([0.1]), np.array([0.2]))

    existing_full = isolated_results_dir / "exp4_stats.csv"
    existing_full.write_text("dataset,metric,p_value\ndsA,stab,0.123\n", encoding="utf-8")
    before_mtime = existing_full.stat().st_mtime_ns
    before_content = existing_full.read_text(encoding="utf-8")

    logger = logging.getLogger("test_k16_finalize_partial")
    out_path = et._finalize_stats_csv(
        "exp4_temporal", [{"dataset": "dsA", "metric": "stab", "p_value": 0.9}], logger,
    )

    assert out_path.name == "exp4_stats_partial.csv"
    assert out_path.exists()
    assert existing_full.stat().st_mtime_ns == before_mtime
    assert existing_full.read_text(encoding="utf-8") == before_content


def test_finalize_stats_csv_zero_coverage_writes_partial_not_full(isolated_results_dir) -> None:
    """n_ok_total > 0 but NO row has a cache (0%) - must also go to
    exp4_stats_partial.csv, not to exp4_stats.csv."""
    rows = [_ok_row("lambda0.0_alpha1.0_smacof", 0)]
    _write_results_csv(isolated_results_dir, rows)  # no cache saved

    logger = logging.getLogger("test_k16_finalize_zero")
    out_path = et._finalize_stats_csv("exp4_temporal", [{"dataset": "dsA", "metric": "stab", "p_value": 0.5}], logger)

    assert out_path.name == "exp4_stats_partial.csv"
