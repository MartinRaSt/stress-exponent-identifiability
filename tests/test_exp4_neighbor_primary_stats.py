# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-16
# License: see the LICENSE file in the repository root
"""K16 (2026-09-16, the author's spec - a hierarchical test for
neighborhood preservation, E4 vs. dynamic t-SNE) tests:

1) `exp4_common.nearest_stab_index` - the shared pairing rule (nearest
   'stab'), including ties.
2) `exp4_neighbor_metrics._build_primary_pairs` - pairing PER DATASET
   according to the same rule, correct sign of the difference (ours -
   dtsne), fail-loud when dtsne/our method is missing in a given dataset.
3) `exp4_neighbor_metrics._primary_test_for_metric` - the exact sign-flip
   test against a hand-computed value (diffs=[1,2,-0.5] -> p=0.5, HL=0.875,
   see test_exp1_holdout_confirmatory.py for an independent verification of
   stats_holdout.py).
4) `_write_primary_stats` end-to-end: the Holm correction ONLY over
   `primary_metrics` (not over anything else), test_role='primary'.
5) `_write_stats` (a descriptive test): column 'test_role' = the
   exploratory constant, NO 'p_value_holm' column (K16 - the correction
   over 672 tests was removed, it was toothless)."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from src.common import checkpoint
from src.experiments import exp4_common
from src.experiments import exp4_neighbor_metrics as enm
from src.experiments.config_experiments import resolve_experiment_config
from src.experiments.stats_holdout import holm_correction


# ---------------------------------------------------------------------------
# 1) exp4_common.nearest_stab_index
# ---------------------------------------------------------------------------

def test_nearest_stab_index_picks_closest() -> None:
    candidates = np.array([0.1, 0.5, 0.9])
    assert exp4_common.nearest_stab_index(0.52, candidates) == 1
    assert exp4_common.nearest_stab_index(0.05, candidates) == 0
    assert exp4_common.nearest_stab_index(10.0, candidates) == 2


def test_nearest_stab_index_tie_break_is_first_occurrence() -> None:
    # 0.5 is EXACTLY equally far (binary-representable values, no rounding
    # ambiguity) from both 0.0 (index 0) and 1.0 (index 1) -> returns the first.
    candidates = np.array([0.0, 1.0])
    assert exp4_common.nearest_stab_index(0.5, candidates) == 0


def test_nearest_stab_index_raises_on_empty() -> None:
    with pytest.raises(ValueError):
        exp4_common.nearest_stab_index(0.5, np.array([]))


# ---------------------------------------------------------------------------
# 2) _build_primary_pairs
# ---------------------------------------------------------------------------

def _med_row(dataset: str, method: str, stab: float, trust_k10: float, jacc_k10: float) -> dict:
    return {"dataset": dataset, "method": method, "stab": stab, "trust_k10": trust_k10, "jacc_k10": jacc_k10}


def test_build_primary_pairs_matches_nearest_stab_and_sign() -> None:
    """dsA: dtsne (stab=0.50) is closer to lambda0.0_alpha1.0_smacof
    (stab=0.48) than to lambda5.0_alpha1.0_sgd (stab=0.90) -> it must be
    paired with the former; diff = ours - dtsne."""
    med = pd.DataFrame([
        _med_row("dsA", "dtsne_lambda0.0", 0.50, trust_k10=0.60, jacc_k10=0.40),
        _med_row("dsA", "lambda0.0_alpha1.0_smacof", 0.48, trust_k10=0.55, jacc_k10=0.30),
        _med_row("dsA", "lambda5.0_alpha1.0_sgd", 0.90, trust_k10=0.99, jacc_k10=0.99),
    ])
    pairs = enm._build_primary_pairs(med, ["trust_k10", "jacc_k10"])
    assert len(pairs) == 2  # 1 dataset x 1 dtsne_lambda x 2 metrics
    row_trust = pairs[pairs["metric"] == "trust_k10"].iloc[0]
    assert row_trust["matched_lambda"] == pytest.approx(0.0)
    assert row_trust["matched_solver"] == "smacof"
    assert row_trust["diff_ours_minus_dtsne"] == pytest.approx(0.55 - 0.60)
    row_jacc = pairs[pairs["metric"] == "jacc_k10"].iloc[0]
    assert row_jacc["diff_ours_minus_dtsne"] == pytest.approx(0.30 - 0.40)
    assert row_trust["stab_diff_abs"] == pytest.approx(abs(0.48 - 0.50))


def test_build_primary_pairs_raises_without_dtsne_method() -> None:
    med = pd.DataFrame([_med_row("dsA", "lambda0.0_alpha1.0_smacof", 0.48, 0.5, 0.5)])
    with pytest.raises(ValueError, match="dtsne_lambda"):
        enm._build_primary_pairs(med, ["trust_k10"])


def test_build_primary_pairs_raises_without_our_method() -> None:
    med = pd.DataFrame([_med_row("dsA", "dtsne_lambda0.0", 0.48, 0.5, 0.5)])
    with pytest.raises(ValueError, match="our family"):
        enm._build_primary_pairs(med, ["trust_k10"])


# ---------------------------------------------------------------------------
# 3) _primary_test_for_metric - proti rucne spocitane hodnote
# ---------------------------------------------------------------------------

def test_primary_test_for_metric_hand_verified() -> None:
    """diffs=[1,2,-0.5] over 3 "datasets": exact two-sided p=0.5 (4 of 8 sign
    combinations have |sum|>=2.5), HL=0.875 (median of the Walsh averages) -
    independently hand-verified (see the module docstring)."""
    diffs = pd.Series([1.0, 2.0, -0.5], index=["dsA", "dsB", "dsC"])
    result = enm._primary_test_for_metric(diffs, "two-sided", n_boot=500, boot_seed=0)
    assert result["n_datasets"] == 3
    assert result["n_datasets_dropped_nan"] == 0
    assert result["p_value"] == pytest.approx(0.5)
    assert result["n_permutations_exact"] == 8
    assert result["hl_estimate"] == pytest.approx(0.875)
    assert result["median_diff"] == pytest.approx(1.0)
    assert result["n_wins_ours"] == 2
    assert result["n_wins_dtsne"] == 1


def test_primary_test_for_metric_drops_nan_datasets() -> None:
    diffs = pd.Series([1.0, np.nan, -0.5], index=["dsA", "dsB", "dsC"])
    result = enm._primary_test_for_metric(diffs, "two-sided", n_boot=200, boot_seed=0)
    assert result["n_datasets"] == 2
    assert result["n_datasets_dropped_nan"] == 1


# ---------------------------------------------------------------------------
# 4)+5) end-to-end pres isolated results dir (_write_primary_stats, _write_stats)
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated_results_dir(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """The same pattern as tests/test_exp4_temporal_stats_k16.py - redirects
    `checkpoint.get_path` (and hence `results_csv_path`) to a temporary directory."""
    data_dir = tmp_path / "results_data"

    def _fake_get_path(key: str):
        if key == "results_data_dir":
            return data_dir
        if key == "embeddings_dir":
            return data_dir / "embeddings"
        raise KeyError(key)

    monkeypatch.setattr(checkpoint, "get_path", _fake_get_path)
    return data_dir


def _write_temporal_csv(data_dir, rows: list[dict]) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df["experiment"] = "exp4_temporal"
    df["seed"] = 0
    df["status"] = "ok"
    df["error"] = ""
    df["wall_time_sec"] = 1.0
    df["qual"] = 0.1
    df["n_snapshots_used"] = 10
    df["n_transitions"] = 9
    df.to_csv(data_dir / "exp4_temporal_results.csv", index=False)


def _dataset_rows(dataset: str, dtsne_stab: float, dtsne_trust10: float, dtsne_jacc10: float, dtsne_trust5: float, dtsne_jacc5: float,
                   our_stab: float, our_trust10: float, our_jacc10: float, our_trust5: float, our_jacc5: float) -> list[dict]:
    return [
        {"dataset": dataset, "method": "dtsne_lambda0.0", "stab": dtsne_stab,
         "trust_k10": dtsne_trust10, "jacc_k10": dtsne_jacc10, "trust_k5": dtsne_trust5, "jacc_k5": dtsne_jacc5},
        {"dataset": dataset, "method": "lambda0.0_alpha1.0_smacof", "stab": our_stab,
         "trust_k10": our_trust10, "jacc_k10": our_jacc10, "trust_k5": our_trust5, "jacc_k5": our_jacc5},
        # another point from our family, deliberately FAR from the dtsne stab,
        # so that pairing unambiguously picks the first (smacof) method above
        {"dataset": dataset, "method": "lambda5.0_alpha1.0_sgd", "stab": our_stab + 5.0,
         "trust_k10": 0.01, "jacc_k10": 0.01, "trust_k5": 0.01, "jacc_k5": 0.01},
    ]


def test_write_primary_stats_holm_only_over_primary_metrics(isolated_results_dir) -> None:
    rows: list[dict] = []
    # 3 datasets, ours SYSTEMATICALLY a bit WORSE (lower) than dtsne on all
    # four metrics - a consistent direction so the exact test gives a clear (small) p.
    for i, ds in enumerate(["dsA", "dsB", "dsC"]):
        rows += _dataset_rows(
            ds, dtsne_stab=0.5, dtsne_trust10=0.80, dtsne_jacc10=0.60, dtsne_trust5=0.85, dtsne_jacc5=0.65,
            our_stab=0.5 + 0.001 * i, our_trust10=0.74, our_jacc10=0.52, our_trust5=0.79, our_jacc5=0.58,
        )
    _write_temporal_csv(isolated_results_dir, rows)

    cfg_primary = resolve_experiment_config("exp4_neighbor_metrics", "full")
    logger = logging.getLogger("test_k16_primary")
    out_path = enm._write_primary_stats(cfg_primary, "full", logger)

    assert out_path is not None
    assert out_path.name == "exp4_neighbor_primary_stats.csv"
    stats_df = pd.read_csv(out_path)
    assert len(stats_df) == 4  # 4 primary_metrics
    assert set(stats_df["metric"]) == {"trust_k10", "jacc_k10", "trust_k5", "jacc_k5"}
    assert (stats_df["test_role"] == "primary").all()
    assert (stats_df["n_datasets"] == 3).all()
    # all 4 diffs are consistently negative (ours < dtsne) -> median_diff < 0
    assert (stats_df["median_diff"] < 0).all()

    # Holm must be computed ONLY over these 4 p-values, not over anything more.
    manual_holm = holm_correction(stats_df["p_value"].tolist())
    np.testing.assert_allclose(sorted(stats_df["p_value_holm"]), sorted(manual_holm), rtol=1e-12)

    pairs_path = out_path.parent / "exp4_neighbor_primary_pairs.csv"
    assert pairs_path.exists()
    pairs_df = pd.read_csv(pairs_path)
    # 3 datasets x 1 dtsne lambda x 4 metrics
    assert len(pairs_df) == 3 * 1 * 4
    assert (pairs_df["matched_solver"] == "smacof").all()


def test_write_primary_stats_missing_metric_column_fails_loud(isolated_results_dir) -> None:
    rows = [
        {"dataset": "dsA", "method": "dtsne_lambda0.0", "stab": 0.5},
        {"dataset": "dsA", "method": "lambda0.0_alpha1.0_smacof", "stab": 0.5},
    ]
    df = pd.DataFrame(rows)
    df["experiment"] = "exp4_temporal"
    df["seed"] = 0
    df["status"] = "ok"
    df["error"] = ""
    df["wall_time_sec"] = 1.0
    df["qual"] = 0.1
    isolated_results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(isolated_results_dir / "exp4_temporal_results.csv", index=False)

    cfg_primary = resolve_experiment_config("exp4_neighbor_metrics", "full")
    logger = logging.getLogger("test_k16_primary_missing_col")
    with pytest.raises(ValueError, match="does not contain the expected"):
        enm._write_primary_stats(cfg_primary, "full", logger)


def _write_neighbor_metrics_csv(data_dir, rows: list[dict]) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df["experiment"] = "exp4_neighbor_metrics"
    df["seed"] = 0
    df["status"] = "ok"
    df["error"] = ""
    df["wall_time_sec"] = 0.1
    df["n_nodes"] = 20
    df["note"] = ""
    df.to_csv(data_dir / "exp4_neighbor_metrics_results.csv", index=False)


def test_write_stats_marks_exploratory_without_holm_column(isolated_results_dir) -> None:
    rows = []
    for t in range(4):
        rows.append({
            "dataset": "dsA", "method": "dtsne_lambda0.0", "lambda": 0.0, "alpha": np.nan, "solver": "dtsne", "t": t,
            "trustworthiness_k5": 0.80, "continuity_k5": 0.80, "knn_jaccard_k5": 0.60,
            "trustworthiness_k10": 0.80, "continuity_k10": 0.80, "knn_jaccard_k10": 0.60,
        })
        rows.append({
            "dataset": "dsA", "method": "lambda0.0_alpha1.0_smacof", "lambda": 0.0, "alpha": 1.0, "solver": "smacof", "t": t,
            "trustworthiness_k5": 0.70, "continuity_k5": 0.70, "knn_jaccard_k5": 0.50,
            "trustworthiness_k10": 0.70, "continuity_k10": 0.70, "knn_jaccard_k10": 0.50,
        })
    _write_neighbor_metrics_csv(isolated_results_dir, rows)

    cfg = resolve_experiment_config("exp4_temporal", "smoke")
    cfg_primary = resolve_experiment_config("exp4_neighbor_metrics", "smoke")
    logger = logging.getLogger("test_k16_write_stats")
    out_path = enm._write_stats(cfg, cfg_primary, "exp4_neighbor_metrics", [5, 10], logger)

    assert out_path is not None
    stats_df = pd.read_csv(out_path)
    assert "test_role" in stats_df.columns
    assert (stats_df["test_role"] == enm._EXPLORATORY_TEST_ROLE).all()
    assert "p_value_holm" not in stats_df.columns
