# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-15
# License: see the LICENSE file in the repository root
"""K15 task B tests for `src/experiments/exp4_temporal.py`: the shared
function `neighbor_preservation_metrics` (identical logic as task A),
converting full names to short keys used in the CSV/npz cache, migrating
the schema of an existing CSV (backup + NaN for old rows), and backward
compatibility of the .npz cache without the new keys (old npz files from
BEFORE task B)."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from src.common import checkpoint
from src.common.checkpoint import RunKey
from src.experiments import exp4_temporal as et


def _square_D_and_Y() -> tuple[np.ndarray, np.ndarray]:
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    diff = pts[:, None, :] - pts[None, :, :]
    D = np.sqrt((diff ** 2).sum(-1))
    return D, pts


def test_neighbor_preservation_metrics_perfect_for_identity() -> None:
    D, Y = _square_D_and_Y()
    metrics, notes = et.neighbor_preservation_metrics(D, Y, [1])
    assert metrics["trustworthiness_k1"] == pytest.approx(1.0)
    assert metrics["continuity_k1"] == pytest.approx(1.0)
    assert metrics["knn_jaccard_k1"] == pytest.approx(1.0)
    assert notes == []


def test_neighbor_preservation_metrics_nan_note_for_too_small_snapshot() -> None:
    D, Y = _square_D_and_Y()
    metrics, notes = et.neighbor_preservation_metrics(D, Y, [10])
    assert np.isnan(metrics["trustworthiness_k10"])
    assert np.isnan(metrics["knn_jaccard_k10"])
    assert len(notes) == 2  # trust/cont and jaccard each report the invalid combination separately


@pytest.mark.parametrize(
    "full_key,expected",
    [("trustworthiness_k5", "trust_k5"), ("continuity_k10", "cont_k10"), ("knn_jaccard_k5", "jacc_k5")],
)
def test_short_neighbor_key_mapping(full_key: str, expected: str) -> None:
    assert et._short_neighbor_key(full_key) == expected


def test_short_neighbor_key_unknown_prefix_fails_loud() -> None:
    with pytest.raises(ValueError):
        et._short_neighbor_key("unexpected_metric_k5")


def test_neighbor_agg_columns_deterministic_order() -> None:
    assert et._neighbor_agg_columns([10, 5]) == [
        "trust_k5", "cont_k5", "jacc_k5", "trust_k10", "cont_k10", "jacc_k10",
    ]


def test_compute_neighbor_arrays_length_mismatch_fails_loud() -> None:
    D, Y = _square_D_and_Y()
    with pytest.raises(ValueError):
        et._compute_neighbor_arrays([D, D], [Y], [1])


def test_median_neighbor_metrics_all_nan_column_stays_nan() -> None:
    arrays = {"trust_k10": np.array([np.nan, np.nan])}
    out = et._median_neighbor_metrics(arrays)
    assert np.isnan(out["trust_k10"])


@pytest.fixture
def isolated_results_dir(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Redirects `checkpoint.get_path` (used by `results_csv_path` inside
    `_migrate_schema_if_needed`) and `exp4_temporal.get_path` (used by
    `_pertrans_cache_path`) to a temporary directory - same pattern as
    tests/test_checkpoint.py, so the test does not touch real results/data."""
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


def test_pertrans_cache_roundtrip_with_neighbor_arrays(isolated_results_dir) -> None:
    key = RunKey("exp4_temporal", "dsA", "lambda0.0_alpha1.0_smacof", 0)
    stab = np.array([0.1, 0.2])
    qual = np.array([0.3, 0.4])
    neighbor_arrays = {"trust_k5": np.array([0.9, 0.8]), "jacc_k5": np.array([0.5, 0.6])}
    et._save_pertrans_cache(key, stab, qual, neighbor_arrays)

    loaded = et._load_pertrans_cache(key)
    assert loaded is not None
    assert set(loaded.keys()) == {"stab", "qual", "trust_k5", "jacc_k5"}
    np.testing.assert_allclose(loaded["trust_k5"], neighbor_arrays["trust_k5"])
    np.testing.assert_allclose(loaded["stab"], stab)


def test_load_pertrans_cache_backward_compatible_without_neighbor_keys(isolated_results_dir) -> None:
    """An old npz (K15 before task B) contains only 'stab'/'qual' - loading
    must not crash on the missing key, these keys are just absent from the result."""
    key = RunKey("exp4_temporal", "dsA", "lambda0.0_alpha1.0_smacof", 0)
    et._save_pertrans_cache(key, np.array([0.1]), np.array([0.2]))  # no neighbor_arrays - old format

    loaded = et._load_pertrans_cache(key)
    assert loaded is not None
    assert set(loaded.keys()) == {"stab", "qual"}
    assert loaded.get("trust_k5") is None  # a missing key = the metric is not available, not a crash


def _old_schema_df() -> pd.DataFrame:
    return pd.DataFrame({
        "experiment": ["exp4_temporal"], "dataset": ["dsA"], "method": ["lambda0.0_alpha1.0_smacof"], "seed": [0],
        "stab": [1.0], "qual": [2.0], "n_snapshots_used": [5], "n_transitions": [4],
        "status": ["ok"], "error": [""], "wall_time_sec": [1.0],
    })


def test_migrate_schema_if_needed_backs_up_and_fills_nan(isolated_results_dir) -> None:
    csv_path = isolated_results_dir / "exp4_temporal_results.csv"
    isolated_results_dir.mkdir(parents=True, exist_ok=True)
    _old_schema_df().to_csv(csv_path, index=False)

    column_keys = ["stab", "qual", "n_snapshots_used", "n_transitions"] + et._neighbor_agg_columns([5, 10])
    et._migrate_schema_if_needed("exp4_temporal", column_keys, logging.getLogger("test_k15_migration"))

    backup = csv_path.with_name(csv_path.name + ".bak_20260915_schema")
    assert backup.exists()
    new_df = pd.read_csv(csv_path)
    for col in et._neighbor_agg_columns([5, 10]):
        assert col in new_df.columns
        assert new_df[col].isna().all()
    assert new_df.loc[0, "stab"] == pytest.approx(1.0)  # existing values unchanged


def test_migrate_schema_if_needed_noop_when_columns_present(isolated_results_dir) -> None:
    """A second call (the schema is already migrated) must not create
    another backup or change anything (idempotence on resume)."""
    csv_path = isolated_results_dir / "exp4_temporal_results.csv"
    isolated_results_dir.mkdir(parents=True, exist_ok=True)
    df = _old_schema_df()
    for col in et._neighbor_agg_columns([5, 10]):
        df[col] = np.nan
    df.to_csv(csv_path, index=False)

    column_keys = ["stab", "qual", "n_snapshots_used", "n_transitions"] + et._neighbor_agg_columns([5, 10])
    et._migrate_schema_if_needed("exp4_temporal", column_keys, logging.getLogger("test_k15_migration"))

    backup = csv_path.with_name(csv_path.name + ".bak_20260915_schema")
    assert not backup.exists()


def test_migrate_schema_if_needed_raises_if_backup_already_exists(isolated_results_dir) -> None:
    csv_path = isolated_results_dir / "exp4_temporal_results.csv"
    isolated_results_dir.mkdir(parents=True, exist_ok=True)
    _old_schema_df().to_csv(csv_path, index=False)
    backup = csv_path.with_name(csv_path.name + ".bak_20260915_schema")
    backup.write_text("dummy existing backup", encoding="utf-8")

    column_keys = ["stab", "qual", "n_snapshots_used", "n_transitions"] + et._neighbor_agg_columns([5, 10])
    with pytest.raises(FileExistsError):
        et._migrate_schema_if_needed("exp4_temporal", column_keys, logging.getLogger("test_k15_migration"))
