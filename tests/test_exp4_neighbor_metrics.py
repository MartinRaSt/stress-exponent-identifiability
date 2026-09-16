# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-15
# License: see the LICENSE file in the repository root
"""K15 task A tests for `src/experiments/exp4_neighbor_metrics.py`: pairing
of trajectory nodes <-> D_t (fail-loud on a mismatch), NaN + a note for
snapshots that are too small (K > the usable limit), deterministic column
names, the Holm-Bonferroni correction, and an integration test for
smoke/quick output isolation."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from src.common.config import get_mode_path, get_path
from src.experiments import exp4_neighbor_metrics as enm

ROOT = get_path("results_data_dir").parent.parent


def _square_D() -> np.ndarray:
    """4 nodes on a unit square - a distance matrix exactly realizable in 2D
    (Y = the same coordinates => perfectly preserved neighborhoods)."""
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    diff = pts[:, None, :] - pts[None, :, :]
    return np.sqrt((diff ** 2).sum(-1)), pts


def _base_task(node_ids_t, D_t, y_by_node, neighbor_k) -> dict:
    return {
        "dataset_name": "dsA", "method_name": "lambda0.0_alpha1.0_smacof__t0", "seed": 0, "t": 0,
        "lam": 0.0, "alpha": 1.0, "solver": "smacof",
        "D_t": D_t, "node_ids_t": node_ids_t, "y_by_node": y_by_node, "neighbor_k": neighbor_k,
    }


def test_run_single_matched_ids_ok_and_perfect_metrics() -> None:
    """Exactly the same coordinates as the original square -> trustworthiness
    = continuity = knn_jaccard = 1 for K=1 (the node <-> D_t pairing is
    verified implicitly by the result coming out exactly 1, not by chance)."""
    D_t, pts = _square_D()
    node_ids_t = [10, 20, 30, 40]
    y_by_node = {str(nid): (float(pts[i, 0]), float(pts[i, 1])) for i, nid in enumerate(node_ids_t)}
    task = _base_task(node_ids_t, D_t, y_by_node, neighbor_k=[1])
    result = enm._run_single(task)
    assert result["status"] == "ok"
    assert result["error"] == ""
    assert result["metrics"]["trustworthiness_k1"] == pytest.approx(1.0)
    assert result["metrics"]["continuity_k1"] == pytest.approx(1.0)
    assert result["metrics"]["knn_jaccard_k1"] == pytest.approx(1.0)
    assert result["extra"]["n_nodes"] == 4
    assert result["extra"]["note"] == ""


def test_run_single_raises_on_missing_node_fail_loud() -> None:
    """A node present in D_t but missing from the saved trajectory must lead
    to an UNCAUGHT exception (fail-loud, K15 spec) - not a NaN row."""
    D_t, pts = _square_D()
    node_ids_t = [10, 20, 30, 40]
    y_by_node = {str(nid): (float(pts[i, 0]), float(pts[i, 1])) for i, nid in enumerate(node_ids_t[:3])}  # node 40 missing
    task = _base_task(node_ids_t, D_t, y_by_node, neighbor_k=[1])
    with pytest.raises(ValueError, match="Node pairing failed"):
        enm._run_single(task)


def test_run_single_raises_on_extra_node_fail_loud() -> None:
    """An extra node in the trajectory (not in D_t) is just as fail-loud as
    a missing node - never a silent skip/truncation."""
    D_t, pts = _square_D()
    node_ids_t = [10, 20, 30, 40]
    y_by_node = {str(nid): (float(pts[i, 0]), float(pts[i, 1])) for i, nid in enumerate(node_ids_t)}
    y_by_node["999"] = (0.5, 0.5)  # extra node
    task = _base_task(node_ids_t, D_t, y_by_node, neighbor_k=[1])
    with pytest.raises(ValueError, match="Node pairing failed"):
        enm._run_single(task)


def test_run_single_nan_and_note_for_too_small_snapshot() -> None:
    """K=10 on a snapshot with 4 nodes is an invalid combination (2n-3K-1<=0
    for trustworthiness/continuity, K>n-1 for Jaccard) - the result MUST be
    NaN + a note in 'note', status stays 'ok' (this is a legitimate edge
    case, not an error)."""
    D_t, pts = _square_D()
    node_ids_t = [10, 20, 30, 40]
    y_by_node = {str(nid): (float(pts[i, 0]), float(pts[i, 1])) for i, nid in enumerate(node_ids_t)}
    task = _base_task(node_ids_t, D_t, y_by_node, neighbor_k=[10])
    result = enm._run_single(task)
    assert result["status"] == "ok"
    assert np.isnan(result["metrics"]["trustworthiness_k10"])
    assert np.isnan(result["metrics"]["continuity_k10"])
    assert np.isnan(result["metrics"]["knn_jaccard_k10"])
    assert "k=10" in result["extra"]["note"]
    assert result["extra"]["n_nodes"] == 4


def test_metric_columns_deterministic_order() -> None:
    cols = enm._metric_columns([10, 5])
    assert cols == [
        "trustworthiness_k5", "continuity_k5", "knn_jaccard_k5",
        "trustworthiness_k10", "continuity_k10", "knn_jaccard_k10",
    ]


def test_holm_correction_matches_manual_step_down() -> None:
    """A manually computed example of the Holm-Bonferroni correction (3 tests)."""
    p = np.array([0.01, 0.04, 0.03])
    # sorted: 0.01(m=3->0.03), 0.03(m=2->0.06), 0.04(m=1->0.04); step-down
    # monotonicity: [0.03, 0.06, 0.06]
    adj = enm._holm_correction(p)
    order = np.argsort(p)
    assert adj[order] == pytest.approx([0.03, 0.06, 0.06])


def test_holm_correction_never_exceeds_one() -> None:
    p = np.array([0.9, 0.95, 0.99])
    adj = enm._holm_correction(p)
    assert np.all(adj <= 1.0)


def _snapshot(paths: list[Path]) -> dict[str, int]:
    return {str(p): p.stat().st_mtime_ns for p in paths if p.exists()}


def test_smoke_run_does_not_touch_full_outputs() -> None:
    """Integration test (K15): a --smoke run must write EXCLUSIVELY to
    results/data/smoke/ - the full exp4_temporal_results.csv/exp4_trajectories.csv/
    exp4_stats.csv remain unchanged (mtime)."""
    smoke_traj = get_mode_path("results_data_dir", "smoke") / "exp4_trajectories.csv"
    if not smoke_traj.exists():
        pytest.skip(f"missing smoke trajectory {smoke_traj} (run src\\run_exp4_temporal.bat smoke)")

    full_paths = [
        get_path("results_data_dir") / "exp4_temporal_results.csv",
        get_path("results_data_dir") / "exp4_trajectories.csv",
        get_path("results_data_dir") / "exp4_stats.csv",
        get_path("results_data_dir") / "exp4_neighbor_metrics_results.csv",
        get_path("results_data_dir") / "exp4_neighbor_stats.csv",
        get_path("results_data_dir") / "exp4_neighbor_primary_stats.csv",
        get_path("results_data_dir") / "exp4_neighbor_primary_pairs.csv",
    ]
    before = _snapshot(full_paths)

    old_argv = sys.argv
    sys.argv = ["exp4_neighbor_metrics", "--smoke"]
    try:
        enm.main()
    finally:
        sys.argv = old_argv

    after = _snapshot(full_paths)
    assert before == after, f"the smoke run changed full outputs: {set(before) ^ set(after) | {k for k in before if before[k] != after.get(k)}}"
