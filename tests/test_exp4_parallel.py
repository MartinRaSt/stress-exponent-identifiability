# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-16
# License: see the LICENSE file in the repository root
"""Tests for E4 parallelization (documentation/2026-09-16_e4_paralelizace.md):

1. the result of a single run must be PICKLABLE (workers return it via
   ProcessPoolExecutor) and must not contain the fitted object,
2. `_preload_snapshots` loads each dataset exactly once (fail-fast before
   the workers start),
3. `parallel.orphan_poll_sec` is read from the config, and a missing/invalid
   value is an error (fail loud), not a silent fallback.
"""
from __future__ import annotations

import logging
import pickle

import numpy as np
import pytest

from src.common import parallel
from src.experiments import exp4_temporal


def test_run_single_result_is_picklable_and_has_no_model_object() -> None:
    """The run result is returned from the worker via pickle: it may only
    contain arrays, not the TemporalSammon/DynamicTSNE object (which also
    holds the input matrices)."""
    task = {
        "family": "sammon", "mode": "smoke", "dataset_name": "__nonexistent_dataset__",
        "lam": 0.0, "alpha": 1.0, "solver": "smacof", "seed": 0,
        "n_components": 2, "max_iter": 5, "tol": 1e-4,
        "time_bin_sec": 3600, "min_snapshot_nodes": 10, "trajectory_seed": 0,
        "sgd_cfg": {"epochs": 1, "mu_max": 1.0, "pairs_per_node": 1, "eps_anneal": 0.1},
        "neighbor_k": [5],
    }
    # a deliberately nonexistent dataset: we care about the SHAPE of the
    # result (the error branch), not the computation - the fit itself is
    # verified by other tests and the smoke run
    result = exp4_temporal._run_single(task)

    assert result["status"] == "error"
    assert "_ts" not in result, "the fitted object must not be returned from the worker"
    assert result["_Y_list"] is None
    pickle.loads(pickle.dumps(result))


def test_preload_snapshots_loads_each_dataset_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Snapshots are loaded in the main process once per (dataset, bin,
    min_nodes), not once per (lambda, alpha, solver, seed) combination."""
    calls: list[tuple] = []

    def fake_load(dataset_name, time_bin_sec, min_snapshot_nodes, mode=None):
        calls.append((dataset_name, time_bin_sec, min_snapshot_nodes))
        return [np.zeros((2, 2))], [[0, 1]]

    monkeypatch.setattr(exp4_temporal, "_load_snapshots", fake_load)
    tasks = [
        {"dataset_name": "a", "time_bin_sec": 60, "min_snapshot_nodes": 10, "mode": "smoke"},
        {"dataset_name": "a", "time_bin_sec": 60, "min_snapshot_nodes": 10, "mode": "smoke"},
        {"dataset_name": "b", "time_bin_sec": 60, "min_snapshot_nodes": 10, "mode": "smoke"},
    ]
    exp4_temporal._preload_snapshots(tasks, logging.getLogger("test"))

    assert calls == [("a", 60, 10), ("b", 60, 10)]


def test_resolve_orphan_poll_sec_reads_config() -> None:
    assert parallel.resolve_orphan_poll_sec() > 0


def test_resolve_orphan_poll_sec_fails_loud_on_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(parallel, "load_config", lambda: {"parallel": {}})
    with pytest.raises(KeyError):
        parallel.resolve_orphan_poll_sec()


def test_resolve_orphan_poll_sec_rejects_nonpositive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(parallel, "load_config", lambda: {"parallel": {"orphan_poll_sec": 0}})
    with pytest.raises(ValueError):
        parallel.resolve_orphan_poll_sec()
