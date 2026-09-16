# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-13
# License: see the LICENSE file in the repository root
"""Tests for the progress log with ETA in
`src/experiments/exp_common.py::_consume_results`
(documentation/2026-09-13_hardening_behu.md, task 1): line frequency per the
config `parallel.progress_log_every_below` / `progress_log_every`, line
format, and the ETA computation from wall_time_sec of completed runs / n_workers."""
from __future__ import annotations

import logging

import pytest

from src.experiments import exp_common
from src.experiments.exp_common import _consume_results, format_progress_line, should_log_progress


def _result(i: int, wall: float, status: str = "ok") -> dict:
    return {
        "dataset_name": f"ds{i}", "method_name": "m", "seed": i, "status": status, "error": "",
        "wall_time_sec": wall, "embedding": None, "metrics": {}, "extra": {},
    }


def test_should_log_progress_small_n_logs_every_run() -> None:
    assert all(should_log_progress(i, 10, every_below=300, every=25) for i in range(1, 11))


def test_should_log_progress_large_n_first_last_and_every_kth() -> None:
    n = 1000
    logged = [i for i in range(1, n + 1) if should_log_progress(i, n, every_below=300, every=25)]
    assert logged[0] == 1 and logged[-1] == n
    assert 2 not in logged and 24 not in logged
    assert all(i % 25 == 0 for i in logged[1:-1])
    assert len(logged) == n // 25 + 1  # 40 multiples of 25 (incl. 1000) + the first


def test_format_progress_line_eta_uses_worker_wall_and_n_workers() -> None:
    # 4 of 10 done, wall sum 40 s -> mean 10 s/task, 6 tasks remain / 2 workers = 30 s
    line = format_progress_line(4, 10, _result(4, 12.5), elapsed_sec=65.0, wall_sum_sec=40.0, n_workers=2)
    assert line.startswith("[4/10] ds4 m seed=4 ok wall=12.5s")
    assert "elapsed=0:01:05" in line
    assert "mean/task=10.0s" in line
    assert "ETA=0:00:30" in line
    assert "n_workers=2" in line


def test_consume_results_logs_progress_lines(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Integration test of the loop: no disk write (append_result/save_embedding
    replaced), 5 runs (N <= every_below) -> 5 lines '[i/5]' incl. ETA, the
    ETA of the last line = 0."""
    written: list = []
    monkeypatch.setattr(exp_common, "append_result", lambda key, row: written.append((key, row)))
    monkeypatch.setattr(exp_common, "save_embedding", lambda key, emb: None)
    monkeypatch.setattr(exp_common, "_progress_log_policy", lambda: (300, 25))
    logger = logging.getLogger("test_progress_log")
    logger.propagate = True
    tasks = [{"dataset_name": f"ds{i}", "method_name": "m", "seed": i} for i in range(5)]
    results = [_result(i, wall=2.0 if i < 4 else 6.0, status="ok" if i != 2 else "error") for i in range(5)]
    with caplog.at_level(logging.INFO, logger="test_progress_log"):
        n_ok, n_err = _consume_results("test_progress", tasks, iter(results), ["status", "error", "wall_time_sec"], logger, n_workers=3)
    assert (n_ok, n_err) == (4, 1)
    assert len(written) == 5
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("[")]
    assert len(lines) == 5
    assert lines[0].startswith("[1/5] ds0 m seed=0 ok wall=2.0s")
    assert "ETA=0:00:00" in lines[-1]
    # after 3 runs: mean (2+2+2)/3 = 2 s, 2 remain, 3 workers -> ETA 1.33 s -> 0:00:01
    assert "mean/task=2.0s" in lines[2] and "ETA=0:00:01" in lines[2]


def test_progress_log_policy_reads_config() -> None:
    every_below, every = exp_common._progress_log_policy()
    assert every_below >= 0 and every >= 1
