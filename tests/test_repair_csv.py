# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-13
# License: see the LICENSE file in the repository root
"""Tests for the `src/experiments/repair_csv.py` CLI on a synthetic CSV in
tmp_path (never on real results/data/*.csv - see CLAUDE.md)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.experiments import repair_csv as rc


def _make_corrupted_csv(tmp_path: Path) -> Path:
    csv_path = tmp_path / "exp7_rank_weights_results.csv"
    csv_path.write_text(
        "experiment,dataset,method,seed,status,error,wall_time_sec\n"
        "exp7_rank_weights,mnist,rank1,0,ok,,1.0\n"
        "exp7_rank_weights,mnist,rank1,1,ok,,1.1\n"
        ",,,,,,\n",  # row 4: an interrupted write (hard reset)
        encoding="utf-8",
    )
    return csv_path


def test_main_check_only_returns_1_and_leaves_csv_unchanged(tmp_path: Path) -> None:
    csv_path = _make_corrupted_csv(tmp_path)
    n_lines_before = len(csv_path.read_text(encoding="utf-8").splitlines())

    rc_exit = rc.main(["--csv", str(csv_path)])
    assert rc_exit == 1
    n_lines_after = len(csv_path.read_text(encoding="utf-8").splitlines())
    assert n_lines_after == n_lines_before  # no change without --repair


def test_main_repair_removes_bad_row_with_backup(tmp_path: Path) -> None:
    csv_path = _make_corrupted_csv(tmp_path)

    rc_exit = rc.main(["--csv", str(csv_path), "--repair"])
    assert rc_exit == 0

    df = pd.read_csv(csv_path)
    assert len(df) == 2
    assert rc.check_csv(csv_path) == []

    backups = list(tmp_path.glob("*.bak_*_repair"))
    assert len(backups) == 1


def test_main_clean_csv_returns_0(tmp_path: Path) -> None:
    csv_path = tmp_path / "exp_clean_results.csv"
    csv_path.write_text(
        "experiment,dataset,method,seed,status,error,wall_time_sec\n"
        "exp_clean,mnist,rank1,0,ok,,1.0\n",
        encoding="utf-8",
    )
    assert rc.main(["--csv", str(csv_path)]) == 0
    assert rc.main(["--csv", str(csv_path), "--repair"]) == 0
