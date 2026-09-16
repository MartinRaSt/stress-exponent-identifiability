# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-13
# License: see the LICENSE file in the repository root
"""K14 (cleanup/hardening): tests for resume robustness of
`src/common/checkpoint.py` - fail-loud detection of rows corrupted by an
interrupted write (NaN in the key columns experiment/dataset/method/seed,
see incident 2026-09-13 18:27: a hard reset left the last row in
exp7_rank_weights_results.csv as 1818 zero bytes) and the fix
(`repair_csv_file`/`src/experiments/repair_csv.py`)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.common import checkpoint
from src.common.checkpoint import (
    RunKey,
    append_result,
    find_corrupted_row_lines,
    is_done,
    repair_csv_file,
)


@pytest.fixture
def isolated_results_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirects `checkpoint.get_path('results_data_dir'/'embeddings_dir')`
    to a temporary directory so the test does not touch the real results/data."""
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


def test_append_result_writes_readable_row(isolated_results_dir: Path) -> None:
    """append_result (with flush+fsync) writes a row that can be read back immediately."""
    key = RunKey("exp_test", "dsA", "methodA", 0)
    append_result(key, {"status": "ok", "error": "", "wall_time_sec": 1.23, "metric_x": 0.5})
    assert is_done(key)
    df = pd.read_csv(isolated_results_dir / "exp_test_results.csv")
    assert len(df) == 1
    assert df.loc[0, "dataset"] == "dsA"


def test_find_corrupted_row_lines_detects_nan_in_key_columns() -> None:
    """A row with NaN in any key column (experiment/dataset/method/seed) is
    detected; row 1 is the header, the first data row is row 2."""
    df = pd.DataFrame(
        {
            "experiment": ["exp7_rank_weights", "exp7_rank_weights", None],
            "dataset": ["mnist", "mnist", "mnist"],
            "method": ["rank1", None, "rank2"],
            "seed": [0, 1, 2],
            "status": ["ok", "ok", "ok"],
        }
    )
    corrupted = find_corrupted_row_lines(df)
    # row 0 (index0, ok) -> line number 2; index1 (missing method) -> line 3;
    # index2 (missing experiment) -> line 4
    assert corrupted == [3, 4]


def test_find_corrupted_row_lines_empty_for_clean_csv() -> None:
    df = pd.DataFrame({"experiment": ["e"], "dataset": ["d"], "method": ["m"], "seed": [0]})
    assert find_corrupted_row_lines(df) == []


def test_is_done_fails_loud_on_corrupted_csv(isolated_results_dir: Path) -> None:
    """Simulation of the incident from 2026-09-13 18:27: the last row of the
    CSV is corrupted (interrupted write) - is_done/_load_existing_keys must
    raise a clear exception instead of silently continuing with bad data."""
    csv_path = isolated_results_dir / "exp_crash_results.csv"
    isolated_results_dir.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(
        "experiment,dataset,method,seed,status,error,wall_time_sec\n"
        "exp_crash,mnist,rank1,0,ok,,1.0\n"
        "exp_crash,mnist,rank2,1,ok,,1.1\n"
        ",,,,,,\n",  # line 4: written incompletely (all keys NaN)
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as exc_info:
        is_done(RunKey("exp_crash", "mnist", "rank1", 0))
    msg = str(exc_info.value)
    assert "corrupted row 4" in msg
    assert "--repair" in msg
    assert "exp_crash_results.csv" in msg


def test_repair_csv_file_removes_bad_rows_with_backup(tmp_path: Path) -> None:
    """repair_csv_file removes corrupted rows, backs up the original file as
    .bak_<date>_repair, and the CSV then passes is_done/find_corrupted without error."""
    csv_path = tmp_path / "exp_crash_results.csv"
    csv_path.write_text(
        "experiment,dataset,method,seed,status,error,wall_time_sec\n"
        "exp_crash,mnist,rank1,0,ok,,1.0\n"
        ",,,,,,\n",
        encoding="utf-8",
    )
    n_removed, backup = repair_csv_file(csv_path)
    assert n_removed == 1
    assert backup is not None and backup.exists()
    assert backup.name.endswith("_repair")

    df_after = pd.read_csv(csv_path)
    assert len(df_after) == 1
    assert find_corrupted_row_lines(df_after) == []

    # a repeated call on an already-clean CSV changes nothing (0, None)
    n_removed2, backup2 = repair_csv_file(csv_path)
    assert n_removed2 == 0
    assert backup2 is None


def test_repair_csv_file_missing_file_is_fail_loud(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        repair_csv_file(tmp_path / "does_not_exist_results.csv")
