# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-09
# License: see the LICENSE file in the repository root
"""
Experiment resumability: every completed run (experiment, dataset, method, seed)
is written as one row to results/data/<experiment>_results.csv, with the
corresponding embedding in results/data/embeddings/<experiment>/<dataset>__<method>__<seed>.npy.
Before a run starts, `is_done` is checked, so on re-run the script resumes
where it left off and skips already-completed combinations.
"""
from __future__ import annotations

import csv
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common.config import ensure_dir, get_path

# Key columns that uniquely identify a result row (RunKey).
# Used to detect corrupted rows on resume (K14, see below).
_KEY_COLUMNS = ("experiment", "dataset", "method", "seed")


@dataclass(frozen=True)
class RunKey:
    """Unique identifier of a single experiment run."""

    experiment: str
    dataset: str
    method: str
    seed: int

    def as_tuple(self) -> tuple[str, str, str, int]:
        return (self.experiment, self.dataset, self.method, self.seed)


def results_csv_path(experiment: str) -> Path:
    """Return the path to the results CSV of the given experiment."""
    results_dir = get_path("results_data_dir")
    return results_dir / f"{experiment}_results.csv"


def embedding_path(key: RunKey) -> Path:
    """Return the path to the .npy file with the saved embedding for the given run."""
    emb_dir = get_path("embeddings_dir") / key.experiment
    return emb_dir / f"{key.dataset}__{key.method}__{key.seed}.npy"


def find_corrupted_row_lines(df: pd.DataFrame) -> list[int]:
    """Return a list of row numbers (1-based; header = row 1, first data
    row = row 2 - matching what the user sees in an editor/`head`) where a
    value (NaN) is missing in one of the key columns `_KEY_COLUMNS`.

    Typical consequence of an interrupted write (hard reset/process crash
    during `append_result`, see incident 2026-09-13 18:27: a hard reset left
    the last row of `exp7_rank_weights_results.csv` as 1818 zero bytes, which
    pandas read in as a row of all NaN)."""
    present_cols = [c for c in _KEY_COLUMNS if c in df.columns]
    if not present_cols:
        return []
    mask = df[present_cols].isna().any(axis=1)
    return [int(i) + 2 for i in df.index[mask]]


def _load_existing_keys(experiment: str) -> set[tuple[str, str, str, int]]:
    """Load the set of already-completed keys (dataset, method, seed) from the existing CSV."""
    csv_path = results_csv_path(experiment)
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path)
    required_cols = {"experiment", "dataset", "method", "seed"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"File {csv_path} is missing expected columns {missing}. "
            "Cannot safely determine completed runs - fix or delete the file manually."
        )
    corrupted = find_corrupted_row_lines(df)
    if corrupted:
        lines_str = ", ".join(str(n) for n in corrupted)
        raise ValueError(
            f"corrupted row {lines_str} in {csv_path} (likely an interrupted write), "
            "remove manually or run with --repair "
            f"(venv\\python.exe -m src.experiments.repair_csv --csv {csv_path} --repair)."
        )
    keys = set(
        zip(
            df["experiment"].astype(str),
            df["dataset"].astype(str),
            df["method"].astype(str),
            df["seed"].astype(int),
        )
    )
    return keys


def repair_csv_file(csv_path: Path) -> tuple[int, Path | None]:
    """Remove from `csv_path` rows corrupted by an interrupted write (NaN in
    the key columns `_KEY_COLUMNS`) - see `find_corrupted_row_lines`.

    The original file is ALWAYS backed up first as `<csv>.bak_<YYYYMMDD>_repair`
    (without overwriting an existing same-day backup - any collision is left
    for the caller/user to resolve manually; a backup is never silently
    overwritten).

    Returns (n_rows_removed, backup_path); (0, None) if the CSV contains no
    corrupted row (nothing changes, no backup)."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Results file does not exist: {csv_path}")
    df = pd.read_csv(csv_path)
    present_cols = [c for c in _KEY_COLUMNS if c in df.columns]
    if not present_cols:
        raise ValueError(f"{csv_path}: contains none of the key columns {_KEY_COLUMNS}, cannot repair.")
    mask = df[present_cols].isna().any(axis=1)
    n_bad = int(mask.sum())
    if n_bad == 0:
        return 0, None

    backup = csv_path.with_name(csv_path.name + ".bak_" + datetime.now().strftime("%Y%m%d") + "_repair")
    if backup.exists():
        raise FileExistsError(
            f"Backup {backup} already exists (--repair already ran today?) - remove it manually if you "
            "want to proceed, so nothing gets silently overwritten."
        )
    shutil.copy(csv_path, backup)
    df[~mask].to_csv(csv_path, index=False)
    return n_bad, backup


def is_done(key: RunKey) -> bool:
    """Return True if a row for the given key already exists in the results CSV."""
    existing = _load_existing_keys(key.experiment)
    return key.as_tuple() in existing


def append_result(key: RunKey, row: dict[str, Any]) -> None:
    """Append one result row to results/data/<experiment>_results.csv.

    Written immediately (append) after each completed run, so that an
    interrupted run does not lose already-computed data. The header is
    written only once.
    """
    csv_path = results_csv_path(key.experiment)
    ensure_dir(csv_path.parent)

    full_row = {
        "experiment": key.experiment,
        "dataset": key.dataset,
        "method": key.method,
        "seed": key.seed,
        **row,
    }

    file_exists = csv_path.exists()
    write_header = True
    if file_exists:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            existing_header = next(csv.reader(f), None)
        if existing_header is not None:
            write_header = False
            missing_cols = set(full_row.keys()) - set(existing_header)
            if missing_cols:
                raise ValueError(
                    f"New row has columns missing from the existing {csv_path}: "
                    f"{missing_cols}. Unify the experiment output schema "
                    "(e.g. delete the old CSV when metrics change)."
                )
            # fill in missing columns in the row (existing columns the row does not have)
            for col in existing_header:
                full_row.setdefault(col, "")
            fieldnames = existing_header
        else:
            fieldnames = list(full_row.keys())
    else:
        fieldnames = list(full_row.keys())

    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(full_row)
        # K14 (2026-09-13 18:27 incident - a hard reset during a write left
        # the last row of exp7_rank_weights_results.csv as 1818 zero bytes):
        # flush + fsync BEFORE closing the file, so every completed row is
        # physically on disk (not just in the OS buffer), and a machine
        # crash corrupts at most the row being written RIGHT NOW, not
        # previously confirmed rows.
        f.flush()
        os.fsync(f.fileno())


def save_embedding(key: RunKey, Y: np.ndarray) -> Path:
    """Save the embedding as .npy into results/data/embeddings/<experiment>/..."""
    path = embedding_path(key)
    ensure_dir(path.parent)
    np.save(path, Y.astype(np.float32))
    return path


def load_embedding(key: RunKey) -> np.ndarray:
    """Load a previously saved embedding for the given key."""
    path = embedding_path(key)
    if not path.exists():
        raise FileNotFoundError(
            f"Embedding for {key.as_tuple()} was not found at path {path}."
        )
    return np.load(path)


def count_done(experiment: str) -> int:
    """Return the number of already-completed runs of the given experiment (rows in the CSV)."""
    return len(_load_existing_keys(experiment))
