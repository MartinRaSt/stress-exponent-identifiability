# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-10
# License: see the LICENSE file in the repository root
"""Removes rows with status=error from a results CSV so that the next
(resumable) run recomputes them - typically after fixing a bug in a method's
code. The original file is backed up as <csv>.bak_<timestamp>.

Usage: venv\python.exe -m src.experiments.clean_error_rows results/data/exp1_dr_benchmark_results.csv [more.csv ...]
"""
from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


def clean_error_rows(csv_path: Path) -> int:
    """Deletes error rows from a single CSV; returns the number of deleted rows. A missing file is an error."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Results file does not exist: {csv_path}")
    df = pd.read_csv(csv_path)
    if "status" not in df.columns:
        raise ValueError(f"{csv_path}: missing column 'status'.")
    err = df[df["status"] == "error"]
    if len(err) == 0:
        print(f"{csv_path}: no error rows, nothing changed.")
        return 0
    backup = csv_path.with_name(csv_path.name + ".bak_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    shutil.copy(csv_path, backup)
    group_cols = [c for c in ("dataset", "method", "solver") if c in err.columns]
    print(f"{csv_path}: {len(df)} rows, {len(err)} error rows (backup {backup.name}):")
    print(err.groupby(group_cols).size().to_string())
    df[df["status"] != "error"].to_csv(csv_path, index=False)
    return int(len(err))


def main(argv: list[str]) -> int:
    """Entry point: cleans all CSV files given as arguments."""
    if not argv:
        print(__doc__)
        return 2
    total = sum(clean_error_rows(Path(p)) for p in argv)
    print(f"Total removed {total} error rows.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
