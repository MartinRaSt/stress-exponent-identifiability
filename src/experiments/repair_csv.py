# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-13
# License: see the LICENSE file in the repository root
"""
K14 (cleanup/hardening): detection and repair of rows corrupted by an
interrupted write to results/data/<experiment>_results.csv - typically a
hard reset/process crash DURING `src.common.checkpoint.append_result` (a
real incident on 2026-09-13 18:27: a hard reset at 17:32 left the last row
of `exp7_rank_weights_results.csv` as 1818 zero bytes; pandas read it as a
row of all NaN, which broke `is_done`/`filter_already_done` on the next
resume).

Without `--repair` the script only CHECKS the CSV and prints the numbers of
corrupted rows (exit 1 if any are found, otherwise exit 0) - safe to run at
any time, changes nothing. With `--repair` it removes the corrupted rows
(the original file is ALWAYS backed up as <csv>.bak_<YYYYMMDD>_repair) and
returns exit 0.

This is NOT a silent fallback: if the CSV has no corrupted rows, `--repair`
reports that and changes nothing; if it does, it always backs up first and
only then overwrites. The corresponding CSV row was (by definition - NaN in
the key columns experiment/dataset/method/seed) incomplete/unusable, so the
next experiment run simply recomputes that combination (resumability is
unaffected).

Usage:
    venv\\python.exe -m src.experiments.repair_csv --csv results/data/exp7_rank_weights_results.csv
    venv\\python.exe -m src.experiments.repair_csv --csv results/data/exp7_rank_weights_results.csv --repair
or: src\\run_repair_csv.bat --csv ... [--repair]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.common.checkpoint import find_corrupted_row_lines, repair_csv_file


def check_csv(csv_path: Path) -> list[int]:
    """Loads the CSV and returns a list of corrupted row numbers (empty list = OK).
    A missing file is an error (fail-loud), not a silent no-op."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Results file does not exist: {csv_path}")
    df = pd.read_csv(csv_path)
    return find_corrupted_row_lines(df)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True, type=Path, help="Path to results/data/<experiment>_results.csv")
    parser.add_argument(
        "--repair", action="store_true",
        help="Actually remove the corrupted rows (with a .bak_<date>_repair backup). Without this flag, check only.",
    )
    args = parser.parse_args(argv)

    corrupted = check_csv(args.csv)
    if not corrupted:
        print(f"{args.csv}: no corrupted rows, nothing changed.")
        return 0

    lines_str = ", ".join(str(n) for n in corrupted)
    print(f"{args.csv}: corrupted rows (NaN in experiment/dataset/method/seed): {lines_str}")

    if not args.repair:
        print("To fix, run again with --repair (removes these rows with a .bak_<date>_repair backup).")
        return 1

    n_removed, backup = repair_csv_file(args.csv)
    print(f"Removed {n_removed} corrupted rows. Backup of the original file: {backup}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
