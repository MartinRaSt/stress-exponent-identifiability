# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-11
# License: see the LICENSE file in the repository root
"""
Selective removal of rows from an experiment results CSV (for a rerun after
fixing a method/dataset) - unlike `clean_error_rows.py` (which only targets
rows with status='error'), this allows deleting specific dataset/method
combinations regardless of status, so that the next (resumable) experiment
run recomputes them.

The original CSV is always backed up as <csv>.bak_<timestamp>. Along with
the rows, the corresponding embedding files
`results/data/embeddings/<experiment>/<dataset>__<method>__<seed>.npy` and
(for E4 - temporal) the pertrans cache
`results/data/embeddings/<experiment>/pertrans/<dataset>__<method>__<seed>.npz`
are also removed (see src/experiments/exp4_temporal.py::_pertrans_cache_path).

The --dataset/--method filters are ANDed (each optional, but at least one
must be given - without a filter the script refuses to delete anything).
Each filter accepts a comma-separated list of names (ORed within a filter).

Usage:
    venv\\python.exe -m src.experiments.remove_rows --csv results/data/exp1_dr_benchmark_results.csv --dataset wine --dry-run
    venv\\python.exe -m src.experiments.remove_rows --csv results/data/exp1_dr_benchmark_results.csv --dataset wine,iris --method pca,mds
or: src\\run_remove_rows.bat --csv ... --dataset ... [--method ...] [--dry-run]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.common.config import get_path


def _experiment_name_from_csv(csv_path: Path) -> str:
    """Derives the experiment name (incl. any 'smoke/'/'quick/' prefix) from
    the CSV path - mirrors `src.common.checkpoint.results_csv_path`
    (<results_data_dir>/<experiment>_results.csv), so it also works for
    results/data/smoke/<exp>_results.csv or results/data/quick/<exp>_results.csv."""
    results_data_dir = get_path("results_data_dir").resolve()
    csv_resolved = csv_path.resolve()
    suffix = "_results.csv"
    if not csv_resolved.name.endswith(suffix):
        raise ValueError(f"Expected a file named '<experiment>{suffix}', got: {csv_path.name}")
    try:
        rel = csv_resolved.relative_to(results_data_dir)
    except ValueError as exc:
        raise ValueError(
            f"CSV {csv_path} is not under results_data_dir ({results_data_dir}) - cannot safely derive "
            "the experiment name (and hence the embedding path)."
        ) from exc
    rel_str = rel.as_posix()
    return rel_str[: -len(suffix)]


def _artifact_paths_for_row(experiment: str, dataset: str, method: str, seed: int) -> list[Path]:
    """Returns all files on disk corresponding to a single row (embedding +
    the E4 pertrans cache, if any) - existence is checked only when deleting."""
    emb_dir = get_path("embeddings_dir") / experiment
    base = f"{dataset}__{method}__{seed}"
    return [emb_dir / f"{base}.npy", emb_dir / "pertrans" / f"{base}.npz"]


def remove_rows(csv_path: Path, datasets: list[str] | None, methods: list[str] | None, dry_run: bool) -> int:
    """Removes from `csv_path` the rows matching the filters (AND between
    dataset/method, OR within each list) and deletes the corresponding
    embedding/pertrans cache files. Returns the number of removed rows (0 if
    nothing matches). A missing CSV or missing expected columns are an
    error (fail-loud), not a silent no-op."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Results file does not exist: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"experiment", "dataset", "method", "seed"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing expected columns {sorted(missing)}.")

    mask = pd.Series(True, index=df.index)
    if datasets is not None:
        mask &= df["dataset"].astype(str).isin(datasets)
    if methods is not None:
        mask &= df["method"].astype(str).isin(methods)

    n_match = int(mask.sum())
    if n_match == 0:
        print(f"{csv_path}: no row matches the filter (dataset={datasets}, method={methods}), nothing changed.")
        return 0

    experiment = _experiment_name_from_csv(csv_path)
    matched = df[mask]
    group_cols = [c for c in ("dataset", "method") if c in matched.columns]
    print(f"{csv_path}: found {n_match}/{len(df)} rows to remove (experiment='{experiment}'):")
    print(matched.groupby(group_cols).size().to_string())

    if dry_run:
        print("--dry-run: no change made (neither the CSV nor the embedding files were deleted).")
        return n_match

    backup = csv_path.with_name(csv_path.name + ".bak_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    shutil.copy(csv_path, backup)
    print(f"Backup of the original CSV: {backup}")

    df[~mask].to_csv(csv_path, index=False)

    n_files_removed = 0
    for _, row in matched.iterrows():
        for p in _artifact_paths_for_row(experiment, str(row["dataset"]), str(row["method"]), int(row["seed"])):
            if p.exists():
                p.unlink()
                n_files_removed += 1
    print(f"Removed rows from CSV: {n_match}")
    print(f"Removed embedding/pertrans cache files: {n_files_removed}")
    return n_match


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True, type=Path, help="Path to results/data/<experiment>_results.csv")
    parser.add_argument("--dataset", default=None, help="Comma-separated dataset name(s), e.g. 'wine,iris'")
    parser.add_argument("--method", default=None, help="Comma-separated method name(s), e.g. 'pca,mds'")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be deleted, without writing to disk")
    args = parser.parse_args(argv)

    if args.dataset is None and args.method is None:
        parser.error(
            "Provide at least one filter (--dataset and/or --method) - without a filter I refuse to delete all rows."
        )

    datasets = [s.strip() for s in args.dataset.split(",") if s.strip()] if args.dataset else None
    methods = [s.strip() for s in args.method.split(",") if s.strip()] if args.method else None

    remove_rows(args.csv, datasets, methods, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
