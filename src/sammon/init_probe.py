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
A probe of the `init_classical_mds` init time on a real E3 distance matrix
(documentation/2026-09-13_hardening_behu.md): loads a cached graph distance
matrix (default pgp resistance, n=10680, see
`src.sammon.gpu_memory_probe.load_cached_distance`) and measures the full
eigh (original path, `iterative_min_n` above n) vs. iterative eigsh (new
path, `iterative_min_n=0`) for a given number of BLAS threads (the E3
worker has 1 thread, see config parallel.blas_threads_per_worker). For
each run, writes a row to `results/data/quick/init_probe.csv` (time, top-p
eigenvalues from Y0, max absolute difference of Y0 vs. the eigh reference
after sign alignment).

Run via: `src\\run_init_probe.bat [--threads 1 8] [--methods eigh eigsh]`
or directly `venv\\python.exe -m src.sammon.init_probe ...`. Default
values from `sammon.init_probe` in config.yaml. Fail-loud: a missing cache
is an error (no synthetic data).
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from src.common.config import ensure_dir, get_path, load_config
from src.common.logging_utils import get_logger
from src.sammon.gpu_memory_probe import load_cached_distance
from src.sammon.init import init_classical_mds

CSV_COLUMNS = [
    "timestamp", "graph", "distance", "n", "n_components", "seed", "method", "blas_threads",
    "init_sec", "eig_top", "max_abs_diff_vs_eigh_signed",
]
METHODS = ("eigh", "eigsh")
# `iterative_min_n` threshold forcing the full eigh (above any realistic n)
_FORCE_DENSE_MIN_N = 10 ** 9


def _csv_path() -> Path:
    return get_path("results_data_dir") / "quick" / "init_probe.csv"


def _append_row(row: dict) -> None:
    path = _csv_path()
    ensure_dir(path.parent)
    write_header = not path.exists()
    if not write_header:
        with open(path, "r", encoding="utf-8", newline="") as f:
            header = next(csv.reader(f))
        if header != CSV_COLUMNS:
            raise RuntimeError(
                f"Schema of {path} ({header}) does not match CSV_COLUMNS ({CSV_COLUMNS}) - rename/back up "
                "the old file; migration is not fabricated."
            )
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def align_signs(Y: np.ndarray, Y_ref: np.ndarray) -> np.ndarray:
    """Flip the sign of Y's columns so they correlate positively with Y_ref's
    columns (eigenvectors are determined only up to sign)."""
    signs = np.sign(np.sum(Y * Y_ref, axis=0))
    signs[signs == 0] = 1.0
    return Y * signs[None, :]


def run_probe(methods: list[str], threads_list: list[int], logger) -> list[dict]:
    import threadpoolctl

    pcfg = load_config()["sammon"]["init_probe"]
    graph, distance, expected_n = str(pcfg["graph"]), str(pcfg["distance"]), int(pcfg["expected_n"])
    n_components, seed = int(pcfg["n_components"]), int(pcfg["seed"])
    for m in methods:
        if m not in METHODS:
            raise ValueError(f"Unknown method '{m}' (expected one of {METHODS}).")
    for t in threads_list:
        if t < 1:
            raise ValueError(f"Number of threads must be >= 1, got {t}.")

    t0 = time.perf_counter()
    D = load_cached_distance(graph, distance, expected_n)
    n = D.shape[0]
    logger.info("Loaded D %s (%s, n=%d) in %.1f s", graph, distance, n, time.perf_counter() - t0)

    rows: list[dict] = []
    Y_ref: np.ndarray | None = None  # first eigh result = reference for differences
    for method in methods:
        min_n = _FORCE_DENSE_MIN_N if method == "eigh" else 0
        for threads in threads_list:
            logger.info("Run %s, BLAS threads=%d ...", method, threads)
            with threadpoolctl.threadpool_limits(limits=threads):
                t_run = time.perf_counter()
                Y0 = init_classical_mds(D, n_components, seed, iterative_min_n=min_n)
                init_sec = time.perf_counter() - t_run
            eig_top = np.sum(Y0 * Y0, axis=0)  # Y0 columns = v_k sqrt(lambda_k) -> |column|^2 = lambda_k
            if method == "eigh" and Y_ref is None:
                Y_ref = Y0.copy()
            diff = float(np.abs(align_signs(Y0, Y_ref) - Y_ref).max()) if Y_ref is not None else float("nan")
            row = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "graph": graph, "distance": distance,
                "n": n, "n_components": n_components, "seed": seed, "method": method, "blas_threads": threads,
                "init_sec": round(init_sec, 3), "eig_top": ";".join(f"{v:.6f}" for v in eig_top),
                "max_abs_diff_vs_eigh_signed": diff,
            }
            _append_row(row)
            rows.append(row)
            logger.info("  -> %s threads=%d: %.2f s, eig_top=%s, max|dY0| vs eigh=%s", method, threads, init_sec, row["eig_top"], diff)
    return rows


def main(argv: list[str] | None = None) -> int:
    pcfg = load_config()["sammon"]["init_probe"]
    parser = argparse.ArgumentParser(description="Probe of init_classical_mds time (eigh vs eigsh) on a cached distance matrix.")
    parser.add_argument("--methods", nargs="+", default=list(pcfg["methods"]), help="subset of %s" % (METHODS,))
    parser.add_argument("--threads", nargs="+", type=int, default=[int(t) for t in pcfg["blas_threads"]])
    args = parser.parse_args(argv)

    logger = get_logger("init_probe")
    from src.experiments.exp_common import keep_system_awake

    with keep_system_awake():
        rows = run_probe(args.methods, args.threads, logger)
    logger.info("Done: %d runs, CSV %s", len(rows), _csv_path())
    return 0


if __name__ == "__main__":
    sys.exit(main())
