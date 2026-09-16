# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-16
# License: see the LICENSE file in the repository root
"""
Estimate the duration of a FULL experiment run from the sum of
`wall_time_sec` in an already-existing results CSV (author's rule from
projectstate.md 2026-09-15: estimate run duration from measured times in
the CSV, NOT from comments in old .bat files).

The estimate is not just sum/n_workers: it uses a greedy schedule
simulation (list scheduling) in the ROW order of the results CSV (i.e. in
the order tasks were written by the previous run) - this also captures the
fact that a single very long run near the end of the queue determines the
lower bound on the total time (for E4, the longest single dtsne run is
over 11 minutes). The estimate does NOT include process-startup overhead or
experiment preparation steps (for E4, pre-loading snapshots).

Run: src\run_estimate_runtime.bat exp4_temporal
"""
from __future__ import annotations

import argparse
import heapq
from pathlib import Path

import pandas as pd

from src.common.config import ensure_dir, get_path
from src.common.parallel import resolve_n_workers


def greedy_makespan(durations_sec: list[float], n_workers: int) -> float:
    """Time to complete all tasks with `n_workers` workers that take tasks
    in the given order (list scheduling) - the same behavior as
    `ProcessPoolExecutor.map` with chunksize=1."""
    if n_workers < 1:
        raise ValueError(f"n_workers={n_workers} must be >= 1.")
    free_at = [0.0] * n_workers
    heapq.heapify(free_at)
    makespan = 0.0
    for duration in durations_sec:
        start = heapq.heappop(free_at)
        end = start + float(duration)
        makespan = max(makespan, end)
        heapq.heappush(free_at, end)
    return makespan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", help="experiment name (e.g. exp4_temporal) or path to a CSV")
    parser.add_argument(
        "--workers", default=None,
        help="comma-separated worker counts to compare (default: 1 and parallel.n_workers from config.yaml). "
             "Must be quoted in .bat, otherwise cmd treats the comma as an argument separator.",
    )
    args = parser.parse_args()

    csv_path = Path(args.experiment)
    if not csv_path.exists():
        csv_path = get_path("results_data_dir") / f"{args.experiment}_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Results CSV '{csv_path}' does not exist - the estimate is only computed from MEASURED times "
            "of a previous run (no blind guessing)."
        )

    df = pd.read_csv(csv_path)
    if "wall_time_sec" not in df.columns:
        raise KeyError(f"CSV '{csv_path}' has no 'wall_time_sec' column - there is nothing else to estimate from.")
    durations = df["wall_time_sec"].astype(float).tolist()

    if args.workers:
        # quotes around the list are required in .bat (cmd treats the comma as
        # a separator) and some shells pass them through here - strip before parsing
        worker_counts = [int(x) for x in args.workers.strip(chr(34)+chr(39)).split(",")]
    else:
        # default comparison: sequential vs. the project's actual allowed parallelism
        worker_counts = [1, resolve_n_workers(None)]

    rows = []
    for w in worker_counts:
        makespan = greedy_makespan(durations, w)
        rows.append({
            "experiment": csv_path.stem, "n_runs": len(durations), "n_workers": w,
            "sum_wall_time_h": sum(durations) / 3600.0,
            "max_single_run_min": max(durations) / 60.0,
            "estimated_makespan_h": makespan / 3600.0,
            "estimated_makespan_min": makespan / 60.0,
        })

    out = pd.DataFrame(rows)
    out_path = ensure_dir(get_path("results_data_dir")) / f"runtime_estimate_{csv_path.stem}.csv"
    out.to_csv(out_path, index=False)
    print(f"Source: {csv_path} ({len(durations)} runs, sum {sum(durations) / 3600.0:.2f} h, "
          f"longest single run {max(durations) / 60.0:.1f} min)")
    for row in rows:
        print(f"  {row['n_workers']:3d} workers -> {row['estimated_makespan_h']:.2f} h "
              f"({row['estimated_makespan_min']:.0f} min)")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
