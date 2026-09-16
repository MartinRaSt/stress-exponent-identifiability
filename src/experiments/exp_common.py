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
Shared helper functions for experiments E1-E5 (`src/experiments/exp1..5*.py`):
the --quick/--full argument, writing a DONE file for unattended .bat runs,
dynamic discovery of canonical metric keys (so the CSV schema automatically
follows the current `src/sammon/metrics.py`, even while another agent
concurrently extends it with `extended=True`), and a generic
checkpointed/parallel loop over a task list (dataset x method x seed and
analogous grids).
"""
from __future__ import annotations

import argparse
import ctypes
import inspect
import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import numpy as np
import pandas as pd

from src.common.checkpoint import RunKey, append_result, is_done, results_csv_path, save_embedding
from src.common.config import get_path, load_config
from src.common.logging_utils import wall_clock
from src.common.parallel import resolve_n_workers, run_parallel_map
from src.common.progress import progress_iter

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


@contextmanager
def keep_system_awake() -> Iterator[None]:
    """Windows: `SetThreadExecutionState` so a long experiment run does not
    let the computer sleep/hibernate (author's requirement, global rule in
    ~/.claude/CLAUDE.md "Long computations and preventing sleep"). Adds a
    second layer on top of the sleep-blocking set at the .bat level
    (`powercfg`, see `src/common/no_sleep_on.bat`) in case
    `venv\\python.exe -m src.experiments...` is run directly without the
    .bat wrapper. A no-op on non-Windows OSes (fail-loud for an unavailable
    `ctypes.windll` would not make sense here - this is an optional
    safeguard, not a functional requirement)."""
    if sys.platform != "win32":
        yield
        return
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    kernel32.SetThreadExecutionState(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED)
    try:
        yield
    finally:
        kernel32.SetThreadExecutionState(_ES_CONTINUOUS)


def add_mode_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Adds the mutually exclusive --quick/--full/--smoke flags (default: --full).

    --smoke: goes through EVERY dataset x method combination (1 seed, a
    heavily reduced budget - n_max, max_iter/epochs, alpha/lambda grid) -
    unlike --quick it does NOT restrict the set of datasets/methods, only
    their size/number of iterations - the goal is to exercise EVERY code
    path (incl. saving embeddings, extended metrics, graph/temporal metrics,
    statistics, DONE marker) within a few minutes. Results are saved to
    results/data/smoke/ (see `resolve_experiment_name`) so they do not
    contaminate quick/full data."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true", help="quick test: a small subset of data/seeds/iterations")
    group.add_argument("--full", action="store_true", help="full run (default)")
    group.add_argument("--smoke", action="store_true", help="test all dataset x method combinations (1 seed, minimal budget), see docstring")
    return parser


def resolve_mode(args: argparse.Namespace) -> str:
    """Returns 'quick'/'smoke'/'full' (default 'full')."""
    if getattr(args, "smoke", False):
        return "smoke"
    if getattr(args, "quick", False):
        return "quick"
    return "full"


def parse_mode_args(description: str) -> str:
    """Shortcut helper: parses argv and returns the mode ('quick'/'smoke'/'full')."""
    parser = argparse.ArgumentParser(description=description)
    add_mode_args(parser)
    args = parser.parse_args()
    return resolve_mode(args)


def add_dataset_scope_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Q1 step 2 (reserse/2026-09-14_specifikace_rozsireni_q1.md, A.5, A.9 item
    6) - adds `--datasets all|core|holdout` (default 'all'): 'core' = the
    original production set (e.g. `exp1_dr_benchmark.datasets`, anchor
    &exp1_datasets, on which the alpha_pred rule was FITTED), 'holdout' = new
    Q1 candidate datasets (`*_holdout`, `q1_holdout_datasets`), 'all' = the
    union of both (no duplicates) - see `resolve_dataset_scope`."""
    parser.add_argument(
        "--datasets", choices=["all", "core", "holdout"], default="all",
        help="'core'=original production datasets (rule fitting), 'holdout'=new Q1 candidates, 'all'=both (default)",
    )
    return parser


def resolve_dataset_scope(args: argparse.Namespace, core: list[str], holdout: list[str]) -> list[str]:
    """Returns the list of datasets per `--datasets` (see `add_dataset_scope_arg`).
    'all' unions `core`+`holdout` without duplicates, `core` stays FIRST
    (order matters for checkpoint/resume - stable task order across runs)."""
    scope = getattr(args, "datasets", "all")
    if scope == "core":
        return list(core)
    if scope == "holdout":
        return list(holdout)
    if scope != "all":
        raise ValueError(f"Unknown --datasets='{scope}' (expected all/core/holdout).")
    combined = list(core)
    for d in holdout:
        if d not in combined:
            combined.append(d)
    return combined


def add_tier_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Q1 step 2 (A.8, A.9 item 6) - adds `--tier all|tier1|tier2` (E1 only,
    `exp1_dr_benchmark.py`): restricts the set of methods used to
    `exp1_dr_benchmark.method_tiers.<tier>` (default 'all' = all methods
    from `methods:`, unchanged original behavior)."""
    parser.add_argument(
        "--tier", choices=["all", "tier1", "tier2"], default="all",
        help="restrict methods to method_tiers.<tier> (default 'all' = all methods from 'methods:')",
    )
    return parser


def resolve_experiment_name(base_name: str, mode: str) -> str:
    """Returns the actual experiment name used for the checkpoint CSV/embeddings.

    'results/data/' (root) is reserved EXCLUSIVELY for `--full` (production
    data of the article). 'quick' and 'smoke' use their own subdirectories
    'results/data/quick/<base_name>' resp. 'results/data/smoke/<base_name>'
    (checkpoint.py builds the path as `<results_data_dir>/<experiment>_results.csv`
    and `<embeddings_dir>/<experiment>/...` - a '/' in the name thus creates
    a nested path, physically separate from the production data without
    changing `src/common/checkpoint.py`)."""
    if mode == "smoke":
        return f"smoke/{base_name}"
    if mode == "quick":
        return f"quick/{base_name}"
    return base_name


def discover_metric_keys(n_probe: int = 20, n_components: int = 2, seed: int = 0) -> tuple[list[str], dict[str, Any]]:
    """Discovers the canonical set of keys returned by `evaluate()` by
    running it on a small synthetic probe (with labels, so that y-dependent
    keys are also obtained).

    The approach is intentionally dynamic (not a hardcoded list) -
    `src/sammon/` is developed concurrently by another agent and may add
    `extended=True` to `evaluate()`; this probe adapts automatically without
    having to duplicate the keys in every experiment. Returns
    (list_of_keys, kwargs_for_evaluate) - kwargs contains {'extended': True}
    if `evaluate` has this parameter.
    """
    from src.sammon.metrics import evaluate

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_probe, 5))
    Y = rng.normal(size=(n_probe, n_components))
    y = np.array([i % 2 for i in range(n_probe)], dtype=np.int64)

    eval_kwargs: dict[str, Any] = {}
    if "extended" in inspect.signature(evaluate).parameters:
        eval_kwargs["extended"] = True

    metrics = evaluate(X, Y, y, "vector", **eval_kwargs)
    return sorted(metrics.keys()), eval_kwargs


def filter_already_done(experiment_name: str, tasks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Splits the task list into (not-yet-done, count-already-done) per the
    checkpoint `results/data/<experiment_name>_results.csv`. A task must have
    the keys 'dataset_name', 'method_name', 'seed'."""
    todo: list[dict[str, Any]] = []
    n_done = 0
    for task in tasks:
        key = RunKey(experiment_name, task["dataset_name"], task["method_name"], task["seed"])
        if is_done(key):
            n_done += 1
        else:
            todo.append(task)
    return todo, n_done


def check_results_schema(experiment_name: str, column_keys: list[str]) -> None:
    """Fail-fast check BEFORE starting workers: an existing results CSV
    (resume) must contain all columns from `column_keys`. Without this
    check, a schema mismatch would only be revealed by `append_result` on
    the first result, by which time the whole ProcessPool is already running
    (a 2026-09-12 incident: 960 E1 tasks computed for hours into a void).
    Missing columns are NOT automatically added - CSV migration is a
    conscious step taken by the author (a .bak backup + adding empty
    columns), see documentation/2026-09-12_gpu_pcg_a_metriky_velkych_grafu.md
    and projectstate.md."""
    csv_path = results_csv_path(experiment_name)
    if not csv_path.exists():
        return
    import csv as _csv

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        header = next(_csv.reader(f), None)
    if header is None:
        return
    missing = [c for c in column_keys if c not in header]
    if missing:
        raise ValueError(
            f"The schema of the results CSV {csv_path} does not match the current experiment "
            f"columns - missing {missing}. Before resuming, add the columns (with a .bak backup) "
            "or rename the file; nothing is added automatically (fail-loud)."
        )


def run_experiment_grid(
    experiment_name: str,
    tasks: list[dict[str, Any]],
    run_single_fn: Callable[[dict[str, Any]], dict[str, Any]],
    column_keys: list[str],
    logger: logging.Logger,
    sequential: bool = False,
    n_workers: int | None = None,
) -> tuple[int, int]:
    """Generic checkpointed loop over `tasks` (dataset x method x seed and
    analogous grids with the same result shape).

    `run_single_fn` must be a top-level (picklable) function returning a
    dict with keys: 'dataset_name', 'method_name', 'seed', 'status', 'error',
    'wall_time_sec', optionally 'embedding' (ndarray|None), 'metrics' (dict),
    'extra' (dict of additional columns, e.g. 'device', 'n_terms').

    `column_keys` is the canonical list of all expected metric+extra columns
    (so the CSV schema is stable across runs, see `discover_metric_keys`).

    `n_workers` (optional, K8): explicitly overrides `parallel.n_workers`
    from config.yaml only for this specific call - e.g. a limited number of
    workers for memory-intensive tasks on large graphs (see
    `exp3_graph_layout.py`, config `exp3_graph_layout.large_graph_max_workers`).
    `None` (default) preserves the original behavior (config value) for all
    other calls unchanged.

    `sequential=True` forces a run without `ProcessPoolExecutor` (e.g. GPU
    runs that cannot be safely shared across processes - see
    exp_sammon_demo.py; or individual very demanding runs on large n, e.g.
    E2 part (b) scaling, where running several such tasks concurrently does
    not make sense anyway - all "sequential" CPU cores are used by a single run).

    For `sequential=True`, additionally (Task 2c, 2026-09-10, see
    documentation/2026-09-10_oprava_sparse_cg_a_paralelismus.md) BLAS threads
    (OpenBLAS/MKL inside numpy/scipy) are limited to `parallel.blas_threads_sequential`
    from config.yaml via `threadpoolctl.threadpool_limits` - unlike the
    parallel branch (where each `ProcessPoolExecutor` worker gets
    `parallel.blas_threads_per_worker`, see `src/common/parallel.py`),
    everything here runs in a single (main) process, so it can be safely
    given more threads.

    Writing to the CSV ALWAYS happens only in this (main) process, right
    after each completed run from the order-preserving iterator - the run is
    thus safely resumable even if interrupted. Returns (n_ok, n_err) from
    this specific run (tasks already done previously are not counted in
    this sum).
    """
    check_results_schema(experiment_name, column_keys)
    with keep_system_awake():
        if sequential:
            import threadpoolctl

            cfg = load_config()
            try:
                blas_threads_sequential = int(cfg["parallel"]["blas_threads_sequential"])
            except KeyError as exc:
                raise KeyError("Missing key 'parallel.blas_threads_sequential' in config.yaml.") from exc
            with threadpoolctl.threadpool_limits(limits=blas_threads_sequential):
                results_iter: Iterable[dict[str, Any]] = (run_single_fn(t) for t in tasks)
                return _consume_results(experiment_name, tasks, results_iter, column_keys, logger, n_workers=1)

        # the actually resolved number of workers (explicit value / config) - the same
        # computation as in run_parallel_map, so the ETA in _consume_results is computed with
        # the actual level of parallelism
        n_workers_resolved = resolve_n_workers(n_workers)
        results_iter = run_parallel_map(run_single_fn, tasks, n_workers=n_workers)
        return _consume_results(experiment_name, tasks, results_iter, column_keys, logger, n_workers=n_workers_resolved)


def _progress_log_policy() -> tuple[int, int]:
    """Returns (progress_log_every_below, progress_log_every) from the
    `parallel` section of config.yaml (fail-loud on missing keys)."""
    cfg = load_config()
    try:
        every_below = int(cfg["parallel"]["progress_log_every_below"])
        every = int(cfg["parallel"]["progress_log_every"])
    except KeyError as exc:
        raise KeyError("Missing key 'parallel.progress_log_every_below' or 'parallel.progress_log_every' in config.yaml.") from exc
    if every_below < 0 or every < 1:
        raise ValueError(f"parallel.progress_log_every_below={every_below} must be >= 0 and progress_log_every={every} >= 1.")
    return every_below, every


def should_log_progress(i: int, n_total: int, every_below: int, every: int) -> bool:
    """Decides whether a log line is written for the i-th (1-based) completed
    run out of `n_total`: always for N <= every_below, otherwise every
    `every`-th run and always the first and last (see config.yaml parallel.progress_log_*)."""
    if n_total <= every_below:
        return True
    return i == 1 or i == n_total or i % every == 0


def format_progress_line(
    i: int, n_total: int, result: dict[str, Any], elapsed_sec: float, wall_sum_sec: float, n_workers: int,
) -> str:
    """Builds a progress log line `[i/N] dataset method seed status wall=..s |
    elapsed=.. | mean/task=.. | ETA=..`. ETA = average wall_time_sec of
    completed runs x remaining count / n_workers - computed from WORKER time
    (wall_time_sec in the results), not from the consumer's time.

    NOTE: the order-preserving `ProcessPoolExecutor.map` returns results in
    submission order, so a long run at the front of the queue delays
    reporting of short runs behind it (they are already done, but not yet
    counted) - the ETA is therefore an upper-bound estimate (overestimates),
    especially when expensive tasks are ordered first."""
    mean_task = wall_sum_sec / max(i, 1)
    remaining = n_total - i
    eta_sec = mean_task * remaining / max(n_workers, 1)
    return (
        f"[{i}/{n_total}] {result['dataset_name']} {result['method_name']} seed={result['seed']} "
        f"{result['status']} wall={float(result['wall_time_sec']):.1f}s | elapsed={_fmt_hms(elapsed_sec)} | "
        f"mean/task={mean_task:.1f}s | ETA={_fmt_hms(eta_sec)} (n_workers={n_workers})"
    )


def _fmt_hms(seconds: float) -> str:
    """Seconds -> 'H:MM:SS' (for readability of long runs in the log)."""
    total = int(round(max(seconds, 0.0)))
    h, rem = divmod(total, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{sec:02d}"


def _consume_results(
    experiment_name: str,
    tasks: list[dict[str, Any]],
    results_iter: Iterable[dict[str, Any]],
    column_keys: list[str],
    logger: logging.Logger,
    n_workers: int = 1,
) -> tuple[int, int]:
    """Shared loop for both the parallel and sequential branches of
    `run_experiment_grid`: writes each result to the CSV (always only in the
    main process), computes (n_ok, n_err) - see the `run_experiment_grid`
    docstring - and, since 2026-09-13
    (documentation/2026-09-13_hardening_behu.md), logs progress with an ETA
    (`format_progress_line`, frequency per `should_log_progress` /
    config.yaml parallel.progress_log_*). `n_workers` is the actual number
    of concurrent workers (for the ETA)."""
    n_ok, n_err = 0, 0
    n_total = len(tasks)
    every_below, every = _progress_log_policy()
    t_start = time.perf_counter()
    wall_sum_sec = 0.0
    with wall_clock(logger, f"{experiment_name} ({n_total} runs, n_workers={n_workers})"):
        for i, result in enumerate(progress_iter(results_iter, desc=experiment_name, total=n_total), start=1):
            key = RunKey(experiment_name, result["dataset_name"], result["method_name"], result["seed"])
            row: dict[str, Any] = {k: np.nan for k in column_keys}
            row.update(result.get("metrics", {}) or {})
            row.update(result.get("extra", {}) or {})
            row["status"] = result["status"]
            row["error"] = result["error"]
            row["wall_time_sec"] = result["wall_time_sec"]
            append_result(key, row)
            if result.get("embedding") is not None:
                save_embedding(key, result["embedding"])
            # success/failure is determined by 'status', NOT by the presence
            # of an embedding - some experiments (e.g. E2 scaling on large n)
            # deliberately do not save the embedding even on a successful run (saves disk space)
            if result["status"] == "ok":
                n_ok += 1
            else:
                n_err += 1
                logger.warning(
                    "Run failed: dataset=%s method=%s seed=%s -> %s",
                    result["dataset_name"], result["method_name"], result["seed"], result["error"],
                )
            wall_sum_sec += float(result["wall_time_sec"])
            if should_log_progress(i, n_total, every_below, every):
                logger.info(format_progress_line(i, n_total, result, time.perf_counter() - t_start, wall_sum_sec, n_workers))
    return n_ok, n_err


def write_done_file(
    experiment_name: str, logger: logging.Logger, extra_info: dict[str, Any] | None = None,
    max_listed_errors: int = 50,
) -> Path:
    """After the experiment finishes, writes `results/data/<experiment_name>_DONE.txt`
    (completion time, number of CSV rows, number of errors, a NAMED list of
    failed dataset/method/seed/error combinations up to `max_listed_errors`) -
    a signal for unattended .bat runs that the run has truly finished (and
    with what result, including WHAT specifically failed)."""
    from src.common.config import ensure_dir

    csv_path = results_csv_path(experiment_name)
    n_rows, n_errors = 0, 0
    error_lines: list[str] = []
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        n_rows = len(df)
        if "status" in df.columns:
            err_df = df[df["status"] == "error"]
            n_errors = len(err_df)
            for _, row in err_df.head(max_listed_errors).iterrows():
                error_lines.append(f"  FAILED: dataset={row.get('dataset')} method={row.get('method')} seed={row.get('seed')} -> {row.get('error')}")
            if n_errors > max_listed_errors:
                error_lines.append(f"  ... and {n_errors - max_listed_errors} more errors (see full CSV: {csv_path}).")

    done_path = get_path("results_data_dir") / f"{experiment_name}_DONE.txt"
    ensure_dir(done_path.parent)
    lines = [
        f"experiment={experiment_name}",
        f"finished_at={datetime.now().isoformat(timespec='seconds')}",
        f"n_rows_csv={n_rows}",
        f"n_errors={n_errors}",
    ]
    for k, v in (extra_info or {}).items():
        lines.append(f"{k}={v}")
    lines.extend(error_lines)
    done_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("DONE file written: %s (rows=%d, errors=%d)", done_path, n_rows, n_errors)
    if n_errors > 0:
        logger.warning("Error summary (%d): see %s", n_errors, done_path)
    return done_path
