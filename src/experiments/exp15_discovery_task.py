# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-18
# License: see the LICENSE file in the repository root
"""
E15 "downstream discovery task" (author request 2026-09-18, motivated by a
DAMI editor comment: "there is not a single task in the whole manuscript
where a better map leads to a better FINDING"). Computes the two discovery
questions of `src/sammon/discovery_task.py` (Q1 `nearest_class_pair`, Q2
`most_dispersed_class`) over the ALREADY SAVED
`results/data/embeddings/exp1_dr_benchmark/<dataset>__<method>__<seed>.npy`
embeddings - NO re-run of any dimensionality-reduction method, exactly the
same "no re-run" pattern as `exp1_cluster_geometry.py` (K2), whose
`_load_dataset_cache` (subsample reproduced EXACTLY as in the production E1
run) is reused directly here rather than duplicated.

Only methods listed in `report.main_methods` (config_experiments.yaml,
single source of truth for the article's main-table method set) are
evaluated - `exp1_dr_benchmark_results.csv` also contains Tier-2/ablation
methods that are out of scope for this downstream-task comparison.

Output schema: one row per (dataset, method, seed, question) - see
`COLUMN_KEYS`. The checkpoint/resume KEY is
`RunKey(experiment, dataset, f"{method}__{question}", seed)` (the composite
method string is the SAME convention `exp6_alpha_curves._method_name_for_alpha`
uses to fold an extra dimension - here `question` - into the
(dataset, method, seed) grid `src/common/checkpoint.py` was built around);
the plain method name is carried separately in the `dr_method` column so
downstream joins against `report.main_methods` need no string surgery. This
resumes CORRECTLY at row (not just task) granularity: if the process is
interrupted between writing the Q1 and the Q2 row of the same
(dataset, method, seed), only the missing question is recomputed on restart
- see `_pending_questions`.

Output: results/data/[<mode>/]exp15_discovery_task_results.csv, DONE file,
log. Run:
  venv\\python.exe -m src.experiments.exp15_discovery_task [--quick|--full|--smoke]
or: src\\run_exp15_discovery_task.bat [quick|full|smoke]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from src.common.checkpoint import RunKey, append_result, is_done, load_embedding, results_csv_path
from src.common.logging_utils import get_logger, wall_clock
from src.common.progress import progress_iter
from src.experiments.config_experiments import load_experiments_config, resolve_experiment_config
from src.experiments.exp1_cluster_geometry import _load_dataset_cache
from src.experiments.exp_common import (
    check_results_schema,
    keep_system_awake,
    parse_mode_args,
    resolve_experiment_name,
    write_done_file,
)
from src.sammon.discovery_task import DISCOVERY_QUESTIONS, DISCOVERY_TASK_KEYS, discovery_task_rows

BASE_EXPERIMENT_NAME = "exp15_discovery_task"
SOURCE_BASE_EXPERIMENT_NAME = "exp1_dr_benchmark"

COLUMN_KEYS = ["dr_method", "question"] + DISCOVERY_TASK_KEYS


def _composite_key(experiment: str, dataset: str, method: str, seed: int, question: str) -> RunKey:
    """RunKey with `question` folded into the method field - see the module docstring."""
    return RunKey(experiment, dataset, f"{method}__{question}", seed)


def _pending_questions(experiment: str, dataset_name: str, method_name: str, seed: int) -> list[str]:
    """Which of `DISCOVERY_QUESTIONS` are NOT yet in the results CSV for this
    (dataset, method, seed) - row-granular resume, see the module docstring."""
    return [
        q for q in DISCOVERY_QUESTIONS
        if not is_done(_composite_key(experiment, dataset_name, method_name, seed, q))
    ]


def _compute_rows_for_task(
    source_experiment: str, dataset_name: str, method_name: str, seed: int, pending: list[str],
    D_in: np.ndarray | None, y: np.ndarray | None, load_error: str | None, tie_relative_tolerance: float,
) -> dict[str, dict[str, Any]]:
    """Compute the result payload for every question in `pending` for one
    (dataset, method, seed) combination - factored out of `main()`'s loop so
    it is directly unit-testable (mock `load_embedding` + a synthetic D_in/y,
    no real dataset/config needed). Returns {question: {..., 'status',
    'error'}}; on ANY failure (missing/mismatched embedding, dataset load
    error) EVERY pending question gets an 'error' row - a partial success
    within one embedding is not meaningful (the same D_out feeds both
    questions)."""
    if load_error is not None:
        return {q: {"question": q, "status": "error", "error": load_error} for q in pending}
    try:
        Y = load_embedding(RunKey(source_experiment, dataset_name, method_name, seed))
        Y = np.asarray(Y, dtype=np.float64)
        if Y.shape[0] != D_in.shape[0]:
            raise ValueError(
                f"Number of embedding points ({Y.shape[0]}) does not match the D_in "
                f"subsample ({D_in.shape[0]}) - check n_samples_used/subsample seed."
            )
        D_out = squareform(pdist(Y, metric="euclidean"))
        computed = discovery_task_rows(D_in, D_out, y, tie_relative_tolerance)
        return {r["question"]: {**r, "status": "ok", "error": ""} for r in computed if r["question"] in pending}
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        return {q: {"question": q, "status": "error", "error": err} for q in pending}


def main() -> None:
    mode = parse_mode_args("E15: downstream discovery task over saved exp1_dr_benchmark embeddings.")
    logger = get_logger(BASE_EXPERIMENT_NAME, mode=mode)
    cfg = resolve_experiment_config(BASE_EXPERIMENT_NAME, mode)
    EXPERIMENT_NAME = resolve_experiment_name(BASE_EXPERIMENT_NAME, mode)
    SOURCE_EXPERIMENT = resolve_experiment_name(SOURCE_BASE_EXPERIMENT_NAME, mode)
    tie_relative_tolerance = float(cfg["tie_relative_tolerance"])

    report_cfg = load_experiments_config()["report"]
    main_methods = list(report_cfg["main_methods"])

    source_csv = results_csv_path(SOURCE_EXPERIMENT)
    if not source_csv.exists():
        raise FileNotFoundError(
            f"Missing input for E15: {source_csv}. First run "
            f"src\\run_exp1_dr_benchmark.bat {mode} (or directly "
            f"venv\\python.exe -m src.experiments.exp1_dr_benchmark --{mode})."
        )
    src_df = pd.read_csv(source_csv)
    ok = src_df[(src_df["status"] == "ok") & (src_df["method"].isin(main_methods))]
    ok = ok[["dataset", "method", "seed", "n_samples_used"]].drop_duplicates()
    if ok.empty:
        raise ValueError(
            f"{source_csv} contains no successful row (status=='ok') for a report.main_methods method - nothing to compute."
        )

    max_datasets = cfg.get("max_datasets")
    if max_datasets is not None:
        keep_datasets = sorted(ok["dataset"].unique())[: int(max_datasets)]
        ok = ok[ok["dataset"].isin(keep_datasets)]
        logger.info("mode=%s: restricting to %d datasets (max_datasets): %s", mode, len(keep_datasets), keep_datasets)

    check_results_schema(EXPERIMENT_NAME, COLUMN_KEYS)

    tasks: list[dict[str, Any]] = [
        {"dataset_name": r.dataset, "method_name": r.method, "seed": int(r.seed), "n_samples_used": int(r.n_samples_used)}
        for r in ok.itertuples(index=False)
    ]

    todo: list[dict[str, Any]] = []
    n_tasks_fully_done = 0
    for task in tasks:
        pending = _pending_questions(EXPERIMENT_NAME, task["dataset_name"], task["method_name"], task["seed"])
        if not pending:
            n_tasks_fully_done += 1
        else:
            task = dict(task)
            task["pending_questions"] = pending
            todo.append(task)
    logger.info(
        "%s (mode=%s): %d dataset/method/seed combinations fully done (skipped), %d with at least one pending question.",
        EXPERIMENT_NAME, mode, n_tasks_fully_done, len(todo),
    )

    dataset_cache: dict[str, tuple[np.ndarray | None, np.ndarray | None, int, str | None]] = {}
    n_ok, n_err = 0, 0
    with keep_system_awake():
        with wall_clock(logger, f"{EXPERIMENT_NAME} ({len(todo)} dataset/method/seed combinations)"):
            for task in progress_iter(todo, desc=EXPERIMENT_NAME, total=len(todo)):
                dataset_name = task["dataset_name"]
                method_name = task["method_name"]
                seed = task["seed"]
                pending = task["pending_questions"]

                if dataset_name not in dataset_cache:
                    dataset_cache[dataset_name] = _load_dataset_cache(dataset_name, task["n_samples_used"])
                D_in, y, _n_classes, load_error = dataset_cache[dataset_name]

                t0 = time.perf_counter()
                rows_by_question = _compute_rows_for_task(
                    SOURCE_EXPERIMENT, dataset_name, method_name, seed, pending, D_in, y, load_error, tie_relative_tolerance,
                )
                wall_time_sec = time.perf_counter() - t0
                for question in pending:
                    payload = rows_by_question[question]
                    out_row: dict[str, Any] = {k: np.nan for k in COLUMN_KEYS}
                    out_row.update({k: v for k, v in payload.items() if k in DISCOVERY_TASK_KEYS})
                    out_row["dr_method"] = method_name
                    out_row["question"] = question
                    out_row["status"] = payload["status"]
                    out_row["error"] = payload["error"]
                    out_row["wall_time_sec"] = wall_time_sec
                    key = _composite_key(EXPERIMENT_NAME, dataset_name, method_name, seed, question)
                    append_result(key, out_row)
                    if payload["status"] == "ok":
                        n_ok += 1
                    else:
                        n_err += 1
                        logger.warning(
                            "Run failed: dataset=%s method=%s seed=%s question=%s -> %s",
                            dataset_name, method_name, seed, question, payload["error"],
                        )

    logger.info("Done. Newly successful rows: %d, newly failed rows: %d, combinations fully skipped: %d.", n_ok, n_err, n_tasks_fully_done)
    write_done_file(EXPERIMENT_NAME, logger, extra_info={"mode": mode})


if __name__ == "__main__":
    main()
