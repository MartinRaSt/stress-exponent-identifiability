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
Exp0 - smoke test of the whole infrastructure: on small datasets
(iris, swiss_roll, karate) runs all compatible methods from the registry
with 2 seeds, computes quality metrics, and saves the results to
results/data/exp0_results.csv (resumable via checkpoint - a repeated
run skips dataset/method/seed combinations already done).

Run: venv\\python.exe -m src.experiments.exp0_smoke
or: src\\run_exp0_smoke.bat
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

# so that `src.*` imports work even when run directly as `python exp0_smoke.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.common.checkpoint import RunKey, append_result, count_done, is_done, results_csv_path, save_embedding
from src.common.config import load_config
from src.common.logging_utils import get_logger, wall_clock
from src.common.parallel import run_parallel_map
from src.common.progress import progress_iter

EXPERIMENT_NAME = "exp0_smoke"


def _loader_kwargs_for(dataset_name: str, exp_cfg: dict[str, Any]) -> dict[str, Any]:
    """Returns any exp0-specific loader parameter overrides (e.g. a smaller swiss_roll)."""
    if dataset_name == "swiss_roll":
        return {"n_samples": exp_cfg["swiss_roll_n_samples"]}
    return {}


def _canonical_metric_keys(cfg: dict[str, Any]) -> list[str]:
    """Builds a fixed list of metric names so the output CSV always has the
    same columns regardless of whether the dataset has labels (y) or the run failed."""
    metrics_cfg = cfg["metrics"]
    keys = [
        "n_samples", "stress_scale_invariant", "sammon_stress",
        "shepard_spearman_rho", "shepard_spearman_pvalue", "auc_rnx",
    ]
    for k in metrics_cfg["neighborhood_k"]:
        keys += [f"trustworthiness_k{k}", f"continuity_k{k}", f"qnx_k{k}", f"rnx_k{k}"]
    keys += [f"neighborhood_hit_k{metrics_cfg['neighborhood_k'][0]}", "silhouette"]
    return keys


def _run_single(task: dict[str, Any]) -> dict[str, Any]:
    """Runs a single (dataset, method, seed) run in a separate worker process.

    Loads the dataset and method again inside the worker (so that the task
    is easily picklable across processes) and returns the result including
    any embedding. Exceptions are not propagated - they are recorded as
    status='error' with a description, so that one incompatible combination
    does not stop the whole experiment run.
    """
    from src.common.seeding import set_seed
    from src.datasets.registry import load_dataset
    from src.methods.registry import get_method
    from src.sammon.metrics import evaluate

    dataset_name = task["dataset_name"]
    method_name = task["method_name"]
    seed = task["seed"]
    n_components = task["n_components"]

    set_seed(seed)
    ds = load_dataset(dataset_name, **task["loader_kwargs"])
    method = get_method(method_name)
    data = ds.X if ds.kind == "vector" else (ds.D if ds.kind == "distance" else ds.graph)

    result: dict[str, Any] = {
        "dataset_name": dataset_name, "method_name": method_name, "seed": seed,
    }
    t0 = time.perf_counter()
    try:
        Y = method.fit_transform(data, ds.kind, seed=seed, n_components=n_components)
        metrics = evaluate(data, Y, ds.y, ds.kind)
        result["status"] = "ok"
        result["error"] = ""
        result["wall_time_sec"] = time.perf_counter() - t0
        result["embedding"] = Y
        result["metrics"] = metrics
    except Exception as exc:  # broad catch is intentional here - see docstring
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["wall_time_sec"] = time.perf_counter() - t0
        result["embedding"] = None
        result["metrics"] = {}
    return result


def main() -> None:
    """Runs the exp0 smoke test over all compatible dataset/method/seed combinations."""
    logger = get_logger(EXPERIMENT_NAME)
    cfg = load_config()
    exp_cfg = cfg["experiments"][EXPERIMENT_NAME]

    # import only here so the registries are populated even when run as a script
    from src.datasets.registry import load_dataset
    from src.methods.common import is_compatible
    from src.methods.registry import get_method, list_registered_methods

    metric_keys = _canonical_metric_keys(cfg)
    n_components = exp_cfg["n_components"]
    seeds = exp_cfg["seeds"]
    methods = list_registered_methods()

    tasks: list[dict[str, Any]] = []
    already_done = 0
    for dataset_name in exp_cfg["datasets"]:
        loader_kwargs = _loader_kwargs_for(dataset_name, exp_cfg)
        ds_probe = load_dataset(dataset_name, **loader_kwargs)
        for method_name in methods:
            method = get_method(method_name)
            if not is_compatible(method.accepts, ds_probe.kind):
                continue
            for seed in seeds:
                key = RunKey(EXPERIMENT_NAME, dataset_name, method_name, seed)
                if is_done(key):
                    already_done += 1
                    continue
                tasks.append({
                    "dataset_name": dataset_name, "method_name": method_name, "seed": seed,
                    "n_components": n_components, "loader_kwargs": loader_kwargs,
                })

    logger.info(
        "Exp0 smoke: %d combinations already done (skipped), %d new to compute.",
        already_done, len(tasks),
    )

    n_ok, n_err = 0, 0
    with wall_clock(logger, f"exp0_smoke ({len(tasks)} runs)"):
        # sequential consumption of results from an order-preserving map() -> each
        # result is written to CSV right after completion, so the run is
        # safely resumable if interrupted (writing happens only in the main process)
        for result in progress_iter(
            run_parallel_map(_run_single, tasks), desc="exp0_smoke", total=len(tasks)
        ):
            key = RunKey(EXPERIMENT_NAME, result["dataset_name"], result["method_name"], result["seed"])
            row: dict[str, Any] = {k: np.nan for k in metric_keys}
            row.update(result["metrics"])
            row["status"] = result["status"]
            row["error"] = result["error"]
            row["wall_time_sec"] = result["wall_time_sec"]
            append_result(key, row)
            if result["embedding"] is not None:
                save_embedding(key, result["embedding"])
                n_ok += 1
            else:
                n_err += 1
                logger.warning(
                    "Run failed: dataset=%s method=%s seed=%s -> %s",
                    result["dataset_name"], result["method_name"], result["seed"], result["error"],
                )

    total_done = count_done(EXPERIMENT_NAME)
    logger.info(
        "Done. Newly successful: %d, newly failed (recorded): %d, skipped: %d, total rows in CSV: %d.",
        n_ok, n_err, already_done, total_done,
    )
    logger.info("Results: %s", results_csv_path(EXPERIMENT_NAME))


if __name__ == "__main__":
    main()
