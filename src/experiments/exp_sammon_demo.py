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
Exp_sammon_demo - demonstration run of the proposed `SammonAlpha` method
(alpha=1, the Sammon equivalent) across all implemented solvers
(pseudo-Newton, SMACOF, naive/stabilized SGD, sparse SMACOF) on CPU and GPU
(where the solver is GPU-capable), on the iris, swiss_roll (n=1000), digits,
and karate datasets.

Results (stress, R_NX AUC, wall-clock time) go to
results/data/exp_sammon_demo.csv (resumable via checkpoint) and an embedding
gallery to results/figures/exp_sammon_demo_<dataset>.pdf +
_embeddings.csv (see src/figures/fig_sammon_demo.py).

Run: venv\\python.exe -m src.experiments.exp_sammon_demo
or: src\\run_exp_sammon_demo.bat
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.common.checkpoint import RunKey, append_result, count_done, is_done, load_embedding, results_csv_path, save_embedding
from src.common.config import ensure_dir, get_path, load_config
from src.common.logging_utils import get_logger, wall_clock
from src.common.progress import progress_iter

EXPERIMENT_NAME = "exp_sammon_demo"

# solver names in config.yaml (experiments.exp_sammon_demo.solvers) ->
# values of the `solver` parameter in SammonAlpha (src/sammon/estimator.py)
_SOLVER_NAME_MAP = {"sgd_stab": "sgd", "sgd_naive": "sgd_naive"}


def _loader_kwargs_for(dataset_name: str, exp_cfg: dict[str, Any]) -> dict[str, Any]:
    if dataset_name == "swiss_roll":
        return {"n_samples": exp_cfg["swiss_roll_n_samples"]}
    return {}


def _run_single(task: dict[str, Any]) -> dict[str, Any]:
    """Runs a single (dataset, solver, device) run: fits SammonAlpha and
    evaluates the metrics. Exceptions (e.g. the known limitation of naive SGD
    on duplicate points) are recorded as status='error'; the run is not fabricated."""
    from src.common.seeding import set_seed
    from src.datasets.registry import load_dataset
    from src.sammon.estimator import SammonAlpha
    from src.sammon.metrics import evaluate

    dataset_name = task["dataset_name"]
    solver = task["solver"]
    device = task["device"]
    seed = task["seed"]
    n_components = task["n_components"]
    alpha = task["alpha"]
    max_iter = task["max_iter"]
    tol = task["tol"]

    set_seed(seed)
    ds = load_dataset(dataset_name, **task["loader_kwargs"])
    data = ds.X if ds.kind == "vector" else (ds.D if ds.kind == "distance" else ds.graph)

    result: dict[str, Any] = {"dataset_name": dataset_name, "solver": solver, "device": device, "seed": seed}
    t0 = time.perf_counter()
    try:
        estimator_solver = _SOLVER_NAME_MAP.get(solver, solver)
        est = SammonAlpha(
            alpha=alpha, solver=estimator_solver, device=device, n_components=n_components,
            max_iter=max_iter, tol=tol, seed=seed, verbose=False,
        )
        Y = est.fit_transform(data, kind=ds.kind)
        metrics = evaluate(data, Y, ds.y, ds.kind)
        result["status"] = "ok"
        result["error"] = ""
        result["wall_time_sec"] = time.perf_counter() - t0
        result["embedding"] = Y
        result["metrics"] = metrics
        result["n_terms"] = est.history_.get("n_terms", np.nan)
    except Exception as exc:  # intentionally broad catch - see docstring
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["wall_time_sec"] = time.perf_counter() - t0
        result["embedding"] = None
        result["metrics"] = {}
        result["n_terms"] = np.nan
    return result


def _canonical_metric_keys() -> list[str]:
    return [
        "n_samples", "stress_scale_invariant", "sammon_stress",
        "shepard_spearman_rho", "auc_rnx", "n_terms",
    ]


def main() -> None:
    logger = get_logger(EXPERIMENT_NAME)
    cfg = load_config()
    exp_cfg = cfg["experiments"][EXPERIMENT_NAME]

    from src.datasets.registry import load_dataset

    metric_keys = _canonical_metric_keys()
    n_components = exp_cfg["n_components"]
    seed = exp_cfg["seed"]
    alpha = exp_cfg["alpha"]
    max_iter = exp_cfg["max_iter"]
    tol = exp_cfg["tol"]
    solvers = exp_cfg["solvers"]
    devices = exp_cfg["devices"]
    gpu_capable = set(exp_cfg["gpu_capable_solvers"])

    tasks: list[dict[str, Any]] = []
    already_done = 0
    for dataset_name in exp_cfg["datasets"]:
        loader_kwargs = _loader_kwargs_for(dataset_name, exp_cfg)
        for solver in solvers:
            devices_for_solver = [d for d in devices if d == "cpu" or solver in gpu_capable]
            for device in devices_for_solver:
                method_name = f"{solver}__{device}"
                key = RunKey(EXPERIMENT_NAME, dataset_name, method_name, seed)
                if is_done(key):
                    already_done += 1
                    continue
                tasks.append({
                    "dataset_name": dataset_name, "solver": solver, "device": device,
                    "seed": seed, "n_components": n_components, "alpha": alpha,
                    "max_iter": max_iter, "tol": tol, "loader_kwargs": loader_kwargs,
                })

    logger.info("Exp_sammon_demo: %d combinations already done (skipped), %d new to compute.", already_done, len(tasks))

    n_ok, n_err = 0, 0
    with wall_clock(logger, f"exp_sammon_demo ({len(tasks)} runs)"):
        # sequential run (n_workers=1 explicitly) - the GPU context (CUDA) is
        # not safely shared across ProcessPoolExecutor processes, so we
        # intentionally do NOT use run_parallel_map with multiple workers here
        for task in progress_iter(tasks, desc="exp_sammon_demo", total=len(tasks)):
            result = _run_single(task)
            method_name = f"{result['solver']}__{result['device']}"
            key = RunKey(EXPERIMENT_NAME, result["dataset_name"], method_name, result["seed"])
            row: dict[str, Any] = {k: np.nan for k in metric_keys}
            row.update(result["metrics"])
            row["n_terms"] = result["n_terms"]
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
                    "Run failed: dataset=%s solver=%s device=%s -> %s",
                    result["dataset_name"], result["solver"], result["device"], result["error"],
                )

    total_done = count_done(EXPERIMENT_NAME)
    logger.info(
        "Done. Newly successful: %d, newly failed: %d, skipped: %d, total rows in CSV: %d.",
        n_ok, n_err, already_done, total_done,
    )
    logger.info("Results: %s", results_csv_path(EXPERIMENT_NAME))

    _make_figures(logger, exp_cfg, seed)


def _make_figures(logger, exp_cfg: dict[str, Any], seed: int) -> None:
    """For each dataset, builds a gallery of embeddings from all successful
    runs (see src/figures/fig_sammon_demo.py)."""
    import pandas as pd

    from src.datasets.registry import load_dataset
    from src.figures.fig_sammon_demo import make_gallery_figure

    df = pd.read_csv(results_csv_path(EXPERIMENT_NAME))
    df_ok = df[(df["status"] == "ok") & (df["experiment"] == EXPERIMENT_NAME) & (df["seed"] == seed)]

    figures_dir = ensure_dir(get_path("results_figures_dir"))
    for dataset_name in exp_cfg["datasets"]:
        sub = df_ok[df_ok["dataset"] == dataset_name]
        if sub.empty:
            logger.warning("No successful run for dataset '%s' - the figure is not created.", dataset_name)
            continue
        loader_kwargs = _loader_kwargs_for(dataset_name, exp_cfg)
        ds = load_dataset(dataset_name, **loader_kwargs)

        panels = []
        for _, row in sub.iterrows():
            key = RunKey(EXPERIMENT_NAME, dataset_name, row["method"], seed)
            try:
                Y = load_embedding(key)
            except FileNotFoundError:
                continue
            solver, device = row["method"].split("__")
            panels.append({
                "solver": solver, "device": device, "Y": Y, "labels": ds.y,
                "stress": float(row["stress_scale_invariant"]), "wall_time_sec": float(row["wall_time_sec"]),
            })

        out_pdf = figures_dir / f"{EXPERIMENT_NAME}_{dataset_name}.pdf"
        out_csv = figures_dir / f"{EXPERIMENT_NAME}_{dataset_name}_embeddings.csv"
        make_gallery_figure(dataset_name, panels, out_pdf, out_csv)
        logger.info("Figure saved: %s (%d panels)", out_pdf, len(panels))


if __name__ == "__main__":
    main()
