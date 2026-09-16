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
E5 - ablation study: alpha-Sammon (SMACOF) over an alpha grid x init
{pca,random,classical_mds} x eps_D quantile (q) on 6 datasets (iris,
digits, coil20, swiss_roll, severed_sphere, football - graph converted to
a shortest-path distance matrix). 5 seeds. Low-level approach (calling
weights/init/smacof_solve directly, not SammonAlpha) - needed because the
eps_D quantile q is not exposed in `SammonAlpha` as a separate parameter
(it is always read from config.yaml sammon.eps_D.q). Results go to
results/data/exp5_ablation_results.csv (resumable).

Run: venv\\python.exe -m src.experiments.exp5_ablation [--quick|--full]
or: src\\run_exp5_ablation.bat [quick|full]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.common.config import load_config
from src.common.logging_utils import get_logger
from src.experiments.config_experiments import resolve_experiment_config
from src.experiments.exp_common import (
    discover_metric_keys,
    filter_already_done,
    parse_mode_args,
    resolve_experiment_name,
    run_experiment_grid,
    write_done_file,
)

BASE_EXPERIMENT_NAME = "exp5_ablation"

_N_OVERRIDE_KEY = {"digits": "digits_n", "coil20": "coil20_n", "swiss_roll": "swiss_roll_n", "severed_sphere": "severed_sphere_n"}


def _load_data(dataset_name: str, cfg: dict[str, Any]) -> tuple[str, np.ndarray, np.ndarray | None]:
    """Returns (kind, D_or_X, y) for the given E5 dataset. The graph dataset
    (football) is converted to a shortest-path distance matrix; vector
    datasets are subsampled to the configured n if needed."""
    from src.datasets.registry import load_dataset
    from src.datasets.subsample import subsample_dataset

    if dataset_name == "football":
        from src.datasets.graph_distance import resistance_distance, shortest_path_distance

        ds = load_dataset("football")
        fn = shortest_path_distance if cfg["football_distance_metric"] == "shortest_path" else resistance_distance
        return "distance", fn(ds.graph), ds.y

    ds = load_dataset(dataset_name)
    override_key = _N_OVERRIDE_KEY.get(dataset_name)
    if override_key is not None:
        ds = subsample_dataset(ds, n_max=cfg[override_key], random_state=42)
    return "vector", ds.X, ds.y


def _run_single(task: dict[str, Any]) -> dict[str, Any]:
    from src.common.seeding import set_seed
    from src.methods.common import to_distance_matrix
    from src.sammon.init import init_classical_mds, init_pca, init_random
    from src.sammon.metrics import evaluate
    from src.sammon.solvers.smacof import smacof_solve
    from src.sammon.weights import alpha_weights, compute_Z_from_W, estimate_eps_D

    dataset_name = task["dataset_name"]
    alpha = task["alpha"]
    init_name = task["init"]
    eps_D_q = task["eps_D_q"]
    seed = task["seed"]
    method_name = task["method_name"]

    result: dict[str, Any] = {"dataset_name": dataset_name, "method_name": method_name, "seed": seed}
    t0 = time.perf_counter()
    try:
        set_seed(seed)
        kind, data, y = _load_data(dataset_name, task["ablation_cfg"])
        D = to_distance_matrix(data, kind)
        n = D.shape[0]
        scfg = task["sammon_cfg"]

        eps_D = estimate_eps_D(D, k=scfg["eps_D"]["k"], q=eps_D_q, kind="distance")
        W = alpha_weights(D, alpha, eps_D)
        Z = compute_Z_from_W(D, W)

        if init_name == "pca":
            if kind != "vector":
                Y0 = init_classical_mds(D, task["n_components"], seed)  # PCA requires X, see SammonAlpha._init_Y0
            else:
                Y0 = init_pca(data, task["n_components"], seed)
        elif init_name == "random":
            scale = float(scfg["init"]["random_scale"]) * float(D[~np.eye(n, dtype=bool)].mean())
            Y0 = init_random(n, task["n_components"], seed, scale=scale)
        else:  # classical_mds
            Y0 = init_classical_mds(D, task["n_components"], seed)

        Y, history = smacof_solve(
            D, W, Z, Y0, max_iter=task["max_iter"], tol=task["tol"], eps_num=scfg["eps_num"],
            dense_pinv_threshold=scfg["smacof"]["dense_pinv_threshold"], cg_max_iter=scfg["smacof"]["cg_max_iter"],
            cg_tol=scfg["smacof"]["cg_tol"], reg_rho=scfg["smacof"]["reg_rho"],
        )
        metrics = evaluate(D, Y, y, "distance", **task["eval_kwargs"])

        result["status"] = "ok"
        result["error"] = ""
        result["wall_time_sec"] = time.perf_counter() - t0
        result["embedding"] = Y
        result["metrics"] = metrics
        result["extra"] = {"alpha": alpha, "init": init_name, "eps_D_q": eps_D_q, "n_iter_smacof": history["n_iter"]}
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["wall_time_sec"] = time.perf_counter() - t0
        result["embedding"] = None
        result["metrics"] = {}
        result["extra"] = {"alpha": alpha, "init": init_name, "eps_D_q": eps_D_q, "n_iter_smacof": np.nan}
    return result


def main() -> None:
    mode = parse_mode_args("E5: ablation study of alpha x init x eps_D quantile on 6 datasets.")
    logger = get_logger(BASE_EXPERIMENT_NAME, mode=mode)
    cfg = resolve_experiment_config(BASE_EXPERIMENT_NAME, mode)
    EXPERIMENT_NAME = resolve_experiment_name(BASE_EXPERIMENT_NAME, mode)
    sammon_cfg = load_config()["sammon"]

    n_components = cfg["n_components"]
    metric_keys, eval_kwargs = discover_metric_keys(n_components=n_components)
    column_keys = metric_keys + ["alpha", "init", "eps_D_q", "n_iter_smacof"]

    tasks: list[dict[str, Any]] = []
    for dataset_name in cfg["datasets"]:
        for alpha in cfg["alpha_grid"]:
            for init_name in cfg["inits"]:
                for eps_D_q in cfg["eps_D_q_grid"]:
                    for seed in cfg["seeds"]:
                        method_name = f"alpha{alpha}_init{init_name}_epsq{eps_D_q}"
                        tasks.append({
                            "dataset_name": dataset_name, "method_name": method_name, "seed": seed,
                            "alpha": alpha, "init": init_name, "eps_D_q": eps_D_q,
                            "n_components": n_components, "max_iter": cfg["max_iter"], "tol": cfg["tol"],
                            "ablation_cfg": cfg, "sammon_cfg": sammon_cfg, "eval_kwargs": eval_kwargs,
                        })

    todo, n_done = filter_already_done(EXPERIMENT_NAME, tasks)
    logger.info("%s (mode=%s): %d combinations already done (skipped), %d new to compute.", EXPERIMENT_NAME, mode, n_done, len(todo))

    n_ok, n_err = run_experiment_grid(EXPERIMENT_NAME, todo, _run_single, column_keys, logger)
    logger.info("Done. Newly successful: %d, newly failed: %d, skipped: %d.", n_ok, n_err, n_done)
    write_done_file(EXPERIMENT_NAME, logger, extra_info={"mode": mode})


if __name__ == "__main__":
    main()
