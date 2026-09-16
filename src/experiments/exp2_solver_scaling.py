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
E2 - solvers: (a) convergence of naive vs. stabilized SGD (digits n=1000,
swiss_roll n=1000, 100 seeds), (b) scaling with n (gaussian_clusters, n up to
50000, smacof/sgd_stab/sparse_smacof/sparse_sgd, CPU and CUDA where
available), (c) approximation error of the sparse model (swiss_roll, digits
n=2000, grid pivots x k, 5 seeds). Three independently checkpointed CSV
outputs: results/data/exp2_convergence_results.csv (+ _curves.csv),
results/data/exp2_scaling_results.csv, results/data/exp2_sparse_results.csv.

Run: venv\\python.exe -m src.experiments.exp2_solver_scaling [--quick|--full]
or: src\\run_exp2_solver_scaling.bat [quick|full]
"""
from __future__ import annotations

import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.common.checkpoint import RunKey, append_result, results_csv_path
from src.common.config import ensure_dir, get_path, load_config
from src.common.logging_utils import get_logger
from src.experiments.config_experiments import resolve_experiment_config
from src.experiments.exp_common import (
    filter_already_done,
    parse_mode_args,
    resolve_experiment_name,
    run_experiment_grid,
    write_done_file,
)

BASE_EXPERIMENT_NAME = "exp2_solver_scaling"
BASE_CONV_EXPERIMENT = "exp2_convergence"
BASE_SCALING_EXPERIMENT = "exp2_scaling"
BASE_SPARSE_EXPERIMENT = "exp2_sparse"

# fixed subsample seed for digits in the convergence part (INDEPENDENT of
# the loop over 100 SGD seeds) - all seeds compare the same subset of points
CONV_SUBSAMPLE_SEED = 42

# solver names in config.yaml (methods.sammon_*) -> values of the `solver`
# parameter in SammonAlpha (src/sammon/estimator.py), same convention as exp_sammon_demo.py
_SOLVER_NAME_MAP = {"sgd_stab": "sgd", "sgd_naive": "sgd_naive"}

# sparse solvers compute "stress" only over the sparse terms (pivots + kNN),
# which is not comparable to the dense stress reported by other solvers (see
# documentation/2026-09-11_kontrola_vysledku_plnych_behu.md, section 3.3) -
# for these solvers, _run_single_scaling additionally computes the full
# (dense) stress over D
_SPARSE_SCALING_SOLVERS = {"sparse_smacof", "sparse_sgd"}


# ---------------------------------------------------------------------------
# shared helper functions
# ---------------------------------------------------------------------------

def _load_D_conv(dataset_name: str, n: int) -> np.ndarray:
    """Loads the distance matrix for the convergence part (a) -
    deterministic across both the main process (reference computation) and workers (individual runs)."""
    from src.datasets.registry import load_dataset
    from src.datasets.subsample import subsample_dataset
    from src.methods.common import to_distance_matrix

    if dataset_name == "digits":
        ds = load_dataset("digits")
        ds = subsample_dataset(ds, n_max=n, random_state=CONV_SUBSAMPLE_SEED)
    elif dataset_name == "swiss_roll":
        ds = load_dataset("swiss_roll", n_samples=n)
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}' for the exp2 convergence part.")
    return to_distance_matrix(ds.X, "vector")


def _reference_smacof_stress(D: np.ndarray, alpha: float, n_components: int, max_iter: int, tol: float, seed: int) -> float:
    """Reference (full) SMACOF run with the same alpha, used as a yardstick
    for the 'converged' flag in the convergence part (a)."""
    from src.sammon.init import init_classical_mds
    from src.sammon.solvers.smacof import smacof_solve
    from src.sammon.weights import alpha_weights, compute_Z_from_W, estimate_eps_D

    scfg = load_config()["sammon"]
    eps_D = estimate_eps_D(D, k=scfg["eps_D"]["k"], q=scfg["eps_D"]["q"], kind="distance")
    W = alpha_weights(D, alpha, eps_D)
    Z = compute_Z_from_W(D, W)
    Y0 = init_classical_mds(D, n_components, seed)
    _, history = smacof_solve(
        D, W, Z, Y0, max_iter=max_iter, tol=tol, eps_num=scfg["eps_num"],
        dense_pinv_threshold=scfg["smacof"]["dense_pinv_threshold"], cg_max_iter=scfg["smacof"]["cg_max_iter"],
        cg_tol=scfg["smacof"]["cg_tol"], reg_rho=scfg["smacof"]["reg_rho"],
    )
    return float(history["stress"][-1])


# ---------------------------------------------------------------------------
# (a) convergence of naive vs. stabilized SGD
# ---------------------------------------------------------------------------

def _run_single_convergence(task: dict[str, Any]) -> dict[str, Any]:
    from src.common.seeding import set_seed
    from src.sammon.init import init_classical_mds
    from src.sammon.solvers.sgd import sgd_solve

    dataset_raw = task["dataset_raw"]
    solver = task["solver"]
    seed = task["seed"]
    n_components = task["n_components"]

    result: dict[str, Any] = {"dataset_name": task["dataset_label"], "method_name": solver, "seed": seed}
    t0 = time.perf_counter()
    try:
        D = _load_D_conv(dataset_raw, task["n"])
        set_seed(seed)
        Y0 = init_classical_mds(D, n_components, seed)
        variant = "naive" if solver == "sgd_naive" else "stabilized"
        Y, history = sgd_solve(
            D, task["alpha"], Y0, epochs=task["epochs"], mu_max=task["mu_max"],
            pairs_per_node=task["pairs_per_node"], eps_anneal=task["eps_anneal"], eps_num=task["eps_num"],
            seed=seed, variant=variant, k_eps=task["k_eps"], q_eps=task["q_eps"],
        )
        final_stress = float(history["stress"][-1])
        converged = int(final_stress < task["converged_ratio"] * task["reference_stress"])

        curve_rows = []
        if seed in task["curve_log_seeds"]:
            for epoch_idx, stress_val in enumerate(history["stress"], start=1):
                curve_rows.append({
                    "dataset": task["dataset_label"], "solver": solver, "seed": seed,
                    "epoch": epoch_idx, "stress_scale_invariant": stress_val,
                })

        result["status"] = "ok"
        result["error"] = ""
        result["wall_time_sec"] = time.perf_counter() - t0
        result["embedding"] = Y
        result["metrics"] = {
            "final_stress": final_stress, "reference_stress": task["reference_stress"],
            "converged": converged, "n_samples": D.shape[0],
        }
        result["extra"] = {}
        result["curve_rows"] = curve_rows
    except Exception as exc:  # intentionally broad - e.g. the known limitation of sgd_naive on duplicate points
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["wall_time_sec"] = time.perf_counter() - t0
        result["embedding"] = None
        result["metrics"] = {}
        result["extra"] = {}
        result["curve_rows"] = []
    return result


def _run_convergence(cfg: dict[str, Any], mode: str, logger, experiment_name: str) -> None:
    ccfg = cfg["convergence"]
    n_components = cfg["n_components"]
    scfg = load_config()["sammon"]["sgd"]

    tasks: list[dict[str, Any]] = []
    ref_stress_by_dataset: dict[str, float] = {}
    for dataset_name in ccfg["datasets"]:
        n = ccfg["digits_n"] if dataset_name == "digits" else ccfg["swiss_roll_n"]
        dataset_label = f"{dataset_name}_n{n}"
        D = _load_D_conv(dataset_name, n)
        ref_stress = _reference_smacof_stress(D, ccfg["alpha"], n_components, cfg["max_iter"], cfg["tol"], seed=0)
        ref_stress_by_dataset[dataset_label] = ref_stress
        logger.info("Convergence: reference SMACOF stress for %s = %.6f (n=%d).", dataset_label, ref_stress, D.shape[0])

        for solver in ccfg["solvers"]:
            for seed in ccfg["seeds"]:
                tasks.append({
                    "dataset_name": dataset_label, "dataset_raw": dataset_name, "dataset_label": dataset_label,
                    "method_name": solver, "n": n, "solver": solver,
                    "seed": seed, "n_components": n_components, "alpha": ccfg["alpha"], "epochs": ccfg["epochs"],
                    "mu_max": scfg["mu_max"], "pairs_per_node": scfg["pairs_per_node"],
                    "eps_anneal": scfg["eps_anneal"], "eps_num": scfg["eps_num"], "k_eps": scfg["k_eps"],
                    "q_eps": scfg["q_eps"], "converged_ratio": ccfg["converged_ratio"],
                    "reference_stress": ref_stress, "curve_log_seeds": set(ccfg["curve_log_seeds"]),
                })

    todo, n_done = filter_already_done(experiment_name, tasks)
    logger.info("%s (mode=%s): %d combinations already done, %d new.", experiment_name, mode, n_done, len(todo))

    from src.common.parallel import run_parallel_map
    from src.common.progress import progress_iter

    curves_csv = results_csv_path(experiment_name).parent / "exp2_convergence_curves.csv"
    ensure_dir(curves_csv.parent)
    column_keys = ["final_stress", "reference_stress", "converged", "n_samples"]

    n_ok, n_err = 0, 0
    for result in progress_iter(run_parallel_map(_run_single_convergence, todo), desc=experiment_name, total=len(todo)):
        key = RunKey(experiment_name, result["dataset_name"], result["method_name"], result["seed"])
        row = {k: np.nan for k in column_keys}
        row.update(result.get("metrics", {}) or {})
        row["status"] = result["status"]
        row["error"] = result["error"]
        row["wall_time_sec"] = result["wall_time_sec"]
        append_result(key, row)
        if result["status"] == "ok":
            n_ok += 1
        else:
            n_err += 1
            logger.warning("Convergence run failed: %s -> %s", key.as_tuple(), result["error"])
        if result["curve_rows"]:
            df_curve = pd.DataFrame(result["curve_rows"])
            df_curve.to_csv(curves_csv, mode="a", header=not curves_csv.exists(), index=False)

    logger.info("Convergence done. Newly OK: %d, newly failed: %d, skipped: %d.", n_ok, n_err, n_done)


# ---------------------------------------------------------------------------
# (b) scaling with n
# ---------------------------------------------------------------------------

def _run_single_scaling(task: dict[str, Any]) -> dict[str, Any]:
    from src.common.seeding import set_seed
    from src.datasets.registry import load_dataset
    from src.sammon.estimator import SammonAlpha

    n = task["n"]
    solver = task["solver"]
    device = task["device"]
    seed = task["seed"]
    dataset_label = f"gaussian_clusters_n{n}"
    method_label = f"{solver}__{device}"

    result: dict[str, Any] = {"dataset_name": dataset_label, "method_name": method_label, "seed": seed}
    t0 = time.perf_counter()
    try:
        set_seed(seed)
        ds = load_dataset(task["base_dataset"], n_samples=n)
        estimator_solver = _SOLVER_NAME_MAP.get(solver, solver)
        est = SammonAlpha(
            alpha=task["alpha"], solver=estimator_solver, device=device, n_components=task["n_components"],
            max_iter=task["max_iter"], tol=task["tol"], seed=seed, verbose=False,
        )

        if device == "cuda":
            from src.sammon.device import measure_gpu_memory

            Y, memory_bytes = measure_gpu_memory(est.fit_transform, ds.X, kind="vector")
        else:
            tracemalloc.start()
            Y = est.fit_transform(ds.X, kind="vector")
            _, memory_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()

        # sparse solvers (sparse_smacof/sparse_sgd) have est.stress_ computed
        # only over the sparse terms (pivots + kNN) - not comparable to
        # dense solvers (see
        # documentation/2026-09-11_kontrola_vysledku_plnych_behu.md, section
        # 3.3). We additionally compute the full (dense) scale-invariant
        # stress over the whole D (n <= max_n_dense, so always available
        # here - see the filter in _run_scaling), the same way as the
        # canonical 'stress_scale_invariant' metric used elsewhere in the
        # project (evaluate(), E1/E2c) - alpha=0 weights, optimal rescaling s*.
        stress_sparse_terms = np.nan
        if solver in _SPARSE_SCALING_SOLVERS:
            from src.methods.common import to_distance_matrix
            from src.sammon.metrics import evaluate

            stress_sparse_terms = float(est.stress_)
            D_full = to_distance_matrix(ds.X, "vector")
            full_metrics = evaluate(D_full, Y, None, "distance")
            stress_scale_invariant = float(full_metrics["stress_scale_invariant"])
        else:
            stress_scale_invariant = float(est.stress_)

        result["status"] = "ok"
        result["error"] = ""
        result["wall_time_sec"] = time.perf_counter() - t0
        result["embedding"] = None  # embeddings for large n (up to 50000) are not saved (disk space)
        result["metrics"] = {
            "stress_scale_invariant": stress_scale_invariant, "stress_sparse_terms": stress_sparse_terms,
            "memory_bytes": int(memory_bytes), "n_samples": n,
        }
        result["extra"] = {}
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["wall_time_sec"] = time.perf_counter() - t0
        result["embedding"] = None
        result["metrics"] = {}
        result["extra"] = {}
    return result


def _run_scaling(cfg: dict[str, Any], mode: str, logger, experiment_name: str) -> None:
    scfg = cfg["scaling"]
    gpu_capable = set(scfg["gpu_capable_solvers"])

    tasks: list[dict[str, Any]] = []
    for n in scfg["n_grid"]:
        if n > scfg["max_n_dense"]:
            logger.info("Scaling: n=%d > max_n_dense=%d, skipping all solvers for this n (a dense O(n^2) representation is required by all current solvers, see documentation/2026-09-09_sammon_core.md).", n, scfg["max_n_dense"])
            continue
        for solver in scfg["solvers"]:
            devices = [d for d in scfg["devices"] if d == "cpu" or solver in gpu_capable]
            for device in devices:
                for seed in scfg["seeds"]:
                    tasks.append({
                        "dataset_name": f"{scfg['base_dataset']}_n{n}", "method_name": f"{solver}__{device}",
                        "n": n, "solver": solver, "device": device, "seed": seed,
                        "base_dataset": scfg["base_dataset"], "alpha": scfg["alpha"],
                        "n_components": cfg["n_components"], "max_iter": cfg["max_iter"], "tol": cfg["tol"],
                    })

    # sequential run (without ProcessPoolExecutor): a mix of CPU/CUDA tasks
    # in one loop - a CUDA context cannot be safely shared across processes
    # (see exp_sammon_demo.py, same convention); a large n already dominates
    # the time on its own, so the loss from not parallelizing CPU runs is negligible.
    todo, n_done = filter_already_done(experiment_name, tasks)
    logger.info("%s (mode=%s): %d combinations already done, %d new.", experiment_name, mode, n_done, len(todo))
    column_keys = ["stress_scale_invariant", "stress_sparse_terms", "memory_bytes", "n_samples"]
    n_ok, n_err = run_experiment_grid(experiment_name, todo, _run_single_scaling, column_keys, logger, sequential=True)
    logger.info("Scaling done. Newly OK: %d, newly failed: %d, skipped: %d.", n_ok, n_err, n_done)


# ---------------------------------------------------------------------------
# (c) sparse model approximation error
# ---------------------------------------------------------------------------

def _load_X_sparse(dataset_name: str, n: int) -> np.ndarray:
    from src.datasets.registry import load_dataset
    from src.datasets.subsample import subsample_dataset

    if dataset_name == "digits":
        ds = load_dataset("digits")
        ds = subsample_dataset(ds, n_max=n, random_state=CONV_SUBSAMPLE_SEED)
        return ds.X
    if dataset_name == "swiss_roll":
        return load_dataset("swiss_roll", n_samples=n).X
    raise ValueError(f"Unknown dataset '{dataset_name}' for the exp2 sparse part.")


def _reference_full_run(X: np.ndarray, alpha: float, n_components: int, max_iter: int, tol: float, seed: int) -> tuple[float, float, float]:
    """Full (non-sparse) SMACOF run via SammonAlpha - returns (stress, auc_rnx, wall_time_sec).

    `stress` is the canonical 'stress_scale_invariant' from `evaluate()`
    (alpha=0 weights, optimal rescaling s*) - the SAME metric that
    `_run_single_sparse` uses for `stress_sparse` (see
    documentation/2026-09-11_kontrola_vysledku_plnych_behu.md, section 3.3),
    so that `stress_gap` compares two comparable numbers. Originally, this
    function returned `est.stress_` (the solver's E_alpha at its own alpha),
    which is a different quantity than the sparse-terms stress and made
    `stress_gap` produce nonsensical (even negative) values."""
    from src.sammon.estimator import SammonAlpha
    from src.sammon.metrics import evaluate

    t0 = time.perf_counter()
    est = SammonAlpha(alpha=alpha, solver="smacof", device="cpu", n_components=n_components, max_iter=max_iter, tol=tol, seed=seed, verbose=False)
    Y = est.fit_transform(X, kind="vector")
    elapsed = time.perf_counter() - t0
    metrics = evaluate(X, Y, None, "vector")
    return float(metrics["stress_scale_invariant"]), float(metrics["auc_rnx"]), elapsed


def _run_single_sparse(task: dict[str, Any]) -> dict[str, Any]:
    from src.common.seeding import set_seed
    from src.sammon.device import resolve_device
    from src.sammon.estimator import SammonAlpha
    from src.sammon.metrics import evaluate

    dataset_name = task["dataset_name"]
    n_pivots = task["n_pivots"]
    k_neighbors = task["k_neighbors"]
    seed = task["seed"]
    method_label = f"sparse_p{n_pivots}_k{k_neighbors}"

    result: dict[str, Any] = {"dataset_name": dataset_name, "method_name": method_label, "seed": seed}
    t0 = time.perf_counter()
    try:
        set_seed(seed)
        X = _load_X_sparse(dataset_name, task["n"])
        # temporarily override the sparse hyperparameters via a direct
        # computation (bypasses SammonAlpha.__init__, which reads
        # n_pivots/k_neighbors only from config.yaml)
        from src.sammon.init import init_classical_mds
        from src.sammon.solvers.sparse import build_sparse_terms, sparse_smacof_solve
        from src.sammon.weights import estimate_eps_D
        from src.methods.common import to_distance_matrix

        D = to_distance_matrix(X, "vector")
        scfg = load_config()["sammon"]
        eps_D = estimate_eps_D(D, k=scfg["eps_D"]["k"], q=scfg["eps_D"]["q"], kind="distance")
        terms = build_sparse_terms(D, task["alpha"], eps_D, n_pivots=n_pivots, k_neighbors=k_neighbors, seed=seed, kind="distance", exact_knn_threshold=scfg["sparse"]["exact_knn_threshold"])
        Y0 = init_classical_mds(D, task["n_components"], seed)
        Y, history = sparse_smacof_solve(
            terms, Y0, max_iter=task["max_iter"], tol=task["tol"], eps_num=scfg["eps_num"],
            cg_max_iter=scfg["smacof"]["cg_max_iter"], cg_tol=scfg["smacof"]["cg_tol"], reg_rho=scfg["smacof"]["reg_rho"],
        )
        elapsed = time.perf_counter() - t0
        metrics = evaluate(D, Y, None, "distance")
        # 'stress_sparse' is now the canonical full (dense) scale-invariant
        # stress over the whole D (comparable to the reference); the
        # original sparse-terms-only value (history['stress'][-1], only over
        # pivots+kNN) is kept separately in 'stress_sparse_terms' (section 3.3)
        stress_sparse = float(metrics["stress_scale_invariant"])
        stress_sparse_terms = float(history["stress"][-1])
        auc_sparse = float(metrics["auc_rnx"])

        ref_stress, ref_auc, ref_time = task["reference"]
        stress_gap = (stress_sparse - ref_stress) / ref_stress if ref_stress > 0 else np.nan
        auc_gap = ref_auc - auc_sparse
        speedup = ref_time / elapsed if elapsed > 0 else np.nan

        result["status"] = "ok"
        result["error"] = ""
        result["wall_time_sec"] = elapsed
        result["embedding"] = Y
        result["metrics"] = {
            "stress_sparse": stress_sparse, "stress_sparse_terms": stress_sparse_terms,
            "auc_sparse": auc_sparse, "stress_gap": stress_gap,
            "auc_gap": auc_gap, "speedup": speedup, "n_terms": int(terms["row"].shape[0]),
            "reference_stress": ref_stress, "reference_auc_rnx": ref_auc,
        }
        result["extra"] = {}
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["wall_time_sec"] = time.perf_counter() - t0
        result["embedding"] = None
        result["metrics"] = {}
        result["extra"] = {}
    return result


def _run_sparse_approx(cfg: dict[str, Any], mode: str, logger, experiment_name: str) -> None:
    spcfg = cfg["sparse_approx"]
    tasks: list[dict[str, Any]] = []
    for dataset_name in spcfg["datasets"]:
        n = spcfg["digits_n"] if dataset_name == "digits" else spcfg["swiss_roll_n"]
        X = _load_X_sparse(dataset_name, n)
        ref_stress, ref_auc, ref_time = _reference_full_run(X, spcfg["alpha"], cfg["n_components"], cfg["max_iter"], cfg["tol"], seed=0)
        logger.info("Sparse gap: reference (full) run %s (n=%d): stress=%.5f auc_rnx=%.4f time=%.2fs.", dataset_name, n, ref_stress, ref_auc, ref_time)
        for n_pivots in spcfg["n_pivots_grid"]:
            for k_neighbors in spcfg["k_neighbors_grid"]:
                for seed in spcfg["seeds"]:
                    tasks.append({
                        "dataset_name": dataset_name, "method_name": f"sparse_p{n_pivots}_k{k_neighbors}",
                        "n": n, "n_pivots": n_pivots, "k_neighbors": k_neighbors,
                        "seed": seed, "alpha": spcfg["alpha"], "n_components": cfg["n_components"],
                        "max_iter": cfg["max_iter"], "tol": cfg["tol"], "reference": (ref_stress, ref_auc, ref_time),
                    })

    todo, n_done = filter_already_done(experiment_name, tasks)
    logger.info("%s (mode=%s): %d combinations already done, %d new.", experiment_name, mode, n_done, len(todo))
    column_keys = [
        "stress_sparse", "stress_sparse_terms", "auc_sparse", "stress_gap", "auc_gap", "speedup", "n_terms",
        "reference_stress", "reference_auc_rnx",
    ]
    n_ok, n_err = run_experiment_grid(experiment_name, todo, _run_single_sparse, column_keys, logger)
    logger.info("Sparse gap done. Newly OK: %d, newly failed: %d, skipped: %d.", n_ok, n_err, n_done)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    mode = parse_mode_args("E2: SGD convergence, scaling with n, sparse model approximation error.")
    logger = get_logger(BASE_EXPERIMENT_NAME, mode=mode)
    cfg = resolve_experiment_config(BASE_EXPERIMENT_NAME, mode)
    experiment_name = resolve_experiment_name(BASE_EXPERIMENT_NAME, mode)
    conv_experiment = resolve_experiment_name(BASE_CONV_EXPERIMENT, mode)
    scaling_experiment = resolve_experiment_name(BASE_SCALING_EXPERIMENT, mode)
    sparse_experiment = resolve_experiment_name(BASE_SPARSE_EXPERIMENT, mode)

    _run_convergence(cfg, mode, logger, conv_experiment)
    _run_scaling(cfg, mode, logger, scaling_experiment)
    _run_sparse_approx(cfg, mode, logger, sparse_experiment)

    # E2 has no single shared results CSV (three independently checkpointed
    # subparts), so the DONE file summarizes the row counts of all three separately
    extra_info: dict[str, Any] = {"mode": mode}
    for sub_experiment in (conv_experiment, scaling_experiment, sparse_experiment):
        p = results_csv_path(sub_experiment)
        n_rows = len(pd.read_csv(p)) if p.exists() else 0
        extra_info[f"n_rows_{sub_experiment}"] = n_rows
    write_done_file(experiment_name, logger, extra_info=extra_info)


if __name__ == "__main__":
    main()
