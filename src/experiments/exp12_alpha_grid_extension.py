# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-17
# License: see the LICENSE file in the repository root
"""
E12: extension of the exp6_alpha_curves alpha grid beyond alpha=3
(documentation/2026-09-17_zadani_exp12_prodlouzeni_mrizky.md).

11 of 51 datasets in `exp6_alpha_curves_results.csv` have their auc_rnx
optimum exactly at alpha=3, the right edge of E6's grid (mostly smooth,
low-dimensional manifolds with very low rho_NN: s_curve, helix, sphere,
torus, trefoil_knot, mobius_strip, klein_bottle_4d, swiss_roll,
twin_peaks). This means the measured gain from tuning alpha on those
datasets is only a LOWER BOUND - the true optimum may lie further out, and
"optimum sits on the grid boundary" is a standard reviewer objection.

This experiment recomputes ONLY the alpha in (3, 6] on the 18 datasets
listed in `exp12_alpha_grid_extension.datasets` (config_experiments.yaml):
the 14 datasets whose E6 optimum sits at/near alpha=3, plus interior-optimum
controls (wall_robot, wine, swiss_roll_hole, ionosphere) which should NOT
drift further out - a sanity check that the extension does not simply shift
every optimum outward.

The grid cap of 6.0 is a NUMERICAL, not a modeling, choice: the weight
dynamic range `rho(alpha) = ((D_max+eps_D)/(D_min+eps_D))^alpha` degrades
the conditioning of the weighted Laplacian used in the Guttman transform
(clanek/sections/03_metoda.tex, remark on the role of eps_D). Measured on
the target datasets (see the "numerical ceiling" table in the zadani
document), `log10(rho)` stays below ~13 for every dataset up to alpha=6 -
comfortably inside float64's ~16 valid digits - while `mfeat_morphological`
already reaches ~17 at alpha=8. Hence alpha_grid stops at 6.0.

Fit and evaluation are IDENTICAL to `exp6_alpha_curves.py::_run_single`
(same estimate_eps_D/alpha_weights/compute_Z_from_W/init_pca/smacof_solve
call sequence, same evaluate(..., extended=True)) so that E6 and E12 rows
of the same (dataset, alpha, seed) family can be concatenated into a single
alpha curve. Crucially, the FIT hyperparameters (seeds, n_max, max_iter,
tol, eps_D (k, q), n_components, device) are ALWAYS read from the FULL
production `exp6_alpha_curves` config (`resolve_experiment_config("exp6_alpha_curves",
"full")`), regardless of exp12's OWN --quick/--smoke/--full mode - exactly
like `exp11_convergence_check.py` reads its E6/E10 source data from the FULL
run. exp12's own mode only restricts WHICH (dataset, alpha) combinations
from `exp12_alpha_grid_extension.datasets`/`.alpha_grid` are processed
(config_experiments.yaml `exp12_alpha_grid_extension.smoke`), so a row
computed in any mode is directly comparable to the corresponding E6 row -
changing exp12's mode must never change what is being measured, only how
much of it.

Two columns are added on top of the exp6_alpha_curves CSV schema (so the
schema check in `run_experiment_grid` accepts appending exp12 rows to the
same combined table used by downstream figures/tables):
  - weight_dynamic_range_log10 = alpha * log10((D_max+eps_D)/(D_min+eps_D))
    (D_max/D_min = the largest/smallest OFF-DIAGONAL entry of the distance
    matrix D actually used for the fit) - the empirical log10 dynamic range
    of the weight matrix W = (D+eps_D)^{-alpha} for this specific
    (dataset, alpha, seed).
  - numerically_reliable = weight_dynamic_range_log10 <= reliability_log10_max
    (threshold from config, NOT dropped silently - rows above the threshold
    are still written to the CSV, just flagged, so nothing is fabricated
    or hidden by a fallback).

Output: results/data/[<mode>/]exp12_alpha_grid_extension_results.csv
(checkpoint, resumable) + results/data/[<mode>/]exp12_alpha_grid_extension_DONE.txt.
Log: results/logs/[<mode>/]exp12_alpha_grid_extension.log.

Run: venv\\python.exe -m src.experiments.exp12_alpha_grid_extension [--quick|--full|--smoke]
or: src\\run_exp12_alpha_grid_extension.bat [quick|full|smoke]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.common.checkpoint import results_csv_path
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

BASE_EXPERIMENT_NAME = "exp12_alpha_grid_extension"
EXP6_BASE_NAME = "exp6_alpha_curves"

# Columns read from the (much larger) exp6_alpha_curves results CSV for the
# closing "combined optimum" log summary - kept narrow on purpose (project
# rule: never load large CSVs in full when only a few columns are needed).
_AUC_SUMMARY_USECOLS = ["dataset", "alpha", "seed", "status", "auc_rnx"]


def _method_name_for_alpha(alpha: float) -> str:
    """Same 'method' naming convention as exp6_alpha_curves.py (f"alpha{alpha}")
    - REQUIRED for E6/E12 rows of the same (dataset, alpha, seed) to line up
    when concatenated into one alpha curve."""
    return f"alpha{alpha}"


def build_column_keys(n_components: int) -> tuple[list[str], dict[str, Any]]:
    """The exp6_alpha_curves CSV schema (dynamically discovered metric keys
    plus 'alpha'/'n_iter_smacof', see exp6_alpha_curves.py::main()) plus the
    two exp12-only columns described in the module docstring. Returns
    (column_keys, eval_kwargs) - eval_kwargs is passed straight to
    `evaluate()` (contains {'extended': True} whenever `evaluate` supports it,
    same as E6)."""
    metric_keys, eval_kwargs = discover_metric_keys(n_components=n_components)
    column_keys = metric_keys + ["alpha", "n_iter_smacof", "weight_dynamic_range_log10", "numerically_reliable"]
    return column_keys, eval_kwargs


def compute_weight_dynamic_range_log10(D: np.ndarray, alpha: float, eps_D: float) -> float:
    """weight_dynamic_range_log10 = alpha * log10((D_max+eps_D)/(D_min+eps_D))
    (documentation/2026-09-17_zadani_exp12_prodlouzeni_mrizky.md, section
    "Co ma experiment udelat") - D_max/D_min are the largest/smallest
    OFF-DIAGONAL entries of D (the diagonal is always 0 and must not enter
    either extremum)."""
    D = np.asarray(D, dtype=np.float64)
    n = D.shape[0]
    if D.ndim != 2 or D.shape != (n, n):
        raise ValueError(f"D must be a square (n x n) distance matrix, got shape {D.shape}.")
    if eps_D <= 0:
        raise ValueError(f"eps_D must be positive, got {eps_D}.")
    mask = ~np.eye(n, dtype=bool)
    D_off = D[mask]
    d_max = float(D_off.max())
    d_min = float(D_off.min())
    ratio = (d_max + eps_D) / (d_min + eps_D)
    return float(alpha) * float(np.log10(ratio))


def is_numerically_reliable(weight_dynamic_range_log10: float, reliability_log10_max: float) -> bool:
    """numerically_reliable = weight_dynamic_range_log10 <= reliability_log10_max
    (config `exp12_alpha_grid_extension.reliability_log10_max`) - a flag, NOT
    a filter: unreliable rows are still written to the CSV (fail-loud
    convention - nothing is silently dropped)."""
    return bool(weight_dynamic_range_log10 <= reliability_log10_max)


def _run_single(task: dict[str, Any]) -> dict[str, Any]:
    """One run: (dataset, alpha, seed) -> one SMACOF fit + evaluate(extended=True),
    exactly the same call sequence as exp6_alpha_curves.py::_run_single, plus
    the two numerical-reliability columns."""
    from src.common.seeding import set_seed
    from src.datasets.registry import load_dataset
    from src.datasets.subsample import subsample_dataset
    from src.methods.common import to_distance_matrix
    from src.sammon.init import init_pca
    from src.sammon.metrics import evaluate
    from src.sammon.solvers.smacof import resolve_gpu_dtype, smacof_solve, smacof_solve_gpu
    from src.sammon.weights import alpha_weights, compute_Z_from_W, estimate_eps_D

    dataset_name = task["dataset_name"]
    alpha = task["alpha"]
    seed = task["seed"]
    method_name = task["method_name"]
    scfg = task["sammon_cfg"]

    result: dict[str, Any] = {"dataset_name": dataset_name, "method_name": method_name, "seed": seed}
    t0 = time.perf_counter()
    try:
        set_seed(seed)
        ds = load_dataset(dataset_name)
        ds = subsample_dataset(ds, n_max=task["n_max"], random_state=task["subsample_seed"])
        X = np.asarray(ds.X, dtype=np.float64)
        D = to_distance_matrix(X, "vector")

        eps_D = estimate_eps_D(D, k=task["eps_D_k"], q=task["eps_D_q"], kind="distance")
        W = alpha_weights(D, alpha, eps_D)
        Z = compute_Z_from_W(D, W)
        Y0 = init_pca(X, task["n_components"], seed)

        if task["device"] == "cuda":
            gd_cfg = scfg["gpu_dense"]
            Y, history = smacof_solve_gpu(
                D, W, Z, Y0, max_iter=task["max_iter"], tol=task["tol"], eps_num=scfg["eps_num"],
                tile_rows=gd_cfg["tile_rows"], cg_max_iter=gd_cfg["cg_max_iter"], cg_tol=gd_cfg["cg_tol"],
                reg_rho=gd_cfg["reg_rho"], pinv_max_bytes=gd_cfg["pinv_max_bytes"],
                resident_max_bytes=gd_cfg["resident_max_bytes"], const_w_rtol=gd_cfg["const_w_rtol"],
                inexact_cg=gd_cfg["inexact_cg"], cg_tol_factor=gd_cfg["cg_tol_factor"],
                cg_tol_max=gd_cfg["cg_tol_max"], cg_check_every=gd_cfg["cg_check_every"],
                stress_blowup_factor=gd_cfg["stress_blowup_factor"],
                dtype=resolve_gpu_dtype(gd_cfg["dtype"]), device="cuda", verbose=False,
            )
        else:
            Y, history = smacof_solve(
                D, W, Z, Y0, max_iter=task["max_iter"], tol=task["tol"], eps_num=scfg["eps_num"],
                dense_pinv_threshold=scfg["smacof"]["dense_pinv_threshold"], cg_max_iter=scfg["smacof"]["cg_max_iter"],
                cg_tol=scfg["smacof"]["cg_tol"], reg_rho=scfg["smacof"]["reg_rho"], verbose=False,
            )

        metrics = evaluate(D, Y, ds.y, "distance", **task["eval_kwargs"])

        weight_dynamic_range_log10 = compute_weight_dynamic_range_log10(D, alpha, eps_D)
        numerically_reliable = is_numerically_reliable(weight_dynamic_range_log10, task["reliability_log10_max"])

        result["status"] = "ok"
        result["error"] = ""
        result["wall_time_sec"] = time.perf_counter() - t0
        result["embedding"] = Y
        result["metrics"] = metrics
        result["extra"] = {
            "alpha": alpha, "n_iter_smacof": history["n_iter"],
            "weight_dynamic_range_log10": weight_dynamic_range_log10,
            "numerically_reliable": numerically_reliable,
        }
    except Exception as exc:  # one incompatible combination must not stop the whole run
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["wall_time_sec"] = time.perf_counter() - t0
        result["embedding"] = None
        result["metrics"] = {}
        result["extra"] = {
            "alpha": alpha, "n_iter_smacof": np.nan,
            "weight_dynamic_range_log10": np.nan, "numerically_reliable": np.nan,
        }
    return result


def _log_combined_optimum_summary(
    exp12_experiment_name: str, exp6_full_experiment_name: str, datasets: list[str],
    extended_alpha_cap: float, logger,
) -> None:
    """Logs, for every dataset in `datasets`, the argmax-median-auc_rnx alpha
    over the COMBINED E6+E12 grid, and flags whether it again sits on the
    (extended) grid edge alpha=`extended_alpha_cap` - the diagnostic
    requested by the zadani document ("Co ma experiment udelat", last
    paragraph). Reads both CSVs with a narrow `usecols` (project rule: never
    load a large results CSV in full for a summary that only needs 5 columns)."""
    exp6_path = results_csv_path(exp6_full_experiment_name)
    exp12_path = results_csv_path(exp12_experiment_name)
    if not exp6_path.exists() or not exp12_path.exists():
        logger.warning("Combined optimum summary skipped - missing %s or %s.", exp6_path, exp12_path)
        return

    df6 = pd.read_csv(exp6_path, usecols=_AUC_SUMMARY_USECOLS)
    df12 = pd.read_csv(exp12_path, usecols=_AUC_SUMMARY_USECOLS)
    combined = pd.concat([df6, df12], ignore_index=True)
    combined = combined[(combined["status"] == "ok") & combined["dataset"].isin(datasets) & combined["auc_rnx"].notna()]
    if combined.empty:
        logger.warning("Combined optimum summary: no successful rows for datasets=%s.", datasets)
        return

    med = combined.groupby(["dataset", "alpha"])["auc_rnx"].median()
    logger.info("Combined E6+E12 optimum (median auc_rnx over seeds), extended grid cap alpha=%.2f:", extended_alpha_cap)
    for dataset_name in datasets:
        if dataset_name not in med.index.get_level_values(0):
            logger.warning("  %-24s: no successful combined rows.", dataset_name)
            continue
        sub = med.loc[dataset_name].sort_index()
        alpha_star = float(sub.idxmax())
        auc_star = float(sub.loc[alpha_star])
        at_edge = " -- STILL AT THE GRID EDGE" if alpha_star >= extended_alpha_cap else ""
        logger.info("  %-24s: alpha*=%.2f (auc_rnx=%.4f)%s", dataset_name, alpha_star, auc_star, at_edge)


def main() -> None:
    mode = parse_mode_args(
        "E12: extension of the exp6_alpha_curves alpha grid from (3, 6] on the datasets whose "
        "E6 optimum sits at/near alpha=3 (documentation/2026-09-17_zadani_exp12_prodlouzeni_mrizky.md)."
    )
    logger = get_logger(BASE_EXPERIMENT_NAME, mode=mode)

    cfg = resolve_experiment_config(BASE_EXPERIMENT_NAME, mode)

    # Fit hyperparameters (seeds, n_max, max_iter, tol, eps_D (k, q),
    # n_components, device) ALWAYS come from the FULL production
    # exp6_alpha_curves config, regardless of exp12's OWN mode - see the
    # module docstring and exp11_convergence_check.py for the same
    # mechanism. exp12's own mode only restricts 'datasets'/'alpha_grid'.
    exp6_cfg_full = resolve_experiment_config(EXP6_BASE_NAME, "full")
    sammon_cfg = load_config()["sammon"]

    from src.experiments.exp1_dr_benchmark import SUBSAMPLE_SEED
    from src.experiments.exp8_prop2_check import _resolve_eps_D_kq
    from src.sammon.device import resolve_device

    eps_D_k, eps_D_q = _resolve_eps_D_kq(exp6_cfg_full, sammon_cfg)
    n_components = int(exp6_cfg_full["n_components"])
    n_max = int(exp6_cfg_full["n_max"])
    max_iter = int(exp6_cfg_full["max_iter"])
    tol = float(exp6_cfg_full["tol"])
    seeds_list = list(exp6_cfg_full["seeds"])

    device_resolved = resolve_device(exp6_cfg_full.get("device", "cpu"))
    if device_resolved == "cuda":
        logger.info(
            "device resolved to 'cuda' (exp6_alpha_curves.device, read from the FULL config) - the "
            "run will be SEQUENTIAL (sequential=True) to avoid concurrent CUDA contexts from multiple "
            "ProcessPoolExecutor workers (same mechanism as exp6_alpha_curves.py/exp2_solver_scaling.py)."
        )

    reliability_log10_max = float(cfg["reliability_log10_max"])
    column_keys, eval_kwargs = build_column_keys(n_components)

    exp6_experiment_name = resolve_experiment_name(EXP6_BASE_NAME, "full")
    EXPERIMENT_NAME = resolve_experiment_name(BASE_EXPERIMENT_NAME, mode)

    datasets: list[str] = list(cfg["datasets"])
    alpha_grid: list[float] = [float(a) for a in cfg["alpha_grid"]]

    logger.info(
        "%s (mode=%s): %d datasets x %d alpha levels x %d seeds, fit config from FULL %s "
        "(n_max=%d, max_iter=%d, tol=%.1e, eps_D k=%d q=%.3f, device=%s), reliability_log10_max=%.1f.",
        BASE_EXPERIMENT_NAME, mode, len(datasets), len(alpha_grid), len(seeds_list), exp6_experiment_name,
        n_max, max_iter, tol, eps_D_k, eps_D_q, device_resolved, reliability_log10_max,
    )

    tasks: list[dict[str, Any]] = []
    for dataset_name in datasets:
        for alpha in alpha_grid:
            method_name = _method_name_for_alpha(alpha)
            for seed in seeds_list:
                tasks.append({
                    "dataset_name": dataset_name, "method_name": method_name, "seed": seed,
                    "alpha": alpha, "n_components": n_components, "n_max": n_max,
                    "subsample_seed": SUBSAMPLE_SEED, "eps_D_k": eps_D_k, "eps_D_q": eps_D_q,
                    "max_iter": max_iter, "tol": tol, "device": device_resolved,
                    "sammon_cfg": sammon_cfg, "eval_kwargs": eval_kwargs,
                    "reliability_log10_max": reliability_log10_max,
                })

    todo, n_done = filter_already_done(EXPERIMENT_NAME, tasks)
    logger.info("%s (mode=%s): %d combinations already done (skipped), %d new to compute.", EXPERIMENT_NAME, mode, n_done, len(todo))

    n_ok, n_err = run_experiment_grid(
        EXPERIMENT_NAME, todo, _run_single, column_keys, logger, sequential=(device_resolved == "cuda"),
    )
    logger.info("Done. Newly successful: %d, newly failed: %d, skipped: %d.", n_ok, n_err, n_done)
    write_done_file(EXPERIMENT_NAME, logger, extra_info={"mode": mode, "device": device_resolved, "exp6_source": exp6_experiment_name})

    # The extended grid's numerical cap is always read from the FULL exp12
    # config (6.0) - even in --quick/--smoke mode, where alpha_grid itself is
    # a smaller subset - so the "still at the edge" diagnostic always refers
    # to the true intended cap, not to whatever subset a given mode ran.
    exp12_cfg_full = resolve_experiment_config(BASE_EXPERIMENT_NAME, "full")
    extended_alpha_cap = max(float(a) for a in exp12_cfg_full["alpha_grid"])
    _log_combined_optimum_summary(EXPERIMENT_NAME, exp6_experiment_name, datasets, extended_alpha_cap, logger)


if __name__ == "__main__":
    main()
