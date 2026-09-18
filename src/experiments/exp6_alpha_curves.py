# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-12
# License: see the LICENSE file in the repository root
"""
K6 (documentation/2026-09-12_plan_smeru_clanku.md) - exp6: alpha curves
(AUC_RNX and stress as a function of the weighted-stress alpha) on a fine
grid alpha = {0, 0.25, ..., 3.0} across all 32 E1 datasets
(`exp1_dr_benchmark.datasets`, a YAML alias - see config_experiments.yaml,
NOT a copy of the list).

A NEW experiment (NOT an extension of E5 - a different schema: E5 is a
3-factor ablation of alpha x init x eps_D_q on 6 datasets, exp6 is a fine
alpha grid on ALL 32 datasets with fixed init='pca'/eps_D_q from
`sammon.eps_D.q`). Basis: reserse/2026-09-12_r1_r6_podklady_smeru_ab.md R2
(nn_ratio) + R6-B (the LOO protocol for K7).

The subsample (n_max=2000, seed=42) and standardization are EXACTLY the
same mechanism as E1 - `src.experiments.exp1_dr_benchmark.SUBSAMPLE_SEED` +
`src.datasets.subsample.subsample_dataset` are reused (imported, not
copied), standardization already happened inside `load_dataset` (see
`src/datasets/registry.py`, the `datasets.standardize` allow-list).

Solver: SMACOF (dense), device per `exp6_alpha_curves.device`
(config_experiments.yaml, DEFAULT 'cpu' - see the rationale directly at the
`resolve_device` call in `main()`: 'auto'/'cuda' on a machine with a GPU
would force a sequential run of ~12.8 h instead of a ~45-50 min parallel CPU
run, measured by extrapolating the production timings of
`sammon_alpha_smacof` in E1). Beware of GPU sharing among parallel workers
(ProcessPoolExecutor would open N CUDA contexts concurrently) - same
mechanism as `exp2_solver_scaling.py` part (b): if `device` resolves to
'cuda' (a manual choice in the config), the run is ALWAYS sequential
(`sequential=True`, one process, no GPU-sharing risk); for 'cpu' (default),
standard ProcessPoolExecutor parallelism runs (16 workers x 1 BLAS thread)
like E1/E5.

Output: results/data/[<mode>/]exp6_alpha_curves_results.csv (checkpoint,
resumable, DONE file) + summary tables
results/tables/[<mode>/]exp6_alpha_optimum.csv/.tex (per-dataset alpha*) and
results/tables/[<mode>/]exp6_factorial_summary.csv/.tex (the marginal effect
of alpha over all datasets - descriptive statistics/average rank, NO
Friedman chi2/p-test over "combinations" - see the `_write_factorial_summary` docstring).

Run: venv\\python.exe -m src.experiments.exp6_alpha_curves [--quick|--full|--smoke]
or: src\\run_exp6_alpha_curves.bat [quick|full|smoke]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.common.checkpoint import results_csv_path
from src.common.config import ensure_dir, get_path, get_tables_dir, load_config
from src.common.logging_utils import get_logger
from src.experiments.config_experiments import resolve_experiment_config
from src.experiments.exp_common import (
    add_dataset_scope_arg,
    add_mode_args,
    discover_metric_keys,
    filter_already_done,
    resolve_dataset_scope,
    resolve_experiment_name,
    resolve_mode,
    run_experiment_grid,
    write_done_file,
)
from src.experiments.report_tables import write_booktabs_tex

BASE_EXPERIMENT_NAME = "exp6_alpha_curves"


def _method_name_for_alpha(alpha: float) -> str:
    """The 'method' name for the RunKey/checkpoint - encoding alpha into the
    method column (same convention as E5: f"alpha{alpha}_init...", here just alpha)."""
    return f"alpha{alpha}"


def _run_single(task: dict[str, Any]) -> dict[str, Any]:
    """One run: (dataset, alpha, seed) -> one SMACOF fit + evaluate(extended=True)."""
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

        eps_D = estimate_eps_D(D, k=scfg["eps_D"]["k"], q=task["eps_D_q"], kind="distance")
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

        result["status"] = "ok"
        result["error"] = ""
        result["wall_time_sec"] = time.perf_counter() - t0
        result["embedding"] = Y
        result["metrics"] = metrics
        result["extra"] = {"alpha": alpha, "n_iter_smacof": history["n_iter"]}
    except Exception as exc:  # one incompatible combination must not stop the whole run
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["wall_time_sec"] = time.perf_counter() - t0
        result["embedding"] = None
        result["metrics"] = {}
        result["extra"] = {"alpha": alpha, "n_iter_smacof": np.nan}
    return result


def _write_alpha_optimum_table(experiment_name: str, mode: str, logger) -> Path | None:
    """`results/tables/exp6_alpha_optimum.csv/.tex`: per-dataset alpha*
    (argmax median auc_rnx over seeds), alpha* per the compromise (max
    auc_rnx with stress <= 1.05x stress(alpha=0)), the gain of alpha* vs.
    alpha=0 and alpha=1."""
    csv_path = results_csv_path(experiment_name)
    if not csv_path.exists():
        logger.warning("exp6_alpha_optimum: %s does not exist, skipping the summary table.", csv_path)
        return None
    df = pd.read_csv(csv_path)
    ok = df[(df["status"] == "ok") & df["auc_rnx"].notna() & df["stress_scale_invariant"].notna()]
    if ok.empty:
        logger.warning("exp6_alpha_optimum: no successful rows, skipping.")
        return None

    med = ok.groupby(["dataset", "alpha"])[["auc_rnx", "stress_scale_invariant"]].median()

    rows: list[dict[str, Any]] = []
    for dataset_name, sub in med.groupby(level=0):
        sub = sub.droplevel(0).sort_index()
        alpha_star_max = float(sub["auc_rnx"].idxmax())
        auc_at_max = float(sub.loc[alpha_star_max, "auc_rnx"])

        if 0.0 not in sub.index:
            continue  # alpha=0 must be in the grid (yardstick) - otherwise skip the dataset (fail-loud in the log below)
        stress0 = float(sub.loc[0.0, "stress_scale_invariant"])
        auc0 = float(sub.loc[0.0, "auc_rnx"])
        stress_limit = 1.05 * stress0
        feasible = sub[sub["stress_scale_invariant"] <= stress_limit]
        if feasible.empty:
            alpha_star_constrained = 0.0
            auc_at_constrained = auc0
        else:
            alpha_star_constrained = float(feasible["auc_rnx"].idxmax())
            auc_at_constrained = float(feasible.loc[alpha_star_constrained, "auc_rnx"])

        auc1 = float(sub.loc[1.0, "auc_rnx"]) if 1.0 in sub.index else np.nan

        rows.append({
            "dataset": dataset_name,
            "alpha_star_max": alpha_star_max, "auc_at_alpha_star_max": auc_at_max,
            "alpha_star_constrained": alpha_star_constrained, "auc_at_alpha_star_constrained": auc_at_constrained,
            "stress_at_alpha0": stress0, "auc_at_alpha0": auc0, "auc_at_alpha1": auc1,
            "gain_vs_alpha0": auc_at_max - auc0,
            "gain_vs_alpha1": (auc_at_max - auc1) if np.isfinite(auc1) else np.nan,
        })

    if not rows:
        logger.warning("exp6_alpha_optimum: no dataset has alpha=0 in the grid, the table is empty.")
        return None

    out = pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)
    tables_dir = get_tables_dir(mode)
    out_csv = tables_dir / "exp6_alpha_optimum.csv"
    out.to_csv(out_csv, index=False)

    write_booktabs_tex(
        out, tables_dir / "exp6_alpha_optimum.tex",
        caption="Per-dataset optimal alpha (argmax median AUC$_{RNX}$ over seeds) and stress-constrained variant",
        label="tab:exp6_alpha_optimum",
        comment_lines=["source: results/data/exp6_alpha_curves_results.csv, see _write_alpha_optimum_table"],
        # one row per dataset (grows with the dataset count) -> too tall for one
        # supplement page (2026-09-18 overflow fix, see write_booktabs_tex docstring).
        long_table=True,
    )

    logger.info("Written: %s (%d datasets).", out_csv, out.shape[0])
    return out_csv


def _write_factorial_summary(experiment_name: str, mode: str, logger) -> Path | None:
    """`results/tables/exp6_factorial_summary.csv/.tex`: the marginal (main)
    effect of alpha across all 32 datasets - median+IQR of auc_rnx/stress
    over datasets and the average rank of alpha (rank computed PER DATASET
    over the alpha grid, then averaged) - a DESCRIPTIVE statistic (analogous
    to the average ranks from a Friedman test), NO chi2/p-value/CD test.

    `src.experiments.stats.friedman_nemenyi` is deliberately NOT used with
    "method" = a combination of all factors as in E5 (117 combinations of
    alpha x init x eps_D_q in one opaque "method", see the WARNING in
    `results/data/stats_exp5_ablation_*.csv` for > 20 methods - the Nemenyi
    q_alpha table in config_experiments.yaml only goes up to k=20). exp6 has
    only 1 factor (alpha, 13 levels) - Friedman/Nemenyi would formally work
    here (k=13<=20), but per the K6 spec ("factorial statistics, NOT
    Friedman over combinations") only a descriptive summary
    (median/IQR/avg rank without a formal test) is deliberately used, so the
    result is not confused with a hypothesis test over unrelated "methods"
    (K12/K12c in S3 will unify the project convention)."""
    csv_path = results_csv_path(experiment_name)
    if not csv_path.exists():
        logger.warning("exp6_factorial_summary: %s does not exist, skipping.", csv_path)
        return None
    df = pd.read_csv(csv_path)
    ok = df[(df["status"] == "ok") & df["auc_rnx"].notna() & df["stress_scale_invariant"].notna()]
    if ok.empty:
        logger.warning("exp6_factorial_summary: no successful rows, skipping.")
        return None

    med = ok.groupby(["dataset", "alpha"])[["auc_rnx", "stress_scale_invariant"]].median()
    wide_auc = med["auc_rnx"].unstack("alpha")
    wide_stress = med["stress_scale_invariant"].unstack("alpha")
    complete = wide_auc.dropna(axis=0, how="any")
    n_dropped = wide_auc.shape[0] - complete.shape[0]

    alphas = sorted(wide_auc.columns.tolist())
    rows: list[dict[str, Any]] = []
    if not complete.empty:
        auc_ranks = rankdata(-complete.to_numpy(dtype=np.float64), axis=1, method="average")
        avg_rank_auc = dict(zip(complete.columns, auc_ranks.mean(axis=0)))
    else:
        avg_rank_auc = {}
    complete_stress = wide_stress.reindex(index=complete.index, columns=complete.columns) if not complete.empty else wide_stress.iloc[0:0]
    if not complete_stress.empty and complete_stress.notna().all(axis=None):
        stress_ranks = rankdata(complete_stress.to_numpy(dtype=np.float64), axis=1, method="average")
        avg_rank_stress = dict(zip(complete_stress.columns, stress_ranks.mean(axis=0)))
    else:
        avg_rank_stress = {}

    for alpha in alphas:
        auc_col = wide_auc[alpha].dropna()
        stress_col = wide_stress[alpha].dropna()
        rows.append({
            "alpha": alpha, "n_datasets": int(auc_col.shape[0]),
            "median_auc_rnx": float(auc_col.median()) if not auc_col.empty else np.nan,
            "iqr_auc_rnx": float(auc_col.quantile(0.75) - auc_col.quantile(0.25)) if not auc_col.empty else np.nan,
            "avg_rank_auc_rnx": float(avg_rank_auc.get(alpha, np.nan)),
            "median_stress": float(stress_col.median()) if not stress_col.empty else np.nan,
            "iqr_stress": float(stress_col.quantile(0.75) - stress_col.quantile(0.25)) if not stress_col.empty else np.nan,
            "avg_rank_stress": float(avg_rank_stress.get(alpha, np.nan)),
            "n_datasets_complete_block": int(complete.shape[0]), "n_datasets_dropped_incomplete": int(n_dropped),
        })

    out = pd.DataFrame(rows).sort_values("alpha").reset_index(drop=True)
    tables_dir = get_tables_dir(mode)
    out_csv = tables_dir / "exp6_factorial_summary.csv"
    out.to_csv(out_csv, index=False)

    cols = list(out.columns)
    lines = [
        "% auto-generated by src/experiments/exp6_alpha_curves.py - do not edit by hand",
        "\\begin{table}[htbp]", "\\centering",
        "\\caption{Marginal effect of alpha over all datasets (descriptive summary, not a Friedman test)}",
        "\\label{tab:exp6_factorial_summary}",
        f"\\begin{{tabular}}{{{'l' * len(cols)}}}", "\\toprule", " & ".join(cols) + " \\\\", "\\midrule",
    ]
    for _, row in out.iterrows():
        cells = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (tables_dir / "exp6_factorial_summary.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    logger.info("Written: %s (%d alpha levels, %d datasets in the complete block, %d dropped).", out_csv, out.shape[0], complete.shape[0], n_dropped)
    return out_csv


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="E6: alpha curves (AUC_RNX/stress vs. alpha) on a fine grid across all 32 E1 datasets (+ optional hold-out Q1 candidates).")
    add_mode_args(parser)
    add_dataset_scope_arg(parser)
    args = parser.parse_args()
    mode = resolve_mode(args)

    logger = get_logger(BASE_EXPERIMENT_NAME, mode=mode)
    cfg = resolve_experiment_config(BASE_EXPERIMENT_NAME, mode)
    EXPERIMENT_NAME = resolve_experiment_name(BASE_EXPERIMENT_NAME, mode)
    sammon_cfg = load_config()["sammon"]

    from src.experiments.exp1_dr_benchmark import SUBSAMPLE_SEED
    from src.sammon.device import resolve_device

    n_components = cfg["n_components"]
    metric_keys, eval_kwargs = discover_metric_keys(n_components=n_components)
    column_keys = metric_keys + ["alpha", "n_iter_smacof"]

    # device: DEFAULT 'cpu' (exp6_alpha_curves.device in
    # config_experiments.yaml), NOT directly sammon.device.default ('auto'
    # -> on this machine always 'cuda', an RTX 3070 Ti is detected). Reason
    # for deviating from the original "device from sammon.device" design:
    # measured on production E1 data (sammon_alpha_smacof, same
    # solver/max_iter/tol as exp6) - extrapolating to the 2080 exp6 runs
    # (n_max=2000) gives ~12.8 h of SEQUENTIAL GPU run (the E2 mechanism,
    # one process, GPU-sharing safe) vs. ~45-50 min of PARALLEL CPU run (16
    # workers x 1 BLAS thread, exactly the E1/E5 mechanism) - the latter
    # matches the planned budget "1-3 h (16 workers)" in
    # documentation/2026-09-12_plan_smeru_clanku.md. All existing
    # sammon_alpha* methods in common/config.yaml also pin device='cpu'
    # explicitly (NOT 'auto') for exactly this reason - exp6 follows the
    # same convention. Anyone who deliberately wants GPU (a smaller grid
    # than full, or a GPU faster on their machine) changes
    # 'exp6_alpha_curves.device: cuda' in config_experiments.yaml - the run
    # then automatically switches to sequential=True (same safety mechanism
    # as exp2_solver_scaling.py part (b)).
    device_resolved = resolve_device(cfg.get("device", "cpu"))
    if device_resolved == "cuda":
        logger.info(
            "device resolved to 'cuda' (exp6_alpha_curves.device) - the run will be SEQUENTIAL "
            "(sequential=True) to avoid concurrent CUDA contexts from multiple ProcessPoolExecutor "
            "workers (same mechanism as exp2_solver_scaling.py part (b))."
        )

    eps_D_q = cfg.get("eps_D_q", sammon_cfg["eps_D"]["q"])

    # Q1 step 2 (A.5, A.9 item 6): --datasets all|core|holdout, see exp1_dr_benchmark.py.
    core_datasets: list[str] = list(cfg["datasets"])
    holdout_datasets: list[str] = list(cfg.get("datasets_holdout", []))
    scope_datasets = resolve_dataset_scope(args, core_datasets, holdout_datasets)
    logger.info("--datasets=%s -> %d datasets (core=%d, holdout=%d).", args.datasets, len(scope_datasets), len(core_datasets), len(holdout_datasets))

    tasks: list[dict[str, Any]] = []
    for dataset_name in scope_datasets:
        for alpha in cfg["alpha_grid"]:
            method_name = _method_name_for_alpha(alpha)
            for seed in cfg["seeds"]:
                tasks.append({
                    "dataset_name": dataset_name, "method_name": method_name, "seed": seed,
                    "alpha": float(alpha), "n_components": n_components, "n_max": int(cfg["n_max"]),
                    "subsample_seed": SUBSAMPLE_SEED, "eps_D_q": float(eps_D_q),
                    "max_iter": int(cfg["max_iter"]), "tol": float(cfg["tol"]),
                    "device": device_resolved, "sammon_cfg": sammon_cfg, "eval_kwargs": eval_kwargs,
                })

    todo, n_done = filter_already_done(EXPERIMENT_NAME, tasks)
    logger.info("%s (mode=%s): %d combinations already done (skipped), %d new to compute.", EXPERIMENT_NAME, mode, n_done, len(todo))

    n_ok, n_err = run_experiment_grid(
        EXPERIMENT_NAME, todo, _run_single, column_keys, logger, sequential=(device_resolved == "cuda"),
    )
    logger.info("Done. Newly successful: %d, newly failed: %d, skipped: %d.", n_ok, n_err, n_done)
    write_done_file(EXPERIMENT_NAME, logger, extra_info={"mode": mode, "device": device_resolved})

    _write_alpha_optimum_table(EXPERIMENT_NAME, mode, logger)
    _write_factorial_summary(EXPERIMENT_NAME, mode, logger)


if __name__ == "__main__":
    main()
