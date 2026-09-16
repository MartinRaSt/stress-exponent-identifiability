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
K13 (documentation/2026-09-12_plan_smeru_clanku.md, supplement S4) - exp7:
neighbor-RANK weights (rank-weighted stress) vs. distance-based weights
(alpha-Sammon, `w_ij = D_ij^-alpha`), on 8 datasets (n_max=1000). A rewrite
of the one-off exploration
`reserse/skripty_20260912_myslitel/pilot_rank_weights.py`
(negative result: rank-weighted stress does not improve AUC_RNX/stress
compared to alpha-Sammon and is slower due to order statistics) into a
reproducible experiment with checkpoint/resume, config (no magic numbers in
code), and a DONE marker - see the `exp7_rank_weights` section in
`src/experiments/config_experiments.yaml`.

Weight schemes:
    dist<alpha>  - `src.sammon.weights.alpha_weights(D, alpha, eps_D)` (same
                   definition as sammon_alpha*_smacof)
    rank<beta>   - `rank_weights(D, beta)` = ((r_ij + r_ji)/2)^-beta, where
                   r_ij is the rank of point j among the neighbors of point i (1 = closest)
    tsne         - baseline from the method registry (`src.methods.registry`),
                   for comparison with a non-stress neighbor-based method

Output: results/data/[<mode>/]exp7_rank_weights_results.csv (checkpoint,
resumable, DONE file) + results/tables/exp7_rank_weights_summary.csv/.tex
(median+-IQR over seeds, then median over datasets, per method).

Run: venv\\python.exe -m src.experiments.exp7_rank_weights [--quick|--full|--smoke]
or: src\\run_exp7_rank_weights.bat [quick|full|smoke]
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
from src.common.config import ensure_dir, get_path, get_tables_dir, load_config
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

BASE_EXPERIMENT_NAME = "exp7_rank_weights"


def rank_weights(D: np.ndarray, beta: float) -> np.ndarray:
    """Neighbor-rank weights: w_ij = ((r_ij + r_ji)/2)^-beta, where r_ij is
    the rank of point j among the neighbors of point i (1 = closest, n-1 =
    farthest; the point itself has rank 0, clamped to >=1 so the power is
    finite). Symmetrization (r_ij+r_ji)/2, because SMACOF requires a
    symmetric W. Zero diagonal (same convention as `alpha_weights`)."""
    n = D.shape[0]
    if D.shape != (n, n):
        raise ValueError(f"D must be a square matrix, got shape {D.shape}.")
    order = np.argsort(D, axis=1)
    rank = np.empty_like(D)
    rows = np.arange(n)[:, None]
    rank[rows, order] = np.arange(n)[None, :]  # 0 = itself, 1 = nearest neighbor
    rank = np.maximum(rank, 1.0)
    rank_sym = 0.5 * (rank + rank.T)
    W = rank_sym ** (-float(beta))
    np.fill_diagonal(W, 0.0)
    return W


def _method_name(kind: str, param: float | None) -> str:
    if kind == "tsne":
        return "tsne"
    return f"{kind}{param:g}"


def _run_single(task: dict[str, Any]) -> dict[str, Any]:
    """One run: (dataset, weight scheme, seed) -> one fit + evaluate(extended=True)."""
    from src.common.seeding import set_seed
    from src.datasets.registry import load_dataset
    from src.datasets.subsample import subsample_dataset
    from src.methods.common import to_distance_matrix
    from src.methods.registry import get_method
    from src.sammon.init import init_pca
    from src.sammon.metrics import evaluate
    from src.sammon.solvers.smacof import smacof_solve
    from src.sammon.weights import alpha_weights, compute_Z_from_W, estimate_eps_D

    dataset_name = task["dataset_name"]
    kind = task["kind"]
    param = task["param"]
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

        if kind == "tsne":
            Y = get_method("tsne").fit_transform(X, "vector", seed, task["n_components"])
        else:
            D = to_distance_matrix(X, "vector")
            if kind == "dist":
                eps_D = estimate_eps_D(D, k=scfg["eps_D"]["k"], q=scfg["eps_D"]["q"], kind="distance")
                W = alpha_weights(D, param, eps_D)
            elif kind == "rank":
                W = rank_weights(D, param)
            else:
                raise ValueError(f"Unknown 'kind'='{kind}' (expected 'dist'/'rank'/'tsne').")
            Z = compute_Z_from_W(D, W)
            Y0 = init_pca(X, task["n_components"], seed)
            Y, _history = smacof_solve(
                D, W, Z, Y0, max_iter=task["max_iter"], tol=task["tol"], eps_num=scfg["eps_num"],
                dense_pinv_threshold=scfg["smacof"]["dense_pinv_threshold"], cg_max_iter=scfg["smacof"]["cg_max_iter"],
                cg_tol=scfg["smacof"]["cg_tol"], reg_rho=scfg["smacof"]["reg_rho"], verbose=False,
            )

        D_eval = to_distance_matrix(X, "vector")
        metrics = evaluate(D_eval, Y, ds.y, "distance", **task["eval_kwargs"])

        result["status"] = "ok"
        result["error"] = ""
        result["wall_time_sec"] = time.perf_counter() - t0
        result["embedding"] = Y
        result["metrics"] = metrics
        result["extra"] = {"weight_kind": kind, "weight_param": param if param is not None else np.nan}
    except Exception as exc:  # one incompatible combination must not stop the whole run
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["wall_time_sec"] = time.perf_counter() - t0
        result["embedding"] = None
        result["metrics"] = {}
        result["extra"] = {"weight_kind": kind, "weight_param": param if param is not None else np.nan}
    return result


def _write_summary_table(experiment_name: str, mode: str, logger) -> Path | None:
    """results/tables/[<mode>/]exp7_rank_weights_summary.csv/.tex: median+-IQR over
    seeds, then median over datasets, per method (weight_kind+weight_param)."""
    csv_path = results_csv_path(experiment_name)
    if not csv_path.exists():
        logger.warning("exp7_rank_weights_summary: %s does not exist, skipping.", csv_path)
        return None
    df = pd.read_csv(csv_path)
    ok = df[df["status"] == "ok"]
    if ok.empty:
        logger.warning("exp7_rank_weights_summary: no successful rows, skipping.")
        return None

    metrics = ["auc_rnx", "trustworthiness_k7", "stress_scale_invariant", "q_global", "shepard_spearman_rho", "wall_time_sec"]
    metrics = [m for m in metrics if m in ok.columns]
    by_ds_method = ok.groupby(["dataset", "method"])[metrics].median()
    summary = by_ds_method.groupby("method")[metrics].median().reset_index()
    summary = summary.rename(columns={m: f"{m}_median" for m in metrics})
    summary = summary.sort_values("auc_rnx_median", ascending=False) if "auc_rnx_median" in summary.columns else summary

    tables_dir = get_tables_dir(mode)
    out_csv = tables_dir / "exp7_rank_weights_summary.csv"
    summary.to_csv(out_csv, index=False)

    cols = list(summary.columns)
    lines = [
        "% auto-generated by src/experiments/exp7_rank_weights.py - do not edit by hand",
        "\\begin{table}[htbp]", "\\centering",
        "\\caption{Rank-weighted vs. distance-weighted (alpha-Sammon) stress - median over seeds then datasets (negative supplement result)}",
        "\\label{tab:exp7_rank_weights_summary}",
        f"\\begin{{tabular}}{{{'l' * len(cols)}}}", "\\toprule", " & ".join(cols) + " \\\\", "\\midrule",
    ]
    for _, row in summary.iterrows():
        cells = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (tables_dir / "exp7_rank_weights_summary.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    logger.info("Written: %s (%d methods).", out_csv, summary.shape[0])
    return out_csv


def main() -> None:
    mode = parse_mode_args("E7 (supplement S4): rank-weighted stress vs. alpha-Sammon distance-based weights.")
    logger = get_logger(BASE_EXPERIMENT_NAME, mode=mode)
    cfg = resolve_experiment_config(BASE_EXPERIMENT_NAME, mode)
    EXPERIMENT_NAME = resolve_experiment_name(BASE_EXPERIMENT_NAME, mode)
    sammon_cfg = load_config()["sammon"]

    from src.experiments.exp1_dr_benchmark import SUBSAMPLE_SEED

    n_components = cfg["n_components"]
    metric_keys, eval_kwargs = discover_metric_keys(n_components=n_components)
    column_keys = metric_keys + ["weight_kind", "weight_param"]

    tasks: list[dict[str, Any]] = []
    for dataset_name in cfg["datasets"]:
        configs: list[tuple[str, float | None]] = [("dist", float(a)) for a in cfg["dist_alpha_grid"]]
        configs += [("rank", float(b)) for b in cfg["rank_beta_grid"]]
        configs += [(m, None) for m in cfg["baseline_methods"]]
        for kind, param in configs:
            method_name = _method_name(kind, param)
            for seed in cfg["seeds"]:
                tasks.append({
                    "dataset_name": dataset_name, "method_name": method_name, "seed": seed,
                    "kind": kind, "param": param, "n_components": n_components, "n_max": int(cfg["n_max"]),
                    "subsample_seed": SUBSAMPLE_SEED, "max_iter": int(cfg["max_iter"]), "tol": float(cfg["tol"]),
                    "sammon_cfg": sammon_cfg, "eval_kwargs": eval_kwargs,
                })

    todo, n_done = filter_already_done(EXPERIMENT_NAME, tasks)
    logger.info("%s (mode=%s): %d combinations already done (skipped), %d new to compute.", EXPERIMENT_NAME, mode, n_done, len(todo))

    n_ok, n_err = run_experiment_grid(EXPERIMENT_NAME, todo, _run_single, column_keys, logger)
    logger.info("Done. Newly successful: %d, newly failed: %d, skipped: %d.", n_ok, n_err, n_done)
    write_done_file(EXPERIMENT_NAME, logger, extra_info={"mode": mode})

    _write_summary_table(EXPERIMENT_NAME, mode, logger)


if __name__ == "__main__":
    main()
