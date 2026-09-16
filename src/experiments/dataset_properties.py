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
K1 (documentation/2026-09-12_plan_smeru_clanku.md) - properties of the input
datasets used in E1 (vector/synthetic, 32 datasets) and E3 (7 graphs x 2
distance metrics = 14 inputs), on EXACTLY the same subsample/standardization
as the respective experiment (see
`src.experiments.exp1_dr_benchmark.SUBSAMPLE_SEED` and
`src.experiments.exp3_graph_layout._build_distance_dataset` - both functions
are imported and reused directly here, so the definitions cannot drift apart).

Properties (motivation: reserse/2026-09-12_proc_sammon_a_smery_clanku.md
direction B - distance concentration predicts the optimal alpha of the
weighted stress):
  nn_ratio_k1/k5/k10 - median distance to the k-th neighbor / median of all
    pairwise distances (i<j)
  dist_cv       - coefficient of variation of all pairwise distances (std/mean)
  dist_skew     - skewness of the pairwise distance distribution (scipy.stats.skew)
  relative_contrast - Aggarwal et al. 2001 (DOI 10.1007/3-540-44503-X_27):
    median_i (max_j D_ij - min_j D_ij) / min_j D_ij
  id_twonn      - intrinsic dimension, Facco et al. 2017 (DOI 10.1038/s41598-017-11873-y):
    mu_i = r2_i/r1_i, linear regression (through the origin) y=-log(1-F(mu)) on
    x=log(mu) over the first `twonn_fraction` points ordered by mu
  id_mle        - intrinsic dimension, Levina & Bickel 2004 (NeurIPS, no DOI,
    https://papers.nips.cc/paper_files/paper/2004/hash/74934548253bcab8490ebd74afed7031-Abstract.html),
    k=`id_mle_k`, average of the local estimates over points
  silhouette_input - sklearn silhouette_score on D (precomputed), only if
    labels exist and >=2 classes, otherwise NaN (never fabricated)
  hubness_skew  - skewness of the distribution of how often a point occurs
    among the k=`hubness_k` nearest neighbors of other points (N_k
    occurrence, Radovanovic et al. 2010)

Exact definitions/justification of choices are provided by `sci-researcher`
(R2) - this script implements the formulas exactly per the author's spec
(see the commit message / the spec for this task); any deviations are
recorded in the documentation, not in the code.

Output:
  results/data/[<mode>/]dataset_properties_results.csv - a resumable
    checkpoint (one row = one dataset, same mechanism as other experiments -
    RunKey with method='properties', seed=0).
  results/data/[<mode>/]dataset_properties.csv - a clean final CSV (only
    successful rows, without internal checkpoint overhead) - INPUT for K2/K3/K4/K12a.

Run: venv\\python.exe -m src.experiments.dataset_properties [--quick|--full|--smoke] [--check-correlation]
or: src\\run_dataset_properties.bat [quick|full|smoke]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from scipy.stats import skew, spearmanr

from src.common.checkpoint import results_csv_path
from src.common.logging_utils import get_logger
from src.experiments.config_experiments import load_experiments_config, resolve_experiment_config
from src.experiments.exp_common import (
    add_dataset_scope_arg,
    add_mode_args,
    filter_already_done,
    resolve_dataset_scope,
    resolve_experiment_name,
    resolve_mode,
    run_experiment_grid,
    write_done_file,
)

BASE_EXPERIMENT_NAME = "dataset_properties"
METHOD_LABEL = "properties"  # dummy 'method' for the RunKey/checkpoint infrastructure (1 row = 1 dataset)
SEED_LABEL = 0

COLUMN_KEYS = [
    "kind", "n", "d", "n_classes",
    "nn_ratio_k1", "nn_ratio_k5", "nn_ratio_k10",
    "dist_cv", "dist_skew", "relative_contrast",
    "id_twonn", "id_mle", "silhouette_input", "hubness_skew",
    "distance_metric", "base_graph",
]

# final columns of the clean CSV (without the checkpoint overhead columns experiment/method/seed/status/error/wall_time_sec)
_FINAL_COLUMNS = ["dataset"] + COLUMN_KEYS


# ---------------------------------------------------------------------------
# Pure numerical functions (testable without a dataset - see tests/test_dataset_properties.py)
# ---------------------------------------------------------------------------

def nn_distances_from_D(D: np.ndarray, k: int) -> np.ndarray:
    """Distance to the k-th nearest neighbor for each point (D without the
    diagonal, 1-indexed - k=1 is the nearest neighbor)."""
    n = D.shape[0]
    if k < 1 or k > n - 1:
        raise ValueError(f"k={k} must be in the range [1, {n - 1}] for n={n}.")
    masked = np.array(D, dtype=np.float64, copy=True)
    np.fill_diagonal(masked, np.inf)
    sorted_d = np.sort(masked, axis=1)
    return sorted_d[:, k - 1]


def nn_ratio(D: np.ndarray, k: int, median_all: float) -> float:
    """Median distance to the k-th neighbor / median of all pairwise distances."""
    nnk = nn_distances_from_D(D, k)
    return float(np.median(nnk) / median_all)


def dist_cv(triu: np.ndarray) -> float:
    """Coefficient of variation (std/mean) of all pairwise distances (i<j)."""
    m = float(triu.mean())
    if m <= 0:
        raise ValueError("Mean distance is 0 - the coefficient of variation is undefined.")
    return float(triu.std() / m)


def dist_skew(triu: np.ndarray) -> float:
    """Skewness of the distribution of all pairwise distances (i<j)."""
    return float(skew(triu))


def relative_contrast(D: np.ndarray, eps: float) -> float:
    """Aggarwal, Hinneburg, Keim 2001 (DOI 10.1007/3-540-44503-X_27):
    median_i (max_j D_ij - min_j D_ij) / min_j D_ij (j != i).

    `eps` is a numerical floor for min_j D_ij (same principle as
    `metrics.eps` in src/sammon/metrics.py::sammon_stress - does not
    fabricate a value, only prevents division by exact zero for duplicate points)."""
    n = D.shape[0]
    masked = np.array(D, dtype=np.float64, copy=True)
    np.fill_diagonal(masked, np.nan)
    row_max = np.nanmax(masked, axis=1)
    row_min = np.nanmin(masked, axis=1)
    row_min_safe = np.maximum(row_min, eps)
    return float(np.median((row_max - row_min) / row_min_safe))


def id_twonn(D: np.ndarray, fraction: float) -> float:
    """Facco, Aletti, Amico, Rondoni, Laio 2017 (DOI 10.1038/s41598-017-11873-y):
    mu_i = r2_i/r1_i (ratio of the distance to the 2nd and 1st nearest
    neighbor), linear regression through the origin y=-log(1-F(mu)) on
    x=log(mu) over the first `fraction` points ordered by mu (F = empirical
    distribution function)."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0,1], got {fraction}.")
    r1 = nn_distances_from_D(D, 1)
    r2 = nn_distances_from_D(D, 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = r2 / np.maximum(r1, 1e-300)
    mu = np.sort(mu[np.isfinite(mu) & (mu > 1.0)])
    n = mu.shape[0]
    if n < 10:
        raise ValueError(f"Too few valid mu=r2/r1 (n={n}) for the TwoNN ID estimate (min. 10).")
    F = np.arange(1, n + 1) / n
    keep = max(2, int(round(fraction * n)))
    x = np.log(mu[:keep])
    y = -np.log(1.0 - np.minimum(F[:keep], 1.0 - 1e-12))
    denom = float((x * x).sum())
    if denom <= 0:
        raise ValueError("Degenerate regression for the TwoNN ID (sum x^2 = 0).")
    return float((x * y).sum() / denom)


def id_mle(D: np.ndarray, k: int) -> float:
    """Levina & Bickel 2004: local estimate
    m_k(i) = [(1/(k-1)) * sum_{j=1}^{k-1} log(r_k(i)/r_j(i))]^-1 for each point,
    global estimate = average over points (per the K1 task spec)."""
    n = D.shape[0]
    if k < 2 or k > n - 1:
        raise ValueError(f"k={k} must be in the range [2, {n - 1}] for n={n}.")
    masked = np.array(D, dtype=np.float64, copy=True)
    np.fill_diagonal(masked, np.inf)
    sorted_d = np.sort(masked, axis=1)[:, :k]  # r_1..r_k for each point
    r_k = sorted_d[:, k - 1 : k]
    with np.errstate(divide="ignore"):
        log_ratios = np.log(r_k / sorted_d[:, : k - 1])
    mean_log = log_ratios.mean(axis=1)
    valid = mean_log > 0
    if not np.any(valid):
        raise ValueError("All local MLE estimates of the intrinsic dimension are degenerate (<=0).")
    m_k = 1.0 / mean_log[valid]
    return float(m_k.mean())


def hubness_skew(D: np.ndarray, k: int) -> float:
    """Skewness of the distribution of N_k(j) - the number of times point j
    occurs among the k nearest neighbors of other points (Radovanovic,
    Nanopoulos, Ivanovic 2010)."""
    n = D.shape[0]
    if k < 1 or k > n - 1:
        raise ValueError(f"k={k} must be in the range [1, {n - 1}] for n={n}.")
    masked = np.array(D, dtype=np.float64, copy=True)
    np.fill_diagonal(masked, np.inf)
    order = np.argsort(masked, axis=1)[:, :k]
    counts = np.bincount(order.ravel(), minlength=n)
    return float(skew(counts))


def silhouette_input(D: np.ndarray, y: np.ndarray | None) -> float:
    """Silhouette on D (precomputed) - only if labels exist, >=2 classes AND
    the number of classes < n (sklearn requires 2 <= n_labels <=
    n_samples-1, the same condition as `evaluate()` in
    src/sammon/metrics.py) - otherwise NaN, a substitute value is never
    fabricated. Some synthetic datasets have a continuous `y` (e.g. position
    on the roll for swiss_roll) - the high number of unique values then
    reliably exceeds n-1 and the metric is correctly NaN."""
    if y is None:
        return float("nan")
    y = np.asarray(y)
    n = D.shape[0]
    classes = np.unique(y)
    if not (2 <= len(classes) < n):
        return float("nan")
    from sklearn.metrics import silhouette_score

    return float(silhouette_score(D, y, metric="precomputed"))


def _safe(fn, *args, **kwargs) -> float:
    """Computes a single property, and on an exception (the metric is
    undefined for the given dataset - e.g. TwoNN requires >=10
    non-degenerate mu=r2/r1, which often fails to hold for small graphs with
    integer (shortest-path) distances due to ties) returns NaN INSTEAD of
    failing the whole row. This is NOT data fabrication - it is the same
    convention as `evaluate()` in src/sammon/metrics.py (metric undefined
    for the given n/k -> NaN), just applied per-property instead of
    per-whole-row, so that one undefined metric does not deprive the row of
    all other (valid) properties."""
    try:
        return fn(*args, **kwargs)
    except Exception:
        return float("nan")


def compute_dataset_properties(
    D: np.ndarray, X: np.ndarray | None, y: np.ndarray | None, kind: str,
    nn_ratio_ks: list[int], id_mle_k: int, hubness_k: int, twonn_fraction: float,
    eps: float,
) -> dict[str, Any]:
    """Computes all dataset properties from the distance matrix D (+
    optional X for the ambient dimension d). Returns a dict with all keys
    from COLUMN_KEYS except 'distance_metric'/'base_graph' (those are added
    by the caller for graph inputs).

    Median of all pairwise distances == 0 is the only FATAL precondition
    (without it neither nn_ratio nor the normalization of other properties
    is defined) - raises an exception for the whole row. All other
    properties are computed independently (see `_safe`) - one being
    undefined (e.g. TwoNN ID on a small discrete graph) does not cause the
    loss of the others."""
    n = D.shape[0]
    triu = D[np.triu_indices(n, k=1)]
    median_all = float(np.median(triu))
    if median_all <= 0:
        raise ValueError("Median of all pairwise distances is 0 - the dataset is degenerate.")

    row: dict[str, Any] = {
        "kind": kind,
        "n": n,
        "d": float(X.shape[1]) if X is not None else float("nan"),
        "n_classes": int(len(np.unique(y))) if y is not None else float("nan"),
        "dist_cv": _safe(dist_cv, triu),
        "dist_skew": _safe(dist_skew, triu),
        "relative_contrast": _safe(relative_contrast, D, eps),
        "id_twonn": _safe(id_twonn, D, twonn_fraction),
        "id_mle": _safe(id_mle, D, id_mle_k),
        "silhouette_input": _safe(silhouette_input, D, y),
        "hubness_skew": _safe(hubness_skew, D, hubness_k),
    }
    for k in nn_ratio_ks:
        row[f"nn_ratio_k{k}"] = _safe(nn_ratio, D, k, median_all)
    return row


# ---------------------------------------------------------------------------
# Data loading - EXACTLY the same subsample/standardization as E1/E3 (reuse
# their functions, no own copy of the subsampling logic)
# ---------------------------------------------------------------------------

def _run_single(task: dict[str, Any]) -> dict[str, Any]:
    """One row = one dataset (a vector one from E1, or a graph distance input from E3)."""
    from scipy.spatial.distance import pdist, squareform

    dataset_name = task["dataset_name"]
    prop_cfg = task["prop_cfg"]
    eps = task["eps"]

    result: dict[str, Any] = {"dataset_name": dataset_name, "method_name": METHOD_LABEL, "seed": SEED_LABEL}
    t0 = time.perf_counter()
    try:
        if task["task_kind"] == "vector_e1":
            from src.datasets.registry import load_dataset
            from src.datasets.subsample import subsample_dataset
            from src.experiments.exp1_dr_benchmark import SUBSAMPLE_SEED

            ds = load_dataset(task["base_name"])
            ds = subsample_dataset(ds, n_max=task["n_max"], random_state=SUBSAMPLE_SEED)
            X = np.asarray(ds.X, dtype=np.float64)
            D = squareform(pdist(X, metric="euclidean"))
            row = compute_dataset_properties(
                D, X, ds.y, "vector",
                prop_cfg["nn_ratio_ks"], prop_cfg["id_mle_k"], prop_cfg["hubness_k"], prop_cfg["twonn_fraction"], eps,
            )
            row["distance_metric"] = ""
            row["base_graph"] = ""
        else:  # graph_e3
            from src.experiments.exp3_graph_layout import _build_distance_dataset

            ds, _g_lcc, _n_original, _n_lcc = _build_distance_dataset(task["base_name"], task["distance_metric"])
            row = compute_dataset_properties(
                ds.D, None, ds.y, "graph_distance",
                prop_cfg["nn_ratio_ks"], prop_cfg["id_mle_k"], prop_cfg["hubness_k"], prop_cfg["twonn_fraction"], eps,
            )
            row["distance_metric"] = task["distance_metric"]
            row["base_graph"] = task["base_name"]

        result["status"] = "ok"
        result["error"] = ""
        result["wall_time_sec"] = time.perf_counter() - t0
        result["embedding"] = None
        result["metrics"] = row
        result["extra"] = {}
    except Exception as exc:  # intentionally broad - one dataset's failure must not stop the others
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["wall_time_sec"] = time.perf_counter() - t0
        result["embedding"] = None
        result["metrics"] = {}
        result["extra"] = {}
    return result


def _write_final_csv(experiment_name: str, logger) -> Path:
    """Reads the checkpoint CSV (dataset_properties_results.csv, only rows
    with status=='ok') and writes the clean output dataset_properties.csv
    (see K3/K4/K12a)."""
    csv_path = results_csv_path(experiment_name)
    df = pd.read_csv(csv_path)
    ok = df[df["status"] == "ok"].copy()
    ok = ok.rename(columns={"dataset": "dataset"})
    final = ok[_FINAL_COLUMNS].sort_values("dataset").reset_index(drop=True)
    out_path = csv_path.parent / "dataset_properties.csv"
    final.to_csv(out_path, index=False)
    logger.info("Final CSV written: %s (%d rows, of which %d failed omitted).", out_path, final.shape[0], (df["status"] != "ok").sum())
    return out_path


def _run_correlation_check(final_csv_path: Path, mode: str, logger) -> None:
    """Sanity-check Spearman correlation of nn_ratio_k1 vs. the best alpha
    from {0,1,2} (per median auc_rnx over seeds in E1) on the E1 datasets -
    see reserse/2026-09-12_proc_sammon_a_smery_clanku.md direction B,
    expected value rho~-0.73 (n=32). Requires an already completed
    exp1_dr_benchmark run (a production --full run, otherwise fail-loud)."""
    from src.experiments.exp_common import resolve_experiment_name as _resolve

    e1_path = results_csv_path(_resolve("exp1_dr_benchmark", mode))
    if not e1_path.exists():
        raise FileNotFoundError(
            f"--check-correlation requires {e1_path} (first run "
            f"src\\run_exp1_dr_benchmark.bat {mode})."
        )
    e1 = pd.read_csv(e1_path)
    e1_ok = e1[e1["status"] == "ok"]
    alpha_cols = ["sammon_alpha0_smacof", "sammon_alpha_smacof", "sammon_alpha2_smacof"]
    alpha_map = {"sammon_alpha0_smacof": 0.0, "sammon_alpha_smacof": 1.0, "sammon_alpha2_smacof": 2.0}
    med = e1_ok.groupby(["dataset", "method"])["auc_rnx"].median().unstack("method")
    missing = [c for c in alpha_cols if c not in med.columns]
    if missing:
        raise KeyError(f"exp1_dr_benchmark_results.csv is missing methods {missing} - cannot determine the best alpha from {{0,1,2}}.")
    sub = med[alpha_cols].dropna(how="any")
    best_alpha = sub.idxmax(axis=1).map(alpha_map)

    props = pd.read_csv(final_csv_path).set_index("dataset")
    joined = best_alpha.to_frame("best_alpha").join(props["nn_ratio_k1"], how="inner")
    if joined.shape[0] < 3:
        raise ValueError(f"Only {joined.shape[0]} datasets shared between E1 and dataset_properties.csv - cannot compute the correlation.")

    rho, pvalue = spearmanr(joined["nn_ratio_k1"], joined["best_alpha"])
    logger.info(
        "--check-correlation: Spearman(nn_ratio_k1, best_alpha_from_{0,1,2}) rho=%+.3f p=%.4g (n=%d) "
        "[expected per the review ~-0.73, n=32].", rho, pvalue, joined.shape[0],
    )
    print(f"Spearman(nn_ratio_k1, best alpha from {{0,1,2}}): rho={rho:+.3f}, p={pvalue:.4g}, n={joined.shape[0]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="K1: properties of the input datasets E1 (vector) + E3 (graph distance inputs).")
    add_mode_args(parser)
    add_dataset_scope_arg(parser)
    parser.add_argument("--check-correlation", action="store_true", help="after completion, computes and prints Spearman(nn_ratio_k1, best alpha from {0,1,2}) against exp1_dr_benchmark_results.csv")
    args = parser.parse_args()
    mode = resolve_mode(args)

    logger = get_logger(BASE_EXPERIMENT_NAME, mode=mode)
    EXPERIMENT_NAME = resolve_experiment_name(BASE_EXPERIMENT_NAME, mode)

    e1_cfg = resolve_experiment_config("exp1_dr_benchmark", mode)
    e3_cfg = resolve_experiment_config("exp3_graph_layout", mode)
    prop_cfg = load_experiments_config()["dataset_properties"]
    from src.common.config import load_config

    eps = float(load_config()["metrics"]["eps"])

    n_max_overrides: dict[str, int] = e1_cfg.get("n_max_overrides", {})
    # Q1 step 2 (A.5, A.9 item 6): --datasets all|core|holdout, see exp1_dr_benchmark.py.
    core_datasets: list[str] = list(e1_cfg["datasets"])
    holdout_datasets: list[str] = list(e1_cfg.get("datasets_holdout", []))
    scope_datasets = resolve_dataset_scope(args, core_datasets, holdout_datasets)
    logger.info("--datasets=%s -> %d vector datasets (core=%d, holdout=%d).", args.datasets, len(scope_datasets), len(core_datasets), len(holdout_datasets))

    tasks: list[dict[str, Any]] = []
    for dataset_name in scope_datasets:
        n_max = int(n_max_overrides.get(dataset_name, e1_cfg["n_max"]))
        tasks.append({
            "dataset_name": dataset_name, "method_name": METHOD_LABEL, "seed": SEED_LABEL,
            "task_kind": "vector_e1", "base_name": dataset_name, "n_max": n_max,
            "prop_cfg": prop_cfg, "eps": eps,
        })
    for base_name in e3_cfg["datasets"]:
        for distance_metric in e3_cfg["distance_metrics"]:
            label = f"{base_name}__{distance_metric}"
            tasks.append({
                "dataset_name": label, "method_name": METHOD_LABEL, "seed": SEED_LABEL,
                "task_kind": "graph_e3", "base_name": base_name, "distance_metric": distance_metric,
                "prop_cfg": prop_cfg, "eps": eps,
            })

    todo, n_done = filter_already_done(EXPERIMENT_NAME, tasks)
    logger.info("%s (mode=%s): %d datasets already done (skipped), %d new to compute.", EXPERIMENT_NAME, mode, n_done, len(todo))

    n_ok, n_err = run_experiment_grid(EXPERIMENT_NAME, todo, _run_single, COLUMN_KEYS, logger)
    logger.info("Done. Newly successful: %d, newly failed: %d, skipped: %d.", n_ok, n_err, n_done)
    write_done_file(EXPERIMENT_NAME, logger, extra_info={"mode": mode})

    final_path = _write_final_csv(EXPERIMENT_NAME, logger)

    if args.check_correlation:
        _run_correlation_check(final_path, mode, logger)


if __name__ == "__main__":
    main()
