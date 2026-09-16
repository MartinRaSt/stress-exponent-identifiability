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
Statistical protocol (reserse/2026-09-09_specifikace_metody.md section 9,
reserse/2026-09-09_pseudokod_a_metriky.md PART B): median/IQR over seeds,
method ranking over datasets, Friedman test (scipy) + Nemenyi post-hoc
(critical difference CD per Demsar 2006), Kendall's W as an effect size.

Input: a long CSV with rows (dataset, method, seed, status, <metrics...>),
e.g. results/data/exp1_dr_benchmark_results.csv. Output:
results/data/stats_<experiment>_<metric>.csv (one row per method: avg_rank,
median of the metric over datasets, CD, Kendall's W, Friedman chi2/p).

Used as a library (see src/main.py) and standalone:
venv\\python.exe -m src.experiments.stats [--quick|--full]
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, spearmanr, studentized_range

from src.common.checkpoint import results_csv_path
from src.common.config import ensure_dir
from src.common.logging_utils import get_logger
from src.experiments.config_experiments import load_experiments_config, resolve_experiment_config
from src.experiments.exp_common import parse_mode_args


def bootstrap_spearman_ci(
    x: np.ndarray, y: np.ndarray, n_boot: int, seed: int, ci_level: float = 0.95,
) -> dict[str, float]:
    """Spearman correlation between `x` and `y` (e.g. over datasets) +
    bootstrap 95% CI by resampling pairs (x_i, y_i) with replacement over
    `n_boot` repetitions (Q1 step 1, see config_experiments.yaml
    exp8_prop2_check.bootstrap - used by `fig_prop2_alpha_prediction.py` for
    alpha_bound vs. alpha_star). Deterministic (`np.random.default_rng(seed)`).
    NaN pairs are dropped BEFORE bootstrapping (same subsample for both the
    point estimate and the CI)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} and {y.shape}.")
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    n = x.shape[0]
    if n < 3:
        return {"rho": float("nan"), "pvalue": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n": int(n)}
    rho, pvalue = spearmanr(x, y)

    rng = np.random.default_rng(seed)
    boot_rhos = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        if np.all(xb == xb[0]) or np.all(yb == yb[0]):
            boot_rhos[b] = np.nan  # without variance the correlation is undefined - 0 is not fabricated
        else:
            boot_rhos[b] = spearmanr(xb, yb)[0]
    alpha_tail = (1.0 - ci_level) / 2.0
    finite_boot = boot_rhos[np.isfinite(boot_rhos)]
    if finite_boot.size == 0:
        ci_lo, ci_hi = float("nan"), float("nan")
    else:
        ci_lo = float(np.quantile(finite_boot, alpha_tail))
        ci_hi = float(np.quantile(finite_boot, 1.0 - alpha_tail))
    return {"rho": float(rho), "pvalue": float(pvalue), "ci_lo": ci_lo, "ci_hi": ci_hi, "n": int(n)}


def aggregate_median_iqr(df: pd.DataFrame, metric: str, group_cols: tuple[str, str] = ("dataset", "method")) -> pd.DataFrame:
    """Median and IQR of the metric over seeds, aggregated at the (dataset,
    method) level (section 9: 'report median and IQR', PART B step 2)."""
    ok = df[(df["status"] == "ok") & df[metric].notna()]
    g = ok.groupby(list(group_cols))[metric]
    out = g.median().rename("median").to_frame()
    out["iqr"] = g.quantile(0.75) - g.quantile(0.25)
    out["n_seeds"] = g.count()
    return out.reset_index()


def nemenyi_q_alpha(n_methods: int, alpha: float = 0.05) -> float:
    """Critical value q_alpha of the Nemenyi post-hoc test (studentized range
    statistic, Demsar 2006, JMLR 7:1-30, Table 5) computed DIRECTLY from the
    studentized range distribution, instead of the original manually
    transcribed table (which ends at k=20 and raised KeyError/WARNING for a
    larger number of methods - K14, see projectstate.md 2026-09-13 12:50
    "Nemenyi q_alpha for k=21 (E1) and k=117 (E5 combinations) is missing"):

        q_alpha = studentized_range.ppf(1 - alpha, k, df=inf) / sqrt(2)

    Agreement with the original table for k<=20 verified in
    `tests/test_stats.py` (tolerance 1e-3)."""
    if n_methods < 2:
        raise ValueError(f"nemenyi_q_alpha requires n_methods>=2, got {n_methods}.")
    return float(studentized_range.ppf(1.0 - alpha, n_methods, np.inf) / np.sqrt(2.0))


def nemenyi_cd(q_alpha: float, n_methods: int, n_datasets: int) -> float:
    """Critical difference of the Nemenyi post-hoc test
    (Demsar 2006, JMLR 7:1-30, eq. 6): CD = q_alpha * sqrt(k(k+1)/(6N))."""
    if n_methods < 2 or n_datasets < 1:
        raise ValueError(f"Nemenyi CD requires n_methods>=2 and n_datasets>=1, got k={n_methods}, N={n_datasets}.")
    return float(q_alpha * np.sqrt(n_methods * (n_methods + 1) / (6.0 * n_datasets)))


def friedman_nemenyi(
    median_wide: pd.DataFrame, direction: str, alpha: float = 0.05,
) -> dict[str, Any]:
    """Friedman test + Nemenyi CD over a wide table (rows=datasets,
    columns=methods, values=median of the metric over seeds). Returns a
    dict with average ranks (1=best), CD, Kendall's W, chi2/p-value.

    Blocks (datasets) with a missing value for any method are dropped
    (Friedman requires complete blocks) - the number dropped is returned in
    'n_datasets_dropped' for transparency (no silent value imputation)."""
    if direction not in ("max", "min"):
        raise ValueError(f"direction must be 'max' or 'min', got '{direction}'.")

    n_before = median_wide.shape[0]
    complete = median_wide.dropna(axis=0, how="any")
    n_dropped = n_before - complete.shape[0]
    n_datasets, n_methods = complete.shape
    if n_datasets < 2 or n_methods < 2:
        raise ValueError(
            f"After dropping incomplete blocks, n_datasets={n_datasets}, n_methods={n_methods} remained - "
            "the Friedman test requires at least 2 complete datasets and 2 methods."
        )

    values = complete.to_numpy(dtype=np.float64)
    rank_source = -values if direction == "max" else values
    ranks = rankdata(rank_source, axis=1, method="average")
    avg_ranks = ranks.mean(axis=0)

    chi2, pvalue = friedmanchisquare(*[values[:, j] for j in range(n_methods)])
    kendall_w = float(chi2 / (n_datasets * (n_methods - 1)))

    q_alpha = nemenyi_q_alpha(n_methods, alpha)
    cd = nemenyi_cd(q_alpha, n_methods, n_datasets)

    return {
        "methods": list(complete.columns), "avg_ranks": avg_ranks, "chi2": float(chi2), "pvalue": float(pvalue),
        "kendall_w": kendall_w, "cd": cd, "q_alpha": q_alpha, "n_datasets": n_datasets, "n_methods": n_methods,
        "n_datasets_dropped": n_dropped,
    }


def run_stats_for_experiment(
    base_experiment_name: str, metric: str, direction: str, logger,
    df: pd.DataFrame | None = None, output_suffix: str = "", mode: str = "full",
) -> pd.DataFrame | None:
    """Loads results/data/[<mode>/]<base_experiment_name>_results.csv (or
    uses an already pre-filtered `df`), computes the aggregation +
    Friedman/Nemenyi for the given primary metric, and writes
    results/data/[<mode>/]stats_<base_experiment_name><output_suffix>_<metric>.csv
    (the `results/data/` root is reserved for `--full`, see `resolve_experiment_name`).

    `output_suffix` (e.g. '_distance'/'_native') allows running several
    independent Friedman tests over different subsets of rows of the same
    experiment (see E3, where graph-native and distance-based methods do
    not share the same 'dataset' labels, so they cannot be compared in a
    single shared Friedman table - a complete block would not exist, see
    the `friedman_nemenyi` docstring).

    Returns the resulting table (or None if data is missing/insufficient -
    a warning is logged, no result is fabricated)."""
    from src.experiments.exp_common import resolve_experiment_name

    experiment_name = resolve_experiment_name(base_experiment_name, mode)
    if df is None:
        csv_path = results_csv_path(experiment_name)
        if not csv_path.exists():
            logger.warning("Stats %s/%s: %s does not exist, skipping.", experiment_name, metric, csv_path)
            return None
        df = pd.read_csv(csv_path)
    if metric not in df.columns:
        logger.warning("Stats %s/%s: column '%s' is not in the CSV, skipping.", experiment_name, metric, metric)
        return None

    agg = aggregate_median_iqr(df, metric)
    if agg.empty:
        logger.warning("Stats %s/%s: no 'ok' rows with a non-null metric, skipping.", experiment_name, metric)
        return None
    wide = agg.pivot_table(index="dataset", columns="method", values="median")

    stats_cfg = load_experiments_config()["stats"]
    alpha = float(stats_cfg["alpha"])

    try:
        fr = friedman_nemenyi(wide, direction, alpha)
    except (ValueError, KeyError) as exc:
        logger.warning("Stats %s/%s: %s - skipping Friedman/Nemenyi.", experiment_name, metric, exc)
        return None

    out_rows = []
    for method_name, avg_rank in zip(fr["methods"], fr["avg_ranks"]):
        med_iqr = agg[agg["method"] == method_name][["median", "iqr"]]
        out_rows.append({
            "experiment": experiment_name, "metric": metric, "direction": direction, "method": method_name,
            "avg_rank": avg_rank, "median_across_datasets": float(med_iqr["median"].median()),
            "iqr_across_datasets": float(med_iqr["iqr"].median()), "cd": fr["cd"], "q_alpha": fr["q_alpha"],
            "kendall_w": fr["kendall_w"], "friedman_chi2": fr["chi2"], "friedman_pvalue": fr["pvalue"],
            "n_datasets": fr["n_datasets"], "n_datasets_dropped": fr["n_datasets_dropped"], "n_methods": fr["n_methods"],
        })
    out_df = pd.DataFrame(out_rows).sort_values("avg_rank")

    # output directory derived from the same (already mode-overridden) name,
    # so it ends up in the correct subdirectory (root only for 'full', otherwise quick/smoke/)
    out_dir = results_csv_path(experiment_name).parent
    out_path = out_dir / f"stats_{base_experiment_name}{output_suffix}_{metric}.csv"
    ensure_dir(out_path.parent)
    out_df.to_csv(out_path, index=False)
    logger.info(
        "Stats saved: %s (k=%d methods, N=%d datasets, %d dropped, Friedman chi2=%.2f p=%.4g, Kendall W=%.3f, CD=%.3f).",
        out_path, fr["n_methods"], fr["n_datasets"], fr["n_datasets_dropped"], fr["chi2"], fr["pvalue"], fr["kendall_w"], fr["cd"],
    )
    return out_df


# experiments suitable for cross-dataset method comparison (dataset=block,
# method=treatment) - E2 (solver scaling/convergence) and E4 (temporal,
# unit=frame, has its own permutation test in exp4_temporal.py) do not
# belong here, see spec section 9
# E5 (117 alpha x init x eps_D_q combinations as "methods", k > 20 = beyond
# the Nemenyi q_alpha table) has NOT been tested with Friedman since K12c -
# the main effects are summarized by
# src/experiments/report_tables.py::write_exp5_factorial_summary
_STATS_EXPERIMENTS = ["exp1_dr_benchmark", "exp3_graph_layout"]


def main() -> None:
    mode = parse_mode_args("Statistical protocol: Friedman/Nemenyi over all primary metrics for E1/E3/E5.")
    logger = get_logger("stats", mode=mode)
    stats_cfg = load_experiments_config()["stats"]
    primary_metrics: dict[str, str] = stats_cfg["primary_metrics"]

    from src.experiments.exp_common import resolve_experiment_name

    for base_experiment_name in _STATS_EXPERIMENTS:
        resolved_name = resolve_experiment_name(base_experiment_name, mode)
        csv_path = results_csv_path(resolved_name)
        if not csv_path.exists():
            logger.warning("%s: %s does not exist, skipping the whole experiment.", resolved_name, csv_path)
            continue
        df = pd.read_csv(csv_path)

        if base_experiment_name == "exp3_graph_layout":
            # native graph layouts (kamada_kawai/spring, dataset label without
            # '__') and distance-based methods (mds/sammon/.../umap, dataset
            # label '<graph>__<distance_metric>') do not share the same 'dataset'
            # blocks -> two independent Friedman tests (see the run_stats_for_experiment docstring)
            df_native = df[~df["dataset"].str.contains("__", regex=False)]
            df_distance = df[df["dataset"].str.contains("__", regex=False)]
            for metric, direction in primary_metrics.items():
                run_stats_for_experiment(base_experiment_name, metric, direction, logger, df=df_distance, output_suffix="_distance", mode=mode)
                run_stats_for_experiment(base_experiment_name, metric, direction, logger, df=df_native, output_suffix="_native", mode=mode)
        else:
            for metric, direction in primary_metrics.items():
                run_stats_for_experiment(base_experiment_name, metric, direction, logger, df=df, mode=mode)


if __name__ == "__main__":
    main()
