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


def spearman_permutation_test(
    x: np.ndarray, y: np.ndarray, n_perm: int, seed: int, alternative: str = "two-sided",
) -> dict[str, float]:
    """Spearman rho + a PERMUTATION p-value (label-permutation of y, `n_perm`
    resamples, seeded), used instead of scipy's asymptotic p-value where the
    unit count (datasets) is small (reserse/2026-09-17_zostreni_propozice2.md
    section 7.2/10, H1: 'Spearman s permutacni p-hodnotou, 10^4 permutaci,
    seed z configu'). NaN pairs are dropped BEFORE permuting (same
    convention as `bootstrap_spearman_ci`).

    Vectorized: all `n_perm` permutations of rank(y) are generated at once
    (an (n_perm x n) index matrix) and correlated against the fixed rank(x)
    via the closed-form Pearson-on-ranks formula (Spearman rho = Pearson
    correlation of the ranks) - avoids a Python-level loop over n_perm calls
    to `scipy.stats.spearmanr`.

    p = (count+1)/(n_perm+1) (conservative convention, matches
    `src.experiments.stats_holdout.mc_sign_flip_pvalue`). Returns
    {'rho', 'pvalue', 'n'}; n<3 -> rho=pvalue=NaN (undefined)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} and {y.shape}.")
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    n = x.shape[0]
    if n < 3:
        return {"rho": float("nan"), "pvalue": float("nan"), "n": int(n)}
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(f"Unknown alternative '{alternative}' (expected two-sided/greater/less).")

    rank_x = rankdata(x)
    rank_y = rankdata(y)
    rx = rank_x - rank_x.mean()
    rx_norm = float(np.sqrt(np.sum(rx**2)))
    if rx_norm <= 0.0 or float(np.std(rank_y)) <= 0.0:
        return {"rho": 0.0, "pvalue": float("nan"), "n": int(n)}  # a constant x or y - correlation undefined, not fabricated as significant
    rho_obs = float(np.corrcoef(rank_x, rank_y)[0, 1])

    rng = np.random.default_rng(seed)
    perm_idx = np.argsort(rng.random((n_perm, n)), axis=1)
    perm_rank_y = rank_y[perm_idx]  # (n_perm, n)
    ry = perm_rank_y - perm_rank_y.mean(axis=1, keepdims=True)
    ry_norm = np.sqrt(np.sum(ry**2, axis=1))
    perm_rhos = (ry * rx[None, :]).sum(axis=1) / (rx_norm * ry_norm)

    eps = 1e-12
    if alternative == "two-sided":
        count = int(np.sum(np.abs(perm_rhos) >= abs(rho_obs) - eps))
    elif alternative == "greater":
        count = int(np.sum(perm_rhos >= rho_obs - eps))
    else:
        count = int(np.sum(perm_rhos <= rho_obs + eps))
    pvalue = float(count + 1) / float(n_perm + 1)
    return {"rho": rho_obs, "pvalue": pvalue, "n": int(n)}


def bootstrap_ci_1d(values: np.ndarray, stat_fn, n_boot: int, seed: int, ci_level: float = 0.95) -> tuple[float, float]:
    """Generic percentile bootstrap CI of `stat_fn` (e.g. np.median) over a
    1-D sample, resampling INDICES with replacement (`n_boot` repetitions,
    seeded). NaN values are dropped first. (n<2 -> (nan,nan), a bootstrap CI
    of a single point is undefined.)"""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    n = v.shape[0]
    if n < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_vals = np.array([stat_fn(v[row]) for row in idx], dtype=np.float64)
    lo_q, hi_q = (1.0 - ci_level) / 2.0, 1.0 - (1.0 - ci_level) / 2.0
    finite = boot_vals[np.isfinite(boot_vals)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(finite, lo_q)), float(np.quantile(finite, hi_q))


def cluster_bootstrap_ols_slope(
    df: pd.DataFrame, x_col: str, y_col: str, cluster_col: str, n_boot: int, seed: int, ci_level: float = 0.95,
) -> dict[str, float]:
    """OLS slope of `y_col ~ x_col` pooled over ALL rows of `df` (e.g. every
    (dataset, alpha, seed) row of exp8_prop2_check), with a bootstrap 95% CI
    obtained by resampling CLUSTERS (`cluster_col`, e.g. 'dataset') with
    replacement and keeping every row of each resampled cluster - i.e. a
    cluster/block bootstrap, appropriate because rows sharing a cluster are
    NOT independent (reserse/2026-09-17_zostreni_propozice2.md section 7.2:
    'nikdy inference pres radky E8/E9... jednotka je dataset').

    Returns {'slope','intercept','pvalue' (asymptotic, from the POINT-estimate
    OLS fit, reported only as a descriptive cross-check - the confirmatory
    inference is the bootstrap CI), 'ci_lo','ci_hi','n_rows','n_clusters'}.
    Rows/clusters with a non-finite x or y are dropped first (fail-loud if
    nothing remains)."""
    from scipy.stats import linregress

    sub = df[[x_col, y_col, cluster_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if sub.empty:
        raise ValueError(f"cluster_bootstrap_ols_slope: no finite ({x_col}, {y_col}) rows.")
    x = sub[x_col].to_numpy(dtype=np.float64)
    y = sub[y_col].to_numpy(dtype=np.float64)
    if np.unique(x).size < 2:
        raise ValueError(f"cluster_bootstrap_ols_slope: '{x_col}' is constant - OLS slope is undefined.")
    fit = linregress(x, y)

    clusters = sub[cluster_col].unique()
    n_clusters = clusters.shape[0]
    grouped = {c: g for c, g in sub.groupby(cluster_col)}
    rng = np.random.default_rng(seed)
    slopes = np.full(n_boot, np.nan, dtype=np.float64)
    for b in range(n_boot):
        chosen = rng.choice(clusters, size=n_clusters, replace=True)
        boot_df = pd.concat([grouped[c] for c in chosen], ignore_index=True)
        if np.unique(boot_df[x_col].to_numpy()).size < 2:
            continue  # degenerate resample (all-one-cluster with constant x) - skipped, not fabricated
        slopes[b] = linregress(boot_df[x_col].to_numpy(dtype=np.float64), boot_df[y_col].to_numpy(dtype=np.float64)).slope

    lo_q, hi_q = (1.0 - ci_level) / 2.0, 1.0 - (1.0 - ci_level) / 2.0
    finite = slopes[np.isfinite(slopes)]
    ci_lo = float(np.quantile(finite, lo_q)) if finite.size else float("nan")
    ci_hi = float(np.quantile(finite, hi_q)) if finite.size else float("nan")
    return {
        "slope": float(fit.slope), "intercept": float(fit.intercept), "pvalue": float(fit.pvalue),
        "ci_lo": ci_lo, "ci_hi": ci_hi, "n_rows": int(sub.shape[0]), "n_clusters": int(n_clusters),
    }


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


def maximal_insignificant_cliques(sorted_avg_ranks: list[float], cd: float) -> list[tuple[int, int]]:
    """Maximal cliques of the Nemenyi 'not significantly different' relation
    (Demsar 2006, JMLR 7:1-30, Fig. 1) for a critical-difference diagram:
    two methods i, j are connected iff |avg_rank_i - avg_rank_j| < cd. Used
    by `src/figures/fig_cd_diagram.py` to draw ONE horizontal bar per
    maximal clique (not one bar per pairwise comparison - see
    projectstate.md 2026-09-17, author feedback on the previous diagram).

    `sorted_avg_ranks` MUST already be sorted ascending (best rank first);
    the relation graph is then a unit interval graph (ranks are points on a
    line, edges = pairs closer than `cd`), so every maximal clique is a
    contiguous index range [start, end] and can be found by, for each
    candidate start index, greedily extending `end` to the right as far as
    the range still spans less than `cd` - then keeping only ranges not
    fully contained in another range (Bron-Kerbosch would also find these,
    but for a sorted 1-D interval graph maximal cliques ARE exactly the
    maximal such windows, so it is not needed).

    Returns a list of (start, end) inclusive index pairs (into
    `sorted_avg_ranks`) with end > start (singletons carry no information -
    a method not covered by any pair is, by construction, significantly
    different from every other method and needs no bar)."""
    n = len(sorted_avg_ranks)
    candidates: list[tuple[int, int]] = []
    for start in range(n):
        end = start
        while end + 1 < n and sorted_avg_ranks[end + 1] - sorted_avg_ranks[start] < cd:
            end += 1
        if end > start:
            candidates.append((start, end))
    maximal = [
        (s, e) for i, (s, e) in enumerate(candidates)
        if not any(s2 <= s and e2 >= e and (s2, e2) != (s, e) for j, (s2, e2) in enumerate(candidates) if j != i)
    ]
    return sorted(set(maximal))


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
