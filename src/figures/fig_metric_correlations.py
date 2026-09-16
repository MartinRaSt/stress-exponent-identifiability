# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-16
# License: see the LICENSE file in the repository root
"""
Correlations between embedding-quality metrics over the results of
Experiment 1.

Reason (documentation/2026-09-16_zadani_vyber_metrik_hlavni_tabulka.md):
Bae, Jeon, and Seo (IEEE VIS 2025, DOI 10.1109/VIS60296.2025.00014) show
that metrics designed for different aspects of an embedding behave
correlated in practice, so the choice of a metric battery itself favors
certain method families. This script measures that on OUR data - it is
the basis for deciding which columns to keep in the main table
`tab:exp1_main`.

What it computes (unit = a (dataset, method) pair, median over seeds):
1. the Spearman correlation matrix of candidate metrics
   (`fig_metric_correlations.metrics`),
2. the same matrix separately for the stress family and for neighbor
   methods (`report.neighbor_methods`) - the correlation structure may
   differ between families,
3. hierarchical clustering of metrics by the distance 1-|rho| (which
   metrics form a single "direction"),
4. a list of pairs above the `redundancy_threshold` (practically
   interchangeable).

Metrics for which a smaller value is better (`lower_is_better`) have their
sign flipped before computation, so that all columns have the "higher =
better" orientation - otherwise two metrics measuring the same direction
would come out negatively correlated.

Outputs (side by side, per the rule "a CSV lies alongside each PDF"):
  results/figures/[<mode>/]fig_metric_correlations.pdf
  results/figures/[<mode>/]fig_metric_correlations.csv          (long form: pair + rho)
  results/figures/[<mode>/]fig_metric_correlations_clusters.csv (metric-to-cluster assignment)

Run: src\\run_fig_metric_correlations.bat [full|quick|smoke]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

import matplotlib.pyplot as plt

from src.experiments.config_experiments import load_experiments_config
from src.figures.fig_common import (
    OKABE_ITO,
    add_quick_arg,
    figures_out_dir,
    parse_fig_mode,
    require_experiment_csv,
    save_csv_alongside,
    save_figure,
)

FIG_NAME = "fig_metric_correlations"


def _config() -> dict:
    """The `fig_metric_correlations` section from config_experiments.yaml
    (fail loud - no silent fallback to a built-in metric list)."""
    cfg = load_experiments_config()
    try:
        own = cfg["fig_metric_correlations"]
    except KeyError as exc:
        raise KeyError(
            "Missing section 'fig_metric_correlations' in src/experiments/config_experiments.yaml "
            "(metric list, lower_is_better, and redundancy_threshold)."
        ) from exc
    for key in ("metrics", "lower_is_better", "redundancy_threshold"):
        if key not in own:
            raise KeyError(f"Missing key 'fig_metric_correlations.{key}' in config_experiments.yaml.")
    try:
        neighbor_methods = cfg["report"]["neighbor_methods"]
        main_methods = cfg["report"]["main_methods"]
    except KeyError as exc:
        raise KeyError("Missing 'report.neighbor_methods' or 'report.main_methods' in config_experiments.yaml.") from exc
    own = dict(own)
    own["neighbor_methods"] = neighbor_methods
    own["main_methods"] = main_methods
    return own


def _unit_table(mode: str, metrics: list[str]) -> pd.DataFrame:
    """Build a table with unit (dataset, method): median over seeds.

    Cluster-geometry metrics live in a separate CSV of Experiment 1
    (`exp1_cluster_geometry`) and are defined only for datasets with >= 3
    classes - joined with a left merge; missing values remain NaN, and
    Spearman is computed pairwise (see `_spearman_matrix`)."""
    bench = require_experiment_csv("exp1_dr_benchmark", mode)
    geom = require_experiment_csv("exp1_cluster_geometry", mode)

    bench = bench[bench["status"] == "ok"] if "status" in bench.columns else bench
    geom = geom[geom["status"] == "ok"] if "status" in geom.columns else geom

    keys = ["dataset", "method"]
    bench_metrics = [m for m in metrics if m in bench.columns]
    geom_metrics = [m for m in metrics if m in geom.columns and m not in bench_metrics]
    missing = [m for m in metrics if m not in bench_metrics + geom_metrics]
    if missing:
        raise KeyError(
            f"Metrics {missing} are neither in exp1_dr_benchmark_results.csv nor in "
            "exp1_cluster_geometry_results.csv - fix the 'fig_metric_correlations.metrics' list "
            "in config_experiments.yaml (no silent omission)."
        )

    left = bench.groupby(keys, as_index=False)[bench_metrics].median(numeric_only=True)
    right = geom.groupby(keys, as_index=False)[geom_metrics].median(numeric_only=True)
    return left.merge(right, on=keys, how="left")


def _orient(df: pd.DataFrame, metrics: list[str], lower_is_better: list[str]) -> pd.DataFrame:
    """Flip the sign of metrics where a smaller value is better, so all
    columns have the 'higher = better' orientation."""
    out = df.copy()
    for m in metrics:
        if m in lower_is_better:
            out[m] = -out[m]
    return out


def _spearman_matrix(df: pd.DataFrame, metrics: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise Spearman correlation of all metric pairs + the number of
    rows used.

    Computed pairwise (not after dropping entire rows with NaN), because
    cluster-geometry metrics exist only for a subset of datasets; it also
    returns a matrix of the number of observations used, so it is possible
    to judge what the correlation is based on."""
    rho = pd.DataFrame(np.nan, index=metrics, columns=metrics, dtype=float)
    n_used = pd.DataFrame(0, index=metrics, columns=metrics, dtype=int)
    for i, a in enumerate(metrics):
        # the diagonal is handled separately: df[[a, a]] would return two
        # identically-named columns, and spearmanr would then get a 2-D
        # input and return a matrix instead of a number
        rho.loc[a, a] = 1.0
        n_used.loc[a, a] = int(df[a].notna().sum())
        for b in metrics[i + 1:]:
            pair = df[[a, b]].dropna()
            n = len(pair)
            value = float(spearmanr(pair[a], pair[b]).statistic) if n >= 3 else np.nan
            rho.loc[a, b] = rho.loc[b, a] = value
            n_used.loc[a, b] = n_used.loc[b, a] = n
    return rho, n_used


def _cluster_metrics(rho: pd.DataFrame, threshold: float) -> pd.Series:
    """Hierarchical clustering of metrics by the distance 1-|rho| (average
    linkage, deterministic). A cut at height 1-threshold: metrics in the
    same cluster are practically interchangeable."""
    dist = (1.0 - rho.abs()).fillna(1.0).to_numpy()
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0  # numerical symmetrization before squareform
    link = linkage(squareform(dist, checks=False), method="average")
    labels = fcluster(link, t=1.0 - threshold, criterion="distance")
    return pd.Series(labels, index=rho.index, name="cluster")


def _long_form(rho: pd.DataFrame, n_used: pd.DataFrame, family: str) -> pd.DataFrame:
    """Convert the correlation matrix to long form (one pair = one row)."""
    rows = []
    metrics = list(rho.index)
    for i, a in enumerate(metrics):
        for b in metrics[i + 1:]:
            rows.append({
                "family": family, "metric_a": a, "metric_b": b,
                "spearman_rho": rho.loc[a, b], "n_units": int(n_used.loc[a, b]),
                "abs_rho": abs(rho.loc[a, b]),
            })
    return pd.DataFrame(rows)


def _heatmap(ax, rho: pd.DataFrame, title: str) -> None:
    """A single correlation map. A diverging scale centered at zero (RdBu_r
    is colorblind-safe for a two-color gradient)."""
    data = rho.to_numpy(dtype=float)
    im = ax.imshow(data, vmin=-1.0, vmax=1.0, cmap="RdBu_r")
    ax.set_xticks(range(len(rho)))
    ax.set_yticks(range(len(rho)))
    ax.set_xticklabels(rho.columns, rotation=90, fontsize=5)
    ax.set_yticklabels(rho.index, fontsize=5)
    ax.set_title(title, fontsize=7)
    for i in range(len(rho)):
        for j in range(len(rho)):
            value = data[i, j]
            if np.isnan(value):
                continue
            ax.text(j, i, f"{value:.2f}".replace("0.", "."), ha="center", va="center",
                    fontsize=3.6, color="white" if abs(value) > 0.6 else "black")
    return im


def main() -> None:
    parser = add_quick_arg(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    mode = parse_fig_mode(args)

    cfg = _config()
    metrics: list[str] = list(cfg["metrics"])
    threshold = float(cfg["redundancy_threshold"])

    table = _unit_table(mode, metrics)
    table = _orient(table, metrics, list(cfg["lower_is_better"]))

    families = {
        "all": table,
        "neighbor": table[table["method"].isin(cfg["neighbor_methods"])],
        "stress": table[~table["method"].isin(cfg["neighbor_methods"])],
    }

    long_frames, clusters_frames, matrices = [], [], {}
    for family, sub in families.items():
        if len(sub) < 3:
            raise ValueError(
                f"Family '{family}' has only {len(sub)} units (dataset x method) - "
                "not enough for correlations; check the input CSV and report.neighbor_methods."
            )
        rho, n_used = _spearman_matrix(sub, metrics)
        matrices[family] = rho
        long_frames.append(_long_form(rho, n_used, family))
        cl = _cluster_metrics(rho, threshold).reset_index()
        cl.columns = ["metric", "cluster"]
        cl.insert(0, "family", family)
        clusters_frames.append(cl)

    long_df = pd.concat(long_frames, ignore_index=True).sort_values(
        ["family", "abs_rho"], ascending=[True, False]
    )
    clusters_df = pd.concat(clusters_frames, ignore_index=True)

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.0), constrained_layout=True)
    titles = {"all": "All methods", "stress": "Stress family", "neighbor": "Neighbour methods"}
    for ax, family in zip(axes, ["all", "stress", "neighbor"]):
        im = _heatmap(ax, matrices[family], titles[family])
    fig.colorbar(im, ax=axes, shrink=0.8, label="Spearman rho (oriented: higher = better)")

    csv_path = save_csv_alongside(long_df, FIG_NAME)
    clusters_path = figures_out_dir() / f"{FIG_NAME}_clusters.csv"
    clusters_df.to_csv(clusters_path, index=False)
    pdf_results, pdf_article = save_figure(fig, FIG_NAME)

    redundant = long_df[(long_df["family"] == "all") & (long_df["abs_rho"] >= threshold)]
    print(f"Units (dataset x method): {len(table)}; metrics: {len(metrics)}; mode: {mode}")
    print(f"Pairs with |rho| >= {threshold} (all methods): {len(redundant)}")
    for _, r in redundant.iterrows():
        print(f"  {r['metric_a']:<24} {r['metric_b']:<24} rho={r['spearman_rho']:+.3f}  n={r['n_units']}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {clusters_path}")
    print(f"Saved: {pdf_results}" + (f" and {pdf_article}" if pdf_article else ""))


if __name__ == "__main__":
    sys.exit(main())
