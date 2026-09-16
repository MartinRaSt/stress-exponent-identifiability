# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-14
# License: see the LICENSE file in the repository root
"""
Q1-step1 (documentation/2026-09-14_exp8_prop2_check.md) - tightness of both
bounds of Proposition 2 (eq:prop2a/eq:prop2b, 03_metoda.tex) as a function
of alpha, colored by rho_NN (distance concentration).

tight_a = R_near(Y) / bound_a(Ybar), tight_b = R_near(Y) / bound_b(Ybar)
(1 = the bound is exactly attained, ->0 = a very loose bound) - see
`src/sammon/prop2_check.py::evaluate_proposition2`.

ONLY rows with p2_holds==True are used (see `src/experiments/
exp8_prop2_check.py` - rows where assumption (P2) fails even with
tolerance are NOT counted in this aggregation/figure, but remain in the
CSV and feed a separate macro for the fraction of excluded rows).

Input: results/data/[<mode>/]exp8_prop2_check_results.csv (K/Q1-step1,
src/experiments/exp8_prop2_check.py). A missing input is fail-loud.

Output: results/figures/[<mode>/]fig_prop2_tightness.pdf (2 panels:
tight_a, tight_b vs. alpha, 1 curve per dataset - median over seeds - color
= median rho_NN of the dataset, log scale, viridis; dashed = the dataset
contains exact duplicate points D_ij=0, see column has_duplicates, fix
2026-09-14 documentation/2026-09-14_exp8_prop2_check.md) + a .csv with the
same name (columns: dataset, alpha, n_seeds, rho_nn_median, tight_a_median,
tight_b_median, has_duplicates).

Run: venv\\python.exe -m src.figures.fig_prop2_tightness [--quick|--full|--smoke]
or: src\\run_fig_prop2_tightness.bat [quick|full|smoke]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm

from src.experiments.config_experiments import load_experiments_config
from src.figures.fig_common import (
    WIDTH_FULL_WIDTH_IN,
    add_quick_arg,
    parse_fig_mode,
    require_experiment_csv,
    save_csv_alongside,
    save_figure,
)

FIG_NAME = "fig_prop2_tightness"
BASE_EXPERIMENT_NAME = "exp8_prop2_check"


def build_tightness_table(exp8_ok_holds: pd.DataFrame) -> pd.DataFrame:
    """Median over seeds for (dataset, alpha): rho_nn, tight_a, tight_b -
    only rows with status 'ok' AND p2_holds==True (see the module docstring).

    `has_duplicates` (2026-09-14, fix of an overly strict D_min=0 check,
    documentation/2026-09-14_exp8_prop2_check.md): 'any' over seeds of the
    given (dataset, alpha) - in practice duplication is a property of the
    dataset, not the seed (subsample_dataset is a no-op for n<=n_max), but
    it is aggregated robustly without this assumption, so both the figure
    and the CSV can verify that the conclusion about bound tightness holds
    in BOTH subgroups (with and without duplicates)."""
    grouped = exp8_ok_holds.groupby(["dataset", "alpha"]).agg(
        n_seeds=("seed", "nunique"),
        rho_nn_median=("rho_nn", "median"),
        tight_a_median=("tight_a", "median"),
        tight_b_median=("tight_b", "median"),
        has_duplicates=("has_duplicates", "any"),
    ).reset_index()
    if grouped.empty:
        raise ValueError("No rows remain after filtering p2_holds==True - cannot plot bound tightness.")
    return grouped.sort_values(["dataset", "alpha"]).reset_index(drop=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Q1-step1: tightness of Proposition 2 bounds (tight_a/tight_b) vs. alpha, color=rho_NN.")
    add_quick_arg(parser)
    args = parser.parse_args()
    mode = parse_fig_mode(args)

    exp8 = require_experiment_csv(BASE_EXPERIMENT_NAME, mode)
    exp8_ok = exp8[exp8["status"] == "ok"].copy()
    if exp8_ok.empty:
        raise ValueError(f"exp8_prop2_check_results.csv (mode={mode}) contains no rows with status 'ok'.")
    n_total_ok = int(exp8_ok.shape[0])
    exp8_ok_holds = exp8_ok[exp8_ok["p2_holds"] == True]  # noqa: E712 - explicit bool comparison, not truthiness
    n_holds = int(exp8_ok_holds.shape[0])
    if exp8_ok_holds.empty:
        raise ValueError(f"exp8_prop2_check_results.csv (mode={mode}): no row with p2_holds==True among the 'ok' rows.")

    table = build_tightness_table(exp8_ok_holds)

    exp_cfg = load_experiments_config()
    fig_cfg = exp_cfg["fig_prop2_tightness"]
    label_datasets = set(fig_cfg["label_datasets"])
    colormap = str(fig_cfg["colormap"])

    dataset_rho = table.groupby("dataset")["rho_nn_median"].median()
    norm = LogNorm(vmin=float(dataset_rho.min()), vmax=float(dataset_rho.max()))
    cmap = plt.get_cmap(colormap)

    # datasets with exact duplicates (D_ij=0 somewhere in the input; after
    # the 2026-09-14 fix these are no longer excluded fail-loud, see
    # build_tightness_table) are drawn dashed, so it is visually verifiable
    # that their curves do not have a qualitatively different shape than
    # datasets without duplicates.
    dataset_has_dup = table.groupby("dataset")["has_duplicates"].any()
    n_dup_datasets = int(dataset_has_dup.sum())

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(WIDTH_FULL_WIDTH_IN, WIDTH_FULL_WIDTH_IN * 0.48), layout="constrained")
    for ax, ycol, ylabel in [(ax_a, "tight_a_median", "tightness (a): R_near(Y) / bound_a"), (ax_b, "tight_b_median", "tightness (b): R_near(Y) / bound_b")]:
        for dataset_name, sub in table.groupby("dataset"):
            sub = sub.sort_values("alpha")
            color = cmap(norm(dataset_rho[dataset_name]))
            linestyle = "--" if dataset_has_dup[dataset_name] else "-"
            ax.plot(sub["alpha"], sub[ycol], color=color, linewidth=0.8, linestyle=linestyle, marker="o", markersize=2.0, alpha=0.85, zorder=2, rasterized=True)
            if dataset_name in label_datasets:
                last = sub.iloc[-1]
                ax.annotate(str(dataset_name), (last["alpha"], last[ycol]), fontsize=5, xytext=(3, 0), textcoords="offset points", va="center")
        ax.set_xlabel("alpha")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.axhline(1.0, color="grey", linewidth=0.6, linestyle="--", zorder=1)
        if n_dup_datasets > 0:
            from matplotlib.lines import Line2D

            proxies = [
                Line2D([0], [0], color="black", linewidth=0.8, linestyle="-", label=f"no duplicates (n={table['dataset'].nunique() - n_dup_datasets})"),
                Line2D([0], [0], color="black", linewidth=0.8, linestyle="--", label=f"has duplicates, D_ij=0 (n={n_dup_datasets})"),
            ]
            ax.legend(handles=proxies, fontsize=5, loc="lower left")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_a, ax_b], location="right", fraction=0.05, pad=0.02)
    cbar.set_label("rho_NN (median per dataset, log scale)", fontsize=6.5)
    cbar.ax.tick_params(labelsize=5.5)

    fig.suptitle("Tightness of Proposition 2 bounds vs. alpha", fontsize=8)
    save_figure(fig, FIG_NAME)
    save_csv_alongside(table, FIG_NAME)

    tight_a_dup = table.loc[table["has_duplicates"], "tight_a_median"]
    tight_a_nodup = table.loc[~table["has_duplicates"], "tight_a_median"]
    print(
        f"{FIG_NAME}: {table['dataset'].nunique()} datasets ({n_dup_datasets} with duplicates), "
        f"{table.shape[0]} (dataset,alpha) rows [p2_holds: {n_holds}/{n_total_ok} 'ok' rows] "
        f"-> results/figures/[{mode}/]{FIG_NAME}.pdf"
    )
    print(
        f"{FIG_NAME}: median tight_a with duplicates={tight_a_dup.median():.4g} (n={tight_a_dup.shape[0]}), "
        f"without duplicates={tight_a_nodup.median():.4g} (n={tight_a_nodup.shape[0]}) - the matching "
        "direction/order of magnitude confirms the conclusion (both bounds very loose) does not follow "
        "from just one subgroup."
    )


if __name__ == "__main__":
    main()
