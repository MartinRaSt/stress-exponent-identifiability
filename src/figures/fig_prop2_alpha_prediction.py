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
Q1-step1 (documentation/2026-09-14_exp8_prop2_check.md) - does
distance-concentration theory (Proposition 2) predict the actual empirical
decrease of R_near with increasing alpha (analysis for task spec item 3)?

alpha_bound(tau) = a theoretical prediction from concentration alone: the
smallest alpha from the exp6_alpha_curves.alpha_grid grid for which
r_eps^alpha <= tau (`src.sammon.prop2_check.compute_alpha_bound` - r_eps
depends only on (dataset, seed), medianed over seeds per dataset).
alpha_star = the empirical optimum (argmax median AUC_RNX over seeds),
already computed in `results/tables/[<mode>/]exp6_alpha_optimum.csv`
(exp6_alpha_curves.py).

Inputs (both must exist in the same mode):
  results/data/[<mode>/]exp8_prop2_check_results.csv (r_eps per (dataset,seed))
  results/tables/[<mode>/]exp6_alpha_optimum.csv (alpha_star_max)
A missing input is fail-loud.

Output: results/figures/[<mode>/]fig_prop2_alpha_prediction.pdf (a scatter
of alpha_bound vs. alpha_star, the diagonal y=x, Spearman rho + a
bootstrap 95% CI from config_experiments.yaml exp8_prop2_check.bootstrap;
outlined points = the dataset contains exact duplicate points D_ij=0, see
has_duplicates, fix 2026-09-14 documentation/2026-09-14_exp8_prop2_check.md)
+ a .csv with the same name (dataset, r_eps_median, has_duplicates,
alpha_bound for EVERY tau from exp8_prop2_check.alpha_bound_tau_grid,
alpha_star_max).

Run: venv\\python.exe -m src.figures.fig_prop2_alpha_prediction [--quick|--full|--smoke]
or: src\\run_fig_prop2_alpha_prediction.bat [quick|full|smoke]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experiments.config_experiments import resolve_experiment_config
from src.experiments.stats import bootstrap_spearman_ci
from src.figures.fig_common import (
    OKABE_ITO,
    WIDTH_SINGLE_COL_IN,
    add_quick_arg,
    get_mode_path,
    parse_fig_mode,
    require_csv,
    require_experiment_csv,
    save_csv_alongside,
    save_figure,
)
from src.sammon.prop2_check import compute_alpha_bound

FIG_NAME = "fig_prop2_alpha_prediction"
BASE_EXPERIMENT_NAME = "exp8_prop2_check"
EXP6_BASE_NAME = "exp6_alpha_curves"


def build_alpha_bound_table(
    exp8_ok: pd.DataFrame, alpha_optimum: pd.DataFrame, alpha_grid: list[float], tau_grid: list[float],
) -> pd.DataFrame:
    """For each dataset: median r_eps over seeds (r_eps is independent of
    alpha, see the `compute_alpha_bound` docstring), alpha_bound(tau) for
    each tau in `tau_grid`, and alpha_star_max from `alpha_optimum` (an
    inner join - a dataset missing from one of the tables is dropped with
    a WARNING outside this function, not silently)."""
    r_eps_by_dataset = exp8_ok.groupby("dataset")["r_eps"].median()
    # has_duplicates (2026-09-14, fix of an overly strict D_min=0 check,
    # documentation/2026-09-14_exp8_prop2_check.md): the dataset has exact
    # duplicate points in some (seed,alpha) row - 'any' over all 'ok' rows
    # of the given dataset, so both the figure and the CSV can verify that
    # the alpha_bound vs. alpha_star relationship does not follow from just
    # one subgroup (with/without duplicates).
    has_dup_by_dataset = exp8_ok.groupby("dataset")["has_duplicates"].any()
    rows: list[dict[str, float | str | bool]] = []
    for dataset_name, r_eps in r_eps_by_dataset.items():
        row: dict[str, float | str | bool] = {
            "dataset": dataset_name, "r_eps_median": float(r_eps),
            "has_duplicates": bool(has_dup_by_dataset.get(dataset_name, False)),
        }
        for tau in tau_grid:
            row[f"alpha_bound_tau{tau}"] = compute_alpha_bound(float(r_eps), float(tau), alpha_grid)
        rows.append(row)
    out = pd.DataFrame(rows).set_index("dataset")
    merged = out.join(alpha_optimum.set_index("dataset")[["alpha_star_max"]], how="inner")
    if merged.empty:
        raise ValueError("The dataset intersection between exp8_prop2_check_results.csv and exp6_alpha_optimum.csv is empty.")
    return merged.reset_index()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Q1-step1: does distance concentration (alpha_bound) predict the empirical alpha optimum (alpha_star)?")
    add_quick_arg(parser)
    args = parser.parse_args()
    mode = parse_fig_mode(args)

    exp8 = require_experiment_csv(BASE_EXPERIMENT_NAME, mode)
    exp8_ok = exp8[exp8["status"] == "ok"].copy()
    if exp8_ok.empty:
        raise ValueError(f"exp8_prop2_check_results.csv (mode={mode}) contains no rows with status 'ok'.")

    optimum_path = get_mode_path("results_tables_dir", mode) / "exp6_alpha_optimum.csv"
    alpha_optimum = require_csv(optimum_path, f"venv\\python.exe -m src.experiments.exp6_alpha_curves --{mode} (K6)")

    exp6_cfg = resolve_experiment_config(EXP6_BASE_NAME, mode)
    prop2_cfg = resolve_experiment_config(BASE_EXPERIMENT_NAME, mode)
    alpha_grid = [float(a) for a in exp6_cfg["alpha_grid"]]
    tau_grid = [float(t) for t in prop2_cfg["alpha_bound_tau_grid"]]
    tau_figure = float(prop2_cfg.get("tau_for_figure", tau_grid[len(tau_grid) // 2]))
    boot_cfg = prop2_cfg["bootstrap"]

    table = build_alpha_bound_table(exp8_ok, alpha_optimum, alpha_grid, tau_grid)

    fig_cfg = resolve_experiment_config("fig_prop2_alpha_prediction", mode)
    tau_for_figure = float(fig_cfg.get("tau_for_figure", tau_figure))
    label_datasets = set(fig_cfg["label_datasets"])
    bound_col = f"alpha_bound_tau{tau_for_figure}"
    if bound_col not in table.columns:
        raise KeyError(f"{bound_col} is not in the resulting table - tau_for_figure={tau_for_figure} must be in exp8_prop2_check.alpha_bound_tau_grid.")

    stat = bootstrap_spearman_ci(
        table[bound_col].to_numpy(), table["alpha_star_max"].to_numpy(),
        n_boot=int(boot_cfg["n_boot"]), seed=int(boot_cfg["seed"]), ci_level=float(boot_cfg["ci_level"]),
    )

    fig, ax = plt.subplots(figsize=(WIDTH_SINGLE_COL_IN, WIDTH_SINGLE_COL_IN * 0.95))
    finite = table[table[bound_col].notna()]
    undefined = table[table[bound_col].isna()]
    # has_duplicates: exact duplicate points (D_ij=0) somewhere in the
    # dataset's input (2026-09-14, documentation/2026-09-14_exp8_prop2_check.md)
    # - distinguished by a marker outline, so it can be verified that the
    # alpha_bound vs. alpha_star relationship does not follow from just one subgroup.
    finite_nodup = finite[~finite["has_duplicates"]]
    finite_dup = finite[finite["has_duplicates"]]
    ax.scatter(finite_nodup[bound_col], finite_nodup["alpha_star_max"], color=OKABE_ITO[5], s=16, rasterized=True, zorder=3, label=f"alpha_bound defined, no duplicates (n={finite_nodup.shape[0]})")
    if not finite_dup.empty:
        ax.scatter(finite_dup[bound_col], finite_dup["alpha_star_max"], facecolors=OKABE_ITO[5], edgecolors="black", linewidths=0.6, s=20, rasterized=True, zorder=4, label=f"alpha_bound defined, has duplicates (n={finite_dup.shape[0]})")
    if not undefined.empty:
        # r_eps too close to 1 (diffusive regime) -> alpha_bound outside the
        # grid (NaN, NOT extrapolated) - plotted as a separate point above
        # the grid max for transparency (never silently dropped, see compute_alpha_bound).
        x_undef = np.full(undefined.shape[0], max(alpha_grid) * 1.08)
        ax.scatter(x_undef, undefined["alpha_star_max"], color=OKABE_ITO[6], marker="x", s=20, zorder=3, label=f"alpha_bound undefined (n={undefined.shape[0]})")

    lims = (0.0, max(alpha_grid) * 1.15)
    ax.plot(lims, lims, color="grey", linewidth=0.7, linestyle="--", zorder=1, label="y = x")
    for _, row in table[table["dataset"].isin(label_datasets)].iterrows():
        x_val = row[bound_col] if np.isfinite(row[bound_col]) else max(alpha_grid) * 1.08
        ax.annotate(str(row["dataset"]), (x_val, row["alpha_star_max"]), fontsize=5, xytext=(3, 3), textcoords="offset points")

    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_xlabel(f"alpha_bound (theory, tau={tau_for_figure:g})")
    ax.set_ylabel("alpha_star (empirical, exp6_alpha_optimum)")
    ax.set_title(
        f"rho={stat['rho']:.2f} [{stat['ci_lo']:.2f}, {stat['ci_hi']:.2f}], p={stat['pvalue']:.3g}, n={stat['n']}",
        fontsize=7,
    )
    ax.legend(fontsize=5.5, loc="lower right")
    fig.suptitle("Concentration-based alpha_bound vs. empirical alpha_star", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, FIG_NAME)
    save_csv_alongside(table, FIG_NAME)

    from src.figures.fig_common import figures_out_dir

    pd.DataFrame([{"tau_for_figure": tau_for_figure, **stat}]).to_csv(figures_out_dir() / f"{FIG_NAME}_spearman.csv", index=False)

    print(f"{FIG_NAME}: {table.shape[0]} datasets, rho={stat['rho']:.3f} [{stat['ci_lo']:.3f},{stat['ci_hi']:.3f}] -> results/figures/[{mode}/]{FIG_NAME}.pdf")


if __name__ == "__main__":
    main()
