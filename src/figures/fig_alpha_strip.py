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
Figure 14d - an alpha x dataset heatmap: R_NX AUC (or scale-invariant
stress) as color, from `results/data/exp5_ablation_results.csv` (the
ablation grid of alpha x init x eps_D quantile) - aggregated by median over
init/eps_D_q/seed for each (dataset, alpha) cell.

Run: venv\\python.exe -m src.figures.fig_alpha_strip [--quick] [--metric auc_rnx]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np

from src.experiments.exp_common import resolve_experiment_name
from src.figures.fig_common import WIDTH_FULL_WIDTH_IN, add_quick_arg, parse_fig_mode, require_experiment_csv, save_csv_alongside, save_figure

FIG_NAME = "fig_alpha_strip"
BASE_EXPERIMENT_NAME = "exp5_ablation"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Alpha x dataset heatmap (E5 ablation).")
    add_quick_arg(parser)
    parser.add_argument("--metric", type=str, default="auc_rnx", help="metric for the color (default auc_rnx)")
    args = parser.parse_args()
    mode = parse_fig_mode(args)
    EXPERIMENT_NAME = resolve_experiment_name(BASE_EXPERIMENT_NAME, mode)

    df = require_experiment_csv(BASE_EXPERIMENT_NAME, mode)
    ok = df[(df["status"] == "ok") & df[args.metric].notna()]
    if ok.empty:
        raise ValueError(f"No successful runs with metric '{args.metric}' in {EXPERIMENT_NAME}_results.csv.")

    pivot = ok.pivot_table(index="dataset", columns="alpha", values=args.metric, aggfunc="median")
    pivot = pivot.sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(WIDTH_FULL_WIDTH_IN, 0.35 * WIDTH_FULL_WIDTH_IN + 0.3 * pivot.shape[0]))
    im = ax.imshow(pivot.to_numpy(), cmap="viridis", aspect="auto")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([f"{a:g}" for a in pivot.columns], fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_xlabel("alpha")
    fig.colorbar(im, ax=ax, label=args.metric, shrink=0.8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.to_numpy()[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5, color="white")

    fig.suptitle(f"alpha x dataset heatmap: median {args.metric} (E5 ablation)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, FIG_NAME)
    save_csv_alongside(pivot.reset_index(), FIG_NAME)
    print(f"{FIG_NAME}: {pivot.shape[0]} datasets x {pivot.shape[1]} alpha values.")


if __name__ == "__main__":
    main()
