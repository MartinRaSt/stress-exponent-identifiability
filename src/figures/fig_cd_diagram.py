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
Figure 14e - a critical difference (Friedman/Nemenyi) diagram: the mean
rank of methods over datasets + the critical difference CD, from
`results/data/stats_<experiment>_<metric>.csv` (src/experiments/stats.py).

Run: venv\\python.exe -m src.figures.fig_cd_diagram [--quick] [--experiment exp1_dr_benchmark] [--metric auc_rnx]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt

from src.figures.fig_common import WIDTH_FULL_WIDTH_IN, add_quick_arg, mode_data_dir, parse_fig_mode, require_csv, save_csv_alongside, save_figure

FIG_NAME = "fig_cd_diagram"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Critical difference diagram (Friedman/Nemenyi).")
    add_quick_arg(parser)
    # default: Friedman/Nemenyi only over report.main_methods (suffix "_main",
    # src/experiments/report_tables.py::write_exp1_main_table, K12b) - k <= 20,
    # so q_alpha exists; over all 21 E1 methods the stats file is not generated
    parser.add_argument("--experiment", type=str, default="exp1_dr_benchmark_main")
    parser.add_argument("--metric", type=str, default="auc_rnx")
    args = parser.parse_args()
    mode = parse_fig_mode(args)

    stats_path = mode_data_dir(mode) / f"stats_{args.experiment}_{args.metric}.csv"
    df = require_csv(stats_path, f"venv\\python.exe -m src.experiments.stats --{mode} (requires {args.experiment}_results.csv)")
    df = df.sort_values("avg_rank")

    cd = float(df["cd"].iloc[0])
    methods = df["method"].tolist()
    ranks = df["avg_rank"].tolist()
    n_methods = len(methods)

    fig, ax = plt.subplots(figsize=(WIDTH_FULL_WIDTH_IN, max(2.0, 0.35 * n_methods)))
    ax.set_xlim(min(ranks) - 0.5, max(ranks) + 0.5)
    ax.set_ylim(0, n_methods + 1)
    ax.invert_yaxis()

    for i, (method_name, rank) in enumerate(zip(methods, ranks)):
        y = i + 1
        ax.scatter([rank], [y], color="black", zorder=3, s=15)
        ax.text(rank, y - 0.25, f"{method_name} ({rank:.2f})", ha="center", fontsize=6)

    # connecting lines between methods whose mean-rank difference is < CD (statistically insignificant difference)
    y_bar = n_methods + 0.6
    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            if abs(ranks[j] - ranks[i]) < cd:
                ax.plot([ranks[i], ranks[j]], [y_bar, y_bar], color="#0072B2", linewidth=2.5, solid_capstyle="round")
                y_bar -= 0.3

    chi2 = df["friedman_chi2"].iloc[0]
    pval = df["friedman_pvalue"].iloc[0]
    kendall_w = df["kendall_w"].iloc[0]
    ax.set_title(
        f"CD diagram: {args.experiment} / {args.metric} (CD={cd:.3f}, Friedman chi2={chi2:.2f}, p={pval:.3g}, Kendall W={kendall_w:.3f})",
        fontsize=7,
    )
    ax.set_xlabel("average rank (1 = best)")
    ax.set_yticks([])
    fig.tight_layout()
    save_figure(fig, f"{FIG_NAME}_{args.experiment}_{args.metric}")
    save_csv_alongside(df, f"{FIG_NAME}_{args.experiment}_{args.metric}")
    print(f"{FIG_NAME}: {args.experiment}/{args.metric}, {n_methods} methods, CD={cd:.3f}.")


if __name__ == "__main__":
    main()
