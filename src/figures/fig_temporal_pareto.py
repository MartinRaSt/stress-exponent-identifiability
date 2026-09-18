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
K9 (documentation/2026-09-12_plan_smeru_clanku.md) - a Pareto curve of
temporal regularization: for each dataset x solver (SMACOF/SGD), a curve
over the lambda grid, x = normalized displacement (stab) relative to
lambda=0 (the ratio stab(lambda)/stab(lambda=0), "1.0" = no change
relative to the independent SMACOF baseline), y = median stress (qual).
Input: `exp4_temporal_results.csv` (exists, WITHOUT a re-run - see the
projectstate.md 2026-09-11 entry).

Side output (K9 requirement "add the relative change to exp4_stats.csv or
a new CSV" - a new file was chosen so as not to change the schema of
exp4_stats.csv used elsewhere): `results/data/[<mode>/]exp4_relative.csv`
(dataset, solver, lam, stab_median, qual_median, stab_ratio_vs_lambda0,
qual_ratio_vs_lambda0, stab_pct_change_vs_lambda0, qual_pct_change_vs_lambda0).
The computation has been handled since 2026-09-14 by the shared module
`src/experiments/exp4_relative.py` (see documentation/2026-09-14_exp4_relative_poradi.md)
- `src/main.py` calls it outside this figure too, so tables/macros do not
run over stale data.

Run: venv\\python.exe -m src.figures.fig_temporal_pareto [--quick|--full|--smoke]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import pandas as pd

from src.experiments.config_experiments import load_experiments_config
from src.experiments.exp4_relative import compute_exp4_relative
from src.figures.fig_common import (
    OKABE_ITO,
    WIDTH_FULL_WIDTH_IN,
    add_quick_arg,
    display_label,
    parse_fig_mode,
    save_csv_alongside,
    save_figure,
)
from src.common.logging_utils import get_logger

FIG_NAME = "fig_temporal_pareto"
BASE_EXPERIMENT_NAME = "exp4_temporal"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="K9: Pareto curve of temporal regularization (displacement vs. stress over lambda).")
    add_quick_arg(parser)
    args = parser.parse_args()
    mode = parse_fig_mode(args)

    # S2 tweak (documentation/2026-09-12_kontrola_vysledku_s1.md section 4):
    # lambda labels at points overlapped for small lambda - restricted to a
    # selected grid from config_experiments.yaml (fig_temporal_pareto.label_lambdas).
    label_lambdas = set(load_experiments_config()["fig_temporal_pareto"]["label_lambdas"])

    logger = get_logger(FIG_NAME, mode=mode)
    relative_path = compute_exp4_relative(mode, logger)
    relative = pd.read_csv(relative_path)

    solvers = sorted(relative["solver"].unique())
    datasets = sorted(relative["dataset"].unique())

    # Author feedback 2026-09-17 (supplement legibility pass): (1) raw
    # "..._temporal"/"solver=dtsne" identifiers replaced by display_label();
    # (2) each panel used to carry its OWN long x-axis label
    # ("stability ratio vs. lambda=0 (...)") via ax.set_xlabel - with 3
    # panels squeezed into one figure width, tight_layout could not keep
    # them from colliding into each other/becoming illegible. The x-axis
    # MEANING is identical across all 3 panels (only the data range
    # differs), so it is now a single shared label centered under the whole
    # figure (fig.supxlabel) instead of 3 competing copies.
    fig, axes = plt.subplots(
        1, len(solvers), figsize=(WIDTH_FULL_WIDTH_IN, WIDTH_FULL_WIDTH_IN * 0.42 + 0.35), sharey=False,
    )
    if len(solvers) == 1:
        axes = [axes]
    for ax, solver in zip(axes, solvers):
        sub_solver = relative[relative["solver"] == solver]
        for i, dataset_name in enumerate(datasets):
            sub = sub_solver[sub_solver["dataset"] == dataset_name].sort_values("lam")
            if sub.empty:
                continue
            color = OKABE_ITO[i % len(OKABE_ITO)]
            ax.plot(
                sub["stab_ratio_vs_lambda0"], sub["qual_median"], marker="o", color=color,
                label=display_label(dataset_name, "dataset"), linewidth=1.0, markersize=3,
            )
            for _, r in sub.iterrows():
                if not any(abs(r["lam"] - lam_sel) < 1e-9 for lam_sel in label_lambdas):
                    continue
                ax.annotate(f"{r['lam']:g}", (r["stab_ratio_vs_lambda0"], r["qual_median"]), fontsize=4.5, xytext=(2, 2), textcoords="offset points")
        ax.axvline(1.0, color="grey", linewidth=0.6, linestyle="--")
        ax.set_ylabel("median stress (qual)", fontsize=6.5)
        ax.set_title(f"Solver: {display_label(solver, 'solver')}", fontsize=7)
        ax.tick_params(axis="both", labelsize=6)
        ax.legend(fontsize=5, ncol=1)
    fig.supxlabel(r"stability ratio vs. $\lambda$=0 (stab($\lambda$)/stab(0))", fontsize=7)
    fig.suptitle("Temporal regularization Pareto curve: stability gain vs. stress cost (E4)", fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    save_figure(fig, FIG_NAME)
    save_csv_alongside(relative, FIG_NAME)

    print(f"{FIG_NAME}: {len(datasets)} datasets x {len(solvers)} solvers, exp4_relative.csv written to {relative_path}.")


if __name__ == "__main__":
    main()
