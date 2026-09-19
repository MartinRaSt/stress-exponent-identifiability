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
Figure 14g - convergence curves (stress vs. epoch): sgd_naive vs. sgd_stab
from `results/data/exp2_convergence_curves.csv` (E2 part a), a log-y axis,
one curve per (dataset, solver, seed) - by default only the first logged
seed per dataset is plotted for legibility (see --seed).

Run: venv\\python.exe -m src.figures.fig_sgd_convergence [--quick]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt

from src.figures.fig_common import ANNOTATION_FONT_PT, LABEL_FONT_PT, OKABE_ITO, WIDTH_SUPPLEMENT_FULL_IN, add_quick_arg, display_label, mode_data_dir, parse_fig_mode, require_csv, save_csv_alongside, save_figure

FIG_NAME = "fig_sgd_convergence"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Convergence curves sgd_naive vs. sgd_stab (E2a).")
    add_quick_arg(parser)
    args = parser.parse_args()
    mode = parse_fig_mode(args)

    curves_path = mode_data_dir(mode) / "exp2_convergence_curves.csv"
    df = require_csv(curves_path, f"venv\\python.exe -m src.experiments.exp2_solver_scaling --{mode}")

    datasets = sorted(df["dataset"].unique())
    # 2026-09-19 (supplement font-size fix): this figure is embedded ONLY in
    # the supplement (clanek_en/supplement/sections/s3_negative_results.tex,
    # [width=\columnwidth]) - drawn at WIDTH_SUPPLEMENT_FULL_IN (390pt, see
    # fig_common.py) so that embed is a no-op scale, instead of
    # WIDTH_SINGLE_COL_IN*len(datasets) (510.24pt for 2 datasets) which was
    # actually WIDER than the 390pt target and left a 0.76x LaTeX SHRINK on
    # top of an already-too-small legend fontsize= (6pt).
    fig, axes = plt.subplots(1, len(datasets), figsize=(WIDTH_SUPPLEMENT_FULL_IN, WIDTH_SUPPLEMENT_FULL_IN / len(datasets) * 0.85), squeeze=False)

    solver_colors = {"sgd_naive": OKABE_ITO[6], "sgd_stab": OKABE_ITO[5]}
    for ax_idx, dataset_name in enumerate(datasets):
        ax = axes[0][ax_idx]
        sub = df[df["dataset"] == dataset_name]
        seed0 = int(sub["seed"].min())
        for solver in sorted(sub["solver"].unique()):
            s = sub[(sub["solver"] == solver) & (sub["seed"] == seed0)].sort_values("epoch")
            if s.empty:
                continue
            ax.plot(s["epoch"], s["stress_scale_invariant"], label=display_label(solver, "solver"), color=solver_colors.get(solver, OKABE_ITO[0]), linewidth=1.2)
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        if ax_idx == 0:
            ax.set_ylabel("scale-invariant stress (log)")
        ax.set_title(f"{display_label(dataset_name, 'dataset')} (seed={seed0})", fontsize=LABEL_FONT_PT)
        ax.legend(fontsize=ANNOTATION_FONT_PT)

    fig.suptitle("SGD convergence: naive vs. stabilized", fontsize=LABEL_FONT_PT + 1)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, FIG_NAME)
    save_csv_alongside(df, FIG_NAME)
    print(f"{FIG_NAME}: {len(datasets)} datasets, {df.shape[0]} curve rows.")


if __name__ == "__main__":
    main()
