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

Author feedback 2026-09-19 (second review): per-point lambda annotations
used to print the bare number (e.g. "10") next to a curve with no unit or
symbol attached - illegible on its own, out of context in a screenshot or a
zoomed crop. Removed: each panel now tags only the ONE lambda value shared
by every curve by construction, lambda=0 (stab_ratio=1, the vertical dashed
line, the independent SMACOF baseline), with the in-figure text
"$\\lambda$=0". Along each curve lambda then increases monotonically from
right (this line) to left, reaching the grid's largest value (`lam_max` in
main(), computed from exp4_relative.csv - not hand-typed) at the curve's
leftmost point; this reading is spelled out in the LaTeX caption
(clanek_en/supplement/sections/s3_negative_results.tex) instead of being
re-derived by the reader from bare numbers.

Run: venv\\python.exe -m src.figures.fig_temporal_pareto [--quick|--full|--smoke]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import pandas as pd

from src.experiments.exp4_relative import compute_exp4_relative
from src.figures.fig_common import (
    ANNOTATION_FONT_PT,
    LABEL_FONT_PT,
    OKABE_ITO,
    WIDTH_SUPPLEMENT_FULL_IN,
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

    # 2026-09-19 (author feedback, second review): per-point bare-number
    # lambda labels ("10" floating with no unit/symbol next to a curve, "the
    # reader has no idea what it means") REMOVED - see the module docstring
    # and the caption in clanek_en/supplement/sections/s3_negative_results.tex
    # for the replacement: the shared vertical dashed line (lambda=0, every
    # panel) is now tagged with a "$\\lambda$=0" text in-figure, and the
    # caption explains that lambda increases from right (this line) to left
    # along each curve, reaching the grid's largest value (lam_max below,
    # computed from the data, never hand-typed) at each curve's leftmost point.
    logger = get_logger(FIG_NAME, mode=mode)
    relative_path = compute_exp4_relative(mode, logger)
    relative = pd.read_csv(relative_path)

    solvers = sorted(relative["solver"].unique())
    datasets = sorted(relative["dataset"].unique())
    lam_max = float(relative["lam"].max())

    # Author feedback 2026-09-17 (supplement legibility pass): (1) raw
    # "..._temporal"/"solver=dtsne" identifiers replaced by display_label();
    # (2) each panel used to carry its OWN long x-axis label
    # ("stability ratio vs. lambda=0 (...)") via ax.set_xlabel - with 3
    # panels squeezed into one figure width, tight_layout could not keep
    # them from colliding into each other/becoming illegible. The x-axis
    # MEANING is identical across all 3 panels (only the data range
    # differs), so it is now a single shared label centered under the whole
    # figure (fig.supxlabel) instead of 3 competing copies.
    # 2026-09-19 (supplement font-size fix): this figure is embedded ONLY in
    # the supplement (clanek_en/supplement/sections/s3_negative_results.tex,
    # [width=\textwidth]) - drawn at WIDTH_SUPPLEMENT_FULL_IN (390pt, see
    # fig_common.py) so that embed is a no-op scale, instead of the
    # DAMI-sized WIDTH_FULL_WIDTH_IN (372pt) that left a small 1.05x LaTeX
    # enlargement on top of already-too-small fontsize= values. The 7
    # datasets are IDENTICAL across all 3 solver panels (only 3 dedicated,
    # per-panel legends existed before) - one shared legend below the
    # figure (constrained_layout reserves real space for it, see the same
    # fix in fig_neighbor_survival.py) frees each ~130pt-wide panel from a
    # 7-entry legend box that no longer fit at the bigger LABEL_FONT_PT.
    fig, axes = plt.subplots(
        1, len(solvers), figsize=(WIDTH_SUPPLEMENT_FULL_IN, WIDTH_SUPPLEMENT_FULL_IN * 0.62), sharey=False,
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.03, wspace=0.08, hspace=0.0)
    if len(solvers) == 1:
        axes = [axes]
    legend_handles: list = []
    legend_labels: list[str] = []
    for ax, solver in zip(axes, solvers):
        sub_solver = relative[relative["solver"] == solver]
        for i, dataset_name in enumerate(datasets):
            sub = sub_solver[sub_solver["dataset"] == dataset_name].sort_values("lam")
            if sub.empty:
                continue
            color = OKABE_ITO[i % len(OKABE_ITO)]
            (line,) = ax.plot(
                sub["stab_ratio_vs_lambda0"], sub["qual_median"], marker="o", color=color,
                label=display_label(dataset_name, "dataset"), linewidth=1.0, markersize=3,
            )
            if display_label(dataset_name, "dataset") not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(display_label(dataset_name, "dataset"))
        ax.axvline(1.0, color="grey", linewidth=0.6, linestyle="--")
        # 2026-09-19 (author feedback, second review): the bare-number
        # per-curve lambda labels are gone (see the comment above main());
        # the ONE value every curve shares by construction - lambda=0 at
        # stab_ratio=1 (this dashed line) - is tagged directly in the
        # figure instead, with the "lambda increases right-to-left, up to
        # lam_max at each curve's leftmost point" reading explained in the
        # caption (same blended-transform tag style as fig_regime_map's
        # t1/t2 threshold lines).
        # 2026-09-19 (author feedback, THIRD review): centered on the line
        # ("ha='center'" straddled the dash-dash-dash exactly through the
        # "=" glyph, reading as a different symbol) and jammed into the
        # top-right corner against the frame. Anchored to the RIGHT of the
        # line instead (ha='right', a few points further left via
        # xytext/offset points) and pulled down from the very top of the
        # axes, so the text sits clearly beside the line, not on it, and
        # clear of both the top and right frame.
        blended = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.annotate(
            r"$\lambda$=0", xy=(1.0, 0.90), xycoords=blended, xytext=(-5, 0), textcoords="offset points",
            fontsize=ANNOTATION_FONT_PT, color="0.2", ha="right", va="center", zorder=5,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=0.5),
        )
        ax.set_ylabel("median stress (qual)", fontsize=LABEL_FONT_PT)
        ax.set_title(f"Solver: {display_label(solver, 'solver')}", fontsize=LABEL_FONT_PT)
        ax.tick_params(axis="both", labelsize=ANNOTATION_FONT_PT)
    fig.legend(handles=legend_handles, labels=legend_labels, loc="outside lower center", ncol=3, fontsize=LABEL_FONT_PT, frameon=False)
    fig.supxlabel(r"stability ratio vs. $\lambda$=0 (stab($\lambda$)/stab(0))", fontsize=LABEL_FONT_PT)
    fig.suptitle("Temporal regularization Pareto curve: stability gain vs. stress cost (E4)", fontsize=LABEL_FONT_PT + 1)
    save_figure(fig, FIG_NAME)
    save_csv_alongside(relative, FIG_NAME)

    print(f"{FIG_NAME}: {len(datasets)} datasets x {len(solvers)} solvers, lambda grid up to {lam_max:g}, exp4_relative.csv written to {relative_path}.")


if __name__ == "__main__":
    main()
