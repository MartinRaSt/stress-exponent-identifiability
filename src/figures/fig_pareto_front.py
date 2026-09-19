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
K3 - a Pareto front (stress_scale_invariant vs auc_rnx) for the main
article panel (medians over 32 datasets, methods from `report.main_methods`)
+ small multiples (32 panels, one per dataset) for the supplement.

Input: `results/tables/pareto_median_front.csv` and
`results/data/pareto_membership.csv` (src/experiments/pareto_analysis.py -
must be run first, see src/run_pareto_analysis.bat).

Run: venv\\python.exe -m src.figures.fig_pareto_front [--quick|--full|--smoke]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from src.common.config import get_mode_path, get_path
from src.experiments.config_experiments import load_experiments_config
from src.experiments.pareto_analysis import pareto_front_mask
from src.figures.fig_common import (
    ANNOTATION_FONT_PT,
    LABEL_FONT_PT,
    OKABE_ITO,
    WIDTH_FULL_WIDTH_IN,
    WIDTH_SUPPLEMENT_FULL_IN,
    add_quick_arg,
    display_label,
    mode_data_dir,
    parse_fig_mode,
    require_csv,
    save_csv_alongside,
    save_figure,
)

FIG_NAME = "fig_pareto_front"
FIG_NAME_SMALL = "fig_pareto_front_small_multiples"

# method family -> (color from OKABE_ITO, marker) - for consistent reading across both panels
_FAMILY_STYLE = {
    "sammon_alpha0_smacof": (OKABE_ITO[5], "o"),
    "sammon_alpha_smacof": (OKABE_ITO[5], "s"),
    "sammon_alpha2_smacof": (OKABE_ITO[5], "^"),
    "sammon_alpha_auto": (OKABE_ITO[6], "D"),
    "mds": (OKABE_ITO[3], "P"),
    "pca": (OKABE_ITO[0], "X"),
    "tsne_auto": (OKABE_ITO[1], "*"),
    "umap_auto": (OKABE_ITO[2], "v"),
    "pacmap": (OKABE_ITO[4], "p"),
    "trimap": (OKABE_ITO[7], "h"),
}
_DEFAULT_STYLE = ("#999999", "o")

# S2 tweak (documentation/2026-09-12_kontrola_vysledku_s1.md section 3):
# members of the Sammon family (+mds, which is practically identical to
# alpha0, see the K3 note in the S1 documentation) cluster in the bottom
# left of the main panel and their labels overlap - a highlighted inset zoom.
_SAMMON_ZOOM_METHODS = ["sammon_alpha0_smacof", "sammon_alpha_smacof", "sammon_alpha2_smacof", "sammon_alpha_auto", "mds"]

# deterministic cycle of label offsets (point-to-text at clock-like
# positions) - reduces overlap of neighboring labels without relying on an
# external package (adjustText is not installed in venv)
_LABEL_OFFSETS = [(3, 3), (3, -9), (-24, 3), (-24, -9), (3, 11), (-24, 11), (3, -16), (-24, -16)]

# 2026-09-19 (supplement font-size fix): at the bigger ANNOTATION_FONT_PT
# (7.4pt, up from 4.5pt) the generic cyclic `_LABEL_OFFSETS` above placed two
# pairs of labels on top of each other in the zoom inset - sammon_alpha2_smacof
# and sammon_alpha_smacof's labels both landed in the gap BETWEEN the two
# points (see results/tables/pareto_median_front.csv: sammon_alpha2_smacof
# and sammon_alpha_auto are 0.0008 apart in stress at IDENTICAL auc_rnx;
# mds/sammon_alpha0_smacof are 0.0002 apart in stress). These five
# hand-tuned offsets (in points, at the actual data geometry) point each
# label away from its nearest neighbor instead of at a fixed clock position.
_ZOOM_LABEL_OFFSETS = {
    # 2026-09-19 fix: the first attempt pointed the top pair's leftmost two
    # labels UP - at ANNOTATION_FONT_PT that put them above the inset box's
    # own top border, where they collided with the MAIN panel's "TriMap"
    # label (an unrelated point the inset's connector lines happen to pass
    # near - see mark_inset below). All three top-pair labels now point
    # DOWN, into the empty band between the top and bottom point pairs,
    # staggered by `dy` so the three (different lengths, different `x`)
    # do not stack on top of each other.
    "sammon_alpha_smacof": (-4, -11),    # leftmost of the top pair -> label down-left
    "sammon_alpha2_smacof": (2, -24),    # middle of the top pair -> label further down (stagger vs. the other two)
    "sammon_alpha_auto": (5, -11),       # rightmost of the top pair -> label down-right (away from the other two)
    # 2026-09-19 fix: pointing the bottom pair's labels DOWN-RIGHT (the
    # first attempt) ran off the inset's right edge and collided with the
    # main panel's own "PHATE"/"MDS (scikit-learn)" labels, which sit just
    # outside the inset at almost the same spot - both now point LEFT
    # (away from the crowded right edge) with `ha="right"` (see
    # `_plot_panel`) so the text ends AT the marker instead of starting there.
    "mds": (-6, 8),                      # top of the bottom pair -> label up-left
    "sammon_alpha0_smacof": (-6, -9),    # bottom of the bottom pair -> label down-left
}

# Short aliases (this zoom inset only - the full names above are still used
# in the main panel and everywhere else): "Kamada-Kawai" alone is 12
# characters and, together with its own offset landing close to
# "Sammon ($\alpha=1$)", was the main remaining source of overlap even after
# the offset fix above; "MDS (scikit-learn)" ran off the panel's right edge.
_ZOOM_LABEL_SHORT = {
    "sammon_alpha2_smacof": r"K.-Kawai ($\alpha=2$)",
    "mds": "MDS (sklearn)",
}


def _style(method: str) -> tuple[str, str]:
    return _FAMILY_STYLE.get(method, _DEFAULT_STYLE)


def _plot_panel(
    ax, sub: pd.DataFrame, front_mask: pd.Series, label_points: bool, fontsize: float,
    skip_label_methods: frozenset[str] = frozenset(),
    label_offsets: dict[str, tuple[float, float]] | None = None,
    label_overrides: dict[str, str] | None = None,
) -> None:
    label_offsets = label_offsets or {}
    label_overrides = label_overrides or {}
    for k, (method_name, row) in enumerate(sub.set_index("method").iterrows()):
        color, marker = _style(method_name)
        on_front = bool(front_mask.get(method_name, False))
        ax.scatter(
            row["stress_scale_invariant"], row["auc_rnx"], color=color, marker=marker,
            s=30 if on_front else 16, edgecolors="black" if on_front else "none",
            linewidths=0.6 if on_front else 0.0, zorder=3 if on_front else 2, rasterized=True,
        )
        if label_points and method_name not in skip_label_methods:
            is_custom = method_name in label_offsets
            dx, dy = label_offsets.get(method_name, _LABEL_OFFSETS[k % len(_LABEL_OFFSETS)])
            text = label_overrides.get(method_name, display_label(method_name, "method"))
            # Custom (hand-tuned) offsets anchor the text AT the marker
            # instead of starting there (matplotlib's annotate default,
            # ha="left") - a negative dx means "label to the left", which
            # should end at the point (ha="right"), not start there and run
            # further left/off the panel.
            ha = ("right" if dx < 0 else "left") if is_custom else "left"
            ax.annotate(
                text, (row["stress_scale_invariant"], row["auc_rnx"]), fontsize=fontsize,
                xytext=(dx, dy), textcoords="offset points", ha=ha,
            )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="K3: Pareto front stress_scale_invariant vs auc_rnx (main panel + small multiples).")
    add_quick_arg(parser)
    args = parser.parse_args()
    mode = parse_fig_mode(args)

    tables_dir = get_mode_path("results_tables_dir", mode)
    median_front_path = tables_dir / "pareto_median_front.csv"
    median_front = require_csv(median_front_path, f"venv\\python.exe -m src.experiments.pareto_analysis --{mode}")

    membership_path = mode_data_dir(mode) / "pareto_membership.csv"
    membership = require_csv(membership_path, f"venv\\python.exe -m src.experiments.pareto_analysis --{mode}")

    exp_cfg = load_experiments_config()
    main_methods: list[str] = exp_cfg["report"]["main_methods"]

    # --- main panel: medians over 32 datasets (report.main_methods) --------
    # 2026-09-19 (supplement font-size fix): this panel is embedded ONLY in
    # the supplement (clanek_en/supplement/sections/s3_negative_results.tex,
    # [width=\columnwidth]) - drawn at WIDTH_SUPPLEMENT_FULL_IN (390pt, see
    # fig_common.py) so that embed is a no-op scale, instead of the literal
    # WIDTH_SINGLE_COL_IN (255.12pt) that left a 1.53x LaTeX enlargement on
    # top of an already-too-small inset-zoom fontsize= (4.5pt).
    fig, ax = plt.subplots(figsize=(WIDTH_SUPPLEMENT_FULL_IN, WIDTH_SUPPLEMENT_FULL_IN * 0.75))
    front_mask_main = median_front.set_index("method")["on_front"]
    # S2 tweak: members of the Sammon family (+mds) are labeled only in the
    # inset zoom (see below) - in the main panel their labels would just
    # clutter the area where the zoom frame is already drawn.
    _plot_panel(ax, median_front, front_mask_main, label_points=True, fontsize=ANNOTATION_FONT_PT, skip_label_methods=frozenset(_SAMMON_ZOOM_METHODS))
    ax.set_xlabel("stress (scale-invariant, lower = better)")
    ax.set_ylabel("AUC$_{RNX}$ (higher = better)")
    ax.set_title("Pareto front: global fidelity vs. local neighborhood ranking\n(median over 32 datasets, E1)", fontsize=LABEL_FONT_PT)

    # S2 tweak: an inset zoom of the bottom-left corner, where members of
    # the Sammon family (+mds) cluster - in the main panel their labels are
    # otherwise illegible due to the small spacing of points (see
    # kontrola_vysledku_s1.md section 3, "Sammon family labels overlap").
    zoom_sub = median_front[median_front["method"].isin(_SAMMON_ZOOM_METHODS)]
    if len(zoom_sub) >= 2:
        xs, ys = zoom_sub["stress_scale_invariant"], zoom_sub["auc_rnx"]
        pad_x = max((xs.max() - xs.min()) * 0.35, xs.max() * 0.02, 1e-6)
        pad_y = max((ys.max() - ys.min()) * 0.35, ys.max() * 0.02, 1e-6)
        axins = inset_axes(ax, width="40%", height="42%", loc="lower right", borderpad=1.6)
        zoom_front_mask = front_mask_main[front_mask_main.index.isin(_SAMMON_ZOOM_METHODS)]
        _plot_panel(
            axins, zoom_sub, zoom_front_mask, label_points=True, fontsize=ANNOTATION_FONT_PT,
            label_offsets=_ZOOM_LABEL_OFFSETS, label_overrides=_ZOOM_LABEL_SHORT,
        )
        axins.set_xlim(xs.min() - pad_x, xs.max() + pad_x)
        axins.set_ylim(ys.min() - pad_y, ys.max() + pad_y)
        axins.tick_params(labelsize=ANNOTATION_FONT_PT)
        axins.set_xlabel("")
        axins.set_ylabel("")
        mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.5", linewidth=0.5)

    fig.tight_layout()
    save_figure(fig, FIG_NAME)
    save_csv_alongside(median_front, FIG_NAME)

    # --- small multiples: 32 datasets, main_methods only, the front
    # recomputed ONLY for the displayed methods (otherwise the "front"
    # would refer to points outside the panel) --------------------------
    sub_main = membership[membership["method"].isin(main_methods)].copy()
    datasets = sorted(sub_main["dataset"].unique())
    n_ds = len(datasets)
    n_cols = 6
    n_rows = -(-n_ds // n_cols)
    fig2, axes = plt.subplots(n_rows, n_cols, figsize=(WIDTH_FULL_WIDTH_IN, WIDTH_FULL_WIDTH_IN / n_cols * n_rows))
    axes_flat = axes.ravel()
    small_multiple_rows = []
    for i, dataset_name in enumerate(datasets):
        ax_i = axes_flat[i]
        sub_ds = sub_main[sub_main["dataset"] == dataset_name].set_index("method")
        local_front = pareto_front_mask(sub_ds, {"auc_rnx": "max", "stress_scale_invariant": "min"})
        _plot_panel(ax_i, sub_ds.reset_index(), local_front, label_points=False, fontsize=4)
        ax_i.set_title(display_label(dataset_name, "dataset"), fontsize=5.5)
        ax_i.tick_params(labelsize=4)
        for method_name, is_front in local_front.items():
            small_multiple_rows.append({
                "dataset": dataset_name, "method": method_name, "on_front_main_methods_only": bool(is_front),
                "auc_rnx": float(sub_ds.loc[method_name, "auc_rnx"]), "stress_scale_invariant": float(sub_ds.loc[method_name, "stress_scale_invariant"]),
            })
    for j in range(n_ds, len(axes_flat)):
        axes_flat[j].axis("off")
    fig2.suptitle("Per-dataset Pareto front (stress vs. AUC$_{RNX}$), main-text methods only", fontsize=8)
    fig2.tight_layout(rect=(0, 0, 1, 0.98))
    save_figure(fig2, FIG_NAME_SMALL)
    save_csv_alongside(pd.DataFrame(small_multiple_rows), FIG_NAME_SMALL)

    print(f"{FIG_NAME}: main panel {median_front.shape[0]} methods, small multiples {n_ds} datasets x {len(main_methods)} methods.")


if __name__ == "__main__":
    main()
