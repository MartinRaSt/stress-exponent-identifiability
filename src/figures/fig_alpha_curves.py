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
K6 (documentation/2026-09-12_plan_smeru_clanku.md) - figures for exp6
(alpha curves, `src/experiments/exp6_alpha_curves.py`):

  fig_alpha_curves.pdf      - the main article figure, 6 datasets from the
                               config (`exp6_alpha_curves.main_figure_datasets`
                               - 3 concentrated high-dimensional + 3
                               manifolds), 2 rows (auc_rnx on top, stress at
                               the bottom) x 6 columns, median +- IQR over
                               seeds vs. alpha.
  fig_alpha_curves_all_p*.pdf - supplement, all datasets from
                               `exp6_alpha_curves.datasets` split into pages
                               of `fig_alpha_curves_all.datasets_per_page`
                               (a grid of `fig_alpha_curves_all.n_cols`
                               columns), each panel auc_rnx (left axis) +
                               stress (right axis, twin). Split across
                               several files so the figure fits on a
                               supplement page (validator 2026-09-14,
                               podklady/2026-09-14_validace_clanku.md
                               1.7 - the original single 4x8 figure
                               overflowed the bottom margin by 186pt); see
                               documentation/2026-09-14_oprava_fig_alpha_curves_all.md.

Input: results/data/[<mode>/]exp6_alpha_curves_results.csv (K6). A missing
input is fail-loud (`require_experiment_csv`) - no figure is fabricated.

Run: venv\\python.exe -m src.figures.fig_alpha_curves [--quick|--full|--smoke]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experiments.config_experiments import load_experiments_config
from src.figures.fig_common import (
    OKABE_ITO,
    WIDTH_FULL_WIDTH_IN,
    WIDTH_SUPPLEMENT_FULL_IN,
    add_quick_arg,
    display_label,
    parse_fig_mode,
    require_experiment_csv,
    save_csv_alongside,
    save_figure,
)

BASE_EXPERIMENT_NAME = "exp6_alpha_curves"
FIG_MAIN = "fig_alpha_curves"
FIG_ALL = "fig_alpha_curves_all"

_COLOR_AUC = OKABE_ITO[5]  # blue
_COLOR_STRESS = OKABE_ITO[6]  # vermillion

# 2026-09-19 proportions fix (author feedback: on a twin-axis panel this
# narrow, e.g. AUC_RNX in [0.370, 0.3825] for one dataset, matplotlib's
# default tick locator had to add 3-4 decimal digits ("0.3825", "0.02975")
# to make consecutive ticks distinguishable, and EVERY panel reserves this
# margin on BOTH sides (AUC ticks on the left, stress ticks on the right) -
# "popisky os zaberou vic sirky nez krivky", measured: axes area fell to
# ~19% of the canvas with a full numeric tick axis on both sides of every
# panel). Per the task's own suggestion ("normalizovat kazdy panel na
# vlastni rozsah a popisovat jen min/max"): the y AXIS (ticks, tick labels)
# is removed entirely for every panel (each panel already has its own,
# independent y-range - there is no shared scale a tick axis could
# usefully communicate across panels anyway) and replaced by two small,
# color-matched text labels IN the panel's own top-left/bottom-left
# (AUC) or top-right/bottom-right (stress) corner, giving the min and max
# of that panel's own curve - the exact two numbers a reader would have
# read off the removed axis, with no per-panel margin reserved for them.
#
# 2026-09-19 ROUND 2 (author feedback, having seen the typeset PDF: a
# monotone curve has one of its own endpoints AT the corner where its label
# sits - e.g. an increasing curve's lowest value is at its bottom-left
# start - and a fixed 10% margins() headroom plus a translucent text
# background was NOT enough at these small panel sizes: the label's own
# line height (needed in POINTS, a fixed physical size) was often a LARGER
# fraction of the panel's height than a blind 10% margin reserved, so the
# label still visually merged with the curve/marker there). Fixed
# geometrically instead of by a fixed guess: `_reserve_label_band` measures
# the axes' ACTUAL rendered height (after the whole figure/grid is laid
# out, via `fig.canvas.draw()`) and expands ylim by EXACTLY enough - in
# points, converted to a fraction of the (now known) axes height - that a
# label of `fontsize` can never reach the data, on either end, regardless
# of where in x the curve's own extremum happens to sit. This needs the
# figure's layout to be FINAL first, so labeling is a two-pass process:
# `_plot_twin_panel` only records (ax, values, color, side) in
# `label_specs` (via `_pending_range_labels`) while plotting; the caller
# draws the finished figure once, then calls `_apply_range_labels` for
# every recorded spec.
_LABEL_LINE_HEIGHT_FACTOR = 1.45  # generous line height (leading) as a multiple of the font's point size
_LABEL_EDGE_GAP_PT = 2.0          # extra clearance between the label's own line box and the data extremum
_MAX_LABEL_BAND_FRAC = 0.45       # safety cap so a degenerate (very short) axes cannot blow up the reserved band


def _pending_range_labels(ax, values: np.ndarray, color: str, side: str, fontsize: float) -> tuple:
    """Removes the y-axis ticks/labels of `ax` (see the module docstring
    above) and returns a (ax, lo, hi, color, side, fontsize) spec for
    `_apply_range_labels` to place AFTER the whole figure is laid out. Does
    NOT touch ylim or draw any text yet - the final axes height is not known
    until the whole grid (titles, other panels) exists."""
    ax.set_yticks([])
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (ax, float("nan"), float("nan"), color, side, fontsize)
    return (ax, float(np.min(finite)), float(np.max(finite)), color, side, fontsize)


def _apply_range_labels(fig, label_specs: list[tuple]) -> None:
    """Second pass (see the module docstring above): `fig` must already be
    FULLY laid out (this function calls `fig.canvas.draw()` once itself,
    then reads each axes' final rendered height) - call this AFTER every
    title/label/panel of `fig` has been set, not before. For every
    (ax, lo, hi, color, side, fontsize) spec: reserves a band of height
    (`_LABEL_LINE_HEIGHT_FACTOR` * fontsize + `_LABEL_EDGE_GAP_PT`) points
    at the top and bottom of `ax` (by expanding ylim - a real reservation,
    not a fixed guess, so the label geometrically CANNOT reach the plotted
    curve on either end) and prints the min/max there, never fabricated."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, lo, hi, color, side, fontsize in label_specs:
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            continue
        axes_height_px = ax.get_window_extent(renderer=renderer).height
        axes_height_pt = axes_height_px * 72.0 / fig.dpi
        line_height_pt = fontsize * _LABEL_LINE_HEIGHT_FACTOR + _LABEL_EDGE_GAP_PT
        frac = min(line_height_pt / axes_height_pt, _MAX_LABEL_BAND_FRAC)
        rng = hi - lo
        new_range = rng / max(1.0 - 2.0 * frac, 1e-6)
        ax.set_ylim(lo - frac * new_range, hi + frac * new_range)

        decimals = min(max(0, int(np.ceil(-np.log10(rng))) + 1), 4)
        x = 0.03 if side == "left" else 0.97
        ax.text(x, 1.0 - frac / 2.0, f"{hi:.{decimals}f}", transform=ax.transAxes, ha=side, va="center", color=color, fontsize=fontsize, zorder=4)
        ax.text(x, frac / 2.0, f"{lo:.{decimals}f}", transform=ax.transAxes, ha=side, va="center", color=color, fontsize=fontsize, zorder=4)


def _median_iqr_by_alpha(df: pd.DataFrame, dataset_name: str, metric: str) -> pd.DataFrame:
    """Median + IQR over seeds for a given dataset/metric, sorted by alpha."""
    sub = df[(df["dataset"] == dataset_name) & df[metric].notna()]
    g = sub.groupby("alpha")[metric]
    out = g.median().rename("median").to_frame()
    out["q25"] = g.quantile(0.25)
    out["q75"] = g.quantile(0.75)
    out["n_seeds"] = g.count()
    return out.reset_index().sort_values("alpha")


def _plot_twin_panel(
    ax_auc,
    dataset_name: str,
    df: pd.DataFrame,
    show_ylabel: bool,
    title_fontsize: float = 7,
    tick_labelsize: float = 6,
    ylabel_fontsize: float = 6.5,
) -> tuple[pd.DataFrame, list[tuple]]:
    """Plot a single panel (auc_rnx on the left axis, stress on the right
    axis, twinx) for a given dataset. Returns (long DataFrame to save to
    CSV, label_specs) - `label_specs` are the (ax, lo, hi, color, side,
    fontsize) tuples `_apply_range_labels` needs, collected here but NOT
    yet drawn (see that function's docstring for why the labeling is a
    deferred second pass over the whole figure)."""
    auc = _median_iqr_by_alpha(df, dataset_name, "auc_rnx")
    stress = _median_iqr_by_alpha(df, dataset_name, "stress_scale_invariant")

    ax_auc.plot(auc["alpha"], auc["median"], color=_COLOR_AUC, marker="o", markersize=2.5, linewidth=1.1, zorder=3)
    ax_auc.fill_between(auc["alpha"], auc["q25"], auc["q75"], color=_COLOR_AUC, alpha=0.2, linewidth=0, zorder=2, rasterized=True)
    label_specs = [_pending_range_labels(ax_auc, pd.concat([auc["median"], auc["q25"], auc["q75"]]).to_numpy(), _COLOR_AUC, "left", tick_labelsize)]
    ax_auc.tick_params(axis="x", labelsize=tick_labelsize)

    ax_stress = ax_auc.twinx()
    ax_stress.plot(stress["alpha"], stress["median"], color=_COLOR_STRESS, marker="s", markersize=2.2, linewidth=1.0, linestyle="--", zorder=3)
    ax_stress.fill_between(stress["alpha"], stress["q25"], stress["q75"], color=_COLOR_STRESS, alpha=0.15, linewidth=0, zorder=1, rasterized=True)
    label_specs.append(_pending_range_labels(ax_stress, pd.concat([stress["median"], stress["q25"], stress["q75"]]).to_numpy(), _COLOR_STRESS, "right", tick_labelsize))

    ax_auc.set_title(display_label(dataset_name, "dataset"), fontsize=title_fontsize)
    if show_ylabel:
        ax_auc.set_ylabel("AUC$_{RNX}$", color=_COLOR_AUC, fontsize=ylabel_fontsize)

    auc["dataset"] = dataset_name
    auc["metric"] = "auc_rnx"
    stress["dataset"] = dataset_name
    stress["metric"] = "stress_scale_invariant"
    return pd.concat([auc, stress], ignore_index=True), label_specs


def _make_main_figure(df: pd.DataFrame, main_datasets: list[str]) -> tuple[plt.Figure, pd.DataFrame]:
    missing = [d for d in main_datasets if d not in df["dataset"].unique()]
    if missing:
        raise KeyError(f"exp6_alpha_curves_results.csv is missing datasets {missing} from main_figure_datasets.")

    n = len(main_datasets)
    # 2026-09-19 proportions fix: fig.tight_layout() is UNRELIABLE with
    # twinx() axes (matplotlib limitation - "Tight layout not applied"
    # observed once the y-axis tick-label footprint changed here, silently
    # shrinking every axes to a sliver); constrained_layout (used
    # everywhere else in this project for the same reason, see
    # fig_graph_layouts.py) solves panel/label spacing from actual bounding
    # boxes instead and does not have this failure mode.
    fig, axes = plt.subplots(
        1, n, figsize=(WIDTH_FULL_WIDTH_IN, WIDTH_FULL_WIDTH_IN * 0.38), sharex=True,
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.04, hspace=0.02)
    if n == 1:
        axes = [axes]
    csv_pieces = []
    all_label_specs: list[tuple] = []
    for i, (ax, dataset_name) in enumerate(zip(axes, main_datasets)):
        piece, label_specs = _plot_twin_panel(ax, dataset_name, df, show_ylabel=(i == 0))
        csv_pieces.append(piece)
        all_label_specs.extend(label_specs)
        ax.set_xlabel("alpha", fontsize=6.5)
    fig.suptitle("AUC$_{RNX}$ (blue, left) and normalized stress (red, right) vs. alpha, median $\\pm$ IQR over 5 seeds", fontsize=8)
    # Second pass (see _apply_range_labels docstring): the figure is now
    # fully laid out (every title/label set), so each panel's true rendered
    # height is known and the min/max labels can be placed with a
    # GEOMETRICALLY guaranteed gap from the curve, not a fixed guess.
    _apply_range_labels(fig, all_label_specs)
    return fig, pd.concat(csv_pieces, ignore_index=True)


def _make_supplement_figure_pages(
    df: pd.DataFrame, all_datasets: list[str], fig_cfg: dict
) -> list[tuple[plt.Figure, pd.DataFrame]]:
    """Split all datasets into pages (`fig_alpha_curves_all.datasets_per_page`)
    and, for each, return (fig, csv_df) - one page = one PDF, so the figure
    fits within the layout height of the supplement (see K6/fix
    2026-09-14, documentation/2026-09-14_oprava_fig_alpha_curves_all.md:
    the original single figure with all 32 panels in one grid overflowed
    the bottom margin of the page). All size/font parameters come from the
    config (`fig_alpha_curves_all` in config_experiments.yaml), no magic
    numbers in the code."""
    present = [d for d in all_datasets if d in df["dataset"].unique()]
    missing = [d for d in all_datasets if d not in present]

    n_cols = int(fig_cfg["n_cols"])
    datasets_per_page = int(fig_cfg["datasets_per_page"])
    panel_height_in = float(fig_cfg["panel_height_in"])
    title_fontsize = float(fig_cfg["title_fontsize"])
    tick_labelsize = float(fig_cfg["tick_labelsize"])
    ylabel_fontsize = float(fig_cfg["ylabel_fontsize"])
    suptitle_fontsize = float(fig_cfg["suptitle_fontsize"])

    pages = [present[i : i + datasets_per_page] for i in range(0, len(present), datasets_per_page)]
    n_pages = len(pages)
    out: list[tuple[plt.Figure, pd.DataFrame]] = []
    for page_idx, page_datasets in enumerate(pages, start=1):
        n = len(page_datasets)
        n_rows = int(np.ceil(n / n_cols))
        # 2026-09-19 (supplement font-size fix): this figure is embedded
        # ONLY in the supplement (clanek_en/supplement/sections/s2_alpha_curves.tex,
        # [width=\textwidth]) - drawn at WIDTH_SUPPLEMENT_FULL_IN (390pt) so
        # that embed is a no-op scale (see fig_common.py), instead of the
        # DAMI-sized WIDTH_FULL_WIDTH_IN (372pt) that used to leave a small
        # 1.05x LaTeX enlargement on top of already-too-small literal
        # fontsize= values from config_experiments.yaml. constrained_layout
        # (not tight_layout+rect) re-solves panel/title/legend spacing from
        # actual bounding boxes now that the fonts below are bigger.
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(WIDTH_SUPPLEMENT_FULL_IN, panel_height_in * n_rows),
            constrained_layout=True,
        )
        fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.03, wspace=0.06, hspace=0.12)
        axes_flat = np.asarray(axes).reshape(-1)
        csv_pieces = []
        all_label_specs: list[tuple] = []
        for i, dataset_name in enumerate(page_datasets):
            piece, label_specs = _plot_twin_panel(
                axes_flat[i],
                dataset_name,
                df,
                show_ylabel=(i % n_cols == 0),
                title_fontsize=title_fontsize,
                tick_labelsize=tick_labelsize,
                ylabel_fontsize=ylabel_fontsize,
            )
            piece["page"] = page_idx
            csv_pieces.append(piece)
            all_label_specs.extend(label_specs)
        for j in range(n, len(axes_flat)):
            axes_flat[j].axis("off")
        missing_note = f" ({len(missing)} missing: {missing})" if (missing and page_idx == n_pages) else ""
        fig.suptitle(
            f"Supplement page {page_idx}/{n_pages}: AUC$_{{RNX}}$ (blue, left) / "
            f"stress (red, right) vs. alpha, {n} datasets" + missing_note,
            fontsize=suptitle_fontsize,
        )
        # Second pass (see _apply_range_labels docstring): deferred until
        # the whole page (all panels, titles, suptitle) is laid out.
        _apply_range_labels(fig, all_label_specs)
        out.append((fig, pd.concat(csv_pieces, ignore_index=True)))
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="K6: alpha curves (AUC_RNX/stress vs. alpha) - main + supplement figure.")
    add_quick_arg(parser)
    args = parser.parse_args()
    mode = parse_fig_mode(args)

    df = require_experiment_csv(BASE_EXPERIMENT_NAME, mode)
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        raise ValueError(f"No successful runs in {BASE_EXPERIMENT_NAME}_results.csv (mode={mode}).")

    exp_cfg = load_experiments_config()["exp6_alpha_curves"]
    main_datasets: list[str] = exp_cfg.get("main_figure_datasets")
    if not main_datasets:
        raise KeyError("config_experiments.yaml exp6_alpha_curves.main_figure_datasets is missing or empty.")
    all_datasets: list[str] = exp_cfg["datasets"]

    fig_main, csv_main = _make_main_figure(ok, main_datasets)
    save_figure(fig_main, FIG_MAIN)
    save_csv_alongside(csv_main, FIG_MAIN)

    fig_all_cfg = load_experiments_config()["fig_alpha_curves_all"]
    pages = _make_supplement_figure_pages(ok, all_datasets, fig_all_cfg)
    for page_idx, (fig_all, csv_all) in enumerate(pages, start=1):
        page_name = f"{FIG_ALL}_p{page_idx}"
        save_figure(fig_all, page_name)
        save_csv_alongside(csv_all, page_name)

    print(
        f"{FIG_MAIN}: {len(main_datasets)} datasets. "
        f"{FIG_ALL}: {ok['dataset'].nunique()} datasets on {len(pages)} pages "
        f"({FIG_ALL}_p1..p{len(pages)})."
    )


if __name__ == "__main__":
    main()
