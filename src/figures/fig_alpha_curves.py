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
    add_quick_arg,
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
) -> pd.DataFrame:
    """Plot a single panel (auc_rnx on the left axis, stress on the right
    axis, twinx) for a given dataset. Returns a long DataFrame (dataset,
    alpha, metric, median, q25, q75, n_seeds) to save to CSV."""
    auc = _median_iqr_by_alpha(df, dataset_name, "auc_rnx")
    stress = _median_iqr_by_alpha(df, dataset_name, "stress_scale_invariant")

    ax_auc.plot(auc["alpha"], auc["median"], color=_COLOR_AUC, marker="o", markersize=2.5, linewidth=1.1, zorder=3)
    ax_auc.fill_between(auc["alpha"], auc["q25"], auc["q75"], color=_COLOR_AUC, alpha=0.2, linewidth=0, zorder=2, rasterized=True)
    ax_auc.tick_params(axis="y", labelcolor=_COLOR_AUC, labelsize=tick_labelsize)
    ax_auc.tick_params(axis="x", labelsize=tick_labelsize)

    ax_stress = ax_auc.twinx()
    ax_stress.plot(stress["alpha"], stress["median"], color=_COLOR_STRESS, marker="s", markersize=2.2, linewidth=1.0, linestyle="--", zorder=3)
    ax_stress.fill_between(stress["alpha"], stress["q25"], stress["q75"], color=_COLOR_STRESS, alpha=0.15, linewidth=0, zorder=1, rasterized=True)
    ax_stress.tick_params(axis="y", labelcolor=_COLOR_STRESS, labelsize=tick_labelsize)

    ax_auc.set_title(dataset_name, fontsize=title_fontsize)
    if show_ylabel:
        ax_auc.set_ylabel("AUC$_{RNX}$", color=_COLOR_AUC, fontsize=ylabel_fontsize)

    auc["dataset"] = dataset_name
    auc["metric"] = "auc_rnx"
    stress["dataset"] = dataset_name
    stress["metric"] = "stress_scale_invariant"
    return pd.concat([auc, stress], ignore_index=True)


def _make_main_figure(df: pd.DataFrame, main_datasets: list[str]) -> tuple[plt.Figure, pd.DataFrame]:
    missing = [d for d in main_datasets if d not in df["dataset"].unique()]
    if missing:
        raise KeyError(f"exp6_alpha_curves_results.csv is missing datasets {missing} from main_figure_datasets.")

    n = len(main_datasets)
    fig, axes = plt.subplots(1, n, figsize=(WIDTH_FULL_WIDTH_IN, WIDTH_FULL_WIDTH_IN * 0.24), sharex=True)
    if n == 1:
        axes = [axes]
    csv_pieces = []
    for i, (ax, dataset_name) in enumerate(zip(axes, main_datasets)):
        piece = _plot_twin_panel(ax, dataset_name, df, show_ylabel=(i == 0))
        csv_pieces.append(piece)
        ax.set_xlabel("alpha", fontsize=6.5)
    fig.suptitle("AUC$_{RNX}$ (blue, left) and normalized stress (red, right) vs. alpha, median $\\pm$ IQR over 5 seeds", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
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
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(WIDTH_FULL_WIDTH_IN, panel_height_in * n_rows))
        axes_flat = np.asarray(axes).reshape(-1)
        csv_pieces = []
        for i, dataset_name in enumerate(page_datasets):
            piece = _plot_twin_panel(
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
        for j in range(n, len(axes_flat)):
            axes_flat[j].axis("off")
        missing_note = f" ({len(missing)} missing: {missing})" if (missing and page_idx == n_pages) else ""
        fig.suptitle(
            f"Supplement page {page_idx}/{n_pages}: AUC$_{{RNX}}$ (blue, left) / "
            f"stress (red, right) vs. alpha, {n} datasets" + missing_note,
            fontsize=suptitle_fontsize,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
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
