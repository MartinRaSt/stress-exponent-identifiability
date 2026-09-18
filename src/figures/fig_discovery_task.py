# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-18
# License: see the LICENSE file in the repository root
"""
E15 downstream discovery task figure (author request 2026-09-18, motivated
by a DAMI editor objection: "there is not a single task in the whole
manuscript where a better map leads to a better FINDING"). Two panels
(Q1 `nearest_class_pair`, Q2 `most_dispersed_class`) show the per-method
ERROR RATE (share of the `report.main_methods` datasets with a wrong hard
answer, see `src/experiments/exp15_discovery_task_stats.py::build_summary`)
as a sorted horizontal bar chart - error rate (not accuracy) is plotted
because the whole point of this figure is honesty about the RESIDUAL
error, which a "share correct" framing would visually bury near 1.0.

Input (must already exist for the given mode - this script NEVER recomputes
E15, only plots its already-generated tables, same convention as
fig_pareto_front.py/fig_neighbor_survival.py):
  results/tables/[<mode>/]exp15_discovery_task_summary.csv
  results/tables/[<mode>/]exp15_discovery_task_pairwise.csv
(both from `src.experiments.exp15_discovery_task_stats`, itself reading
exp15_discovery_task_results.csv from `src.experiments.exp15_discovery_task`).

Honesty requirements (author 2026-09-18, after two referee "overclaiming"
catches on this project - see CLAUDE.md "nic bez zdroje"):
  (a) the bars for the alpha-Sammon/MDS/PCA ("stress") family sit at
      32-51% error, NEVER near zero - every bar carries an on-plot percent
      label so this is impossible to miss even at a glance.
  (b) within the stress family, sammon_alpha_pred (this paper's rule,
      black bar edge, labelled "reference") is NOT significantly different
      from any other stress-family member (McNemar, Holm-corrected within
      each question's family of `report.main_methods` baselines - see
      exp15_discovery_task_stats.py) - drawn as an ABSENCE of the
      significance marker '*' on every stress-family bar. Only the
      neighbor-embedding family (t-SNE/UMAP/PaCMAP/TriMap/PHATE, mostly
      NOT densMAP) is significantly worse. The figure never singles out
      sammon_alpha_pred as the best bar in its own family (no "winner"
      color/marker beyond the neutral "this is the tested reference" edge)
      - the improvement this figure documents belongs to the STRESS
      FAMILY as a whole (vs. neighbor-embedding methods), not to the
      alpha_pred rule specifically.

Bar color = family membership (`report.neighbor_methods` vs. everything
else in `report.main_methods`, the same split already used by
fig_metric_correlations.py's "neighbor"/"stress" panels) - Okabe-Ito
colorblind-safe blue/vermillion. Method order is INDEPENDENT per panel
(sorted by that panel's own error rate, best/lowest error at the top).

Output: results/figures/[<mode>/]fig_discovery_task.pdf (vector PDF,
pdf.fonttype=42, full-page width) + a .csv with one row per
(question, method): family, error_rate, n_correct, n_datasets, is_reference,
p_mcnemar_holm_vs_reference, reject_mcnemar_holm_vs_reference.

Run: venv\\python.exe -m src.figures.fig_discovery_task [--quick|--full|--smoke]
or: src\\run_fig_discovery_task.bat [quick|full|smoke]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from src.common.config import get_mode_path
from src.experiments.config_experiments import load_experiments_config
from src.figures.fig_common import (
    OKABE_ITO,
    WIDTH_FULL_WIDTH_IN,
    add_quick_arg,
    display_label,
    parse_fig_mode,
    require_csv,
    save_csv_alongside,
    save_figure,
)
from src.sammon.discovery_task import DISCOVERY_QUESTIONS

FIG_NAME = "fig_discovery_task"

# Family colors (Okabe-Ito), same indices as fig_alpha_gain_by_regime.py's
# regime palette - purely categorical, not a globally reserved meaning.
_STRESS_COLOR = OKABE_ITO[5]    # blue "#0072B2": mds/pca/sammon_alpha* (report.main_methods minus report.neighbor_methods)
_NEIGHBOR_COLOR = OKABE_ITO[6]  # vermillion "#D55E00": report.neighbor_methods (t-SNE/UMAP/PaCMAP/TriMap/densMAP/PHATE)
_REFERENCE_EDGE_COLOR = "black"
_REFERENCE_EDGE_WIDTH = 1.3
_BAR_HEIGHT = 0.62


def build_plot_table(summary: pd.DataFrame, pairwise: pd.DataFrame, main_methods: list[str], neighbor_methods: set, reference_method: str) -> pd.DataFrame:
    """One row per (question, method) in `main_methods`, in
    `DISCOVERY_QUESTIONS` order: error_rate/n_correct/n_datasets from
    `summary`, family from `neighbor_methods` membership, and the
    Holm-adjusted McNemar p-value/rejection of `method` vs
    `reference_method` from `pairwise` (NaN/False for the reference row
    itself and for any (question, method) pair missing from `pairwise`,
    e.g. below exp15_discovery_task_stats.min_paired_datasets - never a
    fabricated significance flag). Fail loud if `summary` is missing a
    (question, method) combination for any of `main_methods`."""
    rows: list[dict] = []
    for question in DISCOVERY_QUESTIONS:
        for method in main_methods:
            srow = summary[(summary["question"] == question) & (summary["method"] == method)]
            if srow.empty:
                raise ValueError(
                    f"exp15_discovery_task_summary.csv has no row for question='{question}', method='{method}' "
                    "(check report.main_methods against the E15 run)."
                )
            r = srow.iloc[0]
            p_holm = float("nan")
            reject = False
            if method != reference_method:
                prow = pairwise[(pairwise["question"] == question) & (pairwise["method_a"] == reference_method) & (pairwise["method_b"] == method)]
                if not prow.empty:
                    p_holm = float(prow.iloc[0]["p_mcnemar_holm"])
                    reject = bool(prow.iloc[0]["reject_mcnemar_holm"])
            rows.append({
                "question": question, "method": method,
                "family": "neighbor" if method in neighbor_methods else "stress",
                "is_reference": method == reference_method,
                "n_datasets": int(r["n_datasets"]), "error_rate": float(r["error_rate"]),
                "n_correct": int(round(float(r["n_datasets"]) * (1.0 - float(r["error_rate"])))),
                "p_mcnemar_holm_vs_reference": p_holm, "reject_mcnemar_holm_vs_reference": reject,
            })
    return pd.DataFrame(rows)


def _draw_panel(ax, sub: pd.DataFrame, reference_method: str) -> None:
    sub = sub.sort_values("error_rate", ascending=True).reset_index(drop=True)
    y = np.arange(len(sub))
    colors = [_NEIGHBOR_COLOR if fam == "neighbor" else _STRESS_COLOR for fam in sub["family"]]
    edgecolors = [_REFERENCE_EDGE_COLOR if ref else "none" for ref in sub["is_reference"]]
    linewidths = [_REFERENCE_EDGE_WIDTH if ref else 0.0 for ref in sub["is_reference"]]
    ax.barh(y, sub["error_rate"], height=_BAR_HEIGHT, color=colors, edgecolor=edgecolors, linewidth=linewidths, rasterized=True, zorder=3)

    labels = [display_label(m, kind="method") + (" (reference)" if m == reference_method else "") for m in sub["method"]]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_ylim(-0.6, len(sub) - 0.4)
    ax.invert_yaxis()

    for yi, row in sub.iterrows():
        label = f"{100.0 * row['error_rate']:.0f}%"
        if row["reject_mcnemar_holm_vs_reference"]:
            label += " *"
        ax.text(row["error_rate"] + 0.015, yi, label, va="center", ha="left", fontsize=6, zorder=4)

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Error rate (wrong-answer share)", fontsize=6.5)
    ax.tick_params(axis="x", labelsize=6)
    ax.grid(axis="x", color="#dddddd", linewidth=0.5, zorder=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="E15 discovery-task error rate by method, both questions, with Holm-adjusted significance vs. the reference method.")
    add_quick_arg(parser)
    args = parser.parse_args()
    mode = parse_fig_mode(args)

    exp_cfg = load_experiments_config()
    main_methods: list[str] = exp_cfg["report"]["main_methods"]
    neighbor_methods = set(exp_cfg["report"]["neighbor_methods"])
    reference_method: str = exp_cfg["exp15_discovery_task_stats"]["reference_method"]
    alpha: float = float(exp_cfg["exp15_discovery_task_stats"]["alpha"])
    if reference_method not in main_methods:
        raise KeyError(f"exp15_discovery_task_stats.reference_method='{reference_method}' is not in report.main_methods.")

    tables_dir = get_mode_path("results_tables_dir", mode)
    how_to = f"venv\\python.exe -m src.experiments.exp15_discovery_task_stats --{mode} (after src.experiments.exp15_discovery_task --{mode})"
    summary = require_csv(tables_dir / "exp15_discovery_task_summary.csv", how_to)
    pairwise = require_csv(tables_dir / "exp15_discovery_task_pairwise.csv", how_to)

    plot_df = build_plot_table(summary, pairwise, main_methods, neighbor_methods, reference_method)

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_FULL_WIDTH_IN, WIDTH_FULL_WIDTH_IN * 0.70))
    for ax, question in zip(axes, DISCOVERY_QUESTIONS):
        sub = plot_df[plot_df["question"] == question]
        _draw_panel(ax, sub, reference_method)
        ax.set_title(display_label(question, kind="question"), fontsize=8)

    legend_handles = [
        Patch(facecolor=_STRESS_COLOR, edgecolor="none", label="Distance-preserving methods (MDS / PCA / $\\alpha$-Sammon variants)"),
        Patch(facecolor=_NEIGHBOR_COLOR, edgecolor="none", label="Neighbor-embedding methods (t-SNE / UMAP / PaCMAP / TriMap / densMAP / PHATE)"),
        Patch(facecolor="white", edgecolor=_REFERENCE_EDGE_COLOR, linewidth=_REFERENCE_EDGE_WIDTH, label=display_label(reference_method, kind="method") + " (reference for the significance test)"),
        Patch(facecolor="white", edgecolor="white", label=f"* significant vs. reference (McNemar, Holm-adjusted $p<{alpha:g}$)"),
    ]
    fig.suptitle("E15 discovery task: error rate never reaches zero, stress-family bars are mutually indistinguishable", fontsize=7.5, y=0.98)
    fig.subplots_adjust(top=0.84, bottom=0.30, left=0.24, right=0.97, wspace=0.75)
    fig.legend(handles=legend_handles, loc="lower center", ncol=1, fontsize=6, frameon=False, bbox_to_anchor=(0.5, 0.05))
    save_figure(fig, FIG_NAME)
    save_csv_alongside(plot_df, FIG_NAME)

    n_sig = int(plot_df.loc[~plot_df["is_reference"], "reject_mcnemar_holm_vs_reference"].sum())
    n_baselines = int((~plot_df["is_reference"]).sum())
    print(f"{FIG_NAME}: {plot_df['method'].nunique()} methods x {len(DISCOVERY_QUESTIONS)} questions, {n_sig}/{n_baselines} baseline rows significant vs. reference -> results/figures/[{mode}/]{FIG_NAME}.pdf")


if __name__ == "__main__":
    main()
