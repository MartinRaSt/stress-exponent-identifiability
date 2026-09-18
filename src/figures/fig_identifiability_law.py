# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-17
# License: see the LICENSE file in the repository root
"""
Veta 3 (local quadratic law, reserse/2026-09-17_zostreni_propozice2.md
section 4/10, analysis item 5): does the identifiability gap
`Delta_alpha = sigma_0(Y_alpha*)/sigma_0(Y_0*) - 1` follow the predicted
local quadratic law `alpha^2 * I0_exact` (`pred_quadratic` column) along the
smooth branch of minimizers near Y_0*?

Delta (y-axis) vs. alpha^2*I0_exact = pred_quadratic (x-axis), log-log, one
point per (dataset, alpha>0) row with hessian_skipped==False and both
values > 0 (log undefined otherwise - see `_finite_positive` below, NOT
silently clipped to a floor). A reference line of slope 1 (y=x, i.e. exact
agreement Delta==pred_quadratic) is drawn; points scatter around it while
Veta 3 (OSNOVA - the O(alpha^3) remainder is not quantified, reserse
section 4.3) holds, and drift away from it once alpha leaves the local
regime. Color = alpha (categorical, Okabe-Ito).

Input: results/data/[<mode>/]exp10_identifiability_check_results.csv
(src/experiments/exp10_identifiability_check.py). A missing input is
fail-loud (`require_experiment_csv`).

Output: results/figures/[<mode>/]fig_identifiability_law.pdf + .csv (the
exact rows plotted: dataset, alpha, Delta, pred_quadratic, rho_nn,
quadratic_law_holds).

Run: venv\\python.exe -m src.figures.fig_identifiability_law [--quick|--full|--smoke]
or: src\\run_fig_identifiability_law.bat [quick|full|smoke]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from src.experiments.config_experiments import load_experiments_config
from src.figures.fig_common import (
    WIDTH_SINGLE_COL_IN,
    add_quick_arg,
    parse_fig_mode,
    require_experiment_csv,
    save_csv_alongside,
    save_figure,
)

FIG_NAME = "fig_identifiability_law"
BASE_EXPERIMENT_NAME = "exp10_identifiability_check"


def _finite_positive(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Rows with every column in `cols` finite AND strictly positive (needed
    for the log-log axes) - rows failing this (e.g. Delta=0 at a degenerate
    minimizer, or hessian_skipped) are DROPPED, not clipped to a floor
    value (no fabricated data point)."""
    mask = np.ones(df.shape[0], dtype=bool)
    for c in cols:
        v = df[c].to_numpy(dtype=np.float64)
        mask &= np.isfinite(v) & (v > 0.0)
    return df[mask]


def build_plot_table(df_ok: pd.DataFrame) -> pd.DataFrame:
    sub = df_ok[(df_ok["alpha"] > 0.0) & (df_ok["hessian_skipped"] == False)].copy()  # noqa: E712
    sub = _finite_positive(sub, ["Delta", "pred_quadratic"])
    if sub.empty:
        raise ValueError("No (dataset, alpha>0) row has a finite positive (Delta, pred_quadratic) pair - nothing to plot.")
    cols = ["dataset", "alpha", "Delta", "pred_quadratic", "rho_nn", "quadratic_law_holds"]
    return sub[cols].sort_values(["dataset", "alpha"]).reset_index(drop=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Veta 3: Delta vs. alpha^2*I0_exact (local quadratic law), log-log with a reference slope-1 line.")
    add_quick_arg(parser)
    args = parser.parse_args()
    mode = parse_fig_mode(args)

    exp10 = require_experiment_csv(BASE_EXPERIMENT_NAME, mode)
    exp10_ok = exp10[exp10["status"] == "ok"].copy()
    if exp10_ok.empty:
        raise ValueError(f"exp10_identifiability_check_results.csv (mode={mode}) contains no rows with status 'ok'.")

    table = build_plot_table(exp10_ok)

    # 2026-09-17 fix (author: legend with one entry per (alpha, holds/fails)
    # combination - 20+ entries - covered the left half of the plot).
    # color = alpha now goes through ONE continuous colorbar (Normalize +
    # a sequential colormap) instead of a discrete legend entry per alpha
    # value; marker fill (filled=holds, open=fails) is the only remaining
    # visual encoding, so the legend needs at most 3 entries.
    fig_cfg = load_experiments_config()["fig_identifiability_law"]
    cmap = plt.get_cmap(str(fig_cfg["colormap"]))
    legend_loc = str(fig_cfg["legend_loc"])
    norm = Normalize(vmin=float(table["alpha"].min()), vmax=float(table["alpha"].max()))

    holds = table[table["quadratic_law_holds"] == True]  # noqa: E712
    fails = table[table["quadratic_law_holds"] == False]  # noqa: E712

    fig, ax = plt.subplots(figsize=(WIDTH_SINGLE_COL_IN, WIDTH_SINGLE_COL_IN * 0.95))
    if not holds.empty:
        rgba_holds = cmap(norm(holds["alpha"].to_numpy(dtype=float)))
        ax.scatter(holds["pred_quadratic"], holds["Delta"], color=rgba_holds, marker="o", s=14, rasterized=True, zorder=3)
    if not fails.empty:
        rgba_fails = cmap(norm(fails["alpha"].to_numpy(dtype=float)))
        ax.scatter(fails["pred_quadratic"], fails["Delta"], facecolors="none", edgecolors=rgba_fails, marker="o", s=14, linewidths=0.7, rasterized=True, zorder=3)

    lo = float(min(table["pred_quadratic"].min(), table["Delta"].min()))
    hi = float(max(table["pred_quadratic"].max(), table["Delta"].max()))
    ax.plot([lo, hi], [lo, hi], color="grey", linewidth=0.8, linestyle="--", zorder=1)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo * 0.8, hi * 1.25)
    ax.set_ylim(lo * 0.8, hi * 1.25)
    ax.set_xlabel(r"predicted: $\alpha^2 I_0$ (exact Hessian)")
    ax.set_ylabel(r"observed: $\Delta_\alpha$")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=r"$\alpha$", fraction=0.046, pad=0.04)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="0.35", markeredgecolor="0.35", markersize=5, label=f"law holds (n={holds.shape[0]})"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="none", markeredgecolor="0.35", markersize=5, label=f"law fails (n={fails.shape[0]})"),
        Line2D([0], [0], color="grey", linewidth=0.8, linestyle="--", label="slope 1"),
    ]
    ax.legend(handles=legend_handles, fontsize=5.5, loc=legend_loc, framealpha=0.9, borderpad=0.6)
    fig.suptitle("Local quadratic law: predicted vs. observed identifiability gap", fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, FIG_NAME)
    save_csv_alongside(table, FIG_NAME)

    frac_holds = float((table["quadratic_law_holds"] == True).mean())  # noqa: E712
    print(f"{FIG_NAME}: {table.shape[0]} (dataset,alpha) points, fraction quadratic_law_holds={frac_holds:.3f} -> results/figures/[{mode}/]{FIG_NAME}.pdf")


if __name__ == "__main__":
    main()
