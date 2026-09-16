# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-14
# License: see the LICENSE file in the repository root
"""
Q1 step2 (reserse/2026-09-14_specifikace_rozsireni_q1.md, A.9 item 10) - a
paired "slope chart": AUC$_{RNX}$(sammon_alpha0_smacof) -> AUC$_{RNX}$
(sammon_alpha_pred) for every dataset in regime L (low rho_NN,
low_ratio), colored by origin (fitted = the original 9-10 datasets the
rule was FIT on; hold-out = new Q1 candidates, a true out-of-sample
test), shaped by real/synthetic.

Inputs (ALL in the same mode):
  results/data/[<mode>/]exp1_dr_benchmark_results.csv (E1)
  results/data/[<mode>/]dataset_properties.csv (core datasets - regime)
  results/data/[<mode>/]regime_candidates_screen.csv (hold-out eligibility+regime)
  results/data/alpha_pred_rule_frozen_20260913.json (ALWAYS the production path, A.5)

Run: venv\\python.exe -m src.figures.fig_holdout_paired [--quick|--full|--smoke]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experiments.exp1_regime_stratified import classify_regime
from src.experiments.screen_regime_candidates import load_frozen_rule
from src.figures.fig_common import (
    OKABE_ITO,
    WIDTH_SINGLE_COL_IN,
    add_quick_arg,
    mode_data_dir,
    parse_fig_mode,
    require_csv,
    require_experiment_csv,
    save_csv_alongside,
    save_figure,
)

FIG_NAME = "fig_holdout_paired"
PRED = "sammon_alpha_pred"
ALPHA0 = "sammon_alpha0_smacof"


def build_pooled_regime_l_table(mode: str, min_d: int = 3) -> pd.DataFrame:
    """Build a table (dataset, origin['fitted'|'holdout'], is_synthetic,
    auc_alpha0, auc_pred) for all datasets eligible in regime L (core and
    hold-out) with KNOWN values of BOTH methods in E1 - see
    `src.experiments.exp1_holdout_confirmatory` (same eligibility)."""
    from src.experiments.config_experiments import load_experiments_config

    e1 = require_experiment_csv("exp1_dr_benchmark", mode)
    e1_ok = e1[e1["status"] == "ok"]
    med = e1_ok.groupby(["dataset", "method"])["auc_rnx"].median().unstack("method")
    if PRED not in med.columns or ALPHA0 not in med.columns:
        raise KeyError(f"exp1_dr_benchmark_results.csv (mode={mode}) is missing methods {PRED}/{ALPHA0}.")

    props = require_csv(mode_data_dir(mode) / "dataset_properties.csv", f"venv\\python.exe -m src.experiments.dataset_properties --{mode}")
    # NOTE: dataset_properties.csv contains core AND hold-out rows together
    # (--datasets holdout only APPENDS to the same file) - "core" must be
    # explicitly restricted to exp1_dr_benchmark.datasets, see
    # src.experiments.exp1_holdout_confirmatory (same bug/fix).
    core_dataset_names = set(load_experiments_config()["exp1_dr_benchmark"]["datasets"])
    props_vec = props[(props["kind"] == "vector") & (props["d"] >= min_d) & (props["dataset"].isin(core_dataset_names))].copy()
    rule = load_frozen_rule()
    props_vec["regime"] = classify_regime(props_vec["nn_ratio_k1"], rule)
    core_regime_l = set(props_vec.loc[props_vec["regime"] == "low_ratio", "dataset"])
    synthetic_core = set(load_experiments_config()["report"]["synthetic_datasets"])

    screen_path = mode_data_dir(mode) / "regime_candidates_screen.csv"
    holdout_regime_l: set[str] = set()
    holdout_is_synth: dict[str, bool] = {}
    if screen_path.exists():
        screen = pd.read_csv(screen_path)
        eligible = screen[screen["eligible_all"] == True]  # noqa: E712
        holdout_regime_l = set(eligible.loc[eligible["regime_frozen"] == "low_ratio", "dataset"])
        holdout_is_synth = dict(zip(screen["dataset"], ~screen["requires_download"].astype(bool)))

    rows = []
    for ds in sorted(core_regime_l | holdout_regime_l):
        if ds not in med.index:
            continue
        a0, ap = med.loc[ds, ALPHA0], med.loc[ds, PRED]
        if pd.isna(a0) or pd.isna(ap):
            continue
        origin = "holdout" if ds in holdout_regime_l else "fitted"
        is_synth = holdout_is_synth.get(ds, ds in synthetic_core) if origin == "holdout" else ds in synthetic_core
        rows.append({"dataset": ds, "origin": origin, "is_synthetic": bool(is_synth), "auc_alpha0": float(a0), "auc_pred": float(ap)})
    return pd.DataFrame(rows)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Q1 step2: paired slope chart AUC(alpha0)->AUC(alpha_pred), regime L, fitted vs. hold-out.")
    add_quick_arg(parser)
    args = parser.parse_args()
    mode = parse_fig_mode(args)

    table = build_pooled_regime_l_table(mode)
    if table.empty:
        raise ValueError(f"No regime L dataset with valid {ALPHA0}/{PRED} values in E1 (mode={mode}).")

    color_fitted, color_holdout = OKABE_ITO[5], OKABE_ITO[6]
    marker_real, marker_synth = "o", "^"

    fig, ax = plt.subplots(figsize=(WIDTH_SINGLE_COL_IN, WIDTH_SINGLE_COL_IN * 1.05))
    for _, row in table.iterrows():
        color = color_fitted if row["origin"] == "fitted" else color_holdout
        marker = marker_synth if row["is_synthetic"] else marker_real
        ax.plot([0, 1], [row["auc_alpha0"], row["auc_pred"]], color=color, linewidth=0.7, alpha=0.7, zorder=2, rasterized=True)
        ax.scatter([0, 1], [row["auc_alpha0"], row["auc_pred"]], color=color, marker=marker, s=16, zorder=3, rasterized=True)

    ax.set_xlim(-0.15, 1.15)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["alpha=0\n(MDS)", "alpha_pred"])
    ax.set_ylabel("AUC$_{RNX}$ (median over seeds)")
    ax.set_title(f"Regime L (low $\\rho_{{NN}}$): fitted (n={int((table['origin']=='fitted').sum())}) "
                 f"vs. hold-out (n={int((table['origin']=='holdout').sum())})", fontsize=7)

    legend_handles = [
        plt.Line2D([0], [0], color=color_fitted, marker="o", linestyle="-", markersize=4, label="fitted (core)"),
        plt.Line2D([0], [0], color=color_holdout, marker="o", linestyle="-", markersize=4, label="hold-out (Q1)"),
        plt.Line2D([0], [0], color="grey", marker=marker_real, linestyle="none", markersize=4, label="real dataset"),
        plt.Line2D([0], [0], color="grey", marker=marker_synth, linestyle="none", markersize=4, label="synthetic dataset"),
    ]
    ax.legend(handles=legend_handles, fontsize=5.5, loc="lower right")
    fig.tight_layout()
    save_figure(fig, FIG_NAME)
    save_csv_alongside(table, FIG_NAME)

    n_gain = int((table["auc_pred"] > table["auc_alpha0"]).sum())
    print(f"{FIG_NAME}: {table.shape[0]} regime L datasets ({int((table['origin']=='holdout').sum())} hold-out), {n_gain} with improvement pred>alpha0.")


if __name__ == "__main__":
    main()
