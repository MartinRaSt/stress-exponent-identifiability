# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-13
# License: see the LICENSE file in the repository root
"""
A figure for the stratified E1/E6 analysis by distance-concentration regime
(documentation/2026-09-13_e1_stratifikace_rezimu.md) - the basis for the
claim "alpha only matters in the concentrated-distance regime, and the
rule detects this regime". Complements `src/experiments/exp1_regime_stratified.py`
(a table over E1) with a median curve auc_rnx(alpha) from
exp6_alpha_curves_results.csv (K6, a fine alpha grid), normalized to the
value at alpha=0 (= gain over MDS), separately for each of the 3 regimes
(dataset -> regime via `src.experiments.exp1_regime_stratified.classify_regime`,
thresholds from `results/data/alpha_pred_rule.json`, ALWAYS the production
root - same as the classification used in the E1 table).

Procedure:
  1. auc_median(dataset, alpha) = median over seeds (exp6, status=='ok').
  2. gain(dataset, alpha) = auc_median(dataset, alpha) / auc_median(dataset, 0)
     (a dataset without a valid value at alpha=0 is excluded - fail-loud
     log, not a silent fallback).
  3. For each regime and each point of the alpha grid: median + IQR
     (q25/q75) of gain over the datasets in the regime.
  4. Figure: 1 curve per regime (median, IQR band), a vertical dashed line
     at the alpha predicted by the rule for the given regime (a_low/a_mid/a_high
     from the JSON).

Inputs (dataset_properties.csv and exp6_alpha_curves_results.csv MUST be in
the SAME mode --quick/--full/--smoke; alpha_pred_rule.json always from the
production root):
  results/data/[<mode>/]dataset_properties.csv
  results/data/[<mode>/]exp6_alpha_curves_results.csv
  results/data/alpha_pred_rule.json
A missing input is fail-loud - no figure is fabricated.

Output: results/figures/[<mode>/]fig_alpha_gain_by_regime.pdf (vector PDF,
pdf.fonttype=42, width ~\\columnwidth) + a .csv with the same name
(underlying data: regime, alpha, n_datasets, median_gain, q25_gain, q75_gain).

Run: venv\\python.exe -m src.figures.fig_alpha_gain_by_regime [--quick|--full|--smoke]
or: src\\run_fig_alpha_gain_by_regime.bat [quick|full|smoke]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experiments.exp1_regime_stratified import REGIME_ORDER, build_dataset_regime_table
from src.figures.fig_common import (
    OKABE_ITO,
    WIDTH_SINGLE_COL_IN,
    add_quick_arg,
    mode_data_dir,
    parse_fig_mode,
    require_csv,
    save_csv_alongside,
    save_figure,
)
from src.sammon.alpha_predict import load_alpha_pred_rule

FIG_NAME = "fig_alpha_gain_by_regime"
BASE_EXPERIMENT_NAME = "exp6_alpha_curves"

# Neutral band names rho_NN = nn_ratio_k1 (author decision 2026-09-14 - the
# original "concentrated"/"moderate"/"diffuse" were inverted relative to
# distance-concentration theory - high rho_NN = concentrated distances
# (Beyer et al.; Corollary in clanek/sections/03_metoda.tex), see
# documentation/2026-09-14_prejmenovani_rezimu.md).
_REGIME_COLOR = {"low_ratio": OKABE_ITO[6], "mid_ratio": OKABE_ITO[1], "high_ratio": OKABE_ITO[5]}
_REGIME_MARKER = {"low_ratio": "o", "mid_ratio": "s", "high_ratio": "^"}
_REGIME_LABEL = {
    "low_ratio": "low $\\rho_{NN}$ (least concentrated)",
    "mid_ratio": "mid $\\rho_{NN}$",
    "high_ratio": "high $\\rho_{NN}$ (most concentrated)",
}


def compute_gain_curves(exp6_ok: pd.DataFrame, dataset_regime: pd.DataFrame, logger=None) -> pd.DataFrame:
    """Return a long table (regime, alpha, n_datasets, median_gain, q25_gain,
    q75_gain) - see the module docstring, steps 1-3. `exp6_ok` = only the
    'status'=='ok' rows of exp6_alpha_curves_results.csv. `dataset_regime` =
    the output of `build_dataset_regime_table` (columns dataset/regime),
    already restricted to datasets present in `exp6_ok`."""
    auc_med = exp6_ok.groupby(["dataset", "alpha"])["auc_rnx"].median().reset_index()
    wide = auc_med.pivot(index="dataset", columns="alpha", values="auc_rnx")
    if 0.0 not in wide.columns:
        raise KeyError("exp6_alpha_curves_results.csv is missing alpha==0.0 in the grid - 'gain vs. MDS' normalization is not possible.")

    denom = wide[0.0]
    valid_denom = denom[(denom.notna()) & (denom != 0.0)]
    n_dropped = wide.shape[0] - valid_denom.shape[0]
    if n_dropped > 0 and logger is not None:
        logger.warning("%d datasets excluded from the gain curve (missing/zero auc_rnx at alpha=0).", n_dropped)
    gain = wide.loc[valid_denom.index].div(valid_denom, axis=0)

    gain_long = gain.reset_index().melt(id_vars="dataset", var_name="alpha", value_name="gain")
    gain_long = gain_long.merge(dataset_regime[["dataset", "regime"]], on="dataset", how="inner")

    rows = []
    for regime in REGIME_ORDER:
        sub = gain_long[gain_long["regime"] == regime]
        for alpha_value, g in sub.groupby("alpha"):
            values = g["gain"].dropna()
            if values.empty:
                continue
            rows.append({
                "regime": regime, "alpha": float(alpha_value), "n_datasets": int(values.shape[0]),
                "median_gain": float(values.median()), "q25_gain": float(values.quantile(0.25)), "q75_gain": float(values.quantile(0.75)),
            })
    out = pd.DataFrame(rows).sort_values(["regime", "alpha"]).reset_index(drop=True)
    if out.empty:
        raise ValueError("The resulting gain curve is empty - check the dataset intersection between exp6 and dataset_properties.csv.")
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Median curve auc_rnx(alpha)/auc_rnx(0) by distance-concentration regime (K6 x alpha_pred_rule).")
    add_quick_arg(parser)
    args = parser.parse_args()
    mode = parse_fig_mode(args)

    from src.common.logging_utils import get_logger

    logger = get_logger(FIG_NAME, mode=mode)

    props_path = mode_data_dir(mode) / "dataset_properties.csv"
    props = require_csv(props_path, f"venv\\python.exe -m src.experiments.dataset_properties --{mode} (K1)")
    props = props[props["kind"] == "vector"]
    if props.empty:
        raise ValueError(f"{props_path} contains no row with kind=='vector'.")

    rule = load_alpha_pred_rule()  # always the production root - see the module docstring
    dataset_regime = build_dataset_regime_table(props, rule)
    alpha_by_regime = {
        "low_ratio": float(rule["coefficients"]["a_low"]),
        "mid_ratio": float(rule["coefficients"]["a_mid"]),
        "high_ratio": float(rule["coefficients"]["a_high"]),
    }

    from src.figures.fig_common import require_experiment_csv

    exp6 = require_experiment_csv(BASE_EXPERIMENT_NAME, mode)
    exp6_ok = exp6[(exp6["status"] == "ok") & exp6["auc_rnx"].notna()]
    if exp6_ok.empty:
        raise ValueError(f"exp6_alpha_curves_results.csv (mode={mode}) contains no 'ok' rows with auc_rnx.")

    missing_exp6 = set(exp6_ok["dataset"].unique()) - set(dataset_regime["dataset"].unique())
    if missing_exp6:
        raise ValueError(
            f"{len(missing_exp6)} datasets from exp6_alpha_curves_results.csv have no kind=='vector' record in "
            f"dataset_properties.csv (mode={mode}): {sorted(missing_exp6)}."
        )

    curves = compute_gain_curves(exp6_ok, dataset_regime, logger=logger)

    fig, ax = plt.subplots(figsize=(WIDTH_SINGLE_COL_IN, WIDTH_SINGLE_COL_IN * 0.82))
    for regime in REGIME_ORDER:
        sub = curves[curves["regime"] == regime]
        if sub.empty:
            continue
        color = _REGIME_COLOR[regime]
        n_datasets = int(sub["n_datasets"].max())
        ax.plot(
            sub["alpha"], sub["median_gain"], color=color, marker=_REGIME_MARKER[regime], markersize=3.0, linewidth=1.2,
            label=f"{_REGIME_LABEL[regime]} (n={n_datasets})", zorder=3,
        )
        ax.fill_between(sub["alpha"], sub["q25_gain"], sub["q75_gain"], color=color, alpha=0.18, linewidth=0, zorder=2, rasterized=True)
        ax.axvline(alpha_by_regime[regime], color=color, linewidth=1.0, linestyle="--", zorder=1)

    ax.axhline(1.0, color="grey", linewidth=0.6, linestyle=":", zorder=1)
    ax.set_xlabel("alpha")
    ax.set_ylabel("AUC$_{RNX}(\\alpha)$ / AUC$_{RNX}(0)$ (median, IQR band)")
    ax.set_title(
        "Alpha gain over MDS by nearest-neighbor\ndistance ratio $\\rho_{NN}$ (high $\\rho_{NN}$ = concentrated distances)",
        fontsize=7,
    )
    ax.legend(fontsize=6, loc="best")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    save_figure(fig, FIG_NAME)
    save_csv_alongside(curves, FIG_NAME)

    print(f"{FIG_NAME}: {curves['regime'].nunique()} regimes, {curves.shape[0]} curve rows -> results/figures/[{mode}/]{FIG_NAME}.pdf")


if __name__ == "__main__":
    main()
