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
K4 (documentation/2026-09-12_plan_smeru_clanku.md) - a two-regime map
("when do you need t-SNE"): 2 panels over the E1 datasets (vector/synthetic,
kind='vector' in dataset_properties.csv - K1), x = nn_ratio_k1 (log) in
both panels.

Panel 1: y = auc_rnx(tsne_auto) - auc_rnx(sammon_alpha_auto) (median over
seeds, exp1_dr_benchmark_results.csv) - where t-SNE leads the faithful map
locally, and by how much.
Panel 2: y = the best alpha from {0,1,2} (sammon_alpha0_smacof/sammon_alpha_smacof/
sammon_alpha2_smacof by auc_rnx) - a temporary proxy ahead of K6
(exp6_alpha_curves on a finer grid), see the K4 task spec.

Inputs (BOTH must exist in the same mode - see `--quick/--full/--smoke`):
  results/data/[<mode>/]dataset_properties.csv (K1, src/experiments/dataset_properties.py)
  results/data/[<mode>/]exp1_dr_benchmark_results.csv (E1)
A missing K1 CSV is fail-loud (a clear message with the command to
generate it) - a substitute figure is never fabricated.

Run: venv\\python.exe -m src.figures.fig_regime_map [--quick|--full|--smoke]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.experiments.config_experiments import load_experiments_config
from src.figures.fig_common import (
    OKABE_ITO,
    WIDTH_FULL_WIDTH_IN,
    add_quick_arg,
    mode_data_dir,
    parse_fig_mode,
    require_csv,
    require_experiment_csv,
    save_csv_alongside,
    save_figure,
)

FIG_NAME = "fig_regime_map"
_ALPHA012_METHODS = {"sammon_alpha0_smacof": 0.0, "sammon_alpha_smacof": 1.0, "sammon_alpha2_smacof": 2.0}


def _spearman_report(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    valid = x.notna() & y.notna()
    if valid.sum() < 3:
        return float("nan"), float("nan"), int(valid.sum())
    rho, pvalue = spearmanr(x[valid], y[valid])
    return float(rho), float(pvalue), int(valid.sum())


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="K4: a two-regime map (nn_ratio_k1 vs. t-SNE lead / best alpha).")
    add_quick_arg(parser)
    args = parser.parse_args()
    mode = parse_fig_mode(args)

    props_path = mode_data_dir(mode) / "dataset_properties.csv"
    props = require_csv(props_path, f"venv\\python.exe -m src.experiments.dataset_properties --{mode} (K1)")
    props = props[props["kind"] == "vector"].copy()
    if props.empty:
        raise ValueError(f"{props_path} contains no row with kind=='vector' (E1 vector datasets) - K4 requires dataset_properties.py to be run with the same scope as E1.")

    e1 = require_experiment_csv("exp1_dr_benchmark", mode)
    e1_ok = e1[e1["status"] == "ok"]
    med = e1_ok.groupby(["dataset", "method"])["auc_rnx"].median().unstack("method")

    required_methods = ["tsne_auto", "sammon_alpha_auto"] + list(_ALPHA012_METHODS.keys())
    missing = [m for m in required_methods if m not in med.columns]
    if missing:
        raise KeyError(f"exp1_dr_benchmark_results.csv (mode={mode}) is missing methods {missing} needed for K4.")

    tsne_minus_alpha_auto = med["tsne_auto"] - med["sammon_alpha_auto"]
    alpha012_med = med[list(_ALPHA012_METHODS.keys())]
    best_alpha = alpha012_med.idxmax(axis=1).map(_ALPHA012_METHODS)

    merged = props.set_index("dataset").join(tsne_minus_alpha_auto.rename("tsne_minus_alpha_auto"), how="inner")
    merged = merged.join(best_alpha.rename("best_alpha012"), how="inner")
    if merged.empty:
        raise ValueError("The dataset intersection between dataset_properties.csv and exp1_dr_benchmark_results.csv is empty.")

    exp_cfg = load_experiments_config()
    synthetic_datasets = set(exp_cfg["report"]["synthetic_datasets"])
    merged["is_synthetic"] = merged.index.isin(synthetic_datasets)

    # Q1 step2 (reserse/2026-09-14_specifikace_rozsireni_q1.md, A.9 item 10)
    # - new hold-out candidates are marked with a different marker
    # (circle=core, triangle=hold-out) - source: regime_candidates_screen.csv
    # (present only after `run_screen_regime_candidates.bat`, otherwise all
    # core - no change to the original behavior, no fail-loud for a missing file).
    screen_path = mode_data_dir(mode) / "regime_candidates_screen.csv"
    holdout_datasets: set[str] = set()
    if screen_path.exists():
        screen_df = pd.read_csv(screen_path)
        holdout_datasets = set(screen_df["dataset"])
    merged["is_holdout"] = merged.index.isin(holdout_datasets)

    # S2 tweak (documentation/2026-09-12_kontrola_vysledku_s1.md section 3):
    # labels only for a curated subsample (label_datasets) + deterministic
    # jitter in y for panel 2 (best_alpha012 has only 3 discrete values ->
    # points overlap exactly without jitter) - see config_experiments.yaml
    # fig_regime_map.
    regime_cfg = exp_cfg["fig_regime_map"]
    label_datasets = set(regime_cfg["label_datasets"])
    jitter_rng = np.random.default_rng(int(regime_cfg["jitter_seed"]))
    jitter_std = float(regime_cfg["jitter_std"])
    # jitter is deterministic per-dataset (order = sorted index, NOT
    # insertion order into the DataFrame) - the same run produces the same
    # figure on repeated execution regardless of input CSV processing order.
    sorted_names = sorted(merged.index)
    jitter_values = jitter_rng.normal(0.0, jitter_std, size=len(sorted_names))
    jitter_map = dict(zip(sorted_names, jitter_values))
    merged["best_alpha012_jittered"] = [merged.loc[name, "best_alpha012"] + jitter_map[name] for name in merged.index]

    # --- Spearman rho/p, including a variant without synthetic datasets ----
    stats_rows = []
    for label, subset in [("all", merged), ("real_only", merged[~merged["is_synthetic"]])]:
        rho1, p1, n1 = _spearman_report(subset["nn_ratio_k1"], subset["tsne_minus_alpha_auto"])
        rho2, p2, n2 = _spearman_report(subset["nn_ratio_k1"], subset["best_alpha012"])
        stats_rows.append({"subset": label, "panel": "tsne_minus_alpha_auto", "rho": rho1, "pvalue": p1, "n": n1})
        stats_rows.append({"subset": label, "panel": "best_alpha012", "rho": rho2, "pvalue": p2, "n": n2})
    stats_df = pd.DataFrame(stats_rows)

    # --- figure: 2 panels -----------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(WIDTH_FULL_WIDTH_IN, WIDTH_FULL_WIDTH_IN * 0.45))
    color_synth, color_real = OKABE_ITO[6], OKABE_ITO[5]
    panels = [
        (ax1, "tsne_minus_alpha_auto", "AUC$_{RNX}$(tsne_auto) - AUC$_{RNX}$(alpha_auto)", False),
        (ax2, "best_alpha012_jittered", "best alpha in {0,1,2} (by AUC$_{RNX}$, y jittered)", True),
    ]
    for ax, ycol, ylabel, is_panel2 in panels:
        for is_synth, color, label in [(True, color_synth, "synthetic"), (False, color_real, "real")]:
            sub = merged[merged["is_synthetic"] == is_synth]
            ax.scatter(sub["nn_ratio_k1"], sub[ycol], color=color, label=label, s=18, rasterized=True, zorder=3)
            # Q1 step2 (A.9 item 10): hold-out candidates (regime_candidates_screen.csv)
            # are additionally marked with a black open circle (a ring marker) -
            # without changing the synthetic/real color, just distinguishing dataset origin.
            sub_holdout = sub[sub["is_holdout"]]
            if not sub_holdout.empty:
                ax.scatter(
                    sub_holdout["nn_ratio_k1"], sub_holdout[ycol], facecolors="none", edgecolors="black",
                    linewidths=0.6, s=42, rasterized=True, zorder=4, marker="o",
                )
            # S2 tweak: labels only for label_datasets (a curated
            # subsample); a small alternating vertical offset set reduces
            # overlap even among neighboring labeled points (see config
            # fig_regime_map.label_datasets and kontrola_vysledku_s1.md section 3).
            labeled = sub[sub.index.isin(label_datasets)]
            for k, (name, r) in enumerate(labeled.iterrows()):
                y_off = 5 if k % 2 == 0 else -8
                ax.annotate(
                    str(name), (r["nn_ratio_k1"], r[ycol]), fontsize=5, xytext=(3, y_off),
                    textcoords="offset points", va="bottom" if y_off > 0 else "top",
                )
        ax.set_xscale("log")
        ax.set_xlabel("nn_ratio_k1 (log scale)")
        ax.set_ylabel(ylabel)
        if not is_panel2:
            ax.axhline(0.0, color="grey", linewidth=0.6, linestyle="--")
        else:
            for y_ref in (0, 1, 2):
                ax.axhline(y_ref, color="grey", linewidth=0.4, linestyle=":", zorder=1)
    if merged["is_holdout"].any():
        holdout_handle = plt.Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none",
                                     markeredgecolor="black", markersize=5, label="hold-out (Q1)")
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles=handles + [holdout_handle], fontsize=5.5)
    else:
        ax1.legend(fontsize=5.5)
    rho_p1 = stats_df[(stats_df["subset"] == "all") & (stats_df["panel"] == "tsne_minus_alpha_auto")].iloc[0]
    rho_p2 = stats_df[(stats_df["subset"] == "all") & (stats_df["panel"] == "best_alpha012")].iloc[0]
    ax1.set_title(f"rho={rho_p1['rho']:.2f}, p={rho_p1['pvalue']:.3g}, n={int(rho_p1['n'])}", fontsize=6.5)
    ax2.set_title(f"rho={rho_p2['rho']:.2f}, p={rho_p2['pvalue']:.3g}, n={int(rho_p2['n'])}", fontsize=6.5)
    fig.suptitle("Distance concentration predicts local-vs-global regime (E1)", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, FIG_NAME)

    out_csv = merged.reset_index().rename(columns={"index": "dataset"})
    save_csv_alongside(out_csv, FIG_NAME)

    from src.figures.fig_common import figures_out_dir

    stats_df.to_csv(figures_out_dir() / f"{FIG_NAME}_spearman.csv", index=False)

    print(f"{FIG_NAME}: {merged.shape[0]} datasets, rho(all)={rho_p1['rho']:.3f}/{rho_p2['rho']:.3f}.")


if __name__ == "__main__":
    main()
