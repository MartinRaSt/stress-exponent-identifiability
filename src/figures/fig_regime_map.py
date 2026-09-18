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
Panel 2: y = the best alpha over the article's OWN alpha_auto search grid
{0, 0.25, ..., 3} (`numAlphaGridPoints`=13, subsec:alpha_selection), by
auc_rnx, median over seeds, from `exp6_alpha_curves_results.csv` (E6).
Reviewer note 2026-09-18: an earlier version restricted this panel to
alpha in {0,1,2} (the 3 discrete methods run inside E1 itself,
sammon_alpha0_smacof/sammon_alpha_smacof/sammon_alpha2_smacof) - a
leftover from before E6's finer grid existed (see the superseded K4 task
spec). Checked: E6 covers the FULL 13-point grid for all 51 datasets used
here (`exp6_alpha_curves_results.csv`, status=='ok', every dataset has
exactly 13 distinct alpha values) - i.e. the full-grid data has been
available since E6 was run, so there is no reason left to keep the
coarser 3-point proxy. Both vertical threshold lines (below) and this
panel now use the SAME alpha_pred_rule.json this Section's rule (eq:alpha_pred)
is built from, so the right panel visualizes the rule the correlation
numbers back up, not a different quantity.

Both panels also draw the two alpha_pred rule thresholds (t1, t2, in
natural-log(nn_ratio_k1) units - `src.sammon.alpha_predict.predict_alpha_two_threshold`)
as vertical lines, read from `alpha_pred_rule.json` (the SAME file
`fit_alpha_rule.py` writes and `export_numbers.py`'s
`alphaPredThreshold{One,Two}` macros read) - never hand-typed, so the
figure and the numeric thresholds in the article can never drift apart.

Inputs (must exist in the same mode - see `--quick/--full/--smoke`):
  results/data/[<mode>/]dataset_properties.csv (K1, src/experiments/dataset_properties.py)
  results/data/[<mode>/]exp1_dr_benchmark_results.csv (E1, panel 1)
  results/data/[<mode>/]exp6_alpha_curves_results.csv (E6, panel 2)
  results/data/[<mode>/]alpha_pred_rule.json (src/experiments/fit_alpha_rule.py, threshold lines)
A missing input CSV/JSON is fail-loud (a clear message with the command to
generate it) - a substitute figure is never fabricated.

Author feedback 2026-09-17 (second review): raw identifiers (`nn_ratio_k1`,
`tsne_auto`, `alpha_auto`, bare dataset codes) reached the axes/legend
verbatim, and dataset-name labels overlapped each other/the points. Fixed by
routing every rendered string through `display_label` (fig_common.py,
config `display_labels`) and by `_select_label_datasets` - a greedy,
minimum-log-spacing filter over config `fig_regime_map.label_datasets`
(see its docstring) instead of labeling every configured candidate.

Run: venv\\python.exe -m src.figures.fig_regime_map [--quick|--full|--smoke]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.experiments.config_experiments import load_experiments_config
from src.figures.fig_common import (
    OKABE_ITO,
    WIDTH_FULL_WIDTH_IN,
    add_quick_arg,
    display_label,
    mode_data_dir,
    parse_fig_mode,
    require_csv,
    require_experiment_csv,
    save_csv_alongside,
    save_figure,
)

FIG_NAME = "fig_regime_map"


def _load_alpha_pred_rule_coefficients(mode: str) -> dict:
    """`coefficients` dict (t1, t2, a_low, a_mid, a_high) of the fitted
    `two_threshold` alpha_pred rule, read from `alpha_pred_rule.json` - the
    SAME file `export_numbers.py`'s `alphaPredThreshold{One,Two}`/
    `alphaPredAlpha{Low,Mid,High}` macros read
    (`_add_alpha_pred_rule_coefficients`). NEVER hand-typed here: a rule
    refit (`fit_alpha_rule.py`) automatically moves these lines. t1/t2 are
    in natural-log(nn_ratio_k1) units
    (`src.sammon.alpha_predict.predict_alpha_two_threshold`)."""
    rule_path = mode_data_dir(mode) / "alpha_pred_rule.json"
    if not rule_path.exists():
        raise FileNotFoundError(
            f"Missing input data for the figure: {rule_path}\n"
            f"Generate it before running this script: venv\\python.exe -m src.experiments.fit_alpha_rule --{mode}"
        )
    with open(rule_path, "r", encoding="utf-8") as f:
        rule = json.load(f)
    if rule.get("variant") != "two_threshold":
        raise ValueError(f"{rule_path}: fig_regime_map's threshold lines require variant=='two_threshold', got '{rule.get('variant')}'.")
    return {k: float(v) for k, v in rule["coefficients"].items()}


def _best_alpha_full_grid(e6: pd.DataFrame) -> pd.Series:
    """Best alpha per dataset over the article's OWN alpha_auto search grid
    (E6's full alpha column, e.g. {0, 0.25, ..., 3}), by median-over-seeds
    auc_rnx - replaces the earlier {0,1,2} proxy (module docstring). The
    grid itself is whatever `exp6_alpha_curves_results.csv` actually
    contains for this mode (13 points full, fewer for quick/smoke) - never
    hardcoded here."""
    e6_ok = e6[e6["status"] == "ok"]
    med = e6_ok.groupby(["dataset", "alpha"])["auc_rnx"].median()
    wide = med.unstack("alpha")
    return wide.idxmax(axis=1)


def _select_label_datasets(merged: pd.DataFrame, priority_order: list[str], min_log10_gap: float) -> list[str]:
    """Greedily choose which datasets get an annotated name label, so that
    no two labels sit closer than `min_log10_gap` apart in log10(nn_ratio_k1)
    (both panels share this x-axis) - author feedback 2026-09-17: dataset
    labels overlapped each other and the data points; "fewer, readable
    labels beat all labels turned to mush". Preference order: the two
    global extremes of nn_ratio_k1 (always tried first, so the plotted
    range's edges stay represented), then `priority_order` (config
    fig_regime_map.label_datasets) in the given order. A candidate too
    close to an already-accepted label is skipped outright, never forced in."""
    if merged.empty:
        return []
    log_x = np.log10(merged["nn_ratio_k1"])
    extremes = [log_x.idxmin(), log_x.idxmax()]
    seen: set[str] = set()
    candidates = []
    for name in extremes + list(priority_order):
        if name not in seen:
            seen.add(name)
            candidates.append(name)
    selected: list[str] = []
    selected_log_x: list[float] = []
    for name in candidates:
        if name not in merged.index:
            continue
        x = float(log_x.loc[name])
        if any(abs(x - sx) < min_log10_gap for sx in selected_log_x):
            continue
        selected.append(name)
        selected_log_x.append(x)
    return selected


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

    required_methods = ["tsne_auto", "sammon_alpha_auto"]
    missing = [m for m in required_methods if m not in med.columns]
    if missing:
        raise KeyError(f"exp1_dr_benchmark_results.csv (mode={mode}) is missing methods {missing} needed for K4.")
    tsne_minus_alpha_auto = med["tsne_auto"] - med["sammon_alpha_auto"]

    e6 = require_experiment_csv("exp6_alpha_curves", mode)
    best_alpha = _best_alpha_full_grid(e6)

    merged = props.set_index("dataset").join(tsne_minus_alpha_auto.rename("tsne_minus_alpha_auto"), how="inner")
    merged = merged.join(best_alpha.rename("best_alpha_full_grid"), how="inner")
    if merged.empty:
        raise ValueError("The dataset intersection between dataset_properties.csv, exp1_dr_benchmark_results.csv and exp6_alpha_curves_results.csv is empty.")

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
    # jitter in y for panel 2 (the full alpha grid still has repeated
    # optima across datasets, e.g. many datasets peak at alpha=0 or at the
    # grid's upper bound) - see config_experiments.yaml fig_regime_map.
    regime_cfg = exp_cfg["fig_regime_map"]
    label_datasets = set(_select_label_datasets(merged, list(regime_cfg["label_datasets"]), float(regime_cfg["min_log10_x_gap"])))
    jitter_rng = np.random.default_rng(int(regime_cfg["jitter_seed"]))
    jitter_std = float(regime_cfg["jitter_std"])
    # jitter is deterministic per-dataset (order = sorted index, NOT
    # insertion order into the DataFrame) - the same run produces the same
    # figure on repeated execution regardless of input CSV processing order.
    sorted_names = sorted(merged.index)
    jitter_values = jitter_rng.normal(0.0, jitter_std, size=len(sorted_names))
    jitter_map = dict(zip(sorted_names, jitter_values))
    merged["best_alpha_full_grid_jittered"] = [merged.loc[name, "best_alpha_full_grid"] + jitter_map[name] for name in merged.index]

    # alpha_pred rule coefficients (module docstring): t1/t2 (vertical regime
    # thresholds, both panels) and a_low/a_mid/a_high (horizontal reference
    # lines in panel 2 - the alpha VALUE the rule predicts in each regime).
    # x = nn_ratio_k1 is plotted on a log scale but t1/t2 are in
    # natural-log(nn_ratio_k1) units, so the line position is exp(t).
    rule_coef = _load_alpha_pred_rule_coefficients(mode)
    x_t1, x_t2 = float(np.exp(rule_coef["t1"])), float(np.exp(rule_coef["t2"]))

    # --- Spearman rho/p, including a variant without synthetic datasets ----
    stats_rows = []
    for label, subset in [("all", merged), ("real_only", merged[~merged["is_synthetic"]])]:
        rho1, p1, n1 = _spearman_report(subset["nn_ratio_k1"], subset["tsne_minus_alpha_auto"])
        rho2, p2, n2 = _spearman_report(subset["nn_ratio_k1"], subset["best_alpha_full_grid"])
        stats_rows.append({"subset": label, "panel": "tsne_minus_alpha_auto", "rho": rho1, "pvalue": p1, "n": n1})
        stats_rows.append({"subset": label, "panel": "best_alpha_full_grid", "rho": rho2, "pvalue": p2, "n": n2})
    stats_df = pd.DataFrame(stats_rows)

    # --- figure: 2 panels -----------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(WIDTH_FULL_WIDTH_IN, WIDTH_FULL_WIDTH_IN * 0.45))
    color_synth, color_real = OKABE_ITO[6], OKABE_ITO[5]
    auc_rnx_label = display_label("auc_rnx", "metric")
    tsne_label = display_label("tsne_auto", "method")
    alpha_auto_label = display_label("sammon_alpha_auto", "method")
    panel1_ylabel = f"{auc_rnx_label}({tsne_label}) $-$ {auc_rnx_label}({alpha_auto_label})"
    panel2_ylabel = r"best $\alpha$ (full grid, by " + auc_rnx_label + ", y jittered)"
    panels = [
        (ax1, "tsne_minus_alpha_auto", panel1_ylabel, False),
        (ax2, "best_alpha_full_grid_jittered", panel2_ylabel, True),
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
                    display_label(name, "dataset"), (r["nn_ratio_k1"], r[ycol]), fontsize=5, xytext=(3, y_off),
                    textcoords="offset points", va="bottom" if y_off > 0 else "top",
                )
        ax.set_xscale("log")
        ax.set_xlabel(f"{display_label('nn_ratio_k1', 'metric')} (log scale)")
        ax.set_ylabel(ylabel)
        if not is_panel2:
            ax.axhline(0.0, color="grey", linewidth=0.6, linestyle="--")
        else:
            # alpha_pred's own three predicted levels (a_low/a_mid/a_high),
            # NOT the earlier hardcoded (0,1,2) - the horizontal counterpart
            # of the t1/t2 vertical lines below, so this panel shows the
            # exact piecewise-constant rule (eq:alpha_pred), not a generic grid.
            for a_ref in (rule_coef["a_low"], rule_coef["a_mid"], rule_coef["a_high"]):
                ax.axhline(a_ref, color="grey", linewidth=0.4, linestyle=":", zorder=1)
        # alpha_pred regime thresholds t1/t2 (eq:alpha_pred, module docstring):
        # turns this from a plain correlation plot into a picture of the RULE.
        # Label position uses a blended transform (data x, axes-fraction y),
        # INSIDE the top of the axes (never above it, which would collide
        # with the panel title) with a translucent white background so the
        # tag stays legible over a nearby data point.
        blended = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        for x_t, t_label in [(x_t1, r"$t_1$"), (x_t2, r"$t_2$")]:
            ax.axvline(x_t, color="0.35", linewidth=0.8, linestyle="-.", zorder=2)
            ax.text(
                x_t, 0.97, t_label, transform=blended, fontsize=6.5, color="0.2", ha="center", va="top",
                zorder=5, bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=0.5),
            )
    if merged["is_holdout"].any():
        holdout_handle = plt.Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none",
                                     markeredgecolor="black", markersize=5, label="hold-out (independent confirmatory set)")
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles=handles + [holdout_handle], fontsize=5.5)
    else:
        ax1.legend(fontsize=5.5)
    rho_p1 = stats_df[(stats_df["subset"] == "all") & (stats_df["panel"] == "tsne_minus_alpha_auto")].iloc[0]
    rho_p2 = stats_df[(stats_df["subset"] == "all") & (stats_df["panel"] == "best_alpha_full_grid")].iloc[0]
    ax1.set_title(rf"$\rho$={rho_p1['rho']:.2f}, p={rho_p1['pvalue']:.3g}, n={int(rho_p1['n'])}", fontsize=6.5)
    ax2.set_title(rf"$\rho$={rho_p2['rho']:.2f}, p={rho_p2['pvalue']:.3g}, n={int(rho_p2['n'])}", fontsize=6.5)
    fig.suptitle("Distance concentration predicts local-vs-global regime (E1 left, E6 right)", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, FIG_NAME)

    out_csv = merged.reset_index().rename(columns={"index": "dataset"})
    save_csv_alongside(out_csv, FIG_NAME)

    from src.figures.fig_common import figures_out_dir

    stats_df.to_csv(figures_out_dir() / f"{FIG_NAME}_spearman.csv", index=False)

    print(f"{FIG_NAME}: {merged.shape[0]} datasets, rho(all)={rho_p1['rho']:.3f}/{rho_p2['rho']:.3f}.")


if __name__ == "__main__":
    main()
