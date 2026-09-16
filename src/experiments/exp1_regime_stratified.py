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
Stratified E1 analysis by distance-concentration regime
(documentation/2026-09-13_e1_stratifikace_rezimu.md) - the basis for the
article's claim "alpha only matters in the regime of concentrated distances,
and the rule recognizes this regime".

Each dataset (kind=='vector' in dataset_properties.csv, K1) is assigned to
ONE of 3 bands according to the ALREADY DERIVED rule `alpha_pred_rule.json`
(K7, `src/experiments/fit_alpha_rule.py`, 'two_threshold' variant):
  regime='low_ratio'  (log(nn_ratio_k1) < t1)  -> the rule predicts a_low
  regime='mid_ratio'  (t1 <= log(...) < t2)    -> the rule predicts a_mid
  regime='high_ratio' (log(...) >= t2)         -> the rule predicts a_high
The band names are NEUTRAL with respect to the distance-concentration theory
(author's decision 2026-09-14, see
documentation/2026-09-14_prejmenovani_rezimu.md): low rho_NN = "low_ratio"
corresponds to more concentrated distances (Beyer et al., the corollary in
clanek/sections/03_metoda.tex states D_max/D_min -> 1, i.e. rho_NN -> 1 =
"high_ratio"); the original labels "concentrated"/"diffuse" were swapped
relative to this convention - the math and results are UNCHANGED, only the
naming.
The thresholds t1/t2 and the values a_low/a_mid/a_high are ALWAYS read from
the JSON (no hardcoded thresholds) - see `classify_regime`/`regime_alpha_pred`.
The rule is ALWAYS read from the production (root) path
`results/data/alpha_pred_rule.json`
(`src.sammon.alpha_predict.load_alpha_pred_rule`), regardless of this
script's --quick/--full/--smoke mode - just like the `sammon_alpha_pred`
method in E1 always uses only this production path (see the `fit_alpha_rule.py`
docstring for `rule_path`), so the classification here matches the rule
that E1 actually used.

For each regime and each method from `METHODS` (exact 'method' column
values in exp1_dr_benchmark_results.csv - a missing method is fail-loud, see
`_check_methods_present`; note: "alpha=1/classic Sammon" in the spec
corresponds to the column 'sammon_alpha_smacof', NOT 'sammon_alpha1_smacof',
which does not exist in the CSV):
  - median over seeds per dataset, then median+IQR over datasets in the
    regime (for auc_rnx and stress_scale_invariant)
  - paired differences 'sammon_alpha_pred' - 'sammon_alpha0_smacof' and
    'sammon_alpha_pred' - 'sammon_alpha_smacof' (alpha=1) per dataset (on
    the per-dataset medians over seeds): median, IQR, number of datasets
    with a positive difference, Wilcoxon signed-rank test (scipy) within
    the regime if n>=5 datasets with a valid pair, otherwise NaN (no
    silent fallback).

Output: results/tables/[<mode>/]exp1_regime_stratified.csv + .tex (booktabs,
label tab:exp1_regime_stratified) - long format, one row per
(regime, row_type, method/paired-diff, metric); see `_LONG_TABLE_COLUMNS`.

Run: venv\\python.exe -m src.experiments.exp1_regime_stratified [--quick|--full|--smoke]
or: src\\run_exp1_regime_stratified.bat [quick|full|smoke]
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.common.checkpoint import results_csv_path
from src.common.config import get_mode_path, get_tables_dir
from src.common.logging_utils import get_logger
from src.experiments.exp_common import add_mode_args, resolve_experiment_name, resolve_mode
from src.sammon.alpha_predict import load_alpha_pred_rule

MODULE_NAME = "exp1_regime_stratified"
BASE_EXPERIMENT_NAME = "exp1_dr_benchmark"

# Exact 'method' column values in exp1_dr_benchmark_results.csv (verified by
# _check_methods_present before use - fail-loud if any is missing).
# 'sammon_alpha_smacof' = alpha=1 (classic Sammon), 'sammon_alpha0_smacof' =
# alpha=0 (classic metric MDS) - see config_experiments.yaml report.main_methods.
METHODS: list[str] = [
    "sammon_alpha0_smacof",   # alpha=0 (MDS)
    "sammon_alpha_smacof",    # alpha=1 (classic Sammon)
    "sammon_alpha_pred",      # K7: predicted alpha from the rule
    "sammon_alpha_auto",      # grid-search alpha (upper bound for comparison)
    "tsne",
    "umap",
    "densmap",
]
METRICS: list[str] = ["auc_rnx", "stress_scale_invariant"]
# Neutral band names based on rho_NN = nn_ratio_k1 (author's decision
# 2026-09-14 - the original "concentrated"/"moderate"/"diffuse" were swapped
# relative to the distance-concentration theory, see the module docstring).
REGIME_ORDER: list[str] = ["low_ratio", "mid_ratio", "high_ratio"]

# paired comparison of the K7 rule against two fixed alphas (task spec)
PAIR_BASELINES: list[str] = ["sammon_alpha0_smacof", "sammon_alpha_smacof"]
PRED_METHOD = "sammon_alpha_pred"

_LONG_TABLE_COLUMNS = [
    "regime", "n_datasets_regime", "row_type", "method", "metric", "n", "median", "iqr", "n_positive", "wilcoxon_pvalue",
]

WILCOXON_MIN_N = 5


def classify_regime(nn_ratio_k1: np.ndarray | pd.Series, rule: dict[str, Any]) -> np.ndarray:
    """Assigns each row to one of 3 bands according to the 'two_threshold'
    rule (t1<t2 in units of log(nn_ratio_k1)) - see
    `src.sammon.alpha_predict.predict_alpha_two_threshold`. The thresholds
    are TAKEN from `rule` (never hardcoded). Fail-loud if the loaded JSON
    uses a different variant (e.g. 'log_linear') - the 3-band stratification
    is defined only for 'two_threshold'."""
    if rule.get("variant") != "two_threshold":
        raise ValueError(
            f"exp1_regime_stratified requires alpha_pred_rule.json with variant=='two_threshold', "
            f"got '{rule.get('variant')}' - the 3-band stratification is not defined for another variant."
        )
    coef = rule["coefficients"]
    t1 = float(coef["t1"])
    t2 = float(coef["t2"])
    if not t1 < t2:
        raise ValueError(f"Expected t1<t2, got t1={t1}, t2={t2}.")
    nn = np.asarray(nn_ratio_k1, dtype=np.float64)
    if np.any(nn <= 0):
        raise ValueError("nn_ratio_k1 must be positive everywhere (log is undefined).")
    log_nn = np.log(nn)
    return np.where(log_nn < t1, "low_ratio", np.where(log_nn < t2, "mid_ratio", "high_ratio"))


def regime_alpha_pred(rule: dict[str, Any]) -> dict[str, float]:
    """The alpha the rule predicts for each regime (a_low/a_mid/a_high from
    the JSON) - for the vertical dashed line in `fig_alpha_gain_by_regime.py`."""
    coef = rule["coefficients"]
    return {"low_ratio": float(coef["a_low"]), "mid_ratio": float(coef["a_mid"]), "high_ratio": float(coef["a_high"])}


def _check_methods_present(df: pd.DataFrame, methods: list[str], csv_path: Path) -> None:
    present = set(df["method"].unique())
    missing = [m for m in methods if m not in present]
    if missing:
        raise KeyError(
            f"{csv_path} is missing 'method' column values {missing} required by exp1_regime_stratified.METHODS "
            f"(present methods: {sorted(present)})."
        )


def build_dataset_regime_table(props: pd.DataFrame, rule: dict[str, Any]) -> pd.DataFrame:
    """`props` must have the columns 'dataset','nn_ratio_k1' (already
    filtered to kind=='vector'). Returns (dataset, nn_ratio_k1, regime, alpha_pred_regime)."""
    required = {"dataset", "nn_ratio_k1"}
    missing = required - set(props.columns)
    if missing:
        raise KeyError(f"dataset_properties.csv is missing columns {sorted(missing)}.")
    out = props[["dataset", "nn_ratio_k1"]].dropna().copy()
    out["regime"] = classify_regime(out["nn_ratio_k1"], rule)
    alpha_map = regime_alpha_pred(rule)
    out["alpha_pred_regime"] = out["regime"].map(alpha_map)
    return out.reset_index(drop=True)


def median_per_dataset_method(e1_ok: pd.DataFrame, methods: list[str], metrics: list[str]) -> pd.DataFrame:
    """Median over seeds per (dataset, method) for each metric in `metrics`
    (only 'status'=='ok' rows - filtered by the caller). Returns a wide
    table with MultiIndex columns (metric, method), rows=dataset."""
    sub = e1_ok[e1_ok["method"].isin(methods)]
    med = sub.groupby(["dataset", "method"])[metrics].median()
    wide = med.unstack("method")
    return wide


def _regime_method_summary_rows(wide: pd.DataFrame, dataset_regime: pd.Series, methods: list[str], metrics: list[str]) -> list[dict[str, Any]]:
    """Rows with row_type='method': median+IQR over datasets in the regime, per method+metric."""
    rows: list[dict[str, Any]] = []
    for regime in REGIME_ORDER:
        ds_in_regime = dataset_regime[dataset_regime == regime].index
        n_datasets_regime = len(ds_in_regime)
        for metric in metrics:
            for method in methods:
                if (metric, method) not in wide.columns:
                    continue
                values = wide.loc[wide.index.intersection(ds_in_regime), (metric, method)].dropna()
                rows.append({
                    "regime": regime, "n_datasets_regime": n_datasets_regime, "row_type": "method",
                    "method": method, "metric": metric, "n": int(values.shape[0]),
                    "median": float(values.median()) if not values.empty else float("nan"),
                    "iqr": float(values.quantile(0.75) - values.quantile(0.25)) if not values.empty else float("nan"),
                    "n_positive": float("nan"), "wilcoxon_pvalue": float("nan"),
                })
    return rows


def _wilcoxon_pvalue(diffs: np.ndarray) -> float:
    """Wilcoxon signed-rank test if n>=WILCOXON_MIN_N, otherwise NaN (no
    silent fallback - it simply is not computed, per the spec "if n>=5").
    Also returns NaN if all differences are (numerically) zero - a
    degenerate case with no defined test (e.g. the 'high_ratio' regime,
    where the rule predicts exactly alpha=0.0 same as the sammon_alpha0_smacof
    baseline -> identical runs), checked BEFORE calling scipy so that no 0/0
    warning arises from the internal normal-approximation computation. Also
    returns NaN if scipy still raises an exception (another degenerate
    case), with an explicit log at the call site."""
    if diffs.shape[0] < WILCOXON_MIN_N:
        return float("nan")
    if np.allclose(diffs, 0.0, atol=1e-12):
        return float("nan")
    try:
        _, pvalue = wilcoxon(diffs)
    except ValueError:
        return float("nan")
    return float(pvalue)


def _regime_paired_diff_rows(wide: pd.DataFrame, dataset_regime: pd.Series, metrics: list[str], logger) -> list[dict[str, Any]]:
    """Rows with row_type='paired_diff': sammon_alpha_pred - baseline, per regime+metric."""
    rows: list[dict[str, Any]] = []
    for regime in REGIME_ORDER:
        ds_in_regime = dataset_regime[dataset_regime == regime].index
        for metric in metrics:
            if (metric, PRED_METHOD) not in wide.columns:
                continue
            pred_vals = wide.loc[wide.index.intersection(ds_in_regime), (metric, PRED_METHOD)]
            for baseline in PAIR_BASELINES:
                if (metric, baseline) not in wide.columns:
                    continue
                base_vals = wide.loc[wide.index.intersection(ds_in_regime), (metric, baseline)]
                paired = pd.concat([pred_vals.rename("pred"), base_vals.rename("base")], axis=1).dropna()
                diffs = (paired["pred"] - paired["base"]).to_numpy(dtype=np.float64)
                n = diffs.shape[0]
                n_positive = int(np.sum(diffs > 0))
                if n < WILCOXON_MIN_N:
                    logger.info(
                        "Regime=%s metric=%s paired diff %s-%s: n=%d < %d, Wilcoxon p=NaN (no silent fallback).",
                        regime, metric, PRED_METHOD, baseline, n, WILCOXON_MIN_N,
                    )
                pvalue = _wilcoxon_pvalue(diffs)
                rows.append({
                    "regime": regime, "n_datasets_regime": len(ds_in_regime), "row_type": "paired_diff",
                    "method": f"{PRED_METHOD} - {baseline}", "metric": metric, "n": n,
                    "median": float(np.median(diffs)) if n > 0 else float("nan"),
                    "iqr": float(np.quantile(diffs, 0.75) - np.quantile(diffs, 0.25)) if n > 0 else float("nan"),
                    "n_positive": n_positive, "wilcoxon_pvalue": pvalue,
                })
    return rows


def build_regime_stratified_table(
    e1_ok: pd.DataFrame, dataset_regime_df: pd.DataFrame, methods: list[str], metrics: list[str], logger,
) -> pd.DataFrame:
    """Builds the complete long table (`_LONG_TABLE_COLUMNS`) - method
    summaries + paired differences, for all regimes. `dataset_regime_df` =
    the output of `build_dataset_regime_table` (columns dataset/regime)."""
    dataset_regime = dataset_regime_df.set_index("dataset")["regime"]
    wide = median_per_dataset_method(e1_ok, methods, metrics)

    rows = _regime_method_summary_rows(wide, dataset_regime, methods, metrics)
    rows += _regime_paired_diff_rows(wide, dataset_regime, metrics, logger)

    out = pd.DataFrame(rows, columns=_LONG_TABLE_COLUMNS)
    row_type_order = pd.Categorical(out["row_type"], categories=["method", "paired_diff"], ordered=True)
    regime_order_cat = pd.Categorical(out["regime"], categories=REGIME_ORDER, ordered=True)
    out = out.assign(_row_type_cat=row_type_order, _regime_cat=regime_order_cat)
    out = out.sort_values(["_regime_cat", "metric", "_row_type_cat", "method"]).drop(columns=["_row_type_cat", "_regime_cat"])
    return out.reset_index(drop=True)


def _write_booktabs_tex(df: pd.DataFrame, out_path: Path, caption: str, label: str) -> None:
    cols = list(df.columns)
    lines = [
        "% auto-generated by src/experiments/exp1_regime_stratified.py - do not edit by hand",
        "\\begin{table}[htbp]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{'l' * len(cols)}}}", "\\toprule", " & ".join(cols) + " \\\\", "\\midrule",
    ]
    for _, row in df.iterrows():
        cells = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Stratified E1 analysis by distance-concentration regime (alpha_pred_rule.json).")
    add_mode_args(parser)
    args = parser.parse_args()
    mode = resolve_mode(args)

    logger = get_logger(MODULE_NAME, mode=mode)

    e1_experiment = resolve_experiment_name(BASE_EXPERIMENT_NAME, mode)
    e1_path = results_csv_path(e1_experiment)
    if not e1_path.exists():
        raise FileNotFoundError(f"Missing input: {e1_path}. First run src\\run_exp1_dr_benchmark.bat {mode}.")
    e1 = pd.read_csv(e1_path)
    _check_methods_present(e1, METHODS, e1_path)
    e1_ok = e1[(e1["status"] == "ok") & e1["method"].isin(METHODS)]
    if e1_ok.empty:
        raise ValueError(f"{e1_path} contains no 'ok' rows for the required methods {METHODS}.")

    props_path = get_mode_path("results_data_dir", mode) / "dataset_properties.csv"
    if not props_path.exists():
        raise FileNotFoundError(f"Missing input: {props_path}. First run src\\run_dataset_properties.bat {mode}.")
    props = pd.read_csv(props_path)
    props = props[props["kind"] == "vector"]
    if props.empty:
        raise ValueError(f"{props_path} contains no row with kind=='vector'.")

    rule = load_alpha_pred_rule()  # always the production root, see the module docstring
    logger.info("Loaded rule alpha_pred_rule.json: variant=%s, t1=%.6f, t2=%.6f.", rule["variant"], rule["coefficients"]["t1"], rule["coefficients"]["t2"])

    dataset_regime_df = build_dataset_regime_table(props, rule)

    e1_datasets = set(e1_ok["dataset"].unique())
    regime_datasets = set(dataset_regime_df["dataset"].unique())
    missing_in_props = e1_datasets - regime_datasets
    if missing_in_props:
        raise ValueError(
            f"{len(missing_in_props)} datasets from exp1_dr_benchmark_results.csv have no kind=='vector' record in "
            f"dataset_properties.csv (mode={mode}): {sorted(missing_in_props)}. Run both scripts in the same mode."
        )
    dataset_regime_df = dataset_regime_df[dataset_regime_df["dataset"].isin(e1_datasets)].reset_index(drop=True)

    table = build_regime_stratified_table(e1_ok, dataset_regime_df, METHODS, METRICS, logger)

    tables_dir = get_tables_dir(mode)
    out_csv = tables_dir / "exp1_regime_stratified.csv"
    table.to_csv(out_csv, index=False)
    _write_booktabs_tex(
        table, tables_dir / "exp1_regime_stratified.tex",
        "AUC$_{RNX}$/stress stratified by distance-concentration regime (E1, median $\\pm$ IQR over datasets; paired diffs of alpha\\_pred vs. fixed alpha, Wilcoxon signed-rank)",
        "tab:exp1_regime_stratified",
    )
    logger.info("Written: %s (%d rows, %d datasets, regimes n=%s).", out_csv, table.shape[0], dataset_regime_df.shape[0],
                dataset_regime_df["regime"].value_counts().reindex(REGIME_ORDER).to_dict())

    print(
        f"exp1_regime_stratified: {dataset_regime_df.shape[0]} datasets into 3 regimes "
        f"{dataset_regime_df['regime'].value_counts().reindex(REGIME_ORDER).to_dict()}, "
        f"table {table.shape[0]} rows -> {out_csv}"
    )


if __name__ == "__main__":
    main()
