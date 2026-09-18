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
K7 (documentation/2026-09-12_plan_smeru_clanku.md) - derivation of the rule
alpha_pred = f(nn_ratio_k1) from two inputs:
  results/data/[<mode>/]dataset_properties.csv        (K1)
  results/data/[<mode>/]exp6_alpha_curves_results.csv (K6)
BOTH inputs must be in the SAME mode (--quick/--full/--smoke) - a missing K6
output is fail-loud (NO silent fallback to hardcoded numbers, per the spec:
"until exp6 exists, a fallback is NOT allowed"; for development/testing use
--quick or --smoke, which also generate exp6).

Estimates TWO rule variants (R6-B, reserse/2026-09-12_r1_r6_podklady_smeru_ab.md):
  (a) log_linear:    alpha = clip(a + b*ln(nn_ratio_k1), 0, 3)
  (b) two_threshold: alpha = a_low/a_mid/a_high per 2 thresholds in log(nn_ratio_k1)
      (thresholds from the quantile grid
      `sammon_alpha_pred.two_threshold_quantile_grid`, a_low/a_mid/a_high =
      the median alpha* in the given band on the TRAINING part)

Both variants are estimated using LEAVE-ONE-DATASET-OUT (each dataset is
held out once, the rule is fitted on the remaining N-1, alpha_pred for the
held-out dataset is obtained by applying the rule fitted this way). The
variant with the higher LOO median auc_rnx(alpha_pred) over datasets is
selected; the complete LOO table for both variants is saved to CSV anyway,
for transparency.

auc_rnx(alpha_pred) is obtained by LINEAR INTERPOLATION on the alpha grid
from K6 (`np.interp`) - alpha_pred is a continuous value, K6 only has a
discrete grid.

Output:
  results/data/alpha_pred_rule.json   - the selected variant, coefficients,
                                         LOO summary metrics, source files, date
  results/tables/alpha_pred_loo.csv      - the FULL version of the per-dataset
                                         table (14 columns: alpha_pred,
                                         alpha_star, auc_rnx at
                                         alpha_pred/alpha_star/0/1/alpha_auto,
                                         time, both candidate variants), CSV
                                         only (nothing is lost, see the split below).
  results/tables/alpha_pred_loo_core.csv/.tex     - the core for the main
                                         text/discussion (dataset,
                                         nn_ratio_k1, alpha_star, alpha_pred,
                                         auc at pred/star/0/1); label
                                         tab:alpha_pred_loo (2026-09-14 split
                                         from the original 14-column table
                                         for readability, see
                                         documentation/2026-09-14_prejmenovani_rezimu.md).
  results/tables/alpha_pred_loo_variants.csv/.tex - a comparison of both
                                         candidate rule variants + alpha_auto
                                         + K6 timing; label tab:alpha_pred_loo_variants.

Run: venv\\python.exe -m src.experiments.fit_alpha_rule [--quick|--full|--smoke]
or: src\\run_fit_alpha_rule.bat [quick|full|smoke]
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.common.checkpoint import results_csv_path
from src.common.config import ensure_dir, get_mode_path, get_path, get_tables_dir
from src.common.logging_utils import get_logger
from src.experiments.config_experiments import load_experiments_config
from src.experiments.exp_common import add_mode_args, resolve_experiment_name, resolve_mode
from src.experiments.report_tables import write_booktabs_tex
from src.sammon.alpha_predict import (
    ALPHA_MAX,
    ALPHA_MIN,
    default_rule_path,
    predict_alpha_log_linear,
    predict_alpha_two_threshold,
)

MODULE_NAME = "fit_alpha_rule"


def exclude_holdout_datasets(props_raw: pd.DataFrame, holdout_datasets: set[str] | list[str]) -> pd.DataFrame:
    """Q1 step 2 (A.5): removes from `props_raw` (column 'dataset') all rows
    whose dataset is in `holdout_datasets` - a pure function (no I/O),
    testable independently of `main()` (see tests/test_fit_alpha_rule.py)."""
    holdout_set = set(holdout_datasets)
    if not holdout_set:
        return props_raw
    return props_raw[~props_raw["dataset"].isin(holdout_set)].reset_index(drop=True)


def _require_csv(path: Path, hint: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input for fit_alpha_rule: {path}\n{hint}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path} exists but is empty.")
    return df


def _load_alpha_star_and_grid(exp6_path: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Loads exp6 (ok rows), computes the median auc_rnx over seeds for each
    (dataset, alpha), and returns (a per-dataset alpha_star table, a dict
    dataset -> (sorted_alphas, auc_at_alphas) for subsequent interpolation)."""
    df = _require_csv(exp6_path, "First run venv\\python.exe -m src.experiments.exp6_alpha_curves --<mode> (K6).")
    ok = df[(df["status"] == "ok") & df["auc_rnx"].notna()]
    if ok.empty:
        raise ValueError(f"{exp6_path} contains no successful rows with auc_rnx.")

    med = ok.groupby(["dataset", "alpha"])["auc_rnx"].median().reset_index()

    grids: dict[str, np.ndarray] = {}
    stars: list[dict[str, Any]] = []
    for dataset_name, sub in med.groupby("dataset"):
        sub = sub.sort_values("alpha")
        alphas = sub["alpha"].to_numpy(dtype=np.float64)
        aucs = sub["auc_rnx"].to_numpy(dtype=np.float64)
        grids[dataset_name] = (alphas, aucs)
        i_star = int(np.argmax(aucs))
        stars.append({"dataset": dataset_name, "alpha_star": float(alphas[i_star]), "auc_at_alpha_star": float(aucs[i_star])})

    return pd.DataFrame(stars), grids


def _interp_auc(grids: dict[str, np.ndarray], dataset_name: str, alpha: float) -> float:
    """Linear interpolation of auc_rnx(alpha) on the K6 grid for a given
    dataset (`np.interp` - outside the grid range it returns the edge value,
    which does not happen here, because alpha_pred is always clipped to
    [ALPHA_MIN, ALPHA_MAX] and the K6 grid covers this range)."""
    alphas, aucs = grids[dataset_name]
    return float(np.interp(alpha, alphas, aucs))


def _fit_log_linear(train: pd.DataFrame) -> dict[str, float]:
    """OLS: alpha_star ~ a + b*ln(nn_ratio_k1) on the training set (N-1 datasets)."""
    x = np.log(train["nn_ratio_k1"].to_numpy(dtype=np.float64))
    y = train["alpha_star"].to_numpy(dtype=np.float64)
    b, a = np.polyfit(x, y, deg=1)
    return {"a": float(a), "b": float(b)}


def _fit_two_threshold(train: pd.DataFrame, quantile_grid: list[float]) -> dict[str, float]:
    """A grid search over threshold pairs (t1<t2) from the quantiles of
    log(nn_ratio_k1) of the training set; for each pair, a_low/a_mid/a_high
    are the median alpha_star in the respective band (on the TRAINING set),
    and the pair minimizing the MSE between the predicted and actual
    alpha_star is chosen (again only on the training set - no look at the
    held-out LOO dataset)."""
    log_nn = np.log(train["nn_ratio_k1"].to_numpy(dtype=np.float64))
    alpha_star = train["alpha_star"].to_numpy(dtype=np.float64)
    candidates = np.quantile(log_nn, quantile_grid)

    best: dict[str, float] | None = None
    best_mse = np.inf
    for t1 in candidates:
        for t2 in candidates:
            if t2 <= t1:
                continue
            low_mask = log_nn < t1
            mid_mask = (log_nn >= t1) & (log_nn < t2)
            high_mask = log_nn >= t2
            if low_mask.sum() == 0 or mid_mask.sum() == 0 or high_mask.sum() == 0:
                continue
            a_low = float(np.median(alpha_star[low_mask]))
            a_mid = float(np.median(alpha_star[mid_mask]))
            a_high = float(np.median(alpha_star[high_mask]))
            pred = np.where(low_mask, a_low, np.where(mid_mask, a_mid, a_high))
            mse = float(np.mean((alpha_star - pred) ** 2))
            if mse < best_mse:
                best_mse = mse
                best = {"t1": float(t1), "t2": float(t2), "a_low": a_low, "a_mid": a_mid, "a_high": a_high}

    if best is None:
        raise ValueError(
            "No threshold pair (t1,t2) from the quantile grid gives 3 non-empty bands on the "
            "training set - expand 'two_threshold_quantile_grid' or check the data."
        )
    return best


def _run_loo(props: pd.DataFrame, grids: dict[str, np.ndarray], quantile_grid: list[float], logger) -> pd.DataFrame:
    """Leave-one-dataset-out for both variants. `props` must have the
    columns dataset, nn_ratio_k1, alpha_star (already joined with K1+K6, see `main`)."""
    rows: list[dict[str, Any]] = []
    datasets = props["dataset"].tolist()
    for held_out in datasets:
        train = props[props["dataset"] != held_out]
        held = props[props["dataset"] == held_out].iloc[0]
        nn_ratio_k1 = float(held["nn_ratio_k1"])

        coef_ll = _fit_log_linear(train)
        alpha_pred_ll = predict_alpha_log_linear(nn_ratio_k1, **coef_ll)

        coef_tt = _fit_two_threshold(train, quantile_grid)
        alpha_pred_tt = predict_alpha_two_threshold(nn_ratio_k1, **coef_tt)

        rows.append({
            "dataset": held_out, "nn_ratio_k1": nn_ratio_k1, "alpha_star": float(held["alpha_star"]),
            "alpha_pred_log_linear": alpha_pred_ll, "auc_at_alpha_pred_log_linear": _interp_auc(grids, held_out, alpha_pred_ll),
            "alpha_pred_two_threshold": alpha_pred_tt, "auc_at_alpha_pred_two_threshold": _interp_auc(grids, held_out, alpha_pred_tt),
        })
    return pd.DataFrame(rows)


def _write_loo_tables(loo_out: pd.DataFrame, tables_dir: Path, logger) -> None:
    """Typesets the two LOO tables from an already computed `loo_out` frame.
    Split out so that `--tables-only` can re-typeset them from the CSVs on
    disk without re-fitting the rule (the frozen production rule and its file
    timestamp are the evidence of pre-registration and must not be touched).
    """
    # Table A (core): dataset identification + the chosen rule variant vs.
    # fixed alphas - this is cited in the main text/discussion.
    _LOO_CORE_COLUMNS = [
        "dataset", "nn_ratio_k1", "alpha_star", "alpha_pred", "auc_at_alpha_pred",
        "auc_at_alpha_star", "auc_at_alpha0", "auc_at_alpha1",
    ]
    _LOO_CORE_FORMATS = {
        "nn_ratio_k1": ".3f", "alpha_star": ".2f", "alpha_pred": ".2f",
        "auc_at_alpha_pred": ".3f", "auc_at_alpha_star": ".3f", "auc_at_alpha0": ".3f", "auc_at_alpha1": ".3f",
    }
    loo_core = loo_out[_LOO_CORE_COLUMNS]
    loo_core_csv = tables_dir / "alpha_pred_loo_core.csv"
    loo_core.to_csv(loo_core_csv, index=False)
    write_booktabs_tex(
        loo_core, tables_dir / "alpha_pred_loo_core.tex",
        caption="Leave-one-dataset-out validation of the alpha prediction rule: per-dataset $\\rho_{NN}$, $\\alpha^\\ast$, and AUC$_{RNX}$ at $\\alpha_{\\text{pred}}$ vs. fixed $\\alpha\\in\\{0,1\\}$",
        label="tab:alpha_pred_loo",
        float_format={"__default__": "%.4f", **{k: f"%{v}" for k, v in _LOO_CORE_FORMATS.items()}},
        comment_lines=["source: results/data/dataset_properties.csv + exp6_alpha_curves_results.csv (LOO fit of alpha_pred_rule.json), see fit_alpha_rule.py"],
        # one row per dataset (LOO) -> too tall for one supplement page
        # (2026-09-18 overflow fix, see write_booktabs_tex docstring).
        long_table=True,
    )
    logger.info("Written: %s + .tex (%d datasets, %d columns - core).", loo_core_csv, loo_core.shape[0], len(_LOO_CORE_COLUMNS))

    # Table B (diagnostics): comparison of both candidate rule variants +
    # alpha_auto + K6 timing - supplementary info cited in the supplement.
    _LOO_VARIANTS_COLUMNS = [
        "dataset", "auc_at_alpha_auto", "median_wall_time_sec_k6",
        "alpha_pred_log_linear", "auc_at_alpha_pred_log_linear",
        "alpha_pred_two_threshold", "auc_at_alpha_pred_two_threshold",
    ]
    _LOO_VARIANTS_FORMATS = {
        "auc_at_alpha_auto": ".3f", "median_wall_time_sec_k6": ".1f",
        "alpha_pred_log_linear": ".2f", "auc_at_alpha_pred_log_linear": ".3f",
        "alpha_pred_two_threshold": ".2f", "auc_at_alpha_pred_two_threshold": ".3f",
    }
    loo_variants = loo_out[_LOO_VARIANTS_COLUMNS]
    loo_variants_csv = tables_dir / "alpha_pred_loo_variants.csv"
    loo_variants.to_csv(loo_variants_csv, index=False)
    write_booktabs_tex(
        loo_variants, tables_dir / "alpha_pred_loo_variants.tex",
        caption="Leave-one-dataset-out validation of the alpha prediction rule: candidate-variant comparison (log-linear vs. two-threshold) and K6 timing",
        label="tab:alpha_pred_loo_variants",
        float_format={"__default__": "%.4f", **{k: f"%{v}" for k, v in _LOO_VARIANTS_FORMATS.items()}},
        comment_lines=["source: results/data/dataset_properties.csv + exp6_alpha_curves_results.csv + exp1_dr_benchmark_results.csv (auc_at_alpha_auto), see fit_alpha_rule.py"],
        # one row per dataset (LOO) -> too tall for one supplement page
        # (2026-09-18 overflow fix, see write_booktabs_tex docstring).
        long_table=True,
    )
    logger.info("Written: %s + .tex (%d datasets, %d columns - variants).", loo_variants_csv, loo_variants.shape[0], len(_LOO_VARIANTS_COLUMNS))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="K7: derivation + LOO validation of the rule alpha_pred = f(nn_ratio_k1).")
    add_mode_args(parser)
    # Re-typesetting the LaTeX tables must NOT require re-fitting: the
    # production rule was frozen before the hold-out screening and its file
    # timestamp is the evidence of that (a reviewer checked it). With this
    # flag the tables are rebuilt from the CSVs already on disk and neither
    # alpha_pred_rule.json nor any CSV is touched.
    parser.add_argument("--tables-only", action="store_true",
                        help="only re-typeset the .tex tables from the existing CSVs; "
                             "does not re-fit the rule and does not overwrite alpha_pred_rule.json")
    args = parser.parse_args()
    mode = resolve_mode(args)

    logger = get_logger(MODULE_NAME, mode=mode)
    if args.tables_only:
        tables_dir = get_tables_dir(mode)
        loo_csv = tables_dir / "alpha_pred_loo.csv"
        if not loo_csv.exists():
            raise FileNotFoundError(
                f"{loo_csv} does not exist - run this script without --tables-only first."
            )
        _write_loo_tables(pd.read_csv(loo_csv), tables_dir, logger)
        logger.info("--tables-only: tables re-typeset from %s; the rule was NOT re-fitted.", loo_csv)
        return
    exp_cfg = load_experiments_config()["sammon_alpha_pred"]
    gain_frac_tau = float(exp_cfg["gain_frac_tau"])
    quantile_grid = [float(q) for q in exp_cfg["two_threshold_quantile_grid"]]

    props_path = get_mode_path("results_data_dir", mode)
    props_path = props_path / "dataset_properties.csv"
    props_raw = _require_csv(props_path, f"First run venv\\python.exe -m src.experiments.dataset_properties --{mode} (K1).")
    props_raw = props_raw[props_raw["kind"] == "vector"][["dataset", "nn_ratio_k1"]].dropna()

    # Q1 step 2 (reserse/2026-09-14_specifikace_rozsireni_q1.md, A.5) - the
    # hold-out candidate datasets are EXCLUDED from fitting the rule (both
    # LOO and the final coefficients), so they are a true out-of-sample
    # test, NOT part of the training set. Consistency test with the frozen
    # JSON: tests/test_fit_alpha_rule.py.
    holdout_datasets = set(exp_cfg.get("holdout_datasets", []))
    if holdout_datasets:
        n_before = props_raw.shape[0]
        excluded_present = holdout_datasets & set(props_raw["dataset"])
        props_raw = exclude_holdout_datasets(props_raw, holdout_datasets)
        logger.info(
            "sammon_alpha_pred.holdout_datasets: %d of %d datasets in %s excluded from the rule fit (%d were present: %s).",
            len(holdout_datasets), n_before, props_path, len(excluded_present), sorted(excluded_present),
        )

    exp6_path = results_csv_path(resolve_experiment_name("exp6_alpha_curves", mode))
    alpha_star_df, grids = _load_alpha_star_and_grid(exp6_path)

    merged = props_raw.merge(alpha_star_df, on="dataset", how="inner")
    if merged.shape[0] < 4:
        raise ValueError(
            f"The intersection of dataset_properties.csv and exp6_alpha_curves_results.csv has only "
            f"{merged.shape[0]} datasets - LOO requires at least 4 (run both scripts in the same, wider mode)."
        )
    n_missing_props = props_raw.shape[0] - merged.shape[0]
    if n_missing_props > 0:
        logger.warning("%d datasets from dataset_properties.csv have no corresponding rows in exp6 (skipped).", n_missing_props)

    # --- optional: alpha_auto from E1 for an orientational comparison (a different n_max than K6!) ---
    e1_path = results_csv_path(resolve_experiment_name("exp1_dr_benchmark", mode))
    alpha_auto_auc: pd.Series
    if e1_path.exists():
        e1 = pd.read_csv(e1_path)
        e1_ok = e1[(e1["status"] == "ok") & (e1["method"] == "sammon_alpha_auto") & e1["auc_rnx"].notna()]
        alpha_auto_auc = e1_ok.groupby("dataset")["auc_rnx"].median()
    else:
        logger.warning("%s does not exist - the auc_at_alpha_auto column will be NaN (orientational comparison only, K7 does not depend on it).", e1_path)
        alpha_auto_auc = pd.Series(dtype=float)

    loo = _run_loo(merged, grids, quantile_grid, logger)
    loo["auc_at_alpha0"] = [_interp_auc(grids, d, 0.0) for d in loo["dataset"]]
    loo["auc_at_alpha1"] = [_interp_auc(grids, d, 1.0) for d in loo["dataset"]]
    loo["auc_at_alpha_auto"] = loo["dataset"].map(alpha_auto_auc)

    # time: median wall_time_sec from K6 per dataset (informational, not directly the prediction time - see the module docstring)
    exp6_df = pd.read_csv(exp6_path)
    exp6_ok = exp6_df[exp6_df["status"] == "ok"]
    median_time = exp6_ok.groupby("dataset")["wall_time_sec"].median()
    loo["median_wall_time_sec_k6"] = loo["dataset"].map(median_time)

    med_ll = float(loo["auc_at_alpha_pred_log_linear"].median())
    med_tt = float(loo["auc_at_alpha_pred_two_threshold"].median())
    chosen_variant = "log_linear" if med_ll >= med_tt else "two_threshold"
    logger.info("LOO median auc_rnx(alpha_pred): log_linear=%.4f, two_threshold=%.4f -> selected variant '%s'.", med_ll, med_tt, chosen_variant)

    loo["alpha_pred"] = loo[f"alpha_pred_{chosen_variant}"]
    loo["auc_at_alpha_pred"] = loo[f"auc_at_alpha_pred_{chosen_variant}"]

    # gain_frac (R6-B): only for datasets where alpha_auto improves over alpha=0 by more than tau
    denom = loo["auc_at_alpha_auto"] - loo["auc_at_alpha0"]
    eligible = denom.notna() & (denom > gain_frac_tau)
    gain_frac = (loo.loc[eligible, "auc_at_alpha_pred"] - loo.loc[eligible, "auc_at_alpha0"]) / denom.loc[eligible]
    n_eligible = int(eligible.sum())
    if n_eligible > 0:
        gain_frac_median = float(gain_frac.median())
        gain_frac_iqr = float(gain_frac.quantile(0.75) - gain_frac.quantile(0.25))
    else:
        gain_frac_median, gain_frac_iqr = float("nan"), float("nan")
        logger.warning("No dataset satisfies gain_frac_tau=%.4f (alpha_auto - alpha0 > tau) - gain_frac is NaN.", gain_frac_tau)

    # --- derivation of the FINAL rule on ALL datasets (not just the LOO training sets) ---
    if chosen_variant == "log_linear":
        final_coef = _fit_log_linear(merged)
    else:
        final_coef = _fit_two_threshold(merged, quantile_grid)

    now = datetime.now().isoformat(timespec="seconds")
    rule: dict[str, Any] = {
        "variant": chosen_variant,
        "coefficients": final_coef,
        "alpha_bounds": {"min": ALPHA_MIN, "max": ALPHA_MAX},
        "loo_metrics": {
            "n_datasets": int(merged.shape[0]),
            "median_auc_at_alpha_pred_log_linear": med_ll,
            "median_auc_at_alpha_pred_two_threshold": med_tt,
            "median_auc_at_alpha_pred_chosen": float(loo["auc_at_alpha_pred"].median()),
            "median_auc_at_alpha_star": float(alpha_star_df.set_index("dataset").loc[merged["dataset"], "auc_at_alpha_star"].median()),
            "median_auc_at_alpha0": float(loo["auc_at_alpha0"].median()),
            "median_auc_at_alpha1": float(loo["auc_at_alpha1"].median()),
            "n_datasets_gain_frac_eligible": n_eligible,
            "gain_frac_tau": gain_frac_tau,
            "gain_frac_median": gain_frac_median,
            "gain_frac_iqr": gain_frac_iqr,
        },
        "source_files": {
            "dataset_properties": str(props_path), "exp6_alpha_curves_results": str(exp6_path),
            "exp1_dr_benchmark_results": str(e1_path) if e1_path.exists() else None,
        },
        "mode": mode,
        "generated_at": now,
    }

    # `sammon_alpha_pred` (src/methods/sammon_alpha_pred.py) ALWAYS reads
    # ONLY the root `results/data/alpha_pred_rule.json` (Method.fit_transform
    # has no 'mode' parameter, see src/methods/registry.py::Method) -
    # analogous to `numbers.tex` (K12a), the production rule is written ONLY
    # in 'full' mode. 'quick'/'smoke' write to their own subdirectory for
    # inspection/testing, WITHOUT overwriting the already existing
    # production rule.
    if mode == "full":
        rule_path = default_rule_path()
    else:
        rule_path = get_mode_path("results_data_dir", mode) / "alpha_pred_rule.json"
        logger.info(
            "mode='%s' != 'full' - the rule is written ONLY to %s (testing purpose), "
            "the production %s is NOT overwritten. The 'sammon_alpha_pred' method always uses only the production path.",
            mode, rule_path, default_rule_path(),
        )
    ensure_dir(rule_path.parent)
    rule_path.write_text(json.dumps(rule, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Written: %s (variant=%s, LOO median auc_rnx(pred)=%.4f, gain_frac median=%.3f over %d/%d datasets).",
                rule_path, chosen_variant, rule["loo_metrics"]["median_auc_at_alpha_pred_chosen"], gain_frac_median, n_eligible, merged.shape[0])

    tables_dir = get_tables_dir(mode)
    loo_cols = [
        "dataset", "nn_ratio_k1", "alpha_star", "alpha_pred", "auc_at_alpha_pred", "auc_at_alpha_star",
        "auc_at_alpha0", "auc_at_alpha1", "auc_at_alpha_auto", "median_wall_time_sec_k6",
        "alpha_pred_log_linear", "auc_at_alpha_pred_log_linear", "alpha_pred_two_threshold", "auc_at_alpha_pred_two_threshold",
    ]
    loo["auc_at_alpha_star"] = loo["dataset"].map(alpha_star_df.set_index("dataset")["auc_at_alpha_star"])
    loo_out = loo[loo_cols].sort_values("dataset").reset_index(drop=True)

    # Full version (all 14 columns) - no data is lost, it is just split
    # below into 2 more readable LaTeX tables (~5pt on 14 columns in
    # landscape was unreadable, see documentation/2026-09-14_prejmenovani_rezimu.md).
    loo_csv = tables_dir / "alpha_pred_loo.csv"
    loo_out.to_csv(loo_csv, index=False)
    logger.info("Written: %s (%d datasets, %d columns - full version).", loo_csv, loo_out.shape[0], len(loo_cols))

    _write_loo_tables(loo_out, tables_dir, logger)

    print(
        f"fit_alpha_rule: variant='{chosen_variant}', N={merged.shape[0]} datasets, "
        f"LOO median auc_rnx(pred)={rule['loo_metrics']['median_auc_at_alpha_pred_chosen']:.4f}, "
        f"gain_frac median={gain_frac_median:.3f} (n={n_eligible})."
    )


if __name__ == "__main__":
    main()
