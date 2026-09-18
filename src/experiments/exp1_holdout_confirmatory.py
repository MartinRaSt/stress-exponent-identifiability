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
Q1 step 2 (reserse/2026-09-14_specifikace_rozsireni_q1.md, A.6) - confirmatory
and exploratory analysis of hold-out candidate datasets for regime L (low
rho_NN): hypothesis families F_A1-F_A7.

Inputs (ALL must be in the same --quick/--full/--smoke mode):
  results/data/[<mode>/]exp1_dr_benchmark_results.csv (E1, core+holdout runs)
  results/data/[<mode>/]dataset_properties.csv (K1, core datasets - regime)
  results/data/[<mode>/]regime_candidates_screen.csv (Q1 step 2 step 0 - holdout eligibility+regime)
  results/data/[<mode>/]exp6_alpha_curves_results.csv (K6, F_A6 - optional, only a warning if missing)
  results/data/alpha_pred_rule_frozen_20260913.json (A.5 - frozen rule, ALWAYS the production path)

Addendum 2026-09-14 (reserse/2026-09-14_specifikace_rozsireni_q1.md,
"Addendum 2026-09-14: pre-registration change"): ORIGINALLY `phoneme` was
pre-registered (A.3.1 #9) as the ONLY negative control "outside regime L" -
empirically, however, it fell INTO regime L (rho_NN=0.070 < 0.1092), so it
stopped being a control. The control was redesignated to `wall_robot` +
`dry_bean` (empirically mid_ratio, see config_experiments.yaml
q1_negative_control_datasets). IMPORTANT: the F_A3 family below NEVER used a
fixed list of names - `holdout_mid_high` has always been computed
EMPIRICALLY from `regime_frozen` (screen_regime_candidates.csv), so the
pre-registration change does NOT change any output row (`phoneme` already
automatically fell into F_A1/F_A2 via the pooled/holdout regime L,
`wall_robot`/`dry_bean` already automatically fell into F_A3 via mid_ratio) -
the change is ONLY in WHICH datasets are LABELED as the pre-registered
control for citation in the article (see the sanity-check log in main()
below and the macros numHoldoutControlDatasets/holdoutControlDatasetsList in
export_numbers.py).

Families (A.6):
  F_A1 primary (confirmatory, m=2, Holm): pred vs alpha0 (H1a) and pred vs
       alpha1 (H1b) on ALL eligible datasets of regime L (core+holdout, "pooled").
  F_A2 secondary (confirmatory, m=2, Holm): the same, but only on NEW
       (holdout) datasets of regime L (a true out-of-sample test).
  F_A3 no-harm (exploratory, TOST): new datasets of regime M/H - pred vs
       alpha0, equivalence within the band +-tost_margin_auc. Pre-registered
       control (Addendum 2026-09-14): `wall_robot` + `dry_bean` (see above) -
       membership in this family is EMPIRICAL (regime_frozen != low_ratio),
       not enforced by this list.
  F_A4 price (exploratory, m=6, Holm, two-sided): pred vs alpha0 on
       stress_scale_invariant/trustworthiness_k7/continuity_k7/q_local/
       q_global/knn_jaccard_k7, pooled regime L.
  F_A5 vs other methods (exploratory, m=8, Holm, two-sided): pred vs
       {mds, sammon_alpha_auto, tsne_auto, umap_auto, densmap, pacmap,
       trimap, phate}, pooled regime L, auc_rnx.
  F_A6 rule validation (exploratory): Spearman(alpha_pred, alpha*) and
       gain_frac (K6, new datasets only) - requires E6 on the holdout set.
  F_A7 real/synthetic stratification (exploratory, descriptive): median
       Delta (pred-alpha0) separately for real and synthetic datasets in
       pooled regime L.

Each family uses the primary test = exact/MC sign-flip permutation test
(`src.experiments.stats_holdout.sign_flip_test`), cross-check = Wilcoxon,
Holm correction (`holm_correction`) within each family separately (F_A1/F_A2/
F_A4/F_A5 have m>1, F_A3/F_A6/F_A7 do not have a formal Holm family - m<=1).

Output: results/tables/[<mode>/]exp1_holdout_confirmatory.csv + .tex (schema
see A.9 `_CSV_COLUMNS`).

Run: venv\\python.exe -m src.experiments.exp1_holdout_confirmatory [--quick|--full|--smoke]
or: src\\run_exp1_holdout_confirmatory.bat [quick|full|smoke]
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.common.checkpoint import results_csv_path
from src.common.config import get_mode_path, get_tables_dir
from src.common.logging_utils import get_logger
from src.experiments.config_experiments import load_experiments_config, resolve_experiment_config
from src.experiments.exp1_regime_stratified import classify_regime, median_per_dataset_method
from src.experiments.exp_common import add_mode_args, resolve_experiment_name, resolve_mode
from src.experiments.screen_regime_candidates import load_frozen_rule
from src.experiments.report_tables import write_booktabs_tex
from src.experiments.stats_holdout import (
    bootstrap_ci_paired,
    cliff_delta,
    delta_pair,
    hodges_lehmann,
    holm_correction,
    sign_flip_test,
    tost_equivalence,
    wilcoxon_pvalue_safe,
)

MODULE_NAME = "exp1_holdout_confirmatory"

_CSV_COLUMNS = [
    "family", "hypothesis_id", "role", "subset", "regime", "metric", "method_a", "method_b", "alternative",
    "n", "n_positive", "n_zero", "n_negative", "median_diff", "hl_estimate", "mean_diff",
    "ci_low_median", "ci_high_median", "delta_pair", "cliff_delta", "cliff_ci_low", "cliff_ci_high",
    "p_perm", "p_wilcoxon", "p_holm", "reject_holm", "n_perm", "exact_enumeration", "seed",
]

PRED = "sammon_alpha_pred"
ALPHA0 = "sammon_alpha0_smacof"
ALPHA1 = "sammon_alpha_smacof"
F_A4_METRICS = ["stress_scale_invariant", "trustworthiness_k7", "continuity_k7", "q_local", "q_global", "knn_jaccard_k7"]
F_A5_METHODS = ["mds", "sammon_alpha_auto", "tsne_auto", "umap_auto", "densmap", "pacmap", "trimap", "phate"]
_ALL_METHODS = sorted({PRED, ALPHA0, ALPHA1} | set(F_A5_METHODS))


def _paired_arrays(wide: pd.DataFrame, metric: str, method_a: str, method_b: str, datasets: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Returns (a_vals, b_vals, used_datasets) - only datasets from
    `datasets` where BOTH methods have a non-null (non-NaN) `metric` value
    (paired dropna)."""
    if (metric, method_a) not in wide.columns or (metric, method_b) not in wide.columns:
        return np.array([]), np.array([]), []
    sub = wide.loc[wide.index.intersection(datasets), [(metric, method_a), (metric, method_b)]].dropna()
    return sub[(metric, method_a)].to_numpy(dtype=np.float64), sub[(metric, method_b)].to_numpy(dtype=np.float64), list(sub.index)


def paired_diff_row(
    family: str, hypothesis_id: str, role: str, subset: str, regime: str, metric: str,
    method_a: str, method_b: str, alternative: str, a_vals: np.ndarray, b_vals: np.ndarray,
    cfg: dict[str, Any], logger,
) -> dict[str, Any]:
    """Builds a single `_CSV_COLUMNS` row (except p_holm/reject_holm, filled
    in only after grouping by `family` - see `apply_holm_within_families`)."""
    diffs = a_vals - b_vals
    n = diffs.shape[0]
    row: dict[str, Any] = {
        "family": family, "hypothesis_id": hypothesis_id, "role": role, "subset": subset, "regime": regime,
        "metric": metric, "method_a": method_a, "method_b": method_b, "alternative": alternative,
        "n": n, "seed": int(cfg["seed"]),
    }
    if n == 0:
        logger.warning("%s/%s: 0 datasets with a valid pair (%s vs %s, metric %s) - NaN row.", family, hypothesis_id, method_a, method_b, metric)
        row.update({k: np.nan for k in _CSV_COLUMNS if k not in row and k not in ("p_holm", "reject_holm")})
        row["n_perm"], row["exact_enumeration"] = 0, True
        return row

    n_positive, n_zero, n_negative = int(np.sum(diffs > 0)), int(np.sum(diffs == 0)), int(np.sum(diffs < 0))
    if alternative == "tost":
        # TOST (F_A3 no-harm family) is neither a one-sided nor a two-sided
        # shift test: both p-values are computed by the caller via
        # `tost_equivalence` and overwritten into the row. Here, only
        # descriptive statistics are computed (median, CI, Cliff's delta).
        p_perm, n_perm, exact = np.nan, 0, False
        p_wilcoxon = np.nan
    else:
        p_perm, n_perm, exact = sign_flip_test(diffs, alternative, int(cfg["seed"]), int(cfg["n_perm_mc"]), int(cfg["exact_perm_max_n"]))
        p_wilcoxon = wilcoxon_pvalue_safe(diffs, alternative)
    median_diff = float(np.median(diffs))
    ci_low, ci_high = bootstrap_ci_paired(a_vals, b_vals, lambda x, y: float(np.median(x - y)), int(cfg["n_boot"]), int(cfg["seed"]))
    cliff = cliff_delta(a_vals, b_vals)
    cliff_lo, cliff_hi = bootstrap_ci_paired(a_vals, b_vals, cliff_delta, int(cfg["n_boot"]), int(cfg["seed"]) + 1)

    row.update({
        "n_positive": n_positive, "n_zero": n_zero, "n_negative": n_negative,
        "median_diff": median_diff, "hl_estimate": hodges_lehmann(diffs), "mean_diff": float(np.mean(diffs)),
        "ci_low_median": ci_low, "ci_high_median": ci_high, "delta_pair": delta_pair(diffs),
        "cliff_delta": cliff, "cliff_ci_low": cliff_lo, "cliff_ci_high": cliff_hi,
        "p_perm": p_perm, "p_wilcoxon": p_wilcoxon, "n_perm": n_perm, "exact_enumeration": exact,
    })
    return row


def apply_holm_within_families(rows: list[dict[str, Any]], holm_alpha: float) -> list[dict[str, Any]]:
    """Fills in p_holm/reject_holm - Holm correction SEPARATELY within each
    `family` (all non-TOST rows of that family, on the p_perm column). TOST
    rows (alternative=='tost') do not need Holm (a single
    equality/equivalence test) - p_holm=p_perm, reject_holm=(p_perm<holm_alpha) directly."""
    out = [dict(r) for r in rows]
    by_family: dict[str, list[int]] = {}
    for i, r in enumerate(out):
        by_family.setdefault(r["family"], []).append(i)
    for family, idxs in by_family.items():
        tost_idxs = [i for i in idxs if out[i]["alternative"] == "tost"]
        perm_idxs = [i for i in idxs if out[i]["alternative"] != "tost"]
        if perm_idxs:
            pvals = [out[i]["p_perm"] for i in perm_idxs]
            adjusted = holm_correction(pvals)
            for i, p_adj in zip(perm_idxs, adjusted):
                out[i]["p_holm"] = p_adj
                out[i]["reject_holm"] = bool(np.isfinite(p_adj) and p_adj < holm_alpha)
        for i in tost_idxs:
            p = out[i]["p_perm"]
            out[i]["p_holm"] = p
            out[i]["reject_holm"] = bool(np.isfinite(p) and p < holm_alpha)
    return out


def _write_booktabs_tex(df: pd.DataFrame, out_path: Path, caption: str, label: str) -> None:
    """Writes the pre-registered families table through the ONE shared table
    writer (report_tables.write_booktabs_tex), so the header gets readable
    column labels and multi-line \thead cells. The hand-written version this
    replaced printed raw identifiers and came out 1035pt wide against a 548pt
    target, i.e. below the supplement legibility floor.

    `method_a` is dropped: it is the tested rule on every row, it is named in
    the caption, and it was one of the widest columns.
    """
    cols = ["family", "hypothesis_id", "role", "subset", "metric", "method_b",
            "n", "median_diff", "p_perm", "p_holm", "reject_holm"]
    cols = [c for c in cols if c in df.columns]
    write_booktabs_tex(
        df[cols], out_path, caption=caption, label=label,
        comment_lines=["source: results/tables/exp1_holdout_confirmatory.csv, "
                       "see _write_booktabs_tex"],
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Q1 step 2 (A.6): confirmatory/exploratory analysis of hold-out regime-L datasets (F_A1-F_A7).")
    add_mode_args(parser)
    args = parser.parse_args()
    mode = resolve_mode(args)

    logger = get_logger(MODULE_NAME, mode=mode)
    cfg = resolve_experiment_config("exp1_holdout_confirmatory", mode)
    exp_all_cfg = load_experiments_config()
    synthetic_core = set(exp_all_cfg["report"]["synthetic_datasets"])
    holm_alpha = float(cfg["holm_alpha"])
    min_d = int(cfg["eligibility"]["min_d"])

    # --- vstupy ------------------------------------------------------------
    e1_name = resolve_experiment_name("exp1_dr_benchmark", mode)
    e1_path = results_csv_path(e1_name)
    if not e1_path.exists():
        raise FileNotFoundError(f"Missing input: {e1_path}. First run src\\run_exp1_dr_benchmark.bat {mode} --datasets holdout (and core, if not yet done).")
    e1 = pd.read_csv(e1_path)
    e1_ok = e1[e1["status"] == "ok"]

    props_path = get_mode_path("results_data_dir", mode) / "dataset_properties.csv"
    if not props_path.exists():
        raise FileNotFoundError(f"Missing input: {props_path}. First run src\\run_dataset_properties.bat {mode}.")
    props = pd.read_csv(props_path)
    # NOTE: dataset_properties.csv contains both core AND hold-out rows
    # together (--datasets holdout only ADDS to the same file, there is no
    # 'origin' column) - "core" here must be explicitly restricted to
    # exp1_dr_benchmark.datasets (anchor &exp1_datasets, on which the rule
    # was FITTED), otherwise hold-out datasets would leak into the "core"
    # set and F_A1 (pooled) would count them TWICE/mislabel them as fitted.
    core_dataset_names = set(exp_all_cfg["exp1_dr_benchmark"]["datasets"])
    props_vec = props[(props["kind"] == "vector") & (props["dataset"].isin(core_dataset_names))]

    screen_path = get_mode_path("results_data_dir", mode) / "regime_candidates_screen.csv"
    if not screen_path.exists():
        raise FileNotFoundError(f"Missing input: {screen_path}. First run src\\run_screen_regime_candidates.bat {mode}.")
    screen = pd.read_csv(screen_path)

    rule = load_frozen_rule()

    # --- dataset universes ---------------------------------------------------
    props_eligible = props_vec[props_vec["d"] >= min_d].copy()
    props_eligible["regime"] = classify_regime(props_eligible["nn_ratio_k1"], rule)
    core_regime_l = set(props_eligible.loc[props_eligible["regime"] == "low_ratio", "dataset"])

    screen_eligible = screen[screen["eligible_all"] == True]  # noqa: E712 (pandas bool comparison)
    holdout_regime_l = set(screen_eligible.loc[screen_eligible["regime_frozen"] == "low_ratio", "dataset"])
    holdout_mid_high = set(screen_eligible.loc[(screen_eligible["regime_frozen"] != "low_ratio") & (screen_eligible["regime_frozen"] != ""), "dataset"])
    holdout_is_synthetic = dict(zip(screen["dataset"], ~screen["requires_download"].astype(bool)))

    pooled_regime_l = sorted(core_regime_l | holdout_regime_l)
    logger.info(
        "Universe: core regime L=%d (%s), holdout regime L=%d (%s), pooled=%d, holdout M/H no-harm=%d.",
        len(core_regime_l), sorted(core_regime_l), len(holdout_regime_l), sorted(holdout_regime_l), len(pooled_regime_l), len(holdout_mid_high),
    )

    # Addendum 2026-09-14 (pre-registration change): a sanity-check, not an
    # enforcement - just logs whether the pre-registered control datasets
    # actually EMPIRICALLY fell where expected (wall_robot/dry_bean in the
    # M/H no-harm set, phoneme NO LONGER outside regime L). No effect on the
    # computation - just visibility.
    designated_controls = sorted(set(exp_all_cfg.get("q1_negative_control_datasets", [])))
    if designated_controls:
        drifted = [d for d in designated_controls if d not in holdout_mid_high]
        if drifted:
            logger.warning(
                "Addendum 2026-09-14: pre-registered control datasets %s are NOT (all) in the M/H "
                "no-harm set (drift from expectation, record in documentation): %s.",
                designated_controls, drifted,
            )
        else:
            logger.info("Addendum 2026-09-14: pre-registered control datasets %s confirmed in the M/H no-harm set.", designated_controls)
        if "phoneme" in designated_controls:
            logger.warning("Addendum 2026-09-14: 'phoneme' is still listed as a control in config_experiments.yaml - this is AGAINST the addendum (it should have been replaced).")

    wide = median_per_dataset_method(e1_ok, _ALL_METHODS, ["auc_rnx"] + F_A4_METRICS)

    rows: list[dict[str, Any]] = []

    # --- F_A1 primary (confirmatory), pooled regime L, m=2 -------------------
    for hyp_id, method_b in (("H1a", ALPHA0), ("H1b", ALPHA1)):
        a, b, _ds = _paired_arrays(wide, "auc_rnx", PRED, method_b, pooled_regime_l)
        rows.append(paired_diff_row("F_A1", hyp_id, "primary", "pooled", "low_ratio", "auc_rnx", PRED, method_b, "greater", a, b, cfg, logger))

    # --- F_A2 secondary (confirmatory), holdout regime L only, m=2 -------------
    for hyp_id, method_b in (("H1a", ALPHA0), ("H1b", ALPHA1)):
        a, b, _ds = _paired_arrays(wide, "auc_rnx", PRED, method_b, sorted(holdout_regime_l))
        rows.append(paired_diff_row("F_A2", hyp_id, "secondary", "holdout", "low_ratio", "auc_rnx", PRED, method_b, "greater", a, b, cfg, logger))

    # --- F_A3 no-harm (exploratory, TOST), holdout regime M/H ----------------
    a, b, ds_mh = _paired_arrays(wide, "auc_rnx", PRED, ALPHA0, sorted(holdout_mid_high))
    if a.shape[0] >= 2:
        diffs = a - b
        p1, p2, p_tost = tost_equivalence(diffs, float(cfg["tost_margin_auc"]))
        logger.info("F_A3 TOST: n=%d, p1=%.4g, p2=%.4g, p_tost=%.4g (margin=%.3f).", a.shape[0], p1, p2, p_tost, float(cfg["tost_margin_auc"]))
        row = paired_diff_row("F_A3", "no_harm", "exploratory", "holdout", "mid_ratio+high_ratio", "auc_rnx", PRED, ALPHA0, "tost", a, b, cfg, logger)
        row["p_perm"], row["p_wilcoxon"] = p_tost, np.nan
        rows.append(row)
    else:
        logger.warning("F_A3: only %d eligible (eligible_all) regime M/H datasets - skipping (requires >=2).", a.shape[0])

    # --- F_A4 price (exploratory, two-sided), pooled regime L, m=6 ----------
    for metric in F_A4_METRICS:
        a, b, _ds = _paired_arrays(wide, metric, PRED, ALPHA0, pooled_regime_l)
        rows.append(paired_diff_row("F_A4", metric, "exploratory", "pooled", "low_ratio", metric, PRED, ALPHA0, "two-sided", a, b, cfg, logger))

    # --- F_A5 vs other methods (exploratory, two-sided), pooled regime L, m=8 ---
    for method_b in F_A5_METHODS:
        a, b, _ds = _paired_arrays(wide, "auc_rnx", PRED, method_b, pooled_regime_l)
        rows.append(paired_diff_row("F_A5", method_b, "exploratory", "pooled", "low_ratio", "auc_rnx", PRED, method_b, "two-sided", a, b, cfg, logger))

    rows = apply_holm_within_families(rows, holm_alpha)

    # --- F_A7 real/synthetic stratification (exploratory, descriptive) --------
    for label, is_synth in (("pooled_real", False), ("pooled_synthetic", True)):
        ds_subset = [
            d for d in pooled_regime_l
            if (holdout_is_synthetic.get(d, d in synthetic_core) if d in holdout_regime_l else d in synthetic_core) == is_synth
        ]
        a, b, _ds = _paired_arrays(wide, "auc_rnx", PRED, ALPHA0, ds_subset)
        row = paired_diff_row("F_A7", label, "exploratory", label, "low_ratio", "auc_rnx", PRED, ALPHA0, "greater", a, b, cfg, logger)
        row["p_holm"], row["reject_holm"] = np.nan, False  # descriptive family, no formal Holm correction (an m=1 pair would not make sense)
        rows.append(row)

    # --- F_A6 rule validation (exploratory) - requires K6 on the holdout set ----
    try:
        exp6_path = results_csv_path(resolve_experiment_name("exp6_alpha_curves", mode))
        if not exp6_path.exists():
            raise FileNotFoundError(f"{exp6_path} does not exist")
        exp6 = pd.read_csv(exp6_path)
        exp6_ok = exp6[(exp6["status"] == "ok") & exp6["auc_rnx"].notna()]
        exp6_holdout = exp6_ok[exp6_ok["dataset"].isin(holdout_regime_l | holdout_mid_high)]
        if exp6_holdout.empty:
            raise ValueError("no 'ok' K6 rows for hold-out datasets")
        med = exp6_holdout.groupby(["dataset", "alpha"])["auc_rnx"].median().reset_index()
        alpha_star_rows = []
        for dataset_name, sub in med.groupby("dataset"):
            sub = sub.sort_values("alpha")
            i_star = int(np.argmax(sub["auc_rnx"].to_numpy()))
            alpha_star_rows.append({
                "dataset": dataset_name, "alpha_star": float(sub["alpha"].iloc[i_star]),
                "auc_at_alpha_star": float(sub["auc_rnx"].iloc[i_star]),
                "auc_at_alpha0": float(sub.loc[sub["alpha"] == 0.0, "auc_rnx"].iloc[0]) if (sub["alpha"] == 0.0).any() else np.nan,
            })
        alpha_star_df = pd.DataFrame(alpha_star_rows).set_index("dataset")
        rule_coef = rule["coefficients"]
        from src.sammon.alpha_predict import predict_alpha

        joined = props[props["dataset"].isin(alpha_star_df.index)][["dataset", "nn_ratio_k1"]].dropna()
        joined = pd.concat([
            joined, screen[screen["dataset"].isin(alpha_star_df.index)][["dataset", "nn_ratio_k1"]].dropna(),
        ]).drop_duplicates("dataset").set_index("dataset")
        joined = joined.join(alpha_star_df, how="inner")
        joined["alpha_pred"] = joined["nn_ratio_k1"].apply(lambda r: predict_alpha(float(r), rule))

        gain_tau = float(cfg["gain_frac_tau"])
        auc_pred_map = wide[("auc_rnx", PRED)] if ("auc_rnx", PRED) in wide.columns else pd.Series(dtype=float)
        joined["auc_at_alpha_pred"] = joined.index.map(auc_pred_map)
        denom = joined["auc_at_alpha_star"] - joined["auc_at_alpha0"]
        eligible_gain = denom.notna() & (denom > gain_tau) & joined["auc_at_alpha_pred"].notna()
        gain_frac = (joined.loc[eligible_gain, "auc_at_alpha_pred"] - joined.loc[eligible_gain, "auc_at_alpha0"]) / denom.loc[eligible_gain]

        n_a6 = joined.shape[0]
        if n_a6 >= 3:
            rho, pvalue = spearmanr(joined["alpha_pred"], joined["alpha_star"])
        else:
            rho, pvalue = np.nan, np.nan
        logger.info(
            "F_A6: n=%d hold-out datasetu s K6, Spearman(alpha_pred,alpha*)=%.3f (p=%.4g), gain_frac median=%.3f (n=%d).",
            n_a6, rho, pvalue, float(gain_frac.median()) if not gain_frac.empty else float("nan"), int(eligible_gain.sum()),
        )
        rows.append({
            "family": "F_A6", "hypothesis_id": "rule_validation", "role": "exploratory", "subset": "holdout", "regime": "low_ratio+mid_ratio+high_ratio",
            "metric": "spearman_alpha_pred_vs_alpha_star", "method_a": PRED, "method_b": "alpha_star", "alternative": "two-sided",
            "n": n_a6, "n_positive": np.nan, "n_zero": np.nan, "n_negative": np.nan,
            "median_diff": float(gain_frac.median()) if not gain_frac.empty else np.nan,
            "hl_estimate": np.nan, "mean_diff": np.nan, "ci_low_median": np.nan, "ci_high_median": np.nan,
            "delta_pair": np.nan, "cliff_delta": rho, "cliff_ci_low": np.nan, "cliff_ci_high": np.nan,
            "p_perm": np.nan, "p_wilcoxon": pvalue, "p_holm": np.nan, "reject_holm": False,
            "n_perm": 0, "exact_enumeration": True, "seed": int(cfg["seed"]),
        })
    except Exception as exc:
        logger.warning("F_A6 (rule validation) skipped - missing/incomplete K6 hold-out data: %s", exc)

    table = pd.DataFrame(rows, columns=_CSV_COLUMNS)

    tables_dir = get_tables_dir(mode)
    out_csv = tables_dir / "exp1_holdout_confirmatory.csv"
    table.to_csv(out_csv, index=False)
    _write_booktabs_tex(
        table, tables_dir / "exp1_holdout_confirmatory.tex",
        "Confirmatory and exploratory analysis of hold-out regime-L datasets (families F\\_A1-F\\_A7)",
        "tab:exp1_holdout_confirmatory",
    )
    logger.info("Written: %s (%d rows).", out_csv, table.shape[0])
    n_pooled = len(pooled_regime_l)
    print(f"exp1_holdout_confirmatory: pooled regime L n={n_pooled} (core={len(core_regime_l)}, holdout={len(holdout_regime_l)}), {table.shape[0]} rows -> {out_csv}")


if __name__ == "__main__":
    main()
