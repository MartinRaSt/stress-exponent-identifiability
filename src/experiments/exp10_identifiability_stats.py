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
Analyses over `results/data/[<mode>/]exp10_identifiability_check_results.csv`
that `src/experiments/exp10_identifiability_check.py` deliberately left out
(reserse/2026-09-17_zostreni_propozice2.md, section 10, "Analyzy", items
2-6 - item 1, the unit tests, live in tests/test_exp10_identifiability.py).
Analysis unit = DATASET (never a (dataset, alpha, seed) row - see reserse
section 7.2, "nikdy inference pres radky E8/E9... jednotka je dataset").

Item 2 (per-dataset table for alpha in {1,2,3}):
  results/tables/[<mode>/]exp10_identifiability_table.csv/.tex

Items 3-5 (regression, H1/H2/H3, quadratic-law fraction):
  results/data/[<mode>/]exp10_identifiability_stats.csv (long format: one
  row per reported statistic - see STATS_COLUMNS)

  Item 3: OLS regression log(1/tight_a) ~ log(N_over_n_near) over
    exp8_prop2_check rows with p2_holds==True; slope + cluster (dataset)
    bootstrap CI.
  Item 4 (H1/H2/H3):
    H1: Spearman(rho_nn, G_auc_oracle), Spearman(c_1*gamma~_0, G_auc_oracle),
        Spearman(I0_exact, G_auc_oracle) over datasets, permutation p.
    H2: phi = median(G_pred/G_auc_oracle) in the 'low' stratum + cluster
        bootstrap CI.
    H3: TOST equivalence of G_pred to 0 within +-delta_eq in the 'high'
        stratum, delta_eq = median(G_auc_oracle) in that stratum.
  Item 5: fraction of (dataset, alpha>0) rows with quadratic_law_holds==True,
    per alpha and pooled.

Item 6 (s_u vs. the log-normal heuristic) is descriptive-only and left to
`fig_identifiability_law.py`/manual inspection of the CSV - no confirmatory
test is specified for it in reserse section 10.

A missing/empty input CSV is FAIL-LOUD (unlike export_numbers.py's
try_block resilience - this script's whole point is to be run once exp10
has completed; if it hasn't, there is nothing to analyze).

Run: venv\\python.exe -m src.experiments.exp10_identifiability_stats [--quick|--full|--smoke]
or: src\\run_exp10_identifiability_stats.bat [quick|full|smoke]
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.common.checkpoint import results_csv_path
from src.common.config import ensure_dir, get_mode_path
from src.common.logging_utils import get_logger
from src.experiments.config_experiments import resolve_experiment_config
from src.experiments.exp_common import parse_mode_args, resolve_experiment_name
from src.experiments.report_tables import write_booktabs_tex
from src.experiments.stats import bootstrap_ci_1d, cluster_bootstrap_ols_slope, spearman_permutation_test
from src.experiments.stats_holdout import tost_equivalence

BASE_EXPERIMENT_NAME = "exp10_identifiability_check"
E8_BASE_NAME = "exp8_prop2_check"
STATS_CONFIG_KEY = "exp10_identifiability_stats"

STATS_COLUMNS = ["analysis", "label", "value", "ci_low", "ci_high", "pvalue", "n", "note"]

TABLE_COLUMNS = [
    "dataset", "alpha", "n_samples", "rho_nn", "s_u", "s_u_heuristic", "c_alpha",
    "gamma_tilde_Y0", "gamma_tilde_Ya", "r_Y0", "r_Ya",
    "bound_v1", "Delta_prime", "tight_v1", "tight_sandwich",
    "I0_exact", "cert_lower", "alpha_eta", "alpha_dagger",
]


def _log_normal_su_heuristic(rho_nn: pd.Series, n_samples: pd.Series) -> pd.Series:
    """Item 6 (descriptive only, reserse section 10 point 6 / section 3.5):
    s_u_heuristic = log(1/rho_nn) / |Phi^{-1}(ln(2)/(n-1))| - the log-normal
    approximation of s_u implied by observing rho_nn alone (median of the
    minimum of n-1 log-normal draws). NOT a confirmatory test (reserse:
    'jen popisne') - just an extra column for visual comparison with the
    directly measured s_u in the per-dataset table/`fig_identifiability_law`."""
    from scipy.stats import norm

    p_n = np.log(2.0) / (n_samples.astype(np.float64) - 1.0)
    denom = np.abs(norm.ppf(p_n))
    return np.log(1.0 / rho_nn) / denom


def _load_exp10_ok(mode: str) -> pd.DataFrame:
    path = results_csv_path(resolve_experiment_name(BASE_EXPERIMENT_NAME, mode))
    if not path.exists():
        raise FileNotFoundError(f"Missing input: {path}. First run exp10_identifiability_check.py ({mode}).")
    df = pd.read_csv(path)
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        raise ValueError(f"{path} has no rows with status 'ok'.")
    return ok


def build_per_dataset_table(df_ok: pd.DataFrame, alpha_table: list[float], logger) -> pd.DataFrame:
    """Analysis item 2: per-dataset rows for every alpha in `alpha_table`
    that is actually present in the data (reserse section 10, point 2, plus
    the supplement-table columns of section 9.3 - n_samples/rho_nn/s_u)."""
    available_alphas = set(np.round(df_ok["alpha"].unique(), 6))
    used_alphas = [a for a in alpha_table if round(a, 6) in available_alphas]
    missing = sorted(set(alpha_table) - set(used_alphas))
    if missing:
        logger.warning("exp10_identifiability_table: alpha_table values %s are not in the exp10 CSV alpha grid - skipped.", missing)
    if not used_alphas:
        raise ValueError(f"None of alpha_table={alpha_table} are present in the exp10 CSV alpha grid {sorted(available_alphas)}.")
    rows = df_ok[df_ok["alpha"].round(6).isin([round(a, 6) for a in used_alphas])].copy()
    rows["s_u_heuristic"] = _log_normal_su_heuristic(rows["rho_nn"], rows["n_samples"])
    table = rows[TABLE_COLUMNS].sort_values(["dataset", "alpha"]).reset_index(drop=True)
    return table


def dataset_level_frame(df_ok: pd.DataFrame, reference_alpha: float = 1.0) -> pd.DataFrame:
    """One row per dataset, taken at `reference_alpha` (default 1.0 - c_1 in
    the reserse notation): rho_nn, s_u, I0_exact, gamma_tilde_Y0 and
    N_over_n_near do not actually depend on alpha (properties of D/Y0 only -
    see exp10_identifiability_check.py docstring), but G_auc_oracle/G_pred/
    alpha_pred/stratum are only meaningfully read off ONE row per dataset to
    avoid pseudo-replication in the unit=dataset analyses below."""
    sub = df_ok[np.isclose(df_ok["alpha"], reference_alpha)].copy()
    if sub.empty:
        raise ValueError(f"No alpha={reference_alpha} rows in the exp10 CSV - required for the dataset-level analyses (H1/H2/H3, c_1).")
    n_before = sub.shape[0]
    sub = sub.drop_duplicates(subset="dataset")
    if sub.shape[0] != n_before:
        raise ValueError(f"Duplicate (dataset, alpha={reference_alpha}) rows in the exp10 CSV - expected exactly one row per dataset.")
    sub = sub.rename(columns={"c_alpha": "c_1"})
    return sub.set_index("dataset")


def analyze_regression_slack_vs_pairs(df_ok: pd.DataFrame, mode: str, n_boot: int, seed: int, ci_level: float, logger) -> list[dict[str, Any]]:
    """Item 3: log(1/tight_a) ~ log(N_over_n_near), over exp8_prop2_check
    rows with p2_holds==True (ALL seeds - the unit for the cluster bootstrap
    is still 'dataset', not the row)."""
    e8_path = results_csv_path(resolve_experiment_name(E8_BASE_NAME, mode))
    if not e8_path.exists():
        raise FileNotFoundError(f"Missing input: {e8_path}. First run exp8_prop2_check.py ({mode}).")
    e8 = pd.read_csv(e8_path, usecols=["dataset", "alpha", "seed", "tight_a", "p2_holds"])
    e8 = e8[(e8["p2_holds"] == True) & e8["tight_a"].notna() & (e8["tight_a"] > 0.0)]  # noqa: E712
    if e8.empty:
        raise ValueError(f"{e8_path}: no rows with p2_holds==True and a positive tight_a.")

    n_over_n_near = df_ok.groupby("dataset")["N_over_n_near"].median()
    e8 = e8.join(n_over_n_near.rename("N_over_n_near"), on="dataset")
    e8 = e8[e8["N_over_n_near"].notna() & (e8["N_over_n_near"] > 0.0)]
    if e8.empty:
        raise ValueError("No exp8 rows could be matched to a dataset with a positive N_over_n_near from the exp10 CSV.")

    e8["log_inv_tight_a"] = np.log(1.0 / e8["tight_a"])
    e8["log_N_over_n_near"] = np.log(e8["N_over_n_near"])

    fit = cluster_bootstrap_ols_slope(e8, "log_N_over_n_near", "log_inv_tight_a", "dataset", n_boot, seed, ci_level)
    logger.info(
        "Regression log(1/tight_a) ~ log(N_over_n_near): slope=%.3f [%.3f, %.3f] (n_rows=%d, n_clusters=%d, OLS p=%.3g).",
        fit["slope"], fit["ci_lo"], fit["ci_hi"], fit["n_rows"], fit["n_clusters"], fit["pvalue"],
    )
    return [
        {
            "analysis": "regression_slack_vs_pairs", "label": "slope", "value": fit["slope"],
            "ci_low": fit["ci_lo"], "ci_high": fit["ci_hi"], "pvalue": fit["pvalue"], "n": fit["n_rows"],
            "note": f"log(1/tight_a) ~ log(N_over_n_near), exp8_prop2_check_results.csv rows with p2_holds=True, cluster bootstrap over {fit['n_clusters']} datasets (n_boot={n_boot})",
        },
        {
            "analysis": "regression_slack_vs_pairs", "label": "intercept", "value": fit["intercept"],
            "ci_low": float("nan"), "ci_high": float("nan"), "pvalue": float("nan"), "n": fit["n_rows"],
            "note": "same regression, intercept (point estimate only, no bootstrap CI computed)",
        },
    ]


def analyze_h1_spearman(ds: pd.DataFrame, n_perm: int, seed: int, logger) -> list[dict[str, Any]]:
    """H1: does the Veta-1 ceiling (c_1*gamma~_0, I0_exact) or rho_nn predict
    the oracle AUC gain G_auc_oracle over datasets? (reserse section 7.2/10)."""
    g_auc = ds["G_auc_oracle"].to_numpy(dtype=np.float64)
    specs = [
        ("rho_nn_vs_G_auc_oracle", ds["rho_nn"].to_numpy(dtype=np.float64)),
        ("cGammaTildeZero_vs_G_auc_oracle", (ds["c_1"] * ds["gamma_tilde_Y0"]).to_numpy(dtype=np.float64)),
        ("I0exact_vs_G_auc_oracle", ds["I0_exact"].to_numpy(dtype=np.float64)),
    ]
    rows: list[dict[str, Any]] = []
    for label, x in specs:
        out = spearman_permutation_test(x, g_auc, n_perm=n_perm, seed=seed, alternative="two-sided")
        logger.info("H1 %s: rho=%.3f, p=%.4g (n=%d).", label, out["rho"], out["pvalue"], out["n"])
        rows.append({
            "analysis": "H1_spearman", "label": label, "value": out["rho"], "ci_low": float("nan"), "ci_high": float("nan"),
            "pvalue": out["pvalue"], "n": out["n"],
            "note": f"Spearman over datasets (alpha=1.0 row), permutation p (n_perm={n_perm}, seed={seed}), two-sided",
        })
    return rows


def analyze_h2_phi_low(ds: pd.DataFrame, n_boot: int, seed: int, ci_level: float, logger) -> list[dict[str, Any]]:
    """H2: in the 'low' rho_nn stratum (concentrated data, high alpha_pred),
    phi = median(G_pred/G_auc_oracle) should be clearly positive (the rule
    captures a meaningful share of the oracle gain ceiling)."""
    low = ds[ds["stratum"] == "low"].copy()
    ratio = (low["G_pred"] / low["G_auc_oracle"]).replace([np.inf, -np.inf], np.nan)
    ratio = ratio[low["G_auc_oracle"].abs() > 0.0]  # G_auc_oracle==0 -> ratio undefined, dropped (not fabricated as 0/0)
    n_dropped = int(low.shape[0] - ratio.dropna().shape[0])
    if n_dropped > 0:
        logger.warning("H2 phi_low: dropped %d/%d 'low'-stratum datasets with undefined G_pred/G_auc_oracle (G_auc_oracle==0 or NaN).", n_dropped, low.shape[0])
    ratio = ratio.dropna().to_numpy(dtype=np.float64)
    if ratio.size == 0:
        raise ValueError("H2 phi_low: no 'low'-stratum dataset has a defined G_pred/G_auc_oracle ratio.")
    phi = float(np.median(ratio))
    ci_lo, ci_hi = bootstrap_ci_1d(ratio, np.median, n_boot, seed, ci_level)
    logger.info("H2 phi_low: median=%.3f [%.3f, %.3f] (n=%d).", phi, ci_lo, ci_hi, ratio.size)
    return [{
        "analysis": "H2_phi_low", "label": "phi", "value": phi, "ci_low": ci_lo, "ci_high": ci_hi,
        "pvalue": float("nan"), "n": int(ratio.size),
        "note": f"median(G_pred/G_auc_oracle) over 'low'-stratum datasets, percentile bootstrap CI (n_boot={n_boot}, seed={seed})",
    }]


def analyze_h3_tost_high(ds: pd.DataFrame, logger) -> list[dict[str, Any]]:
    """H3: in the 'high' rho_nn stratum (diffuse data, alpha_pred~0), G_pred
    should be equivalent to 0 within +-delta_eq = median(G_auc_oracle) in
    that stratum (small oracle ceiling -> small effects there are the
    theory's prediction, not a failure - reserse section 7.1/7.2)."""
    high = ds[ds["stratum"] == "high"].copy()
    g_pred = high["G_pred"].dropna().to_numpy(dtype=np.float64)
    n_dropped = int(high.shape[0] - g_pred.size)
    if n_dropped > 0:
        logger.warning("H3 tost_high: dropped %d/%d 'high'-stratum datasets with undefined G_pred.", n_dropped, high.shape[0])
    if g_pred.size < 2:
        raise ValueError("H3 tost_high: fewer than 2 'high'-stratum datasets with a defined G_pred - TOST is undefined.")
    delta_eq = float(high["G_auc_oracle"].median())
    p1, p2, p_tost = tost_equivalence(g_pred, delta_eq)
    logger.info("H3 tost_high: mean(G_pred)=%.4f, delta_eq=%.4f, p_tost=%.4g (n=%d).", float(np.mean(g_pred)), delta_eq, p_tost, g_pred.size)
    return [{
        "analysis": "H3_tost_high", "label": "G_pred", "value": float(np.mean(g_pred)), "ci_low": float("nan"), "ci_high": float("nan"),
        "pvalue": p_tost, "n": int(g_pred.size),
        "note": f"TOST equivalence of G_pred to 0 within +-delta_eq={delta_eq:.4f} (median G_auc_oracle in the 'high' stratum), p1={p1:.4g}, p2={p2:.4g}",
    }]


def analyze_quadratic_law_fraction(df_ok: pd.DataFrame, tol: float) -> list[dict[str, Any]]:
    """Item 5: fraction of (dataset, alpha>0) rows with quadratic_law_holds
    True, among rows where the local index was actually computed
    (hessian_skipped==False) - per alpha and pooled."""
    sub = df_ok[(df_ok["alpha"] > 0.0) & (df_ok["hessian_skipped"] == False)].copy()  # noqa: E712
    rows: list[dict[str, Any]] = []
    if sub.empty:
        return rows
    pooled_frac = float(sub["quadratic_law_holds"].mean())
    rows.append({
        "analysis": "quadratic_law_fraction", "label": "pooled", "value": pooled_frac, "ci_low": float("nan"), "ci_high": float("nan"),
        "pvalue": float("nan"), "n": int(sub.shape[0]),
        "note": f"fraction of (dataset,alpha>0) rows with |Delta-alpha^2*I0_exact|<=tol*Delta, tol={tol} (quadratic_law_tol), hessian_skipped=False, pooled over alpha",
    })
    for alpha_value, group in sub.groupby("alpha"):
        rows.append({
            "analysis": "quadratic_law_fraction", "label": f"alpha_{alpha_value:g}", "value": float(group["quadratic_law_holds"].mean()),
            "ci_low": float("nan"), "ci_high": float("nan"), "pvalue": float("nan"), "n": int(group.shape[0]),
            "note": f"same, alpha={alpha_value:g}",
        })
    return rows


def _safe_extend(rows: list[dict[str, Any]], description: str, fn, logger) -> None:
    """Runs one analysis block and extends `rows` with its output; on
    failure (typically too few datasets in a stratum/table - expected in
    --smoke/--quick with only a handful of datasets, see the module
    docstring) logs a WARNING and SKIPS just that block, matching the
    resilience convention of `export_numbers.MacroCollector.try_block` -
    the other analyses still run and the CSV is still written with whatever
    could be computed (never a fabricated substitute row)."""
    try:
        rows.extend(fn())
    except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
        logger.warning("Analysis '%s' SKIPPED (likely too few datasets for --smoke/--quick): %s", description, exc)


def compute_stats(df_ok: pd.DataFrame, mode: str, cfg: dict[str, Any], exp10_cfg: dict[str, Any], logger) -> pd.DataFrame:
    seed = int(cfg["seed"])
    n_perm = int(cfg["n_perm"])
    n_boot = int(cfg["n_boot"])
    ci_level = float(cfg["ci_level"])
    tol = float(exp10_cfg["quadratic_law_tol"])

    rows: list[dict[str, Any]] = []
    _safe_extend(rows, "regression log(1/tight_a) ~ log(N_over_n_near)", lambda: analyze_regression_slack_vs_pairs(df_ok, mode, n_boot, seed, ci_level, logger), logger)

    try:
        ds = dataset_level_frame(df_ok, reference_alpha=1.0)
    except ValueError as exc:
        logger.warning("Dataset-level frame (alpha=1.0) unavailable - H1/H2/H3 SKIPPED: %s", exc)
        ds = None
    if ds is not None:
        _safe_extend(rows, "H1 Spearman ceiling vs. oracle gain", lambda: analyze_h1_spearman(ds, n_perm, seed, logger), logger)
        _safe_extend(rows, "H2 phi (low stratum)", lambda: analyze_h2_phi_low(ds, n_boot, seed, ci_level, logger), logger)
        _safe_extend(rows, "H3 TOST (high stratum)", lambda: analyze_h3_tost_high(ds, logger), logger)

    _safe_extend(rows, "quadratic law fraction", lambda: analyze_quadratic_law_fraction(df_ok, tol), logger)

    return pd.DataFrame(rows, columns=STATS_COLUMNS)


def main() -> None:
    mode = parse_mode_args(
        "exp10 identifiability check - analyses left out of exp10_identifiability_check.py "
        "(reserse/2026-09-17_zostreni_propozice2.md section 10, items 2-6): per-dataset table, "
        "regression, H1/H2/H3, quadratic-law fraction."
    )
    logger = get_logger("exp10_identifiability_stats", mode=mode)

    cfg = resolve_experiment_config(STATS_CONFIG_KEY, mode)
    exp10_cfg = resolve_experiment_config(BASE_EXPERIMENT_NAME, mode)

    df_ok = _load_exp10_ok(mode)

    # --- item 2: per-dataset table -------------------------------------------------
    alpha_table = [float(a) for a in cfg["alpha_table"]]
    table = build_per_dataset_table(df_ok, alpha_table, logger)
    tables_dir = ensure_dir(get_mode_path("results_tables_dir", mode))
    table_csv = tables_dir / "exp10_identifiability_table.csv"
    table.to_csv(table_csv, index=False)
    write_booktabs_tex(
        table, tables_dir / "exp10_identifiability_table.tex",
        caption="Identifiability quantities (Lemma 2/Veta 1/2/3) per dataset, alpha in the reported grid.",
        label="tab:exp10_identifiability_table",
        comment_lines=[f"source: exp10_identifiability_check_results.csv (mode={mode})"],
    )
    logger.info("Table written: %s (%d rows).", table_csv, table.shape[0])

    # --- items 3-6: stats -----------------------------------------------------------
    stats_df = compute_stats(df_ok, mode, cfg, exp10_cfg, logger)
    data_dir = ensure_dir(get_mode_path("results_data_dir", mode))
    stats_csv = data_dir / "exp10_identifiability_stats.csv"
    stats_df.to_csv(stats_csv, index=False)
    logger.info("Stats written: %s (%d rows).", stats_csv, stats_df.shape[0])
    print(f"exp10_identifiability_stats ({mode}): {table_csv} ({table.shape[0]} rows), {stats_csv} ({stats_df.shape[0]} rows).")


if __name__ == "__main__":
    main()
