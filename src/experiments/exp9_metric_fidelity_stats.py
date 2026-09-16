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
Q1 step 3 (reserse/2026-09-14_specifikace_rozsireni_q1.md, PART B.5) -
pre-registered statistics over `exp9_metric_fidelity_results.csv`: analysis
unit = replicate r (methods paired across the SAME replicate/seed).

Families:
  F_B1 (primary, m=9=3 scenarios x 3 `primary_baselines`, Holm FWER 0.05):
       sammon_alpha_pred vs. {tsne_auto, umap_auto, densmap} on the primary
       metric of the scenario (S1 centroid_lre min, S2 cophenetic_pearson
       max, S3 geodesic_stress_si min).
  F_B2 (secondary, m=9=3 scenarios x 3 `secondary_baselines`, Holm separately):
       the same vs. {pacmap, trimap, phate}.
  F_B3 (exploratory, WITHOUT Holm, report only): pred vs. `explorational_vs`
       on the primary metric of the scenario - the broader set of secondary
       metrics/S4 is NOT implemented in this version (time budget for
       K/Q1 step 3), see documentation/2026-09-14_q1_krok3_metricka_vernost.md.

Test: an exact sign-flip permutation test (T=mean Delta) - for
replicates<=exp9_metric_fidelity_stats.exact_perm_max_n a FULL enumeration
of 2**n sign vectors (for R=20: 2**20=1048576), otherwise Monte-Carlo
(`n_perm_mc`, seed from the config). Cross-check: one-sided/two-sided
Wilcoxon signed-rank test (scipy). Holm correction within EACH family
separately (F_B1, F_B2). Effect: median Delta, Hodges-Lehmann estimator,
delta_pair, Cliff's delta (unpaired) + a bootstrap 95% CI (percentile,
B=n_boot, seed from the config, resampling REPLICATES with replacement).

Output: results/tables/[<mode>/]exp9_metric_fidelity_stats.csv +
exp9_metric_fidelity.tex (median +- IQR of the primary/secondary metric,
method x scenario, significance marks after Holm).

Run: venv\\python.exe -m src.experiments.exp9_metric_fidelity_stats [--quick|--full|--smoke]
or: src\\run_exp9_metric_fidelity_stats.bat [quick|full|smoke]
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
from src.common.config import ensure_dir, get_mode_path
from src.common.logging_utils import get_logger
from src.experiments.config_experiments import resolve_experiment_config
from src.experiments.exp_common import parse_mode_args, resolve_experiment_name

BASE_EXPERIMENT_NAME = "exp9_metric_fidelity"
STATS_CONFIG_KEY = "exp9_metric_fidelity_stats"

STATS_COLUMNS: list[str] = [
    "family", "hypothesis_id", "role", "scenario", "metric", "direction", "method_a", "method_b", "alternative",
    "n", "n_positive", "n_zero", "median_diff", "ci_low_median", "ci_high_median",
    "delta_pair", "cliff_delta", "cliff_ci_low", "cliff_ci_high",
    "p_perm", "p_wilcoxon", "p_holm", "reject_holm", "n_perm", "exact_enumeration", "seed",
]


def sign_flip_test(diff: np.ndarray, alternative: str, seed: int, n_perm_mc: int, exact_max_n: int) -> tuple[float, int, bool]:
    """Sign-flip permutation test on the statistic T=mean(diff) (B.5).
    n<=exact_max_n -> full enumeration of 2**n sign vectors; otherwise
    Monte-Carlo (`n_perm_mc` repetitions, seeded). p per B.5: exactly the
    fraction of permutations with T_perm (>=/<=) T_obs; for MC (count+1)/(N_perm+1)."""
    diff = np.asarray(diff, dtype=np.float64)
    n = diff.shape[0]
    if n == 0:
        raise ValueError("sign_flip_test: empty array of differences.")
    T_obs = float(np.mean(diff))
    eps = 1e-12

    if n <= exact_max_n:
        idx = np.arange(2 ** n, dtype=np.int64)
        bits = (idx[:, None] >> np.arange(n)[None, :]) & 1
        signs = 1.0 - 2.0 * bits.astype(np.float64)
        Ts = signs @ diff / n
        n_perm = int(2 ** n)
        exact = True
        if alternative == "greater":
            p = float(np.sum(Ts >= T_obs - eps)) / n_perm
        elif alternative == "less":
            p = float(np.sum(Ts <= T_obs + eps)) / n_perm
        elif alternative == "two-sided":
            p = float(np.sum(np.abs(Ts) >= abs(T_obs) - eps)) / n_perm
        else:
            raise ValueError(f"Unknown alternative '{alternative}'.")
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice(np.array([1.0, -1.0]), size=(n_perm_mc, n))
        Ts = signs @ diff / n
        n_perm = n_perm_mc
        exact = False
        if alternative == "greater":
            p = (float(np.sum(Ts >= T_obs)) + 1.0) / (n_perm + 1.0)
        elif alternative == "less":
            p = (float(np.sum(Ts <= T_obs)) + 1.0) / (n_perm + 1.0)
        elif alternative == "two-sided":
            p = (float(np.sum(np.abs(Ts) >= abs(T_obs))) + 1.0) / (n_perm + 1.0)
        else:
            raise ValueError(f"Unknown alternative '{alternative}'.")
    return float(p), n_perm, exact


def hodges_lehmann(diff: np.ndarray) -> float:
    """Median of the Walsh averages (Delta_i+Delta_j)/2, i<=j (B.5/A.6)."""
    diff = np.asarray(diff, dtype=np.float64)
    n = diff.shape[0]
    iu = np.triu_indices(n)
    walsh = (diff[iu[0]] + diff[iu[1]]) / 2.0
    return float(np.median(walsh))


def delta_pair_stat(diff: np.ndarray) -> tuple[float, int, int, int]:
    """delta_pair = (#Delta_i>0 - #Delta_i<0)/n (A.6/B.5)."""
    diff = np.asarray(diff, dtype=np.float64)
    n = diff.shape[0]
    n_pos = int(np.sum(diff > 0))
    n_neg = int(np.sum(diff < 0))
    n_zero = n - n_pos - n_neg
    return float((n_pos - n_neg) / n), n_pos, n_zero, n_neg


def cliff_delta_stat(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta (unpaired, Cliff 1993, DOI 10.1037/0033-2909.114.3.494):
    (#{a_i>b_j} - #{a_i<b_j}) / (n_a*n_b)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff_matrix = a[:, None] - b[None, :]
    gt = float(np.sum(diff_matrix > 0))
    lt = float(np.sum(diff_matrix < 0))
    return float((gt - lt) / (a.shape[0] * b.shape[0]))


def bootstrap_ci_paired(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int, ci_level: float = 0.95) -> dict[str, tuple[float, float]]:
    """Bootstrap percentile 95% CI (median Delta, Cliff's delta) - resampling
    REPLICATES (paired indices) with replacement, seeded (A.6/B.5)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = a.shape[0]
    rng = np.random.default_rng(seed)
    med_diff_boot = np.empty(n_boot, dtype=np.float64)
    cliff_boot = np.empty(n_boot, dtype=np.float64)
    diff = a - b
    for k in range(n_boot):
        idx = rng.integers(0, n, size=n)
        med_diff_boot[k] = np.median(diff[idx])
        cliff_boot[k] = cliff_delta_stat(a[idx], b[idx])
    lo_q, hi_q = (1.0 - ci_level) / 2.0, 1.0 - (1.0 - ci_level) / 2.0
    return {
        "median_diff": (float(np.quantile(med_diff_boot, lo_q)), float(np.quantile(med_diff_boot, hi_q))),
        "cliff_delta": (float(np.quantile(cliff_boot, lo_q)), float(np.quantile(cliff_boot, hi_q))),
    }


def holm_correction(pvalues: list[float]) -> list[float]:
    """The Holm step-down procedure (Holm 1979): sort p_(1)<=...<=p_(m);
    adjusted p_(j) = max_{i<=j} min(1, (m-i+1)*p_(i)). Returns the adjusted p
    values in the ORIGINAL input order."""
    m = len(pvalues)
    order = np.argsort(pvalues)
    sorted_p = np.array(pvalues)[order]
    adjusted_sorted = np.empty(m, dtype=np.float64)
    running_max = 0.0
    for i in range(m):
        val = min(1.0, (m - i) * sorted_p[i])
        running_max = max(running_max, val)
        adjusted_sorted[i] = running_max
    adjusted = np.empty(m, dtype=np.float64)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()


def _wilcoxon_p(diff: np.ndarray, alternative: str) -> float:
    diff = np.asarray(diff, dtype=np.float64)
    nonzero = diff[diff != 0]
    if nonzero.shape[0] < 1:
        return float("nan")
    try:
        _, p = wilcoxon(diff, alternative=alternative, zero_method="wilcox", mode="auto")
    except ValueError:
        return float("nan")
    return float(p)


def _paired_values(df_ok: pd.DataFrame, scenario: str, metric: str, method_a: str, method_b: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (values_a, values_b) paired over 'replicate' (the B.5 analysis
    unit) for the given scenario/metric - only replicates where BOTH methods
    have 'ok' status and a non-null/non-NaN metric."""
    sub = df_ok[(df_ok["scenario"] == scenario) & df_ok["method"].isin([method_a, method_b])]
    wide = sub.pivot_table(index="replicate", columns="method", values=metric, aggfunc="first")
    if method_a not in wide.columns or method_b not in wide.columns:
        return np.array([]), np.array([])
    wide = wide[[method_a, method_b]].dropna()
    return wide[method_a].to_numpy(dtype=np.float64), wide[method_b].to_numpy(dtype=np.float64)


def _build_row(
    family: str, role: str, scenario: str, metric: str, direction: str, method_a: str, method_b: str,
    a_vals: np.ndarray, b_vals: np.ndarray, seed: int, n_boot: int, n_perm_mc: int, exact_max_n: int,
) -> dict[str, Any] | None:
    if a_vals.shape[0] < 2:
        return None
    diff = a_vals - b_vals
    alternative = "less" if direction == "min" else "greater"
    p_perm, n_perm, exact = sign_flip_test(diff, alternative, seed, n_perm_mc, exact_max_n)
    p_wilcoxon = _wilcoxon_p(diff, alternative)
    delta_pair, n_pos, n_zero, _n_neg = delta_pair_stat(diff)
    cliff = cliff_delta_stat(a_vals, b_vals)
    boot = bootstrap_ci_paired(a_vals, b_vals, n_boot, seed)
    return {
        "family": family, "hypothesis_id": f"{family}_{scenario}_{method_b}", "role": role, "scenario": scenario,
        "metric": metric, "direction": direction, "method_a": method_a, "method_b": method_b, "alternative": alternative,
        "n": int(a_vals.shape[0]), "n_positive": n_pos, "n_zero": n_zero,
        "median_diff": float(np.median(diff)), "ci_low_median": boot["median_diff"][0], "ci_high_median": boot["median_diff"][1],
        "delta_pair": delta_pair, "cliff_delta": cliff, "cliff_ci_low": boot["cliff_delta"][0], "cliff_ci_high": boot["cliff_delta"][1],
        "p_perm": p_perm, "p_wilcoxon": p_wilcoxon, "p_holm": float("nan"), "reject_holm": False,
        "n_perm": n_perm, "exact_enumeration": exact, "seed": seed,
    }


def compute_stats(df: pd.DataFrame, cfg: dict[str, Any], stats_cfg: dict[str, Any], logger) -> pd.DataFrame:
    df_ok = df[df["status"] == "ok"].copy()
    seed = int(stats_cfg["seed"])
    n_boot = int(stats_cfg["n_boot"])
    n_perm_mc = int(stats_cfg["n_perm_mc"])
    exact_max_n = int(stats_cfg["exact_perm_max_n"])

    primary_metric: dict[str, str] = cfg["primary_metric"]
    primary_direction: dict[str, str] = cfg["primary_direction"]
    pred_method = str(cfg["pred_method"])
    scenarios = list(cfg["active_scenarios"])

    rows: list[dict[str, Any]] = []

    def _run_family(family: str, role: str, baselines: list[str]) -> None:
        family_rows: list[dict[str, Any]] = []
        for scenario in scenarios:
            metric = primary_metric[scenario]
            direction = primary_direction[scenario]
            for baseline in baselines:
                a_vals, b_vals = _paired_values(df_ok, scenario, metric, pred_method, baseline)
                row = _build_row(family, role, scenario, metric, direction, pred_method, baseline, a_vals, b_vals, seed, n_boot, n_perm_mc, exact_max_n)
                if row is None:
                    logger.warning("%s %s vs %s: not enough paired replicates, skipping.", scenario, pred_method, baseline)
                    continue
                family_rows.append(row)
        if not family_rows:
            return
        pvals = [r["p_perm"] for r in family_rows]
        adjusted = holm_correction(pvals)
        alpha = float(stats_cfg["alpha_onesided"])
        for r, p_holm in zip(family_rows, adjusted):
            r["p_holm"] = p_holm
            r["reject_holm"] = bool(p_holm <= alpha)
            rows.append(r)

    _run_family("F_B1", "primary", list(cfg["primary_baselines"]))
    _run_family("F_B2", "secondary", list(cfg["secondary_baselines"]))

    # F_B3 (exploratory, without Holm) - pred vs. explorational_vs, the
    # primary metric of the scenario, two-sided (B.5: "pred vs alpha0
    # two-sided ..."); p_holm stays NaN/reject_holm False (not a
    # confirmatory decision).
    for scenario in scenarios:
        metric = primary_metric[scenario]
        for baseline in list(cfg["explorational_vs"]):
            a_vals, b_vals = _paired_values(df_ok, scenario, metric, pred_method, baseline)
            row = _build_row("F_B3", "exploratory", scenario, metric, "two-sided", pred_method, baseline, a_vals, b_vals, seed, n_boot, n_perm_mc, exact_max_n)
            if row is None:
                logger.warning("F_B3 %s %s vs %s: not enough paired replicates, skipping.", scenario, pred_method, baseline)
                continue
            row["direction"] = "two-sided"
            rows.append(row)

    if not rows:
        # an empty result (e.g. smoke with 1 replicate - sign-flip/Holm
        # requires >=2 paired replicates) - return an empty DataFrame with
        # the CORRECT schema (header), so the CSV remains readable by
        # pandas.read_csv (fail-loud elsewhere, not "No columns to parse from file").
        return pd.DataFrame(columns=STATS_COLUMNS)
    out = pd.DataFrame(rows)
    return out[STATS_COLUMNS]


def _write_tex_summary(stats_df: pd.DataFrame, out_dir, logger) -> None:
    """B.7 item 4: a simple booktabs table of the median primary metric per
    scenario/method (F_B1+F_B2 rows) with significance marks after Holm."""
    if stats_df.empty:
        logger.warning("exp9_metric_fidelity.tex: empty statistics, skipping.")
        return
    lines = [
        "% AUTO-GENERATED: src/experiments/exp9_metric_fidelity_stats.py - DO NOT EDIT BY HAND.",
        "\\begin{tabular}{llrrrrl}",
        "\\toprule",
        "scenario & method B & median $\\Delta$ & delta\\_pair & Cliff $\\delta$ & $p_{\\mathrm{Holm}}$ & significance \\\\",
        "\\midrule",
    ]
    for _, r in stats_df[stats_df["role"].isin(["primary", "secondary"])].iterrows():
        sig = "*" if r["reject_holm"] else ""
        lines.append(
            f"{r['scenario']} & {r['method_b']} & {r['median_diff']:.4f} & {r['delta_pair']:.3f} & "
            f"{r['cliff_delta']:.3f} & {r['p_holm']:.4g} & {sig} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    out_path = out_dir / "exp9_metric_fidelity.tex"
    ensure_dir(out_path.parent)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Written: %s", out_path)


def main() -> None:
    mode = parse_mode_args("E9 stats (Q1 step 3): families F_B1/F_B2/F_B3 over exp9_metric_fidelity_results.csv.")
    logger = get_logger("exp9_metric_fidelity_stats", mode=mode)
    cfg = resolve_experiment_config(BASE_EXPERIMENT_NAME, mode)
    stats_cfg = resolve_experiment_config(STATS_CONFIG_KEY, mode)
    experiment_name = resolve_experiment_name(BASE_EXPERIMENT_NAME, mode)
    csv_path = results_csv_path(experiment_name)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing input: {csv_path}. First run exp9_metric_fidelity.py ({mode}).")
    df = pd.read_csv(csv_path)

    stats_df = compute_stats(df, cfg, stats_cfg, logger)
    out_dir = get_mode_path("results_tables_dir", mode)
    ensure_dir(out_dir)
    out_csv = out_dir / "exp9_metric_fidelity_stats.csv"
    stats_df.to_csv(out_csv, index=False)
    logger.info("Written: %s (%d rows).", out_csv, len(stats_df))

    _write_tex_summary(stats_df, out_dir, logger)


if __name__ == "__main__":
    main()
