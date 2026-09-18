# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-18
# License: see the LICENSE file in the repository root
"""
E15 statistics: paired comparison of methods over
`exp15_discovery_task_results.csv` (`src/experiments/exp15_discovery_task.py`).

Unit of analysis = DATASET (never a raw dataset/method/seed row - the
project rule against pseudoreplication, see CLAUDE.md). Per (dataset,
dr_method, question), the `n_seeds` per-seed 'correct'/'soft_rho' values are
FIRST reduced to one value via the MEDIAN across seeds
(`_aggregate_over_seeds`) - only THEN are datasets compared across methods.

A dataset counts as "answered correctly" by a method only if a STRICT
MAJORITY of its seeds got the hard question right (median_correct > 0.5);
an exact 5-5 split (possible with an even seed count) counts as "wrong" -
a fixed, documented, non-fabricated tie-break (never a coin flip).

Two paired tests per question, `reference_method` (config) against every
other `report.main_methods` method:
  - hard score (`dataset_correct`, 0/1): McNemar's test
    (`statsmodels.stats.contingency_tables.mcnemar`) on the 2x2 table of
    (reference correct/wrong) x (baseline correct/wrong) over datasets
    where BOTH have a defined answer.
  - soft score (`median_soft_rho`, continuous): two-sided Wilcoxon
    signed-rank test on the paired per-dataset differences.
Both are Holm-corrected WITHIN their family (all baselines of one
question, one correction for the hard tests, one for the soft tests) - see
`_holm_adjust` (NaN-safe: a test that could not be computed, e.g. zero
discordant pairs, does not shrink the family size for the tests that
COULD be computed, and never receives a fabricated corrected p-value).

Output: results/tables/[<mode>/]exp15_discovery_task_summary.csv (one row
per method x question: n_datasets, error_rate, median/IQR soft rho) and
results/tables/[<mode>/]exp15_discovery_task_pairwise.csv (one row per
baseline x question: McNemar + Wilcoxon results, Holm-adjusted).

Run: venv\\python.exe -m src.experiments.exp15_discovery_task_stats [--quick|--full|--smoke]
or: src\\run_exp15_discovery_task_stats.bat [quick|full|smoke]
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar

from src.common.checkpoint import results_csv_path
from src.common.config import ensure_dir, get_mode_path
from src.common.logging_utils import get_logger
from src.experiments.config_experiments import load_experiments_config, resolve_experiment_config
from src.experiments.exp_common import parse_mode_args, resolve_experiment_name
from src.sammon.discovery_task import DISCOVERY_QUESTIONS

BASE_EXPERIMENT_NAME = "exp15_discovery_task"
STATS_CONFIG_KEY = "exp15_discovery_task_stats"

SUMMARY_COLUMNS = [
    "question", "method", "n_datasets", "error_rate",
    "median_soft_rho", "iqr_low_soft_rho", "iqr_high_soft_rho",
]
PAIRWISE_COLUMNS = [
    "question", "method_a", "method_b", "n_datasets_hard",
    "n_a_correct", "n_b_correct", "n_discordant_a_only", "n_discordant_b_only",
    "p_mcnemar", "p_mcnemar_holm", "reject_mcnemar_holm", "mcnemar_exact",
    "n_datasets_soft", "median_soft_rho_a", "median_soft_rho_b", "median_soft_rho_diff",
    "p_wilcoxon_soft", "p_wilcoxon_soft_holm", "reject_wilcoxon_soft_holm",
]


def _aggregate_over_seeds(df_ok: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, dr_method, question): median over seeds of
    'correct' and 'soft_rho' - see the module docstring for the majority
    tie-break rule. `dataset_correct` is NaN (never fabricated) wherever
    the underlying question was undefined for that dataset (insufficient/
    excessive classes - `median_correct` is NaN in that case)."""
    g = df_ok.groupby(["dataset", "dr_method", "question"], as_index=False).agg(
        n_seeds=("seed", "nunique"),
        median_correct=("correct", "median"),
        median_soft_rho=("soft_rho", "median"),
    )
    g["dataset_correct"] = np.where(g["median_correct"].notna(), (g["median_correct"] > 0.5).astype(float), np.nan)
    return g


def build_summary(agg: pd.DataFrame) -> pd.DataFrame:
    """Metric per method: 'error_rate' = share of datasets with a WRONG hard
    answer (`podil datasetu se spatnou odpovedi` from the task spec) +
    median/IQR of the soft rank correlation."""
    rows: list[dict[str, Any]] = []
    for (question, method), sub in agg.groupby(["question", "dr_method"]):
        sub_valid = sub.dropna(subset=["median_correct"])
        n = len(sub_valid)
        if n == 0:
            continue
        error_rate = float((1.0 - sub_valid["dataset_correct"]).mean())
        soft = sub_valid["median_soft_rho"].dropna()
        rows.append({
            "question": question, "method": method, "n_datasets": n, "error_rate": error_rate,
            "median_soft_rho": float(soft.median()) if len(soft) else float("nan"),
            "iqr_low_soft_rho": float(soft.quantile(0.25)) if len(soft) else float("nan"),
            "iqr_high_soft_rho": float(soft.quantile(0.75)) if len(soft) else float("nan"),
        })
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS) if rows else pd.DataFrame(columns=SUMMARY_COLUMNS)


def _paired_wide(agg: pd.DataFrame, question: str, value_col: str, method_a: str, method_b: str) -> pd.DataFrame | None:
    sub = agg[agg["question"] == question]
    wide = sub.pivot_table(index="dataset", columns="dr_method", values=value_col, aggfunc="first")
    if method_a not in wide.columns or method_b not in wide.columns:
        return None
    return wide[[method_a, method_b]].dropna()


def _holm_adjust(pvalues: list[float]) -> list[float]:
    """Holm step-down (Holm 1979), NaN-safe: NaN entries pass through as NaN
    and are EXCLUDED from the family size m (m = number of tests that could
    actually be computed) - an uncomputable test must not shrink other
    tests' threshold, nor receive a fabricated corrected p-value."""
    pvalues = list(pvalues)
    valid_idx = [i for i, p in enumerate(pvalues) if np.isfinite(p)]
    out = [float("nan")] * len(pvalues)
    if not valid_idx:
        return out
    valid_p = np.array([pvalues[i] for i in valid_idx], dtype=np.float64)
    m = len(valid_p)
    order = np.argsort(valid_p)
    sorted_p = valid_p[order]
    adjusted_sorted = np.empty(m, dtype=np.float64)
    running_max = 0.0
    for i in range(m):
        val = min(1.0, (m - i) * sorted_p[i])
        running_max = max(running_max, val)
        adjusted_sorted[i] = running_max
    adjusted = np.empty(m, dtype=np.float64)
    adjusted[order] = adjusted_sorted
    for local_i, global_i in enumerate(valid_idx):
        out[global_i] = float(adjusted[local_i])
    return out


def _wilcoxon_two_sided(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=np.float64)
    nonzero = diff[diff != 0]
    if nonzero.shape[0] < 1:
        return float("nan")
    try:
        _, p = wilcoxon(diff, alternative="two-sided", zero_method="wilcox", mode="auto")
    except ValueError:
        return float("nan")
    return float(p)


def build_pairwise(
    agg: pd.DataFrame, reference_method: str, baselines: list[str], questions: tuple[str, ...],
    min_paired_datasets: int, mcnemar_exact_max_n: int, alpha: float, logger,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for question in questions:
        family_rows: list[dict[str, Any]] = []
        for baseline in baselines:
            pc = _paired_wide(agg, question, "dataset_correct", reference_method, baseline)
            row: dict[str, Any] = {"question": question, "method_a": reference_method, "method_b": baseline}
            row.update({c: np.nan for c in PAIRWISE_COLUMNS if c not in row})

            if pc is not None and len(pc) >= min_paired_datasets:
                a = pc[reference_method].to_numpy(dtype=np.float64)
                b = pc[baseline].to_numpy(dtype=np.float64)
                both_correct = int(((a == 1) & (b == 1)).sum())
                both_wrong = int(((a == 0) & (b == 0)).sum())
                a_only = int(((a == 1) & (b == 0)).sum())  # reference correct, baseline wrong
                b_only = int(((a == 0) & (b == 1)).sum())  # reference wrong, baseline correct
                n_disc = a_only + b_only
                row.update({
                    "n_datasets_hard": len(pc), "n_a_correct": int(a.sum()), "n_b_correct": int(b.sum()),
                    "n_discordant_a_only": a_only, "n_discordant_b_only": b_only,
                })
                if n_disc == 0:
                    # perfectly concordant - McNemar is undefined (no
                    # information to distinguish the two marginals), NOT a
                    # p=1.0 fabrication.
                    row["p_mcnemar"] = float("nan")
                    row["mcnemar_exact"] = False
                else:
                    exact = n_disc <= mcnemar_exact_max_n
                    table = [[both_correct, a_only], [b_only, both_wrong]]
                    result = mcnemar(table, exact=exact, correction=True)
                    row["p_mcnemar"] = float(result.pvalue)
                    row["mcnemar_exact"] = bool(exact)
            else:
                if pc is not None:
                    logger.warning(
                        "%s: %s vs %s hard score - only %d paired datasets (< min_paired_datasets=%d), skipping McNemar.",
                        question, reference_method, baseline, len(pc), min_paired_datasets,
                    )

            ps = _paired_wide(agg, question, "median_soft_rho", reference_method, baseline)
            if ps is not None and len(ps) >= min_paired_datasets:
                a_soft = ps[reference_method].to_numpy(dtype=np.float64)
                b_soft = ps[baseline].to_numpy(dtype=np.float64)
                diff = a_soft - b_soft
                row.update({
                    "n_datasets_soft": len(ps), "median_soft_rho_a": float(np.median(a_soft)),
                    "median_soft_rho_b": float(np.median(b_soft)), "median_soft_rho_diff": float(np.median(diff)),
                    "p_wilcoxon_soft": _wilcoxon_two_sided(diff),
                })
            else:
                if ps is not None:
                    logger.warning(
                        "%s: %s vs %s soft score - only %d paired datasets (< min_paired_datasets=%d), skipping Wilcoxon.",
                        question, reference_method, baseline, len(ps), min_paired_datasets,
                    )

            family_rows.append(row)

        if not family_rows:
            continue
        holm_mcnemar = _holm_adjust([r["p_mcnemar"] for r in family_rows])
        holm_wilcoxon = _holm_adjust([r["p_wilcoxon_soft"] for r in family_rows])
        for r, p_holm_m, p_holm_w in zip(family_rows, holm_mcnemar, holm_wilcoxon):
            r["p_mcnemar_holm"] = p_holm_m
            r["reject_mcnemar_holm"] = bool(np.isfinite(p_holm_m) and p_holm_m <= alpha)
            r["p_wilcoxon_soft_holm"] = p_holm_w
            r["reject_wilcoxon_soft_holm"] = bool(np.isfinite(p_holm_w) and p_holm_w <= alpha)
            rows.append(r)

    return pd.DataFrame(rows, columns=PAIRWISE_COLUMNS) if rows else pd.DataFrame(columns=PAIRWISE_COLUMNS)


def main() -> None:
    mode = parse_mode_args("E15 stats: paired reference_method vs. report.main_methods comparison.")
    logger = get_logger("exp15_discovery_task_stats", mode=mode)
    stats_cfg = resolve_experiment_config(STATS_CONFIG_KEY, mode)
    experiment_name = resolve_experiment_name(BASE_EXPERIMENT_NAME, mode)

    csv_path = results_csv_path(experiment_name)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing input: {csv_path}. First run exp15_discovery_task.py (--{mode}).")
    df = pd.read_csv(csv_path)
    df_ok = df[df["status"] == "ok"].copy()
    if df_ok.empty:
        raise ValueError(f"{csv_path} contains no successful ('ok') row - nothing to aggregate.")

    report_cfg = load_experiments_config()["report"]
    main_methods = list(report_cfg["main_methods"])
    reference_method = str(stats_cfg["reference_method"])
    if reference_method not in main_methods:
        raise ValueError(
            f"exp15_discovery_task_stats.reference_method='{reference_method}' is not in report.main_methods={main_methods}."
        )
    baselines = [m for m in main_methods if m != reference_method]
    min_paired_datasets = int(stats_cfg["min_paired_datasets"])
    mcnemar_exact_max_n = int(stats_cfg["mcnemar_exact_max_n"])
    alpha = float(stats_cfg["alpha"])

    agg = _aggregate_over_seeds(df_ok)

    summary = build_summary(agg)
    pairwise = build_pairwise(agg, reference_method, baselines, DISCOVERY_QUESTIONS, min_paired_datasets, mcnemar_exact_max_n, alpha, logger)

    out_dir = get_mode_path("results_tables_dir", mode)
    ensure_dir(out_dir)
    summary_path = out_dir / "exp15_discovery_task_summary.csv"
    pairwise_path = out_dir / "exp15_discovery_task_pairwise.csv"
    summary.to_csv(summary_path, index=False)
    pairwise.to_csv(pairwise_path, index=False)
    logger.info("Written: %s (%d rows).", summary_path, len(summary))
    logger.info("Written: %s (%d rows).", pairwise_path, len(pairwise))


if __name__ == "__main__":
    main()
