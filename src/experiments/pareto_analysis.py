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
K3 (documentation/2026-09-12_plan_smeru_clanku.md) - Pareto analysis over
`exp1_dr_benchmark_results.csv` (median over seeds per dataset x method, NO
re-run of the DR methods): for each dataset and each metric pair (see
config_experiments.yaml pareto_analysis.fronts) determines which methods are
on the local (per-dataset) Pareto front and which are strictly dominated by
some "neighbor" method (config report.neighbor_methods - tsne/tsne_auto/umap/
umap_auto/pacmap/trimap).

Standard Pareto dominance definition: point B dominates point A if B is >=
A in ALL normalized (higher-is-better) objectives and > A in at least one.
A method is "on the front" if it is not dominated by ANY other available
method on that dataset.

Outputs:
  results/data/pareto_membership.csv - (dataset, method, <metrics of both
    pairs>, on_front_<front_name> for each pair from the config,
    dominated_by_neighbor_method)
  results/tables/pareto_summary.csv/.tex - per-method aggregation (number of
    datasets on the auc_stress/qlocal_qglobal front, number of datasets
    dominated by a neighbor method, total number of datasets)
  results/tables/pareto_median_front.csv - a control "median over datasets"
    front RESTRICTED to `report.main_methods` (for comparison with the main
    panel of the fig_pareto_front.pdf figure and with the numbers in
    reserse/2026-09-12_proc_sammon_a_smery_clanku.md section 1)

Run: venv\\python.exe -m src.experiments.pareto_analysis [--full|--quick|--smoke]
or: src\\run_pareto_analysis.bat [full|quick|smoke]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.common.checkpoint import results_csv_path
from src.common.config import ensure_dir, get_mode_path, get_path, get_tables_dir
from src.common.logging_utils import get_logger
from src.experiments.config_experiments import load_experiments_config
from src.experiments.exp_common import add_mode_args, resolve_experiment_name, resolve_mode

BASE_EXPERIMENT_NAME = "pareto_analysis"
SOURCE_BASE_EXPERIMENT_NAME = "exp1_dr_benchmark"

# expected values from reserse/2026-09-12_proc_sammon_a_smery_clanku.md
# section 1, for a sanity-check log (see `_log_expected_value_check`) - NEVER
# used as a substitute for the computed values, only for logging differences.
_EXPECTED_MEDIAN_FRONT = {"sammon_alpha0_smacof", "sammon_alpha_auto", "tsne_auto"}
_EXPECTED_PER_DATASET_FRONT_COUNT = {"tsne_auto": 30, "sammon_alpha_auto": 27, "sammon_alpha0_smacof": 22}
_EXPECTED_ALPHA_AUTO_DOMINATED = 0


def pareto_front_mask(values: pd.DataFrame, directions: dict[str, str]) -> pd.Series:
    """Returns a bool Series (index = values.index): True if a given row is
    NOT dominated by any other row of `values` (standard Pareto dominance,
    see the module docstring). `directions`: column name -> 'max'/'min'."""
    normalized = values.copy()
    for col, direction in directions.items():
        if direction not in ("max", "min"):
            raise ValueError(f"Direction '{direction}' for column '{col}' must be 'max' or 'min'.")
        if direction == "min":
            normalized[col] = -normalized[col]
    arr = normalized[list(directions.keys())].to_numpy(dtype=np.float64)
    n = arr.shape[0]
    on_front = np.ones(n, dtype=bool)
    for i in range(n):
        if not np.all(np.isfinite(arr[i])):
            on_front[i] = False
            continue
        for j in range(n):
            if i == j or not np.all(np.isfinite(arr[j])):
                continue
            if np.all(arr[j] >= arr[i]) and np.any(arr[j] > arr[i]):
                on_front[i] = False
                break
    return pd.Series(on_front, index=values.index)


def dominated_by_mask(values: pd.DataFrame, directions: dict[str, str], candidate_index) -> pd.Series:
    """Returns a bool Series: True for rows of `values` that are strictly
    dominated by at least one row whose index is in `candidate_index`
    (e.g. neighbor methods)."""
    normalized = values.copy()
    for col, direction in directions.items():
        if direction == "min":
            normalized[col] = -normalized[col]
    cols = list(directions.keys())
    result = pd.Series(False, index=values.index)
    candidate_index = [c for c in candidate_index if c in values.index]
    for i in values.index:
        vi = normalized.loc[i, cols].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(vi)):
            continue
        for j in candidate_index:
            if j == i:
                continue
            vj = normalized.loc[j, cols].to_numpy(dtype=np.float64)
            if not np.all(np.isfinite(vj)):
                continue
            if np.all(vj >= vi) and np.any(vj > vi):
                result.loc[i] = True
                break
    return result


def _median_per_dataset_method(df: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    ok = df[df["status"] == "ok"]
    return ok.groupby(["dataset", "method"])[metric_cols].median().reset_index()


def _log_expected_value_check(logger, median_front: set[str], per_dataset_counts: pd.Series, alpha_auto_dominated: int) -> None:
    """An honest log of any difference from the expectation in the literature
    review (see the K3 task spec: 'if you get a different result, do not
    hide it'). Does NOT modify any result."""
    if median_front != _EXPECTED_MEDIAN_FRONT:
        logger.warning(
            "K3 check: median front (main_methods) came out as %s, expected from the review %s.",
            sorted(median_front), sorted(_EXPECTED_MEDIAN_FRONT),
        )
    else:
        logger.info("K3 check: median front OK, matches the review (%s).", sorted(median_front))

    for method, expected in _EXPECTED_PER_DATASET_FRONT_COUNT.items():
        actual = int(per_dataset_counts.get(method, -1))
        if actual != expected:
            logger.warning("K3 check: %s on the front for %d datasets, expected %d (review).", method, actual, expected)
        else:
            logger.info("K3 check: %s on the front for %d datasets - OK (matches the review).", method, actual)

    if alpha_auto_dominated != _EXPECTED_ALPHA_AUTO_DOMINATED:
        logger.warning(
            "K3 check: sammon_alpha_auto dominated by a neighbor method on %d datasets, expected %d (review).",
            alpha_auto_dominated, _EXPECTED_ALPHA_AUTO_DOMINATED,
        )
    else:
        logger.info("K3 check: sammon_alpha_auto never dominated by a neighbor method - OK (matches the review).")


def _write_booktabs_tex(df: pd.DataFrame, out_path: Path, caption: str, label: str) -> None:
    cols = list(df.columns)
    lines = [
        "% auto-generated by src/experiments/pareto_analysis.py - do not edit by hand",
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

    parser = argparse.ArgumentParser(description="K3: Pareto analysis over exp1_dr_benchmark_results.csv (no re-run).")
    add_mode_args(parser)
    args = parser.parse_args()
    mode = resolve_mode(args)

    logger = get_logger(BASE_EXPERIMENT_NAME, mode=mode)
    exp_cfg = load_experiments_config()
    fronts_cfg: dict[str, dict[str, str]] = exp_cfg["pareto_analysis"]["fronts"]
    neighbor_methods: list[str] = exp_cfg["report"]["neighbor_methods"]
    main_methods: list[str] = exp_cfg["report"]["main_methods"]

    src_experiment = resolve_experiment_name(SOURCE_BASE_EXPERIMENT_NAME, mode)
    src_path = results_csv_path(src_experiment)
    if not src_path.exists():
        raise FileNotFoundError(
            f"Missing input for K3: {src_path}. First run "
            f"src\\run_exp1_dr_benchmark.bat {mode}."
        )
    df = pd.read_csv(src_path)

    metric_cols = sorted({c for front in fronts_cfg.values() for c in front.keys()})
    missing = [c for c in metric_cols if c not in df.columns]
    if missing:
        raise KeyError(f"{src_path} is missing columns {missing} required by pareto_analysis.fronts in config_experiments.yaml.")

    med = _median_per_dataset_method(df, metric_cols)

    membership_rows: list[pd.DataFrame] = []
    for dataset_name, sub in med.groupby("dataset"):
        sub = sub.set_index("method")
        row = pd.DataFrame(index=sub.index)
        row["dataset"] = dataset_name
        for col in metric_cols:
            row[col] = sub[col]
        for front_name, directions in fronts_cfg.items():
            row[f"on_front_{front_name}"] = pareto_front_mask(sub, directions)
        auc_stress_directions = fronts_cfg.get("auc_stress")
        if auc_stress_directions is not None:
            row["dominated_by_neighbor_method"] = dominated_by_mask(sub, auc_stress_directions, neighbor_methods)
        else:
            row["dominated_by_neighbor_method"] = np.nan
        row = row.reset_index().rename(columns={"index": "method"})
        membership_rows.append(row)

    membership = pd.concat(membership_rows, ignore_index=True)
    membership = membership[["dataset", "method"] + metric_cols + [f"on_front_{f}" for f in fronts_cfg] + ["dominated_by_neighbor_method"]]
    membership = membership.sort_values(["dataset", "method"]).reset_index(drop=True)

    data_dir = ensure_dir(get_mode_path("results_data_dir", mode))
    membership_path = data_dir / "pareto_membership.csv"
    membership.to_csv(membership_path, index=False)
    logger.info("Written: %s (%d rows).", membership_path, membership.shape[0])

    # --- per-method summary ---------------------------------------------------
    summary_rows = []
    for method_name, sub in membership.groupby("method"):
        row = {"method": method_name, "n_datasets_total": sub.shape[0]}
        for front_name in fronts_cfg:
            row[f"n_datasets_on_front_{front_name}"] = int(sub[f"on_front_{front_name}"].sum())
        row["n_datasets_dominated_by_neighbor_method"] = int(sub["dominated_by_neighbor_method"].sum())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("n_datasets_on_front_auc_stress", ascending=False).reset_index(drop=True)

    tables_dir = get_tables_dir(mode)
    summary_csv = tables_dir / "pareto_summary.csv"
    summary.to_csv(summary_csv, index=False)
    _write_booktabs_tex(summary, tables_dir / "pareto_summary.tex", "Pareto front membership per method (E1, 32 datasets)", "tab:pareto_summary")
    logger.info("Written: %s (%d methods).", summary_csv, summary.shape[0])

    # --- median-over-datasets front (main_methods), sanity-check number -------
    med_across = med.groupby("method")[metric_cols].median()
    med_main = med_across.loc[[m for m in main_methods if m in med_across.index]]
    auc_stress_directions = fronts_cfg.get("auc_stress", {"auc_rnx": "max", "stress_scale_invariant": "min"})
    median_front_mask = pareto_front_mask(med_main, auc_stress_directions)
    median_front_df = med_main.copy()
    median_front_df["on_front"] = median_front_mask
    median_front_df = median_front_df.reset_index().sort_values("method")
    median_front_path = tables_dir / "pareto_median_front.csv"
    median_front_df.to_csv(median_front_path, index=False)
    logger.info("Written: %s.", median_front_path)

    per_dataset_counts = summary.set_index("method")["n_datasets_on_front_auc_stress"]
    alpha_auto_dominated = int(summary.set_index("method").get("n_datasets_dominated_by_neighbor_method", pd.Series(dtype=int)).get("sammon_alpha_auto", -1))
    _log_expected_value_check(
        logger, set(median_front_df.loc[median_front_df["on_front"], "method"]), per_dataset_counts, alpha_auto_dominated,
    )

    print(f"pareto_analysis: {membership.shape[0]} rows in pareto_membership.csv, {summary.shape[0]} methods in pareto_summary.csv.")


if __name__ == "__main__":
    main()
