# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-15
# License: see the LICENSE file in the repository root
"""
K15 (2026-09-15, task A, documentation/2026-09-15_e4_metrika_sousedstvi.md):
post-hoc NEIGHBORHOOD PRESERVATION metrics (trustworthiness, continuity -
Venna & Kaski 2001 DOI 10.1109/IJCNN.2001.939861, 2006 DOI
10.1016/j.neunet.2006.05.013; kNN Jaccard - a derived metric, see
src/sammon/metrics.py::knn_jaccard) for the E4 temporal experiment, computed
OVER ALREADY SAVED TRAJECTORIES (results/data/[<mode>/]exp4_trajectories.csv,
one representative seed = exp4_temporal.trajectory_seed).

Reason (author, 2026-09-15): the comparison of our temporal method with
dynamic t-SNE in E4 has so far relied only on the `qual` = STRESS metric,
which is the objective function of the SMACOF/SGD family (dtsne minimizes
KL divergence) - vulnerable to reviewer criticism. This script computes a
metric independent of the objective function of either compared method.

For ALL 3 seeds (not just trajectory_seed), the same metrics are computed
DIRECTLY in `exp4_temporal.py` (task B, per-frame arrays in the .npz cache
'trust_k<K>'/'cont_k<K>'/'jacc_k<K>', aggregates (median) in the results
CSV) - this script is an INDEPENDENT, immediately usable post-hoc addition
over a SINGLE seed that does not require a new temporal fit run.

Data/snapshot source: the SAME `_load_snapshots` and configuration (key
'exp4_temporal', `dataset_params`, mode override) as `exp4_temporal.py` - no
duplication of loading snapshot distance matrices (imported from
src/experiments/exp4_temporal.py; that file is not modified except for task
B below).

Pairing of trajectory nodes (column 'node' in exp4_trajectories.csv) <-> the
row/column order of D_t (returned by `_load_snapshots`) is ALWAYS explicitly
verified as equality of the SETS of identifiers (not just length/index) - a
mismatch is a fail-loud error (an uncaught exception, the whole script
crashes), NEVER a silent skip or a fabricated substitute value.

K values for trustworthiness/continuity/kNN Jaccard come from the config key
`exp4_temporal.neighbor_k` (config_experiments.yaml) - no magic numbers in
the code. If a snapshot is too small for a given K (invalid combination
`2n-3K-1<=0`, see metrics.py::_trustworthiness_continuity, resp. K outside
[1,n-1] for Jaccard), the metric is written as NaN and the reason goes into
the 'note' column - this is NOT fabrication or a silent fallback, it is a
legitimately undefined value (the validation condition is delegated to the
functions themselves in `src/sammon/metrics.py`, no duplicate implementation).

Outputs:
- results/data/[<mode>/]exp4_neighbor_metrics_results.csv (via
  src/common/checkpoint.py - the same naming convention as ALL other
  experiments in the project, `<experiment>_results.csv`; one row per
  (dataset, lambda, alpha, solver, t))
- results/data/[<mode>/]exp4_neighbor_stats.csv (a paired permutation test
  of all our methods (solver in {smacof, sgd}, any lambda/alpha) against
  the dtsne baseline `dtsne_lambda0.0` (analogous to lambda=0 "no
  regularization" in the Sammon family), independence unit = SNAPSHOT,
  paired over (dataset, t); an EXCLUSIVELY DESCRIPTIVE/EXPLORATORY test
  (column 'test_role'), WITHOUT multiple-testing correction - see
  `_write_stats` and K16 below)
- results/data/[<mode>/]exp4_neighbor_primary_pairs.csv (K16, detail of the
  pairing of our family <-> dtsne per dataset x dtsne_lambda x metric - see
  `_build_primary_pairs`)
- results/data/[<mode>/]exp4_neighbor_primary_stats.csv (K16, 2026-09-16,
  the author's "hierarchical test" spec: the PRIMARY confirmatory test,
  independence unit = DATASET, over ALREADY SAVED data from
  `exp4_temporal_results.csv` (all 3 seeds, not just trajectory_seed) - an
  exact sign-flip test over datasets, Holm-Bonferroni correction ONLY over
  `exp4_neighbor_metrics.primary_metrics` - see `_write_primary_stats` and
  documentation/2026-09-16_e4_hierarchicky_test.md)

Limitation: `_write_stats` (the descriptive test) uses only ONE seed
(trajectory_seed) - see task B in exp4_temporal.py for all 3 seeds;
`_write_primary_stats` (the K16 primary test), in contrast, USES all 3 seeds
(reads exp4_temporal_results.csv directly, not the trajectories).

Run: venv\\python.exe -m src.experiments.exp4_neighbor_metrics [--quick|--full|--smoke]
or: src\\run_exp4_neighbor_metrics.bat [quick|full|smoke]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.common.checkpoint import results_csv_path
from src.common.config import load_config
from src.common.logging_utils import get_logger
from src.experiments import exp4_common
from src.experiments.config_experiments import resolve_experiment_config
from src.experiments.exp_common import (
    check_results_schema,
    filter_already_done,
    parse_mode_args,
    resolve_experiment_name,
    run_experiment_grid,
    write_done_file,
)
from src.experiments.stats_holdout import (
    bootstrap_ci_paired,
    exact_sign_flip_pvalue,
    hodges_lehmann,
    holm_correction,
)

BASE_EXPERIMENT_NAME = "exp4_neighbor_metrics"
_EXP4_TEMPORAL_NAME = "exp4_temporal"
# Baseline "without temporal smoothing" in the dtsne family - analogous to
# lambda=0 in the Sammon family (see the docstring above and the
# exp4_temporal.py comment on 'dtsne:' - lam_dt=0.0 is ALWAYS the first
# element in `exp4_temporal.dtsne.lambdas`).
_DTSNE_BASELINE_LAMBDA = 0.0
# K16 (2026-09-16): the descriptive (exploratory) test over snapshots has NO
# claim to significance (see _write_stats) - each row of
# exp4_neighbor_stats.csv is labeled with this string in the 'test_role'
# column, the primary test has its own file exp4_neighbor_primary_stats.csv
# with 'test_role'="primary".
_EXPLORATORY_TEST_ROLE = "exploratory_no_multiplicity_correction"
_PRIMARY_TEST_ROLE = "primary"


def _metric_columns(neighbor_k: list[int]) -> list[str]:
    """Names of the metric columns for the given K grid (deterministic
    order: sorted ascending by K, within each K always trustworthiness/continuity/jaccard)."""
    cols: list[str] = []
    for k in sorted(neighbor_k):
        cols += [f"trustworthiness_k{k}", f"continuity_k{k}", f"knn_jaccard_k{k}"]
    return cols


def _build_tasks(cfg: dict[str, Any], mode: str, traj_df: pd.DataFrame, neighbor_k: list[int]) -> list[dict[str, Any]]:
    """Builds one task per (dataset, lambda, alpha, solver, t) - each task
    carries only D_t (the distance matrix of THAT snapshot) and Y (the
    embedding of the nodes of that snapshot for the given method), not the
    whole time series, so the amount of data passed to ProcessPoolExecutor
    is limited to a single snapshot."""
    from src.experiments.exp4_temporal import _load_snapshots

    trajectory_seed = cfg["trajectory_seed"]
    dataset_params = cfg["dataset_params"]

    tasks: list[dict[str, Any]] = []
    grouped = traj_df.groupby(["dataset", "lambda", "alpha", "solver"], dropna=False, sort=False)
    for (dataset_name, lam, alpha, solver), sub in grouped:
        if dataset_name not in dataset_params:
            raise KeyError(
                f"exp4_temporal.dataset_params has no entry for dataset '{dataset_name}' "
                f"(found in exp4_trajectories.csv) - the configuration does not match the data."
            )
        ds_params = dataset_params[dataset_name]
        list_of_D, node_ids = _load_snapshots(
            dataset_name, ds_params["time_bin_sec"], ds_params["min_snapshot_nodes"], mode=mode,
        )
        method_name = f"dtsne_lambda{lam}" if solver == "dtsne" else f"lambda{lam}_alpha{alpha}_{solver}"

        for t, sub_t in sub.groupby("t", sort=True):
            t = int(t)
            if t >= len(list_of_D):
                raise ValueError(
                    f"Trajectory for dataset={dataset_name} lambda={lam} alpha={alpha} solver={solver} "
                    f"contains t={t}, but _load_snapshots returned only {len(list_of_D)} usable snapshots "
                    "for this configuration - a mismatch between the saved trajectories and the current config."
                )
            y_by_node = {str(row.node): (float(row.x), float(row.y)) for row in sub_t.itertuples(index=False)}
            tasks.append({
                "dataset_name": dataset_name, "lam": float(lam), "alpha": alpha, "solver": solver, "t": t,
                "method_name": f"{method_name}__t{t}", "seed": int(trajectory_seed),
                "D_t": list_of_D[t], "node_ids_t": node_ids[t], "y_by_node": y_by_node,
                "neighbor_k": neighbor_k,
            })
    return tasks


def _run_single(task: dict[str, Any]) -> dict[str, Any]:
    """One (dataset, lambda, alpha, solver, snapshot t): verifies node
    pairing (fail-loud on mismatch - see the module docstring), computes
    trustworthiness/continuity/kNN Jaccard for all K in `task['neighbor_k']`.

    An invalid (n, K) combination is NOT an exception of this function
    (metrics.py reports it internally as a ValueError, which is caught here
    and converted to NaN + a note) - unlike a node identifier mismatch,
    which is NOT caught and must crash the whole run (a genuine data/config
    error, not a legitimate edge case)."""
    from src.sammon.metrics import _neighbor_ranks, _trustworthiness_continuity, knn_jaccard

    t0 = time.perf_counter()
    dataset_name = task["dataset_name"]
    method_name = task["method_name"]
    seed = task["seed"]
    t = task["t"]
    D_t = task["D_t"]
    node_ids_t = task["node_ids_t"]
    y_by_node = task["y_by_node"]
    n = D_t.shape[0]

    ids_str = [str(nid) for nid in node_ids_t]
    missing = [nid for nid in ids_str if nid not in y_by_node]
    if missing:
        raise ValueError(
            f"Node pairing failed (fail-loud, K15): dataset={dataset_name} method={method_name} t={t} - "
            f"{len(missing)} nodes from D_t are missing in the saved trajectory (e.g. {missing[:5]})."
        )
    extra_ids = set(y_by_node.keys()) - set(ids_str)
    if extra_ids:
        raise ValueError(
            f"Node pairing failed (fail-loud, K15): dataset={dataset_name} method={method_name} t={t} - "
            f"the trajectory contains {len(extra_ids)} extra nodes that are not in D_t (e.g. {sorted(extra_ids)[:5]})."
        )

    Y = np.array([y_by_node[nid] for nid in ids_str], dtype=np.float64)
    d_emb = np.sqrt(((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1))
    rank_orig, order_orig = _neighbor_ranks(D_t)
    rank_emb, order_emb = _neighbor_ranks(d_emb)

    metrics: dict[str, Any] = {}
    notes: list[str] = []
    for k in task["neighbor_k"]:
        try:
            tw, cont = _trustworthiness_continuity(rank_orig, order_orig, rank_emb, order_emb, k, n)
        except ValueError as exc:
            tw, cont = float("nan"), float("nan")
            notes.append(f"k={k}: {exc}")
        metrics[f"trustworthiness_k{k}"] = tw
        metrics[f"continuity_k{k}"] = cont

        try:
            jac = knn_jaccard(order_orig, order_emb, k)
        except ValueError as exc:
            jac = float("nan")
            notes.append(f"k={k}: {exc}")
        metrics[f"knn_jaccard_k{k}"] = jac

    return {
        "dataset_name": dataset_name, "method_name": method_name, "seed": seed,
        "status": "ok", "error": "", "wall_time_sec": time.perf_counter() - t0, "embedding": None,
        "metrics": metrics,
        "extra": {
            "lambda": task["lam"], "alpha": task["alpha"], "solver": task["solver"], "t": t,
            "n_nodes": n, "note": "; ".join(notes),
        },
    }


def _holm_correction(p_values: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni correction (step-down) - returns adjusted p-values in
    the SAME order as the `p_values` input (the input does not need to be sorted)."""
    m = len(p_values)
    order = np.argsort(p_values, kind="stable")
    sorted_p = p_values[order]
    adjusted_sorted = np.empty(m, dtype=np.float64)
    running_max = 0.0
    for i in range(m):
        adj = min((m - i) * sorted_p[i], 1.0)
        running_max = max(running_max, adj)
        adjusted_sorted[i] = running_max
    adjusted = np.empty(m, dtype=np.float64)
    adjusted[order] = adjusted_sorted
    return adjusted


def _write_stats(
    cfg: dict[str, Any], cfg_primary: dict[str, Any], experiment_name: str, neighbor_k: list[int], logger,
) -> Path | None:
    """Paired permutation test (unit = snapshot, paired over (dataset,t)) of
    all our methods (solver != 'dtsne') against the dtsne baseline
    `dtsne_lambda{_DTSNE_BASELINE_LAMBDA}` - imports `_paired_permutation_test`
    from exp4_temporal.py (no duplicate test implementation).

    K16 (2026-09-16, author): these tests are EXCLUSIVELY DESCRIPTIVE
    (exploratory) - the independence unit is the SNAPSHOT, not the dataset,
    so with 672 tests in the file the Holm correction with the original
    `exp4_temporal.n_permutations`=1000 gave the smallest attainable
    corrected p-value of 672/1001~0.67 (no test could reject H0 regardless
    of effect size - author, 2026-09-15). The Holm correction OVER THE WHOLE
    FAMILY was therefore REMOVED (see `_write_primary_stats` for the
    confirmatory test with correction only over the 4 primary metrics) -
    each row is labeled with the `test_role` column (value
    `_EXPLORATORY_TEST_ROLE`), the number of permutations increased to
    `exp4_neighbor_metrics.descriptive_n_permutations` (config, independent
    of the primary test)."""
    from src.experiments.exp4_temporal import _paired_permutation_test

    csv_path = results_csv_path(experiment_name)
    if not csv_path.exists():
        logger.warning("No results CSV (%s) - exp4_neighbor_stats.csv is not written.", csv_path)
        return None
    df = pd.read_csv(csv_path)
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        logger.warning("No successful rows in %s - exp4_neighbor_stats.csv is not written.", csv_path)
        return None

    dtsne_lambdas = cfg.get("dtsne", {}).get("lambdas", [])
    if _DTSNE_BASELINE_LAMBDA not in dtsne_lambdas:
        raise ValueError(
            f"exp4_temporal.dtsne.lambdas does not contain the baseline value {_DTSNE_BASELINE_LAMBDA} - "
            "the comparison with dynamic t-SNE (K15) is undefined."
        )

    metric_cols = _metric_columns(neighbor_k)
    if "descriptive_n_permutations" not in cfg_primary or "descriptive_permutation_seed" not in cfg_primary:
        raise KeyError(
            "exp4_neighbor_metrics.descriptive_n_permutations/descriptive_permutation_seed is missing in "
            "config_experiments.yaml (K16 - the descriptive test has its own number of permutations, not "
            "exp4_temporal.n_permutations)."
        )
    n_perm = cfg_primary["descriptive_n_permutations"]
    perm_seed = cfg_primary["descriptive_permutation_seed"]

    rows: list[dict[str, Any]] = []
    for dataset_name, ok_ds in ok.groupby("dataset"):
        baseline = ok_ds[(ok_ds["solver"] == "dtsne") & np.isclose(ok_ds["lambda"], _DTSNE_BASELINE_LAMBDA)]
        if baseline.empty:
            logger.warning(
                "Dataset %s: missing dtsne baseline lambda=%.1f in %s - skipping the permutation test.",
                dataset_name, _DTSNE_BASELINE_LAMBDA, csv_path,
            )
            continue
        baseline_by_t = baseline.set_index("t")

        ours = ok_ds[ok_ds["solver"] != "dtsne"]
        for (lam, alpha, solver), sub in ours.groupby(["lambda", "alpha", "solver"]):
            sub_by_t = sub.set_index("t")
            common_t = sorted(set(sub_by_t.index) & set(baseline_by_t.index))
            if not common_t:
                continue
            for metric in metric_cols:
                a = sub_by_t.loc[common_t, metric].to_numpy(dtype=np.float64)
                b = baseline_by_t.loc[common_t, metric].to_numpy(dtype=np.float64)
                valid = np.isfinite(a) & np.isfinite(b)
                n_paired = int(valid.sum())
                if n_paired == 0:
                    continue
                diff = a[valid] - b[valid]
                observed, p_value = _paired_permutation_test(diff, n_perm, perm_seed)
                rows.append({
                    "dataset": dataset_name, "lambda": lam, "alpha": alpha, "solver": solver,
                    "metric": metric, "baseline": f"dtsne_lambda{_DTSNE_BASELINE_LAMBDA}",
                    "n_paired": n_paired, "mean_diff_vs_baseline": observed, "p_value": p_value,
                })

    if not rows:
        logger.warning("No pairing against the dtsne baseline found - exp4_neighbor_stats.csv is not written.")
        return None

    stats_df = pd.DataFrame(rows)
    # K16 (2026-09-16): NO Holm correction over this (672-row, per-snapshot)
    # family - it would be toothless (see the docstring above) and
    # confusing. Instead, rows are clearly labeled as descriptive/exploratory -
    # see `_write_primary_stats` for the confirmatory test with correction.
    stats_df["test_role"] = _EXPLORATORY_TEST_ROLE

    stats_csv = csv_path.parent / "exp4_neighbor_stats.csv"
    stats_df.to_csv(stats_csv, index=False)
    logger.info(
        "Descriptive (exploratory) permutation test against the dtsne baseline saved: %s (%d rows, n_permutations=%d, "
        "WITHOUT multiple-testing correction - see exp4_neighbor_primary_stats.csv for the confirmatory conclusion).",
        stats_csv, len(stats_df), n_perm,
    )
    return stats_csv


def _load_temporal_medians(mode: str, metrics: list[str]) -> pd.DataFrame:
    """K16 primary test: loads `exp4_temporal_results.csv` (ALL 3 seeds -
    unlike `_write_stats`, which uses `exp4_neighbor_metrics_results.csv`
    with 1 seed), returns a DataFrame with columns
    ['dataset','method','stab',*metrics] medianed OVER SEEDS (same
    aggregation as `export_numbers.py::_add_exp4_dtsne_baseline_numbers`,
    which however further medians over datasets too - the K16 primary test,
    in contrast, KEEPS the dataset unit, see _build_primary_pairs)."""
    exp4_temporal_name = resolve_experiment_name(_EXP4_TEMPORAL_NAME, mode)
    csv_path = results_csv_path(exp4_temporal_name)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} does not exist - first run src\\run_exp4_temporal.bat {mode} "
            "(K16 primary test computes over already saved results, no new fit)."
        )
    df = pd.read_csv(csv_path)
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        raise ValueError(f"{csv_path} has no 'ok' rows - the K16 primary test cannot be computed.")
    missing_cols = [m for m in metrics if m not in ok.columns]
    if missing_cols:
        raise ValueError(
            f"{csv_path} does not contain the expected neighbor-preservation metric columns {missing_cols} - "
            "a new full run of exp4_temporal.py is needed (K15 task B added trust_k<K>/jacc_k<K>)."
        )
    cols = ["stab"] + metrics
    med = ok.groupby(["dataset", "method"], as_index=False)[cols].median()
    return med


def _build_primary_pairs(med: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """For EVERY dataset and EVERY dtsne lambda, finds the point in our
    family with the closest 'stab' value (`exp4_common.nearest_stab_index` -
    the same rule as `export_numbers.py::_add_exp4_dtsne_baseline_numbers`,
    K12a) and records the difference `metric_ours - metric_dtsne` for each
    of `metrics`. Returns a LONG DataFrame, one row per
    (dataset, dtsne_lambda, metric).

    Fail-loud: a dataset with NOT EVEN ONE dtsne method or NOT EVEN ONE
    method of our family is a configuration/data error (not a silent skip)."""
    rows: list[dict[str, Any]] = []
    for dataset_name, sub in med.groupby("dataset"):
        dtsne_rows: dict[float, pd.Series] = {}
        our_rows: dict[tuple[float, str], pd.Series] = {}
        for _, row in sub.iterrows():
            method = row["method"]
            m_dtsne = exp4_common.DTSNE_METHOD_RE.match(method)
            if m_dtsne:
                dtsne_rows[float(m_dtsne.group("lam"))] = row
                continue
            m_ours = exp4_common.OUR_TEMPORAL_METHOD_RE.match(method)
            if m_ours:
                our_rows[(float(m_ours.group("lam")), m_ours.group("solver"))] = row
        if not dtsne_rows:
            raise ValueError(f"Dataset '{dataset_name}': no 'dtsne_lambda*' method in exp4_temporal_results.csv.")
        if not our_rows:
            raise ValueError(
                f"Dataset '{dataset_name}': no method of our family 'lambda*_alpha*_<solver>' in "
                "exp4_temporal_results.csv."
            )
        our_keys = list(our_rows.keys())
        our_stab = np.array([our_rows[k]["stab"] for k in our_keys], dtype=np.float64)
        for dtsne_lam, drow in sorted(dtsne_rows.items()):
            idx = exp4_common.nearest_stab_index(float(drow["stab"]), our_stab)
            matched_lam, matched_solver = our_keys[idx]
            orow = our_rows[(matched_lam, matched_solver)]
            stab_diff_abs = float(abs(float(orow["stab"]) - float(drow["stab"])))
            for metric in metrics:
                dtsne_val = float(drow[metric])
                ours_val = float(orow[metric])
                rows.append({
                    "dataset": dataset_name, "dtsne_lambda": dtsne_lam,
                    "matched_lambda": matched_lam, "matched_solver": matched_solver,
                    "metric": metric, "dtsne_value": dtsne_val, "ours_value": ours_val,
                    # positive = our method is BETTER (higher trust/jacc) than dtsne at comparable stability
                    "diff_ours_minus_dtsne": ours_val - dtsne_val,
                    "dtsne_stab": float(drow["stab"]), "ours_stab": float(orow["stab"]),
                    "stab_diff_abs": stab_diff_abs,
                })
    return pd.DataFrame(rows)


def _primary_test_for_metric(
    diffs_by_dataset: pd.Series, alternative: str, n_boot: int, boot_seed: int,
) -> dict[str, Any]:
    """One confirmatory test for one metric: an exact sign-flip test over
    datasets (`stats_holdout.exact_sign_flip_pvalue`), the median difference
    + the Hodges-Lehmann estimator + a paired percentile bootstrap CI of the
    median (`stats_holdout.bootstrap_ci_paired`, stat_fn=median difference;
    b=zeros, see the `_write_primary_stats` docstring). NaN datasets (e.g.
    K=10 on a small snapshot, K15) are dropped (`.dropna()`), the number
    dropped is recorded, a substitute value is NEVER fabricated."""
    finite = diffs_by_dataset.dropna()
    n_dropped = int(diffs_by_dataset.shape[0] - finite.shape[0])
    diffs = finite.to_numpy(dtype=np.float64)
    n_datasets = int(diffs.shape[0])
    if n_datasets == 0:
        raise ValueError("Primary test: no dataset has a valid (non-NaN) value for this metric.")

    p_value, n_perm_exact = exact_sign_flip_pvalue(diffs, alternative)
    median_diff = float(np.median(diffs))
    hl_estimate = hodges_lehmann(diffs)
    if n_datasets >= 2:
        zeros = np.zeros_like(diffs)
        ci_low, ci_high = bootstrap_ci_paired(diffs, zeros, lambda x, y: float(np.median(x - y)), n_boot, boot_seed)
    else:
        ci_low, ci_high = float("nan"), float("nan")

    return {
        "n_datasets": n_datasets, "n_datasets_dropped_nan": n_dropped,
        "median_diff": median_diff, "hl_estimate": hl_estimate,
        "ci_low": ci_low, "ci_high": ci_high,
        "p_value": p_value, "n_permutations_exact": n_perm_exact,
        "n_wins_ours": int(np.sum(diffs > 0)), "n_wins_dtsne": int(np.sum(diffs < 0)),
        "n_ties": int(np.sum(diffs == 0)),
    }


def _write_primary_stats(cfg_primary: dict[str, Any], mode: str, logger) -> Path | None:
    """K16 (2026-09-16, author) primary confirmatory test: independence unit
    = DATASET (n=number of datasets in exp4_temporal.datasets, 7 in 'full'
    mode). The pairing of our family with the dtsne baseline uses EXACTLY
    the same rule ("nearest stab") as the `numExpFourDtsneDominatedPairs`
    macro in export_numbers.py, imported from `exp4_common.py` (no duplicate
    logic). Per dataset, the MEDIAN difference is taken over all dtsne
    lambdas, then an exact sign-flip test (`stats_holdout.exact_sign_flip_pvalue`)
    over datasets - Holm correction ONLY over `primary_metrics` (unlike
    `_write_stats`, which is purely descriptive/exploratory without correction).

    Returns None (only logs a WARNING) if the input CSV is missing/empty -
    fail-loud exceptions (missing columns, a dataset without a dtsne/our
    method) are NOT caught here, see
    `_load_temporal_medians`/`_build_primary_pairs`."""
    metrics = cfg_primary.get("primary_metrics")
    if not metrics:
        raise KeyError("exp4_neighbor_metrics.primary_metrics is missing in config_experiments.yaml (K16 spec).")
    alternative = cfg_primary.get("primary_alternative")
    if not alternative:
        raise KeyError("exp4_neighbor_metrics.primary_alternative is missing in config_experiments.yaml (K16 spec).")
    if "n_boot" not in cfg_primary or "bootstrap_seed" not in cfg_primary:
        raise KeyError("exp4_neighbor_metrics.n_boot/bootstrap_seed is missing in config_experiments.yaml (K16 spec).")
    n_boot = int(cfg_primary["n_boot"])
    boot_seed = int(cfg_primary["bootstrap_seed"])

    med = _load_temporal_medians(mode, metrics)
    pairs_df = _build_primary_pairs(med, metrics)

    out_dir = results_csv_path(resolve_experiment_name(BASE_EXPERIMENT_NAME, mode)).parent
    pairs_csv = out_dir / "exp4_neighbor_primary_pairs.csv"
    pairs_df.to_csv(pairs_csv, index=False)

    median_abs_stab_diff = float(np.median(pairs_df["stab_diff_abs"].to_numpy(dtype=np.float64)))

    summary_rows: list[dict[str, Any]] = []
    for metric in metrics:
        diffs_by_dataset = pairs_df[pairs_df["metric"] == metric].groupby("dataset")["diff_ours_minus_dtsne"].median()
        result = _primary_test_for_metric(diffs_by_dataset, alternative, n_boot, boot_seed)
        result["metric"] = metric
        summary_rows.append(result)

    raw_pvalues = [row["p_value"] for row in summary_rows]
    holm_adjusted = holm_correction(raw_pvalues)
    for row, p_holm in zip(summary_rows, holm_adjusted):
        row["p_value_holm"] = p_holm
        row["alternative"] = alternative
        row["test_role"] = _PRIMARY_TEST_ROLE
        row["median_abs_stab_diff"] = median_abs_stab_diff
        row["power_caveat"] = (
            f"exact sign-flip test, n_datasets={row['n_datasets']} "
            f"(2^n_datasets={row['n_permutations_exact']} permutations) - a small n limits statistical power; "
            "read together with median_diff/hl_estimate/ci_low/ci_high, not just the p-value."
        )

    stats_df = pd.DataFrame(summary_rows)[[
        "metric", "test_role", "n_datasets", "n_datasets_dropped_nan",
        "median_diff", "hl_estimate", "ci_low", "ci_high",
        "p_value", "p_value_holm", "alternative", "n_permutations_exact",
        "n_wins_ours", "n_wins_dtsne", "n_ties", "median_abs_stab_diff", "power_caveat",
    ]]
    stats_csv = out_dir / "exp4_neighbor_primary_stats.csv"
    stats_df.to_csv(stats_csv, index=False)
    logger.info(
        "K16 primary test saved: %s (%d metrics, n_datasets<=%d) + pairing detail %s (%d rows).",
        stats_csv, len(stats_df), int(pairs_df.groupby("metric")["dataset"].nunique().max()), pairs_csv, len(pairs_df),
    )
    return stats_csv


def main() -> None:
    mode = parse_mode_args(
        "E4 post-hoc (K15): neighborhood preservation metrics (trustworthiness/continuity/kNN Jaccard) "
        "over exp4_trajectories.csv (1 seed)."
    )
    logger = get_logger(BASE_EXPERIMENT_NAME, mode=mode)
    cfg = resolve_experiment_config(_EXP4_TEMPORAL_NAME, mode)
    # K16 (2026-09-16): a dedicated config block for the hierarchical test
    # (primary metrics/alternative/bootstrap + descriptive permutation
    # count) - separate from `exp4_temporal:`, see the config_experiments.yaml
    # comment for 'exp4_neighbor_metrics:'.
    cfg_primary = resolve_experiment_config(BASE_EXPERIMENT_NAME, mode)
    EXPERIMENT_NAME = resolve_experiment_name(BASE_EXPERIMENT_NAME, mode)

    neighbor_k = cfg.get("neighbor_k")
    if not neighbor_k:
        raise KeyError(
            "exp4_temporal.neighbor_k is missing in config_experiments.yaml - the K values for "
            "trustworthiness/continuity/knn_jaccard must be in the config (K15 spec), no magic "
            "numbers in the code."
        )
    column_keys = _metric_columns(neighbor_k) + ["lambda", "alpha", "solver", "t", "n_nodes", "note"]

    # Fail-fast schema check BEFORE loading trajectories/snapshots (K15
    # project rule - a schema mismatch should not be revealed only after the
    # expensive loading of all snapshots of all datasets).
    check_results_schema(EXPERIMENT_NAME, column_keys)

    exp4_temporal_name = resolve_experiment_name(_EXP4_TEMPORAL_NAME, mode)
    trajectories_csv = results_csv_path(exp4_temporal_name).parent / "exp4_trajectories.csv"
    if not trajectories_csv.exists():
        raise FileNotFoundError(
            f"{trajectories_csv} does not exist - first run src\\run_exp4_temporal.bat {mode} "
            "(exp4_neighbor_metrics.py computes over already saved trajectories, it does not fabricate data)."
        )
    traj_df = pd.read_csv(trajectories_csv)
    required_traj_cols = {"dataset", "lambda", "alpha", "solver", "t", "node", "x", "y"}
    missing_traj_cols = required_traj_cols - set(traj_df.columns)
    if missing_traj_cols:
        raise ValueError(f"{trajectories_csv} does not contain the expected columns {missing_traj_cols}.")

    tasks = _build_tasks(cfg, mode, traj_df, neighbor_k)
    logger.info(
        "%s (mode=%s): built %d tasks (dataset x lambda x alpha x solver x snapshot) from %s.",
        EXPERIMENT_NAME, mode, len(tasks), trajectories_csv,
    )

    todo, n_done = filter_already_done(EXPERIMENT_NAME, tasks)
    logger.info("%s: %d tasks already done (skipped), %d new to compute.", EXPERIMENT_NAME, n_done, len(todo))

    n_workers = load_config()["parallel"].get("n_workers")
    n_ok, n_err = run_experiment_grid(EXPERIMENT_NAME, todo, _run_single, column_keys, logger, n_workers=n_workers)
    logger.info("Done. Newly successful: %d, newly failed: %d, skipped: %d.", n_ok, n_err, n_done)

    _write_stats(cfg, cfg_primary, EXPERIMENT_NAME, neighbor_k, logger)
    _write_primary_stats(cfg_primary, mode, logger)

    write_done_file(EXPERIMENT_NAME, logger, extra_info={"mode": mode, "n_tasks": len(tasks)})


if __name__ == "__main__":
    main()
