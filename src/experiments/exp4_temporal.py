# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-09
# License: see the LICENSE file in the repository root
"""
E4 - temporal Sammon: `TemporalSammon` (src/sammon/temporal.py) across all
datasets in `exp4_temporal.datasets` (config_experiments.yaml) - time series
of contact/communication networks aggregated into time bins: SocioPatterns
`primary_school_temporal`, `hospital_temporal`, `high_school_temporal`,
`invs13_temporal`, `sfhh_temporal`, `ht09_temporal`, and the non-SocioPatterns
control `email_eu_core_temporal` (loaders in src/datasets/temporal.py,
per-dataset time_bin_sec/min_snapshot_nodes in exp4_temporal.dataset_params,
see documentation/2026-09-13_e4_rozsireni_datasetu.md) - a lambda grid
(src/common/config.yaml: sammon.temporal.lambda_grid) x alpha.
lambda=0 in this grid IS the baseline described in the spec ("independent
SMACOF layouts with Procrustes alignment, anchor-init without
regularization, CASoN 2013 style") - `TemporalSammon.fit` for lam=0 uses a
warm-start initialization (anchors/neighbor centroids) WITHOUT the quadratic
anchoring term (see src/sammon/temporal.py:TemporalSammon.fit, lam_t=lam),
exactly matching this description, so it is not implemented separately.

External dynamic baseline (2026-09-14, documentation/2026-09-14_e4_dtsne_baseline.md):
"Dynamic t-SNE" (Rauber, Falcao, Telea 2016, DOI 10.2312/eurovisshort.20161164,
src/sammon/dynamic_tsne.py::DynamicTSNE) - methods `dtsne_lambda<L>` for
L in `exp4_temporal.dtsne.lambdas` (common/config.yaml, separate from
`sammon.temporal.lambda_grid` used by the Sammon/SGD family), runs in the
SAME CSV/pipeline across the same datasets/seeds as the SMACOF/SGD family
(column 'method' = 'dtsne_lambda<L>', 'solver' in the exp4_stats.csv
bucketing is 'dtsne', 'alpha' is NaN - dtsne has no alpha weighting). The
schema of the results CSV is UNCHANGED (stab/qual/n_snapshots_used/n_transitions
are the same definitions - qual for dtsne is the UNWEIGHTED scale-invariant
stress `scale_invariant_stress`, not KL divergence, so it is comparable to
the other methods), so no migration/backup of the existing full CSV is needed.

Outputs:
- results/data/exp4_temporal_results.csv (dataset, stab, qual per lambda x alpha x seed,
  resumable; K15 task B 2026-09-15: additionally aggregates (median over snapshots) of the
  neighborhood-preservation metrics 'trust_k<K>'/'cont_k<K>'/'jacc_k<K>' for K from
  exp4_temporal.neighbor_k, ALL seeds - see documentation/2026-09-15_e4_metrika_sousedstvi.md;
  existing full CSV migrated, backup *.bak_20260915_schema)
- results/data/exp4_trajectories.csv (dataset, t, node, x, y, lambda - one representative seed)
- results/data/exp4_stats.csv (a paired permutation test over snapshots, SEPARATELY per dataset,
  lambda vs. the lambda=0 baseline within the same dataset; the 'metric' column has, since K15,
  also included 'trust_k<K>'/'cont_k<K>'/'jacc_k<K>' if available in the .npz cache of both sides)
- results/data/embeddings/exp4_temporal/pertrans/*.npz (K15: additionally per-snapshot arrays
  'trust_k<K>'/'cont_k<K>'/'jacc_k<K>' alongside the original 'stab'/'qual' - backward
  compatible, old npz files without these keys are loaded unchanged, see `_load_pertrans_cache`)
- results/data/exp4_neighbor_metrics_results.csv, exp4_neighbor_stats.csv (K15 task A,
  src/experiments/exp4_neighbor_metrics.py - POST-HOC, trajectory_seed only, see that module)

Run: venv\\python.exe -m src.experiments.exp4_temporal [--quick|--full]
or: src\\run_exp4_temporal.bat [quick|full] (a complete run including dtsne);
for an isolated run of only the dtsne family (e.g. after the Sammon/SGD part
is already done) see src\\run_q1_step4_e4_dtsne.bat - runs the SAME
script/CSV, just a convenience wrapper with its own log/DONE message for
this specific task.
"""
from __future__ import annotations

import functools
import shutil
import sys
import time
import warnings
from multiprocessing import parent_process
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.common.checkpoint import RunKey, append_result, is_done
from src.common.config import ensure_dir, get_path, load_config
from src.common.logging_utils import get_logger
from src.experiments.config_experiments import resolve_experiment_config
from src.experiments.exp_common import (
    check_results_schema,
    filter_already_done,
    parse_mode_args,
    resolve_experiment_name,
    write_done_file,
)

BASE_EXPERIMENT_NAME = "exp4_temporal"


def neighbor_preservation_metrics(D_t: np.ndarray, Y_t: np.ndarray, neighbor_k: list[int]) -> tuple[dict[str, float], list[str]]:
    """Trustworthiness/continuity/kNN Jaccard (Venna & Kaski 2001/2006) for
    ONE snapshot - shared implementation between `exp4_temporal.py` (K15
    task B, all seeds, per-snapshot arrays in the .npz cache) and
    `exp4_neighbor_metrics.py` (K15 task A, post-hoc 1 seed over
    exp4_trajectories.csv) - EXCLUSIVELY existing functions from
    `src/sammon/metrics.py`, no duplicate implementation of the validation
    conditions. An invalid (n, K) combination -> NaN + a note in the second
    return element (never a fabricated value).

    Returns (metrics, notes) - the metric keys are 'trustworthiness_k<K>',
    'continuity_k<K>', 'knn_jaccard_k<K>' for each K in `neighbor_k`."""
    from src.sammon.metrics import _neighbor_ranks, _trustworthiness_continuity, knn_jaccard

    n = D_t.shape[0]
    Y_t = np.asarray(Y_t, dtype=np.float64)
    d_emb = np.sqrt(((Y_t[:, None, :] - Y_t[None, :, :]) ** 2).sum(-1))
    rank_orig, order_orig = _neighbor_ranks(D_t)
    rank_emb, order_emb = _neighbor_ranks(d_emb)

    metrics: dict[str, float] = {}
    notes: list[str] = []
    for k in neighbor_k:
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
    return metrics, notes


def _short_neighbor_key(full_key: str) -> str:
    """Converts the full metric name ('trustworthiness_k5') to the short
    prefix used in the exp4_temporal.py CSV/npz cache ('trust_k5') - see the
    K15 task B spec (example names 'trust_k10'/'jacc_k10'). An unknown
    prefix is a fail-loud error (signals a name change in `metrics.py`/
    `neighbor_preservation_metrics`, not a silent skip)."""
    for prefix, short in (("trustworthiness_", "trust_"), ("continuity_", "cont_"), ("knn_jaccard_", "jacc_")):
        if full_key.startswith(prefix):
            return short + full_key[len(prefix):]
    raise ValueError(f"Unexpected neighborhood-preservation metric key (K15): '{full_key}'.")


def _neighbor_agg_columns(neighbor_k: list[int]) -> list[str]:
    """Names of the aggregated (median over snapshots) neighborhood-preservation
    metric columns in the results CSV, deterministic order (ascending K,
    within each K trust/cont/jacc) - see K15 task B."""
    cols: list[str] = []
    for k in sorted(neighbor_k):
        cols += [f"trust_k{k}", f"cont_k{k}", f"jacc_k{k}"]
    return cols


def _migrate_schema_if_needed(experiment_name: str, column_keys: list[str], logger) -> None:
    """K15 task B: when adding the aggregated neighborhood-preservation
    metric columns (`_neighbor_agg_columns`) to an existing results CSV,
    first backs up the original file (`.bak_20260915_schema`, fail-loud if
    it already exists - no silent overwrite of the backup) and adds the
    missing columns as NaN for ALL existing rows (nothing is computed
    blindly - the values are filled in only by a full rerun, see the
    projectstate.md/CLAUDE.md schema migration rule)."""
    from src.common.checkpoint import results_csv_path

    csv_path = results_csv_path(experiment_name)
    if not csv_path.exists():
        return
    import csv as _csv

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        header = next(_csv.reader(f), None)
    if header is None:
        return
    missing = [c for c in column_keys if c not in header]
    if not missing:
        return

    backup = csv_path.with_name(csv_path.name + ".bak_20260915_schema")
    if backup.exists():
        raise FileExistsError(
            f"Backup {backup} already exists (the K15 schema migration already ran) - remove it "
            "manually if you want to repeat the migration, so nothing is silently overwritten."
        )
    shutil.copy(csv_path, backup)
    df = pd.read_csv(csv_path)
    for col in missing:
        df[col] = np.nan
    df.to_csv(csv_path, index=False)
    logger.warning(
        "K15 schema migration %s: added %d new neighborhood-preservation metric columns (%s) as NaN "
        "for existing rows, backup saved to %s. Values are filled in only by a full rerun.",
        csv_path, len(missing), missing, backup,
    )


@functools.lru_cache(maxsize=None)
def _load_snapshots(dataset_name: str, time_bin_sec: float, min_snapshot_nodes: int, mode: str | None = None) -> tuple[list[np.ndarray], list[list]]:
    """Loads the time series, converts each window to a (largest connected
    component) shortest-path distance matrix, and skips empty/too-small
    windows (e.g. overnight breaks without contact). Returns
    (list_of_D, node_ids_per_snapshot).

    The result is cached (`lru_cache`) by (dataset_name, time_bin_sec,
    min_snapshot_nodes) - `main()` calls this function repeatedly for each
    (lambda, alpha, solver, seed) combination, but the input snapshots are
    always the same (deterministic) for a given dataset+parameters, so
    recomputing the shortest-path distances repeatedly would be wasteful
    (see the Speed rule).
    """
    import networkx as nx

    from src.datasets.graph_distance import shortest_path_distance
    from src.datasets.registry import load_dataset

    ds = load_dataset(dataset_name, time_bin_sec=time_bin_sec)
    snapshots = ds.meta["snapshots"]

    list_of_D: list[np.ndarray] = []
    node_ids: list[list] = []
    n_skipped = 0
    for g in snapshots:
        active = [n for n, d in g.degree() if d > 0]
        sub = g.subgraph(active)
        components = list(nx.connected_components(sub))
        if not components:
            n_skipped += 1
            continue
        largest = max(components, key=len)
        if len(largest) < min_snapshot_nodes:
            n_skipped += 1
            continue
        lcc = sub.subgraph(largest).copy()
        ids = list(lcc.nodes())
        D = shortest_path_distance(lcc)
        list_of_D.append(D)
        node_ids.append(ids)

    if len(list_of_D) < 2:
        raise ValueError(
            f"After filtering out empty/small windows, only {len(list_of_D)} usable snapshots remained "
            f"(out of {len(snapshots)}) - at least 2 are needed for temporal analysis. Reduce "
            "min_snapshot_nodes or increase time_bin_sec in config_experiments.yaml."
        )

    # A known phenomenon (handled by another agent in the solver, only
    # logged here): between consecutive usable snapshots there is sometimes
    # no shared node at all (e.g. the whole active circle of people changes
    # overnight/over the weekend) - Procrustes alignment of such a
    # transition then has no reference point.
    n_no_common_transitions = sum(
        1 for t in range(1, len(node_ids)) if not (set(node_ids[t]) & set(node_ids[t - 1]))
    )
    # Log only in the MAIN process: since parallelization (2026-09-16), this
    # function is also called by ProcessPoolExecutor workers, and
    # `get_logger` would create its own file
    # results/logs/exp4_temporal_<timestamp>.log in each of them. The main
    # process loads (and logs) the snapshots of all datasets in advance, see
    # `_preload_snapshots` called before the pool starts.
    if parent_process() is None:
        logger = get_logger(BASE_EXPERIMENT_NAME, mode=mode)
        logger.info(
            "%s (bin=%ss, min_nodes=%d): %d usable snapshots (out of %d, %d skipped), "
            "%d transitions with no node shared with the previous snapshot.",
            dataset_name, time_bin_sec, min_snapshot_nodes, len(list_of_D), len(snapshots), n_skipped,
            n_no_common_transitions,
        )
    return list_of_D, node_ids


def _preload_snapshots(tasks: list[dict[str, Any]], logger) -> None:
    """Loads the snapshots of each used dataset once in the MAIN process.

    Two reasons: (1) fail-fast - a missing/broken dataset crashes BEFORE the
    workers start, not after hours of computation (same rule as the CSV
    schema check); (2) log lines about the number of usable snapshots are
    produced exactly once and in the correct file (workers already skip
    `_load_snapshots` logging)."""
    seen: set[tuple[str, float, int]] = set()
    for task in tasks:
        sig = (task["dataset_name"], task["time_bin_sec"], task["min_snapshot_nodes"])
        if sig in seen:
            continue
        seen.add(sig)
        _load_snapshots(task["dataset_name"], task["time_bin_sec"], task["min_snapshot_nodes"], mode=task.get("mode"))
    logger.info("Preloaded %d datasets (snapshots checked before workers start).", len(seen))


def _per_transition_metrics(ts) -> tuple[np.ndarray, np.ndarray]:
    """Recomputes per-snapshot/per-transition stability and quality values
    (instead of just the aggregated mean/median from
    TemporalSammon.stability()/.quality()) - needed for the permutation test
    (unit = snapshot/transition, spec sections 6.5/9). Uses only PUBLIC
    TemporalSammon attributes (Y_list_, node_ids_, history_list_) and the
    same Procrustes alignment function as src/sammon/temporal.py (imported,
    that file is not modified - it is being developed concurrently by
    another agent).
    """
    # src/sammon/temporal_metrics.py is the canonical (shared)
    # implementation of Procrustes alignment (see TemporalSammon.stability(),
    # a thin wrapper around it) - re-exported without an underscore, unlike
    # the earlier private copy directly in temporal.py
    from src.sammon.temporal_metrics import orthogonal_procrustes_no_scale

    stab_per_transition: list[float] = []
    for t in range(1, len(ts.Y_list_)):
        Y_prev, ids_prev = ts.Y_list_[t - 1], ts.node_ids_[t - 1]
        Y_cur, ids_cur = ts.Y_list_[t], ts.node_ids_[t]
        id_to_prev = {nid: i for i, nid in enumerate(ids_prev)}
        common_cur_idx = [i for i, nid in enumerate(ids_cur) if nid in id_to_prev]
        if not common_cur_idx:
            continue
        common_prev_idx = [id_to_prev[ids_cur[i]] for i in common_cur_idx]
        Y_cur_common = orthogonal_procrustes_no_scale(Y_cur[common_cur_idx], Y_prev[common_prev_idx])
        disp = np.linalg.norm(Y_cur_common - Y_prev[common_prev_idx], axis=1)
        n_cur = Y_cur.shape[0]
        if n_cur > 1:
            mask_full = ~np.eye(n_cur, dtype=bool)
            d_cur = np.sqrt(((Y_cur[:, None, :] - Y_cur[None, :, :]) ** 2).sum(-1))
            L_t = float(d_cur[mask_full].mean())
        else:
            L_t = 1.0
        stab_per_transition.append(float(disp.mean() / L_t))

    qual_per_snapshot = np.array([hist["stress"][-1] for hist in ts.history_list_], dtype=np.float64)
    return np.array(stab_per_transition, dtype=np.float64), qual_per_snapshot


def _compute_neighbor_arrays(list_of_D: list[np.ndarray], Y_list: list[np.ndarray], neighbor_k: list[int]) -> dict[str, np.ndarray]:
    """K15 task B: per-snapshot arrays of the neighborhood-preservation
    metrics (`neighbor_preservation_metrics`) over the WHOLE time series,
    converted to the short keys (`_short_neighbor_key`) used both in the
    .npz cache and the results CSV. `list_of_D[t]`/`Y_list[t]` must have the
    same node order (guaranteed by `TemporalSammon.fit`/`DynamicTSNE.fit`
    iterating directly over these two equal-length/equally-ordered
    sequences - see their `fit`)."""
    if len(list_of_D) != len(Y_list):
        raise ValueError(f"list_of_D ({len(list_of_D)}) and Y_list ({len(Y_list)}) have different lengths - a snapshot mismatch.")
    per_t: dict[str, list[float]] = {}
    for D_t, Y_t in zip(list_of_D, Y_list):
        metrics, _notes = neighbor_preservation_metrics(D_t, Y_t, neighbor_k)
        for full_key, val in metrics.items():
            per_t.setdefault(_short_neighbor_key(full_key), []).append(val)
    return {k: np.array(v, dtype=np.float64) for k, v in per_t.items()}


def _median_neighbor_metrics(neighbor_arrays: dict[str, np.ndarray]) -> dict[str, float]:
    """Median over snapshots for each key in `neighbor_arrays` (an aggregate
    for a row of the results CSV) - NaN if ALL snapshots for the given K are
    undefined (see `neighbor_preservation_metrics`), never a fabricated
    substitute value."""
    out: dict[str, float] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # "All-NaN slice" - expected, see NaN above
        for key, arr in neighbor_arrays.items():
            out[key] = float(np.nanmedian(arr)) if arr.size > 0 else float("nan")
    return out


def _run_single_sammon(task: dict[str, Any]) -> dict[str, Any]:
    """One TemporalSammon (SMACOF/SGD family) run over the whole time series
    for the given (lambda, alpha, seed)."""
    from src.common.seeding import set_seed
    from src.sammon.temporal import TemporalSammon

    dataset_name = task["dataset_name"]
    lam = task["lam"]
    alpha = task["alpha"]
    solver = task["solver"]
    seed = task["seed"]
    method_name = f"lambda{lam}_alpha{alpha}_{solver}"

    result: dict[str, Any] = {"dataset_name": dataset_name, "method_name": method_name, "seed": seed}
    t0 = time.perf_counter()
    try:
        set_seed(seed)
        list_of_D, node_ids = _load_snapshots(dataset_name, task["time_bin_sec"], task["min_snapshot_nodes"], mode=task.get("mode"))
        sgd_cfg = task["sgd_cfg"]
        ts = TemporalSammon(
            n_components=task["n_components"], max_iter=task["max_iter"], tol=task["tol"], seed=seed,
            solver=solver,
            # SGD anchor step (section 6.3): gamma_mode='fixed' per the
            # recommendation of the underlying kernel (eta_t would weaken
            # the anchoring force too much for temporal anchoring under
            # strong annealing toward the end of the run)
            sgd_epochs=sgd_cfg["epochs"], sgd_mu_max=sgd_cfg["mu_max"], sgd_pairs_per_node=sgd_cfg["pairs_per_node"],
            sgd_eps_anneal=sgd_cfg["eps_anneal"], sgd_variant="stabilized", gamma_mode="fixed", gamma_fixed=0.1,
        )
        ts.fit(list_of_D, node_ids, lam=lam, alpha=alpha)
        stab = ts.stability()
        qual = ts.quality()
        stab_per_t, qual_per_t = _per_transition_metrics(ts)
        # K15 task B: neighborhood-preservation metrics for ALL seeds (not
        # just trajectory_seed as in exp4_neighbor_metrics.py task A) - the
        # same neighbor_preservation_metrics function, key exp4_temporal.neighbor_k.
        neighbor_arrays = _compute_neighbor_arrays(list_of_D, ts.Y_list_, task["neighbor_k"])

        result["status"] = "ok"
        result["error"] = ""
        result["wall_time_sec"] = time.perf_counter() - t0
        result["embedding"] = None  # embeddings are saved separately to exp4_trajectories.csv (trajectory_seed only)
        result["metrics"] = {
            "stab": stab, "qual": qual, "n_snapshots_used": len(list_of_D), "n_transitions": len(stab_per_t),
            **_median_neighbor_metrics(neighbor_arrays),
        }
        result["extra"] = {"lam": lam, "alpha": alpha, "solver": solver}
        # 2026-09-16 (parallelization): only SERIALIZABLE data (a list of
        # Y_t arrays) is returned from the worker, NOT the fitted
        # TemporalSammon/DynamicTSNE object - that would have to be pickled
        # in its entirety (incl. the input distance matrices) by
        # ProcessPoolExecutor.map.
        result["_Y_list"] = [np.asarray(Y) for Y in ts.Y_list_] if seed == task["trajectory_seed"] else None
        result["_node_ids"] = node_ids if seed == task["trajectory_seed"] else None
        result["_stab_per_t"] = stab_per_t
        result["_qual_per_t"] = qual_per_t
        result["_neighbor_arrays"] = neighbor_arrays
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["wall_time_sec"] = time.perf_counter() - t0
        result["embedding"] = None
        result["metrics"] = {}
        result["extra"] = {"lam": lam, "alpha": alpha, "solver": solver}
        result["_Y_list"] = None
        result["_node_ids"] = None
        result["_stab_per_t"] = np.array([])
        result["_qual_per_t"] = np.array([])
        result["_neighbor_arrays"] = {}
    return result


def _run_single_dtsne(task: dict[str, Any]) -> dict[str, Any]:
    """One DynamicTSNE (external baseline, K12) run over the whole time
    series for the given (lambda_dt, seed) - see src/sammon/dynamic_tsne.py."""
    from src.common.seeding import set_seed
    from src.sammon.dynamic_tsne import DynamicTSNE

    dataset_name = task["dataset_name"]
    lam_dt = task["lam"]
    seed = task["seed"]
    method_name = f"dtsne_lambda{lam_dt}"

    result: dict[str, Any] = {"dataset_name": dataset_name, "method_name": method_name, "seed": seed}
    t0 = time.perf_counter()
    try:
        set_seed(seed)
        list_of_D, node_ids = _load_snapshots(dataset_name, task["time_bin_sec"], task["min_snapshot_nodes"], mode=task.get("mode"))
        ts = DynamicTSNE(
            n_components=task["n_components"], perplexity=task["dtsne_perplexity"],
            max_iter=task["dtsne_max_iter"], seed=seed, dtsne_cfg=task["dtsne_cfg"],
        )
        ts.fit(list_of_D, node_ids, lam_dt=lam_dt)
        stab = ts.stability()
        qual = ts.quality()
        stab_per_t, qual_per_t = _per_transition_metrics(ts)
        # K15 task B: the same neighborhood-preservation metric also for the
        # dtsne family (the same exp4_temporal.neighbor_k key, the same
        # shared function).
        neighbor_arrays = _compute_neighbor_arrays(list_of_D, ts.Y_list_, task["neighbor_k"])

        result["status"] = "ok"
        result["error"] = ""
        result["wall_time_sec"] = time.perf_counter() - t0
        result["embedding"] = None
        result["metrics"] = {
            "stab": stab, "qual": qual, "n_snapshots_used": len(list_of_D), "n_transitions": len(stab_per_t),
            **_median_neighbor_metrics(neighbor_arrays),
        }
        # 'alpha' makes no sense for dtsne (no alpha weighting) - a NaN
        # sentinel, 'solver'='dtsne' distinguishes the family in the
        # exp4_relative.py/exp4_stats.csv bucketing
        result["extra"] = {"lam": lam_dt, "alpha": float("nan"), "solver": "dtsne"}
        # 2026-09-16 (parallelization): only SERIALIZABLE data (a list of
        # Y_t arrays) is returned from the worker, NOT the fitted
        # TemporalSammon/DynamicTSNE object - that would have to be pickled
        # in its entirety (incl. the input distance matrices) by
        # ProcessPoolExecutor.map.
        result["_Y_list"] = [np.asarray(Y) for Y in ts.Y_list_] if seed == task["trajectory_seed"] else None
        result["_node_ids"] = node_ids if seed == task["trajectory_seed"] else None
        result["_stab_per_t"] = stab_per_t
        result["_qual_per_t"] = qual_per_t
        result["_neighbor_arrays"] = neighbor_arrays
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["wall_time_sec"] = time.perf_counter() - t0
        result["embedding"] = None
        result["metrics"] = {}
        result["extra"] = {"lam": lam_dt, "alpha": float("nan"), "solver": "dtsne"}
        result["_Y_list"] = None
        result["_node_ids"] = None
        result["_stab_per_t"] = np.array([])
        result["_qual_per_t"] = np.array([])
        result["_neighbor_arrays"] = {}
    return result


def _run_single(task: dict[str, Any]) -> dict[str, Any]:
    """Dispatches by the method family (K12: 'sammon' = TemporalSammon
    SMACOF/SGD, 'dtsne' = the DynamicTSNE external baseline) - both families
    share the same results schema (stab, qual, n_snapshots_used, n_transitions)."""
    if task["family"] == "dtsne":
        return _run_single_dtsne(task)
    return _run_single_sammon(task)


def _pertrans_cache_path(key: RunKey) -> Path:
    """Path to the .npz cache with per-transition/per-snapshot arrays (stab,
    qual) for a given run - needed for the permutation test even after a
    resume, when the run was not recomputed in this session (only loaded
    from the checkpoint as 'done'). `key.experiment` already contains any
    'smoke/' prefix, so the cache is automatically separated the same way as
    the results CSV."""
    d = get_path("embeddings_dir") / key.experiment / "pertrans"
    ensure_dir(d)
    return d / f"{key.dataset}__{key.method}__{key.seed}.npz"


def _save_pertrans_cache(
    key: RunKey, stab_per_t: np.ndarray, qual_per_t: np.ndarray, neighbor_arrays: dict[str, np.ndarray] | None = None,
) -> None:
    """Saves per-snapshot/per-transition arrays to the .npz cache: 'stab',
    'qual' (original, unchanged) and, from K15 (task B), optionally
    'trust_k<K>'/'cont_k<K>'/'jacc_k<K>' neighborhood-preservation metrics
    (`neighbor_arrays`, if available) - see `_load_pertrans_cache` for
    backward compatibility."""
    payload: dict[str, np.ndarray] = {"stab": stab_per_t, "qual": qual_per_t}
    if neighbor_arrays:
        payload.update(neighbor_arrays)
    np.savez(_pertrans_cache_path(key), **payload)


def _load_pertrans_cache(key: RunKey) -> dict[str, np.ndarray] | None:
    """Loads ALL arrays stored in the .npz cache as a dict. K15: old npz
    files (before task B) contain only 'stab'/'qual' - the returned dict
    then simply does not contain the keys 'trust_k<K>'/'cont_k<K>'/
    'jacc_k<K>' (a missing key = the metric is not available for this run,
    the caller uses `dict.get`, NEVER crashes on a missing key)."""
    p = _pertrans_cache_path(key)
    if not p.exists():
        return None
    with np.load(p) as data:
        return {k: data[k] for k in data.files}


def _paired_permutation_test(diff: np.ndarray, n_permutations: int, seed: int) -> tuple[float, float]:
    """Paired sign-flip Monte-Carlo permutation test on the array of paired
    differences `diff` (independence unit = snapshot/transition, spec
    section 9). Returns (observed mean difference, two-sided p-value).

    K16 (2026-09-15, fix): the standard Monte-Carlo p-value estimate
    `p = (1 + count) / (1 + n_permutations)` (conservative convention - same
    formula/terminology as `stats_holdout.py::mc_sign_flip_pvalue` and
    `exp9_metric_fidelity_stats.py::sign_flip_test`, consistent across the
    project). The original formula `count / n_permutations` could return
    EXACTLY p=0.0 (a statistically incorrect claim - seen e.g. in
    results/data/smoke/exp4_neighbor_stats.csv before this fix), the new
    formula guarantees p > 0 for any finite `n_permutations`."""
    diff = np.asarray(diff, dtype=np.float64)
    observed = float(diff.mean())
    if len(diff) == 0:
        raise ValueError("Empty array of paired differences - the permutation test cannot be performed.")
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, len(diff)))
    null_dist = (signs * diff[None, :]).mean(axis=1)
    count = int(np.sum(np.abs(null_dist) >= abs(observed)))
    p_value = float(1 + count) / float(1 + n_permutations)
    return observed, p_value


def _compute_stats_coverage(experiment_name: str) -> tuple[int, int]:
    """K16 (2026-09-15, protection against result loss): computes
    (n_with_cache, n_ok_total) - how many rows with `status='ok'` in the
    results CSV have a corresponding .npz pertrans cache available
    (`_pertrans_cache_path`).

    Reason: the FULL npz cache can be lost/deleted independently of the
    results CSV (see the 2026-09-15 incident, the cache was deleted during
    K15 smoke verification) - without this check, a further (even partial)
    run could overwrite a valid `exp4_stats.csv` with incomplete statistics
    built only from whatever happened to remain in the cache."""
    from src.common.checkpoint import results_csv_path

    csv_path = results_csv_path(experiment_name)
    if not csv_path.exists():
        return 0, 0
    df = pd.read_csv(csv_path)
    if "status" not in df.columns:
        return 0, 0
    ok = df[df["status"] == "ok"]
    n_ok = len(ok)
    n_with_cache = 0
    for row in ok.itertuples(index=False):
        key = RunKey(experiment_name, str(row.dataset), str(row.method), int(row.seed))
        if _pertrans_cache_path(key).exists():
            n_with_cache += 1
    return n_with_cache, n_ok


def _finalize_stats_csv(experiment_name: str, stats_rows: list[dict[str, Any]], logger) -> Path:
    """K16 (2026-09-15, protection against result loss): writes `stats_rows`
    to `exp4_stats.csv` (full overwrite) ONLY when there is 100% npz cache
    coverage for all 'ok' runs in the results CSV (`_compute_stats_coverage`) -
    otherwise (partial coverage, e.g. after cache loss - the 2026-09-15
    incident) writes to `exp4_stats_partial.csv` and LEAVES the existing
    (possibly more complete/valid) `exp4_stats.csv` UNCHANGED. Returns the
    path to the file it actually wrote to."""
    from src.common.checkpoint import results_csv_path

    n_with_cache, n_ok_total = _compute_stats_coverage(experiment_name)
    coverage_pct = 100.0 * n_with_cache / n_ok_total if n_ok_total > 0 else 0.0
    stats_dir = results_csv_path(experiment_name).parent
    if n_ok_total > 0 and n_with_cache == n_ok_total:
        stats_csv = stats_dir / "exp4_stats.csv"
        pd.DataFrame(stats_rows).to_csv(stats_csv, index=False)
        logger.info(
            "Permutation test saved: %s (%d rows, npz cache coverage 100%% = %d/%d 'ok' runs).",
            stats_csv, len(stats_rows), n_with_cache, n_ok_total,
        )
        return stats_csv

    stats_csv_partial = stats_dir / "exp4_stats_partial.csv"
    pd.DataFrame(stats_rows).to_csv(stats_csv_partial, index=False)
    logger.warning(
        "!!! K16 PROTECTION: npz pertrans cache coverage only %d/%d (%.1f%%) of successful ('ok') runs - "
        "the existing full exp4_stats.csv is NOT OVERWRITTEN (it might be more complete/valid than "
        "this partial computation). Incomplete statistics saved to %s. A full exp4_stats.csv requires "
        "restoring/recomputing the missing .npz cache for ALL 'ok' runs (e.g. by removing the "
        "affected rows from the results CSV and recomputing them).",
        n_with_cache, n_ok_total, coverage_pct, stats_csv_partial,
    )
    return stats_csv_partial


def main() -> None:
    mode = parse_mode_args("E4: temporal alpha-Sammon (lambda grid) on SocioPatterns temporal datasets.")
    # K12 (2026-09-14, fix during dtsne baseline verification): `mode=mode`
    # MUST be passed right at the first logger initialization (see the
    # `get_logger` docstring - a handler is created only on the first call
    # for a given logger name, further calls with the same name return the
    # already-initialized logger unchanged path) - without this parameter, a
    # smoke/quick run would ALWAYS write the log to results/logs/ ('full'
    # mode), which violates the "Separate smoke/quick from full outputs"
    # rule (~/.claude/CLAUDE.md).
    logger = get_logger(BASE_EXPERIMENT_NAME, mode=mode)
    cfg = resolve_experiment_config(BASE_EXPERIMENT_NAME, mode)
    EXPERIMENT_NAME = resolve_experiment_name(BASE_EXPERIMENT_NAME, mode)
    # lambda_grid: for smoke it can be overridden directly in
    # config_experiments.yaml (exp4_temporal.smoke.lambda_grid), otherwise
    # (quick/full) sammon.temporal.lambda_grid from the main config.yaml is
    # used (a section maintained concurrently by another agent, only read here)
    lambda_grid = cfg.get("lambda_grid", load_config()["sammon"]["temporal"]["lambda_grid"])
    sgd_cfg = load_config()["sammon"]["sgd"]
    # K15 (task B): K values for the neighborhood-preservation metrics -
    # SHARED key with exp4_neighbor_metrics.py (post-hoc task A), no
    # duplication of the K grid.
    neighbor_k = cfg.get("neighbor_k")
    if not neighbor_k:
        raise KeyError(
            "exp4_temporal.neighbor_k is missing in config_experiments.yaml - the K values for "
            "trustworthiness/continuity/knn_jaccard must be in the config (K15 spec), no magic "
            "numbers in the code."
        )
    # K12 (documentation/2026-09-14_e4_dtsne_baseline.md): the external
    # "dynamic t-SNE" baseline. `perplexity` is DELIBERATELY its OWN key
    # `exp4_temporal.dtsne.perplexity` (NOT shared with
    # `methods.tsne.perplexity` used in E1) - E1 works on whole static
    # graphs (on the order of hundreds to tens of thousands of nodes), while
    # E4 snapshots, after `min_snapshot_nodes` filtering, have only ~10-20
    # to hundreds of nodes (see the `dataset_params` comment above); the
    # shared E1 default perplexity=30 is invalid for the smallest allowed
    # snapshots (min_snapshot_nodes) (requires perplexity < n, see
    # `_binary_search_perplexity`) and would be fail-loud rejected precisely
    # on those smallest snapshots - hence a dedicated config key, safely
    # tuned to `min_snapshot_nodes` (n >> perplexity); no silent fallback to
    # the E1 value. max_iter has its own key there too (smoke/quick use a
    # shorter run, see config_experiments.yaml).
    dtsne_grid_cfg = cfg.get("dtsne", {})
    dtsne_lambdas = dtsne_grid_cfg.get("lambdas", [])
    if dtsne_lambdas and "perplexity" not in dtsne_grid_cfg:
        raise KeyError(
            "exp4_temporal.dtsne.perplexity is missing in config_experiments.yaml - it is required "
            "explicitly (must not be silently taken from methods.tsne.perplexity, see the comment above)."
        )
    if dtsne_lambdas and "max_iter" not in dtsne_grid_cfg:
        raise KeyError("exp4_temporal.dtsne.max_iter is missing in config_experiments.yaml - it is required explicitly.")
    dtsne_perplexity = dtsne_grid_cfg.get("perplexity")
    dtsne_max_iter = dtsne_grid_cfg.get("max_iter")
    dtsne_cfg = load_config()["sammon"]["temporal"]["dtsne"]

    tasks: list[dict[str, Any]] = []
    for dataset_name in cfg["datasets"]:
        if dataset_name not in cfg["dataset_params"]:
            raise KeyError(
                f"exp4_temporal.dataset_params in config_experiments.yaml has no entry for dataset "
                f"'{dataset_name}' (listed in exp4_temporal.datasets) - add time_bin_sec/min_snapshot_nodes."
            )
        ds_params = cfg["dataset_params"][dataset_name]
        for lam in lambda_grid:
            for alpha in cfg["alphas"]:
                for solver in cfg["solvers"]:
                    for seed in cfg["seeds"]:
                        tasks.append({
                            "family": "sammon",
                            "mode": mode,
                            "dataset_name": dataset_name, "lam": lam, "alpha": alpha, "solver": solver, "seed": seed,
                            "n_components": cfg["n_components"], "max_iter": cfg["max_iter"], "tol": cfg["tol"],
                            "time_bin_sec": ds_params["time_bin_sec"], "min_snapshot_nodes": ds_params["min_snapshot_nodes"],
                            "trajectory_seed": cfg["trajectory_seed"], "method_name": f"lambda{lam}_alpha{alpha}_{solver}",
                            "sgd_cfg": sgd_cfg, "neighbor_k": neighbor_k,
                        })
        for lam_dt in dtsne_lambdas:
            for seed in cfg["seeds"]:
                tasks.append({
                    "family": "dtsne",
                    "mode": mode,
                    "dataset_name": dataset_name, "lam": lam_dt, "alpha": float("nan"), "solver": "dtsne", "seed": seed,
                    "n_components": cfg["n_components"],
                    "time_bin_sec": ds_params["time_bin_sec"], "min_snapshot_nodes": ds_params["min_snapshot_nodes"],
                    "trajectory_seed": cfg["trajectory_seed"], "method_name": f"dtsne_lambda{lam_dt}",
                    "dtsne_perplexity": dtsne_perplexity, "dtsne_max_iter": dtsne_max_iter, "dtsne_cfg": dtsne_cfg,
                    "neighbor_k": neighbor_k,
                })

    # K15 (task B): new aggregated neighborhood-preservation metric columns
    # (median over snapshots) - the columns are unchanged for the dtsne
    # family too (see method_name/'solver'='dtsne' above).
    column_keys = ["stab", "qual", "n_snapshots_used", "n_transitions"] + _neighbor_agg_columns(neighbor_k)

    # K15 schema migration: existing full/quick/smoke CSV files from BEFORE
    # task B lack the new columns - they are added as NaN (backup
    # .bak_20260915_schema, no blind computation, see
    # `_migrate_schema_if_needed`). MUST be BEFORE `check_results_schema`
    # (K12 rule: fail-fast schema check BEFORE the run starts), otherwise
    # the fail-loud check would also crash a run that the migration itself would fix.
    _migrate_schema_if_needed(EXPERIMENT_NAME, column_keys, logger)
    check_results_schema(EXPERIMENT_NAME, column_keys)

    todo, n_done = filter_already_done(EXPERIMENT_NAME, tasks)
    # Fail-fast before the workers start (see `_preload_snapshots`). Only
    # after `filter_already_done`, so that for a fully completed run (todo
    # empty) shortest paths are not needlessly computed.
    if todo:
        _preload_snapshots(todo, logger)
    logger.info("%s (mode=%s): %d combinations already done (skipped), %d new to compute.", EXPERIMENT_NAME, mode, n_done, len(todo))

    # Parallel run over (dataset, method, seed) - 2026-09-16, replaces the
    # original sequential loop (the full grid of 441 runs took 3.5 h and
    # used none of the 12 allowed workers; measurement in projectstate.md
    # 2026-09-15). Tasks are independent: each loads its own snapshots
    # (`_load_snapshots`, lru_cache within the worker) and fits on its own.
    # Uses `run_parallel_map` = an order-preserving ProcessPoolExecutor.map
    # (never as_completed - determinism of the CSV write order).
    # Collection of side outputs remains in the MAIN process: the worker
    # returns only arrays (`_Y_list`, `_stab_per_t`, `_qual_per_t`,
    # `_neighbor_arrays`) and all WRITES (append_result, npz cache,
    # exp4_trajectories.csv) are done by the main process right after each
    # completed run - the run thus remains resumable even if interrupted midway.
    n_ok, n_err = 0, 0
    from src.common.checkpoint import results_csv_path

    trajectories_csv = results_csv_path(EXPERIMENT_NAME).parent / "exp4_trajectories.csv"
    ensure_dir(trajectories_csv.parent)

    from src.common.logging_utils import wall_clock
    from src.common.parallel import resolve_n_workers, run_parallel_map
    from src.common.progress import progress_iter

    # `_progress_log_policy` is an internal function of a sibling module in
    # the same package - reading the logging frequency from the config is
    # deliberately NOT duplicated.
    from src.experiments.exp_common import _progress_log_policy, format_progress_line, keep_system_awake, should_log_progress

    n_workers_resolved = resolve_n_workers(None)
    every_below, every = _progress_log_policy()
    logger.info("%s: %d runs on %d workers (parallel.n_workers).", EXPERIMENT_NAME, len(todo), n_workers_resolved)
    wall_sum_sec = 0.0
    t_start = time.perf_counter()

    with keep_system_awake(), wall_clock(logger, f"{EXPERIMENT_NAME} ({len(todo)} runs)"):
        results_iter = run_parallel_map(_run_single, todo)
        for i, result in enumerate(progress_iter(results_iter, desc=EXPERIMENT_NAME, total=len(todo)), start=1):
            key = RunKey(EXPERIMENT_NAME, result["dataset_name"], result["method_name"], result["seed"])
            row = {k: np.nan for k in column_keys}
            row.update(result.get("metrics", {}) or {})
            row["status"] = result["status"]
            row["error"] = result["error"]
            row["wall_time_sec"] = result["wall_time_sec"]
            append_result(key, row)

            if result["status"] == "ok":
                n_ok += 1
                lam, alpha, solver = result["extra"]["lam"], result["extra"]["alpha"], result["extra"]["solver"]
                # per-transition/per-snapshot arrays are saved to disk (NOT
                # just kept in the memory of this run), so the permutation
                # test below works correctly even after a resume (when most
                # combinations were already done in a previous run and are
                # not recomputed in this session)
                _save_pertrans_cache(key, result["_stab_per_t"], result["_qual_per_t"], result.get("_neighbor_arrays", {}))

                if result["_Y_list"] is not None:
                    traj_rows = []
                    for t, (Y_t, ids_t) in enumerate(zip(result["_Y_list"], result["_node_ids"])):
                        for node_idx, node_id in enumerate(ids_t):
                            traj_rows.append({
                                "dataset": result["dataset_name"], "lambda": lam, "alpha": alpha, "solver": solver,
                                "t": t, "node": node_id, "x": Y_t[node_idx, 0], "y": Y_t[node_idx, 1],
                            })
                    pd.DataFrame(traj_rows).to_csv(trajectories_csv, mode="a", header=not trajectories_csv.exists(), index=False)
            else:
                n_err += 1
                logger.warning("Temporal run failed: %s -> %s", key.as_tuple(), result["error"])

            # Progress log with an ETA computed from WORKER time
            # (wall_time_sec) and the actual number of workers - the same
            # policy as for other experiments (`_consume_results` in exp_common.py).
            wall_sum_sec += float(result["wall_time_sec"])
            if should_log_progress(i, len(todo), every_below, every):
                logger.info(format_progress_line(i, len(todo), result, time.perf_counter() - t_start, wall_sum_sec, n_workers_resolved))

    logger.info("Done. Newly successful: %d, newly failed: %d, skipped: %d.", n_ok, n_err, n_done)

    # permutation test: each (lambda,alpha) != baseline (lambda=0, same
    # alpha) against the baseline, paired by (seed, snapshot/transition) -
    # see _paired_permutation_test. Reconstructed from ALL successfully
    # completed combinations (checkpoint + npz cache), NOT just those
    # computed in this specific run - otherwise exp4_stats.csv after a
    # resume (when most are already done) would contain only a fraction of
    # the data. The key includes `dataset_name` - each dataset has a
    # different number/size of snapshots and measures different phenomena,
    # so snapshots/transitions of different datasets are NOT mutually
    # interchangeable units, and the permutation test is computed
    # SEPARATELY for each dataset.
    # K15 (task B): the bucket now holds ALL arrays stored in the npz cache
    # (key = metric name: 'stab', 'qual', and optionally
    # 'trust_k<K>'/'cont_k<K>'/'jacc_k<K>'), not just the fixed
    # 'stab'/'qual' as before K15 - the generalization below allows adding a
    # permutation test for new metrics without changing this loop on future
    # extensions. Old npz files (before task B) simply do not contain the
    # neighborhood-preservation metric keys -> those are simply not counted
    # for that run (backward compatibility, no crash).
    stats_rows_input: dict[tuple[str, float, float, str], dict[str, list[np.ndarray]]] = {}
    for task in tasks:
        key = RunKey(EXPERIMENT_NAME, task["dataset_name"], task["method_name"], task["seed"])
        if not is_done(key):
            continue
        cached = _load_pertrans_cache(key)
        if cached is None:
            continue  # the run exists in the CSV, but without an npz cache (e.g. it errored out) - skip
        bucket = stats_rows_input.setdefault(
            (task["dataset_name"], task["lam"], task["alpha"], task["solver"]), {}
        )
        for metric_name, arr in cached.items():
            bucket.setdefault(metric_name, []).append(arr)

    stats_rows: list[dict[str, Any]] = []
    for dataset_name in cfg["datasets"]:
        for alpha in cfg["alphas"]:
            for solver in cfg["solvers"]:
                baseline_key = (dataset_name, 0.0, alpha, solver)
                if baseline_key not in stats_rows_input:
                    logger.warning(
                        "The lambda=0 baseline for dataset=%s/alpha=%s/solver=%s is not available (not yet done) - "
                        "skipping the permutation test.", dataset_name, alpha, solver,
                    )
                    continue
                baseline_bucket = stats_rows_input[baseline_key]

                for (d, lam, a, s), bucket in sorted(stats_rows_input.items()):
                    if d != dataset_name or a != alpha or s != solver or lam == 0.0:
                        continue
                    common_metrics = sorted(set(bucket.keys()) & set(baseline_bucket.keys()))
                    for metric_name in common_metrics:
                        arr_lam = np.concatenate(bucket[metric_name]) if bucket[metric_name] else np.array([])
                        arr_base = np.concatenate(baseline_bucket[metric_name]) if baseline_bucket[metric_name] else np.array([])
                        n_pair = min(len(arr_lam), len(arr_base))
                        if n_pair == 0:
                            continue
                        diff_obs, p_val = _paired_permutation_test(
                            arr_lam[:n_pair] - arr_base[:n_pair], cfg["n_permutations"], cfg["permutation_seed"],
                        )
                        stats_rows.append({
                            "dataset": dataset_name, "lam": lam, "alpha": alpha, "solver": solver, "metric": metric_name,
                            "n_paired": n_pair, "mean_diff_vs_baseline": diff_obs, "p_value": p_val,
                        })

    if stats_rows:
        _finalize_stats_csv(EXPERIMENT_NAME, stats_rows, logger)
    else:
        logger.warning("No pairing for the permutation test (likely all runs are still missing) - exp4_stats.csv is not written.")

    write_done_file(EXPERIMENT_NAME, logger, extra_info={"mode": mode})


if __name__ == "__main__":
    main()
