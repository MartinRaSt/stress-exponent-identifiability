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
E13: "how many of the k nearest neighbors survive the projection to 2D"
(documentation/2026-09-17_zadani_exp13_preziti_sousedu.md) - a plain-language
companion metric to `auc_rnx`/`knn_jaccard_k7`/`q_global`, meant for readers
who are not specialists in dimensionality-reduction metrics.

TRAP THIS EXPERIMENT EXISTS TO AVOID: `knn_jaccard_k7` in
`exp6_alpha_curves_results.csv` is a MEAN Jaccard overlap over points. The
mean number of surviving neighbors can NOT be recovered from it via the
single-point identity m = 2*k*J/(1+J) - that identity holds pointwise, not
for an average of differing per-point J values (Jensen's inequality, since
m(J) is concave). The mean survivor count is therefore computed DIRECTLY
here via `src.sammon.metrics.knn_overlap_count`, never derived from the
already-averaged `knn_jaccard_k7` column (which is only carried along as
`knn_jaccard_reference`, a cross-check, never a source).

NO new DR run. Inputs, all reused (not redefined):
  - `results/data/[<mode>/]exp6_alpha_curves_results.csv`: cached embeddings
    (`src.common.checkpoint.load_embedding`/`RunKey`, same lookup convention
    as `src.experiments.exp10_identifiability_check`) and the metric columns
    `stress_scale_invariant`, `shepard_spearman_rho`, `q_global`,
    `knn_jaccard_k{k}` (already computed - never recomputed here) for the
    ONE canonical seed `exp6_alpha_curves.seeds[exp13_neighbor_survival.embedding_seed_index]`.
  - `results/data/[<mode>/]exp12_alpha_grid_extension_results.csv` (OPTIONAL):
    the same cached-embedding/metric-column source as exp6_alpha_curves
    above, but for alpha in (3, 6] on the 18 datasets it covers (see that
    module's docstring). Used ONLY to extend the oracle alpha_best search
    (below) beyond exp6's alpha=3 cap; if this file does not exist for the
    current mode, exp13 falls back to the exp6-only grid with a WARNING -
    never a fatal error (a curated extension experiment, not a prerequisite).
  - `results/data/[<mode>/]dataset_properties.csv` (kind=='vector',
    nn_ratio_k1) + the single production `results/data/alpha_pred_rule.json`
    (NOT mode-specific, same convention as `src.methods.sammon_alpha_pred`
    and `exp10_identifiability_check`): regime classification via
    `src.experiments.exp1_regime_stratified.classify_regime` and the rule's
    own predicted alpha via `src.sammon.alpha_predict.predict_alpha`.

`--datasets all|core|holdout` (default 'all', same flag/helper as
`exp1_dr_benchmark.py`/`exp6_alpha_curves.py` -
`exp_common.add_dataset_scope_arg`/`resolve_dataset_scope`) selects which
datasets this experiment covers: 'core' = `exp6_cfg["datasets"]` (the 32
original production datasets the alpha_pred rule was fitted on), 'holdout' =
`exp6_cfg.get("datasets_holdout", [])` (the 19 Q1 hold-out candidates),
'all' = the union (51 datasets, no duplicates, core first) - the default,
because this table is DESCRIPTIVE, not confirmatory: excluding the hold-out
set previously hid 6 of the 10 datasets whose oracle alpha_best exceeds
alpha=3 (anisotropic_ellipsoid, klein_bottle_4d, banknote, trefoil_knot,
twin_peaks, mfeat_morphological), understating the benefit of tuning alpha.
Each output row carries which set its dataset came from in the
`dataset_set` column ('core'/'holdout'), and the policy table's caption
records how many datasets and which `--datasets` scope it was built from.

Row = (dataset, alpha, seed=canonical, k). For EVERY dataset in the selected
`--datasets` scope (no cherry-picking within that scope, see the module
docstring's "Co NEDELAT"), `alpha` ranges over the UNION of:
  - the fixed policy grid `exp13_neighbor_survival.alpha_table` (e.g.
    alpha=0/1/2, shown as the table's fixed-alpha rows),
  - that dataset's own oracle alpha_best (argmax of the median auc_rnx over
    the UNION of the exp6_alpha_curves grid and, when available, the
    exp12_alpha_grid_extension grid - the same "oracle" concept as
    `sammon_alpha_auto`/`G_auc_oracle` elsewhere in the project). The
    experiment ('exp6' or 'exp12') that alpha_best/its median auc_rnx came
    from is recorded per dataset in the output column `alpha_best_source`,
    and its cached embedding/metric row is looked up in THAT SAME
    experiment's cache - never a silent fallback to the other one (see
    `_collect_dataset_alphas`),
  - that dataset's own rule-predicted alpha_pred (`alpha_pred_rule.json`,
    always in [0, 3] - `src.sammon.alpha_predict.ALPHA_MAX` - hence always
    sourced from exp6_alpha_curves).
A dataset for which a given alpha has no successful row in its source
experiment for the canonical seed (e.g. a --smoke run's narrower alpha grid
does not contain the production alpha_pred value) is SKIPPED for that one
alpha with a WARNING - never fabricated, never silently substituted, but
also not a fatal error (this experiment's own --quick/--smoke narrows the
alpha grid it draws from, same "code path exercised, not full coverage"
convention as `exp10_identifiability_check`).

order_orig (the ranking of all points by distance in the ORIGINAL space) is
computed ONCE per dataset via a `functools.lru_cache`-backed loader
(`_dataset_order_orig`), not once per alpha - see that function's docstring
for the caching rationale (a best-effort per-worker-process optimization,
not a correctness requirement: a cache miss in a different worker process
simply recomputes, it never returns a stale value).

Output: results/data/[<mode>/]exp13_neighbor_survival_results.csv
(checkpoint, resumable) + results/data/[<mode>/]exp13_neighbor_survival_DONE.txt,
plus the two-panel policy table
results/tables/[<mode>/]exp13_neighbor_survival.csv/.tex (booktabs, built at
the end of `main()` from the just-completed results CSV - see
`_write_survival_table`).

Run: venv\\python.exe -m src.experiments.exp13_neighbor_survival [--quick|--full|--smoke] [--datasets all|core|holdout]
or: src\\run_exp13_neighbor_survival.bat [quick|full|smoke] [--datasets all|core|holdout]
"""
from __future__ import annotations

import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.common.checkpoint import RunKey, load_embedding, results_csv_path
from src.common.config import get_mode_path, get_tables_dir
from src.common.logging_utils import get_logger
from src.experiments.config_experiments import resolve_experiment_config
from src.experiments.exp_common import (
    add_dataset_scope_arg,
    add_mode_args,
    filter_already_done,
    resolve_dataset_scope,
    resolve_experiment_name,
    resolve_mode,
    run_experiment_grid,
    write_done_file,
)
from src.experiments.report_tables import write_booktabs_tex

BASE_EXPERIMENT_NAME = "exp13_neighbor_survival"
EXP6_BASE_NAME = "exp6_alpha_curves"
# Optional extension of the exp6_alpha_curves alpha grid (alpha in (3, 6] on
# 18 datasets, see exp12_alpha_grid_extension.py's module docstring). Named
# via a module constant, same convention as EXP6_BASE_NAME above, so the
# source experiment name is never hardcoded at more than this one place.
EXP12_BASE_NAME = "exp12_alpha_grid_extension"

# 'exp6' or 'exp12': which grid a dataset's oracle alpha_best (and its median
# auc_rnx) was found in - see `_merge_alpha_median_auc`.
ALPHA_BEST_SOURCE_EXP6 = "exp6"
ALPHA_BEST_SOURCE_EXP12 = "exp12"

# Canonical CSV schema (documentation/2026-09-17_zadani_exp13_preziti_sousedu.md,
# section "Vystup"). 'dataset'/'method'/'seed'/'experiment' are added
# automatically by src.common.checkpoint.append_result (from the RunKey);
# 'status'/'error'/'wall_time_sec' are added automatically by
# src.experiments.exp_common.run_experiment_grid.
COLUMN_KEYS = [
    "alpha", "k", "n_samples", "regime", "nn_ratio_k1", "alpha_best_source", "dataset_set",
    "neighbors_kept", "neighbors_kept_frac", "knn_jaccard_reference",
    "stress_scale_invariant", "shepard_spearman_rho", "q_global",
]

DATASET_SET_CORE = "core"
DATASET_SET_HOLDOUT = "holdout"


def dataset_set_label(dataset_name: str, holdout_datasets: set[str] | list[str]) -> str:
    """'core' or 'holdout', per `--datasets` (see `add_dataset_scope_arg`):
    core and holdout are disjoint BY DESIGN (config_experiments.yaml never
    lists the same dataset in both `exp1_dr_benchmark.datasets` and
    `datasets_holdout` - see exp_common.add_dataset_scope_arg's docstring),
    so membership in `holdout_datasets` alone decides the label."""
    return DATASET_SET_HOLDOUT if dataset_name in holdout_datasets else DATASET_SET_CORE

# Pretty row labels for the fixed-alpha policies shown in the table (a pure
# LABELING convenience, not a numeric magic constant - any alpha_table value
# not listed here just falls back to a generic "alpha=<value>" label).
_ALPHA_POLICY_LABELS = {0.0: "alpha=0 (MDS)", 1.0: "alpha=1 (Sammon)", 2.0: "alpha=2 (Kamada-Kawai)"}
POLICY_LABEL_PRED = "alpha_pred (rule)"
POLICY_LABEL_BEST = "alpha_best (oracle)"

PANEL_LOW_RATIO = "low_ratio (tuning helps)"
PANEL_REST = "mid_or_high_ratio (tuning does not help)"

_SURVIVAL_TABLE_COLUMNS = [
    "panel", "policy", "n_datasets", "k",
    "neighbors_kept", "stress_scale_invariant", "shepard_spearman_rho", "q_global",
]


def policy_label(alpha: float) -> str:
    """Pretty row label for a FIXED alpha value shown in the table (see
    `_ALPHA_POLICY_LABELS`); falls back to a generic label for any value
    not in the lookup."""
    return _ALPHA_POLICY_LABELS.get(float(alpha), f"alpha={float(alpha):g}")


def _method_name_for_alpha(alpha: float) -> str:
    """Same 'method' naming convention as exp6_alpha_curves.py/exp10_identifiability_check.py
    (the RunKey used to load a cached embedding MUST match exactly)."""
    return f"alpha{alpha}"


def _method_name_for_task(alpha: float, k: int) -> str:
    """The 'method' name for THIS experiment's own RunKey/checkpoint -
    encodes both alpha and k, since a row here is (dataset, alpha, k) and a
    plain RunKey only uniquely identifies (experiment, dataset, method, seed)."""
    return f"{_method_name_for_alpha(alpha)}_k{k}"


@lru_cache(maxsize=8)
def _dataset_order_orig(dataset_name: str, n_max: int, subsample_seed: int) -> tuple[np.ndarray, int]:
    """Loads+subsamples a dataset (EXACTLY the exp6_alpha_curves pipeline)
    and returns (order_orig, n_samples), computed ONCE per (dataset, n_max,
    subsample_seed) via `functools.lru_cache`.

    Rationale (documentation/2026-09-17_zadani_exp13_preziti_sousedu.md item
    1): `main()` builds the task list dataset-by-dataset, and
    `src.common.parallel.run_parallel_map` (ProcessPoolExecutor.map) hands
    out CONSECUTIVE tasks to the same worker process by default, so
    consecutive alpha/k tasks of the SAME dataset usually reuse this cache
    within one worker instead of recomputing the O(n^2 log n)
    `_neighbor_ranks` sort for every alpha. This is a best-effort
    optimization, NOT a correctness requirement: a cache miss (a different
    worker process, which starts with an empty cache) simply recomputes the
    same deterministic value - it can never return a stale or wrong result
    for a different dataset."""
    from src.datasets.registry import load_dataset
    from src.datasets.subsample import subsample_dataset
    from src.methods.common import to_distance_matrix
    from src.sammon.metrics import _neighbor_ranks

    ds = load_dataset(dataset_name)
    ds = subsample_dataset(ds, n_max=n_max, random_state=subsample_seed)
    D = ds.D if ds.D is not None else to_distance_matrix(ds.X, ds.kind)
    D = np.asarray(D, dtype=np.float64)
    _, order_orig = _neighbor_ranks(D)
    return order_orig, D.shape[0]


def _load_dataset_policy_info(
    mode: str, merged_median_auc: dict[str, dict[float, tuple[float, str]]],
) -> dict[str, dict[str, Any]]:
    """{dataset: {'nn_ratio_k1':.., 'regime':.., 'alpha_pred':.., 'alpha_best':..,
    'alpha_best_source':..}} - regime/alpha_pred via the SAME rule and
    nn_ratio_k1 source as `exp1_regime_stratified.classify_regime`/
    `exp10_identifiability_check` (`results/data/[<mode>/]dataset_properties.csv`,
    kind=='vector', and the single production `results/data/alpha_pred_rule.json`);
    alpha_best is the oracle argmax of `merged_median_auc[dataset]` (median
    auc_rnx per alpha, each alpha tagged with its source experiment - see
    `_merge_alpha_median_auc`) over the UNION of the exp6 alpha grid and, when
    available, the exp12 extension grid actually available for this mode (the
    same "oracle" concept as `sammon_alpha_auto`/`G_auc_oracle` elsewhere in
    the project). `alpha_best_source` records which experiment ('exp6' or
    'exp12') alpha_best came from, so callers know which cache to load its
    embedding/metric row from - never a silent fallback to the other one."""
    from src.experiments.exp1_regime_stratified import classify_regime
    from src.sammon.alpha_predict import load_alpha_pred_rule, predict_alpha

    props_path = get_mode_path("results_data_dir", mode) / "dataset_properties.csv"
    if not props_path.exists():
        raise FileNotFoundError(
            f"Missing dataset_properties.csv: {props_path}\n"
            f"Run first: venv\\python.exe -m src.experiments.dataset_properties --{mode} (K1)."
        )
    props = pd.read_csv(props_path, usecols=["dataset", "kind", "nn_ratio_k1"])
    props = props[(props["kind"] == "vector") & props["nn_ratio_k1"].notna()].reset_index(drop=True)
    rule = load_alpha_pred_rule()
    regimes = classify_regime(props["nn_ratio_k1"].to_numpy(dtype=np.float64), rule)

    out: dict[str, dict[str, Any]] = {}
    for row, regime in zip(props.itertuples(index=False), regimes):
        dataset_name = str(row.dataset)
        nn = float(row.nn_ratio_k1)
        auc_by_alpha = merged_median_auc.get(dataset_name)
        if not auc_by_alpha:
            continue  # no successful exp6 row for this dataset in this mode - excluded below with a clear error, not here
        alpha_best, _auc_best, source_best = _pick_alpha_best(auc_by_alpha)
        out[dataset_name] = {
            "nn_ratio_k1": nn,
            "regime": str(regime),
            "alpha_pred": predict_alpha(nn, rule),
            "alpha_best": alpha_best,
            "alpha_best_source": source_best,
        }
    return out


def _pick_alpha_best(auc_by_alpha: dict[float, tuple[float, str]]) -> tuple[float, float, str]:
    """argmax over {alpha: (median_auc_rnx, source)} -> (alpha_best, auc_best,
    source_best). Extracted as a small pure function so the oracle selection
    itself is unit-testable without any of `_load_dataset_policy_info`'s file
    I/O (dataset_properties.csv, alpha_pred_rule.json)."""
    alpha_best, (auc_best, source_best) = max(auc_by_alpha.items(), key=lambda item: item[1][0])
    return float(alpha_best), float(auc_best), source_best


def _load_median_auc(experiment_name: str, run_hint: str) -> dict[str, dict[float, float]]:
    """{dataset: {alpha: median_auc_rnx over seeds}} - same aggregation as
    exp6_alpha_curves.py::_derive_alpha_optimum / exp10_identifiability_check.py.
    Works for any exp6-schema-compatible results CSV (exp6_alpha_curves or
    exp12_alpha_grid_extension share the same (dataset, alpha, seed, status,
    auc_rnx) columns by construction, see exp12_alpha_grid_extension.py's
    module docstring)."""
    path = results_csv_path(experiment_name)
    if not path.exists():
        raise FileNotFoundError(f"Missing results: {path}\n{run_hint}")
    df = pd.read_csv(path, usecols=["dataset", "alpha", "auc_rnx", "status"])
    ok = df[(df["status"] == "ok") & df["auc_rnx"].notna()]
    if ok.empty:
        raise ValueError(f"{path} contains no successful rows with auc_rnx.")
    med = ok.groupby(["dataset", "alpha"])["auc_rnx"].median()
    out: dict[str, dict[float, float]] = {}
    for (dataset_name, alpha), value in med.items():
        out.setdefault(str(dataset_name), {})[float(alpha)] = float(value)
    return out


def _merge_alpha_median_auc(
    e6_median_auc: dict[str, dict[float, float]],
    e12_median_auc: dict[str, dict[float, float]] | None,
) -> dict[str, dict[float, tuple[float, str]]]:
    """Unions the exp6 and (optional) exp12 alpha grids per dataset into
    {dataset: {alpha: (median_auc_rnx, source)}}, `source` in
    {ALPHA_BEST_SOURCE_EXP6, ALPHA_BEST_SOURCE_EXP12} - the input to the
    oracle argmax in `_load_dataset_policy_info`. exp12 only extends the grid
    strictly ABOVE exp6's alpha=3 cap for a curated subset of datasets
    (exp12_alpha_grid_extension.py's module docstring), so the two grids are
    disjoint BY CONSTRUCTION; an alpha present in both for the same dataset
    is therefore a configuration error, not something to silently resolve."""
    merged: dict[str, dict[float, tuple[float, str]]] = {
        dataset: {alpha: (auc, ALPHA_BEST_SOURCE_EXP6) for alpha, auc in by_alpha.items()}
        for dataset, by_alpha in e6_median_auc.items()
    }
    if e12_median_auc:
        for dataset_name, by_alpha in e12_median_auc.items():
            bucket = merged.setdefault(dataset_name, {})
            for alpha, auc in by_alpha.items():
                if alpha in bucket:
                    raise ValueError(
                        f"dataset={dataset_name}: alpha={alpha} present in BOTH exp6_alpha_curves and "
                        "exp12_alpha_grid_extension - the two alpha grids must be disjoint (exp12 only "
                        "extends alpha beyond exp6's cap); fix the grids in config_experiments.yaml."
                    )
                bucket[alpha] = (auc, ALPHA_BEST_SOURCE_EXP12)
    return merged


def _try_load_e12_source(
    exp12_experiment_name: str, canonical_seed: int, k_values: list[int], logger,
) -> tuple[dict[str, dict[float, float]] | None, dict[tuple[str, float], dict[str, Any]]]:
    """Best-effort load of the OPTIONAL exp12_alpha_grid_extension source:
    (median auc_rnx per (dataset, alpha) for the oracle, plus the raw rows
    for the canonical seed). Returns (None, {}) with a WARNING - never an
    exception - if exp12 has not been run for this mode: unlike
    exp6_alpha_curves, exp12 is not a prerequisite for exp13 (module
    docstring, item 1: 'chovej se jako dosud a nepadej')."""
    path = results_csv_path(exp12_experiment_name)
    if not path.exists():
        logger.warning(
            "%s: exp12_alpha_grid_extension results not found at %s - the alpha_best oracle will use the "
            "exp6_alpha_curves grid only (run exp12_alpha_grid_extension.py first to extend it beyond alpha=3).",
            BASE_EXPERIMENT_NAME, path,
        )
        return None, {}
    hint = "(unreachable - file existence already checked above)"
    median_auc = _load_median_auc(exp12_experiment_name, run_hint=hint)
    rows = _load_alpha_rows(exp12_experiment_name, canonical_seed, k_values, run_hint=hint)
    return median_auc, rows


def _load_alpha_rows(
    experiment_name: str, seed: int, k_values: list[int], run_hint: str,
) -> dict[tuple[str, float], dict[str, Any]]:
    """{(dataset, alpha): row_dict} from an exp6-schema-compatible results CSV
    (exp6_alpha_curves or exp12_alpha_grid_extension - see
    `_merge_alpha_median_auc`), filtered to the canonical seed and
    status=='ok' - the source of the ALREADY-COMPUTED metric columns
    (`stress_scale_invariant`, `shepard_spearman_rho`, `q_global`,
    `knn_jaccard_k{k}`), never recomputed here."""
    path = results_csv_path(experiment_name)
    if not path.exists():
        raise FileNotFoundError(f"Missing results: {path}\n{run_hint}")
    header = list(pd.read_csv(path, nrows=0).columns)
    jaccard_cols = [f"knn_jaccard_k{k}" for k in k_values if f"knn_jaccard_k{k}" in header]
    usecols = ["dataset", "alpha", "seed", "status", "stress_scale_invariant", "shepard_spearman_rho", "q_global", *jaccard_cols]
    df = pd.read_csv(path, usecols=usecols)
    df = df[(df["seed"] == seed) & (df["status"] == "ok")]
    out: dict[tuple[str, float], dict[str, Any]] = {}
    for row in df.itertuples(index=False):
        out[(str(row.dataset), float(row.alpha))] = row._asdict()
    return out


def _collect_dataset_alphas(alpha_table: list[float], info: dict[str, Any]) -> list[tuple[float, str]]:
    """Ordered, de-duplicated (alpha, source) pairs for one dataset: the
    fixed alpha_table policies and alpha_pred always come from exp6
    (alpha_pred is clipped to [0, 3], see `src.sammon.alpha_predict.ALPHA_MAX`
    - it can never fall in the exp12 extension range); alpha_best's source is
    whichever grid its oracle optimum was found in (`info['alpha_best_source']`,
    set by `_load_dataset_policy_info`/`_merge_alpha_median_auc`). Returned in
    a fixed source per alpha - never an opportunistic 'try exp6 then exp12'
    search - so a caller always knows exactly which cache to read."""
    candidates: list[tuple[float, str]] = [(float(a), ALPHA_BEST_SOURCE_EXP6) for a in alpha_table]
    candidates.append((float(info["alpha_pred"]), ALPHA_BEST_SOURCE_EXP6))
    candidates.append((float(info["alpha_best"]), info["alpha_best_source"]))

    out: list[tuple[float, str]] = []
    for alpha, source in candidates:
        if any(abs(alpha - seen_alpha) <= 1e-9 for seen_alpha, _ in out):
            continue
        out.append((alpha, source))
    return out


def _run_single(task: dict[str, Any]) -> dict[str, Any]:
    """One row: (dataset, alpha, k) -> knn_overlap_count (+ the metric
    columns already computed by exp6_alpha_curves, passed through unchanged)."""
    from src.methods.common import to_distance_matrix
    from src.sammon.metrics import _neighbor_ranks, knn_overlap_count

    dataset_name = task["dataset_name"]
    alpha = task["alpha"]
    k = task["k"]
    seed = task["seed"]
    method_name = task["method_name"]

    result: dict[str, Any] = {"dataset_name": dataset_name, "method_name": method_name, "seed": seed}
    t0 = time.perf_counter()
    try:
        order_orig, n_samples = _dataset_order_orig(dataset_name, task["n_max"], task["subsample_seed"])

        Y = load_embedding(RunKey(task["embedding_experiment_name"], dataset_name, _method_name_for_alpha(alpha), seed))
        if Y.shape[0] != n_samples:
            raise ValueError(
                f"Cached embedding for dataset={dataset_name} alpha={alpha} seed={seed} has {Y.shape[0]} rows, "
                f"expected {n_samples} (n_max/subsample_seed mismatch with exp6_alpha_curves)."
            )
        d_emb = to_distance_matrix(np.asarray(Y, dtype=np.float64), "vector")
        _, order_emb = _neighbor_ranks(d_emb)

        neighbors_kept = knn_overlap_count(order_orig, order_emb, k)

        metrics: dict[str, Any] = {
            "n_samples": n_samples, "regime": task["regime"], "nn_ratio_k1": task["nn_ratio_k1"],
            "alpha_best_source": task["alpha_best_source"], "dataset_set": task["dataset_set"],
            "neighbors_kept": neighbors_kept, "neighbors_kept_frac": neighbors_kept / k,
            "knn_jaccard_reference": task["knn_jaccard_reference"],
            "stress_scale_invariant": task["stress_scale_invariant"],
            "shepard_spearman_rho": task["shepard_spearman_rho"],
            "q_global": task["q_global"],
        }
        result["status"] = "ok"
        result["error"] = ""
        result["metrics"] = metrics
    except Exception as exc:  # one incompatible combination must not stop the whole run
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["metrics"] = {}
    result["extra"] = {"alpha": alpha, "k": k}
    result["wall_time_sec"] = time.perf_counter() - t0
    result["embedding"] = None  # exp13 does not generate any new embedding, only analyzes existing ones
    return result


def _select_per_dataset_alpha(df_ok: pd.DataFrame, alpha_by_dataset: dict[str, float], atol: float = 1e-9) -> pd.DataFrame:
    """Selects, for each dataset in `alpha_by_dataset`, the single row of
    `df_ok` whose alpha matches (used for the alpha_pred/alpha_best policy
    rows, where alpha varies per dataset). A dataset without a matching row
    (e.g. skipped in `main()` as a smoke-grid gap) is simply absent from the
    result - never fabricated."""
    rows = []
    for dataset_name, alpha in alpha_by_dataset.items():
        sub = df_ok[(df_ok["dataset"] == dataset_name) & (np.abs(df_ok["alpha"] - alpha) <= atol)]
        if not sub.empty:
            rows.append(sub.iloc[0])
    return pd.DataFrame(rows) if rows else df_ok.iloc[0:0]


def _panel_medians(sub: pd.DataFrame) -> dict[str, float]:
    return {
        "n_datasets": int(sub["dataset"].nunique()),
        "neighbors_kept": float(sub["neighbors_kept"].median()) if not sub.empty else float("nan"),
        "stress_scale_invariant": float(sub["stress_scale_invariant"].median()) if not sub.empty else float("nan"),
        "shepard_spearman_rho": float(sub["shepard_spearman_rho"].median()) if not sub.empty else float("nan"),
        "q_global": float(sub["q_global"].median()) if not sub.empty else float("nan"),
    }


def build_survival_table(
    df: pd.DataFrame, policy_info: dict[str, dict[str, Any]], alpha_table: list[float], k: int,
    scope_datasets: list[str] | set[str] | None = None,
) -> pd.DataFrame:
    """Builds the two-panel policy table (documentation/
    2026-09-17_zadani_exp13_preziti_sousedu.md, section 3): one row per
    (panel, policy) with the MEDIAN over datasets in that panel/regime-group
    of neighbors_kept/stress_scale_invariant/shepard_spearman_rho/q_global.
    Panel A = 'low_ratio' (tuning is predicted to help); panel B = every
    other regime pooled together (tuning is predicted not to help) - exactly
    the split requested in the task, not a 3-way regime breakdown.

    `scope_datasets` (the PRESENTLY selected `--datasets all|core|holdout`
    scope, see `add_dataset_scope_arg`) restricts the aggregation to exactly
    those datasets - both the rows of `df` (which may, on disk, contain
    leftover rows from a PREVIOUS run in a different scope, e.g. 120 rows
    from an earlier core-only run) and the panel membership derived from
    `policy_info` (which is loaded from dataset_properties.csv/exp6 for ALL
    known datasets, not just the current scope). `None` (the default, used
    by callers/tests that already pre-filtered both inputs) disables this
    extra filter."""
    ok = df[(df["status"] == "ok") & (df["k"] == k)]
    if scope_datasets is not None:
        scope_set = set(scope_datasets)
        ok = ok[ok["dataset"].isin(scope_set)]
        policy_info = {d: info for d, info in policy_info.items() if d in scope_set}

    panels = {
        PANEL_LOW_RATIO: [d for d, info in policy_info.items() if info["regime"] == "low_ratio"],
        PANEL_REST: [d for d, info in policy_info.items() if info["regime"] != "low_ratio"],
    }

    rows: list[dict[str, Any]] = []
    for panel_name, datasets_in_panel in panels.items():
        panel_ok = ok[ok["dataset"].isin(datasets_in_panel)]

        for alpha in alpha_table:
            sub = panel_ok[np.abs(panel_ok["alpha"] - alpha) <= 1e-9]
            rows.append({"panel": panel_name, "policy": policy_label(alpha), "k": k, **_panel_medians(sub)})

        alpha_pred_by_dataset = {d: policy_info[d]["alpha_pred"] for d in datasets_in_panel}
        sub_pred = _select_per_dataset_alpha(panel_ok, alpha_pred_by_dataset)
        rows.append({"panel": panel_name, "policy": POLICY_LABEL_PRED, "k": k, **_panel_medians(sub_pred)})

        alpha_best_by_dataset = {d: policy_info[d]["alpha_best"] for d in datasets_in_panel}
        sub_best = _select_per_dataset_alpha(panel_ok, alpha_best_by_dataset)
        rows.append({"panel": panel_name, "policy": POLICY_LABEL_BEST, "k": k, **_panel_medians(sub_best)})

    out = pd.DataFrame(rows)[_SURVIVAL_TABLE_COLUMNS]
    return out


def _write_survival_table(
    experiment_name: str, mode: str, policy_info: dict[str, dict[str, Any]], alpha_table: list[float], k: int,
    scope_datasets: list[str], dataset_scope: str, logger,
) -> Path | None:
    csv_path = results_csv_path(experiment_name)
    if not csv_path.exists():
        logger.warning("exp13_neighbor_survival table: %s does not exist, skipping.", csv_path)
        return None
    df = pd.read_csv(csv_path)
    if df[(df["status"] == "ok") & (df["k"] == k) & df["dataset"].isin(scope_datasets)].empty:
        logger.warning(
            "exp13_neighbor_survival table: no 'ok' rows for k=%d within --datasets=%s scope in %s, skipping.",
            k, dataset_scope, csv_path,
        )
        return None

    table = build_survival_table(df, policy_info, alpha_table, k, scope_datasets=scope_datasets)

    tables_dir = get_tables_dir(mode)
    out_csv = tables_dir / "exp13_neighbor_survival.csv"
    table.to_csv(out_csv, index=False)

    out_tex = tables_dir / "exp13_neighbor_survival.tex"
    write_booktabs_tex(
        table, out_tex,
        caption=(
            "Neighbor survival (mean count of the k nearest input-space neighbors "
            "still present among the k nearest neighbors in the 2D embedding, out of k) "
            "versus metric cost, by alpha-selection policy, split by distance-concentration regime. "
            "Values are medians over datasets within each panel. "
            f"Covers {len(scope_datasets)} datasets (--datasets={dataset_scope})."
        ),
        label="tab:exp13_neighbor_survival",
        comment_lines=[
            f"source: {csv_path.name} (src/experiments/exp13_neighbor_survival.py), k={k}",
            "panel A = low_ratio regime (tuning alpha is predicted to help); "
            "panel B = mid_ratio+high_ratio pooled (tuning is predicted not to help)",
            f"dataset scope: --datasets={dataset_scope} ({len(scope_datasets)} datasets, see the 'dataset_set' results column)",
        ],
        # 'panel'/'policy' values are the PANEL_LOW_RATIO/PANEL_REST and
        # policy_label(alpha)/POLICY_LABEL_PRED/POLICY_LABEL_BEST constants
        # above, which are exactly the config_experiments.yaml
        # display_labels.regime/.policy keys (see fig_neighbor_survival.py,
        # which renders the same values the same way).
        value_labels={"panel": "regime", "policy": "policy"},
    )
    logger.info("Survival table written: %s / %s (--datasets=%s, %d datasets)", out_csv, out_tex, dataset_scope, len(scope_datasets))
    return out_tex


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="E13: how many of the k nearest neighbors survive the 2D projection, by alpha-selection "
        "policy (documentation/2026-09-17_zadani_exp13_preziti_sousedu.md)."
    )
    add_mode_args(parser)
    add_dataset_scope_arg(parser)
    args = parser.parse_args()
    mode = resolve_mode(args)
    logger = get_logger(BASE_EXPERIMENT_NAME, mode=mode)

    exp6_cfg = resolve_experiment_config(EXP6_BASE_NAME, mode)
    exp13_cfg = resolve_experiment_config(BASE_EXPERIMENT_NAME, mode)

    from src.experiments.exp1_dr_benchmark import SUBSAMPLE_SEED

    seeds_list = list(exp6_cfg["seeds"])
    seed_index = int(exp13_cfg["embedding_seed_index"])
    if not (0 <= seed_index < len(seeds_list)):
        raise ValueError(f"exp13_neighbor_survival.embedding_seed_index={seed_index} out of range for exp6_alpha_curves.seeds={seeds_list} (mode={mode}).")
    canonical_seed = int(seeds_list[seed_index])

    exp6_experiment_name = resolve_experiment_name(EXP6_BASE_NAME, mode)
    exp12_experiment_name = resolve_experiment_name(EXP12_BASE_NAME, mode)

    # --datasets all|core|holdout (default 'all'), same flag/helper as
    # exp1_dr_benchmark.py/exp6_alpha_curves.py (see the module docstring for
    # why 'all' - 51 datasets - is the default here, unlike exp6/exp1's 'all').
    core_datasets: list[str] = list(exp6_cfg["datasets"])
    holdout_datasets: list[str] = list(exp6_cfg.get("datasets_holdout", []))
    holdout_set = set(holdout_datasets)
    all_datasets = resolve_dataset_scope(args, core_datasets, holdout_datasets)
    logger.info(
        "--datasets=%s -> %d datasets (core=%d, holdout=%d).",
        args.datasets, len(all_datasets), len(core_datasets), len(holdout_datasets),
    )
    k_values = [int(k) for k in exp13_cfg["k_values"]]
    alpha_table = sorted(float(a) for a in exp13_cfg["alpha_table"])

    e6_run_hint = "Run first: venv\\python.exe -m src.experiments.exp6_alpha_curves --<mode> (K6)."
    e6_median_auc = _load_median_auc(exp6_experiment_name, run_hint=e6_run_hint)
    e6_rows = _load_alpha_rows(exp6_experiment_name, canonical_seed, k_values, run_hint=e6_run_hint)
    e12_median_auc, e12_rows = _try_load_e12_source(exp12_experiment_name, canonical_seed, k_values, logger)

    merged_median_auc = _merge_alpha_median_auc(e6_median_auc, e12_median_auc)
    policy_info = _load_dataset_policy_info(mode, merged_median_auc)

    n_from_e12 = sum(1 for info in policy_info.values() if info["alpha_best_source"] == ALPHA_BEST_SOURCE_EXP12)
    logger.info(
        "%s (mode=%s): %d datasets, alpha_table=%s, k_values=%s, canonical seed=%d "
        "(embedding sources: exp6=%s, exp12=%s [%s]). alpha_best oracle sourced from exp12 for %d/%d datasets.",
        BASE_EXPERIMENT_NAME, mode, len(all_datasets), alpha_table, k_values, canonical_seed,
        exp6_experiment_name, exp12_experiment_name, "available" if e12_median_auc else "unavailable",
        n_from_e12, len(policy_info),
    )

    # exp12's own fit hyperparameters (incl. n_max) always come from the FULL
    # production exp6_alpha_curves config, regardless of exp12's own mode
    # (exp12_alpha_grid_extension.py's module docstring) - so a task whose
    # embedding is sourced from exp12 must subsample with the FULL exp6 n_max,
    # even when exp13 itself runs in --quick/--smoke mode (whose exp6_cfg
    # n_max is smaller and would not match the cached exp12 embedding's row count).
    exp6_full_n_max = int(resolve_experiment_config(EXP6_BASE_NAME, "full")["n_max"])

    EXPERIMENT_NAME = resolve_experiment_name(BASE_EXPERIMENT_NAME, mode)

    tasks: list[dict[str, Any]] = []
    for dataset_name in all_datasets:
        info = policy_info.get(dataset_name)
        if info is None:
            raise ValueError(
                f"No usable dataset_properties.csv/exp6_alpha_curves data for dataset='{dataset_name}' (mode={mode}) - "
                "run dataset_properties.py and exp6_alpha_curves.py first (K1/K6)."
            )

        for alpha, source in _collect_dataset_alphas(alpha_table, info):
            if source == ALPHA_BEST_SOURCE_EXP6:
                row = e6_rows.get((dataset_name, alpha))
                embedding_experiment_name = exp6_experiment_name
                row_n_max = int(exp6_cfg["n_max"])
            elif source == ALPHA_BEST_SOURCE_EXP12:
                row = e12_rows.get((dataset_name, alpha))
                embedding_experiment_name = exp12_experiment_name
                row_n_max = exp6_full_n_max
            else:
                raise ValueError(f"dataset={dataset_name}: unknown alpha_best_source={source!r} (expected 'exp6' or 'exp12').")

            if row is None:
                logger.warning(
                    "dataset=%s: alpha=%.4g (source=%s) has no %s row for the canonical seed=%d (mode=%s) "
                    "- skipping this alpha for this dataset (a narrower --%s alpha grid, or a genuinely failed run).",
                    dataset_name, alpha, source, source, canonical_seed, mode, mode,
                )
                continue

            for k in k_values:
                jac_col = f"knn_jaccard_k{k}"
                jac_val = row.get(jac_col)
                knn_jaccard_reference = float(jac_val) if jac_val is not None and pd.notna(jac_val) else float("nan")
                tasks.append({
                    "dataset_name": dataset_name, "method_name": _method_name_for_task(alpha, k),
                    "seed": canonical_seed, "alpha": alpha, "k": k,
                    "n_max": row_n_max, "subsample_seed": SUBSAMPLE_SEED,
                    "embedding_experiment_name": embedding_experiment_name,
                    "regime": info["regime"], "nn_ratio_k1": info["nn_ratio_k1"],
                    "alpha_best_source": info["alpha_best_source"],
                    "dataset_set": dataset_set_label(dataset_name, holdout_set),
                    "knn_jaccard_reference": knn_jaccard_reference,
                    "stress_scale_invariant": float(row["stress_scale_invariant"]),
                    "shepard_spearman_rho": float(row["shepard_spearman_rho"]),
                    "q_global": float(row["q_global"]) if pd.notna(row["q_global"]) else float("nan"),
                })

    todo, n_done = filter_already_done(EXPERIMENT_NAME, tasks)
    logger.info(
        "%s (mode=%s): %d combinations already done (skipped), %d new to compute.",
        EXPERIMENT_NAME, mode, n_done, len(todo),
    )

    n_ok, n_err = run_experiment_grid(EXPERIMENT_NAME, todo, _run_single, COLUMN_KEYS, logger)
    logger.info("Done. Newly successful: %d, newly failed: %d, skipped: %d.", n_ok, n_err, n_done)
    write_done_file(
        EXPERIMENT_NAME, logger,
        extra_info={
            "mode": mode, "datasets_scope": args.datasets, "exp6_source": exp6_experiment_name,
            "exp12_source": exp12_experiment_name if e12_median_auc else "unavailable",
            "canonical_seed": canonical_seed, "n_datasets_alpha_best_from_exp12": n_from_e12,
        },
    )

    _write_survival_table(EXPERIMENT_NAME, mode, policy_info, alpha_table, k_values[0], all_datasets, args.datasets, logger)


if __name__ == "__main__":
    main()
