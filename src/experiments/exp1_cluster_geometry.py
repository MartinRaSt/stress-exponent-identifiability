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
K2 (documentation/2026-09-12_plan_smeru_clanku.md) - inter-cluster geometry
(src/sammon/cluster_geometry.py) computed over the ALREADY SAVED E1
embeddings
(results/data/embeddings/exp1_dr_benchmark/<dataset>__<method>__<seed>.npy) -
NO re-run of the DR methods, just computing 4 new metrics from existing results.

For each (dataset, method, seed) successful row in exp1_dr_benchmark_results.csv:
  - datasets with < 3 classes (or no labels) -> a row with NaN metrics and
    note='insufficient_classes_or_no_labels' (status='ok' - this is NOT a
    run failure, the metric is simply undefined for that dataset, see the
    K2 spec)
  - others -> D_in recomputed from the SAME subsample as in the E1 run (see
    the `n_samples_used` column + `src.experiments.exp1_dr_benchmark.SUBSAMPLE_SEED`),
    D_out from the saved Y, 4 metrics (src/sammon/cluster_geometry.py)

D_in is shared by all methods/seeds for a given dataset - it is computed
EXACTLY ONCE per dataset (not 180x, as a generic per-task worker would do),
see `_load_dataset_cache`.

Output: results/data/[<mode>/]exp1_cluster_geometry_results.csv
(experiment, dataset, method, seed, 4 metrics, n_classes, note, status,
error, wall_time_sec), resumable (checkpoint), DONE file.

Run: venv\\python.exe -m src.experiments.exp1_cluster_geometry [--quick|--full|--smoke]
or: src\\run_exp1_cluster_geometry.bat [quick|full|smoke]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from src.common.checkpoint import RunKey, append_result, is_done, load_embedding, results_csv_path
from src.common.config import load_config
from src.common.logging_utils import get_logger, wall_clock
from src.common.progress import progress_iter
from src.experiments.config_experiments import resolve_experiment_config
from src.experiments.exp_common import (
    filter_already_done,
    keep_system_awake,
    parse_mode_args,
    resolve_experiment_name,
    write_done_file,
)
from src.sammon.cluster_geometry import (
    CLUSTER_GEOMETRY_KEYS,
    MAX_CLASSES_FOR_GEOMETRY,
    MIN_CLASSES_FOR_GEOMETRY,
    cluster_geometry_metrics,
)

BASE_EXPERIMENT_NAME = "exp1_cluster_geometry"
SOURCE_BASE_EXPERIMENT_NAME = "exp1_dr_benchmark"

COLUMN_KEYS = CLUSTER_GEOMETRY_KEYS + ["n_classes", "note"]

NOTE_INSUFFICIENT_CLASSES = "insufficient_classes_or_no_labels"
NOTE_EXCESSIVE_CLASSES = "excessive_classes_likely_continuous_label"


def _load_dataset_cache(dataset_name: str, n_max: int) -> tuple[np.ndarray | None, np.ndarray | None, int, str | None]:
    """Loads and subsamples the dataset EXACTLY the same way as in the
    production E1 run (see `src.experiments.exp1_dr_benchmark.SUBSAMPLE_SEED`,
    `n_max` = the recorded `n_samples_used` from exp1_dr_benchmark_results.csv -
    deterministically reproduces the same subsample independent of the
    current config.yaml). Returns (D_in, y, n_classes, load_error) -
    `load_error` is None on success."""
    from src.datasets.registry import load_dataset
    from src.datasets.subsample import subsample_dataset
    from src.experiments.exp1_dr_benchmark import SUBSAMPLE_SEED

    try:
        ds = load_dataset(dataset_name)
        ds = subsample_dataset(ds, n_max=n_max, random_state=SUBSAMPLE_SEED)
        X = np.asarray(ds.X, dtype=np.float64)
        D_in = squareform(pdist(X, metric="euclidean"))
        y = ds.y
        n_classes = int(len(np.unique(y))) if y is not None else 0
        return D_in, y, n_classes, None
    except Exception as exc:
        return None, None, 0, f"{type(exc).__name__}: {exc}"


def main() -> None:
    mode = parse_mode_args("K2: inter-cluster geometry over saved exp1_dr_benchmark embeddings.")
    logger = get_logger(BASE_EXPERIMENT_NAME, mode=mode)
    cfg = resolve_experiment_config(BASE_EXPERIMENT_NAME, mode)
    EXPERIMENT_NAME = resolve_experiment_name(BASE_EXPERIMENT_NAME, mode)
    SOURCE_EXPERIMENT = resolve_experiment_name(SOURCE_BASE_EXPERIMENT_NAME, mode)
    cg_k = int(load_config()["sammon"]["metrics_extended"]["cluster_geometry_k"])

    source_csv = results_csv_path(SOURCE_EXPERIMENT)
    if not source_csv.exists():
        raise FileNotFoundError(
            f"Missing input for K2: {source_csv}. First run "
            f"src\\run_exp1_dr_benchmark.bat {mode} (or directly "
            f"venv\\python.exe -m src.experiments.exp1_dr_benchmark --{mode})."
        )
    src_df = pd.read_csv(source_csv)
    ok = src_df[src_df["status"] == "ok"][["dataset", "method", "seed", "n_samples_used"]].drop_duplicates()
    if ok.empty:
        raise ValueError(f"{source_csv} contains no successful row (status=='ok') - nothing to compute.")

    max_datasets = cfg.get("max_datasets")
    if max_datasets is not None:
        keep_datasets = sorted(ok["dataset"].unique())[: int(max_datasets)]
        ok = ok[ok["dataset"].isin(keep_datasets)]
        logger.info("mode=%s: restricting to %d datasets (max_datasets): %s", mode, len(keep_datasets), keep_datasets)

    tasks: list[dict[str, Any]] = [
        {"dataset_name": r.dataset, "method_name": r.method, "seed": int(r.seed), "n_samples_used": int(r.n_samples_used)}
        for r in ok.itertuples(index=False)
    ]
    todo, n_done = filter_already_done(EXPERIMENT_NAME, tasks)
    logger.info("%s (mode=%s): %d combinations already done (skipped), %d new to compute.", EXPERIMENT_NAME, mode, n_done, len(todo))

    # D_in/y is computed EXACTLY ONCE per dataset (shared by all methods/seeds
    # of that dataset), not per-task - see the module docstring.
    dataset_cache: dict[str, tuple[np.ndarray | None, np.ndarray | None, int, str | None]] = {}
    n_ok, n_err = 0, 0
    with keep_system_awake():
        with wall_clock(logger, f"{EXPERIMENT_NAME} ({len(todo)} runs)"):
            for task in progress_iter(todo, desc=EXPERIMENT_NAME, total=len(todo)):
                dataset_name = task["dataset_name"]
                method_name = task["method_name"]
                seed = task["seed"]
                key = RunKey(EXPERIMENT_NAME, dataset_name, method_name, seed)

                if dataset_name not in dataset_cache:
                    dataset_cache[dataset_name] = _load_dataset_cache(dataset_name, task["n_samples_used"])
                D_in, y, n_classes, load_error = dataset_cache[dataset_name]

                t0 = time.perf_counter()
                out_row: dict[str, Any] = {k: np.nan for k in COLUMN_KEYS}
                if load_error is not None:
                    status, error = "error", load_error
                elif n_classes < MIN_CLASSES_FOR_GEOMETRY:
                    status, error = "ok", ""
                    out_row["n_classes"] = n_classes
                    out_row["note"] = NOTE_INSUFFICIENT_CLASSES
                    for gk in CLUSTER_GEOMETRY_KEYS:
                        out_row[gk] = float("nan")
                elif n_classes > MAX_CLASSES_FOR_GEOMETRY:
                    # several synthetic manifolds (swiss_roll, s_curve, sphere,
                    # severed_sphere, helix, torus) have `y` as a continuous
                    # coordinate, not a categorical class - see
                    # src/sammon/cluster_geometry.py::MAX_CLASSES_FOR_GEOMETRY
                    status, error = "ok", ""
                    out_row["n_classes"] = n_classes
                    out_row["note"] = NOTE_EXCESSIVE_CLASSES
                    for gk in CLUSTER_GEOMETRY_KEYS:
                        out_row[gk] = float("nan")
                else:
                    try:
                        Y = load_embedding(RunKey(SOURCE_EXPERIMENT, dataset_name, method_name, seed))
                        Y = np.asarray(Y, dtype=np.float64)
                        if Y.shape[0] != D_in.shape[0]:
                            raise ValueError(
                                f"Number of embedding points ({Y.shape[0]}) does not match the D_in "
                                f"subsample ({D_in.shape[0]}) - check n_samples_used/subsample seed."
                            )
                        D_out = squareform(pdist(Y, metric="euclidean"))
                        metrics = cluster_geometry_metrics(D_in, D_out, y, k_neighbors=cg_k)
                        out_row.update(metrics)
                        out_row["n_classes"] = n_classes
                        out_row["note"] = ""
                        status, error = "ok", ""
                    except Exception as exc:
                        status, error = "error", f"{type(exc).__name__}: {exc}"

                out_row["status"] = status
                out_row["error"] = error
                out_row["wall_time_sec"] = time.perf_counter() - t0
                append_result(key, out_row)
                if status == "ok":
                    n_ok += 1
                else:
                    n_err += 1
                    logger.warning("Run failed: dataset=%s method=%s seed=%s -> %s", dataset_name, method_name, seed, error)

    logger.info("Done. Newly successful: %d, newly failed: %d, skipped: %d.", n_ok, n_err, n_done)
    write_done_file(EXPERIMENT_NAME, logger, extra_info={"mode": mode})


if __name__ == "__main__":
    main()
