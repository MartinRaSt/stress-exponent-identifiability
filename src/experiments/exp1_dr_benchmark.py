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
E1 - main comparative benchmark: all vector + synthetic datasets from the
registry (subsampled to n_max) x all dimensionality-reduction/layout methods
(classic: PCA/MDS/Sammon/Isomap/LLE/t-SNE/UMAP/PaCMAP/TriMap, plus the
proposed alpha-Sammon family) x 10 independent seeds. Results (incl.
embeddings) go to results/data/exp1_dr_benchmark_results.csv (resumable via
checkpoint).

Run: venv\\python.exe -m src.experiments.exp1_dr_benchmark [--quick|--full]
or: src\\run_exp1_dr_benchmark.bat [quick|full]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.common.logging_utils import get_logger
from src.experiments.config_experiments import resolve_experiment_config
from src.experiments.exp_common import (
    add_dataset_scope_arg,
    add_mode_args,
    add_tier_arg,
    discover_metric_keys,
    filter_already_done,
    resolve_dataset_scope,
    resolve_experiment_name,
    resolve_mode,
    run_experiment_grid,
    write_done_file,
)

BASE_EXPERIMENT_NAME = "exp1_dr_benchmark"

# fixed subsample seed (INDEPENDENT of the method-seed loop): all methods
# and all seeds are compared on the SAME subset of points of a given
# dataset, so that differences between seeds reflect only the randomness of
# the method itself, not a different random data selection (see
# src/datasets/subsample.py, random_state parameter)
SUBSAMPLE_SEED = 42


def _run_single(task: dict[str, Any]) -> dict[str, Any]:
    """Computes one run (dataset, method, seed) in a separate worker process."""
    from src.common.seeding import set_seed
    from src.datasets.registry import load_dataset
    from src.datasets.subsample import subsample_dataset
    from src.methods.common import method_config
    from src.methods.registry import get_method
    from src.sammon.metrics import evaluate

    dataset_name = task["dataset_name"]
    method_name = task["method_name"]
    seed = task["seed"]
    n_components = task["n_components"]
    n_max = task["n_max"]
    eval_kwargs = task["eval_kwargs"]

    set_seed(seed)
    result: dict[str, Any] = {"dataset_name": dataset_name, "method_name": method_name, "seed": seed}
    t0 = time.perf_counter()
    try:
        ds = load_dataset(dataset_name)
        ds = subsample_dataset(ds, n_max=n_max, random_state=SUBSAMPLE_SEED)
        method = get_method(method_name)
        data = ds.X if ds.kind == "vector" else (ds.D if ds.kind == "distance" else ds.graph)

        Y = method.fit_transform(data, ds.kind, seed=seed, n_components=n_components)
        metrics = evaluate(data, Y, ds.y, ds.kind, **eval_kwargs)

        try:
            device = str(method_config(method_name).get("device", "cpu"))
        except KeyError:
            device = "cpu"
        # methods with an automatic hyperparameter grid search (sammon_alpha_auto,
        # sammon_multiscale, tsne_auto, umap_auto) store the selected value
        # into self.last_selected_hyperparam after fit_transform (see
        # src/methods/sammon_alpha.py, src/methods/tsne_auto.py,
        # src/methods/umap_auto.py) - other methods do not have the attribute -> "".
        selected_hyperparam = str(getattr(method, "last_selected_hyperparam", ""))

        result["status"] = "ok"
        result["error"] = ""
        result["wall_time_sec"] = time.perf_counter() - t0
        result["embedding"] = Y
        result["metrics"] = metrics
        result["extra"] = {"device": device, "n_samples_used": ds.n_samples, "selected_hyperparam": selected_hyperparam}
    except Exception as exc:  # intentionally broad catch - one incompatible combination must not stop the whole run
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["wall_time_sec"] = time.perf_counter() - t0
        result["embedding"] = None
        result["metrics"] = {}
        result["extra"] = {"device": "", "n_samples_used": np.nan, "selected_hyperparam": ""}
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="E1: benchmark of dimensionality-reduction/layout methods across all vector+synthetic datasets.")
    add_mode_args(parser)
    add_dataset_scope_arg(parser)
    add_tier_arg(parser)
    args = parser.parse_args()
    mode = resolve_mode(args)

    logger = get_logger(BASE_EXPERIMENT_NAME, mode=mode)
    cfg = resolve_experiment_config(BASE_EXPERIMENT_NAME, mode)
    EXPERIMENT_NAME = resolve_experiment_name(BASE_EXPERIMENT_NAME, mode)

    from src.methods.common import is_compatible
    from src.methods.registry import get_method

    metric_keys, eval_kwargs = discover_metric_keys(n_components=cfg["n_components"])
    column_keys = metric_keys + ["device", "n_samples_used", "selected_hyperparam"]

    n_max_overrides: dict[str, int] = cfg.get("n_max_overrides", {})
    n_components = cfg["n_components"]

    # Q1 step 2 (reserse/2026-09-14_specifikace_rozsireni_q1.md, A.5, A.9 item 6):
    # --datasets all|core|holdout merges 'datasets' (core, used to fit rules)
    # and 'datasets_holdout' (new Q1 candidates) in code - the configuration
    # must NEVER have a shared list (see A.5 - the hold-out must not affect fitting).
    core_datasets: list[str] = list(cfg["datasets"])
    holdout_datasets: list[str] = list(cfg.get("datasets_holdout", []))
    scope_datasets = resolve_dataset_scope(args, core_datasets, holdout_datasets)
    logger.info("--datasets=%s -> %d datasets (core=%d, holdout=%d).", args.datasets, len(scope_datasets), len(core_datasets), len(holdout_datasets))

    # --tier all|tier1|tier2 (E1 only, A.8/A.9 item 6): restricts methods to
    # method_tiers.<tier> - 'all' (default) has no effect on the original behavior.
    method_tiers: dict[str, list[str]] = cfg.get("method_tiers", {})
    tier_methods: set[str] | None = None
    if args.tier != "all":
        if args.tier not in method_tiers:
            raise KeyError(f"--tier='{args.tier}' requires exp1_dr_benchmark.method_tiers.{args.tier} in config_experiments.yaml.")
        tier_methods = set(method_tiers[args.tier])
        logger.info("--tier=%s -> %d methods (%s).", args.tier, len(tier_methods), sorted(tier_methods))

    # smoke: asymmetric coverage - all methods only on 'full_coverage_datasets'
    # (small, fast), on the remaining datasets only 'reduced_methods' (see
    # config_experiments.yaml exp1_dr_benchmark.smoke comment - the goal is
    # under 5 minutes total; the full 32 datasets x 16 methods would be too
    # slow, mainly due to OpenML download/parsing of mnist_784/fashion_mnist)
    full_coverage_datasets = set(cfg.get("full_coverage_datasets", []))
    full_coverage_n_max = cfg.get("full_coverage_n_max")
    reduced_methods = cfg.get("reduced_methods")

    tasks: list[dict[str, Any]] = []
    for dataset_name in scope_datasets:
        if reduced_methods is not None and dataset_name in full_coverage_datasets:
            n_max = int(full_coverage_n_max)
            methods_for_dataset = cfg["methods"]
        elif reduced_methods is not None:
            n_max = int(n_max_overrides.get(dataset_name, cfg["n_max"]))
            methods_for_dataset = reduced_methods
        else:
            n_max = int(n_max_overrides.get(dataset_name, cfg["n_max"]))
            methods_for_dataset = cfg["methods"]
        if tier_methods is not None:
            methods_for_dataset = [m for m in methods_for_dataset if m in tier_methods]
        for method_name in methods_for_dataset:
            method = get_method(method_name)
            if not is_compatible(method.accepts, "vector"):
                logger.warning(
                    "Method '%s' (accepts=%s) does not support kind='vector', skipping for dataset '%s'.",
                    method_name, method.accepts, dataset_name,
                )
                continue
            for seed in cfg["seeds"]:
                tasks.append({
                    "dataset_name": dataset_name, "method_name": method_name, "seed": seed,
                    "n_components": n_components, "n_max": n_max, "eval_kwargs": eval_kwargs,
                })

    todo, n_done = filter_already_done(EXPERIMENT_NAME, tasks)
    logger.info("%s (mode=%s): %d combinations already done (skipped), %d new to compute.", EXPERIMENT_NAME, mode, n_done, len(todo))

    n_ok, n_err = run_experiment_grid(EXPERIMENT_NAME, todo, _run_single, column_keys, logger)
    logger.info("Done. Newly successful: %d, newly failed: %d, skipped: %d.", n_ok, n_err, n_done)
    write_done_file(EXPERIMENT_NAME, logger, extra_info={"mode": mode})


if __name__ == "__main__":
    main()
