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
Q1 step 1 (documentation/2026-09-14_exp8_prop2_check.md) - post-hoc EMPIRICAL
TEST of Proposition 2 (clanek/sections/03_metoda.tex, lines ~165-194: local
residual R_near(Y) and rho_NN) OVER ALREADY COMPLETED embeddings from
Experiment 6 (`results/data/[<mode>/]embeddings/exp6_alpha_curves` via
`resolve_experiment_name`). NO new DR computation - only loading the .npy
embedding, recomputing D/eps_D EXACTLY the same way as
`exp6_alpha_curves.py::_run_single` (load_dataset, subsample_dataset,
to_distance_matrix, estimate_eps_D - all REUSED, not redefined), and
evaluating the pure numerical part in `src/sammon/prop2_check.py`.

For each row (dataset, seed, alpha>0) from the `exp6_alpha_curves` grid
(same mode - datasets/n_max/alpha_grid/seeds/eps_D_q are taken DIRECTLY
from `resolve_experiment_config('exp6_alpha_curves', mode)`, NO own copy of
these lists, see the K6/K8 convention "YAML anchor, not a copy"):
  Ybar = embedding at alpha=0.0 (same dataset/seed)         [RunKey('exp6_alpha_curves', ..., 'alpha0.0', seed)]
  Y    = embedding at the given alpha>0 (same dataset/seed) [RunKey('exp6_alpha_curves', ..., f'alpha{alpha}', seed)]
theta/m/rho_NN are computed over the SAME distance matrix D using the
existing functions `src.experiments.dataset_properties.nn_distances_from_D`/
`nn_ratio` (no new definition of rho_NN).

The assumption (P2) sigma_alpha(Y) <= sigma_alpha(Ybar) is NOT assumed, it
is VERIFIED (column `p2_holds`); because SMACOF only converges to
`exp6_alpha_curves.tol` (not to the exact global minimum), a tolerance
`exp8_prop2_check.p2_tolerance_rel/p2_tolerance_abs` is used
(`resolve_p2_holds`, src/sammon/prop2_check.py) - rows where P2 does not
hold even with the tolerance are NOT silently discarded: they remain in the
CSV with `p2_holds=False` and feed into a separate macro for the fraction of
discarded rows (export_numbers.py), while all other aggregations/figures use
only `p2_holds=True` rows.

Rows with exact duplicates in the input data (D_ij=0) are NOT discarded
just because D_min=0 - `kappa_eps=(theta+eps_D)/(D_min+eps_D)` is finite
for `eps_D>0` (that is the point of the regularization, see
remark[role eps_D], 03_metoda.tex); fail-loud (`ValueError`) only occurs if
BOTH effective regularization is missing (`eps_D <= exp8_prop2_check.eps_d_zero_tol`)
AND zero off-diagonal distances exist - see `src/sammon/prop2_check.py::
evaluate_proposition2` and documentation/2026-09-14_exp8_prop2_check.md
("Too strict a check on D_min=0", 2026-09-14). The CSV columns
`n_zero_pairs` (number of zero off-diagonal pairs) and `has_duplicates`
(bool) allow verifying that the analysis conclusions do not stem only from
datasets without duplicates.

Output: results/data/[<mode>/]exp8_prop2_check_results.csv (checkpoint,
resumable via the same mechanism as E1-E7, `exp_common.py`) +
results/data/[<mode>/]exp8_prop2_check_DONE.txt.

Run: venv\\python.exe -m src.experiments.exp8_prop2_check [--quick|--full|--smoke]
or: src\\run_exp8_prop2_check.bat [quick|full|smoke]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.common.checkpoint import RunKey, load_embedding
from src.common.config import load_config
from src.common.logging_utils import get_logger
from src.experiments.config_experiments import resolve_experiment_config
from src.experiments.exp_common import (
    filter_already_done,
    parse_mode_args,
    resolve_experiment_name,
    run_experiment_grid,
    write_done_file,
)

BASE_EXPERIMENT_NAME = "exp8_prop2_check"
EXP6_BASE_NAME = "exp6_alpha_curves"

# Canonical list of metric columns (static, NOT `discover_metric_keys` -
# this experiment does not call `sammon.metrics.evaluate`, it uses its own
# set of Proposition 2 quantities defined in `src/sammon/prop2_check.py`).
# The order follows the task spec (Q1 step 1) - "alpha" is a numeric copy of
# alpha encoded in `method` (RunKey), added for convenient analysis in pandas.
COLUMN_KEYS = [
    "n_samples", "rho_nn", "theta", "m", "d_min", "n_zero_pairs", "has_duplicates",
    "eps_d", "r_eps", "kappa_eps",
    "sigma_alpha_Y", "sigma_alpha_Ybar", "p2_holds", "p2_slack",
    "R_near_Y", "R_mid_Y", "R_far_Y", "R_near_Ybar", "R_mid_Ybar", "R_far_Ybar",
    "bound_a", "bound_b", "slack_a", "slack_b", "tight_a", "tight_b",
    "n_pairs_near", "n_pairs_mid", "n_pairs_far",
    "r_near_ratio", "predicted_factor_r_eps_alpha",
    "alpha",
]


def _method_name_for_alpha(alpha: float) -> str:
    """Same convention for encoding alpha into 'method' as `exp6_alpha_curves.py`
    (the RunKey used to load the embedding MUST use exactly the same name)."""
    return f"alpha{alpha}"


def _resolve_eps_D_kq(exp6_cfg: dict[str, Any], sammon_cfg: dict[str, Any]) -> tuple[int, float]:
    """Replicates EXACTLY the logic of `exp6_alpha_curves.py::main()` for the
    (k, q) eps_D estimate - k always from `sammon.eps_D.k`, q from
    `exp6_alpha_curves.eps_D_q` (for the given mode) if present, otherwise
    also from `sammon.eps_D.q`. Without this consistency, the eps_D used
    here would not match the weights with which the embedding was actually
    computed (see the config_experiments.yaml comment for exp8_prop2_check)."""
    k = int(sammon_cfg["eps_D"]["k"])
    q = float(exp6_cfg.get("eps_D_q", sammon_cfg["eps_D"]["q"]))
    return k, q


def _run_single(task: dict[str, Any]) -> dict[str, Any]:
    """One row: (dataset, alpha>0, seed) -> loads D/eps_D/Y/Ybar and
    evaluates Proposition 2 (`src.sammon.prop2_check.evaluate_proposition2`)."""
    from src.datasets.registry import load_dataset
    from src.datasets.subsample import subsample_dataset
    from src.experiments.dataset_properties import nn_distances_from_D, nn_ratio
    from src.methods.common import to_distance_matrix
    from src.sammon.prop2_check import evaluate_proposition2, resolve_p2_holds
    from src.sammon.weights import estimate_eps_D

    dataset_name = task["dataset_name"]
    alpha = task["alpha"]
    seed = task["seed"]
    method_name = task["method_name"]

    result: dict[str, Any] = {"dataset_name": dataset_name, "method_name": method_name, "seed": seed}
    t0 = time.perf_counter()
    try:
        # EXACTLY the same path to D as exp6_alpha_curves.py::_run_single -
        # no new DR computation, just re-assembling the input distance matrix.
        ds = load_dataset(dataset_name)
        ds = subsample_dataset(ds, n_max=task["n_max"], random_state=task["subsample_seed"])
        X = np.asarray(ds.X, dtype=np.float64)
        D = to_distance_matrix(X, "vector")
        eps_D = estimate_eps_D(D, k=task["eps_D_k"], q=task["eps_D_q"], kind="distance")

        n = D.shape[0]
        median_all = float(np.median(D[np.triu_indices(n, k=1)]))
        theta = float(np.median(nn_distances_from_D(D, 1)))
        m = median_all
        rho_nn = nn_ratio(D, 1, median_all)  # == theta / m, REUSED function from K1 (dataset_properties.py)

        y_key = RunKey(task["exp6_experiment_name"], dataset_name, method_name, seed)
        ybar_key = RunKey(task["exp6_experiment_name"], dataset_name, task["ybar_method_name"], seed)
        Y = load_embedding(y_key)
        Ybar = load_embedding(ybar_key)

        metrics = evaluate_proposition2(D, Y, Ybar, alpha, eps_D, theta, m, task["eps_d_zero_tol"])
        metrics["p2_holds"] = resolve_p2_holds(
            metrics["sigma_alpha_Y"], metrics["sigma_alpha_Ybar"], task["p2_tolerance_rel"], task["p2_tolerance_abs"],
        )
        metrics["rho_nn"] = rho_nn
        metrics["eps_d"] = eps_D
        metrics["theta"] = theta
        metrics["m"] = m

        result["status"] = "ok"
        result["error"] = ""
        result["metrics"] = metrics
        result["extra"] = {"alpha": alpha}
    except Exception as exc:  # one incompatible combination must not stop the whole run
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["metrics"] = {}
        result["extra"] = {"alpha": alpha}
    result["wall_time_sec"] = time.perf_counter() - t0
    result["embedding"] = None  # exp8 does not generate any new embedding, only analyzes existing ones
    return result


def main() -> None:
    mode = parse_mode_args(
        "Q1 step 1: post-hoc empirical test of Proposition 2 (local residual and rho_NN) over completed exp6_alpha_curves embeddings."
    )
    logger = get_logger(BASE_EXPERIMENT_NAME, mode=mode)

    exp6_cfg = resolve_experiment_config(EXP6_BASE_NAME, mode)
    prop2_cfg = resolve_experiment_config(BASE_EXPERIMENT_NAME, mode)
    sammon_cfg = load_config()["sammon"]

    from src.experiments.exp1_dr_benchmark import SUBSAMPLE_SEED

    eps_D_k, eps_D_q = _resolve_eps_D_kq(exp6_cfg, sammon_cfg)
    exp6_experiment_name = resolve_experiment_name(EXP6_BASE_NAME, mode)
    ybar_method_name = _method_name_for_alpha(0.0)

    p2_tolerance_rel = float(prop2_cfg["p2_tolerance_rel"])
    p2_tolerance_abs = float(prop2_cfg["p2_tolerance_abs"])
    eps_d_zero_tol = float(prop2_cfg["eps_d_zero_tol"])

    alpha_grid = [float(a) for a in exp6_cfg["alpha_grid"] if float(a) > 0.0]
    if not alpha_grid:
        raise ValueError(f"exp6_alpha_curves.alpha_grid (mode={mode}) contains no alpha > 0 - nothing to verify.")
    if 0.0 not in [float(a) for a in exp6_cfg["alpha_grid"]]:
        raise ValueError(f"exp6_alpha_curves.alpha_grid (mode={mode}) does not contain alpha=0.0 - missing reference Ybar.")

    EXPERIMENT_NAME = resolve_experiment_name(BASE_EXPERIMENT_NAME, mode)

    tasks: list[dict[str, Any]] = []
    for dataset_name in exp6_cfg["datasets"]:
        for alpha in alpha_grid:
            method_name = _method_name_for_alpha(alpha)
            for seed in exp6_cfg["seeds"]:
                tasks.append({
                    "dataset_name": dataset_name, "method_name": method_name, "seed": seed, "alpha": alpha,
                    "n_max": int(exp6_cfg["n_max"]), "subsample_seed": SUBSAMPLE_SEED,
                    "eps_D_k": eps_D_k, "eps_D_q": eps_D_q,
                    "exp6_experiment_name": exp6_experiment_name, "ybar_method_name": ybar_method_name,
                    "p2_tolerance_rel": p2_tolerance_rel, "p2_tolerance_abs": p2_tolerance_abs,
                    "eps_d_zero_tol": eps_d_zero_tol,
                })

    todo, n_done = filter_already_done(EXPERIMENT_NAME, tasks)
    logger.info(
        "%s (mode=%s): %d combinations already done (skipped), %d new to compute (embedding source: %s).",
        EXPERIMENT_NAME, mode, n_done, len(todo), exp6_experiment_name,
    )

    n_ok, n_err = run_experiment_grid(EXPERIMENT_NAME, todo, _run_single, COLUMN_KEYS, logger)
    logger.info("Done. Newly successful: %d, newly failed: %d, skipped: %d.", n_ok, n_err, n_done)
    write_done_file(EXPERIMENT_NAME, logger, extra_info={"mode": mode, "exp6_source": exp6_experiment_name})


if __name__ == "__main__":
    main()
