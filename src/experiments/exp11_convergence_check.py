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
E11: convergence-vs-local-minimum diagnostic for the exp10_identifiability_check
non-negativity violations (documentation/2026-09-17_zadani_exp11_konvergence.md).

`results/data/exp10_identifiability_check_results.csv` has 13 of its 384
alpha>0 rows with `both_gaps_nonneg == False` (a negative `Delta` or
`Delta_prime`) - but Veta 1 (Lemma 2's identity plus the global minimality
of Y_0*/Y_alpha*) requires BOTH gaps to be non-negative
(clanek/sections/03_metoda.tex, eq:gaps). E10 does not run any solver itself
- it loads Y_0*/Y_alpha* from the exp6_alpha_curves cache
(`src.common.checkpoint.load_embedding`), which were fitted with SMACOF at
`sammon.smacof`/`exp6_alpha_curves.max_iter=300`, `tol=1e-4`. A negative gap
can therefore mean either (a) the cached embedding is merely under-converged
(a looser stopping tolerance/iteration budget), or (b) SMACOF genuinely
settled in a DIFFERENT local minimum than the one Veta 1's global-minimality
argument assumes (SMACOF/Guttman majorization is only guaranteed to reach a
stationary point, not the global minimum - de Leeuw 2009, \\citep{deleeuw2009smacof}).

This experiment recomputes Delta/Delta_prime for each flagged (dataset,
alpha) in THREE variants, one row per (dataset, alpha, variant):

  - 'baseline': the cached Y_0*/Y_alpha* from exp6_alpha_curves, UNCHANGED -
    a reproduction check, not a variant: it must match the corresponding row
    of `exp10_identifiability_check_results.csv` to `baseline_match_tol`
    (fail-loud otherwise - if it does not match, exp11 is computing
    something different from E10, which would be a bug in exp11, not a
    legitimate finding).
  - 'tight': SMACOF refit from scratch with the SAME PCA initialization as
    exp6_alpha_curves, but a much tighter stopping criterion
    (`exp11_convergence_check.tight.max_iter/tol`, e.g. 5000/1e-10 vs. the
    production 300/1e-4). If the negative gap disappears here, it was a
    tolerance/iteration-budget artifact.
  - 'warm': SMACOF refit with a CROSS (warm-start) initialization instead of
    PCA - the alpha=0 objective started from the cached alpha-weighted
    optimum Y_alpha*, and the alpha objective started from the cached
    alpha=0 optimum Y_0* - using the SAME tight tolerance as 'tight'. If the
    negative gap disappears only here (not in 'tight'), the PCA-initialized
    SMACOF run was stuck in a genuinely different local minimum: a measured
    limitation of the solver, to be reported as such in the article, not
    papered over.
  - If the negative gap survives even 'warm', the cause is something else
    (e.g. a genuine near-tie between two comparably good local minima, or an
    eps_D/numerics issue) and must be reported, not hidden.

IMPORTANT (see also the config_experiments.yaml comment above
`exp11_convergence_check:`): the (dataset, alpha) targets and the reference
E10 values used by the 'baseline' check are ALWAYS read from the FULL
production `exp6_alpha_curves`/`exp10_identifiability_check` runs
(`results/data/exp6_alpha_curves_results.csv`,
`results/data/exp10_identifiability_check_results.csv` and their embeddings
cache), regardless of exp11_convergence_check's OWN --quick/--smoke/--full
mode - this experiment is a targeted re-diagnosis of a FIXED, already
identified set of violating rows, not a recomputation on differently-sized
data. exp11's own mode only restricts which (dataset, alpha) targets are
processed and how tight the 'tight'/'warm' solver budget is
(`exp11_convergence_check.smoke.*` in config_experiments.yaml), so --smoke
still exercises every code path (including the fail-loud baseline check)
quickly - glass (one of the smoke targets) has only n=214 points, so even a
tight SMACOF refit finishes in well under a second.

NOTE on `glass` and `sammon.smacof.small_n_threshold`/`tol_small_n`: glass
has n=214, i.e. below `small_n_threshold=500`. However, the 'tight' variant
here uses `exp11_convergence_check.tight.tol` UNCONDITIONALLY, regardless of
n (it does NOT branch on small_n) - so 'tight' is NOT simply "the small_n
tolerance instead of the default one". It is a separate, uniformly applied,
much tighter tolerance for every dataset in `targets` (small or large), used
purely to distinguish "solver tolerance" from "different local minimum".

Output: results/data/[<mode>/]exp11_convergence_check_results.csv
(checkpoint, resumable) + results/data/[<mode>/]exp11_convergence_check_DONE.txt.

Run: venv\\python.exe -m src.experiments.exp11_convergence_check [--quick|--full|--smoke]
or: src\\run_exp11_convergence_check.bat [quick|full|smoke]
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.common.checkpoint import RunKey, load_embedding, results_csv_path
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

BASE_EXPERIMENT_NAME = "exp11_convergence_check"
EXP6_BASE_NAME = "exp6_alpha_curves"
EXP10_BASE_NAME = "exp10_identifiability_check"

VARIANTS = ("baseline", "tight", "warm")

# Canonical CSV schema (documentation/2026-09-17_zadani_exp11_konvergence.md,
# section "Vystup"). 'dataset'/'method'/'seed'/'experiment' are added
# automatically by src.common.checkpoint.append_result (from the RunKey);
# 'status'/'error'/'wall_time_sec' are added automatically by
# src.experiments.exp_common.run_experiment_grid.
COLUMN_KEYS = [
    "alpha", "variant", "n_samples", "eps_d",
    "sigma0_Y0", "sigma0_Ya", "sigmaA_Y0", "sigmaA_Ya",
    "Delta", "Delta_prime", "both_gaps_nonneg",
    "n_iter_alpha0", "n_iter_alpha", "converged_alpha0", "converged_alpha",
    "delta_vs_baseline",
]

# Fields checked (absolute difference) by `_check_baseline_match`.
_BASELINE_MATCH_FIELDS = [
    "n_samples", "eps_d", "sigma0_Y0", "sigma0_Ya", "sigmaA_Y0", "sigmaA_Ya", "Delta", "Delta_prime",
]

_E10_REF_USECOLS = [
    "dataset", "alpha", "seed", "status",
    "n_samples", "eps_d", "sigma0_Y0", "sigma0_Ya", "sigmaA_Y0", "sigmaA_Ya",
    "Delta", "Delta_prime", "both_gaps_nonneg",
]

_E6_NITER_USECOLS = ["dataset", "method", "seed", "n_iter_smacof", "status"]


def _method_name_for_alpha(alpha: float) -> str:
    """Same 'method' naming convention as exp6_alpha_curves.py/exp10_identifiability_check.py
    (the RunKey used to load a cached embedding MUST match exactly)."""
    return f"alpha{alpha}"


def _method_name_for_task(alpha: float, variant: str) -> str:
    """The 'method' name for THIS experiment's own RunKey/checkpoint - encodes
    both alpha and variant, since a row here is (dataset, alpha, variant)."""
    return f"{_method_name_for_alpha(alpha)}_{variant}"


def _check_baseline_match(computed: dict[str, Any], ref: dict[str, Any], tol: float, dataset_name: str, alpha: float) -> None:
    """Fail-loud check that the freshly computed 'baseline' variant
    reproduces the corresponding row of the FULL
    `results/data/exp10_identifiability_check_results.csv` within `tol`
    RELATIVE difference per field (`tol` is also used as an absolute floor,
    so fields that are legitimately ~0 still compare sanely).

    The tolerance MUST be relative: the stress fields are of magnitude ~1e4,
    where a single ulp is already ~2e-12, so an absolute 1e-12 bound could
    never be met by a correct computation (observed 2026-09-17 on
    dataset=glass, alpha=0.75: |diff|=1.8e-12, relative 1.5e-16 = eps).

    This is a reproduction guarantee, not a numerical tolerance to relax: a
    mismatch means exp11 is computing something DIFFERENT from E10 (wrong
    eps_D/(k,q), wrong n_max/subsample seed, or a wrong embedding lookup),
    i.e. a bug in exp11 itself, not a legitimate finding about SMACOF
    convergence. Raises ValueError listing every mismatched field (never
    silently records a wrong 'baseline' row)."""
    mismatches: list[str] = []
    for field in _BASELINE_MATCH_FIELDS:
        a = float(computed[field])
        b = float(ref[field])
        if not math.isclose(a, b, rel_tol=tol, abs_tol=tol):
            scale = max(abs(a), abs(b), 1.0)
            mismatches.append(
                f"{field}: exp11={a!r} vs exp10={b!r} "
                f"(|diff|={abs(a - b):.3e}, relative={abs(a - b) / scale:.3e} > rel_tol={tol:.3e})"
            )
    if mismatches:
        raise ValueError(
            f"exp11 'baseline' variant for (dataset={dataset_name}, alpha={alpha}) does NOT reproduce "
            f"results/data/exp10_identifiability_check_results.csv within relative baseline_match_tol={tol:.3e} "
            "- exp11 is computing something different from E10 (bug), not a numerical variant:\n  "
            + "\n  ".join(mismatches)
        )


def _delta_vs_baseline(delta_value: float, delta_ref: float, variant: str) -> float:
    """delta_vs_baseline = Delta(variant) - Delta(baseline) for the same
    (dataset, alpha) (documentation/2026-09-17_zadani_exp11_konvergence.md,
    section "Vystup"). Exactly 0.0 for variant=='baseline' BY DEFINITION
    (not computed via subtraction, to avoid spurious floating-point noise
    for what is conceptually an identity: baseline compared to itself)."""
    if variant == "baseline":
        return 0.0
    return float(delta_value) - float(delta_ref)


def _load_e10_reference_rows(exp10_experiment_name: str, seed: int) -> dict[tuple[str, float], dict[str, Any]]:
    """{(dataset, alpha): row_dict} from the FULL exp10_identifiability_check
    results CSV (status=='ok' rows only), filtered to the canonical seed -
    the reference values the 'baseline' variant must reproduce, and the
    Delta reference used for `delta_vs_baseline` in every variant (module
    docstring: exp11 ALWAYS reads the FULL E10 output, regardless of its own mode)."""
    path = results_csv_path(exp10_experiment_name)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing exp10_identifiability_check results: {path}\n"
            "Run first: venv\\python.exe -m src.experiments.exp10_identifiability_check --full."
        )
    df = pd.read_csv(path, usecols=_E10_REF_USECOLS)
    df = df[(df["seed"] == seed) & (df["status"] == "ok")]
    out: dict[tuple[str, float], dict[str, Any]] = {}
    for _, row in df.iterrows():
        out[(str(row["dataset"]), float(row["alpha"]))] = row.to_dict()
    return out


def _load_e6_n_iter_map(exp6_experiment_name: str, seed: int) -> dict[tuple[str, str], float]:
    """{(dataset, method): n_iter_smacof} from the FULL exp6_alpha_curves
    results CSV, filtered to the canonical seed.

    Used only for the 'baseline' variant's n_iter_alpha0/n_iter_alpha
    columns: 'baseline' does not rerun SMACOF (by definition it reuses the
    cached embedding as-is), but E6's OWN already-recorded iteration count
    is directly informative for the diagnosis - did the cached embedding
    even reach the tol stopping criterion, or did it hit
    exp6_alpha_curves.max_iter (a real, not fabricated, number from the
    completed E6 run)."""
    path = results_csv_path(exp6_experiment_name)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing exp6_alpha_curves results: {path}\n"
            "Run first: venv\\python.exe -m src.experiments.exp6_alpha_curves --full (K6)."
        )
    df = pd.read_csv(path, usecols=_E6_NITER_USECOLS)
    df = df[df["seed"] == seed]
    out: dict[tuple[str, str], float] = {}
    for _, row in df.iterrows():
        out[(str(row["dataset"]), str(row["method"]))] = float(row["n_iter_smacof"])
    return out


def _fit_smacof(D: np.ndarray, alpha: float, eps_D: float, Y_init: np.ndarray, max_iter: int, tol: float, sammon_cfg: dict[str, Any]) -> tuple[np.ndarray, dict]:
    """One SMACOF fit for a given alpha - the SAME call convention as
    exp6_alpha_curves.py::_run_single's CPU branch (device is always 'cpu'
    here: exp6_alpha_curves.device is 'cpu' in production, and this
    diagnostic does not need the GPU code path)."""
    from src.sammon.solvers.smacof import smacof_solve
    from src.sammon.weights import alpha_weights, compute_Z_from_W

    W = alpha_weights(D, alpha, eps_D)
    Z = compute_Z_from_W(D, W)
    Y, history = smacof_solve(
        D, W, Z, Y_init, max_iter=max_iter, tol=tol, eps_num=sammon_cfg["eps_num"],
        dense_pinv_threshold=sammon_cfg["smacof"]["dense_pinv_threshold"],
        cg_max_iter=sammon_cfg["smacof"]["cg_max_iter"], cg_tol=sammon_cfg["smacof"]["cg_tol"],
        reg_rho=sammon_cfg["smacof"]["reg_rho"], verbose=False,
    )
    return Y, history


def _run_single(task: dict[str, Any]) -> dict[str, Any]:
    """One row: (dataset, alpha, variant) -> Delta/Delta_prime under one of
    the three ways of obtaining Y_0*/Y_alpha* described in the module
    docstring."""
    from src.datasets.registry import load_dataset
    from src.datasets.subsample import subsample_dataset
    from src.methods.common import to_distance_matrix
    from src.sammon.identifiability import veta1_quantities
    from src.sammon.init import init_pca
    from src.sammon.weights import estimate_eps_D

    dataset_name = task["dataset_name"]
    alpha = task["alpha"]
    variant = task["variant"]
    seed = task["seed"]
    method_name = task["method_name"]

    result: dict[str, Any] = {"dataset_name": dataset_name, "method_name": method_name, "seed": seed}
    t0 = time.perf_counter()
    try:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant='{variant}' (expected one of {VARIANTS}).")

        ds = load_dataset(dataset_name)
        ds = subsample_dataset(ds, n_max=task["n_max"], random_state=task["subsample_seed"])
        X = np.asarray(ds.X, dtype=np.float64)
        D = to_distance_matrix(X, "vector")
        eps_D = estimate_eps_D(D, k=task["eps_D_k"], q=task["eps_D_q"], kind="distance")
        n = D.shape[0]

        exp6_name = task["exp6_experiment_name"]
        Y0_cached = load_embedding(RunKey(exp6_name, dataset_name, _method_name_for_alpha(0.0), seed))
        Ya_cached = load_embedding(RunKey(exp6_name, dataset_name, _method_name_for_alpha(alpha), seed))
        sammon_cfg = task["sammon_cfg"]

        n_iter_alpha0: float = float("nan")
        n_iter_alpha: float = float("nan")
        converged_alpha0: Any = float("nan")
        converged_alpha: Any = float("nan")

        if variant == "baseline":
            # No SMACOF rerun - reuse the cached E6 embeddings exactly as
            # they are (this IS what E10 loaded). n_iter/converged come from
            # E6's OWN already-recorded iteration count (real data, not
            # recomputed nor fabricated - see `_load_e6_n_iter_map`).
            Y0, Ya = Y0_cached, Ya_cached
            n_iter_alpha0 = task["baseline_n_iter_alpha0"]
            n_iter_alpha = task["baseline_n_iter_alpha"]
            converged_alpha0 = bool(n_iter_alpha0 < task["exp6_max_iter"])
            converged_alpha = bool(n_iter_alpha < task["exp6_max_iter"])
        elif variant == "tight":
            # Same PCA initialization as exp6_alpha_curves, tighter stopping criterion.
            Y0_init = init_pca(X, task["n_components"], seed)
            Ya_init = init_pca(X, task["n_components"], seed)
            Y0, hist0 = _fit_smacof(D, 0.0, eps_D, Y0_init, task["tight_max_iter"], task["tight_tol"], sammon_cfg)
            Ya, histA = _fit_smacof(D, alpha, eps_D, Ya_init, task["tight_max_iter"], task["tight_tol"], sammon_cfg)
            n_iter_alpha0, n_iter_alpha = float(hist0["n_iter"]), float(histA["n_iter"])
            converged_alpha0 = bool(n_iter_alpha0 < task["tight_max_iter"])
            converged_alpha = bool(n_iter_alpha < task["tight_max_iter"])
        else:  # variant == "warm"
            # Cross (warm-start) initialization: the alpha=0 objective started
            # from the cached alpha-weighted optimum Y_alpha*, and the alpha
            # objective started from the cached alpha=0 optimum Y_0* - same
            # tight tolerance as 'tight'. Tests whether SMACOF converges back
            # to (about) the same point from a different, already-good start.
            Y0, hist0 = _fit_smacof(D, 0.0, eps_D, Ya_cached, task["tight_max_iter"], task["tight_tol"], sammon_cfg)
            Ya, histA = _fit_smacof(D, alpha, eps_D, Y0_cached, task["tight_max_iter"], task["tight_tol"], sammon_cfg)
            n_iter_alpha0, n_iter_alpha = float(hist0["n_iter"]), float(histA["n_iter"])
            converged_alpha0 = bool(n_iter_alpha0 < task["tight_max_iter"])
            converged_alpha = bool(n_iter_alpha < task["tight_max_iter"])

        v1 = veta1_quantities(D, Y0, Ya, alpha, eps_D)

        computed = {
            "n_samples": n, "eps_d": eps_D,
            "sigma0_Y0": v1["sigma0_Y0"], "sigma0_Ya": v1["sigma0_Ya"],
            "sigmaA_Y0": v1["sigmaA_Y0"], "sigmaA_Ya": v1["sigmaA_Ya"],
            "Delta": v1["Delta"], "Delta_prime": v1["Delta_prime"],
        }

        e10_ref = task["e10_ref"]
        if variant == "baseline":
            _check_baseline_match(computed, e10_ref, task["baseline_match_tol"], dataset_name, alpha)

        delta_vs_baseline = _delta_vs_baseline(v1["Delta"], float(e10_ref["Delta"]), variant)

        metrics: dict[str, Any] = {
            "n_samples": n, "eps_d": eps_D,
            "sigma0_Y0": v1["sigma0_Y0"], "sigma0_Ya": v1["sigma0_Ya"],
            "sigmaA_Y0": v1["sigmaA_Y0"], "sigmaA_Ya": v1["sigmaA_Ya"],
            "Delta": v1["Delta"], "Delta_prime": v1["Delta_prime"], "both_gaps_nonneg": v1["both_gaps_nonneg"],
            "n_iter_alpha0": n_iter_alpha0, "n_iter_alpha": n_iter_alpha,
            "converged_alpha0": converged_alpha0, "converged_alpha": converged_alpha,
            "delta_vs_baseline": delta_vs_baseline,
        }

        result["status"] = "ok"
        result["error"] = ""
        result["metrics"] = metrics
        result["extra"] = {"alpha": alpha, "variant": variant}
    except Exception as exc:  # one incompatible combination must not stop the whole run
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["metrics"] = {}
        result["extra"] = {"alpha": alpha, "variant": variant}
    result["wall_time_sec"] = time.perf_counter() - t0
    result["embedding"] = None  # exp11 does not generate any DR embedding for downstream use, only diagnoses convergence
    return result


def main() -> None:
    mode = parse_mode_args(
        "E11: convergence-vs-local-minimum diagnostic for the exp10_identifiability_check "
        "both_gaps_nonneg==False rows (documentation/2026-09-17_zadani_exp11_konvergence.md)."
    )
    logger = get_logger(BASE_EXPERIMENT_NAME, mode=mode)

    exp11_cfg = resolve_experiment_config(BASE_EXPERIMENT_NAME, mode)

    # Underlying data (E6 embeddings, E10 reference CSV) is ALWAYS the FULL
    # production run - see the module docstring and the config_experiments.yaml
    # comment above `exp11_convergence_check:`.
    exp6_cfg = resolve_experiment_config(EXP6_BASE_NAME, "full")
    exp10_cfg = resolve_experiment_config(EXP10_BASE_NAME, "full")
    sammon_cfg = load_config()["sammon"]

    from src.experiments.exp1_dr_benchmark import SUBSAMPLE_SEED
    from src.experiments.exp8_prop2_check import _resolve_eps_D_kq

    eps_D_k, eps_D_q = _resolve_eps_D_kq(exp6_cfg, sammon_cfg)
    exp6_experiment_name = resolve_experiment_name(EXP6_BASE_NAME, "full")
    exp10_experiment_name = resolve_experiment_name(EXP10_BASE_NAME, "full")

    seeds_list = list(exp6_cfg["seeds"])
    seed_index = int(exp10_cfg["embedding_seed_index"])
    if not (0 <= seed_index < len(seeds_list)):
        raise ValueError(
            f"exp10_identifiability_check.embedding_seed_index={seed_index} out of range for "
            f"exp6_alpha_curves.seeds={seeds_list} (full mode)."
        )
    canonical_seed = int(seeds_list[seed_index])
    exp6_max_iter = int(exp6_cfg["max_iter"])
    n_components = int(exp6_cfg["n_components"])

    tight_max_iter = int(exp11_cfg["tight"]["max_iter"])
    tight_tol = float(exp11_cfg["tight"]["tol"])
    baseline_match_tol = float(exp11_cfg["baseline_match_tol"])

    e10_ref = _load_e10_reference_rows(exp10_experiment_name, canonical_seed)
    e6_niter = _load_e6_n_iter_map(exp6_experiment_name, canonical_seed)

    logger.info(
        "%s (mode=%s): data source FULL %s / %s, canonical seed=%d, tight max_iter=%d tol=%.1e.",
        BASE_EXPERIMENT_NAME, mode, exp6_experiment_name, exp10_experiment_name, canonical_seed, tight_max_iter, tight_tol,
    )

    EXPERIMENT_NAME = resolve_experiment_name(BASE_EXPERIMENT_NAME, mode)

    tasks: list[dict[str, Any]] = []
    missing: list[tuple[str, float]] = []
    for target in exp11_cfg["targets"]:
        dataset_name = str(target["dataset"])
        for alpha_raw in target["alphas"]:
            alpha = float(alpha_raw)
            key = (dataset_name, alpha)
            if key not in e10_ref:
                missing.append(key)
                continue
            alpha0_method = _method_name_for_alpha(0.0)
            alpha_method = _method_name_for_alpha(alpha)
            niter_key0 = (dataset_name, alpha0_method)
            niter_keyA = (dataset_name, alpha_method)
            if niter_key0 not in e6_niter or niter_keyA not in e6_niter:
                raise ValueError(
                    f"Missing exp6_alpha_curves n_iter_smacof for dataset={dataset_name}, "
                    f"method in {{{alpha0_method}, {alpha_method}}}, seed={canonical_seed} - "
                    "run exp6_alpha_curves --full first (K6)."
                )
            for variant in VARIANTS:
                tasks.append({
                    "dataset_name": dataset_name, "method_name": _method_name_for_task(alpha, variant),
                    "seed": canonical_seed, "alpha": alpha, "variant": variant,
                    "n_max": int(exp6_cfg["n_max"]), "subsample_seed": SUBSAMPLE_SEED,
                    "eps_D_k": eps_D_k, "eps_D_q": eps_D_q,
                    "exp6_experiment_name": exp6_experiment_name, "n_components": n_components,
                    "sammon_cfg": sammon_cfg, "tight_max_iter": tight_max_iter, "tight_tol": tight_tol,
                    "baseline_match_tol": baseline_match_tol, "e10_ref": e10_ref[key],
                    "exp6_max_iter": exp6_max_iter,
                    "baseline_n_iter_alpha0": e6_niter[niter_key0], "baseline_n_iter_alpha": e6_niter[niter_keyA],
                })

    if missing:
        raise ValueError(
            "exp11_convergence_check.targets references (dataset, alpha) pairs missing from the FULL "
            f"exp10_identifiability_check results (status=='ok', seed={canonical_seed}): {missing}. "
            "Re-run exp10_identifiability_check --full first, or fix the 'targets' list in config_experiments.yaml."
        )

    logger.info(
        "%s (mode=%s): %d (dataset, alpha) targets x %d variants = %d tasks.",
        EXPERIMENT_NAME, mode, len(tasks) // len(VARIANTS), len(VARIANTS), len(tasks),
    )

    todo, n_done = filter_already_done(EXPERIMENT_NAME, tasks)
    logger.info("%s (mode=%s): %d combinations already done (skipped), %d new to compute.", EXPERIMENT_NAME, mode, n_done, len(todo))

    n_ok, n_err = run_experiment_grid(EXPERIMENT_NAME, todo, _run_single, COLUMN_KEYS, logger)
    logger.info("Done. Newly successful: %d, newly failed: %d, skipped: %d.", n_ok, n_err, n_done)
    write_done_file(
        EXPERIMENT_NAME, logger,
        extra_info={"mode": mode, "exp6_source": exp6_experiment_name, "exp10_source": exp10_experiment_name, "canonical_seed": canonical_seed},
    )


if __name__ == "__main__":
    main()
