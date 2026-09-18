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
Numerical verification of the identifiability-of-the-stress-exponent theory
(reserse/2026-09-17_zostreni_propozice2.md, section 10 - "Specifikace
numerickeho overeni pro python-codera"; called "E9"/"exp9_identifiability_check"
there, RENAMED here to E10/exp10_identifiability_check to avoid a name
collision with the already-existing `src/experiments/exp9_metric_fidelity.py`).

NO new DR run. Inputs, all reused (not redefined):
  - `src/experiments/exp8_prop2_check.py`: the loader
    (load_dataset/subsample_dataset/to_distance_matrix/estimate_eps_D), the
    eps_D (k,q) resolution (`_resolve_eps_D_kq`), and theta/m/rho_NN
    (`src.experiments.dataset_properties.nn_distances_from_D`/`nn_ratio`) -
    so D and eps_D are IDENTICAL to those used when the exp6_alpha_curves
    embeddings were fitted.
  - `src/sammon/identifiability.py`: all the pure numerical functions of
    Lemma 2/Veta 1 (identity + bound), Tvrzeni 3/4 (c_alpha, rho_NN), Veta 2
    (dual certificate), Veta 3 (local quadratic index I0) - see that
    module's docstring for the exact formulas and their provenance.
  - `results/data/[<mode>/]exp6_alpha_curves_results.csv`: the already
    completed embeddings (loaded via `src.common.checkpoint.load_embedding`)
    and the auc_rnx grid (median over seeds per (dataset, alpha), same
    aggregation as `exp6_alpha_curves.py::_derive_alpha_optimum`).
  - `results/data/[<mode>/]exp8_prop2_check_results.csv`: ONLY the columns
    `dataset, alpha, seed, tight_a, tight_b, bound_a, bound_b, R_near_Y,
    has_duplicates, p2_holds` (reserse section 10) - used for the E8 slack
    decomposition S1..S4 (alpha>0 rows only).
  - `results/data/dataset_properties.csv` (mode-specific path, K1) +
    `results/data/alpha_pred_rule.json` (NOT mode-specific, the single
    production rule - same convention already used by
    `src.methods.sammon_alpha_pred` in every mode): nn_ratio_k1 ->
    alpha_pred/stratum via `src.sammon.alpha_predict.predict_alpha` and
    `src.sammon.identifiability.stratum_from_rule`.

Row = (dataset, alpha) for alpha in the exp6_alpha_curves alpha grid
(INCLUDING alpha=0, where every gap collapses to 0 by construction - reserse
section 2.5). Unlike exp8_prop2_check.py (one row per (dataset, alpha,
seed)), a row here uses a single CANONICAL seed
(`exp6_alpha_curves.seeds[exp10_identifiability_check.embedding_seed_index]`,
default index 0) - the identity/bound/certificate/local-index are properties
of ONE pair of configurations (Y_0*, Y_alpha*) at a time (reserse section 10,
"Radek = (dataset, alpha)"), not something to average over seeds.

The Veta 2 certificate and the local index I0 need ALL alpha-grid embeddings
of a dataset (as witnesses) resp. only the alpha=0 embedding - both are
reloaded independently for every (dataset, alpha) task (same
reload-per-task convention as exp8_prop2_check.py, which also reloads
D/eps_D fresh for every task despite it being shared across a dataset's
alpha/seed grid - simplicity/parallelizability over avoiding redundant
compute; the Hessian-based I0 is the one genuinely expensive part, guarded
by `n_max_hessian`).

Output: results/data/[<mode>/]exp10_identifiability_check_results.csv
(checkpoint, resumable) + results/data/[<mode>/]exp10_identifiability_check_DONE.txt.

Run: venv\\python.exe -m src.experiments.exp10_identifiability_check [--quick|--full|--smoke]
or: src\\run_exp10_identifiability_check.bat [quick|full|smoke]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.common.checkpoint import RunKey, load_embedding, results_csv_path
from src.common.config import get_mode_path, load_config
from src.common.logging_utils import get_logger
from src.experiments.config_experiments import resolve_experiment_config
from src.experiments.exp_common import (
    filter_already_done,
    parse_mode_args,
    resolve_experiment_name,
    run_experiment_grid,
    write_done_file,
)

BASE_EXPERIMENT_NAME = "exp10_identifiability_check"
EXP6_BASE_NAME = "exp6_alpha_curves"
EXP8_BASE_NAME = "exp8_prop2_check"

# Canonical CSV schema. Order follows the bullet list of reserse section 10.
COLUMN_KEYS = [
    # identification
    "n_samples", "n_pairs", "p_dim", "eps_d", "has_duplicates",
    # statistics of D (independent of alpha, repeated on every row)
    "s_u", "skew_v", "kurt_v", "cv_D", "cv_D_eps", "rho_nn", "r_eps", "kappa_eps",
    "n_near", "n_mid", "n_far", "N_over_n_near", "alpha_dagger",
    # weight statistics
    "c_alpha", "c_alpha_first", "c_alpha_second",
    # Y_0 / Y_alpha configuration
    "sigma0_Y0", "sigma0_Ya", "sigmaA_Y0", "sigmaA_Ya",
    "Delta", "Delta_prime", "both_gaps_nonneg",
    "gamma_tilde_Y0", "gamma_tilde_Ya", "r_Y0", "r_Ya",
    "lambda_ratio_check", "identity_residual",
    "bound_v1", "bound_nontrivial", "tight_v1",
    # sandwich bound (Lemma 1), for comparison
    "bound_sandwich", "tight_sandwich",
    # local index I0 (Veta 3, alpha=0 configuration only)
    "I0_gn", "I0_exact", "lambda_min_pos_H0", "s_u_pi_sq", "hessian_skipped",
    "pred_quadratic", "quadratic_law_holds",
    # Veta 2 certificate
    "witness_alpha", "cert_lower", "alpha_eta", "cert_lower_direct", "witness_alpha_direct",
    # E8 slack decomposition (alpha>0 only)
    "S1", "S2", "S3", "S4", "check_S",
    # quality (E6 auc_rnx grid, alpha_pred rule)
    "auc_alpha", "auc_0", "G_auc_oracle", "auc_pred", "G_pred", "alpha_pred", "stratum",
    "alpha",
]


def _method_name_for_alpha(alpha: float) -> str:
    """Same 'method' naming convention as exp6_alpha_curves.py/exp8_prop2_check.py
    (the RunKey used to load an embedding MUST match exactly)."""
    return f"alpha{alpha}"


def _load_e6_median_auc(exp6_experiment_name: str) -> dict[str, dict[float, float]]:
    """{dataset: {alpha: median_auc_rnx over seeds}} - same aggregation as
    exp6_alpha_curves.py::_derive_alpha_optimum (groupby(dataset,alpha).median())."""
    path = results_csv_path(exp6_experiment_name)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing exp6_alpha_curves results: {path}\n"
            "Run first: venv\\python.exe -m src.experiments.exp6_alpha_curves --<mode> (K6)."
        )
    df = pd.read_csv(path, usecols=["dataset", "alpha", "auc_rnx", "status"])
    ok = df[(df["status"] == "ok") & df["auc_rnx"].notna()]
    if ok.empty:
        raise ValueError(f"{path} contains no successful rows with auc_rnx.")
    med = ok.groupby(["dataset", "alpha"])["auc_rnx"].median()
    out: dict[str, dict[float, float]] = {}
    for (dataset, alpha), value in med.items():
        out.setdefault(dataset, {})[float(alpha)] = float(value)
    return out


def _load_alpha_pred_info(mode: str) -> dict[str, dict[str, Any]]:
    """{dataset: {'alpha_pred': ..., 'stratum': ...}} from
    results/data/[<mode>/]dataset_properties.csv (nn_ratio_k1) and the
    single production rule results/data/alpha_pred_rule.json (NOT
    mode-specific - same file used by `src.methods.sammon_alpha_pred` in
    every mode)."""
    from src.sammon.alpha_predict import load_alpha_pred_rule, predict_alpha
    from src.sammon.identifiability import stratum_from_rule

    props_path = get_mode_path("results_data_dir", mode) / "dataset_properties.csv"
    if not props_path.exists():
        raise FileNotFoundError(
            f"Missing dataset_properties.csv: {props_path}\n"
            f"Run first: venv\\python.exe -m src.experiments.dataset_properties --{mode} (K1)."
        )
    props = pd.read_csv(props_path, usecols=["dataset", "kind", "nn_ratio_k1"])
    props = props[(props["kind"] == "vector") & props["nn_ratio_k1"].notna()]
    rule = load_alpha_pred_rule()

    out: dict[str, dict[str, Any]] = {}
    for _, row in props.iterrows():
        nn = float(row["nn_ratio_k1"])
        if nn <= 0:
            continue
        out[str(row["dataset"])] = {
            "alpha_pred": predict_alpha(nn, rule),
            "stratum": stratum_from_rule(nn, rule),
        }
    return out


_E8_USECOLS = ["dataset", "alpha", "seed", "tight_a", "tight_b", "bound_a", "bound_b", "R_near_Y", "has_duplicates", "p2_holds"]


def _load_e8_rows(mode: str, seed: int) -> dict[tuple[str, float], dict[str, Any]]:
    """{(dataset, alpha): row_dict} from exp8_prop2_check_results.csv,
    filtered to the SAME canonical seed as the embeddings used here (reads
    ONLY the columns listed in reserse section 10, `_E8_USECOLS`)."""
    e8_experiment_name = resolve_experiment_name(EXP8_BASE_NAME, mode)
    path = results_csv_path(e8_experiment_name)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing exp8_prop2_check results: {path}\n"
            f"Run first: venv\\python.exe -m src.experiments.exp8_prop2_check --{mode} (Q1 step 1)."
        )
    df = pd.read_csv(path, usecols=_E8_USECOLS)
    df = df[df["seed"] == seed]
    out: dict[tuple[str, float], dict[str, Any]] = {}
    for _, row in df.iterrows():
        out[(str(row["dataset"]), float(row["alpha"]))] = row.to_dict()
    return out


def _run_single(task: dict[str, Any]) -> dict[str, Any]:
    """One row: (dataset, alpha) -> loads D/eps_D/Y0/Ya/witnesses and
    evaluates every quantity of reserse/2026-09-17_zostreni_propozice2.md,
    section 10, via `src.sammon.identifiability`."""
    from src.datasets.registry import load_dataset
    from src.datasets.subsample import subsample_dataset
    from src.experiments.dataset_properties import nn_distances_from_D, nn_ratio
    from src.methods.common import to_distance_matrix
    from src.sammon.identifiability import (
        alpha_dagger as alpha_dagger_fn,
        alpha_eta_threshold,
        certificate_direct,
        certificate_lower_bound,
        coefficient_of_variation,
        local_identifiability_index,
        log_distance_moments,
        r_eps_kappa_eps,
        sandwich_bound,
        slack_decomposition,
        veta1_quantities,
        weight_cv_exact,
        weight_cv_first_order,
        weight_cv_second_order,
    )
    from src.sammon.prop2_check import block_masks, block_residual, pair_residuals, sigma_alpha_value
    from src.sammon.weights import estimate_eps_D

    dataset_name = task["dataset_name"]
    alpha = task["alpha"]
    seed = task["seed"]
    method_name = task["method_name"]

    result: dict[str, Any] = {"dataset_name": dataset_name, "method_name": method_name, "seed": seed}
    t0 = time.perf_counter()
    try:
        ds = load_dataset(dataset_name)
        ds = subsample_dataset(ds, n_max=task["n_max"], random_state=task["subsample_seed"])
        X = np.asarray(ds.X, dtype=np.float64)
        D = to_distance_matrix(X, "vector")
        eps_D = estimate_eps_D(D, k=task["eps_D_k"], q=task["eps_D_q"], kind="distance")

        n = D.shape[0]
        N_pairs = n * (n - 1) // 2
        iu = np.triu_indices(n, k=1)
        D_triu = D[iu]
        median_all = float(np.median(D_triu))
        theta = float(np.median(nn_distances_from_D(D, 1)))
        m = median_all
        rho_nn = nn_ratio(D, 1, median_all)

        exp6_name = task["exp6_experiment_name"]
        Y0 = load_embedding(RunKey(exp6_name, dataset_name, _method_name_for_alpha(0.0), seed))
        Ya = load_embedding(RunKey(exp6_name, dataset_name, method_name, seed))
        p_dim = Y0.shape[1]

        has_duplicates = bool(np.any(D_triu <= 0.0))

        moments = log_distance_moments(D_triu, eps_D)
        cv_D = coefficient_of_variation(D_triu)
        cv_D_eps = coefficient_of_variation(D_triu + eps_D)
        r_eps, kappa_eps = r_eps_kappa_eps(D_triu, theta, m, eps_D)
        near, mid, far = block_masks(D_triu, theta, m)
        n_near, n_mid, n_far = int(near.sum()), int(mid.sum()), int(far.sum())
        N_over_n_near = float(N_pairs / n_near) if n_near > 0 else float("nan")
        a_dagger = alpha_dagger_fn(n, r_eps)

        c_alpha = weight_cv_exact(D_triu, alpha, eps_D)
        c_alpha_first = weight_cv_first_order(alpha, moments["s_u"])
        c_alpha_second = weight_cv_second_order(alpha, moments["s_u"], moments["skew_v"])

        v1 = veta1_quantities(D, Y0, Ya, alpha, eps_D)
        bound_sandwich = sandwich_bound(D_triu, eps_D, alpha)
        tight_sandwich = (v1["Delta_prime"] / bound_sandwich) if bound_sandwich > 0.0 else float("nan")

        try:
            # `local_identifiability_index` fail-louds (ValueError) on
            # coincident points in Y0 (d_e=0 for some pair) - Veta 3
            # explicitly assumes a nondegenerate local minimizer without
            # coincidences (reserse section 4.1). This DOES happen on real
            # data (e.g. iris has exact duplicate rows, and a low-iteration
            # SMACOF - smoke/quick mode - can also converge to near/exact
            # coincidences for other datasets) - it is an expected
            # degeneracy of Y0, not a bug, so only the local-index columns
            # are downgraded to NaN/hessian_skipped=True; the rest of the
            # row (identity, bound, certificate, E8 slack, quality) does NOT
            # depend on Y0 being coincidence-free and stays valid.
            local = local_identifiability_index(
                D, Y0, eps_D, n_max_hessian=task["n_max_hessian"], pinv_rel_tol=task["hessian_pinv_rel_tol"],
            )
        except ValueError:
            local = {"I0_gn": float("nan"), "I0_exact": float("nan"), "lambda_min_pos_H0": float("nan"), "s_u_pi_sq": float("nan"), "hessian_skipped": True}
        pred_quadratic = float(alpha**2 * local["I0_exact"]) if not local["hessian_skipped"] else float("nan")
        if local["hessian_skipped"]:
            quadratic_law_holds = False
        else:
            quadratic_law_holds = bool(abs(v1["Delta"] - pred_quadratic) <= task["quadratic_law_tol"] * abs(v1["Delta"]))

        # --- Veta 2 certificate: witnesses = every embedding on the FULL exp6 alpha grid (same dataset/seed) ---
        rho_Y0 = pair_residuals(D, Y0)[1]
        R_near_Y0 = block_residual(rho_Y0, near)
        R_mid_Y0 = block_residual(rho_Y0, mid)

        witness_blocks: list[tuple[float, float, float, float]] = []
        witness_sigmas: list[tuple[float, float]] = []
        for a2 in task["witness_alpha_grid"]:
            Y2 = load_embedding(RunKey(exp6_name, dataset_name, _method_name_for_alpha(a2), seed))
            rho2 = pair_residuals(D, Y2)[1]
            witness_blocks.append((a2, block_residual(rho2, near), block_residual(rho2, mid), block_residual(rho2, far)))
            sigma2 = sigma_alpha_value(D, Y2, alpha, eps_D) if alpha > 0.0 else float(np.sum(rho2**2))
            witness_sigmas.append((a2, sigma2))

        cert = certificate_lower_bound(R_near_Y0, R_mid_Y0, alpha, r_eps, kappa_eps, witness_blocks)
        cert_direct = certificate_direct(v1["sigmaA_Y0"], witness_sigmas)
        eta_cfg = task["alpha_eta_bisection"]
        witness_block_for_eta = next((rn, rm, rf) for (a2, rn, rm, rf) in witness_blocks if a2 == cert["witness_alpha"])
        alpha_eta = alpha_eta_threshold(
            R_near_Y0, witness_block_for_eta, task["eta"], r_eps, kappa_eps,
            eta_cfg["alpha_max"], eta_cfg["xtol"], eta_cfg["near_zero_rel_tol"],
        )

        # --- E8 slack decomposition (alpha>0 only) ---
        if alpha > 0.0:
            e8_row = task["e8_rows"].get((dataset_name, alpha))
            if e8_row is not None:
                slack = slack_decomposition(
                    D, Ya, alpha, eps_D, theta, m,
                    bound_a=float(e8_row["bound_a"]), bound_b=float(e8_row["bound_b"]),
                    R_near_Y=float(e8_row["R_near_Y"]), tight_a=float(e8_row["tight_a"]),
                )
            else:
                slack = {"S1": float("nan"), "S2": float("nan"), "S3": float("nan"), "S4": float("nan"), "check_S": float("nan")}
        else:
            slack = {"S1": float("nan"), "S2": float("nan"), "S3": float("nan"), "S4": float("nan"), "check_S": float("nan")}

        # --- quality (E6 auc_rnx grid, alpha_pred rule) ---
        auc_by_alpha = task["e6_auc_for_dataset"]
        auc_alpha = auc_by_alpha.get(alpha, float("nan"))
        auc_0 = auc_by_alpha.get(0.0, float("nan"))
        G_auc_oracle = max(auc_by_alpha.values()) - auc_0 if auc_by_alpha else float("nan")
        pred_info = task["alpha_pred_info"]
        alpha_pred = pred_info["alpha_pred"] if pred_info is not None else float("nan")
        stratum = pred_info["stratum"] if pred_info is not None else ""
        auc_pred = float("nan")
        if pred_info is not None:
            for a2, auc2 in auc_by_alpha.items():
                if abs(a2 - alpha_pred) < 1e-9:
                    auc_pred = auc2
                    break
        G_pred = (auc_pred - auc_0) if np.isfinite(auc_pred) and np.isfinite(auc_0) else float("nan")

        metrics: dict[str, Any] = {
            "n_samples": n, "n_pairs": N_pairs, "p_dim": p_dim, "eps_d": eps_D, "has_duplicates": has_duplicates,
            "s_u": moments["s_u"], "skew_v": moments["skew_v"], "kurt_v": moments["kurt_v"],
            "cv_D": cv_D, "cv_D_eps": cv_D_eps, "rho_nn": rho_nn, "r_eps": r_eps, "kappa_eps": kappa_eps,
            "n_near": n_near, "n_mid": n_mid, "n_far": n_far, "N_over_n_near": N_over_n_near, "alpha_dagger": a_dagger,
            "c_alpha": c_alpha, "c_alpha_first": c_alpha_first, "c_alpha_second": c_alpha_second,
            "sigma0_Y0": v1["sigma0_Y0"], "sigma0_Ya": v1["sigma0_Ya"], "sigmaA_Y0": v1["sigmaA_Y0"], "sigmaA_Ya": v1["sigmaA_Ya"],
            "Delta": v1["Delta"], "Delta_prime": v1["Delta_prime"], "both_gaps_nonneg": v1["both_gaps_nonneg"],
            "gamma_tilde_Y0": v1["gamma_tilde_Y0"], "gamma_tilde_Ya": v1["gamma_tilde_Ya"], "r_Y0": v1["r_Y0"], "r_Ya": v1["r_Ya"],
            "lambda_ratio_check": v1["lambda_ratio_check"], "identity_residual": v1["identity_residual"],
            "bound_v1": v1["bound_v1"], "bound_nontrivial": v1["bound_nontrivial"], "tight_v1": v1["tight_v1"],
            "bound_sandwich": bound_sandwich, "tight_sandwich": tight_sandwich,
            "I0_gn": local["I0_gn"], "I0_exact": local["I0_exact"], "lambda_min_pos_H0": local["lambda_min_pos_H0"],
            "s_u_pi_sq": local["s_u_pi_sq"], "hessian_skipped": local["hessian_skipped"],
            "pred_quadratic": pred_quadratic, "quadratic_law_holds": quadratic_law_holds,
            "witness_alpha": cert["witness_alpha"], "cert_lower": cert["cert_lower"], "alpha_eta": alpha_eta,
            "cert_lower_direct": cert_direct["cert_lower_direct"], "witness_alpha_direct": cert_direct["witness_alpha_direct"],
            "S1": slack["S1"], "S2": slack["S2"], "S3": slack["S3"], "S4": slack["S4"], "check_S": slack["check_S"],
            "auc_alpha": auc_alpha, "auc_0": auc_0, "G_auc_oracle": G_auc_oracle,
            "auc_pred": auc_pred, "G_pred": G_pred, "alpha_pred": alpha_pred, "stratum": stratum,
        }

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
    result["embedding"] = None  # exp10 does not generate any new embedding, only analyzes existing ones
    return result


def main() -> None:
    mode = parse_mode_args(
        "Identifiability check of the stress exponent (Lemma 2/Veta 1/2/3, "
        "reserse/2026-09-17_zostreni_propozice2.md section 10) over completed exp6_alpha_curves embeddings."
    )
    logger = get_logger(BASE_EXPERIMENT_NAME, mode=mode)

    exp6_cfg = resolve_experiment_config(EXP6_BASE_NAME, mode)
    exp10_cfg = resolve_experiment_config(BASE_EXPERIMENT_NAME, mode)
    sammon_cfg = load_config()["sammon"]

    from src.experiments.exp1_dr_benchmark import SUBSAMPLE_SEED
    from src.experiments.exp8_prop2_check import _resolve_eps_D_kq

    eps_D_k, eps_D_q = _resolve_eps_D_kq(exp6_cfg, sammon_cfg)
    exp6_experiment_name = resolve_experiment_name(EXP6_BASE_NAME, mode)

    seeds_list = list(exp6_cfg["seeds"])
    seed_index = int(exp10_cfg["embedding_seed_index"])
    if not (0 <= seed_index < len(seeds_list)):
        raise ValueError(f"exp10_identifiability_check.embedding_seed_index={seed_index} out of range for exp6_alpha_curves.seeds={seeds_list} (mode={mode}).")
    canonical_seed = int(seeds_list[seed_index])

    witness_alpha_grid = sorted(float(a) for a in exp6_cfg["alpha_grid"])
    if 0.0 not in witness_alpha_grid:
        raise ValueError(f"exp6_alpha_curves.alpha_grid (mode={mode}) does not contain alpha=0.0 - missing reference Y_0.")

    all_datasets = list(exp6_cfg["datasets"])
    if mode == "smoke":
        n_datasets = int(exp10_cfg["n_datasets"])
        row_datasets = all_datasets[:n_datasets]
        alpha_values = {float(a) for a in exp10_cfg["alpha_values"]}
        row_alpha_grid = [a for a in witness_alpha_grid if a in alpha_values]
        if not row_alpha_grid:
            raise ValueError(f"exp10_identifiability_check.smoke.alpha_values={sorted(alpha_values)} does not intersect the exp6 alpha grid {witness_alpha_grid}.")
    else:
        row_datasets = all_datasets
        row_alpha_grid = witness_alpha_grid

    n_max_hessian = int(exp10_cfg["n_max_hessian"])
    hessian_pinv_rel_tol = float(exp10_cfg["hessian_pinv_rel_tol"])
    eta = float(exp10_cfg["eta"])
    quadratic_law_tol = float(exp10_cfg["quadratic_law_tol"])
    alpha_eta_bisection = {
        "alpha_max": float(exp10_cfg["alpha_eta_bisection"]["alpha_max"]),
        "xtol": float(exp10_cfg["alpha_eta_bisection"]["xtol"]),
        "near_zero_rel_tol": float(exp10_cfg["alpha_eta_bisection"]["near_zero_rel_tol"]),
    }

    logger.info("%s (mode=%s): %d datasets, alpha grid %s, canonical seed=%d (witness grid %s).",
                BASE_EXPERIMENT_NAME, mode, len(row_datasets), row_alpha_grid, canonical_seed, witness_alpha_grid)

    e6_auc = _load_e6_median_auc(exp6_experiment_name)
    alpha_pred_info_all = _load_alpha_pred_info(mode)
    e8_rows = _load_e8_rows(mode, canonical_seed)

    EXPERIMENT_NAME = resolve_experiment_name(BASE_EXPERIMENT_NAME, mode)

    tasks: list[dict[str, Any]] = []
    for dataset_name in row_datasets:
        e6_auc_for_dataset = e6_auc.get(dataset_name, {})
        alpha_pred_info = alpha_pred_info_all.get(dataset_name)
        for alpha in row_alpha_grid:
            method_name = _method_name_for_alpha(alpha)
            tasks.append({
                "dataset_name": dataset_name, "method_name": method_name, "seed": canonical_seed, "alpha": alpha,
                "n_max": int(exp6_cfg["n_max"]), "subsample_seed": SUBSAMPLE_SEED,
                "eps_D_k": eps_D_k, "eps_D_q": eps_D_q, "exp6_experiment_name": exp6_experiment_name,
                "witness_alpha_grid": witness_alpha_grid,
                "n_max_hessian": n_max_hessian, "hessian_pinv_rel_tol": hessian_pinv_rel_tol,
                "eta": eta, "quadratic_law_tol": quadratic_law_tol, "alpha_eta_bisection": alpha_eta_bisection,
                "e6_auc_for_dataset": e6_auc_for_dataset, "alpha_pred_info": alpha_pred_info, "e8_rows": e8_rows,
            })

    todo, n_done = filter_already_done(EXPERIMENT_NAME, tasks)
    logger.info(
        "%s (mode=%s): %d combinations already done (skipped), %d new to compute (embedding source: %s).",
        EXPERIMENT_NAME, mode, n_done, len(todo), exp6_experiment_name,
    )

    n_ok, n_err = run_experiment_grid(EXPERIMENT_NAME, todo, _run_single, COLUMN_KEYS, logger)
    logger.info("Done. Newly successful: %d, newly failed: %d, skipped: %d.", n_ok, n_err, n_done)
    write_done_file(EXPERIMENT_NAME, logger, extra_info={"mode": mode, "exp6_source": exp6_experiment_name, "canonical_seed": canonical_seed})


if __name__ == "__main__":
    main()
