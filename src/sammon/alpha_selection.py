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
Automatic alpha selection: grid search + an independent validation metric,
R_NX AUC (section 5.1), and a multi-scale mixture as a robust alternative
(section 5.2).

E_alpha itself is NOT comparable across different alpha values (each alpha
defines a different weight normalization, section 5.1) - hence an
independent metric, `src.sammon.metrics.evaluate` (default 'auc_rnx'), is
used for selection.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.sammon.init import init_classical_mds, init_random
from src.sammon.metrics import evaluate
from src.sammon.solvers.smacof import smacof_solve
from src.sammon.weights import alpha_weights, compute_Z_from_W, estimate_eps_D, multiscale_weights


def _subsample_D(D: np.ndarray, n_val: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Random subsample of the distance matrix down to at most n_val points (fixed seed)."""
    n = D.shape[0]
    if n <= n_val:
        return D, np.arange(n)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=n_val, replace=False))
    return D[np.ix_(idx, idx)], idx


def _fit_one_alpha(
    D: np.ndarray, alpha: float, n_components: int, max_iter: int, tol: float,
    eps_num: float, eps_D_k: int, eps_D_q: float, dense_pinv_threshold: int,
    cg_max_iter: int, cg_tol: float, reg_rho: float, seed: int,
) -> tuple[np.ndarray, dict]:
    """Fit Y for a given alpha using SMACOF on a (subsample of the) distance matrix D."""
    eps_D = estimate_eps_D(D, k=eps_D_k, q=eps_D_q, kind="distance")
    W = alpha_weights(D, alpha, eps_D)
    Z = compute_Z_from_W(D, W)
    Y0 = init_classical_mds(D, n_components, seed)
    Y, history = smacof_solve(
        D, W, Z, Y0, max_iter=max_iter, tol=tol, eps_num=eps_num,
        dense_pinv_threshold=dense_pinv_threshold, cg_max_iter=cg_max_iter,
        cg_tol=cg_tol, reg_rho=reg_rho, verbose=False,
    )
    return Y, history


def select_alpha(
    D_or_X: np.ndarray,
    grid: list[float],
    n_val: int,
    seed: int,
    kind: str = "distance",
    metric: str = "auc_rnx",
    n_components: int = 2,
    max_iter: int = 300,
    tol: float = 1.0e-4,
    eps_num: float = 1.0e-9,
    eps_D_k: int = 5,
    eps_D_q: float = 0.1,
    dense_pinv_threshold: int = 5000,
    cg_max_iter: int = 500,
    cg_tol: float = 1.0e-6,
    reg_rho: float = 1.0e-8,
) -> tuple[float, pd.DataFrame]:
    """Grid search over `grid` alpha values; for each one, fits SMACOF on a
    validation subset (n_val points, fixed seed) and evaluates an
    independent metric (default R_NX AUC, section 5.1). Returns
    (best_alpha, table).

    Returns a table with columns: alpha, <metric>, stress_scale_invariant
    (alpha=0 as a shared yardstick), auc_rnx, wall_time_sec.
    """
    from src.methods.common import to_distance_matrix

    D_full = to_distance_matrix(D_or_X, kind)
    D_val, idx = _subsample_D(D_full, n_val, seed)

    rows: list[dict] = []
    for alpha in grid:
        t0 = time.perf_counter()
        Y, history = _fit_one_alpha(
            D_val, alpha, n_components, max_iter, tol, eps_num, eps_D_k, eps_D_q,
            dense_pinv_threshold, cg_max_iter, cg_tol, reg_rho, seed,
        )
        metrics = evaluate(D_val, Y, None, "distance")
        # shared yardstick E_0^scale-inv on the same Y (independent of the alpha used for training)
        W0 = alpha_weights(D_val, 0.0, estimate_eps_D(D_val, eps_D_k, eps_D_q, "distance"))
        Z0 = compute_Z_from_W(D_val, W0)
        from src.sammon.stress import scale_invariant_stress
        e0 = scale_invariant_stress(D_val, Y, W0, Z0, eps_num)
        rows.append({
            "alpha": alpha, "auc_rnx": metrics.get("auc_rnx", np.nan),
            "stress_scale_invariant_own_alpha": metrics["stress_scale_invariant"],
            "stress_scale_invariant_alpha0": e0,
            "n_smacof_iter": history["n_iter"], "wall_time_sec": time.perf_counter() - t0,
        })

    table = pd.DataFrame(rows)
    if metric not in table.columns:
        raise KeyError(f"Metric '{metric}' is not in the grid-search results table (available: {list(table.columns)}).")
    best_idx = table[metric].astype(float).idxmax()
    best_alpha = float(table.loc[best_idx, "alpha"])
    return best_alpha, table


def multiscale(
    D_or_X: np.ndarray,
    alphas: list[float],
    seed: int,
    kind: str = "distance",
    n_components: int = 2,
    max_iter: int = 300,
    tol: float = 1.0e-4,
    eps_num: float = 1.0e-9,
    eps_D_k: int = 5,
    eps_D_q: float = 0.1,
    C: float = 1.0,
    dense_pinv_threshold: int = 5000,
    cg_max_iter: int = 500,
    cg_tol: float = 1.0e-6,
    reg_rho: float = 1.0e-8,
) -> tuple[np.ndarray, dict]:
    """Multi-scale weight mixture (section 5.2): trains a single embedding on
    W = sum_k c_k D^{-alpha_k}, a robust alternative to a single chosen alpha."""
    from src.methods.common import to_distance_matrix

    D = to_distance_matrix(D_or_X, kind)
    eps_D = estimate_eps_D(D, k=eps_D_k, q=eps_D_q, kind="distance")
    W, c_by_alpha = multiscale_weights(D, alphas, eps_D, C=C)

    n = D.shape[0]
    mask = ~np.eye(n, dtype=bool)
    Z = float((W[mask] * D[mask] ** 2).sum() / 2.0)
    if Z <= 0:
        raise ValueError("Z <= 0 for multiscale W - invalid data.")

    Y0 = init_classical_mds(D, n_components, seed)
    Y, history = smacof_solve(
        D, W, Z, Y0, max_iter=max_iter, tol=tol, eps_num=eps_num,
        dense_pinv_threshold=dense_pinv_threshold, cg_max_iter=cg_max_iter,
        cg_tol=cg_tol, reg_rho=reg_rho, verbose=False,
    )
    history["c_by_alpha"] = c_by_alpha
    history["eps_D"] = eps_D
    return Y, history
