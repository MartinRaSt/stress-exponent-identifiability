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
Pure numerical functions for the empirical test of Proposition 2 (local
residual and rho_NN), clanek/sections/03_metoda.tex, lines ~165-194
(Notation, Assumptions, Proposition 2, proof). No IO or dataset loading -
that is done by `src/experiments/exp8_prop2_check.py` (re-uses
`load_dataset`, `subsample_dataset`, `to_distance_matrix`, `estimate_eps_D`,
same as `exp6_alpha_curves.py`, so that D and eps_D are identical to the
embeddings being verified).

Notation (matches the article):
  D_ij        - input distance matrix (n x n), i<j
  Y, Ybar     - two configurations (Y = run at a given alpha, Ybar =
                reference configuration; in exp8 always the alpha=0
                embedding for the same (dataset, seed))
  rho_ij(Y)   = D_ij - d_ij(Y)
  w_ij        = (D_ij + eps_D)^{-alpha}                  (`alpha_weights`, weights.py - REUSED, not redefined)
  sigma_alpha(Y) = sum_{i<j} w_ij rho_ij(Y)^2
  theta, m    - see eq:nn_ratio (03_metoda.tex): median distance to the 1st
                neighbor, median of all pairwise distances (rho_NN = theta/m)
  r_eps       = (theta+eps_D)/(m+eps_D)
  kappa_eps   = (theta+eps_D)/(D_min+eps_D)  - FINITE for every eps_D > 0,
                even if D_min = 0 (duplicate points) - that is the whole
                point of the eps_D regularization (03_metoda.tex,
                remark[role eps_D]); fail-loud triggers only when
                eps_D <= eps_d_zero_tol (config) AND D_min = 0 hold
                SIMULTANEOUSLY (see `evaluate_proposition2`; 2026-09-14 fix
                of an overly strict check, see
                documentation/2026-09-14_exp8_prop2_check.md)
  phi_alpha(D)= ((theta+eps_D)/(D+eps_D))^alpha
  P_near/P_mid/P_far - see `block_masks` (a pair with D_ij=0 always belongs
                to P_near, since 0 <= theta)
  R_S(Y)      = sum_{(i,j) in S} rho_ij(Y)^2
  n_zero_pairs/has_duplicates - number of off-diagonal pairs with D_ij=0,
                resp. whether any exist (diagnostic CSV columns, not part
                of Proposition 2 itself)

Assumption (P2) tested empirically (NOT assumed): sigma_alpha(Y) <=
sigma_alpha(Ybar) - `evaluate_proposition2` returns `p2_holds` (bool); rows
where it fails to hold are NOT silently dropped (see exp8_prop2_check.py,
column p2_holds).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform

from src.sammon.weights import alpha_weights


def _validate_D(D: np.ndarray) -> np.ndarray:
    D = np.asarray(D, dtype=np.float64)
    n = D.shape[0]
    if D.ndim != 2 or D.shape != (n, n):
        raise ValueError(f"D must be a square matrix, got shape {D.shape}.")
    return D


def pair_residuals(D: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (D_triu, rho) for all i<j: D_ij and rho_ij(Y) = D_ij - d_ij(Y).

    `Y` is a configuration (n x p); d_ij(Y) is the Euclidean row distance."""
    D = _validate_D(D)
    n = D.shape[0]
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim != 2 or Y.shape[0] != n:
        raise ValueError(f"Y must have shape (n, p) with the same n as D ({n}), got {Y.shape}.")
    iu = np.triu_indices(n, k=1)
    dY = squareform(pdist(Y, metric="euclidean"))
    D_triu = D[iu]
    rho = D_triu - dY[iu]
    return D_triu, rho


def sigma_alpha_value(D: np.ndarray, Y: np.ndarray, alpha: float, eps_D: float) -> float:
    """sigma_alpha(Y) = sum_{i<j} w_ij rho_ij(Y)^2, w_ij from `alpha_weights`
    (src/sammon/weights.py - same weight definition as when fitting
    alpha-Sammon, REUSED, not redefined)."""
    D = _validate_D(D)
    n = D.shape[0]
    iu = np.triu_indices(n, k=1)
    W = alpha_weights(D, alpha, eps_D)
    _, rho = pair_residuals(D, Y)
    return float(np.sum(W[iu] * rho**2))


def block_masks(D_triu: np.ndarray, theta: float, m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """P_near = {D_ij <= theta}, P_far = {D_ij >= m}, P_mid = the rest
    (03_metoda.tex, Notation before Proposition 2). Requires theta <= m
    (follows from theta/m being the median distance to the 1st neighbor,
    resp. the median of all distances - fail-loud if the input violates
    this assumption)."""
    if theta > m:
        raise ValueError(f"theta={theta} > m={m}: the definition of P_near/P_far/P_mid assumes theta <= m (see eq:nn_ratio).")
    near = D_triu <= theta
    far = D_triu >= m
    mid = ~near & ~far
    return near, mid, far


def block_residual(rho: np.ndarray, mask: np.ndarray) -> float:
    """R_S(Y) = sum_{(i,j) in S} rho_ij(Y)^2."""
    return float(np.sum(rho[mask] ** 2))


def phi_alpha(D_triu: np.ndarray, theta: float, eps_D: float, alpha: float) -> np.ndarray:
    """phi_alpha(D) = ((theta+eps_D)/(D+eps_D))^alpha (03_metoda.tex, Notation)."""
    return np.power((theta + eps_D) / (D_triu + eps_D), alpha)


def evaluate_proposition2(
    D: np.ndarray, Y: np.ndarray, Ybar: np.ndarray, alpha: float, eps_D: float, theta: float, m: float,
    eps_d_zero_tol: float,
) -> dict[str, float | bool | int]:
    """Compute all quantities of Proposition 2 (03_metoda.tex,
    eq:prop2a/eq:prop2b) for a single pair of configurations (Y at the
    given alpha, Ybar the reference). Returns a dict used directly as a CSV
    row in `exp8_prop2_check.py` (extended there with
    rho_nn/eps_d/theta/m/n_pairs_total, which are not part of this pure
    computation).

    `eps_d_zero_tol` (from config.yaml `exp8_prop2_check.eps_d_zero_tol`) is
    the numerical tolerance for "eps_D is practically zero". By the
    definition in 03_metoda.tex (Notation, `remark[role eps_D]`),
    `kappa_eps = (theta+eps_D)/(D_min+eps_D)` is FINITE and well-defined for
    EVERY `eps_D > 0`, even if `D_min = 0` (near-duplicate/duplicate points)
    - that is precisely the point of the `eps_D` regularization. Fail-loud
    therefore triggers ONLY when the regularization is effectively absent
    (`eps_D <= eps_d_zero_tol`) AND zero off-diagonal distances exist
    (`D_min <= 0`) at the same time - in that case
    `kappa_eps = theta/D_min` would be undefined/unstable. For
    `eps_D > eps_d_zero_tol` the row is computed normally regardless of
    D_min (2026-09-14: the original check `d_min <= 0 -> ValueError` was too
    strict - it dropped whole datasets with exact duplicates, see
    documentation/2026-09-14_exp8_prop2_check.md)."""
    if alpha <= 0:
        raise ValueError(f"evaluate_proposition2 is defined for alpha > 0 (Ybar=alpha=0 is the reference), got alpha={alpha}.")
    if eps_D < 0:
        raise ValueError(f"eps_D must be >= 0, got {eps_D}.")
    if eps_d_zero_tol < 0:
        raise ValueError(f"eps_d_zero_tol must be >= 0, got {eps_d_zero_tol}.")

    D = _validate_D(D)
    n = D.shape[0]
    D_triu, rho_Y = pair_residuals(D, Y)
    _, rho_Ybar = pair_residuals(D, Ybar)
    d_min = float(D_triu.min())
    n_zero_pairs = int(np.sum(D_triu <= 0.0))
    has_duplicates = n_zero_pairs > 0
    if has_duplicates and eps_D <= eps_d_zero_tol:
        raise ValueError(
            f"D contains {n_zero_pairs} zero off-diagonal distances (duplicate points) and "
            f"eps_D={eps_D} is below the numerical tolerance eps_d_zero_tol={eps_d_zero_tol} (no "
            "effective regularization) - kappa_eps = theta/D_min would be undefined/unstable. For "
            "eps_D > eps_d_zero_tol, kappa_eps = (theta+eps_D)/(D_min+eps_D) is finite even at "
            "D_min=0 (see 03_metoda.tex, remark[role eps_D]) - check why eps_D came out so small "
            "for this data (estimate_eps_D)."
        )

    near, mid, far = block_masks(D_triu, theta, m)

    sigma_Y = sigma_alpha_value(D, Y, alpha, eps_D)
    sigma_Ybar = sigma_alpha_value(D, Ybar, alpha, eps_D)
    p2_holds = bool(sigma_Y <= sigma_Ybar)

    R_near_Y = block_residual(rho_Y, near)
    R_mid_Y = block_residual(rho_Y, mid)
    R_far_Y = block_residual(rho_Y, far)
    R_near_Ybar = block_residual(rho_Ybar, near)
    R_mid_Ybar = block_residual(rho_Ybar, mid)
    R_far_Ybar = block_residual(rho_Ybar, far)

    r_eps = (theta + eps_D) / (m + eps_D)
    kappa_eps = (theta + eps_D) / (d_min + eps_D)
    phi = phi_alpha(D_triu, theta, eps_D, alpha)

    bound_a = float(np.sum(phi * rho_Ybar**2))
    bound_b = (kappa_eps**alpha) * R_near_Ybar + R_mid_Ybar + (r_eps**alpha) * R_far_Ybar

    r_near_ratio = (R_near_Y / R_near_Ybar) if R_near_Ybar > 0 else float("nan")
    predicted_factor = r_eps**alpha

    return {
        "n_samples": n,
        "d_min": d_min,
        "n_zero_pairs": n_zero_pairs,
        "has_duplicates": has_duplicates,
        "r_eps": r_eps,
        "kappa_eps": kappa_eps,
        "sigma_alpha_Y": sigma_Y,
        "sigma_alpha_Ybar": sigma_Ybar,
        "p2_holds": p2_holds,
        "p2_slack": sigma_Ybar - sigma_Y,
        "R_near_Y": R_near_Y, "R_mid_Y": R_mid_Y, "R_far_Y": R_far_Y,
        "R_near_Ybar": R_near_Ybar, "R_mid_Ybar": R_mid_Ybar, "R_far_Ybar": R_far_Ybar,
        "bound_a": bound_a,
        "bound_b": bound_b,
        "slack_a": bound_a - R_near_Y,
        "slack_b": bound_b - R_near_Y,
        "tight_a": (R_near_Y / bound_a) if bound_a > 0 else float("nan"),
        "tight_b": (R_near_Y / bound_b) if bound_b > 0 else float("nan"),
        "n_pairs_near": int(near.sum()), "n_pairs_mid": int(mid.sum()), "n_pairs_far": int(far.sum()),
        "r_near_ratio": r_near_ratio,
        "predicted_factor_r_eps_alpha": predicted_factor,
    }


def resolve_p2_holds(sigma_Y: float, sigma_Ybar: float, tol_rel: float, tol_abs: float) -> bool:
    """Decide whether assumption (P2) sigma_alpha(Y) <= sigma_alpha(Ybar) holds
    WITH TOLERANCE (config_experiments.yaml exp8_prop2_check.p2_tolerance_rel/abs):
    SMACOF only converges to `exp6_alpha_curves.tol`, not to the exact global
    minimum, so a small numerical overshoot must not lead to a false
    rejection of P2. `tol_abs` additionally handles the case sigma_Ybar~0."""
    if tol_rel < 0 or tol_abs < 0:
        raise ValueError(f"p2_tolerance_rel/abs must be >= 0, got rel={tol_rel}, abs={tol_abs}.")
    threshold = sigma_Ybar + tol_abs + tol_rel * abs(sigma_Ybar)
    return bool(sigma_Y <= threshold)


def compute_alpha_bound(r_eps: float, tau: float, alpha_grid: list[float]) -> float:
    """Smallest alpha from `alpha_grid` (alpha_grid may contain alpha=0,
    which is ignored) for which r_eps^alpha <= tau - the theoretical
    prediction of the threshold beyond which the discounted contribution of
    P_far to the local residual drops below `tau` (interpretation following
    Proposition 2, 03_metoda.tex; see config_experiments.yaml
    fig_prop2_alpha_prediction/exp8_prop2_check).
    r_eps depends only on (dataset, seed) - theta/m/eps_D are fixed across
    the whole alpha grid for a given (dataset, seed), so it suffices to
    compute it once.
    Returns NaN if no alpha in the grid satisfies the condition (r_eps too
    close to 1, diffusive regime - NOT extrapolated beyond the grid, see
    Corollary 1)."""
    if not (0.0 < tau < 1.0):
        raise ValueError(f"tau must be in (0,1), got {tau}.")
    if not (0.0 < r_eps <= 1.0):
        raise ValueError(f"r_eps must be in (0,1], got {r_eps}.")
    grid = sorted(float(a) for a in alpha_grid if a > 0)
    if not grid:
        raise ValueError("alpha_grid does not contain any alpha > 0.")
    for alpha in grid:
        if r_eps**alpha <= tau:
            return float(alpha)
    return float("nan")
