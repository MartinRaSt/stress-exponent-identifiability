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
Pure numerical functions for the identifiability check of the stress
exponent (reserse/2026-09-17_zostreni_propozice2.md, section 10 - the
specification for python-coder; referred to there as "E9" / experiment
"exp9_identifiability_check", renamed here to E10/exp10_identifiability_check
to avoid a name collision with the already-existing
`src/experiments/exp9_metric_fidelity.py`).

No IO or dataset loading here - that is done by
`src/experiments/exp10_identifiability_check.py`, which reuses the loader of
`src/experiments/exp8_prop2_check.py` (load_dataset, subsample_dataset,
to_distance_matrix, estimate_eps_D, theta/m/rho_NN via
`src.experiments.dataset_properties`) so that D, eps_D and the weighted
stress are computed EXACTLY the same way as the embeddings being analyzed.

Notation follows reserse/2026-09-17_zostreni_propozice2.md sections 1-6:
  D_e = D_ij, i<j (N = n(n-1)/2 pairs)
  u_e = log(D_e + eps_D), v_e = u_e - ubar (centered log-distance)
  w_e = (D_e + eps_D)^{-alpha}  (alpha=0 -> w_e = 1)
  rho_e(Y) = D_e - d_e(Y)       (residual of configuration Y)
  sigma_alpha(Y) = sum_e w_e rho_e(Y)^2   (RAW weighted stress, not
                    normalized by Z_alpha - see prop2_check.sigma_alpha_value,
                    reused here, not redefined)
  pi_e(Y) = rho_e(Y)^2 / sigma_0(Y)       (residual profile, a probability
                    vector over pairs)
  gamma~(Y) = CV_e(rho_e(Y)^2)            (coefficient of variation of the
                    squared residuals - "how concentrated is the residual
                    mass on a few pairs")
  c_alpha = CV_e(w_e)                     (coefficient of variation of the
                    weights - "how concentrated is the distance
                    distribution")
  r_alpha(Y) = corr_e(w_e, rho_e(Y)^2)    (Pearson correlation of weights
                    and squared residuals over the N pairs)

Lemma 2 (identity):  Lambda_alpha(Y)/wbar = 1 + r_alpha(Y) c_alpha gamma~(Y)
Veta 1 (bound on the identifiability gaps Delta, Delta'):
  (1+Delta_alpha)(1+Delta'_alpha) = (1 + r_0 c gamma~_0)/(1 + r_a c gamma~_alpha)   [[exact identity]]
  0 <= Delta_alpha, Delta'_alpha <= (1+c gamma~_0)/(1-c gamma~_alpha) - 1   [[bound, valid iff c*gamma~_alpha < 1]]
Veta 3 (local quadratic law): Delta_alpha ~ alpha^2 I_0 + O(alpha^3),
  I_0 = Q/(2 sigma_0(Y_0*)), Q = g^T H_0^+ g, g = 2 J^T(rho o v) - the
  pseudo-inverse H_0^+ MUST cut exactly the p(p+1)/2 zero directions of
  rigid motions (translations + rotations of the whole configuration),
  otherwise Q silently diverges/blows up - this is the single most likely
  place for a silent bug (see the docstring of `local_identifiability_index`
  and the "three points on a line" unit test in
  tests/test_exp10_identifiability.py, which is the main safeguard against
  it).
Veta 2 (dual/certificate): a lower bound on Delta'_alpha via a witness
  configuration Y~ from the alpha grid, see `certificate_lower_bound`.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from src.sammon.prop2_check import block_masks, block_residual, pair_residuals, sigma_alpha_value

# --- generic statistics -----------------------------------------------------


def coefficient_of_variation(x: np.ndarray) -> float:
    """CV(x) = sd(x)/mean(x) (population sd, divide by N - matches the
    definition s_u^2 = (1/N) sum v_e^2 in the article, NOT the N-1 sample
    variance). Returns NaN for mean(x) <= 0 (e.g. gamma~ of a configuration
    with sigma_0 = 0 is not defined - see reserse section 2.5, "sigma_0(Y)=0")."""
    x = np.asarray(x, dtype=np.float64)
    mean_x = float(np.mean(x))
    if mean_x <= 0:
        return float("nan")
    var_x = float(np.mean((x - mean_x) ** 2))
    return float(np.sqrt(var_x) / mean_x)


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation over pairs. Returns 0.0 (not NaN) when either
    array is constant (sd=0): in that degenerate case the correlation is
    mathematically undefined, but in every use in this module it multiplies
    a factor that is itself 0 (e.g. r_alpha(Y_0*)*c*gamma~(Y_0*) with
    gamma~(Y_0*)=0), so returning 0 keeps the product correct without
    fabricating a spurious correlation value."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    sx = float(x.std())
    sy = float(y.std())
    if sx <= 0.0 or sy <= 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def log_distance_moments(D_triu: np.ndarray, eps_D: float) -> dict[str, float]:
    """u_e = log(D_e + eps_D), v_e = u_e - ubar (Tvrzeni 3/4, reserse section 3).

    Returns s_u (population sd of v), skew_v (population skewness of v),
    kurt_v (population EXCESS kurtosis of v, i.e. kurtosis - 3), ubar.
    Degenerate case s_u=0 (all distances equal, e.g. a perfectly regular
    configuration) returns skew_v=kurt_v=0.0 rather than 0/0 = NaN."""
    if eps_D < 0:
        raise ValueError(f"eps_D must be >= 0, got {eps_D}.")
    D_triu = np.asarray(D_triu, dtype=np.float64)
    if np.any(D_triu + eps_D <= 0):
        raise ValueError("D_e + eps_D <= 0 for at least one pair - log(D_e+eps_D) is undefined.")
    u = np.log(D_triu + eps_D)
    ubar = float(u.mean())
    v = u - ubar
    m2 = float(np.mean(v**2))
    s_u = float(np.sqrt(m2))
    if s_u <= 0.0:
        return {"s_u": 0.0, "skew_v": 0.0, "kurt_v": 0.0, "ubar": ubar}
    m3 = float(np.mean(v**3))
    m4 = float(np.mean(v**4))
    skew_v = m3 / s_u**3
    kurt_v = m4 / s_u**4 - 3.0
    return {"s_u": s_u, "skew_v": skew_v, "kurt_v": kurt_v, "ubar": ubar}


# --- weight statistics (Tvrzeni 3, c_alpha) ---------------------------------


def weights_from_Dtriu(D_triu: np.ndarray, alpha: float, eps_D: float) -> np.ndarray:
    """w_e = (D_e + eps_D)^{-alpha} (alpha=0 -> w_e=1 regardless of eps_D,
    same convention as `src.sammon.weights.alpha_weights`, reused there for
    the full DxD matrix - this is the pair-vector equivalent, avoiding
    building a full (n x n) matrix just to read off the upper triangle).

    Unlike `src.sammon.weights.alpha_weights`, `eps_D=0` is ALLOWED here for
    alpha>0 as long as every D_e>0 (reserse section 2.5, "eps_D=0, D_min>0:
    u=log D, vse konecne. OK." - the closed-form three-point family is
    exactly this case). Fail-loud triggers only when the base D_e+eps_D
    would be non-positive (duplicate points with no regularization)."""
    D_triu = np.asarray(D_triu, dtype=np.float64)
    if alpha < 0:
        raise ValueError(f"alpha must be >= 0, got {alpha}.")
    if eps_D < 0:
        raise ValueError(f"eps_D must be >= 0, got {eps_D}.")
    if alpha == 0.0:
        return np.ones_like(D_triu)
    base = D_triu + eps_D
    if np.any(base <= 0.0):
        raise ValueError(
            f"D_e + eps_D <= 0 for at least one pair (alpha={alpha} > 0 needs a strictly positive base) - "
            "likely duplicate points (D_e=0) combined with eps_D=0."
        )
    return np.power(base, -alpha)


def weight_cv_exact(D_triu: np.ndarray, alpha: float, eps_D: float) -> float:
    """c_alpha = sqrt(mean(w^2)/mean(w)^2 - 1), the EXACT (not asymptotic)
    coefficient of variation of the weights (Notation, reserse section 1.1).
    c_0 = 0 exactly (w=1 for alpha=0)."""
    if alpha == 0.0:
        return 0.0
    w = weights_from_Dtriu(D_triu, alpha, eps_D)
    mean_w = float(w.mean())
    mean_w2 = float(np.mean(w**2))
    ratio = mean_w2 / mean_w**2 - 1.0
    if -1e-9 < ratio < 0.0:
        ratio = 0.0  # floating-point noise around the c=0 boundary
    if ratio < 0.0:
        raise ValueError(f"c_alpha^2={ratio} < 0 beyond numerical tolerance (mean_w2={mean_w2}, mean_w={mean_w}) - check the input weights.")
    return float(np.sqrt(ratio))


def weight_cv_first_order(alpha: float, s_u: float) -> float:
    """c_alpha ~ alpha*s_u (Tvrzeni 3, first order in alpha*s_u -> 0)."""
    return float(alpha * s_u)


def weight_cv_second_order(alpha: float, s_u: float, skew_v: float) -> float:
    """c_alpha ~ alpha*s_u*(1 - skew_v*alpha*s_u/2) (Tvrzeni 3, second order)."""
    return float(alpha * s_u * (1.0 - skew_v * alpha * s_u / 2.0))


def r_eps_kappa_eps(D_triu: np.ndarray, theta: float, m: float, eps_D: float) -> tuple[float, float]:
    """r_eps = (theta+eps_D)/(m+eps_D), kappa_eps = (theta+eps_D)/(D_min+eps_D)
    (Notation, reused identically to `src.sammon.prop2_check.evaluate_proposition2`,
    but exposed standalone here since exp10 rows also need it for alpha=0,
    where `evaluate_proposition2` itself is undefined - it requires alpha>0)."""
    d_min = float(np.min(D_triu))
    r_eps = (theta + eps_D) / (m + eps_D)
    kappa_eps = (theta + eps_D) / (d_min + eps_D)
    return float(r_eps), float(kappa_eps)


def alpha_dagger(n: int, r_eps: float) -> float:
    """alpha_dagger = log(2(n-1))/log(1/r_eps) (Veta 2, reserse section 5.2) -
    the weight-mass threshold beyond which near pairs carry at least as much
    weighted mass as far pairs, for EVERY configuration (rigorous variant,
    not requiring a witness). NaN if r_eps=1 (log(1/r_eps)=0 -> undefined /
    infinite threshold - never identifiable at any finite alpha)."""
    if not (0.0 < r_eps <= 1.0):
        raise ValueError(f"r_eps must be in (0,1], got {r_eps}.")
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}.")
    denom = np.log(1.0 / r_eps)
    if denom <= 0.0:
        return float("nan")
    return float(np.log(2.0 * (n - 1)) / denom)


# --- Lemma 2 / Veta 1 (identity and bound on the identifiability gaps) -----


def veta1_quantities(D: np.ndarray, Y0: np.ndarray, Ya: np.ndarray, alpha: float, eps_D: float) -> dict[str, Any]:
    """All quantities of Lemma 2 (identity) and Veta 1 (bound) for one row
    (dataset, alpha): Y0 = the alpha=0 minimizer (reference), Ya = the
    alpha-weighted minimizer (same dataset/seed). For alpha=0, Y0 and Ya are
    the SAME embedding and every gap collapses to 0 by construction (reserse
    section 2.5, "alpha=0: w=1, c_0=0, (L2) je 1=1, Delta=Delta'=0").

    Raises ValueError (fail-loud) if sigma_0(Y0) <= 0 or sigma_alpha(Ya) <= 0
    (perfectly embeddable data - Delta/Delta' are then genuinely undefined,
    reserse section 2.5, NOT approximated by a fallback)."""
    if eps_D < 0:
        raise ValueError(f"eps_D must be >= 0, got {eps_D}.")
    D_triu, rho_Y0 = pair_residuals(D, Y0)
    _, rho_Ya = pair_residuals(D, Ya)
    w = weights_from_Dtriu(D_triu, alpha, eps_D)
    rho2_Y0 = rho_Y0**2
    rho2_Ya = rho_Ya**2

    sigma0_Y0 = float(np.sum(rho2_Y0))
    sigma0_Ya = float(np.sum(rho2_Ya))
    sigmaA_Y0 = float(np.sum(w * rho2_Y0))
    sigmaA_Ya = float(np.sum(w * rho2_Ya))
    if sigma0_Y0 <= 0.0:
        raise ValueError("sigma_0(Y_0*) <= 0 - the data is perfectly embeddable, Delta/Delta' are undefined (reserse section 2.5).")
    if sigmaA_Ya <= 0.0:
        raise ValueError("sigma_alpha(Y_alpha*) <= 0 - degenerate weighted stress, Delta' is undefined.")

    c = weight_cv_exact(D_triu, alpha, eps_D)
    gamma_Y0 = coefficient_of_variation(rho2_Y0)
    gamma_Ya = coefficient_of_variation(rho2_Ya)
    gamma_Y0_safe = 0.0 if np.isnan(gamma_Y0) else gamma_Y0
    gamma_Ya_safe = 0.0 if np.isnan(gamma_Ya) else gamma_Ya
    r_Y0 = pearson_corr(w, rho2_Y0)
    r_Ya = pearson_corr(w, rho2_Ya)

    Delta = sigma0_Ya / sigma0_Y0 - 1.0
    Delta_prime = sigmaA_Y0 / sigmaA_Ya - 1.0
    both_gaps_nonneg = bool(Delta >= -1e-9 and Delta_prime >= -1e-9)

    lambda_ratio_check = (1.0 + r_Y0 * c * gamma_Y0_safe) / (1.0 + r_Ya * c * gamma_Ya_safe)
    identity_residual = (1.0 + Delta) * (1.0 + Delta_prime) - lambda_ratio_check

    c_gamma_a = c * gamma_Ya_safe
    bound_nontrivial = bool(c_gamma_a < 1.0)
    if bound_nontrivial:
        bound_v1 = (1.0 + c * gamma_Y0_safe) / (1.0 - c_gamma_a) - 1.0
        tight_v1 = Delta_prime / bound_v1 if bound_v1 > 0.0 else float("nan")
    else:
        bound_v1 = float("nan")
        tight_v1 = float("nan")

    return {
        "sigma0_Y0": sigma0_Y0, "sigma0_Ya": sigma0_Ya, "sigmaA_Y0": sigmaA_Y0, "sigmaA_Ya": sigmaA_Ya,
        "Delta": Delta, "Delta_prime": Delta_prime, "both_gaps_nonneg": both_gaps_nonneg,
        "gamma_tilde_Y0": gamma_Y0, "gamma_tilde_Ya": gamma_Ya, "r_Y0": r_Y0, "r_Ya": r_Ya,
        "lambda_ratio_check": lambda_ratio_check, "identity_residual": identity_residual,
        "c_alpha": c, "bound_v1": bound_v1, "bound_nontrivial": bound_nontrivial, "tight_v1": tight_v1,
    }


def sandwich_bound(D_triu: np.ndarray, eps_D: float, alpha: float) -> float:
    """bound_sandwich = ((D_max+eps_D)/(D_min+eps_D))^alpha - 1 (Lemma 1,
    the original sandwich bound - for comparison with the sharper Veta 1
    bound; reserse sections 1.2/2.3)."""
    D_triu = np.asarray(D_triu, dtype=np.float64)
    d_max = float(D_triu.max())
    d_min = float(D_triu.min())
    return float(((d_max + eps_D) / (d_min + eps_D)) ** alpha - 1.0)


def log_lambda_alpha(D: np.ndarray, Y: np.ndarray, alpha: float, eps_D: float) -> float:
    """log Lambda_alpha(Y) = log(sigma_alpha(Y)/sigma_0(Y)) (reserse section
    1.2 - the scale-correct multiplicative object that replaces the
    unbounded additive difference of normalized stresses). Used by the
    Protipriklad 1 (counterexample) unit test.

    Uses `weights_from_Dtriu` (not `prop2_check.sigma_alpha_value` /
    `weights.alpha_weights`) so that eps_D=0 is allowed whenever D has no
    duplicate pairs - the counterexample (reserse section 1.2b) is stated
    with eps_D=0."""
    D_triu, rho = pair_residuals(D, Y)
    w = weights_from_Dtriu(D_triu, alpha, eps_D)
    sigma_a = float(np.sum(w * rho**2))
    sigma_0 = float(np.sum(rho**2))
    if sigma_0 <= 0.0:
        raise ValueError("sigma_0(Y) <= 0 - Lambda_alpha(Y) is undefined.")
    return float(np.log(sigma_a / sigma_0))


# --- Veta 3 (local quadratic law, index I_0) --------------------------------


def _rigid_motion_dof(p: int) -> int:
    """p(p+1)/2 rigid-motion degrees of freedom (translations + rotations)
    in R^p - the dimension of the null space that MUST be cut from the
    pseudo-inverse of the Hessian/Gram matrix (Veta 3, reserse section 4.1)."""
    return p * (p + 1) // 2


def _assemble_block_matrix(
    n: int, p: int, iu: np.ndarray, ju: np.ndarray, diag_term: np.ndarray, offdiag_term: np.ndarray,
) -> np.ndarray:
    """Scatter-add N per-pair (p x p) blocks into a dense (np x np) matrix:
    diag_term[e] is added to the (i,i) and (j,j) diagonal blocks, and
    offdiag_term[e] to the (i,j) and (j,i) off-diagonal blocks, for every
    pair e=(i,j)=(iu[e],ju[e]). Uses a COO sparse matrix as an accumulator
    (duplicate (row,col) entries from different pairs sharing a point are
    summed automatically on conversion to a dense array), which avoids an
    explicit Python loop over the N pairs."""
    from scipy.sparse import coo_matrix

    N = iu.shape[0]
    a_idx, b_idx = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    a_idx = a_idx.ravel()
    b_idx = b_idx.ravel()  # length p*p, matches diag_term.reshape(N, p*p) element order

    vals_diag = diag_term.reshape(N, p * p)
    vals_offdiag = offdiag_term.reshape(N, p * p)

    rows_ii = iu[:, None] * p + a_idx[None, :]
    cols_ii = iu[:, None] * p + b_idx[None, :]
    rows_jj = ju[:, None] * p + a_idx[None, :]
    cols_jj = ju[:, None] * p + b_idx[None, :]
    rows_ij = iu[:, None] * p + a_idx[None, :]
    cols_ij = ju[:, None] * p + b_idx[None, :]
    rows_ji = ju[:, None] * p + a_idx[None, :]
    cols_ji = iu[:, None] * p + b_idx[None, :]

    all_rows = np.concatenate([rows_ii.ravel(), rows_jj.ravel(), rows_ij.ravel(), rows_ji.ravel()])
    all_cols = np.concatenate([cols_ii.ravel(), cols_jj.ravel(), cols_ij.ravel(), cols_ji.ravel()])
    all_vals = np.concatenate([vals_diag.ravel(), vals_diag.ravel(), vals_offdiag.ravel(), vals_offdiag.ravel()])

    M = coo_matrix((all_vals, (all_rows, all_cols)), shape=(n * p, n * p)).toarray()
    return M


def _pinv_quadratic_form(M: np.ndarray, x: np.ndarray, rel_tol: float) -> tuple[float, int, float]:
    """x^T M^+ x via the eigendecomposition of the symmetric matrix M, with
    the pseudo-inverse cutting every eigenvalue with |eigenvalue| <=
    rel_tol*max(|eigenvalues|) (the p(p+1)/2 rigid-motion directions,
    Veta 3). Returns (quadratic_form, n_cut, lambda_min_pos) where n_cut is
    the number of directions actually cut (a diagnostic - the unit test
    checks that it equals p(p+1)/2 for a nondegenerate configuration) and
    lambda_min_pos is the smallest eigenvalue classified as significant and
    positive (NaN if none)."""
    eigvals, eigvecs = np.linalg.eigh(M)
    max_abs = float(np.max(np.abs(eigvals))) if eigvals.size else 0.0
    threshold = rel_tol * max_abs
    significant = np.abs(eigvals) > threshold
    inv_eigvals = np.zeros_like(eigvals)
    inv_eigvals[significant] = 1.0 / eigvals[significant]
    proj = eigvecs.T @ x
    quad = float(np.sum(inv_eigvals * proj**2))
    n_cut = int(np.sum(~significant))
    pos_eigvals = eigvals[significant & (eigvals > 0.0)]
    lambda_min_pos = float(np.min(pos_eigvals)) if pos_eigvals.size > 0 else float("nan")
    return quad, n_cut, lambda_min_pos


def local_identifiability_index(
    D: np.ndarray, Y0: np.ndarray, eps_D: float, n_max_hessian: int, pinv_rel_tol: float,
) -> dict[str, Any]:
    """I_0 (exact, via the full Hessian H_0) and I_0^GN (Gauss-Newton
    approximation, via G=J^T J) of Veta 3 (reserse section 4), computed ONLY
    from the alpha=0 minimizer Y0 (independent of alpha - `pred_quadratic =
    alpha^2 * I0_exact` is computed by the caller per row).

    g = 2 J^T(rho o v), Q = g^T H_0^+ g, I0_exact = Q/(2 sigma_0(Y0));
    Q_GN = 2||P_J(rho o v)||^2 = 2 b^T G^+ b, I0_gn = Q_GN/(2 sigma_0(Y0)) =
    b^T G^+ b/sigma_0(Y0), b = g/2. Both pseudo-inverses cut the
    p(p+1)/2-dimensional rigid-motion null space (`_pinv_quadratic_form`).

    Skips the (O(n^3) via eigh of an (np x np) matrix) computation for
    n > n_max_hessian, returning NaN with hessian_skipped=True (reserse
    section 10: "Pro n > n_max_hessian NaN a priznak hessian_skipped").

    Fail-loud on coincident points (d_e(Y0)=0 for some pair) - Veta 3
    explicitly assumes a local minimizer WITHOUT coincidences."""
    Y0 = np.asarray(Y0, dtype=np.float64)
    n, p = Y0.shape
    if n > n_max_hessian:
        return {
            "I0_gn": float("nan"), "I0_exact": float("nan"), "lambda_min_pos_H0": float("nan"),
            "s_u_pi_sq": float("nan"), "hessian_skipped": True,
        }

    iu, ju = np.triu_indices(n, k=1)
    diff = Y0[iu] - Y0[ju]
    d = np.linalg.norm(diff, axis=1)
    if np.any(d <= 0.0):
        raise ValueError("Y0 contains coincident points (d_e=0 for at least one pair) - Veta 3 requires a nondegenerate local minimizer without coincidences.")

    D_triu = np.asarray(D, dtype=np.float64)[iu, ju]
    if eps_D < 0:
        raise ValueError(f"eps_D must be >= 0, got {eps_D}.")
    u = np.log(D_triu + eps_D)
    v = u - float(u.mean())
    rho = D_triu - d

    sigma0_Y0 = float(np.sum(rho**2))
    if sigma0_Y0 <= 0.0:
        raise ValueError("sigma_0(Y0) <= 0 - the local index I0 is undefined.")
    pi = rho**2 / sigma0_Y0
    s_u_pi_sq = float(np.sum(pi * v**2))

    direction = diff / d[:, None]  # unit vectors e_hat_ij, shape (N, p)
    eye_p = np.eye(p)
    uu = np.einsum("ei,ej->eij", direction, direction)  # N x p x p

    # b = J^T(rho o v): contribution to block i is -rho_e*v_e*e_hat_ij, to block j is +rho_e*v_e*e_hat_ij
    contrib = -(rho * v)[:, None] * direction
    b = np.zeros((n, p))
    np.add.at(b, iu, contrib)
    np.add.at(b, ju, -contrib)
    b_flat = b.reshape(-1)

    diag_term_G = uu
    offdiag_term_G = -uu
    G = _assemble_block_matrix(n, p, iu, ju, diag_term_G, offdiag_term_G)

    diag_term_H = 2.0 * (uu - (rho / d)[:, None, None] * (eye_p[None, :, :] - uu))
    offdiag_term_H = -diag_term_H
    H0 = _assemble_block_matrix(n, p, iu, ju, diag_term_H, offdiag_term_H)

    expected_cut = _rigid_motion_dof(p)

    bT_Gp_b, n_cut_G, _ = _pinv_quadratic_form(G, b_flat, pinv_rel_tol)
    I0_gn = bT_Gp_b / sigma0_Y0

    bT_H0p_b, n_cut_H0, lambda_min_pos_H0 = _pinv_quadratic_form(H0, b_flat, pinv_rel_tol)
    I0_exact = 2.0 * bT_H0p_b / sigma0_Y0

    return {
        "I0_gn": float(I0_gn), "I0_exact": float(I0_exact), "lambda_min_pos_H0": lambda_min_pos_H0,
        "s_u_pi_sq": s_u_pi_sq, "hessian_skipped": False,
        "n_cut_G": n_cut_G, "n_cut_H0": n_cut_H0, "expected_rigid_dof": expected_cut,
    }


# --- Veta 2 (dual certificate) ----------------------------------------------


def witness_denominator(R_near: float, R_mid: float, R_far: float, alpha: float, kappa_eps: float, r_eps: float) -> float:
    """Denominator of (V2): kappa_eps^alpha R_near(Y~) + R_mid(Y~) + r_eps^alpha R_far(Y~)."""
    return float((kappa_eps**alpha) * R_near + R_mid + (r_eps**alpha) * R_far)


def certificate_lower_bound(
    R_near_Y0: float, R_mid_Y0: float, alpha: float, r_eps: float, kappa_eps: float,
    witness_blocks: Sequence[tuple[float, float, float, float]],
) -> dict[str, Any]:
    """(V2): sigma_alpha(Y_0)/sigma_alpha(Y_alpha*) >= sigma_alpha(Y_0)/sigma_alpha(Y~)
    >= [R_near(Y_0) + r_eps^alpha R_mid(Y_0)] / min_witness den(witness, alpha),
    minimized over the given witnesses (any configuration is a valid
    witness, reserse section 5.1 - here the whole alpha-grid of already
    computed embeddings). `cert_lower = ... - 1` is a certified LOWER bound
    on Delta'_alpha (can be negative = no certificate from this witness set).

    witness_blocks: sequence of (alpha_prime, R_near, R_mid, R_far) tuples,
    one per candidate configuration Y_alpha_prime (computed with the SAME
    theta/m as Y_0, i.e. the SAME P_near/P_mid/P_far masks)."""
    if not witness_blocks:
        raise ValueError("witness_blocks must not be empty - need at least one candidate configuration for the certificate.")
    numerator = R_near_Y0 + (r_eps**alpha) * R_mid_Y0
    dens = [(a2, witness_denominator(rn, rm, rf, alpha, kappa_eps, r_eps)) for (a2, rn, rm, rf) in witness_blocks]
    witness_alpha, best_den = min(dens, key=lambda t: t[1])
    if best_den <= 0.0:
        raise ValueError("Witness denominator <= 0 - degenerate configuration(s), cannot build the Veta 2 certificate.")
    cert_lower = numerator / best_den - 1.0
    return {"cert_lower": float(cert_lower), "witness_alpha": float(witness_alpha)}


def certificate_direct(sigmaA_Y0: float, witness_sigmas: Sequence[tuple[float, float]]) -> dict[str, Any]:
    """cert_lower_direct = sigma_alpha(Y_0)/min_witness sigma_alpha(Y_witness) - 1,
    computed directly from the already-evaluated weighted stresses of the
    candidate configurations (a numerical sanity check of
    `certificate_lower_bound` against the actual grid, not a closed-form
    bound). witness_sigmas: sequence of (alpha_prime, sigma_alpha(Y_alpha_prime))."""
    if not witness_sigmas:
        raise ValueError("witness_sigmas must not be empty.")
    witness_alpha, min_val = min(witness_sigmas, key=lambda t: t[1])
    if min_val <= 0.0:
        raise ValueError("min witness sigma_alpha <= 0 - cannot build the direct certificate.")
    return {"cert_lower_direct": float(sigmaA_Y0 / min_val - 1.0), "witness_alpha_direct": float(witness_alpha)}


def alpha_eta_threshold(
    R_near_Y0: float, witness_block: tuple[float, float, float], eta: float, r_eps: float, kappa_eps: float,
    alpha_max: float, xtol: float, near_zero_rel_tol: float,
) -> float:
    """alpha_eta: the smallest alpha for which Delta'_alpha >= eta is
    certified by (V2) via the single fixed witness `witness_block` =
    (R_near, R_mid, R_far) of Y~ (reserse section 5.1). Solves
    R_near(Y_0) = (1+eta)*[kappa_eps^alpha*R_near(Y~) + R_mid(Y~) + r_eps^alpha*R_far(Y~)]
    for alpha >= 0.

    Closed form (reserse eq. in section 5.1) when R_near(Y~) is negligible
    relative to R_mid(Y~)+R_far(Y~) (below `near_zero_rel_tol`): R_loc(Y~)
    does not depend on alpha, and the equation is log-linear in r_eps^alpha.
    Otherwise kappa_eps^alpha grows with alpha while r_eps^alpha shrinks, so
    the general equation is solved numerically by bisection (scipy.optimize.brentq)
    on [0, alpha_max]. Returns NaN if the conflict condition
    R_near(Y_0) > (1+eta)*R_loc(Y~)|_{alpha=0} fails, or if no sign change is
    found on [0, alpha_max] (no threshold within the search range - NOT
    extrapolated, same convention as `prop2_check.compute_alpha_bound`)."""
    if eta <= 0.0:
        raise ValueError(f"eta must be > 0, got {eta}.")
    if not (0.0 < r_eps <= 1.0):
        raise ValueError(f"r_eps must be in (0,1], got {r_eps}.")
    if kappa_eps < 1.0:
        raise ValueError(f"kappa_eps must be >= 1, got {kappa_eps}.")
    R_near_w, R_mid_w, R_far_w = witness_block

    def f(alpha: float) -> float:
        return R_near_Y0 - (1.0 + eta) * witness_denominator(R_near_w, R_mid_w, R_far_w, alpha, kappa_eps, r_eps)

    if f(0.0) <= 0.0:
        return float("nan")  # conflict condition fails already at alpha=0 - never certified in [0, alpha_max]

    total_w = R_near_w + R_mid_w + R_far_w
    near_negligible = total_w <= 0.0 or (R_near_w / total_w) <= near_zero_rel_tol
    if near_negligible:
        # R_loc(Y~) = R_mid_w independent of alpha -> closed-form solution for r_eps^alpha
        target = R_near_Y0 / (1.0 + eta) - R_mid_w
        if target <= 0.0 or R_far_w <= 0.0:
            return float("nan")
        ratio = target / R_far_w
        if not (0.0 < ratio <= 1.0):
            return float("nan")  # r_eps^alpha=ratio outside (0,1] -> no valid alpha>=0
        return float(np.log(ratio) / np.log(r_eps))

    if f(alpha_max) > 0.0:
        return float("nan")  # no sign change on [0, alpha_max] - not extrapolated beyond the search range
    from scipy.optimize import brentq

    return float(brentq(f, 0.0, alpha_max, xtol=xtol))


# --- E8 slack decomposition S1..S4 (reserse section 6) ----------------------


def sigma_alpha_masked(D: np.ndarray, Y: np.ndarray, alpha: float, eps_D: float, mask: np.ndarray) -> float:
    """sum_{e in mask} w_e rho_e(Y)^2 - the weighted stress restricted to a
    subset of pairs (e.g. P_near), used for the E8 slack decomposition S2/S3
    (sigma_alpha^near(Y), reserse section 6.1)."""
    D_triu, rho = pair_residuals(D, Y)
    w = weights_from_Dtriu(D_triu, alpha, eps_D)
    return float(np.sum(w[mask] * rho[mask] ** 2))


def slack_decomposition(
    D: np.ndarray, Y: np.ndarray, alpha: float, eps_D: float, theta: float, m: float,
    bound_a: float, bound_b: float, R_near_Y: float, tight_a: float,
) -> dict[str, float]:
    """1/tight_a = S1*S2*S3 (reserse section 6.1): S1 = sigma_alpha(Ybar)/sigma_alpha(Y)
    (optimization gain vs. the E8 reference Ybar=embedding at alpha=0.0,
    recovered from bound_a = (theta+eps_D)^alpha * sigma_alpha(Ybar) - the
    E8 identity `bound_a = sum phi_alpha(D) rho(Ybar)^2`, see
    `src.sammon.prop2_check.evaluate_proposition2`); S2 = sigma_alpha(Y)/sigma_alpha^near(Y)
    (budget factor, structural, Theta(n)); S3 = (theta+eps_D)^alpha *
    sigma_alpha^near(Y)/R_near(Y) (within-block weight heterogeneity,
    bounded in [1, kappa_eps^alpha]); S4 = bound_b/bound_a (E8's own steps
    (5)-(7), read directly from the E8 CSV). check_S = S1*S2*S3*tight_a - 1
    must be ~0 for matching (dataset, alpha) rows (identity, not a bound)."""
    D_triu, _ = pair_residuals(D, Y)
    near, _mid, _far = block_masks(D_triu, theta, m)
    sigma_alpha_Y = sigma_alpha_value(D, Y, alpha, eps_D)
    sigma_alpha_Y_near = sigma_alpha_masked(D, Y, alpha, eps_D, near)
    theta_eps_alpha = (theta + eps_D) ** alpha

    sigma_alpha_Ybar = bound_a / theta_eps_alpha if theta_eps_alpha > 0.0 else float("nan")
    S1 = sigma_alpha_Ybar / sigma_alpha_Y if sigma_alpha_Y > 0.0 else float("nan")
    S2 = sigma_alpha_Y / sigma_alpha_Y_near if sigma_alpha_Y_near > 0.0 else float("nan")
    S3 = theta_eps_alpha * sigma_alpha_Y_near / R_near_Y if R_near_Y > 0.0 else float("nan")
    S4 = bound_b / bound_a if bound_a > 0.0 else float("nan")
    check_S = S1 * S2 * S3 * tight_a - 1.0
    return {"S1": S1, "S2": S2, "S3": S3, "S4": S4, "check_S": check_S}


# --- alpha_pred stratum (reserse section 3.5 / 5, `sammon_alpha_pred` rule) -


def stratum_from_rule(nn_ratio_k1: float, rule: dict[str, Any]) -> str:
    """low/mid/high stratum of nn_ratio_k1 per the SAME thresholds as the
    production `two_threshold` alpha_pred rule (`src.sammon.alpha_predict.predict_alpha_two_threshold`,
    reused - not redefined). 'low' = low nn_ratio_k1 = concentrated data
    (high alpha_pred), 'high' = high nn_ratio_k1 (low alpha_pred) - see the
    2026-09-14 label fix (low/mid/high_ratio, not low/mid/high 'regime').
    Fail-loud for any rule variant other than 'two_threshold' (no discrete
    strata are defined for 'log_linear')."""
    if nn_ratio_k1 <= 0:
        raise ValueError(f"nn_ratio_k1 must be positive, got {nn_ratio_k1}.")
    variant = rule.get("variant")
    if variant != "two_threshold":
        raise ValueError(f"stratum_from_rule only supports the 'two_threshold' rule variant, got '{variant}'.")
    coef = rule["coefficients"]
    t1 = float(coef["t1"])
    t2 = float(coef["t2"])
    log_nn = float(np.log(nn_ratio_k1))
    if log_nn < t1:
        return "low"
    if log_nn < t2:
        return "mid"
    return "high"
