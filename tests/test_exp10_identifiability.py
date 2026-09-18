# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-17
# License: see the LICENSE file in the repository root
"""Tests for the pure numerical functions of the identifiability check
(src/sammon/identifiability.py), reserse/2026-09-17_zostreni_propozice2.md
section 10 ("Specifikace numerickeho overeni pro python-codera", there named
E9/exp9_identifiability_check - renamed here to E10/exp10_identifiability_check
to avoid a collision with the already-existing exp9_metric_fidelity).

The main safeguard is the "three points on a line" family (reserse section
2.4): a closed-form exactly solvable case that must reproduce the
hand-computed table to 3-4 significant digits, INCLUDING the local index I0
via the pseudo-inverse of the full (np x np) Hessian (the single most likely
place for a silent bug - cutting exactly p(p+1)/2 rigid-motion directions)."""
from __future__ import annotations

import numpy as np
import pytest

from src.sammon.identifiability import (
    _pinv_quadratic_form,
    _rigid_motion_dof,
    alpha_dagger,
    certificate_direct,
    certificate_lower_bound,
    coefficient_of_variation,
    local_identifiability_index,
    log_distance_moments,
    log_lambda_alpha,
    pearson_corr,
    r_eps_kappa_eps,
    sandwich_bound,
    slack_decomposition,
    stratum_from_rule,
    veta1_quantities,
    weight_cv_exact,
    weight_cv_first_order,
    witness_denominator,
)
from src.sammon.prop2_check import evaluate_proposition2, pair_residuals

# ============================================================================
# Three points on a line (reserse section 2.4): a=1, b=1.5, p=1, eps_D=0.
# Order 2-1-3 on the line -> D_triu (i<j over indices [pt2=0, pt1=1, pt3=2]):
#   D(0,1)=D_21=a, D(0,2)=D_23=b, D(1,2)=D_13=a
# x*(alpha) = (a + t*b)/(1+2t), t=(a/b)^alpha; x*(0)=(a+b)/3.
# Y = [[0], [x], [2x]] (Euclidean distance on R^1 reproduces d_01=x, d_02=2x, d_12=x).
# ============================================================================
_A = 1.0
_B = 1.5
_EPS_D = 0.0


def _three_point_D() -> np.ndarray:
    return np.array([
        [0.0, _A, _B],
        [_A, 0.0, _A],
        [_B, _A, 0.0],
    ])


def _three_point_Y(x: float) -> np.ndarray:
    return np.array([[0.0], [x], [2.0 * x]])


def _x_star(alpha: float) -> float:
    t = (_A / _B) ** alpha
    return (_A + t * _B) / (1.0 + 2.0 * t)


_Y0 = _three_point_Y(_x_star(0.0))  # alpha=0 minimizer, x0=(a+b)/3

# hand-computed reference table, reserse section 2.4
_TABLE = {
    1.0: dict(Delta=0.040816, c_alpha=0.176777, gamma_tilde_Ya=0.415879, r_Ya=-1.0, bound_v1=0.079343, Delta_prime=0.037064, I0=0.036534),
    2.0: dict(Delta=0.172996, c_alpha=0.321346, gamma_tilde_Ya=0.813241, r_Ya=-1.0, bound_v1=0.353755, Delta_prime=0.154106, I0=0.036534),
    3.0: dict(Delta=0.390527, c_alpha=0.433444, gamma_tilde_Ya=1.097337, r_Ya=-1.0, bound_v1=0.907207, Delta_prime=0.371501, I0=0.036534),
}
# I0 = 2*log(b/a)^2/9 is the SAME at every alpha (Veta 3: I0 is a property of
# Y0 alone, not of alpha) - reserse section 2.4/4.2 "presne resitelna rodina
# ... dosahuje I_0 = s_{u,pi}^2 = s_u^2 = 2L^2/9 s rovnosti".
_I0_EXPECTED = 2.0 * np.log(_B / _A) ** 2 / 9.0


@pytest.mark.parametrize("alpha", [1.0, 2.0, 3.0])
def test_three_point_family_matches_hand_computed_table(alpha: float) -> None:
    D = _three_point_D()
    Ya = _three_point_Y(_x_star(alpha))
    expected = _TABLE[alpha]

    out = veta1_quantities(D, _Y0, Ya, alpha, _EPS_D)

    assert out["Delta"] == pytest.approx(expected["Delta"], abs=2e-4)
    assert out["c_alpha"] == pytest.approx(expected["c_alpha"], abs=2e-4)
    assert out["gamma_tilde_Ya"] == pytest.approx(expected["gamma_tilde_Ya"], abs=2e-3)
    assert out["r_Ya"] == pytest.approx(expected["r_Ya"], abs=1e-9)  # exact -1 (Cauchy-Schwarz equality, two-valued weights)
    assert out["gamma_tilde_Y0"] == pytest.approx(0.0, abs=1e-9)  # Y0* has EQUAL squared residuals (well-known MDS property)
    assert out["bound_nontrivial"] is True
    assert out["bound_v1"] == pytest.approx(expected["bound_v1"], abs=2e-3)
    assert out["Delta_prime"] == pytest.approx(expected["Delta_prime"], abs=2e-3)
    assert out["tight_v1"] == pytest.approx(expected["Delta_prime"] / expected["bound_v1"], abs=5e-3)


@pytest.mark.parametrize("alpha", [1.0, 2.0, 3.0])
def test_three_point_family_identity_residual_is_zero(alpha: float) -> None:
    """Lemma 2 / Veta 1(a) is an IDENTITY, not an inequality - a nonzero
    residual means a bug in the implementation (reserse section 10, point 2)."""
    D = _three_point_D()
    Ya = _three_point_Y(_x_star(alpha))
    out = veta1_quantities(D, _Y0, Ya, alpha, _EPS_D)
    assert abs(out["identity_residual"]) < 1e-10


def test_three_point_family_local_index_matches_2L2_over_9() -> None:
    """Veta 3 / reserse section 4.2: in this family the Gauss-Newton
    approximation is EXACT and I0 = s_u^2 = 2*log(b/a)^2/9 for every alpha
    (I0 is a property of Y0* alone). This is the main pseudo-inverse
    safeguard: p=1 -> exactly 1 rigid-motion direction (translation; no
    rotation in 1D) must be cut from the 3x3 Hessian."""
    D = _three_point_D()
    out = local_identifiability_index(D, _Y0, _EPS_D, n_max_hessian=100, pinv_rel_tol=1e-8)

    assert out["hessian_skipped"] is False
    assert out["expected_rigid_dof"] == _rigid_motion_dof(1) == 1
    assert out["n_cut_G"] == 1
    assert out["n_cut_H0"] == 1
    assert out["I0_exact"] == pytest.approx(_I0_EXPECTED, abs=2e-4)
    assert out["I0_gn"] == pytest.approx(_I0_EXPECTED, abs=2e-4)  # GN is exact in this family (linear residuals)
    assert out["s_u_pi_sq"] == pytest.approx(_I0_EXPECTED, abs=2e-4)  # I0 = s_{u,pi}^2 with equality here


def test_local_index_skips_for_n_above_threshold() -> None:
    D = _three_point_D()
    out = local_identifiability_index(D, _Y0, _EPS_D, n_max_hessian=2, pinv_rel_tol=1e-8)
    assert out["hessian_skipped"] is True
    assert np.isnan(out["I0_exact"])
    assert np.isnan(out["I0_gn"])


def test_local_index_rejects_coincident_points() -> None:
    D = _three_point_D()
    Y_coincident = np.array([[0.0], [0.0], [1.5]])
    with pytest.raises(ValueError, match="coincident"):
        local_identifiability_index(D, Y_coincident, _EPS_D, n_max_hessian=100, pinv_rel_tol=1e-8)


def test_pinv_quadratic_form_cuts_rigid_translation_in_2d() -> None:
    """A minimal direct check of `_pinv_quadratic_form` independent of the
    stress Hessian: for a symmetric matrix with a KNOWN null space (here a
    graph Laplacian-like block matrix with the all-ones vector in its
    kernel), the pseudo-inverse must cut exactly that direction and the
    quadratic form of a vector ORTHOGONAL to the kernel must equal the
    ordinary (non-pseudo) inverse result."""
    # 3x3 Laplacian of a path graph 0-1-2 (rows/cols sum to 0, kernel = span{[1,1,1]})
    L = np.array([[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]])
    x = np.array([1.0, 0.0, -1.0])  # orthogonal to the kernel (sums to 0)
    quad, n_cut, lambda_min_pos = _pinv_quadratic_form(L, x, rel_tol=1e-8)
    assert n_cut == 1
    # restrict to the 2D orthogonal complement and invert directly
    eigvals, eigvecs = np.linalg.eigh(L)
    significant = np.abs(eigvals) > 1e-8 * np.max(np.abs(eigvals))
    V = eigvecs[:, significant]
    Lr = V.T @ L @ V
    xr = V.T @ x
    quad_expected = float(xr @ np.linalg.solve(Lr, xr))
    assert quad == pytest.approx(quad_expected, abs=1e-9)
    assert lambda_min_pos == pytest.approx(min(eigvals[significant]), abs=1e-9)


# ============================================================================
# Counterexample 1 (reserse section 1.2b / 10): a near-equidistant Gaussian
# cluster plus one outlier point. c_alpha stays small (data concentrates as
# d grows), but dilating the outlier point makes |log Lambda_alpha(Y) -
# log Lambda_alpha(Y_ref)| bounded away from 0 by more than 0.5*alpha*log(L)
# - i.e. NO bound of the form C*alpha*CV(D) (uniform over Y) can hold.
# ============================================================================


def _counterexample_D(n_cluster: int, d: int, L: float, seed: int) -> np.ndarray:
    """n = n_cluster+1 points: point 0 is the outlier at distance ~L from
    every cluster point, points 1..n_cluster form a near-equidistant
    high-dimensional Gaussian cluster (pairwise distances concentrate as
    d grows, no explicit rescaling needed - Beyer/Francois concentration)."""
    rng = np.random.default_rng(seed)
    from scipy.spatial.distance import pdist, squareform

    cluster = rng.normal(size=(n_cluster, d))
    D_cluster = squareform(pdist(cluster, metric="euclidean"))
    mean_cluster_dist = float(D_cluster[np.triu_indices(n_cluster, k=1)].mean())
    D_cluster = D_cluster / mean_cluster_dist  # rescale so cluster pairwise distances are ~1

    n = n_cluster + 1
    D = np.zeros((n, n))
    D[1:, 1:] = D_cluster
    D[0, 1:] = L
    D[1:, 0] = L
    return D


def test_counterexample1_small_c_alpha_but_large_lambda_shift(exp10_cfg) -> None:
    cfg = exp10_cfg["counterexample"]
    n_cluster = int(cfg["n_cluster"])
    d = int(cfg["d"])
    L = float(cfg["L"])
    seed = int(cfg["seed"])
    alpha = float(cfg["alpha"])
    dilation_factor = float(cfg["dilation_factor"])
    c_alpha_max = float(cfg["c_alpha_max"])

    D = _counterexample_D(n_cluster, d, L, seed)
    n = D.shape[0]
    D_triu = D[np.triu_indices(n, k=1)]

    c_alpha = weight_cv_exact(D_triu, alpha, eps_D=0.0)
    assert c_alpha < c_alpha_max, f"c_alpha={c_alpha} not small (expected < {c_alpha_max}) - concentration assumption violated for this n/d."

    # a generic low-dim reference configuration (PCA of the cluster part,
    # outlier placed at its own true relative position) - NOT constructed
    # from D_ij directly, just needs non-degenerate, order-1 residuals.
    rng = np.random.default_rng(seed + 1)
    Y_ref = rng.normal(size=(n, 2))

    Y = Y_ref.copy()
    # dilate the outlier point (index 0) far away along its existing direction from the cluster centroid
    centroid = Y_ref[1:].mean(axis=0)
    direction = Y_ref[0] - centroid
    norm_dir = np.linalg.norm(direction)
    assert norm_dir > 0.0
    direction = direction / norm_dir
    M = dilation_factor * n
    Y[0] = centroid + M * direction

    log_lambda_Y = log_lambda_alpha(D, Y, alpha, eps_D=0.0)
    log_lambda_Yref = log_lambda_alpha(D, Y_ref, alpha, eps_D=0.0)
    shift = abs(log_lambda_Y - log_lambda_Yref)
    threshold = 0.5 * alpha * np.log(L)
    assert shift > threshold, f"shift={shift} not > 0.5*alpha*log(L)={threshold} - counterexample did not reproduce (see reserse section 1.2b)."


@pytest.fixture()
def exp10_cfg() -> dict:
    from src.experiments.config_experiments import resolve_experiment_config

    return resolve_experiment_config("exp10_identifiability_check", "full")


def test_gaussian_su_close_to_trigamma_heuristic() -> None:
    """reserse section 3.3: for X,X' ~ N(0,I_d), population s_u^2 = psi_1(d/2)/4
    ~ 1/(2d); d=100 -> s_u ~ sqrt(psi_1(50))/2 ~ 0.071 (reserse section 10, point 1)."""
    from scipy.special import polygamma

    d = 100
    rng = np.random.default_rng(20260917)
    n = 400
    X = rng.normal(size=(n, d))
    from scipy.spatial.distance import pdist

    D_triu = pdist(X, metric="euclidean")
    stats = log_distance_moments(D_triu, eps_D=0.0)
    expected_s_u = float(np.sqrt(polygamma(1, d / 2.0)) / 2.0)
    assert expected_s_u == pytest.approx(0.071, abs=2e-3)
    assert stats["s_u"] == pytest.approx(expected_s_u, rel=0.15)


# --- smaller unit tests for the remaining pure functions --------------------


def test_coefficient_of_variation_basic_and_degenerate() -> None:
    assert coefficient_of_variation(np.array([1.0, 1.0, 1.0])) == pytest.approx(0.0)
    assert np.isnan(coefficient_of_variation(np.array([0.0, 0.0])))
    x = np.array([1.0, 2.0, 3.0])
    assert coefficient_of_variation(x) == pytest.approx(x.std() / x.mean())


def test_pearson_corr_degenerate_returns_zero() -> None:
    assert pearson_corr(np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0])) == 0.0
    assert pearson_corr(np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0])) == pytest.approx(-1.0)


def test_r_eps_kappa_eps_and_alpha_dagger() -> None:
    D_triu = np.array([1.0, 2.0, 3.0, 4.0])
    r_eps, kappa_eps = r_eps_kappa_eps(D_triu, theta=1.5, m=2.5, eps_D=0.5)
    assert r_eps == pytest.approx(2.0 / 3.0)
    assert kappa_eps == pytest.approx(2.0 / 1.5)
    ad = alpha_dagger(n=1000, r_eps=0.5)
    assert ad == pytest.approx(np.log(2 * 999) / np.log(2.0))
    assert np.isnan(alpha_dagger(n=10, r_eps=1.0))


def test_weight_cv_first_order_matches_alpha_times_su() -> None:
    assert weight_cv_first_order(2.0, 0.3) == pytest.approx(0.6)


def test_sandwich_bound_matches_Lemma1() -> None:
    D_triu = np.array([1.0, 4.0])
    b = sandwich_bound(D_triu, eps_D=0.0, alpha=1.0)
    assert b == pytest.approx(4.0 - 1.0)


def test_stratum_from_rule_thresholds() -> None:
    rule = {"variant": "two_threshold", "coefficients": {"t1": -2.0, "t2": -1.0, "a_low": 2.5, "a_mid": 0.75, "a_high": 0.0}}
    assert stratum_from_rule(np.exp(-3.0), rule) == "low"
    assert stratum_from_rule(np.exp(-1.5), rule) == "mid"
    assert stratum_from_rule(np.exp(0.0), rule) == "high"
    with pytest.raises(ValueError):
        stratum_from_rule(np.exp(-3.0), {"variant": "log_linear", "coefficients": {}})


def test_witness_denominator_and_certificate_lower_bound() -> None:
    den = witness_denominator(R_near=1.0, R_mid=2.0, R_far=3.0, alpha=1.0, kappa_eps=2.0, r_eps=0.5)
    assert den == pytest.approx(2.0 * 1.0 + 2.0 + 0.5 * 3.0)
    witnesses = [(0.0, 1.0, 2.0, 3.0), (1.0, 0.5, 1.0, 1.5)]
    out = certificate_lower_bound(R_near_Y0=4.0, R_mid_Y0=1.0, alpha=1.0, r_eps=0.5, kappa_eps=2.0, witness_blocks=witnesses)
    den0 = witness_denominator(1.0, 2.0, 3.0, 1.0, 2.0, 0.5)
    den1 = witness_denominator(0.5, 1.0, 1.5, 1.0, 2.0, 0.5)
    best_den = min(den0, den1)
    numerator = 4.0 + 0.5 * 1.0
    assert out["cert_lower"] == pytest.approx(numerator / best_den - 1.0)


def test_certificate_direct() -> None:
    out = certificate_direct(sigmaA_Y0=10.0, witness_sigmas=[(0.0, 8.0), (1.0, 5.0)])
    assert out["cert_lower_direct"] == pytest.approx(10.0 / 5.0 - 1.0)
    assert out["witness_alpha_direct"] == pytest.approx(1.0)


# ============================================================================
# check_S identity (reserse section 10, point 3): S1*S2*S3*tight_a - 1 ~ 0
# for a matching (dataset, alpha) row, reusing the same 4-point example as
# tests/test_prop2_check.py (so bound_a/bound_b/R_near_Y/tight_a are taken
# from `evaluate_proposition2` itself, not hand-computed again).
# ============================================================================
_D4 = np.array([
    [0.0, 1.0, 2.0, 5.0],
    [1.0, 0.0, 3.0, 4.0],
    [2.0, 3.0, 0.0, 6.0],
    [5.0, 4.0, 6.0, 0.0],
])
_EPS_D4 = 0.5
_ALPHA4 = 1.0
_THETA4 = 1.5
_M4 = 3.5
# NOTE: _Y4 deliberately does NOT reproduce D(0,1)=1 exactly (unlike
# tests/test_prop2_check.py's Y_case_A/B, both of which have d(0,1)=1 and
# thus R_near_Y=0 identically for the near pair (0,1) - a degenerate case
# where S3=R_near_Y/... is 0/0=NaN by construction and cannot exercise the
# check_S identity below).
_Y4 = np.array([[0.0], [1.3], [2.0], [5.0]])     # plays the role of Y (SMACOF result), R_near_Y != 0
_YBAR4 = np.array([[0.0], [1.0], [3.0], [4.0]])  # plays the role of Ybar (alpha=0 embedding)


def test_check_S_identity_matches_evaluate_proposition2() -> None:
    e8 = evaluate_proposition2(_D4, _Y4, _YBAR4, _ALPHA4, _EPS_D4, _THETA4, _M4, eps_d_zero_tol=1e-12)
    assert e8["R_near_Y"] > 0.0  # sanity: this test only exercises the identity if R_near_Y != 0 (see note above)
    out = slack_decomposition(
        _D4, _Y4, _ALPHA4, _EPS_D4, _THETA4, _M4,
        bound_a=e8["bound_a"], bound_b=e8["bound_b"], R_near_Y=e8["R_near_Y"], tight_a=e8["tight_a"],
    )
    assert abs(out["check_S"]) < 1e-9
