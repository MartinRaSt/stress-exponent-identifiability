# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-14
# License: see the LICENSE file in the repository root
"""Tests for the pure numerical functions of Proposition 2
(src/sammon/prop2_check.py) on a small manually computed example (4 points,
1D configuration Y/Ybar - see documentation/2026-09-14_exp8_prop2_check.md
for the derivation of the values used below)."""
from __future__ import annotations

import numpy as np
import pytest

from src.sammon.prop2_check import (
    block_masks,
    block_residual,
    compute_alpha_bound,
    evaluate_proposition2,
    pair_residuals,
    phi_alpha,
    resolve_p2_holds,
    sigma_alpha_value,
)

# --- a manually computed example (4 points) -------------------------------------
# D_triu order (i<j): (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
_D = np.array([
    [0.0, 1.0, 2.0, 5.0],
    [1.0, 0.0, 3.0, 4.0],
    [2.0, 3.0, 0.0, 6.0],
    [5.0, 4.0, 6.0, 0.0],
])
_EPS_D = 0.5
_ALPHA = 1.0
_THETA = 1.5  # median NN distances [1,1,2,4] (see the documentation)
_M = 3.5      # median of all pairwise distances [1,2,3,4,5,6]

# 1D configuration (Euclidean distance = |difference|)
_Y_CASE_A = np.array([[0.0], [1.0], [3.0], [4.0]])  # a "worse" configuration (larger sigma_alpha)
_Y_CASE_B = np.array([[0.0], [1.0], [2.0], [5.0]])  # a "better" configuration (smaller sigma_alpha) - plays the role of Ybar


def test_pair_residuals_matches_hand_computation() -> None:
    D_triu, rho = pair_residuals(_D, _Y_CASE_A)
    np.testing.assert_allclose(D_triu, [1.0, 2.0, 5.0, 3.0, 4.0, 6.0])
    # d(Y_caseA): d01=1,d02=3,d03=4,d12=2,d13=3,d23=1
    np.testing.assert_allclose(rho, [0.0, -1.0, 1.0, 1.0, 1.0, 5.0])


def test_block_masks_near_mid_far() -> None:
    D_triu, _ = pair_residuals(_D, _Y_CASE_A)
    near, mid, far = block_masks(D_triu, _THETA, _M)
    np.testing.assert_array_equal(near, [True, False, False, False, False, False])
    np.testing.assert_array_equal(mid, [False, True, False, True, False, False])
    np.testing.assert_array_equal(far, [False, False, True, False, True, True])
    # P_far must have at least half of all N=6 pairs (03_metoda.tex, Notation)
    assert far.sum() >= D_triu.shape[0] / 2


def test_block_masks_rejects_theta_greater_than_m() -> None:
    with pytest.raises(ValueError):
        block_masks(np.array([1.0, 2.0]), theta=3.0, m=2.0)


def test_sigma_alpha_value_matches_hand_computation() -> None:
    sigma_a = sigma_alpha_value(_D, _Y_CASE_A, _ALPHA, _EPS_D)
    sigma_b = sigma_alpha_value(_D, _Y_CASE_B, _ALPHA, _EPS_D)
    assert sigma_a == pytest.approx(4.935908, abs=1e-4)
    assert sigma_b == pytest.approx(2.527473, abs=1e-4)


def test_phi_alpha_matches_hand_computation() -> None:
    D_triu = np.array([1.0, 2.0, 5.0, 3.0, 4.0, 6.0])
    phi = phi_alpha(D_triu, _THETA, _EPS_D, _ALPHA)
    expected = np.array([4 / 3, 0.8, 4 / 11, 4 / 7, 4 / 9, 4 / 13])
    np.testing.assert_allclose(phi, expected, atol=1e-9)


_EPS_D_ZERO_TOL = 1e-12  # matches config_experiments.yaml exp8_prop2_check.eps_d_zero_tol


def test_evaluate_proposition2_p2_fails_when_Y_is_worse() -> None:
    """Y=Y_caseA has a larger sigma_alpha than Ybar=Y_caseB -> P2 does not
    hold (it must not be silently discarded, it is just marked p2_holds=False)."""
    out = evaluate_proposition2(_D, _Y_CASE_A, _Y_CASE_B, _ALPHA, _EPS_D, _THETA, _M, _EPS_D_ZERO_TOL)
    assert out["p2_holds"] is False
    assert out["sigma_alpha_Y"] == pytest.approx(4.935908, abs=1e-4)
    assert out["sigma_alpha_Ybar"] == pytest.approx(2.527473, abs=1e-4)
    assert out["n_zero_pairs"] == 0
    assert out["has_duplicates"] is False


def test_evaluate_proposition2_bounds_hold_when_p2_holds() -> None:
    """Y=Y_caseB (smaller sigma_alpha) vs. Ybar=Y_caseA (larger sigma_alpha)
    -> P2 holds; both bounds (a) and (b) must hold and match the manual
    computation (see documentation/2026-09-14_exp8_prop2_check.md)."""
    out = evaluate_proposition2(_D, _Y_CASE_B, _Y_CASE_A, _ALPHA, _EPS_D, _THETA, _M, _EPS_D_ZERO_TOL)

    assert out["p2_holds"] is True

    # R_near/mid/far(Y=Y_caseB)
    assert out["R_near_Y"] == pytest.approx(0.0, abs=1e-9)
    assert out["R_mid_Y"] == pytest.approx(4.0, abs=1e-9)
    assert out["R_far_Y"] == pytest.approx(9.0, abs=1e-9)

    # R_near/mid/far(Ybar=Y_caseA)
    assert out["R_near_Ybar"] == pytest.approx(0.0, abs=1e-9)
    assert out["R_mid_Ybar"] == pytest.approx(2.0, abs=1e-9)
    assert out["R_far_Ybar"] == pytest.approx(27.0, abs=1e-9)

    assert out["r_eps"] == pytest.approx(0.5, abs=1e-9)
    assert out["kappa_eps"] == pytest.approx(4.0 / 3.0, abs=1e-9)

    assert out["bound_a"] == pytest.approx(9.871817, abs=1e-4)
    assert out["bound_b"] == pytest.approx(15.5, abs=1e-9)

    # both inequalities (a), (b) must hold when P2 holds
    assert out["R_near_Y"] <= out["bound_a"] + 1e-9
    assert out["R_near_Y"] <= out["bound_b"] + 1e-9
    assert out["slack_a"] >= -1e-9
    assert out["slack_b"] >= -1e-9

    # the predicted factor for P_far: r_eps^alpha = 0.5^1 = 0.5
    assert out["predicted_factor_r_eps_alpha"] == pytest.approx(0.5, abs=1e-9)


def test_evaluate_proposition2_rejects_alpha_zero() -> None:
    with pytest.raises(ValueError):
        evaluate_proposition2(_D, _Y_CASE_A, _Y_CASE_B, 0.0, _EPS_D, _THETA, _M, _EPS_D_ZERO_TOL)


# --- a manually computed example with a duplicate point (D_01 = 0) -----------------
# D_triu order (i<j): (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
# Point 0 and point 1 are identical (D_01=0) - theta/m are computed by hand
# the same way as in exp8_prop2_check.py (theta = median NN distance, m =
# median of all pairwise distances), so the pair (0,1) correctly falls into P_near (0 <= theta).
_D_DUP = np.array([
    [0.0, 0.0, 2.0, 5.0],
    [0.0, 0.0, 3.0, 4.0],
    [2.0, 3.0, 0.0, 6.0],
    [5.0, 4.0, 6.0, 0.0],
])
# NN distances: point0->0 (point1), point1->0 (point0), point2->2 (point0), point3->4 (point1)
# -> [0,0,2,4], median = 1.0; all pairwise distances [0,2,5,3,4,6], median = 3.5
_THETA_DUP = 1.0
_M_DUP = 3.5
_Y_DUP_A = np.array([[0.0], [0.2], [3.0], [4.0]])  # "worse" (larger sigma_alpha)
_Y_DUP_B = np.array([[0.0], [0.1], [2.0], [5.0]])  # "better" (smaller sigma_alpha) -> Ybar


def test_block_masks_zero_distance_pair_falls_in_p_near() -> None:
    """A pair with D_ij=0 must belong to P_near, because 0 <= theta
    (03_metoda.tex, Notation before Proposition 2) - an explicit test on a
    duplicate point (K/Q1 task item 3: verify P_near/P_mid/P_far for D_ij=0 pairs)."""
    D_triu, _ = pair_residuals(_D_DUP, _Y_DUP_A)
    np.testing.assert_allclose(D_triu, [0.0, 2.0, 5.0, 3.0, 4.0, 6.0])
    near, mid, far = block_masks(D_triu, _THETA_DUP, _M_DUP)
    # D_triu = [0.0(dup), 2.0, 5.0, 3.0, 4.0, 6.0], theta=1.0, m=3.5:
    # only the dup-pair (D=0.0) is <= theta -> P_near; the rest is split into mid/far by m.
    np.testing.assert_array_equal(near, [True, False, False, False, False, False])
    np.testing.assert_array_equal(mid, [False, True, False, True, False, False])
    np.testing.assert_array_equal(far, [False, False, True, False, True, True])


def test_evaluate_proposition2_duplicate_point_ok_with_positive_eps_d() -> None:
    """D_min=0 (points 0 and 1 identical) with eps_D > eps_d_zero_tol must
    pass without error - kappa_eps=(theta+eps_D)/(0+eps_D) is finite
    (regularization), exactly the point of `remark[role eps_D]`
    (03_metoda.tex). This is the main regression test for the 2026-09-14 fix
    (originally D_min<=0 always raised a ValueError regardless of eps_D)."""
    out = evaluate_proposition2(_D_DUP, _Y_DUP_B, _Y_DUP_A, _ALPHA, _EPS_D, _THETA_DUP, _M_DUP, _EPS_D_ZERO_TOL)
    assert out["d_min"] == pytest.approx(0.0, abs=1e-12)
    assert out["n_zero_pairs"] == 1
    assert out["has_duplicates"] is True
    # kappa_eps = (theta+eps_D)/(d_min+eps_D) = (1.0+0.5)/(0.0+0.5) = 3.0
    assert out["kappa_eps"] == pytest.approx(3.0, abs=1e-9)
    assert np.isfinite(out["kappa_eps"])
    assert np.isfinite(out["bound_a"])
    assert np.isfinite(out["bound_b"])


def test_evaluate_proposition2_duplicate_point_fails_when_eps_d_below_tolerance() -> None:
    """D_min=0 and eps_D <= eps_d_zero_tol (here directly eps_D=0, no
    regularization) must fail LOUDLY - kappa_eps would be undefined
    (division by zero), no silent fallback."""
    with pytest.raises(ValueError, match="zero off-diagonal distances"):
        evaluate_proposition2(_D_DUP, _Y_DUP_B, _Y_DUP_A, _ALPHA, eps_D=0.0, theta=_THETA_DUP, m=_M_DUP, eps_d_zero_tol=_EPS_D_ZERO_TOL)


def test_evaluate_proposition2_no_duplicates_flag_false() -> None:
    """A dataset without duplicates (the original _D) must have
    has_duplicates=False, n_zero_pairs=0, independent of eps_d_zero_tol."""
    out = evaluate_proposition2(_D, _Y_CASE_B, _Y_CASE_A, _ALPHA, _EPS_D, _THETA, _M, _EPS_D_ZERO_TOL)
    assert out["has_duplicates"] is False
    assert out["n_zero_pairs"] == 0


def test_block_residual_simple() -> None:
    rho = np.array([1.0, -2.0, 3.0])
    mask = np.array([True, False, True])
    assert block_residual(rho, mask) == pytest.approx(1.0 + 9.0)


def test_resolve_p2_holds_strict_and_tolerance() -> None:
    assert resolve_p2_holds(1.0, 2.0, tol_rel=0.0, tol_abs=0.0) is True
    assert resolve_p2_holds(2.0, 1.0, tol_rel=0.0, tol_abs=0.0) is False
    # a small excess within the tolerance is considered satisfied
    assert resolve_p2_holds(1.0001, 1.0, tol_rel=0.0, tol_abs=1e-3) is True
    assert resolve_p2_holds(1.1, 1.0, tol_rel=0.0, tol_abs=1e-3) is False
    with pytest.raises(ValueError):
        resolve_p2_holds(1.0, 1.0, tol_rel=-0.1, tol_abs=0.0)


def test_compute_alpha_bound_matches_hand_computation() -> None:
    # r_eps=0.5, tau=0.05 -> the smallest alpha with 0.5^alpha <= 0.05 is log2(20)=4.32 -> on the grid {..,4,5} it is 5
    grid = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert compute_alpha_bound(0.5, 0.05, grid) == pytest.approx(5.0)
    # r_eps=0.1, tau=0.05 -> 0.1^1=0.1 > 0.05, 0.1^2=0.01 <= 0.05 -> alpha_bound=2
    assert compute_alpha_bound(0.1, 0.05, grid) == pytest.approx(2.0)
    # beyond the grid range -> NaN (no extrapolation)
    assert np.isnan(compute_alpha_bound(0.99, 0.05, grid))
    with pytest.raises(ValueError):
        compute_alpha_bound(1.5, 0.05, grid)
    with pytest.raises(ValueError):
        compute_alpha_bound(0.5, 1.5, grid)
