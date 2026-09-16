# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-09
# License: see the LICENSE file in the repository root
"""Tests for section 10.1: the analytic gradient of E_alpha vs. a central
numerical derivative on a small synthetic dataset, across several seeds and alpha values."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

from src.sammon.init import init_random
from src.sammon.stress import diag_hessian, gradient, stress_alpha
from src.sammon.weights import alpha_weights, compute_Z_from_W, estimate_eps_D


def _make_problem(seed: int, alpha: float, n: int = 20, p: int = 2):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    D = squareform(pdist(X))
    eps_D = estimate_eps_D(D, k=5, q=0.1, kind="distance")
    W = alpha_weights(D, alpha, eps_D)
    Z = compute_Z_from_W(D, W)
    Y = init_random(n, p, seed=seed + 100, scale=0.5)
    return D, W, Z, Y


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("alpha", [0.0, 1.0, 2.0])
def test_gradient_matches_numerical_derivative(seed: int, alpha: float) -> None:
    """Central differences (h=1e-6) vs. the analytic gradient, rtol=1e-4 (section 10.1)."""
    D, W, Z, Y = _make_problem(seed, alpha)
    eps_num = 1e-9
    g_analytic = gradient(D, Y, W, Z, eps_num)

    h = 1e-6
    g_numeric = np.zeros_like(Y)
    n, p = Y.shape
    for i in range(n):
        for k in range(p):
            Yp = Y.copy(); Yp[i, k] += h
            Ym = Y.copy(); Ym[i, k] -= h
            g_numeric[i, k] = (
                stress_alpha(D, Yp, W, Z, eps_num) - stress_alpha(D, Ym, W, Z, eps_num)
            ) / (2 * h)

    np.testing.assert_allclose(g_analytic, g_numeric, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("alpha", [0.0, 1.0, 2.0])
def test_diag_hessian_matches_numerical_second_derivative(alpha: float) -> None:
    """The diagonal Hessian (section 1.3) vs. the numerical second derivative of the stress."""
    D, W, Z, Y = _make_problem(seed=3, alpha=alpha)
    eps_num = 1e-9
    h_analytic = diag_hessian(D, Y, W, Z, eps_num)

    h = 1e-4
    n, p = Y.shape
    h_numeric = np.zeros_like(Y)
    for i in range(n):
        for k in range(p):
            Yp = Y.copy(); Yp[i, k] += h
            Ym = Y.copy(); Ym[i, k] -= h
            f0 = stress_alpha(D, Y, W, Z, eps_num)
            fp = stress_alpha(D, Yp, W, Z, eps_num)
            fm = stress_alpha(D, Ym, W, Z, eps_num)
            h_numeric[i, k] = (fp - 2 * f0 + fm) / (h ** 2)

    np.testing.assert_allclose(h_analytic, h_numeric, rtol=2e-2, atol=1e-3)


def test_alpha_weights_alpha0_is_uniform() -> None:
    """alpha=0 must give w_ij=1 for all off-diagonal pairs (Kruskal stress-1^2, section 1.1)."""
    rng = np.random.default_rng(0)
    D = squareform(pdist(rng.normal(size=(15, 4))))
    W = alpha_weights(D, 0.0, eps_D=1.0)
    mask = ~np.eye(15, dtype=bool)
    assert np.allclose(W[mask], 1.0)
    assert np.allclose(np.diag(W), 0.0)


def test_estimate_eps_D_positive_and_reasonable() -> None:
    rng = np.random.default_rng(0)
    D = squareform(pdist(rng.normal(size=(50, 4))))
    eps_D = estimate_eps_D(D, k=5, q=0.1, kind="distance")
    assert eps_D > 0
    assert eps_D < D.max()
