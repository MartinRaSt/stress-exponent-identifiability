# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-13
# License: see the LICENSE file in the repository root
"""Tests for the iterative path of `init_classical_mds` (documentation/
2026-09-13_hardening_behu.md, task 3): eigsh over a LinearOperator vs. a
full eigh at n=1500 (above the threshold set in the test) - Y0 agrees after
sign alignment (rtol 1e-6), the stress after SMACOF agrees, the result does
not depend on the number of BLAS threads, the threshold is read from the config."""
from __future__ import annotations

import numpy as np
import pytest
import threadpoolctl
from scipy.spatial.distance import pdist, squareform

from src.common.config import load_config
from src.sammon.init import _resolve_iterative_min_n, init_classical_mds
from src.sammon.init_probe import align_signs
from src.sammon.solvers.smacof import smacof_solve
from src.sammon.weights import alpha_weights, compute_Z_from_W, estimate_eps_D

N = 1500
P = 2
SEED = 3
THRESHOLD_IN_TEST = 1000  # n=1500 > threshold -> the iterative path
FORCE_DENSE = 10 ** 9


@pytest.fixture(scope="module")
def D() -> np.ndarray:
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(N, 6)) @ np.diag([5.0, 3.0, 1.0, 0.5, 0.3, 0.1])
    return squareform(pdist(X))


@pytest.fixture(scope="module")
def Y0_pair(D):
    Y_dense = init_classical_mds(D, P, SEED, iterative_min_n=FORCE_DENSE)
    Y_iter = init_classical_mds(D, P, SEED, iterative_min_n=THRESHOLD_IN_TEST)
    return Y_dense, Y_iter


def test_iterative_matches_dense_after_sign_alignment(Y0_pair) -> None:
    Y_dense, Y_iter = Y0_pair
    assert Y_iter.shape == (N, P)
    np.testing.assert_allclose(align_signs(Y_iter, Y_dense), Y_dense, rtol=1e-6, atol=1e-6 * np.abs(Y_dense).max())


def test_iterative_sign_convention_largest_abs_component_positive(Y0_pair) -> None:
    _, Y_iter = Y0_pair
    idx = np.argmax(np.abs(Y_iter), axis=0)
    assert np.all(Y_iter[idx, np.arange(P)] > 0)


def test_stress_after_smacof_identical(D, Y0_pair) -> None:
    """Reflecting Y0 does not change the SMACOF trajectory (stress is
    reflection-invariant) - the final stress from both inits must agree."""
    Y_dense, Y_iter = Y0_pair
    cfg = load_config()["sammon"]
    eps_D = estimate_eps_D(D, k=cfg["eps_D"]["k"], q=cfg["eps_D"]["q"], kind="distance")
    W = alpha_weights(D, 1.0, eps_D)
    Z = compute_Z_from_W(D, W)
    kw = dict(max_iter=20, tol=1e-6, eps_num=cfg["eps_num"], dense_pinv_threshold=cfg["smacof"]["dense_pinv_threshold"],
              cg_max_iter=cfg["smacof"]["cg_max_iter"], cg_tol=cfg["smacof"]["cg_tol"], reg_rho=cfg["smacof"]["reg_rho"])
    _, h_dense = smacof_solve(D, W, Z, Y_dense, **kw)
    _, h_iter = smacof_solve(D, W, Z, Y_iter, **kw)
    assert h_dense["stress"][-1] == pytest.approx(h_iter["stress"][-1], rel=1e-6)


def test_iterative_independent_of_blas_threads(D) -> None:
    with threadpoolctl.threadpool_limits(limits=1):
        Y1 = init_classical_mds(D, P, SEED, iterative_min_n=THRESHOLD_IN_TEST)
    with threadpoolctl.threadpool_limits(limits=4):
        Y4 = init_classical_mds(D, P, SEED, iterative_min_n=THRESHOLD_IN_TEST)
    np.testing.assert_allclose(Y1, Y4, rtol=1e-8, atol=1e-10)


def test_threshold_from_config_and_small_n_uses_dense() -> None:
    thr = _resolve_iterative_min_n(None)
    assert thr == int(load_config()["sammon"]["init"]["classical_mds_iterative_min_n"]) and thr > 0
    # a small n (below the config threshold) -> the result matches the original eigh path exactly
    rng = np.random.default_rng(0)
    D_small = squareform(pdist(rng.normal(size=(60, 4))))
    Y_default = init_classical_mds(D_small, P, 0)
    Y_dense = init_classical_mds(D_small, P, 0, iterative_min_n=FORCE_DENSE)
    np.testing.assert_array_equal(Y_default, Y_dense)


def test_iterative_rejects_p_ge_n() -> None:
    D_tiny = np.abs(np.random.default_rng(0).normal(size=(3, 3)))
    D_tiny = 0.5 * (D_tiny + D_tiny.T)
    np.fill_diagonal(D_tiny, 0.0)
    with pytest.raises(ValueError):
        init_classical_mds(D_tiny, 3, 0, iterative_min_n=1)
