# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-14
# License: see the LICENSE file in the repository root
"""Q1 step 3 (B.7 item 9): invariance under Y->s*R*Y+t (s>0, R orthogonal
incl. reflections, t a shift) < 1e-9 for every metric in
`src/sammon/truth_metrics.py`; the sigma_G closed form == a numerical
minimization; TA=1 for delta=u; beta_r=1 for rho=c*r."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import pdist, squareform

from src.sammon.truth_metrics import (
    aspect_ratio_error,
    centroid_pearson,
    centroid_spearman,
    cophenetic_pearson,
    geodesic_lre,
    geodesic_pearson,
    geodesic_stress_scale_inv,
    log_distortion,
    log_ratio_error,
    mds_upper_bound_cpcc,
    radius_lre,
    radius_slope,
    triplet_accuracy,
    upper_triangle_pairs,
)

_TOL = 1e-9


def _random_orthogonal(dim: int, seed: int, reflect: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(dim, dim))
    Q, R = np.linalg.qr(A)
    Q = Q * np.sign(np.diag(R))[None, :]
    if reflect:
        Q[:, 0] *= -1.0
    return Q


def _transform(Z: np.ndarray, s: float, Q: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (Z @ Q.T) * s + t[None, :]


@pytest.fixture(params=[False, True], ids=["rotation", "reflection"])
def transform_params(request):
    rng = np.random.default_rng(123)
    reflect = request.param
    s = float(rng.uniform(0.1, 10.0))
    Q = _random_orthogonal(2, seed=7 if not reflect else 8, reflect=reflect)
    t = rng.normal(size=2) * 100.0
    return s, Q, t


def test_centroid_pearson_spearman_lre_logdist_invariant(transform_params) -> None:
    s, Q, t = transform_params
    rng = np.random.default_rng(1)
    K = 8
    Y = rng.normal(size=(K, 2)) * 5.0
    Delta = squareform(pdist(rng.normal(size=(K, 2)) * 10.0 + 3.0))

    delta = squareform(pdist(Y))
    delta_t = squareform(pdist(_transform(Y, s, Q, t)))

    assert abs(centroid_pearson(delta, Delta) - centroid_pearson(delta_t, Delta)) < _TOL
    assert abs(centroid_spearman(delta, Delta) - centroid_spearman(delta_t, Delta)) < _TOL

    dp, Dp = upper_triangle_pairs(delta), upper_triangle_pairs(Delta)
    dpt = upper_triangle_pairs(delta_t)
    assert abs(log_ratio_error(dp, Dp) - log_ratio_error(dpt, Dp)) < _TOL
    assert abs(log_distortion(dp, Dp) - log_distortion(dpt, Dp)) < _TOL


def test_radius_slope_and_lre_invariant(transform_params) -> None:
    s, _Q, _t = transform_params
    rng = np.random.default_rng(2)
    rho = np.abs(rng.normal(size=8)) + 1.0
    r = np.abs(rng.normal(size=8)) + 1.0
    rho_scaled = rho * s
    assert abs(radius_slope(rho, r) - radius_slope(rho_scaled, r)) < _TOL
    assert abs(radius_lre(rho, r) - radius_lre(rho_scaled, r)) < _TOL


def test_cophenetic_pearson_and_triplet_accuracy_invariant(transform_params) -> None:
    s, Q, t = transform_params
    u = np.array([[0, 2, 2, 4], [2, 0, 2, 4], [2, 2, 0, 4], [4, 4, 4, 0]], dtype=float)
    rng = np.random.default_rng(3)
    Y = rng.normal(size=(4, 2)) * 3.0
    delta = squareform(pdist(Y))
    delta_t = squareform(pdist(_transform(Y, s, Q, t)))
    assert abs(cophenetic_pearson(delta, u) - cophenetic_pearson(delta_t, u)) < _TOL
    assert abs(triplet_accuracy(u, delta) - triplet_accuracy(u, delta_t)) < _TOL


def test_triplet_accuracy_is_one_when_delta_equals_u() -> None:
    u = np.array([[0, 2, 2, 4], [2, 0, 2, 4], [2, 2, 0, 4], [4, 4, 4, 0]], dtype=float)
    assert triplet_accuracy(u, u) == pytest.approx(1.0)


def test_radius_slope_is_one_for_proportional_radii() -> None:
    rng = np.random.default_rng(4)
    r = np.abs(rng.normal(size=10)) + 1.0
    c = 3.7
    rho = c * r
    assert radius_slope(rho, r) == pytest.approx(1.0, abs=1e-8)


def test_geodesic_metrics_invariant(transform_params) -> None:
    s, Q, t = transform_params
    rng = np.random.default_rng(5)
    n = 30
    P = rng.normal(size=(n, 2))
    G = squareform(pdist(P))
    base = P + rng.normal(size=(n, 2)) * 0.1
    d = squareform(pdist(base))
    d_t = squareform(pdist(_transform(base, s, Q, t)))

    assert abs(geodesic_stress_scale_inv(d, G) - geodesic_stress_scale_inv(d_t, G)) < _TOL
    assert abs(geodesic_pearson(d, G) - geodesic_pearson(d_t, G)) < _TOL
    assert abs(geodesic_lre(d, G) - geodesic_lre(d_t, G)) < _TOL


def test_geodesic_stress_scale_inv_closed_form_matches_numeric_minimization() -> None:
    rng = np.random.default_rng(6)
    n = 25
    P = rng.normal(size=(n, 2))
    G = squareform(pdist(P))
    d = squareform(pdist(P + rng.normal(size=(n, 2)) * 0.2))
    dp, Gp = upper_triangle_pairs(d), upper_triangle_pairs(G)

    res = minimize_scalar(lambda scale: float(np.sum((scale * dp - Gp) ** 2)))
    sigma_numeric = float(np.sum((res.x * dp - Gp) ** 2) / np.sum(Gp ** 2))
    sigma_closed = geodesic_stress_scale_inv(d, G)
    assert abs(sigma_closed - sigma_numeric) < 1e-8


def test_aspect_ratio_error_invariant(transform_params) -> None:
    s, Q, t = transform_params
    rng = np.random.default_rng(9)
    T = rng.normal(size=(60, 2)) * np.array([5.0, 1.0])
    base = T + rng.normal(size=(60, 2)) * 0.01
    base_t = _transform(base, s, Q, t)
    assert abs(aspect_ratio_error(base, T) - aspect_ratio_error(base_t, T)) < _TOL


def test_mds_upper_bound_cpcc_deterministic() -> None:
    u = np.array([[0, 2, 2, 4], [2, 0, 2, 4], [2, 2, 0, 4], [4, 4, 4, 0]], dtype=float)
    v1 = mds_upper_bound_cpcc(u)
    v2 = mds_upper_bound_cpcc(u)
    assert v1 == pytest.approx(v2)
    assert -1.0 <= v1 <= 1.0 + 1e-9


def test_log_ratio_error_requires_positive_values() -> None:
    with pytest.raises(ValueError):
        log_ratio_error(np.array([1.0, -1.0]), np.array([1.0, 1.0]))
