# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-14
# License: see the LICENSE file in the repository root
"""Q1 step 3 (B.7 item 9): determinism of the generators + checks of the
latent geometry (S1: Delta from the oracle_Y projection vs. Delta from the
input to within 1% relative, separation satisfied; S2: u has exactly 4
values, every triplet has a unique nearest pair; S3: G symmetric, G_ij >=
||p_i-p_j|| and G_ij <= ||p_i-p_j||/0.9003 (with noise tolerance),
sigma_G(oracle) < 1e-6 without noise)."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

from src.datasets.synthetic_truth import (
    make_bent_sheet,
    make_density_pair,
    make_planar_clusters,
    make_tree_clusters,
)
from src.sammon.truth_metrics import geodesic_stress_scale_inv

_S1_CFG = dict(K=8, n_k=60, sigma_min=0.5, sigma_max=2.0, center_range=25.0, sep_factor=4.0, max_separation_attempts=10000, d=10)
_S2_CFG = dict(depth=4, n_leaf=30, a1=16.0, decay=0.5, sigma_leaf=0.5, d=10)
_S3_CFG = dict(n=400, S=3.0, H=1.0, Theta_over_pi=0.5, beta_a=2.0, beta_b=5.0, noise=0.02, d=10)
_S4_CFG = dict(n1=40, n2=360, sigma=1.0, center_distance=20.0, d=10)


def test_make_planar_clusters_deterministic() -> None:
    X1, y1, t1 = make_planar_clusters(90100, _S1_CFG)
    X2, y2, t2 = make_planar_clusters(90100, _S1_CFG)
    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)
    assert np.allclose(t1["Delta"], t2["Delta"])


def test_make_planar_clusters_separation_and_oracle_projection() -> None:
    X, y, truth = make_planar_clusters(90100, _S1_CFG)
    K = _S1_CFG["K"]
    sigma_min, sigma_max = _S1_CFG["sigma_min"], _S1_CFG["sigma_max"]
    sigma_latent = sigma_min * (sigma_max / sigma_min) ** (np.arange(K) / (K - 1))
    centers = truth["centers_latent"]
    diff = centers[:, None, :] - centers[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    min_required = _S1_CFG["sep_factor"] * (sigma_latent[:, None] + sigma_latent[None, :])
    np.fill_diagonal(dist, np.inf)
    np.fill_diagonal(min_required, -np.inf)
    assert np.all(dist >= min_required)

    oracle_Y = truth["oracle_Y"]
    centroids_2d = np.stack([oracle_Y[y == k].mean(axis=0) for k in range(K)], axis=0)
    Delta_2d = squareform(pdist(centroids_2d))
    Delta = truth["Delta"]
    iu = np.triu_indices(K, k=1)
    rel_err = np.abs(Delta_2d[iu] - Delta[iu]) / Delta[iu]
    assert np.max(rel_err) < 0.01


def test_make_tree_clusters_deterministic_and_ultrametric_structure() -> None:
    X1, y1, t1 = make_tree_clusters(90200, _S2_CFG)
    X2, y2, t2 = make_tree_clusters(90200, _S2_CFG)
    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)

    u = t1["u"]
    n_leaves = u.shape[0]
    assert n_leaves == 2 ** _S2_CFG["depth"]
    distinct = sorted(set(u[np.triu_indices(n_leaves, k=1)].tolist()))
    assert len(distinct) == _S2_CFG["depth"]

    # every triplet has a unique (strict) nearest pair according to u
    for i, j, k in combinations(range(n_leaves), 3):
        vals = (u[i, j], u[i, k], u[j, k])
        m = min(vals)
        assert sum(1 for v in vals if v == m) == 1


def test_make_bent_sheet_deterministic_and_geodesic_bounds() -> None:
    X1, y1, t1 = make_bent_sheet(90300, _S3_CFG)
    X2, y2, t2 = make_bent_sheet(90300, _S3_CFG)
    assert y1 is None and y2 is None
    assert np.array_equal(X1, X2)
    assert np.allclose(t1["G"], t2["G"])

    G = t1["G"]
    assert np.allclose(G, G.T)
    ambient_d = squareform(pdist(X1))
    chord_arc_ratio_min = 0.9003
    # noise tolerance: the difference of two independent 10-dimensional noise
    # vectors (each N(0,noise^2*I_10)) has a norm on the order of
    # noise*sqrt(2*10)~0.09; for n=400 (79800 pairs) we use a generous
    # multiplicative margin so the test is not statistically flaky
    # (deterministic seed 90300, see the empirically measured max difference
    # ~0.12 at this seed)
    tol = 15.0 * _S3_CFG["noise"]
    assert np.all(ambient_d >= chord_arc_ratio_min * G - tol)
    assert np.all(ambient_d <= G + tol)


def test_make_bent_sheet_oracle_sigma_g_near_zero_without_noise() -> None:
    cfg = dict(_S3_CFG)
    cfg["noise"] = 0.0
    X, y, truth = make_bent_sheet(90300, cfg)
    T = truth["oracle_Y"]
    d_oracle = squareform(pdist(T))
    sigma = geodesic_stress_scale_inv(d_oracle, truth["G"])
    assert sigma < 1e-6


def test_make_density_pair_deterministic_and_shape() -> None:
    X1, y1, t1 = make_density_pair(90400, _S4_CFG)
    X2, y2, t2 = make_density_pair(90400, _S4_CFG)
    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)
    assert X1.shape == (_S4_CFG["n1"] + _S4_CFG["n2"], _S4_CFG["d"])
    assert t1["r_ratio"] == pytest.approx(t2["r_ratio"])
    assert t1["r_ratio"] > 0
