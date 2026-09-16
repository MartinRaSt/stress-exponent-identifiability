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
Q1-step3 (reserse/2026-09-14_specifikace_rozsireni_q1.md, PART B.2) -
generators of synthetic scenarios with KNOWN latent geometry (ground truth),
for the calibrated metric-fidelity task
`src/experiments/exp9_metric_fidelity.py`. Unlike `src/datasets/synthetic.py`
(K2, registered datasets without known ground truth), these generators are
NOT in `src.datasets.registry` - they are called directly, returning
(X, y, truth), where `truth` contains the latent geometry needed by
`src/sammon/truth_metrics.py`.

Parameters of EVERY scenario come exclusively from
`config_experiments.yaml: exp9_metric_fidelity.scenarios.<name>` (no magic
constants here) - `cfg` is the already-resolved (quick/smoke-overridden)
dict for the given scenario, `seed` is the ALREADY-SUMMED generative seed
(SEED_BASE_s + r, see B.2) - the order of `rng` calls in each function is
part of the specification (determinism).
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def make_planar_clusters(seed: int, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """S1 "planar clusters" (B.2): K clusters with centers in the plane
    (e_1,e_2), a geometric progression of standard deviations, rejection
    sampling of the WHOLE center configuration (min. separation >=
    sep_factor*(sigma_k+sigma_l)).

    Returns (X (n,d), y (n,) cluster index, truth):
        truth['centroids_in'] (K,d) - sample centroids (from the realized data)
        truth['Delta']        (K,K) - centroid distance matrix (ground truth)
        truth['r']            (K,)  - RMS cluster radii (ground truth)
        truth['centers_latent'] (K,2) - latent (pre-noise) centers
        truth['sigma_latent']   (K,) - latent standard deviations
        truth['oracle_Y']     (n,2) - orthogonal projection onto (e_1,e_2)
    """
    K = int(cfg["K"])
    n_k = int(cfg["n_k"])
    sigma_min = float(cfg["sigma_min"])
    sigma_max = float(cfg["sigma_max"])
    center_range = float(cfg["center_range"])
    sep_factor = float(cfg["sep_factor"])
    max_attempts = int(cfg["max_separation_attempts"])
    d = int(cfg["d"])

    rng = _rng(seed)
    sigma_latent = sigma_min * (sigma_max / sigma_min) ** (np.arange(K) / (K - 1))

    centers_latent = None
    for _attempt in range(max_attempts):
        candidate = rng.uniform(-center_range, center_range, size=(K, 2))
        diff = candidate[:, None, :] - candidate[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        min_required = sep_factor * (sigma_latent[:, None] + sigma_latent[None, :])
        np.fill_diagonal(dist, np.inf)
        np.fill_diagonal(min_required, -np.inf)
        if np.all(dist >= min_required):
            centers_latent = candidate
            break
    if centers_latent is None:
        raise RuntimeError(
            f"make_planar_clusters: failed to find a separated configuration of K={K} "
            f"centers in {max_attempts} attempts (sep_factor={sep_factor}) - fail-loud, no substitute configuration."
        )

    X_parts = []
    y_parts = []
    for k in range(K):
        eps = rng.normal(size=(n_k, d))
        center_full = np.zeros(d)
        center_full[:2] = centers_latent[k]
        X_k = center_full[None, :] + sigma_latent[k] * eps
        X_parts.append(X_k)
        y_parts.append(np.full(n_k, k, dtype=np.int64))
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    centroids_in = np.stack([X[y == k].mean(axis=0) for k in range(K)], axis=0)
    diff_c = centroids_in[:, None, :] - centroids_in[None, :, :]
    Delta = np.sqrt(np.sum(diff_c ** 2, axis=-1))
    r = np.array([
        float(np.sqrt(np.mean(np.sum((X[y == k] - centroids_in[k]) ** 2, axis=-1))))
        for k in range(K)
    ])
    oracle_Y = X[:, :2].copy()

    truth = {
        "centroids_in": centroids_in, "Delta": Delta, "r": r,
        "centers_latent": centers_latent, "sigma_latent": sigma_latent, "oracle_Y": oracle_Y,
    }
    return X, y, truth


def make_tree_clusters(seed: int, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """S2 "binary tree clusters" (B.2): a balanced binary tree of depth
    `depth` -> 2**depth leaf clusters, level offsets a_l = a1*decay**(l-1).
    Every internal node v at level l draws a direction u_v uniformly on
    S^(d-1) (a normalized N(0,I_d)); children have centers mu_v +- a_l*u_v
    (order: minus first, then plus - see the tree construction below).

    Returns (X (n,d), y (n,) = leaf index 0..2**depth-1, truth):
        truth['u']            (L,L) - leaf ultrametric (L=2**depth)
        truth['Delta']        (L,L) - realized leaf-centroid distances
        truth['leaf_of_point'] (n,) - same as y (explicit key per B.7)
        truth['tree_levels']  (depth,) - a_l values used at each level
    """
    depth = int(cfg["depth"])
    n_leaf = int(cfg["n_leaf"])
    a1 = float(cfg["a1"])
    decay = float(cfg["decay"])
    sigma_leaf = float(cfg["sigma_leaf"])
    d = int(cfg["d"])

    rng = _rng(seed)
    a_levels = np.array([a1 * (decay ** (level - 1)) for level in range(1, depth + 1)])

    # tree construction level by level: centers[l] is an array (2**l, d) of
    # node centers at level l (0 = root); the node order in the array IS the
    # binary path (b1 = MSB), because every parent adds its two children in
    # order [minus, plus].
    centers = np.zeros((1, d))
    for level in range(1, depth + 1):
        n_parents = centers.shape[0]
        directions = rng.normal(size=(n_parents, d))
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        if np.any(norms <= 0):
            raise RuntimeError("make_tree_clusters: zero direction while sampling S^(d-1) (probability 0, check the seed).")
        directions = directions / norms
        a_l = a_levels[level - 1]
        new_centers = np.empty((2 * n_parents, d))
        new_centers[0::2] = centers - a_l * directions
        new_centers[1::2] = centers + a_l * directions
        centers = new_centers

    n_leaves = centers.shape[0]
    if n_leaves != 2 ** depth:
        raise RuntimeError(f"make_tree_clusters: expected {2**depth} leaves, got {n_leaves}.")

    X_parts = []
    y_parts = []
    for leaf in range(n_leaves):
        eps = rng.normal(size=(n_leaf, d))
        X_leaf = centers[leaf][None, :] + sigma_leaf * eps
        X_parts.append(X_leaf)
        y_parts.append(np.full(n_leaf, leaf, dtype=np.int64))
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    # ultrametric u_kl = 2*a_{l(k,l)}, l(k,l) = level of the first split
    # (position of the most significant differing bit of leaves k,l binary paths)
    leaf_idx = np.arange(n_leaves)
    xor = leaf_idx[:, None] ^ leaf_idx[None, :]
    u = np.zeros((n_leaves, n_leaves))
    for k in range(n_leaves):
        for l in range(n_leaves):
            if k == l:
                continue
            bit_len = int(xor[k, l]).bit_length()  # most significant set bit
            level_div = depth - bit_len + 1  # 1-indexed split level (1=root)
            u[k, l] = 2.0 * a_levels[level_div - 1]

    centroids = np.stack([X[y == leaf].mean(axis=0) for leaf in range(n_leaves)], axis=0)
    diff_c = centroids[:, None, :] - centroids[None, :, :]
    Delta = np.sqrt(np.sum(diff_c ** 2, axis=-1))

    truth = {"u": u, "Delta": Delta, "leaf_of_point": y.copy(), "tree_levels": a_levels}
    return X, y, truth


def make_bent_sheet(seed: int, cfg: dict[str, Any]) -> tuple[np.ndarray, None, dict[str, Any]]:
    """S3 "bent sheet" (B.2): a developable quarter-cylinder surface
    (intrinsic coordinates s~U(0,S), h~H*Beta(beta_a,beta_b)), embedded via
    orthogonal rotation into R^d + Gaussian noise.

    Returns (X (n,d), None, truth):
        truth['G']  (n,n) - exact geodesic (intrinsic) metric
        truth['T']  (n,2) - intrinsic coordinates (s,h) - `oracle_truth`
        truth['oracle_Y'] (n,2) - = T
    """
    n = int(cfg["n"])
    S = float(cfg["S"])
    H = float(cfg["H"])
    Theta = float(cfg["Theta_over_pi"]) * np.pi
    beta_a = float(cfg["beta_a"])
    beta_b = float(cfg["beta_b"])
    noise = float(cfg["noise"])
    d = int(cfg["d"])

    if Theta <= 0:
        raise ValueError(f"Theta must be positive, got Theta_over_pi={cfg['Theta_over_pi']}.")
    R = S / Theta

    rng = _rng(seed)
    s = rng.uniform(0.0, S, size=n)
    h = H * rng.beta(beta_a, beta_b, size=n)
    p = np.stack([R * np.sin(s / R), h, R * (1.0 - np.cos(s / R))], axis=1)

    A = rng.normal(size=(d, d))
    Q, Rmat = np.linalg.qr(A)
    Q = Q * np.sign(np.diag(Rmat))[None, :]
    eps = rng.normal(size=(n, d))

    p_full = np.zeros((n, d))
    p_full[:, :3] = p
    X = p_full @ Q.T + noise * eps

    ds = s[:, None] - s[None, :]
    dh = h[:, None] - h[None, :]
    G = np.sqrt(ds ** 2 + dh ** 2)
    T = np.stack([s, h], axis=1)

    truth = {"G": G, "T": T, "oracle_Y": T.copy()}
    return X, None, truth


def make_density_pair(seed: int, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """S4 "density pair" (B.2, exploratory/optional) - 2 clusters with the
    same latent sigma but a very different n (a test of neighbor methods
    equalizing cluster sizes given the same radius).

    Returns (X (n,d), y (n,) in {0,1}, truth): truth['r_ratio'] = realized
    RMS radius ratio r_1/r_2 (~1, see B.2 S4)."""
    n1 = int(cfg["n1"])
    n2 = int(cfg["n2"])
    sigma = float(cfg["sigma"])
    center_distance = float(cfg["center_distance"])
    d = int(cfg["d"])

    rng = _rng(seed)
    c1 = np.zeros(d)
    c2 = np.zeros(d)
    c2[0] = center_distance

    X1 = c1[None, :] + sigma * rng.normal(size=(n1, d))
    X2 = c2[None, :] + sigma * rng.normal(size=(n2, d))
    X = np.concatenate([X1, X2], axis=0)
    y = np.concatenate([np.zeros(n1, dtype=np.int64), np.ones(n2, dtype=np.int64)])

    mu1 = X1.mean(axis=0)
    mu2 = X2.mean(axis=0)
    r1 = float(np.sqrt(np.mean(np.sum((X1 - mu1) ** 2, axis=-1))))
    r2 = float(np.sqrt(np.mean(np.sum((X2 - mu2) ** 2, axis=-1))))
    if r2 <= 0:
        raise ValueError("make_density_pair: r2 <= 0 (degenerate cluster 2).")

    truth = {"r_ratio": r1 / r2}
    return X, y, truth


SCENARIO_GENERATORS = {
    "S1": make_planar_clusters,
    "S2": make_tree_clusters,
    "S3": make_bent_sheet,
    "S4": make_density_pair,
}
