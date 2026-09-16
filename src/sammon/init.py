# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-09
# License: see the LICENSE file in the repository root
"""Initialization strategies for Y0 (random, PCA, classical/Torgerson MDS).

All functions are seeded via the explicit `seed` parameter (no global
random state)."""
from __future__ import annotations

import numpy as np


def init_random(n: int, p: int, seed: int, scale: float = 1.0) -> np.ndarray:
    """Random initialization Y0 ~ N(0, scale^2), shape (n, p)."""
    if seed is None:
        raise ValueError("Seed must not be None.")
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=scale, size=(n, p))


def init_pca(X: np.ndarray, p: int, seed: int) -> np.ndarray:
    """Initialization with the first p principal components of point data
    X (n x d).

    Requires vector data (kind='vector'); for a distance matrix use
    `init_classical_mds`. PCA is deterministic up to the sign of the axes;
    the seed is passed for a uniform API and any small jitter needed for
    degenerate (n<=p) data.
    """
    if seed is None:
        raise ValueError("Seed must not be None.")
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    if d < p:
        raise ValueError(f"Number of input dimensions d={d} is smaller than the requested n_components={p}.")
    from sklearn.decomposition import PCA

    n_components_eff = min(p, n - 1) if n > 1 else p
    pca = PCA(n_components=n_components_eff, random_state=seed)
    Y0 = pca.fit_transform(X)
    if n_components_eff < p:
        # fill the missing columns with small random noise (e.g. n<=p, degenerate case)
        rng = np.random.default_rng(seed)
        extra = rng.normal(scale=1e-6, size=(n, p - n_components_eff))
        Y0 = np.hstack([Y0, extra])
    return Y0


def _resolve_iterative_min_n(iterative_min_n: int | None) -> int:
    """Threshold n above which `init_classical_mds` uses the iterative eigsh
    instead of the full eigh: an explicit value, otherwise
    `sammon.init.classical_mds_iterative_min_n` from config.yaml (a missing
    key is an error, no silent fallback)."""
    if iterative_min_n is not None:
        return int(iterative_min_n)
    from src.common.config import load_config

    try:
        return int(load_config()["sammon"]["init"]["classical_mds_iterative_min_n"])
    except KeyError as exc:
        raise KeyError("Missing key 'sammon.init.classical_mds_iterative_min_n' in config.yaml.") from exc


def _top_eigpairs_dense(D: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """Full `numpy.linalg.eigh` of the doubly-centered matrix
    B = -0.5 J D^2 J (O(n^3), ~3 temporary n x n matrices) - the original
    path, left UNCHANGED for n <= threshold (results for small datasets do
    not change, including the sign of eigenvectors, which is determined by
    LAPACK)."""
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    B = 0.5 * (B + B.T)  # numerical symmetrization

    eigvals, eigvecs = np.linalg.eigh(B)
    order = np.argsort(eigvals)[::-1]
    return eigvals[order][:p], eigvecs[:, order][:, :p]


def _top_eigpairs_iterative(D: np.ndarray, p: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """2026-09-13 (documentation/2026-09-13_hardening_behu.md): the p largest
    eigenpairs of B = -0.5 J D^2 J via `scipy.sparse.linalg.eigsh(k=p,
    which='LA')` over a `LinearOperator` - B is never materialized; the
    matvec B x = -0.5 * center(D^2 center(x)) is O(n^2) over a single copy
    of D^2 (no additional n x n matrix). The starting vector v0 is
    deterministic from `seed` (`np.random.default_rng(seed)`), tol=0 =
    machine precision (ARPACK), so the result does not depend on the number
    of BLAS threads (difference between 1 vs. 8 threads on pgp ~2e-15, see
    tests/test_init_iterative.py). Sign convention: the largest-magnitude
    component of each eigenvector is made positive (the eigh path does not
    normalize signs - determined by LAPACK; for Sammon/SMACOF, Y0 is
    equivalent up to reflection, stress is reflection-invariant).
    Measurement on pgp (n=10680, 1 BLAS thread): eigh ~4-5 min vs.
    eigsh ~3 s (results/data/quick/init_probe.csv)."""
    from scipy.sparse.linalg import LinearOperator, eigsh

    n = D.shape[0]
    if p >= n:
        raise ValueError(f"Iterative eigsh requires p={p} < n={n}.")
    D2 = D * D  # a single additional n x n copy

    def matvec(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).ravel()
        xc = x - x.mean()
        y = D2 @ xc
        y -= y.mean()
        return -0.5 * y

    op = LinearOperator((n, n), matvec=matvec, dtype=np.float64)
    v0 = np.random.default_rng(seed).standard_normal(n)
    eigvals, eigvecs = eigsh(op, k=p, which="LA", v0=v0, tol=0.0)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    # sign convention: the largest-magnitude component is positive
    idx_max = np.argmax(np.abs(eigvecs), axis=0)
    signs = np.sign(eigvecs[idx_max, np.arange(eigvecs.shape[1])])
    signs[signs == 0] = 1.0
    return eigvals, eigvecs * signs[None, :]


def init_classical_mds(D: np.ndarray, p: int, seed: int, iterative_min_n: int | None = None) -> np.ndarray:
    """Classical (Torgerson) MDS: double centering of -0.5*D^2 and a
    spectral decomposition on the p largest eigenvalues. Used as a
    warm-start init for distance/graph data where no point representation
    X is available.

    For n > `iterative_min_n` (default from config.yaml
    `sammon.init.classical_mds_iterative_min_n`), the iterative
    `_top_eigpairs_iterative` (eigsh over a LinearOperator, O(n^2) per
    matvec) is used instead of the full `_top_eigpairs_dense` (eigh O(n^3))
    - see both docstrings.
    """
    if seed is None:
        raise ValueError("Seed must not be None.")
    D = np.asarray(D, dtype=np.float64)
    n = D.shape[0]
    if D.shape != (n, n):
        raise ValueError(f"D must be square (n x n), got {D.shape}.")

    if n > _resolve_iterative_min_n(iterative_min_n):
        eigvals, eigvecs = _top_eigpairs_iterative(D, p, seed)
    else:
        eigvals, eigvecs = _top_eigpairs_dense(D, p)

    eigvals_pos = np.clip(eigvals, a_min=0.0, a_max=None)
    Y0 = eigvecs * np.sqrt(eigvals_pos)[None, :]

    n_neg_or_zero = int((eigvals <= 1e-12).sum())
    if n_neg_or_zero > 0:
        # some of the p largest eigenvalues are <=0 (the data is not fully
        # embeddable in Euclidean p-space) - fill in small random
        # coordinates so the optimization has somewhere to start from (no
        # metric is fabricated, it is just a warm start for the subsequent
        # gradient solver)
        rng = np.random.default_rng(seed)
        Y0[:, p - n_neg_or_zero:] += rng.normal(scale=1e-6, size=(n, n_neg_or_zero))
    return Y0
