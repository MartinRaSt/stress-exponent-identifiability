# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-09
# License: see the LICENSE file in the repository root
"""
The weight family w_ij = D_ij^{-alpha} for alpha-Sammon (see
reserse/2026-09-09_specifikace_metody.md, sections 1 and 3.3a).

All functions operate on the full (n x n) distance matrix D (symmetric,
zero diagonal). The diagonal of the resulting weight matrix is always 0
(the pair (i,i) is never used).
"""
from __future__ import annotations

import numpy as np


def _validate_D(D: np.ndarray) -> np.ndarray:
    """Check the shape and nonnegativity of the distance matrix, return a float64 copy."""
    D = np.asarray(D, dtype=np.float64)
    n = D.shape[0]
    if D.ndim != 2 or D.shape != (n, n):
        raise ValueError(f"The distance matrix must be square (n x n), got shape {D.shape}.")
    if np.any(D < 0):
        raise ValueError("The distance matrix contains negative values - invalid input.")
    return D


def estimate_eps_D(D_or_X: np.ndarray, k: int, q: float, kind: str = "distance") -> float:
    """Estimate the regularization constant eps_D as the q-quantile of the
    distance to the k-th nearest neighbor over all points (section 3.3a of
    the spec).

    Parameters:
        D_or_X: a distance matrix (kind='distance') or point data
            (kind='vector'); Euclidean distance is used for 'vector'.
        k: neighbor rank (e.g. k=5 -> the 5th nearest neighbor).
        q: quantile of the distribution {d(x_i, kNN_k(x_i))}, q in [0, 1].
        kind: 'distance' or 'vector'.

    Returns: a positive real number eps_D (if the result would be <=0, this
    is an input error - all points would have to be identical).
    """
    if kind == "vector":
        from scipy.spatial.distance import pdist, squareform

        D = squareform(pdist(np.asarray(D_or_X, dtype=np.float64), metric="euclidean"))
    elif kind == "distance":
        D = _validate_D(D_or_X)
    else:
        raise ValueError(f"Unknown kind='{kind}' for estimate_eps_D (expected 'distance' or 'vector').")

    n = D.shape[0]
    if k < 1 or k > n - 1:
        raise ValueError(f"k={k} must be in range [1, n-1={n - 1}].")
    if not (0.0 <= q <= 1.0):
        raise ValueError(f"q={q} must be in [0, 1].")

    D_masked = D.copy()
    np.fill_diagonal(D_masked, np.inf)
    # k-th smallest value in each row (0-based index k-1 after sorting)
    knn_k_dist = np.partition(D_masked, kth=k - 1, axis=1)[:, k - 1]
    eps_D = float(np.quantile(knn_k_dist, q))
    if eps_D <= 0:
        raise ValueError(
            "Estimated eps_D is <= 0 (too many identical/close points for the given k,q). "
            "A fallback value is not fabricated - check the input data or lower k/q."
        )
    return eps_D


def alpha_weights(D: np.ndarray, alpha: float, eps_D: float) -> np.ndarray:
    """Compute the weight matrix w_ij = (D_ij + eps_D)^{-alpha}, diagonal = 0.

    The `+eps_D` regularization (section 3.3a) prevents an explosion of
    weights for nearly-coincident points (D_ij -> 0) when alpha > 0; for
    alpha == 0 it always gives w=1 regardless of eps_D, which matches the
    definition of E_0 (Kruskal's stress-1^2).
    """
    D = _validate_D(D)
    if alpha < 0:
        raise ValueError(f"alpha must be >= 0, got {alpha}.")
    if eps_D <= 0:
        raise ValueError(f"eps_D must be positive, got {eps_D}.")

    if alpha == 0.0:
        W = np.ones_like(D)
    else:
        W = np.power(D + eps_D, -alpha)
    np.fill_diagonal(W, 0.0)
    return W


def compute_Z(D: np.ndarray, alpha: float) -> float:
    """Compute the normalization constant Z = sum_{i<j} D_ij^{-alpha} D_ij^2
    (without eps_D regularization - Z is defined directly from the original
    definition of E_alpha, section 1.1; for D_ij=0 the contribution to Z is
    zero in the limit for alpha<2, otherwise Z would diverge - such cases
    (duplicate points) report an error).
    """
    D = _validate_D(D)
    n = D.shape[0]
    mask = ~np.eye(n, dtype=bool)
    Dm = D[mask]
    if alpha == 0.0:
        w = np.ones_like(Dm)
    else:
        if np.any((Dm == 0) & (alpha >= 2.0)):
            raise ValueError(
                "D contains zero off-diagonal distances (duplicate points) and alpha>=2 - "
                "Z would diverge. Remove duplicates or use alpha_weights with eps_D."
            )
        with np.errstate(divide="ignore"):
            w = np.where(Dm > 0, np.power(Dm, -alpha), 0.0)
    Z = float((w * Dm ** 2).sum() / 2.0)
    if Z <= 0:
        raise ValueError("Z <= 0 - the normalized stress E_alpha is not defined for this data.")
    return Z


def compute_Z_from_W(D: np.ndarray, W: np.ndarray) -> float:
    """Compute the normalization constant Z directly from an already-built
    weight matrix W (Z = sum_{i<j} W_ij D_ij^2). Used everywhere W is
    regularized (e.g. `alpha_weights` with eps_D) - Z thus remains
    consistent with the objective actually being minimized, even for data
    with exact duplicates (D_ij=0), where the "pure" definition
    Z = sum D_ij^{-alpha} D_ij^2 (see `compute_Z`) would diverge for
    alpha>=2."""
    D = _validate_D(D)
    W = np.asarray(W, dtype=np.float64)
    if W.shape != D.shape:
        raise ValueError(f"W and D must have the same shape, got W={W.shape}, D={D.shape}.")
    n = D.shape[0]
    mask = ~np.eye(n, dtype=bool)
    Z = float((W[mask] * D[mask] ** 2).sum() / 2.0)
    if Z <= 0:
        raise ValueError("Z <= 0 - the normalized stress E_alpha is not defined for this data/weights.")
    return Z


def multiscale_weights(
    D: np.ndarray, alphas: list[float], eps_D: float, C: float = 1.0
) -> tuple[np.ndarray, dict[float, float]]:
    """Multi-scale weight mixture w_ij = sum_k c_k * (D_ij+eps_D)^{-alpha_k} (section 5.2).

    c_k = C / sum_{i<j} D_ij^{2-alpha_k} normalizes each level so it
    contributes the same constant C to the stress "budget".

    Returns (W, c_by_alpha) - the resulting weight matrix and a dict of the
    coefficients c_k used, for diagnostics/logging.
    """
    D = _validate_D(D)
    n = D.shape[0]
    mask = ~np.eye(n, dtype=bool)
    if C <= 0:
        raise ValueError(f"C must be positive, got {C}.")

    W = np.zeros_like(D)
    c_by_alpha: dict[float, float] = {}
    for alpha_k in alphas:
        Dm = D[mask]
        if alpha_k == 2.0:
            denom = float(mask.sum())
        else:
            with np.errstate(divide="ignore"):
                denom = float(np.power(Dm, 2.0 - alpha_k).sum())
        if denom <= 0:
            raise ValueError(f"Normalization denominator for alpha_k={alpha_k} is <= 0 - invalid data.")
        c_k = C / denom
        c_by_alpha[alpha_k] = c_k
        Wk = alpha_weights(D, alpha_k, eps_D)
        W = W + c_k * Wk
    np.fill_diagonal(W, 0.0)
    return W, c_by_alpha
