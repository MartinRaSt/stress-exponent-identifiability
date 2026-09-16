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
Stress E_alpha, gradient, diagonal Hessian, and optimal scaling s*
(reserse/2026-09-09_specifikace_metody.md, section 1). Each function has a
numpy and a torch variant with the same signature; dispatch is done based
on the type of the array `Y` (np.ndarray -> numpy branch, torch.Tensor ->
torch branch).

Sum normalization convention: the full (n x n) off-diagonal mask is used
(each pair i!=j is summed twice, once as (i,j) and once as (j,i)). Because
both the numerator and denominator of all ratios (E_alpha, s*) are doubled
this way simultaneously, the resulting values are identical to the
definition over i<j (section 1.1) - this style is chosen for easy
vectorization (no triu indexing) and consistency with `src/sammon/metrics.py`.
"""
from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is always installed in this project
    torch = None  # type: ignore


def _is_torch(x: Any) -> bool:
    return torch is not None and isinstance(x, torch.Tensor)


# ---------------------------------------------------------------------------
# numpy branch
# ---------------------------------------------------------------------------

def _pairwise_distances_np(Y: np.ndarray, eps_num: float) -> np.ndarray:
    diff = Y[:, None, :] - Y[None, :, :]
    d = np.sqrt((diff ** 2).sum(-1))
    return np.where(d < eps_num, eps_num, d)


def _stress_alpha_np(D: np.ndarray, Y: np.ndarray, W: np.ndarray, Z: float, eps_num: float) -> float:
    n = D.shape[0]
    mask = ~np.eye(n, dtype=bool)
    d = _pairwise_distances_np(Y, eps_num)
    sigma = float((mask * W * (D - d) ** 2).sum() / 2.0)
    return sigma / Z


def _optimal_scale_np(D: np.ndarray, d: np.ndarray, W: np.ndarray) -> float:
    n = D.shape[0]
    mask = ~np.eye(n, dtype=bool)
    num = float((mask * W * D * d).sum())
    den = float((mask * W * d ** 2).sum())
    if den <= 0:
        raise ValueError("Denominator of the optimal scaling s* is <= 0 - Y is degenerate (all points coincide).")
    return num / den


def _scale_invariant_stress_np(D: np.ndarray, Y: np.ndarray, W: np.ndarray, Z: float, eps_num: float) -> float:
    d = _pairwise_distances_np(Y, eps_num)
    s_star = _optimal_scale_np(D, d, W)
    n = D.shape[0]
    mask = ~np.eye(n, dtype=bool)
    sigma = float((mask * W * (D - s_star * d) ** 2).sum() / 2.0)
    return sigma / Z


def _gradient_np(D: np.ndarray, Y: np.ndarray, W: np.ndarray, Z: float, eps_num: float) -> np.ndarray:
    n, p = Y.shape
    mask = ~np.eye(n, dtype=bool)
    diff = Y[:, None, :] - Y[None, :, :]  # (n, n, p)
    d = _pairwise_distances_np(Y, eps_num)
    coeff = np.where(mask, W * (D - d) / d, 0.0)  # (n, n)
    grad = -2.0 / Z * (coeff[:, :, None] * diff).sum(axis=1)
    return grad


def _diag_hessian_np(D: np.ndarray, Y: np.ndarray, W: np.ndarray, Z: float, eps_num: float) -> np.ndarray:
    n, p = Y.shape
    mask = ~np.eye(n, dtype=bool)
    diff = Y[:, None, :] - Y[None, :, :]  # (n, n, p)
    d = _pairwise_distances_np(Y, eps_num)
    bracket = (1.0 - D / d)[:, :, None] + D[:, :, None] * diff ** 2 / d[:, :, None] ** 3
    hess = 2.0 / Z * (np.where(mask, W, 0.0)[:, :, None] * bracket).sum(axis=1)
    return hess


# ---------------------------------------------------------------------------
# torch branch
# ---------------------------------------------------------------------------

def _pairwise_distances_torch(Y: "torch.Tensor", eps_num: float):
    diff = Y[:, None, :] - Y[None, :, :]
    d = torch.sqrt((diff ** 2).sum(-1))
    return torch.clamp(d, min=eps_num)


def _mask_torch(n: int, device, dtype):
    eye = torch.eye(n, device=device, dtype=torch.bool)
    return ~eye


def _stress_alpha_torch(D, Y, W, Z: float, eps_num: float) -> float:
    n = D.shape[0]
    mask = _mask_torch(n, Y.device, Y.dtype).to(D.dtype)
    d = _pairwise_distances_torch(Y, eps_num)
    sigma = (mask * W * (D - d) ** 2).sum() / 2.0
    return float((sigma / Z).item())


def _optimal_scale_torch(D, d, W) -> float:
    n = D.shape[0]
    mask = _mask_torch(n, D.device, D.dtype).to(D.dtype)
    num = (mask * W * D * d).sum()
    den = (mask * W * d ** 2).sum()
    if float(den.item()) <= 0:
        raise ValueError("Denominator of the optimal scaling s* is <= 0 - Y is degenerate (all points coincide).")
    return float((num / den).item())


def _scale_invariant_stress_torch(D, Y, W, Z: float, eps_num: float) -> float:
    d = _pairwise_distances_torch(Y, eps_num)
    s_star = _optimal_scale_torch(D, d, W)
    n = D.shape[0]
    mask = _mask_torch(n, D.device, D.dtype).to(D.dtype)
    sigma = (mask * W * (D - s_star * d) ** 2).sum() / 2.0
    return float((sigma / Z).item())


def _gradient_torch(D, Y, W, Z: float, eps_num: float):
    n, p = Y.shape
    mask = _mask_torch(n, Y.device, Y.dtype).to(Y.dtype)
    diff = Y[:, None, :] - Y[None, :, :]
    d = _pairwise_distances_torch(Y, eps_num)
    coeff = mask * W * (D - d) / d
    grad = -2.0 / Z * (coeff[:, :, None] * diff).sum(dim=1)
    return grad


def _diag_hessian_torch(D, Y, W, Z: float, eps_num: float):
    n, p = Y.shape
    mask = _mask_torch(n, Y.device, Y.dtype).to(Y.dtype)
    diff = Y[:, None, :] - Y[None, :, :]
    d = _pairwise_distances_torch(Y, eps_num)
    bracket = (1.0 - D / d)[:, :, None] + D[:, :, None] * diff ** 2 / d[:, :, None] ** 3
    hess = 2.0 / Z * ((mask * W)[:, :, None] * bracket).sum(dim=1)
    return hess


# ---------------------------------------------------------------------------
# public interface (dispatch by type of Y)
# ---------------------------------------------------------------------------

def pairwise_distances(Y, eps_num: float):
    """Euclidean distances between rows of Y (n x n), clipped below at eps_num."""
    if _is_torch(Y):
        return _pairwise_distances_torch(Y, eps_num)
    return _pairwise_distances_np(np.asarray(Y, dtype=np.float64), eps_num)


def stress_alpha(D, Y, W, Z: float, eps_num: float) -> float:
    """Normalized stress E_alpha(Y) = sigma(Y)/Z (sections 1.1, 2.1)."""
    if _is_torch(Y):
        return _stress_alpha_torch(D, Y, W, Z, eps_num)
    return _stress_alpha_np(np.asarray(D, dtype=np.float64), np.asarray(Y, dtype=np.float64), np.asarray(W, dtype=np.float64), Z, eps_num)


def optimal_scale(D, d, W) -> float:
    """Optimal scaling s* = argmin_s E_alpha(s*Y) in closed form (section 1.4)."""
    if _is_torch(d):
        return _optimal_scale_torch(D, d, W)
    return _optimal_scale_np(np.asarray(D, dtype=np.float64), np.asarray(d, dtype=np.float64), np.asarray(W, dtype=np.float64))


def scale_invariant_stress(D, Y, W, Z: float, eps_num: float) -> float:
    """E_alpha^{scale-inv}(Y) = E_alpha(s* Y) (section 1.4) - used for
    cross-method comparison (embeddings from other methods have an
    arbitrary scale)."""
    if _is_torch(Y):
        return _scale_invariant_stress_torch(D, Y, W, Z, eps_num)
    return _scale_invariant_stress_np(np.asarray(D, dtype=np.float64), np.asarray(Y, dtype=np.float64), np.asarray(W, dtype=np.float64), Z, eps_num)


def gradient(D, Y, W, Z: float, eps_num: float):
    """Analytic gradient dE_alpha/dY (n x p), see section 1.2."""
    if _is_torch(Y):
        return _gradient_torch(D, Y, W, Z, eps_num)
    return _gradient_np(np.asarray(D, dtype=np.float64), np.asarray(Y, dtype=np.float64), np.asarray(W, dtype=np.float64), Z, eps_num)


def diag_hessian(D, Y, W, Z: float, eps_num: float):
    """Diagonal approximation of the Hessian (pseudo-Newton, section 1.3), shape (n x p)."""
    if _is_torch(Y):
        return _diag_hessian_torch(D, Y, W, Z, eps_num)
    return _diag_hessian_np(np.asarray(D, dtype=np.float64), np.asarray(Y, dtype=np.float64), np.asarray(W, dtype=np.float64), Z, eps_num)
