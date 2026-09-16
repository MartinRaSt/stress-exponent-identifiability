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
Reference implementation of the classical Sammon projection (Sammon, 1969):
"A Nonlinear Mapping for Data Structure Analysis", IEEE Trans. Computers.

Uses the original pseudo-Newton update rule (diagonal Hessian approximation)
with a fixed step alpha. The implementation is O(n^2) per iteration and
serves as the reference baseline against which the proposed improvement is
compared (see src/sammon/, to be added later).
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from src.methods.common import method_config, to_distance_matrix
from src.methods.registry import Method, register_method


def sammon_mapping(
    D: np.ndarray,
    n_components: int,
    max_iter: int,
    alpha: float,
    tol: float,
    eps: float,
    seed: int,
    step_halving: bool = True,
    max_halvings: int = 10,
) -> tuple[np.ndarray, list[float]]:
    """Classical Sammon algorithm (pseudo-Newton) over the distance matrix D.

    Returns (Y, stress_history), where Y is the (n x n_components) resulting
    configuration and stress_history is a list of Sammon stress E values
    across iterations (for convergence diagnostics).
    """
    n = D.shape[0]
    if D.shape != (n, n):
        raise ValueError(f"Distance matrix must be square, got shape {D.shape}.")

    Dstar = np.asarray(D, dtype=np.float64).copy()
    if np.any(Dstar < 0):
        raise ValueError("Distance matrix contains negative values - invalid input.")
    np.fill_diagonal(Dstar, 0.0)

    mask = ~np.eye(n, dtype=bool)
    c = Dstar[mask].sum() / 2.0
    if c <= 0:
        raise ValueError("Sum of all original distances is 0 - Sammon stress is undefined.")
    Dstar_safe = np.where(Dstar < eps, eps, Dstar)

    rng = np.random.default_rng(seed)
    mean_scale = Dstar[mask].mean()
    Y = rng.normal(loc=0.0, scale=0.1 * mean_scale, size=(n, n_components))

    stress_history: list[float] = []
    prev_stress = None

    for _ in range(max_iter):
        diff_full = Y[:, None, :] - Y[None, :, :]  # (n, n, q)
        d = np.sqrt((diff_full ** 2).sum(-1))
        d_safe = np.where(d < eps, eps, d)

        delta = Dstar - d  # (n, n)
        stress = float((mask * (delta ** 2) / Dstar_safe).sum() / (2.0 * c))
        stress_history.append(stress)

        base = np.where(mask, 1.0 / (Dstar_safe * d_safe), 0.0)  # (n, n)

        # pseudo-Newton step direction (diagonal Hessian) - computed once;
        # any step-halving only rescales the same direction
        step = np.zeros_like(Y)
        for q in range(n_components):
            diff_q = Y[:, q][:, None] - Y[:, q][None, :]  # (n, n)
            grad_q = -2.0 / c * (mask * base * delta * diff_q).sum(axis=1)

            bracket = delta - (diff_q ** 2 / d_safe) * (1.0 + delta / d_safe)
            hess_q = -2.0 / c * (mask * base * bracket).sum(axis=1)
            hess_q_safe = np.where(np.abs(hess_q) < eps, -eps, hess_q)

            step[:, q] = grad_q / np.abs(hess_q_safe)

        # step-halving (2026-09-11): if stress increases after a step, the step
        # is reverted and alpha is halved (max. max_halvings times) - without
        # this, pseudo-Newton with a fixed MF=0.3 diverges on high-dimensional
        # data (E1: cnae9, isolet)
        alpha_cur = alpha
        n_halvings = 0
        while True:
            Y_new = Y - alpha_cur * step
            if not step_halving or n_halvings >= max_halvings:
                break
            d_new = np.sqrt(((Y_new[:, None, :] - Y_new[None, :, :]) ** 2).sum(-1))
            stress_new = float((mask * ((Dstar - d_new) ** 2) / Dstar_safe).sum() / (2.0 * c))
            if stress_new <= stress:
                break
            alpha_cur *= 0.5
            n_halvings += 1

        Y = Y_new

        if prev_stress is not None and abs(prev_stress - stress) < tol * max(prev_stress, eps):
            break
        prev_stress = stress

    return Y, stress_history


class SammonClassicMethod(Method):
    name = "sammon_classic"
    accepts = "both"

    def fit_transform(self, data: np.ndarray | nx.Graph, kind: str, seed: int, n_components: int = 2) -> np.ndarray:
        cfg = method_config("sammon_classic")
        D = to_distance_matrix(data, kind)
        Y, _ = sammon_mapping(
            D,
            n_components=n_components,
            max_iter=cfg["max_iter"],
            alpha=cfg["alpha"],
            tol=cfg["tol"],
            eps=cfg["eps"],
            seed=seed,
            step_halving=bool(cfg["step_halving"]),
            max_halvings=int(cfg["max_halvings"]),
        )
        return Y


register_method("sammon_classic")(SammonClassicMethod)
