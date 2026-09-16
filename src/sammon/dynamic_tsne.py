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
External baseline for E4 (temporal networks): "Dynamic t-SNE" (Rauber,
Falcao, Telea 2016, "Visualizing Time-Dependent Data Using Dynamic t-SNE",
EuroVis Short Papers, DOI 10.2312/eurovisshort.20161164, key
'rauber2016dynamic' in clanek/references.bib).

Custom numpy implementation (no dependency on openTSNE/sklearn TSNE, which
do not allow a custom objective function with a temporal penalty): a
sequence of t-SNE embeddings Y^(t), each initialized from Y^(t-1) for nodes
shared with the previous snapshot (warm start), with objective function

    L_t(Y^(t)) = KL_t(Y^(t)) + (lambda_dt / (2 N_t)) * sum_{i in common(t-1,t)} ||y_i^(t) - y_i^(t-1)||^2

where KL_t is the standard t-SNE cost (Kullback-Leibler divergence
P_t || Q(Y^(t))), N_t is the number of nodes of snapshot t, and P_t is
computed from the SAME input distance matrix D_t (shortest-path, see
src/experiments/exp4_temporal.py::_load_snapshots) as used by our
TemporalSammon (SMACOF/SGD family) - a fair comparison on identical input.

New nodes (without a previous position) are initialized with the same
scheme as in `src.sammon.temporal.TemporalSammon.fit` (K14 jitter): the
centroid of the nearest continuing neighbors + small seeded jitter,
otherwise random initialization. This file does NOT import or modify
`src/sammon/temporal.py` (developed concurrently by another agent) - the
logic is deliberately duplicated here with its own configuration constants
(`sammon.temporal.dtsne` in common/config.yaml).
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.spatial.distance import pdist, squareform

from src.sammon.init import init_classical_mds
from src.sammon.metrics import scale_invariant_stress
from src.sammon.temporal_metrics import stability

logger = logging.getLogger("sammon.dynamic_tsne")


# ---------------------------------------------------------------------------
# P matrix (perplexity binary search) - same definition as classic t-SNE
# (van der Maaten & Hinton, 2008), uses D^2 directly (precomputed metric).
# ---------------------------------------------------------------------------

def _binary_search_perplexity(
    D_sq: np.ndarray, perplexity: float, tol: float, max_iter: int,
) -> np.ndarray:
    """For each row, find beta_i=1/(2 sigma_i^2) such that the entropy of
    the conditional Gaussian P_{.|i} matches log(perplexity). Returns the
    conditional (unsymmetrized) matrix P_{j|i} (diagonal 0)."""
    n = D_sq.shape[0]
    if perplexity <= 0 or perplexity >= n:
        raise ValueError(
            f"perplexity={perplexity} must be in (0, n={n}) - the snapshot has too few nodes for "
            "the requested perplexity. Lower 'methods.tsne.perplexity' or raise min_snapshot_nodes."
        )
    target_entropy = np.log(perplexity)
    P = np.zeros((n, n), dtype=np.float64)
    mask_row = ~np.eye(n, dtype=bool)

    for i in range(n):
        d_i = D_sq[i, mask_row[i]]
        # Log-sum-exp shift by d_i.min() (standard numerical stabilization,
        # e.g. sklearn.manifold._utils._binary_search_perplexity uses an
        # analogous trick): the entropy H=log(sum_j exp(-beta*d_j))+beta*E[d]
        # is invariant to a constant shift d_j -> d_j-c (see the derivation
        # in documentation/2026-09-14_e4_dtsne_baseline.md), so the resulting
        # normalized P_{.|i} is UNCHANGED - the shift only prevents
        # exp(-beta*d_i) from underflowing to exactly 0.0 at large beta.
        # Without this shift, graph-distance inputs with small integer
        # distances and many ties at the minimum (typical for shortest-path
        # matrices of contact networks, see the K12 smoke run of
        # primary_school_temporal t=1, 35 nodes sharing minimum distance 1)
        # require beta large enough that exp(-beta*d_min) itself underflows
        # to 0.0 (sum_p==0.0) before the binary search converges - the
        # fail-loud exception below would then trigger even on entirely
        # regular data.
        d_shift = float(d_i.min())
        d_shifted = d_i - d_shift
        beta_min, beta_max = -np.inf, np.inf
        beta = 1.0
        p_i = None
        for _ in range(max_iter):
            p_i = np.exp(-d_shifted * beta)
            sum_p = p_i.sum()
            if sum_p <= 0.0 or not np.isfinite(sum_p):
                raise ValueError(
                    f"Row {i}: the sum of Gaussian weights is infinite/zero (beta={beta}) - "
                    "the distance matrix contains invalid (NaN/Inf) values."
                )
            entropy = np.log(sum_p) + beta * (d_shifted * p_i).sum() / sum_p
            p_i = p_i / sum_p
            diff = entropy - target_entropy
            if abs(diff) < tol:
                break
            if diff > 0.0:
                beta_min = beta
                beta = beta * 2.0 if not np.isfinite(beta_max) else (beta + beta_max) / 2.0
            else:
                beta_max = beta
                beta = beta / 2.0 if not np.isfinite(beta_min) else (beta + beta_min) / 2.0
        P[i, mask_row[i]] = p_i
    return P


def compute_p_matrix(D: np.ndarray, perplexity: float, tol: float, max_iter: int, eps_num: float) -> np.ndarray:
    """Symmetrized P matrix (n x n) from the distance matrix D (section 2 of
    t-SNE, van der Maaten & Hinton 2008): P_ij = (P_{j|i} + P_{i|j}) / (2n),
    diagonal 0, lower-clipped to `eps_num` (numerical stabilization of KL)."""
    D = np.asarray(D, dtype=np.float64)
    n = D.shape[0]
    D_sq = D ** 2
    P_cond = _binary_search_perplexity(D_sq, perplexity, tol, max_iter)
    P = (P_cond + P_cond.T) / (2.0 * n)
    np.fill_diagonal(P, 0.0)
    P = np.maximum(P, eps_num)
    np.fill_diagonal(P, 0.0)
    return P


# ---------------------------------------------------------------------------
# KL(P||Q) cost + gradient (Student-t kernel) + quadratic temporal penalty
# ---------------------------------------------------------------------------

def _student_t_q(Y: np.ndarray, eps_num: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (Q, inv_1pd2) - Q is the normalized Student-t affinity,
    inv_1pd2 = 1/(1+||y_i-y_j||^2) (diagonal 0), also needed for the gradient."""
    diff = Y[:, None, :] - Y[None, :, :]
    dist_sq = (diff ** 2).sum(-1)
    inv = 1.0 / (1.0 + dist_sq)
    np.fill_diagonal(inv, 0.0)
    Z = inv.sum()
    if Z <= 0.0:
        raise ValueError("Sum of Student-t affinities is <= 0 - Y is degenerate (all points coincide).")
    Q = inv / Z
    Q = np.maximum(Q, eps_num)
    np.fill_diagonal(Q, 0.0)
    return Q, inv


def kl_cost(Y: np.ndarray, P: np.ndarray, eps_num: float) -> float:
    """KL(P || Q(Y)), with P having a zero diagonal (see `compute_p_matrix`)."""
    Q, _ = _student_t_q(Y, eps_num)
    mask = P > 0.0
    return float((P[mask] * np.log(P[mask] / Q[mask])).sum())


def kl_gradient(Y: np.ndarray, P: np.ndarray, eps_num: float) -> np.ndarray:
    """Analytic gradient dKL/dY (n x p), standard t-SNE form
    grad_i = 4 * sum_j (p_ij - q_ij) * (1+||y_i-y_j||^2)^{-1} * (y_i - y_j)."""
    Q, inv = _student_t_q(Y, eps_num)
    coeff = (P - Q) * inv
    diff = Y[:, None, :] - Y[None, :, :]
    return 4.0 * (coeff[:, :, None] * diff).sum(axis=1)


def anchor_penalty_cost(Y: np.ndarray, anchors: np.ndarray, mask: np.ndarray, lam_dt: float, n_t: int) -> float:
    """(lambda_dt / (2 N_t)) * sum_i mask_i ||y_i - a_i||^2."""
    if lam_dt <= 0.0:
        return 0.0
    diff = (Y - anchors) * mask[:, None]
    return float(lam_dt / (2.0 * n_t) * (diff ** 2).sum())


def anchor_penalty_gradient(Y: np.ndarray, anchors: np.ndarray, mask: np.ndarray, lam_dt: float, n_t: int) -> np.ndarray:
    """Gradient of the above penalty: (lambda_dt / N_t) * mask_i * (y_i - a_i)."""
    if lam_dt <= 0.0:
        return np.zeros_like(Y)
    return (lam_dt / n_t) * mask[:, None] * (Y - anchors)


def cost_and_grad(
    Y: np.ndarray, P: np.ndarray, anchors: np.ndarray, mask: np.ndarray, lam_dt: float, n_t: int, eps_num: float,
) -> tuple[float, np.ndarray]:
    """Full objective function L_t(Y) = KL_t(Y) + temporal penalty, and its
    gradient - a single place used both for the finite-difference test and
    the optimizer."""
    cost = kl_cost(Y, P, eps_num) + anchor_penalty_cost(Y, anchors, mask, lam_dt, n_t)
    grad = kl_gradient(Y, P, eps_num) + anchor_penalty_gradient(Y, anchors, mask, lam_dt, n_t)
    return cost, grad


# ---------------------------------------------------------------------------
# Optimization of a single snapshot (momentum gradient descent, van der
# Maaten & Hinton 2008 - adaptive gains, early exaggeration, momentum switch).
# ---------------------------------------------------------------------------

def _resolve_learning_rate(n: int, early_exaggeration: float, learning_rate_min: float) -> float:
    """sklearn TSNE 'auto' formula: max(n / early_exaggeration / 4, learning_rate_min)."""
    return max(n / early_exaggeration / 4.0, learning_rate_min)


def _optimize_frame(
    Y0: np.ndarray, P: np.ndarray, anchors: np.ndarray, mask: np.ndarray, lam_dt: float,
    max_iter: int, apply_exaggeration: bool, cfg: dict,
) -> tuple[np.ndarray, float]:
    """Momentum gradient descent for a single snapshot (van der Maaten &
    Hinton 2008 gain/momentum scheme, explicit step) for the KL part of the
    objective, with a NON-IMPLICIT (proximal) step for the quadratic
    temporal penalty.

    Reason for the proximal/implicit step (forward-backward splitting): for
    large lambda_dt the quadratic term has high curvature
    (Hessian ~ lambda_dt/N_t), and an explicit gradient step with the (van
    der Maaten) fixed learning_rate designed for the KL part diverges
    (instability for eta*curvature > 2, verified empirically - see
    tests/test_dynamic_tsne.py::test_large_lambda_does_not_diverge). The
    proximal step for the quadratic term is exact (closed form) and
    unconditionally stable for any lambda_dt >= 0; the KL part remains the
    standard explicit scheme (nonconvex, gain/momentum heuristic)."""
    n = Y0.shape[0]
    eps_num = cfg["eps_num"]
    early_exaggeration = cfg["early_exaggeration"]
    early_exaggeration_iter = cfg["early_exaggeration_iter"]
    momentum_init = cfg["momentum_init"]
    momentum_final = cfg["momentum_final"]
    momentum_switch_iter = cfg["momentum_switch_iter"]
    min_gain = cfg["min_gain"]
    learning_rate = _resolve_learning_rate(n, early_exaggeration, cfg["learning_rate_min"])
    w_anchor = (lam_dt / n) * mask[:, None]  # (n,1), 0 outside continuing nodes or when lam_dt=0

    Y = Y0.copy()
    velocity = np.zeros_like(Y)
    gains = np.ones_like(Y)

    for it in range(max_iter):
        momentum = momentum_init if it < momentum_switch_iter else momentum_final
        P_used = P * early_exaggeration if (apply_exaggeration and it < early_exaggeration_iter) else P
        grad_kl = kl_gradient(Y, P_used, eps_num)

        sign_flip = np.sign(grad_kl) != np.sign(velocity)
        gains = np.where(sign_flip, gains + 0.2, gains * 0.8)
        gains = np.clip(gains, min_gain, None)

        eta = learning_rate * gains  # effective step of the explicit KL step (per node x dimension)
        velocity = momentum * velocity - eta * grad_kl
        Y_provisional = Y + velocity

        if lam_dt > 0.0:
            # closed form of the proximal step: minimizes
            # 0.5/eta*(y-Y_provisional)^2 + w_anchor*(y-anchor)^2 over y
            Y = (Y_provisional / eta + w_anchor * anchors) / (1.0 / eta + w_anchor)
        else:
            Y = Y_provisional

    kl_final = kl_cost(Y, P, eps_num)
    return Y, kl_final


VALID_DTSNE_KEYS = (
    "early_exaggeration", "early_exaggeration_iter", "momentum_init", "momentum_final",
    "momentum_switch_iter", "learning_rate_min", "min_gain", "eps_num", "perplexity_tol",
    "perplexity_max_iter", "new_node_k_near", "new_node_jitter_scale", "new_node_random_scale",
)


class DynamicTSNE:
    """Dynamic t-SNE (Rauber, Falcao, Telea 2016) over a sequence of
    snapshots D_t.

    `fit(list_of_D, node_ids_per_snapshot, lam_dt)` returns `self`, storing
    `self.Y_list_`, `self.node_ids_`, `self.history_list_` - the same API
    shape as `src.sammon.temporal.TemporalSammon`, so `exp4_temporal.py` can
    share most of the evaluation code (stability/quality) between the two
    method families.
    """

    def __init__(self, n_components: int, perplexity: float, max_iter: int, seed: int, dtsne_cfg: dict) -> None:
        missing = [k for k in VALID_DTSNE_KEYS if k not in dtsne_cfg]
        if missing:
            raise KeyError(f"dtsne_cfg is missing keys {missing} (see common/config.yaml sammon.temporal.dtsne).")
        self.n_components = n_components
        self.perplexity = perplexity
        self.max_iter = max_iter
        self.seed = seed
        self.cfg = dict(dtsne_cfg)

    def fit(self, list_of_D: list[np.ndarray], node_ids_per_snapshot: list[list], lam_dt: float) -> "DynamicTSNE":
        if len(list_of_D) != len(node_ids_per_snapshot):
            raise ValueError("list_of_D and node_ids_per_snapshot must have the same length (number of snapshots).")

        Y_list: list[np.ndarray] = []
        history_list: list[dict] = []
        prev_Y: np.ndarray | None = None
        prev_ids: list | None = None

        for t, (D_t, ids_t) in enumerate(zip(list_of_D, node_ids_per_snapshot)):
            D_t = np.asarray(D_t, dtype=np.float64)
            n_t = D_t.shape[0]
            if len(ids_t) != n_t:
                raise ValueError(f"Snapshot t={t}: number of node ids ({len(ids_t)}) does not match D.shape[0]={n_t}.")

            P = compute_p_matrix(D_t, self.perplexity, self.cfg["perplexity_tol"], self.cfg["perplexity_max_iter"], self.cfg["eps_num"])

            if prev_Y is None:
                Y0 = init_classical_mds(D_t, self.n_components, self.seed + t)
                mask = np.zeros(n_t, dtype=np.float64)
                anchors = np.zeros((n_t, self.n_components), dtype=np.float64)
                lam_t = 0.0
                apply_exaggeration = True
            else:
                id_to_prev_idx = {nid: i for i, nid in enumerate(prev_ids)}
                mask = np.zeros(n_t, dtype=np.float64)
                anchors = np.zeros((n_t, self.n_components), dtype=np.float64)
                Y0 = np.zeros((n_t, self.n_components), dtype=np.float64)
                rng = np.random.default_rng(self.seed + t)

                continuing_local = []
                for i, nid in enumerate(ids_t):
                    if nid in id_to_prev_idx:
                        mask[i] = 1.0
                        anchors[i] = prev_Y[id_to_prev_idx[nid]]
                        Y0[i] = anchors[i]
                        continuing_local.append(i)

                # new nodes: centroid of continuing neighbors (+ jitter),
                # otherwise random init - scheme identical to
                # src/sammon/temporal.py K14, but with its own configuration
                # constants (dtsne_cfg)
                new_idx = [i for i in range(n_t) if mask[i] == 0.0]
                if new_idx:
                    scale = float(D_t[~np.eye(n_t, dtype=bool)].mean()) if n_t > 1 else 1.0
                    k_near_cfg = int(self.cfg["new_node_k_near"])
                    jitter_scale = float(self.cfg["new_node_jitter_scale"])
                    random_scale = float(self.cfg["new_node_random_scale"])
                    for i in new_idx:
                        if continuing_local:
                            dists_to_continuing = D_t[i, continuing_local]
                            k_near = min(k_near_cfg, len(continuing_local))
                            nearest = np.array(continuing_local)[np.argsort(dists_to_continuing)[:k_near]]
                            jitter = rng.normal(scale=jitter_scale * scale, size=self.n_components)
                            Y0[i] = Y0[nearest].mean(axis=0) + jitter
                        else:
                            Y0[i] = rng.normal(scale=random_scale * scale, size=self.n_components)

                if mask.sum() == 0.0:
                    logger.warning(
                        "Snapshot t=%d has no node in common with the previous snapshot (mask.sum()==0) - "
                        "using lam_t=0.0 (no anchors).", t,
                    )
                    lam_t = 0.0
                else:
                    lam_t = lam_dt
                apply_exaggeration = False

            Y_t, kl_final = _optimize_frame(Y0, P, anchors, mask, lam_t, self.max_iter, apply_exaggeration, self.cfg)

            # 'stress' here is the scale-invariant stress (src.sammon.metrics.
            # scale_invariant_stress(D, d), same definition as E1
            # 'stress_scale_invariant') - t-SNE has no concept of alpha
            # weighting, so for a fair comparison with the alpha-Sammon
            # family the UNWEIGHTED scale-invariant metric is used (see
            # documentation/2026-09-14_e4_dtsne_baseline.md). Stored under
            # the key 'stress' (not 'kl') so that
            # `exp4_temporal.py::_per_transition_metrics` and the
            # `TemporalSammon`-compatible `.quality()` below work unchanged
            # for both method families.
            d_t = squareform(pdist(Y_t))
            stress_si = scale_invariant_stress(D_t, d_t)
            Y_list.append(Y_t)
            history_list.append({"stress": [stress_si], "kl": [kl_final], "n_t": n_t})
            prev_Y = Y_t
            prev_ids = ids_t

        self.Y_list_ = Y_list
        self.node_ids_ = node_ids_per_snapshot
        self.history_list_ = history_list
        self.lam_dt_ = lam_dt
        return self

    def stability(self, procrustes_align: bool = True) -> float:
        """Same definition as `TemporalSammon.stability()` (a thin wrapper
        around the canonical `src.sammon.temporal_metrics.stability`)."""
        if not hasattr(self, "Y_list_"):
            raise RuntimeError("Call fit() first.")
        return stability(self.Y_list_, self.node_ids_, procrustes_align=procrustes_align)

    def quality(self) -> float:
        """Median scale-invariant stress over all snapshots (see the
        comment on storing 'stress' in `fit()` above)."""
        if not hasattr(self, "history_list_"):
            raise RuntimeError("Call fit() first.")
        vals = [hist["stress"][-1] for hist in self.history_list_]
        return float(np.median(vals))
