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
Temporal Sammon: anchor-based regularization of stability between
consecutive snapshots of a dynamic network
(reserse/2026-09-09_specifikace_metody.md, section 6). The anchoring term
is incorporated directly into SMACOF (modified Guttman step
(V+lambda*M)Y = B(Y)Y + lambda*M*A, section 6.2).
"""
from __future__ import annotations

import logging
import time

import numpy as np

from src.sammon.init import init_classical_mds, init_random
from src.sammon.solvers.sgd import sgd_solve
from src.sammon.solvers.smacof import smacof_solve
from src.sammon.stress import scale_invariant_stress
from src.sammon.temporal_metrics import stability as _stability_metric
from src.sammon.weights import alpha_weights, compute_Z_from_W, estimate_eps_D

VALID_TEMPORAL_SOLVERS = {"smacof", "sgd"}

logger = logging.getLogger("sammon.temporal")


class TemporalSammon:
    """Temporal alpha-Sammon over a sequence of snapshots D_t (section 6).

    fit(list_of_D, node_ids_per_snapshot, lam, alpha, ...) returns a list
    of embeddings Y_t and stores them in `self.Y_list_`, `self.node_ids_`.
    """

    def __init__(
        self,
        n_components: int = 2,
        max_iter: int = 300,
        tol: float = 1.0e-4,
        eps_num: float = 1.0e-9,
        eps_D_k: int = 5,
        eps_D_q: float = 0.1,
        dense_pinv_threshold: int = 5000,
        cg_max_iter: int = 500,
        cg_tol: float = 1.0e-6,
        reg_rho: float = 1.0e-8,
        seed: int = 0,
        solver: str = "smacof",
        sgd_epochs: int = 100,
        sgd_mu_max: float = 0.5,
        sgd_pairs_per_node: int = 60,
        sgd_eps_anneal: float = 0.01,
        sgd_variant: str = "stabilized",
        gamma_mode: str = "eta_t",
        gamma_fixed: float = 0.1,
    ) -> None:
        if solver not in VALID_TEMPORAL_SOLVERS:
            raise ValueError(f"Unknown solver='{solver}' (expected one of {sorted(VALID_TEMPORAL_SOLVERS)}).")
        if gamma_mode not in ("eta_t", "fixed"):
            raise ValueError(f"Unknown gamma_mode='{gamma_mode}' (expected 'eta_t' or 'fixed').")
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.eps_num = eps_num
        self.eps_D_k = eps_D_k
        self.eps_D_q = eps_D_q
        self.dense_pinv_threshold = dense_pinv_threshold
        self.cg_max_iter = cg_max_iter
        self.cg_tol = cg_tol
        self.reg_rho = reg_rho
        self.seed = seed
        self.solver = solver
        self.sgd_epochs = sgd_epochs
        self.sgd_mu_max = sgd_mu_max
        self.sgd_pairs_per_node = sgd_pairs_per_node
        self.sgd_eps_anneal = sgd_eps_anneal
        self.sgd_variant = sgd_variant
        self.gamma_mode = gamma_mode
        self.gamma_fixed = gamma_fixed

    def fit(
        self,
        list_of_D: list[np.ndarray],
        node_ids_per_snapshot: list[list],
        lam: float,
        alpha: float,
    ) -> "TemporalSammon":
        """Fit each snapshot t=0..T-1 sequentially.

        t=0: plain SMACOF (lam=0, no anchors - nothing to anchor to).
        t>0: nodes shared with t-1 get an anchor a_i=y_i(t-1) and m_i=1;
        new nodes (m_i=0) are initialized at the centroid of neighbors that
        already existed at t-1 (randomly if none exist - section 6.4).
        """
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

            eps_D = estimate_eps_D(D_t, k=self.eps_D_k, q=self.eps_D_q, kind="distance")
            W = alpha_weights(D_t, alpha, eps_D)
            Z = compute_Z_from_W(D_t, W)

            if prev_Y is None:
                Y0 = init_classical_mds(D_t, self.n_components, self.seed + t)
                mask = np.zeros(n_t, dtype=np.float64)
                anchors = np.zeros((n_t, self.n_components), dtype=np.float64)
                lam_t = 0.0
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

                # new nodes: centroid of neighbors that already existed at
                # t-1 (via D_t - "neighbor" = one of the k nearest points in
                # D_t with m_i=1); if no such neighbors exist, random
                # initialization (section 6.4)
                new_idx = [i for i in range(n_t) if mask[i] == 0.0]
                if new_idx:
                    scale = float(D_t[~np.eye(n_t, dtype=bool)].mean()) if n_t > 1 else 1.0
                    for i in new_idx:
                        if continuing_local:
                            dists_to_continuing = D_t[i, continuing_local]
                            k_near = min(5, len(continuing_local))
                            nearest = np.array(continuing_local)[np.argsort(dists_to_continuing)[:k_near]]
                            # K14 (2026-09-13, documentation/2026-09-13_k14_uklid_hardening.md):
                            # when a snapshot has few nodes (typically < ~15)
                            # and only a HANDFUL of continuing ones (in the
                            # extreme case exactly 1, see invs13_temporal
                            # t=15, 13 nodes), ALL new nodes without jitter
                            # would get an IDENTICAL Y0 (the mean of the same
                            # single-element set). The SGD/Gauss-Seidel
                            # pairwise update then uses a direction vector
                            # (Y_i-Y_j) that is zero for exactly matching
                            # coordinates - no force ever separates the
                            # points again and the whole snapshot collapses
                            # to a single point (stability() then correctly
                            # fails loud with "mean pairwise distance <= 0").
                            # A small seeded deviation from the centroid
                            # (order 1e-3*scale, an optimizer initialization
                            # detail, not a fabricated result) breaks this
                            # numerical degeneracy - see
                            # test_new_node_jitter_breaks_exact_duplicate_init.
                            jitter = rng.normal(scale=1.0e-3 * scale, size=self.n_components)
                            Y0[i] = Y0[nearest].mean(axis=0) + jitter
                        else:
                            Y0[i] = rng.normal(scale=0.1 * scale, size=self.n_components)

                if mask.sum() == 0.0:
                    # no node in common with the previous snapshot (e.g. an
                    # overnight gap) -> nothing to anchor to; the system
                    # (V+lam*M) would be singular at M=0 (see documentation/
                    # 2026-09-11_kontrola_vysledku_plnych_behu.md, section 3.1)
                    logger.warning(
                        "Snapshot t=%d has no node in common with the previous snapshot "
                        "(mask.sum()==0) - using lam_t=0.0 (no anchors).", t,
                    )
                    lam_t = 0.0
                else:
                    lam_t = lam

            if self.solver == "smacof":
                Y_t, hist = smacof_solve(
                    D_t, W, Z, Y0, max_iter=self.max_iter, tol=self.tol, eps_num=self.eps_num,
                    dense_pinv_threshold=self.dense_pinv_threshold, cg_max_iter=self.cg_max_iter,
                    cg_tol=self.cg_tol, reg_rho=self.reg_rho, anchors=anchors, mask=mask, lam=lam_t,
                    verbose=False,
                )
            else:  # solver == "sgd" (section 6.3 - anchor step after each epoch)
                gamma = None if self.gamma_mode == "eta_t" else self.gamma_fixed
                Y_t, hist = sgd_solve(
                    D_t, alpha, Y0, epochs=self.sgd_epochs, mu_max=self.sgd_mu_max,
                    pairs_per_node=self.sgd_pairs_per_node, eps_anneal=self.sgd_eps_anneal,
                    eps_num=self.eps_num, seed=self.seed + t, variant=self.sgd_variant,
                    k_eps=self.eps_D_k, q_eps=self.eps_D_q, anchors=anchors, mask=mask, lam=lam_t,
                    gamma=gamma, verbose=False,
                )
            hist["eps_D"] = eps_D
            hist["Z"] = Z
            Y_list.append(Y_t)
            history_list.append(hist)
            prev_Y = Y_t
            prev_ids = ids_t

        self.Y_list_ = Y_list
        self.node_ids_ = node_ids_per_snapshot
        self.history_list_ = history_list
        self.lam_ = lam
        self.alpha_ = alpha
        return self

    def stability(self, procrustes_align: bool = True) -> float:
        """Stability (section 6.5): mean normalized displacement of shared
        nodes between consecutive snapshots, after Procrustes alignment
        without scaling (a fair comparison even for lambda=0, where
        snapshots are independent).

        A thin wrapper around the canonical implementation
        `src.sammon.temporal_metrics.stability` (also shared with baseline
        methods outside TemporalSammon)."""
        if not hasattr(self, "Y_list_"):
            raise RuntimeError("Call fit() first.")
        return _stability_metric(self.Y_list_, self.node_ids_, procrustes_align=procrustes_align)

    def quality(self) -> float:
        """Quality (section 6.5): median E_alpha^scale-inv over all snapshots."""
        if not hasattr(self, "history_list_"):
            raise RuntimeError("Call fit() first.")
        vals = [hist["stress"][-1] for hist in self.history_list_]
        return float(np.median(vals))
