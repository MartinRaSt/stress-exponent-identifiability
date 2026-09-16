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
Stochastic gradient descent for graph drawing / MDS (Zheng, Pawar, Goodman
2018) and its stabilized variant for alpha>0 (see
reserse/2026-09-09_specifikace_metody.md, section 3).

Two variants:
    naive      - the original Zheng 2018 scheme (w_ij=D_ij^{-alpha} without
                 eps_D regularization, all pairs every epoch, mu=min(1, eta*w)).
    stabilized - eps_D weight regularization, robust annealing (median),
                 mu_max clipping, C*n sampled pairs per epoch (section 3.3).

The CPU epoch kernel is a sequential Gauss-Seidel loop over pairs,
accelerated with `numba.njit` (deterministic for a fixed pair order). The
GPU variant (round-robin, node-disjoint batches) is in `sgd_gpu.py`.
"""
from __future__ import annotations

import time

import numpy as np
from numba import njit

from src.common.progress import progress_iter
from src.sammon.stress import scale_invariant_stress
from src.sammon.weights import alpha_weights, compute_Z, compute_Z_from_W, estimate_eps_D


@njit(cache=True)
def _sgd_epoch_numba(Y: np.ndarray, D: np.ndarray, W: np.ndarray, i_idx: np.ndarray, j_idx: np.ndarray,
                      eta_t: float, mu_max: float, eps_num: float) -> tuple[int, np.ndarray]:
    """One epoch of sequential (Gauss-Seidel) SGD updates over the given
    list of pairs (i_idx, j_idx). Modifies Y in-place. Returns
    (n_saturated, mu_array)."""
    n_pairs = i_idx.shape[0]
    p = Y.shape[1]
    n_sat = 0
    mus = np.empty(n_pairs, dtype=np.float64)
    for t in range(n_pairs):
        i = i_idx[t]
        j = j_idx[t]
        dsq = 0.0
        for k in range(p):
            diff = Y[i, k] - Y[j, k]
            dsq += diff * diff
        d = np.sqrt(dsq)
        if d < eps_num:
            d = eps_num
        w = W[i, j]
        mu = eta_t * w
        if mu > mu_max:
            mu = mu_max
            n_sat += 1
        mus[t] = mu
        r = 0.5 * mu * (d - D[i, j]) / d
        for k in range(p):
            delta = r * (Y[i, k] - Y[j, k])
            Y[i, k] -= delta
            Y[j, k] += delta
    return n_sat, mus


def _all_pairs(n: int) -> tuple[np.ndarray, np.ndarray]:
    iu = np.triu_indices(n, k=1)
    return iu[0].astype(np.int64), iu[1].astype(np.int64)


def _sample_pairs_for_epoch(n: int, n_pairs_total: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Samples `n_pairs_total` distinct pairs (i<j) without replacement within an epoch."""
    max_pairs = n * (n - 1) // 2
    n_sample = min(n_pairs_total, max_pairs)
    flat_idx = rng.choice(max_pairs, size=n_sample, replace=False)
    # convert the linear index to (i,j) via triu order - for small n it suffices to generate all pairs
    all_i, all_j = _all_pairs(n)
    order = rng.permutation(n_sample)  # additional shuffling of order (section 3.3: shuffle(pairs))
    return all_i[flat_idx][order], all_j[flat_idx][order]


def sgd_solve(
    D: np.ndarray,
    alpha: float,
    Y0: np.ndarray,
    epochs: int,
    mu_max: float,
    pairs_per_node: int,
    eps_anneal: float,
    eps_num: float,
    seed: int,
    variant: str = "stabilized",
    k_eps: int = 5,
    q_eps: float = 0.1,
    anchors: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    lam: float = 0.0,
    gamma: float | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, dict]:
    """SGD solver (naive or stabilized variant), section 3, with an optional
    temporal anchor step (section 6.3).

    Parameters:
        D: distance matrix (n x n).
        alpha: exponent of the weight family w_ij = D_ij^{-alpha}.
        Y0: initial configuration (n x p).
        epochs: number of epochs T.
        mu_max: clipping (used only for variant='stabilized'; for 'naive'
            the effective cap is 1.0, see section 3.1).
        pairs_per_node: C - number of sampled pairs per node per epoch
            (used only for 'stabilized'; 'naive' uses all pairs).
        eps_anneal: epsilon in eta_min = eps_anneal / w_max.
        eps_num: numerical stabilization of d_ij.
        seed: seed for pair sampling and ordering.
        variant: 'naive' or 'stabilized'.
        k_eps, q_eps: eps_D estimation parameters (only for 'stabilized').
        anchors, mask, lam: temporal anchoring (section 6.3) - after each
            epoch, an additional step is performed for nodes with m_i>0:
            y_i <- y_i - gamma_t*lambda*m_i*(y_i-a_i). lam=0 (default) =>
            no anchor step (no change in behavior). lam>0 requires the
            'anchors' (a_i) and 'mask' (m_i in {0,1}) to be provided.
        gamma: gamma_t in the anchor step; None (default) => gamma_t=eta_t
            (shared annealing, section 6.3 - "gamma_t may share the eta_t
            annealing"), otherwise a fixed constant for all epochs.
        verbose: tqdm progress bar.

    Returns (Y, history): history contains 'stress' (E_alpha^scale-inv per
    epoch), 'sat_ratio', 'mu_median', 'mu_iqr', 'time_sec', 'eps_D' (only
    for stabilized), 'variant'.
    """
    if variant not in ("naive", "stabilized"):
        raise ValueError(f"Unknown SGD variant '{variant}' (expected 'naive' or 'stabilized').")
    if lam > 0 and (anchors is None or mask is None):
        raise ValueError("lam > 0 requires 'anchors' and 'mask' to be provided (temporal anchoring, section 6.3).")

    D = np.asarray(D, dtype=np.float64)
    n = D.shape[0]
    Y = np.asarray(Y0, dtype=np.float64).copy()
    rng = np.random.default_rng(seed)

    eps_D = None
    if variant == "stabilized":
        eps_D = estimate_eps_D(D, k=k_eps, q=q_eps, kind="distance")
        W = alpha_weights(D, alpha, eps_D)
    else:
        # naive: original weights without regularization (D_ij=0 would
        # cause division by zero - fail loud, we do not fabricate)
        if alpha > 0 and np.any(D[~np.eye(n, dtype=bool)] == 0):
            raise ValueError(
                "Naive SGD requires D_ij>0 for all off-diagonal pairs when alpha>0 "
                "(duplicate points -> w_ij diverges). Use variant='stabilized'."
            )
        with np.errstate(divide="ignore"):
            W = np.zeros_like(D) if alpha == 0.0 else np.power(D, -alpha, where=(D > 0), out=np.full_like(D, np.inf))
        if alpha == 0.0:
            W[:] = 1.0
        np.fill_diagonal(W, 0.0)

    Z = compute_Z_from_W(D, W) if variant == "stabilized" else compute_Z(D, alpha)

    mask_off = ~np.eye(n, dtype=bool)
    w_vals = W[mask_off]
    if variant == "stabilized":
        w_med = float(np.median(w_vals))
        w_max = float(w_vals.max())
        eta_max = 1.0 / w_med
        eta_min = eps_anneal / w_max
        n_pairs_total = pairs_per_node * n
    else:
        w_min = float(w_vals[w_vals > 0].min())
        w_max = float(w_vals.max())
        eta_max = 1.0 / w_min
        eta_min = eps_anneal / w_max
        n_pairs_total = n * (n - 1) // 2  # naive: all pairs every epoch

    stress_hist: list[float] = []
    sat_ratio_hist: list[float] = []
    mu_median_hist: list[float] = []
    mu_iqr_hist: list[float] = []
    time_hist: list[float] = []
    t_start = time.perf_counter()

    iterator = progress_iter(range(1, epochs + 1), desc=f"sgd_{variant}") if verbose else range(1, epochs + 1)
    for t in iterator:
        if epochs > 1:
            eta_t = eta_max * (eta_min / eta_max) ** ((t - 1) / (epochs - 1))
        else:
            eta_t = eta_max

        epoch_rng = np.random.default_rng(rng.integers(0, 2**63 - 1))
        if variant == "stabilized":
            i_idx, j_idx = _sample_pairs_for_epoch(n, n_pairs_total, epoch_rng)
        else:
            all_i, all_j = _all_pairs(n)
            order = epoch_rng.permutation(len(all_i))
            i_idx, j_idx = all_i[order], all_j[order]

        mu_max_eff = mu_max if variant == "stabilized" else 1.0
        n_sat, mus = _sgd_epoch_numba(Y, D, W, i_idx, j_idx, eta_t, mu_max_eff, eps_num)

        if lam > 0:
            # anchor step (section 6.3): after the epoch, extra O(n), outside
            # the numba kernel (not a hot pair-loop, just one vectorized numpy step)
            gamma_t = eta_t if gamma is None else gamma
            Y -= gamma_t * lam * mask[:, None] * (Y - anchors)

        stress = scale_invariant_stress(D, Y, W, Z, eps_num)
        stress_hist.append(stress)
        sat_ratio_hist.append(n_sat / len(i_idx))
        mu_median_hist.append(float(np.median(mus)))
        q75, q25 = np.percentile(mus, [75, 25])
        mu_iqr_hist.append(float(q75 - q25))
        time_hist.append(time.perf_counter() - t_start)

    history = {
        "stress": stress_hist, "sat_ratio": sat_ratio_hist, "mu_median": mu_median_hist,
        "mu_iqr": mu_iqr_hist, "time_sec": time_hist, "variant": variant,
        "eps_D": eps_D, "eta_max": eta_max, "eta_min": eta_min, "lam": lam,
    }
    return Y, history


# ---------------------------------------------------------------------------
# GPU variant (sections 3.3, 12.2): round-robin (node-disjoint) batching as
# the default (exact match with the sequential Gauss-Seidel scheme within a
# batch), Jacobi as a simpler/approximate ablation (possible node collisions
# within a single batch).
#
# PERFORMANCE WARNING: the original implementation built rounds via greedy
# (node-disjoint) edge-coloring over an ALREADY-SAMPLED list of pairs (a
# numba loop over pairs) - the number of rounds thus grew with the maximum
# node degree in that list (up to O(n) for a dense selection, e.g. naive SGD
# with all pairs), which for n~1000 led to thousands of GPU kernel launches
# per epoch, and the GPU was 30-60x SLOWER than CPU (measured, see
# projectstate.md). Replaced with a scheme with a FIXED number of rounds
# independent of n: `pairs_per_node` (C) random perfect matchings per epoch
# (each round = a random permutation of all n nodes split into pairs, O(1)
# numpy operations, no loop over pairs/rounds). Used for BOTH GPU variants
# (naive and stabilized) - `_round_robin_tournament_base` below implements
# an alternative exactly-covering scheme (a classic round-robin tournament,
# n-1 rounds) for possible future use where an exact match with the CPU
# "all pairs every epoch" (section 3.1) is required even on GPU, at the
# cost of O(n) rounds.
# ---------------------------------------------------------------------------

def _perfect_matching_round(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """One round of a random perfect matching of all n nodes (for odd n, one
    random node is skipped in this round) - O(n), no loop over pairs."""
    perm = rng.permutation(n)
    if n % 2 == 1:
        perm = perm[:-1]
    return perm[0::2].astype(np.int64), perm[1::2].astype(np.int64)


def _round_robin_tournament_base(n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Base schedule of a classic round-robin tournament (circle method) for
    ABSTRACT player labels 0..n-1: n-1 (even n) / n (odd n, with a "bye")
    rounds, each pair exactly once. Computed ONCE (independent of the
    epoch); the actual nodes get a permutation of these indices every epoch
    (see the calling code), which preserves "all pairs every epoch" at
    O(1) overhead per epoch."""
    has_bye = n % 2 == 1
    n_eff = n + 1 if has_bye else n
    players = np.arange(n_eff)
    if has_bye:
        players[-1] = -1  # "bye" - this player does not play in this round
    fixed = players[0]
    rest = players[1:].copy()
    n_rounds = n_eff - 1
    half = n_eff // 2
    rounds: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(n_rounds):
        arr = np.concatenate(([fixed], rest))
        i_idx = arr[:half]
        j_idx = arr[half:][::-1]
        valid = (i_idx >= 0) & (j_idx >= 0)
        rounds.append((i_idx[valid].astype(np.int64), j_idx[valid].astype(np.int64)))
        rest = np.roll(rest, 1)
    return rounds


def sgd_solve_gpu(
    D: np.ndarray,
    alpha: float,
    Y0: np.ndarray,
    epochs: int,
    mu_max: float,
    pairs_per_node: int,
    eps_anneal: float,
    eps_num: float,
    seed: int,
    variant: str = "stabilized",
    batch_mode: str = "round_robin",
    k_eps: int = 5,
    q_eps: float = 0.1,
    device: str = "cuda",
    dtype: "object" = None,
    log_every: int = 1,
    verbose: bool = False,
) -> tuple[np.ndarray, dict]:
    """GPU variant of stabilized/naive SGD (sections 3.3, 12.2).

    `batch_mode='round_robin'` (default): each epoch consists of a FIXED
    number of node-disjoint rounds (independent of n) - the stabilized
    variant uses C random perfect matchings (`pairs_per_node` rounds), the
    naive variant uses a classic round-robin tournament covering all pairs
    exactly once (n-1 rounds). Each round is a fully vectorized GPU batch
    without a python loop over pairs and without synchronization within the
    round - see the comment at `_perfect_matching_round`/
    `_round_robin_tournament_base` for the reason (the original greedy
    edge-coloring over already-sampled pairs scaled with the max node
    degree and was 30-60x slower on GPU than on CPU).
    `batch_mode='jacobi'`: the entire pair list of the epoch is processed as
    a single batch via `index_add_` without collision handling (faster, but
    approximate - a documented ablation, NOT validated by a CPU-agreement
    test).
    `log_every`: the stress (requires transferring Y to CPU + an O(n^2)
    recomputation) is logged only every `log_every`-th epoch (and always
    the last epoch) - the dominant cost for large n/many epochs; sat_ratio
    is logged every epoch (only a scalar synchronization).
    """
    import torch

    if batch_mode not in ("round_robin", "jacobi"):
        raise ValueError(f"Unknown batch_mode '{batch_mode}' (expected 'round_robin' or 'jacobi').")
    if variant not in ("naive", "stabilized"):
        raise ValueError(f"Unknown SGD variant '{variant}'.")

    torch_dtype = dtype if dtype is not None else torch.float32
    dev = torch.device(device)

    D_np = np.asarray(D, dtype=np.float64)
    n = D_np.shape[0]

    eps_D = None
    if variant == "stabilized":
        eps_D = estimate_eps_D(D_np, k=k_eps, q=q_eps, kind="distance")
        W_np = alpha_weights(D_np, alpha, eps_D)
    else:
        if alpha > 0 and np.any(D_np[~np.eye(n, dtype=bool)] == 0):
            raise ValueError("Naive SGD requires D_ij>0 for alpha>0 (duplicate points).")
        with np.errstate(divide="ignore"):
            W_np = np.zeros_like(D_np) if alpha == 0.0 else np.power(D_np, -alpha, where=(D_np > 0), out=np.full_like(D_np, np.inf))
        if alpha == 0.0:
            W_np[:] = 1.0
        np.fill_diagonal(W_np, 0.0)

    Z = compute_Z_from_W(D_np, W_np) if variant == "stabilized" else compute_Z(D_np, alpha)

    D_t = torch.tensor(D_np, device=dev, dtype=torch_dtype)
    W_t = torch.tensor(W_np, device=dev, dtype=torch_dtype)
    Y = torch.tensor(np.asarray(Y0, dtype=np.float64), device=dev, dtype=torch_dtype).clone()

    mask_off = ~np.eye(n, dtype=bool)
    w_vals = W_np[mask_off]
    if variant == "stabilized":
        eta_max = 1.0 / float(np.median(w_vals))
        eta_min = eps_anneal / float(w_vals.max())
    else:
        eta_max = 1.0 / float(w_vals[w_vals > 0].min())
        eta_min = eps_anneal / float(w_vals.max())

    rng = np.random.default_rng(seed)

    stress_hist: list[float] = []
    sat_ratio_hist: list[float] = []
    time_hist: list[float] = []
    epoch_logged: list[int] = []
    t_start = time.perf_counter()
    mu_max_eff = mu_max if variant == "stabilized" else 1.0

    iterator = progress_iter(range(1, epochs + 1), desc=f"sgd_gpu_{variant}_{batch_mode}") if verbose else range(1, epochs + 1)
    for t in iterator:
        eta_t = eta_max * (eta_min / eta_max) ** ((t - 1) / (epochs - 1)) if epochs > 1 else eta_max
        epoch_rng = np.random.default_rng(rng.integers(0, 2**63 - 1))

        n_sat_epoch_t = torch.zeros((), device=dev, dtype=torch.int64)
        n_pairs_epoch = 0

        if batch_mode == "round_robin":
            # WARNING - a documented deviation from CPU: `variant='naive'` on
            # GPU does NOT cover all n(n-1)/2 pairs every epoch (an exact
            # round-robin tournament would require n-1 sequential rounds,
            # i.e. O(n) GPU launches per epoch - measured as 30-100x slower
            # than CPU for n~1000, see projectstate.md). Instead it uses the
            # same FIXED number of `pairs_per_node` random perfect matching
            # rounds as the stabilized variant (a smaller, but comparable
            # sample of pairs per epoch). The CPU `variant='naive'` remains
            # exact (all pairs, section 3.1) - use CPU for exact agreement
            # with the literature.
            rounds_np = [_perfect_matching_round(n, epoch_rng) for _ in range(pairs_per_node)]

            for i_np, j_np in rounds_np:
                if i_np.shape[0] == 0:
                    continue
                i_r = torch.as_tensor(i_np, device=dev, dtype=torch.long)
                j_r = torch.as_tensor(j_np, device=dev, dtype=torch.long)
                Yi, Yj = Y[i_r], Y[j_r]
                d = torch.clamp(torch.linalg.norm(Yi - Yj, dim=1), min=eps_num)
                w = W_t[i_r, j_r]
                mu = torch.clamp(eta_t * w, max=mu_max_eff)
                n_sat_epoch_t += (eta_t * w > mu_max_eff).sum()
                n_pairs_epoch += i_np.shape[0]
                Dij = D_t[i_r, j_r]
                r_coef = (0.5 * mu * (d - Dij) / d).unsqueeze(1)
                delta = r_coef * (Yi - Yj)
                Y.index_add_(0, i_r, -delta)
                Y.index_add_(0, j_r, delta)
        else:  # jacobi - approximate, possible collisions within a single batch
            if variant == "stabilized":
                i_np, j_np = _sample_pairs_for_epoch(n, pairs_per_node * n, epoch_rng)
            else:
                all_i, all_j = _all_pairs(n)
                order = epoch_rng.permutation(len(all_i))
                i_np, j_np = all_i[order], all_j[order]
            n_pairs_epoch = len(i_np)
            i_r = torch.as_tensor(i_np, device=dev, dtype=torch.long)
            j_r = torch.as_tensor(j_np, device=dev, dtype=torch.long)
            Yi, Yj = Y[i_r], Y[j_r]
            d = torch.clamp(torch.linalg.norm(Yi - Yj, dim=1), min=eps_num)
            w = W_t[i_r, j_r]
            mu = torch.clamp(eta_t * w, max=mu_max_eff)
            n_sat_epoch_t = (eta_t * w > mu_max_eff).sum()
            Dij = D_t[i_r, j_r]
            r_coef = (0.5 * mu * (d - Dij) / d).unsqueeze(1)
            delta = r_coef * (Yi - Yj)
            Y.index_add_(0, i_r, -delta)
            Y.index_add_(0, j_r, delta)

        is_last = t == epochs
        if (t % log_every == 0) or is_last:
            Y_np = Y.detach().cpu().numpy().astype(np.float64)
            stress_hist.append(scale_invariant_stress(D_np, Y_np, W_np, Z, eps_num))
            sat_ratio_hist.append(float(n_sat_epoch_t.item()) / max(n_pairs_epoch, 1))
            time_hist.append(time.perf_counter() - t_start)
            epoch_logged.append(t)

    history = {
        "stress": stress_hist, "sat_ratio": sat_ratio_hist, "time_sec": time_hist,
        "epoch_logged": epoch_logged, "variant": variant, "batch_mode": batch_mode,
        "eps_D": eps_D, "eta_max": eta_max, "eta_min": eta_min, "device": device,
    }
    return Y.detach().cpu().numpy().astype(np.float64), history
