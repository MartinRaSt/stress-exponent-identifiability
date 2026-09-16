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
Metrics for dynamic (temporal) embeddings (rows 24-26 of the metrics table,
reserse/2026-09-09_pseudokod_a_metriky.md PART B): stability between
snapshots (section 6.5 of the spec), trajectory smoothness (2nd-order
numerical difference), and the mental map metric (Archambault & Purchase
2013).

All functions operate on a list of embeddings `Y_list` (one `np.ndarray`
(n_t x p) per snapshot) and a corresponding list `node_ids_list` (list of
node identifiers for the given snapshot, in the same order as the rows of
Y_list[t]). This is the canonical implementation -
`TemporalSammon.stability()` in `src/sammon/temporal.py` is a thin wrapper
around `stability()` from this module.
"""
from __future__ import annotations

import numpy as np


def orthogonal_procrustes_no_scale(Y_src: np.ndarray, Y_ref: np.ndarray) -> np.ndarray:
    """Align Y_src to Y_ref by rotation/reflection/translation WITHOUT
    scaling (section 6.5 of the spec - a fair stability comparison even for
    independent snapshots). Returns the aligned Y_src."""
    mu_src = Y_src.mean(axis=0)
    mu_ref = Y_ref.mean(axis=0)
    Xs = Y_src - mu_src
    Xr = Y_ref - mu_ref
    U, _, Vt = np.linalg.svd(Xs.T @ Xr)
    R = U @ Vt
    return Xs @ R + mu_ref


def stability(Y_list: list[np.ndarray], node_ids_list: list[list], procrustes_align: bool = True) -> float:
    """Stability of a dynamic embedding (section 6.5, eq. stab(lambda)):
    mean normalized displacement of shared nodes between consecutive
    snapshots, after Procrustes alignment without scaling (a fair
    comparison even for independently fitted snapshots, e.g. the lambda=0
    baseline).

    stab = mean_t [ mean_{i in common(t-1,t)} ||y_i(t)-y_i(t-1)|| / L(t) ],
    L(t) = mean pairwise distance in Y_t.
    """
    if len(Y_list) != len(node_ids_list):
        raise ValueError("Y_list and node_ids_list must have the same length.")
    per_transition = []
    for t in range(1, len(Y_list)):
        Y_prev, ids_prev = Y_list[t - 1], node_ids_list[t - 1]
        Y_cur, ids_cur = Y_list[t], node_ids_list[t]
        id_to_prev = {nid: i for i, nid in enumerate(ids_prev)}
        common_cur_idx = [i for i, nid in enumerate(ids_cur) if nid in id_to_prev]
        if not common_cur_idx:
            continue
        common_prev_idx = [id_to_prev[ids_cur[i]] for i in common_cur_idx]

        Y_cur_common = Y_cur[common_cur_idx]
        Y_prev_common = Y_prev[common_prev_idx]
        if procrustes_align:
            Y_cur_common = orthogonal_procrustes_no_scale(Y_cur_common, Y_prev_common)

        disp = np.linalg.norm(Y_cur_common - Y_prev_common, axis=1)
        n_cur = Y_cur.shape[0]
        if n_cur > 1:
            mask_full = ~np.eye(n_cur, dtype=bool)
            d_cur = np.sqrt(((Y_cur[:, None, :] - Y_cur[None, :, :]) ** 2).sum(-1))
            L_t = float(d_cur[mask_full].mean())
        else:
            L_t = 1.0
        if L_t <= 0:
            raise ValueError(
                f"Mean pairwise distance in Y_t (t={t}, n={n_cur}) is <= 0 - the embedding collapsed to "
                "a single point, stability cannot be normalized. Common for snapshots with very few (~1) "
                "nodes continuing from the previous snapshot (see src/sammon/temporal.py, K14 jitter for "
                "new-node initialization) - check min_snapshot_nodes/time_bin_sec for this dataset."
            )
        per_transition.append(float(disp.mean() / L_t))

    if not per_transition:
        raise ValueError("No transition has common nodes - stability cannot be computed.")
    return float(np.mean(per_transition))


def trajectory_smoothness(Y_list: list[np.ndarray], node_ids_list: list[list], align: bool = False) -> float:
    """Trajectory smoothness (row 25 of the metrics table): mean squared
    2nd-order numerical difference of node trajectories over time,

    smooth = (1/n) sum_i (1/(T-2)) sum_{t=2}^{T-1} ||y_i(t+1)-2y_i(t)+y_i(t-1)||^2,

    computed over nodes present in ALL snapshots (T>=3). Lower = smoother
    trajectory (0 for exactly linear motion - constant "velocity" between
    snapshots has a zero 2nd difference). `align=True` aligns each snapshot
    to the first via a Procrustes transform (without scale) before
    computing - default False, because the metric definition (no source in
    the DR literature, see metrics table row 25) operates directly on the
    output coordinates; alignment is only useful when comparing
    independently oriented baseline snapshots."""
    if len(Y_list) != len(node_ids_list):
        raise ValueError("Y_list and node_ids_list must have the same length.")
    T = len(Y_list)
    if T < 3:
        raise ValueError(f"Trajectory smoothness requires at least 3 snapshots (T>=3), got T={T}.")

    common = set(node_ids_list[0])
    for ids in node_ids_list[1:]:
        common &= set(ids)
    if not common:
        raise ValueError("No node is present in all snapshots - trajectory smoothness cannot be computed.")
    common_list = [nid for nid in node_ids_list[0] if nid in common]

    aligned = []
    for Y_t, ids_t in zip(Y_list, node_ids_list):
        id_to_idx = {nid: i for i, nid in enumerate(ids_t)}
        aligned.append(Y_t[[id_to_idx[nid] for nid in common_list]])
    if align:
        ref = aligned[0]
        aligned = [ref] + [orthogonal_procrustes_no_scale(Y_t, ref) for Y_t in aligned[1:]]

    total = 0.0
    for t in range(1, T - 1):
        diff2 = aligned[t + 1] - 2.0 * aligned[t] + aligned[t - 1]
        total += float((diff2 ** 2).sum(axis=1).mean())
    return float(total / (T - 2))


def mental_map_preservation(Y_list: list[np.ndarray], node_ids_list: list[list], k: int, align: bool = True) -> float:
    """Mental map metric (row 26 of the metrics table; Archambault, Purchase
    2013, DOI 10.1016/j.ijhcs.2013.08.004; GD 2012, DOI
    10.1007/978-3-642-36763-2_42): fraction of preserved K nearest
    neighbors between consecutive snapshots for shared nodes, after
    Procrustes alignment (without scale). Averaged over all transitions
    and nodes.

    For each transition (t-1,t): shared nodes are aligned (Y_t to Y_{t-1});
    for each shared node i, the set of K nearest neighbors (among shared
    nodes) in Y_{t-1} is compared to that in the aligned Y_t; the metric is
    the mean fraction of overlapping neighbors."""
    if len(Y_list) != len(node_ids_list):
        raise ValueError("Y_list and node_ids_list must have the same length.")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}.")

    fractions: list[float] = []
    for t in range(1, len(Y_list)):
        ids_prev, ids_cur = node_ids_list[t - 1], node_ids_list[t]
        id_to_prev = {nid: i for i, nid in enumerate(ids_prev)}
        common_cur_idx = [i for i, nid in enumerate(ids_cur) if nid in id_to_prev]
        n_common = len(common_cur_idx)
        if n_common < k + 1:
            continue  # too few shared nodes for a meaningful K neighbors
        common_prev_idx = [id_to_prev[ids_cur[i]] for i in common_cur_idx]

        Y_prev_common = Y_list[t - 1][common_prev_idx]
        Y_cur_common = Y_list[t][common_cur_idx]
        if align:
            Y_cur_common = orthogonal_procrustes_no_scale(Y_cur_common, Y_prev_common)

        k_eff = min(k, n_common - 1)

        def _knn_order(pts: np.ndarray) -> np.ndarray:
            dist = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
            np.fill_diagonal(dist, np.inf)
            return np.argsort(dist, axis=1, kind="mergesort")

        order_prev = _knn_order(Y_prev_common)
        order_cur = _knn_order(Y_cur_common)

        for i in range(n_common):
            nb_prev = set(order_prev[i, :k_eff].tolist())
            nb_cur = set(order_cur[i, :k_eff].tolist())
            fractions.append(len(nb_prev & nb_cur) / k_eff)

    if not fractions:
        raise ValueError("No transition has enough shared nodes (>k) for the mental map metric.")
    return float(np.mean(fractions))
