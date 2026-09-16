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
Dimensionality-reduction quality metrics used to compare methods:
scale-invariant normalized stress, Sammon stress, Shepard Spearman
correlation, trustworthiness & continuity, co-ranking Q_NX/R_NX/AUC_RNX
(Lee & Verleysen, 2009), neighborhood hit, silhouette, and the Procrustes
distance between two embeddings.

All metrics that work with neighbor ranks (trustworthiness, continuity,
co-ranking) share a single O(n^2) rank matrix for the original and
projected space, to avoid repeated O(n^2 log n) sorting.
"""
from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
from scipy.spatial import procrustes
from scipy.stats import spearmanr
from sklearn.metrics import davies_bouldin_score, silhouette_score

from src.common.config import load_config
from src.methods.common import to_distance_matrix


def _neighbor_ranks(dist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute, for each row, the rank order of all points (1 = nearest neighbor).

    The diagonal (self-distance) is masked to +inf, so every point gets the
    highest rank (n) in its own row and is easy to exclude when working with
    neighborhoods. Returns (ranks, order): `ranks[i, j]` is the rank of
    point j from the perspective of point i; `order[i, k]` is the index of
    the k-th nearest neighbor of point i (0-based, k=0..n-2 are true
    neighbors, k=n-1 is the point itself).
    """
    n = dist.shape[0]
    masked = np.array(dist, dtype=np.float64, copy=True)
    np.fill_diagonal(masked, np.inf)
    order = np.argsort(masked, axis=1, kind="mergesort")
    ranks = np.empty((n, n), dtype=np.int32)
    row_idx = np.arange(n)[:, None]
    ranks[row_idx, order] = (np.arange(n)[None, :] + 1).astype(np.int32)
    return ranks, order


def _trustworthiness_continuity(
    rank_orig: np.ndarray, order_orig: np.ndarray, rank_emb: np.ndarray, order_emb: np.ndarray, k: int, n: int,
) -> tuple[float, float]:
    """Trustworthiness and continuity for a given K (Venna & Kaski, 2001)."""
    denom = n * k * (2 * n - 3 * k - 1)
    if denom <= 0:
        raise ValueError(f"Invalid combination n={n}, K={k} for trustworthiness/continuity (2n-3K-1<=0).")
    norm = 2.0 / denom

    knn_emb = order_emb[:, :k]
    r_orig_at_knn_emb = np.take_along_axis(rank_orig, knn_emb, axis=1)
    t_penalty = np.maximum(0, r_orig_at_knn_emb - k).sum()
    trustworthiness = 1.0 - norm * t_penalty

    knn_orig = order_orig[:, :k]
    r_emb_at_knn_orig = np.take_along_axis(rank_emb, knn_orig, axis=1)
    c_penalty = np.maximum(0, r_emb_at_knn_orig - k).sum()
    continuity = 1.0 - norm * c_penalty

    return float(trustworthiness), float(continuity)


def _coranking_matrix(rank_orig: np.ndarray, rank_emb: np.ndarray, n: int) -> np.ndarray:
    """Build the (n-1, n-1) co-ranking matrix Q (Lee & Verleysen, 2009)."""
    mask = ~np.eye(n, dtype=bool)
    r1 = (rank_orig[mask] - 1).astype(np.int64)
    r2 = (rank_emb[mask] - 1).astype(np.int64)
    size = n - 1
    idx = r1 * size + r2
    counts = np.bincount(idx, minlength=size * size)
    return counts.reshape(size, size)


def _qnx_curve(Q: np.ndarray, n: int) -> np.ndarray:
    """Return the array Q_NX(K) for K=1..n-1 (index 0 corresponds to K=1), Lee & Verleysen 2009 eq. 5.1."""
    size = Q.shape[0]
    cum_q = np.cumsum(np.cumsum(Q, axis=0), axis=1)
    ks = np.arange(1, size + 1)
    return np.diag(cum_q) / (ks * n)


def _rnx_curve(Q: np.ndarray, n: int) -> np.ndarray:
    """Return the array R_NX(K) for K=1..n-2 (index 0 corresponds to K=1); the last element (K=n-1) is NaN."""
    size = Q.shape[0]
    qnx = _qnx_curve(Q, n)
    ks = np.arange(1, size + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rnx = ((n - 1) * qnx - ks) / (n - 1 - ks)
    rnx[-1] = np.nan  # K = n-1: division by zero, undefined by construction
    return rnx


def _auc_rnx(rnx: np.ndarray) -> float:
    """AUC_RNX with logarithmic weighting w(K)=1/K (Lee & Verleysen, 2009)."""
    k_values = np.arange(1, len(rnx) + 1)
    valid = ~np.isnan(rnx)
    weights = 1.0 / k_values[valid]
    return float((rnx[valid] * weights).sum() / weights.sum())


def scale_invariant_stress(D: np.ndarray, d: np.ndarray) -> float:
    """Scale-invariant normalized stress with optimal scaling s*.

    s* = sum(D*d) / sum(d^2); stress = sum (D - s*d)^2 / sum D^2 (off-diagonal).
    """
    n = D.shape[0]
    mask = ~np.eye(n, dtype=bool)
    Dm, dm = D[mask], d[mask]
    denom_s = (dm ** 2).sum()
    if denom_s <= 0:
        raise ValueError("Sum of squared embedding distances is 0 - stress is not defined.")
    s_star = (Dm * dm).sum() / denom_s
    denom_stress = (Dm ** 2).sum()
    if denom_stress <= 0:
        raise ValueError("Sum of squared original distances is 0 - stress is not defined.")
    return float(((Dm - s_star * dm) ** 2).sum() / denom_stress)


def sammon_stress(D: np.ndarray, d: np.ndarray, eps: float) -> float:
    """Original Sammon stress: sum_{i<j} (D_ij-d_ij)^2/D_ij / sum_{i<j} D_ij."""
    n = D.shape[0]
    triu = np.triu_indices(n, k=1)
    Dt, dt = D[triu], d[triu]
    c = Dt.sum()
    if c <= 0:
        raise ValueError("Sum of original distances is 0 - Sammon stress is not defined.")
    Dt_safe = np.where(Dt < eps, eps, Dt)
    return float(((Dt - dt) ** 2 / Dt_safe).sum() / c)


def shepard_spearman(D: np.ndarray, d: np.ndarray, n_sample_pairs: int, random_state: int) -> tuple[float, float]:
    """Spearman correlation between original and projected distances on a subsample of pairs."""
    n = D.shape[0]
    triu = np.triu_indices(n, k=1)
    Dt, dt = D[triu], d[triu]
    rng = np.random.default_rng(random_state)
    n_sample = min(n_sample_pairs, len(Dt))
    idx = rng.choice(len(Dt), size=n_sample, replace=False)
    rho, pvalue = spearmanr(Dt[idx], dt[idx])
    return float(rho), float(pvalue)


def neighborhood_hit(order_emb: np.ndarray, y: np.ndarray, k: int) -> float:
    """Fraction of the K nearest neighbors in the embedding with the same class as the given point."""
    knn_emb = order_emb[:, :k]
    same_label = y[knn_emb] == y[:, None]
    return float(same_label.mean())


def procrustes_distance(Y1: np.ndarray, Y2: np.ndarray) -> float:
    """Mean Procrustes distance between two embeddings (after optimal
    alignment by rotation/scale/translation). Returns scipy `disparity`
    (sum of squared residuals after normalization to unit norm), see
    scipy.spatial.procrustes.
    """
    _, _, disparity = procrustes(np.asarray(Y1, dtype=np.float64), np.asarray(Y2, dtype=np.float64))
    return float(disparity)


# ---------------------------------------------------------------------------
# Extended metric set (extended=True in evaluate()) - PART B of the metrics
# table, reserse/2026-09-09_pseudokod_a_metriky.md. Parameters (n_dcor, K
# for LCMC/Jaccard) in config.yaml sammon.metrics_extended (deliberately
# kept separate from the `metrics:` section used by the original `evaluate()`).
# ---------------------------------------------------------------------------

def kruskal_stress1(D: np.ndarray, d: np.ndarray) -> float:
    """Kruskal's stress-1 (Kruskal 1964, DOI 10.1007/BF02289565):
    S_1 = sqrt( sum_{i<j} (D_ij-d_ij)^2 / sum_{i<j} d_ij^2 )."""
    n = D.shape[0]
    triu = np.triu_indices(n, k=1)
    Dt, dt = D[triu], d[triu]
    denom = (dt ** 2).sum()
    if denom <= 0:
        raise ValueError("Sum of squared projected distances is 0 - Kruskal's stress-1 is not defined.")
    return float(np.sqrt(((Dt - dt) ** 2).sum() / denom))


def shepard_pearson(D: np.ndarray, d: np.ndarray, n_sample_pairs: int, random_state: int) -> tuple[float, float]:
    """Pearson correlation coefficient between original and projected
    distances on a subsample of pairs (analogous to `shepard_spearman`, a
    standard complementary metric)."""
    from scipy.stats import pearsonr

    n = D.shape[0]
    triu = np.triu_indices(n, k=1)
    Dt, dt = D[triu], d[triu]
    rng = np.random.default_rng(random_state)
    n_sample = min(n_sample_pairs, len(Dt))
    idx = rng.choice(len(Dt), size=n_sample, replace=False)
    r, pvalue = pearsonr(Dt[idx], dt[idx])
    return float(r), float(pvalue)


def _double_center(M: np.ndarray) -> np.ndarray:
    """Double centering of a matrix (subtracts the row and column mean, adds
    back the overall mean) - the basis of computing distance covariance
    (Szekely et al. 2007)."""
    return M - M.mean(axis=1, keepdims=True) - M.mean(axis=0, keepdims=True) + M.mean()


def distance_correlation(D: np.ndarray, d: np.ndarray, n_dcor: int, random_state: int) -> float:
    """Distance correlation dCor(D,d) between the original and projected
    distance matrix (Szekely, Rizzo, Bakirov 2007, DOI
    10.1214/009053607000000505), applied directly to the already-existing
    (n x n) distance matrices D, d (i.e. D, d play the role of "a_kl"/"b_kl"
    in the classic definition of dCor for two random vectors - the
    distances are already computed, only double centering is needed).

    dCov(D,d)^2 = mean(A*B), dVar(X)^2 = mean(A*A), A/B = the double-centered
    D/d. dCor = sqrt(dCov^2) / (dVar(D)^2 * dVar(d)^2)^{1/4}.

    For n > n_dcor, a random subsample of points is used (O(n^2) would
    otherwise be prohibitive for large n, see config
    sammon.metrics_extended.n_dcor)."""
    n = D.shape[0]
    if n > n_dcor:
        rng = np.random.default_rng(random_state)
        idx = np.sort(rng.choice(n, size=n_dcor, replace=False))
        D = D[np.ix_(idx, idx)]
        d = d[np.ix_(idx, idx)]

    A = _double_center(np.asarray(D, dtype=np.float64))
    B = _double_center(np.asarray(d, dtype=np.float64))
    dcov2 = max(float((A * B).mean()), 0.0)
    dvar_a2 = float((A * A).mean())
    dvar_b2 = float((B * B).mean())
    if dvar_a2 <= 0 or dvar_b2 <= 0:
        raise ValueError("dVar(D) or dVar(d) is 0 (degenerate data) - distance correlation is not defined.")
    return float(np.sqrt(dcov2) / (dvar_a2 * dvar_b2) ** 0.25)


def qlocal_qglobal(Q: np.ndarray, rnx: np.ndarray, n: int) -> tuple[float, float, int]:
    """Split the R_NX(K) curve into local and global parts at the point K*
    that maximizes LCMC(K) = Q_NX(K) - K/(n-1) (Lee & Verleysen 2009, DOI
    10.1016/j.neucom.2008.12.017; Lee, Renard, Bernard, Dupont, Verleysen
    2013, DOI 10.1016/j.neucom.2012.12.036). The K* maximizing LCMC
    corresponds to the point where the gap between Q_NX(K) and the
    random-embedding baseline K/(n-1) is largest, i.e. it minimizes the
    "area" between the curve and the baseline at the breakpoint - a
    practical operationalization of "K* minimizing area" from the task
    decomposition.

    Q_local = mean R_NX(K) for K<=K*, Q_global = mean R_NX(K) for K>K*.
    Returns (Q_local, Q_global, K*)."""
    qnx = _qnx_curve(Q, n)
    ks = np.arange(1, len(qnx) + 1)
    lcmc = qnx - ks / (n - 1)
    if not np.any(np.isfinite(lcmc)):
        raise ValueError("LCMC curve is undefined everywhere - cannot determine K*.")
    k_star_idx = int(np.nanargmax(np.where(np.isfinite(lcmc), lcmc, -np.inf)))
    q_local = float(np.nanmean(rnx[: k_star_idx + 1]))
    tail = rnx[k_star_idx + 1 :]
    q_global = float(np.nanmean(tail)) if tail.size > 0 and np.any(np.isfinite(tail)) else np.nan
    return q_local, q_global, int(ks[k_star_idx])


def lcmc_at_k(Q: np.ndarray, n: int, k: int) -> float:
    """LCMC(K) = Q_NX(K) - K/(n-1) (Chen & Buja 2009, DOI 10.1198/jasa.2009.0111)."""
    if k < 1 or k > Q.shape[0]:
        raise ValueError(f"k={k} must be in range [1, {Q.shape[0]}].")
    qnx = _qnx_curve(Q, n)
    return float(qnx[k - 1] - k / (n - 1))


def knn_jaccard(order_orig: np.ndarray, order_emb: np.ndarray, k: int) -> float:
    """Mean Jaccard overlap of the K nearest neighbors between the original
    and projected space: J(i) = |N_K^D(i) n N_K^Y(i)| / |N_K^D(i) u N_K^Y(i)|,
    averaged over all points i (a standard derived metric, cf.
    trustworthiness/continuity - Venna & Kaski 2006)."""
    n = order_orig.shape[0]
    if k < 1 or k > n - 1:
        raise ValueError(f"k={k} must be in range [1, {n - 1}].")
    knn_orig = order_orig[:, :k]
    knn_emb = order_emb[:, :k]
    jac = np.empty(n, dtype=np.float64)
    for i in range(n):
        so = set(knn_orig[i].tolist())
        se = set(knn_emb[i].tolist())
        union = so | se
        jac[i] = len(so & se) / len(union) if union else 0.0
    return float(jac.mean())


def distance_consistency(Y: np.ndarray, y: np.ndarray) -> float:
    """Distance consistency DSC (Sips, Neubert, Lewis, Hanrahan 2009, DOI
    10.1111/j.1467-8659.2009.01467.x): fraction of points whose nearest
    class centroid in the projection Y matches their true class."""
    y = np.asarray(y)
    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError("DSC requires at least 2 classes.")
    centroids = np.array([Y[y == c].mean(axis=0) for c in classes])
    dists = np.linalg.norm(Y[:, None, :] - centroids[None, :, :], axis=2)
    pred = classes[np.argmin(dists, axis=1)]
    return float((pred == y).mean())


def evaluate(
    D_or_X: np.ndarray | nx.Graph,
    Y: np.ndarray,
    y: np.ndarray | None,
    kind: str,
    extended: bool = False,
) -> dict[str, Any]:
    """Compute all quality metrics of the embedding Y relative to the original data.

    Parameters:
        D_or_X: original data - a point matrix (kind='vector'), a distance
            matrix (kind='distance'), or a graph (kind='graph').
        Y: the resulting embedding (n x n_components).
        y: optional class labels (n,), used for neighborhood hit and silhouette.
        kind: type of the original data, see `src.methods.common.to_distance_matrix`.
        extended: if True, adds the extended metric set (PART B table -
            Kruskal stress-1, Shepard Pearson, distance correlation,
            Q_local/Q_global, LCMC, kNN Jaccard, Davies-Bouldin, distance
            consistency). Default False, since several of these (dCor, DB)
            have non-negligible extra cost for large n.

    Returns a dict metric -> value (float or NaN if the metric is not
    defined for the given sample count/K - a fallback value is never
    fabricated).
    """
    cfg = load_config()["metrics"]
    cfg_ext = load_config()["sammon"]["metrics_extended"]

    D = to_distance_matrix(D_or_X, kind)
    d = to_distance_matrix(np.asarray(Y, dtype=np.float64), "vector")
    n = D.shape[0]
    if d.shape[0] != n:
        raise ValueError(f"Number of points in the embedding ({d.shape[0]}) does not match the original data ({n}).")

    results: dict[str, Any] = {"n_samples": n}

    results["stress_scale_invariant"] = scale_invariant_stress(D, d)
    results["sammon_stress"] = sammon_stress(D, d, eps=cfg["eps"])

    rho, pvalue = shepard_spearman(D, d, n_sample_pairs=cfg["shepard_sample_pairs"], random_state=cfg["random_state"])
    results["shepard_spearman_rho"] = rho
    results["shepard_spearman_pvalue"] = pvalue

    rank_orig, order_orig = _neighbor_ranks(D)
    rank_emb, order_emb = _neighbor_ranks(d)

    for k in cfg["neighborhood_k"]:
        if k >= n - 1 or k < 1:
            results[f"trustworthiness_k{k}"] = np.nan
            results[f"continuity_k{k}"] = np.nan
            continue
        t, c = _trustworthiness_continuity(rank_orig, order_orig, rank_emb, order_emb, k, n)
        results[f"trustworthiness_k{k}"] = t
        results[f"continuity_k{k}"] = c

    Q = None
    rnx = None
    if n <= cfg["coranking_max_n"]:
        Q = _coranking_matrix(rank_orig, rank_emb, n)
        rnx = _rnx_curve(Q, n)
        results["auc_rnx"] = _auc_rnx(rnx)
        for k in cfg["neighborhood_k"]:
            if 1 <= k <= n - 2:
                results[f"qnx_k{k}"] = float(Q[:k, :k].sum() / (k * n))
                results[f"rnx_k{k}"] = float(rnx[k - 1])
            else:
                results[f"qnx_k{k}"] = np.nan
                results[f"rnx_k{k}"] = np.nan
    else:
        results["auc_rnx"] = np.nan  # above the coranking_max_n limit from config.yaml (O(n^2) memory cost)

    if y is not None:
        y = np.asarray(y)
        k_nh = cfg["neighborhood_k"][0]
        if k_nh < n:
            results[f"neighborhood_hit_k{k_nh}"] = neighborhood_hit(order_emb, y, k_nh)
        else:
            results[f"neighborhood_hit_k{k_nh}"] = np.nan

        n_classes = len(np.unique(y))
        if 2 <= n_classes < n:
            results["silhouette"] = float(silhouette_score(Y, y))
        else:
            results["silhouette"] = np.nan

    if extended:
        results["kruskal_stress1"] = kruskal_stress1(D, d)

        pear_r, pear_p = shepard_pearson(D, d, n_sample_pairs=cfg["shepard_sample_pairs"], random_state=cfg["random_state"])
        results["shepard_pearson_r"] = pear_r
        results["shepard_pearson_pvalue"] = pear_p

        results["distance_correlation"] = distance_correlation(D, d, n_dcor=cfg_ext["n_dcor"], random_state=cfg["random_state"])

        k_lcmc = cfg_ext["lcmc_k"]
        k_jac = cfg_ext["jaccard_k"]
        if 1 <= k_jac <= n - 1:
            results[f"knn_jaccard_k{k_jac}"] = knn_jaccard(order_orig, order_emb, k_jac)
        else:
            results[f"knn_jaccard_k{k_jac}"] = np.nan

        if Q is not None and rnx is not None:
            if 1 <= k_lcmc <= Q.shape[0]:
                results[f"lcmc_k{k_lcmc}"] = lcmc_at_k(Q, n, k_lcmc)
            else:
                results[f"lcmc_k{k_lcmc}"] = np.nan
            q_local, q_global, k_star = qlocal_qglobal(Q, rnx, n)
            results["q_local"] = q_local
            results["q_global"] = q_global
            results["q_local_global_k_star"] = k_star
        else:
            results[f"lcmc_k{k_lcmc}"] = np.nan
            results["q_local"] = np.nan
            results["q_global"] = np.nan
            results["q_local_global_k_star"] = np.nan

        if y is not None:
            y = np.asarray(y)
            n_classes = len(np.unique(y))
            if 2 <= n_classes < n:
                results["davies_bouldin"] = float(davies_bouldin_score(Y, y))
                results["distance_consistency"] = distance_consistency(Y, y)
            else:
                results["davies_bouldin"] = np.nan
                results["distance_consistency"] = np.nan

        # K2 (documentation/2026-09-12_plan_smeru_clanku.md): between-cluster
        # geometry (centroid_dist_spearman, class_spread_spearman,
        # class_spread_lie_factor, centroid_knn_preservation) - see
        # src/sammon/cluster_geometry.py. NaN if y is missing or < 3 classes
        # (a fallback value is never fabricated).
        from src.sammon.cluster_geometry import cluster_geometry_metrics

        results.update(cluster_geometry_metrics(D, d, y, k_neighbors=cfg_ext["cluster_geometry_k"]))

    return results
