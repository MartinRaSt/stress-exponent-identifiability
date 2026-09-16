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
Q1-step3 (reserse/2026-09-14_specifikace_rozsireni_q1.md, PART B.3) - "metric
fidelity" metrics of the embedding Y relative to the KNOWN latent ground
truth (in contrast to `src/sammon/metrics.py`, which measures against the
INPUT matrix D, not the ground truth).

Every embedding-dependent metric MUST be invariant to
    Y -> Y' = s*R*Y + t,   s>0, R orthogonal (including reflections), t translation,
because the compared methods (t-SNE/UMAP/densMAP vs. the stress/MDS family)
produce embeddings with a different scale, rotation, and mirroring. The
invariance derivation for each metric is given in its docstring; the
numerical test (tolerance < 1e-9) is in `tests/test_truth_metrics.py`.

Conventions:
- Pairwise metrics (centroid_*, cophenetic_pearson, geodesic_*) accept
  SQUARE symmetric distance matrices (achieved, truth) and internally
  extract the upper triangle (i<j) - see `_pairs`.
- `log_ratio_error`/`log_distortion` are GENERIC functions over already
  extracted 1D arrays of positive values (pairs OR radii) - the caller
  (e.g. `src/experiments/exp9_metric_fidelity.py`) extracts the
  pairs/radii itself, so a single implementation serves LRE_c, LD_c, and
  LRE_G/LD_G alike.
- `radius_slope`/`radius_lre` accept a 1D array of length K (radii per cluster).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr


def _pairs(M: np.ndarray) -> np.ndarray:
    """Return the values M[i,j] for i<j (M must be square (K,K) or (n,n))."""
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"Expected a square matrix, got shape {M.shape}.")
    n = M.shape[0]
    if n < 2:
        raise ValueError(f"Matrix must have at least 2 rows/columns for pairs, got n={n}.")
    iu = np.triu_indices(n, k=1)
    return M[iu]


def upper_triangle_pairs(M: np.ndarray) -> np.ndarray:
    """Public alias for `_pairs` for use outside this module
    (`src/experiments/exp9_metric_fidelity.py`, tests) - extracts the
    values M[i,j] for i<j from a square symmetric distance matrix."""
    return _pairs(M)


def _safe_corr(fn, a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"The achieved and truth arrays must have the same shape, got {a.shape} and {b.shape}.")
    if a.size < 2:
        raise ValueError(f"Correlation requires at least 2 values, got {a.size}.")
    if np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")  # correlation is undefined without variance - 0 is not fabricated
    rho, _ = fn(a, b)
    return float(rho)


def centroid_pearson(delta: np.ndarray, Delta: np.ndarray) -> float:
    """r_c = corr({Delta_kl},{delta_kl}) over k<l (B.3 table).

    Invariance: delta_kl is the Euclidean distance between two points in Y,
    so under Y->sRY+t, delta_kl' = s*delta_kl holds (rotation/reflection/
    translation do not change distances, the scale is a positive constant).
    The Pearson correlation coefficient is invariant to a positive affine
    transformation of either variable (corr(s*X+c, Y) = corr(X,Y) for s>0),
    so r_c(s*delta, Delta) = r_c(delta, Delta).
    """
    return _safe_corr(pearsonr, _pairs(delta), _pairs(Delta))


def centroid_spearman(delta: np.ndarray, Delta: np.ndarray) -> float:
    """rho_c = Spearman({Delta_kl},{delta_kl}) over k<l.

    Invariance: Spearman is a rank statistic - a positive scale (and any
    increasing transformation) does not change the order of values, so it
    is invariant exactly like centroid_pearson, and additionally also to
    nonlinear monotonic distortions.
    """
    return _safe_corr(spearmanr, _pairs(delta), _pairs(Delta))


def log_ratio_error(achieved_pairs: np.ndarray, truth_pairs: np.ndarray) -> float:
    """LRE = med_i |e_i - mean(e)|, e_i = log(achieved_i) - log(truth_i)
    (B.3: log-ratio error - a generic function used for LRE_c, LD_c, and
    LRE_G; the caller extracts pairs/radii in advance).

    Invariance: under the scale achieved_i -> s*achieved_i (s>0),
    e_i' = log(s) + e_i holds for ALL i (a constant shift), so
    mean(e') = mean(e) + log(s) and e_i' - mean(e') = e_i - mean(e) exactly -
    the shift cancels out, so LRE is invariant. Rotation/reflection/
    translation of Y does not change achieved_i at all (they are Euclidean
    distances), so LRE is invariant to the full Y->sRY+t.
    """
    a = np.asarray(achieved_pairs, dtype=np.float64)
    b = np.asarray(truth_pairs, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"achieved_pairs and truth_pairs must have the same shape, got {a.shape} and {b.shape}.")
    if np.any(a <= 0) or np.any(b <= 0):
        raise ValueError("log_ratio_error requires all values to be positive (log is undefined for <=0).")
    e = np.log(a) - np.log(b)
    return float(np.median(np.abs(e - np.mean(e))))


def log_distortion(achieved_pairs: np.ndarray, truth_pairs: np.ndarray) -> float:
    """LD = max_i e_i - min_i e_i (log Lipschitz distortion), e_i as in
    `log_ratio_error`.

    Invariance: e_i' = e_i + log(s) for all i (see `log_ratio_error`); the
    max-min difference cancels a constant shift -> invariant.
    """
    a = np.asarray(achieved_pairs, dtype=np.float64)
    b = np.asarray(truth_pairs, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"achieved_pairs and truth_pairs must have the same shape, got {a.shape} and {b.shape}.")
    if np.any(a <= 0) or np.any(b <= 0):
        raise ValueError("log_distortion requires all values to be positive (log is undefined for <=0).")
    e = np.log(a) - np.log(b)
    return float(np.max(e) - np.min(e))


def radius_slope(rho: np.ndarray, r: np.ndarray) -> float:
    """beta_r = OLS slope of log(rho_k) = beta_r*log(r_k) + b over k=1..K
    (B.3: 1 = radius ratios preserved, 0 = sizes equalized - a typical
    consequence of t-SNE/UMAP, see Kobak & Berens 2019).

    Invariance: under the scale rho_k -> s*rho_k, log(rho_k') = log(s)+log(rho_k)
    holds - a constant shift of the OLS dependent variable only changes the
    intercept b, the slope beta_r stays exactly the same. Rotation/
    reflection/translation of Y does not change the RMS radii rho_k at all
    (they are computed from Euclidean distances to the centroid).
    """
    rho = np.asarray(rho, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    if rho.shape != r.shape:
        raise ValueError(f"rho and r must have the same shape, got {rho.shape} and {r.shape}.")
    if rho.size < 2:
        raise ValueError(f"radius_slope requires at least 2 clusters, got {rho.size}.")
    if np.any(rho <= 0) or np.any(r <= 0):
        raise ValueError("radius_slope requires all radii to be positive (log is undefined for <=0).")
    slope, _intercept = np.polyfit(np.log(r), np.log(rho), 1)
    return float(slope)


def radius_lre(rho: np.ndarray, r: np.ndarray) -> float:
    """LRE_r = log_ratio_error(rho, r) over K clusters (B.3)."""
    return log_ratio_error(rho, r)


def cophenetic_pearson(delta: np.ndarray, u: np.ndarray) -> float:
    """CPCC = corr({u_kl},{delta_kl}) - the cophenetic correlation (Sokal &
    Rohlf 1962, DOI 10.2307/1217208) between the ultrametric u and the
    achieved centroid distances delta. Invariance: same as `centroid_pearson`."""
    return _safe_corr(pearsonr, _pairs(delta), _pairs(u))


def triplet_accuracy(u: np.ndarray, delta: np.ndarray) -> float:
    """TA = fraction of triplets (k,l,m) for which the nearest pair
    according to the ultrametric u (u_kl < u_km == u_lm) is also the
    nearest according to the achieved delta
    (delta_kl < min(delta_km, delta_lm)) - B.3. Chance level = 1/3.

    Invariance: a purely ordinal statistic (an "is smaller than"
    comparison) - invariant to any strictly increasing transformation of
    delta (hence also to Y->sRY+t, s>0)."""
    u = np.asarray(u, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    K = u.shape[0]
    if u.shape != delta.shape or u.shape != (K, K):
        raise ValueError(f"u and delta must be square of the same shape, got {u.shape} and {delta.shape}.")
    if K < 3:
        raise ValueError(f"triplet_accuracy requires at least 3 clusters, got K={K}.")
    from itertools import combinations

    n_correct = 0
    n_total = 0
    for i, j, k in combinations(range(K), 3):
        nearest_u = int(np.argmin((u[i, j], u[i, k], u[j, k])))
        nearest_delta = int(np.argmin((delta[i, j], delta[i, k], delta[j, k])))
        n_total += 1
        if nearest_u == nearest_delta:
            n_correct += 1
    return float(n_correct) / float(n_total)


def geodesic_stress_scale_inv(d: np.ndarray, G: np.ndarray) -> float:
    """sigma_G = 1 - (sum_{i<j} G_ij d_ij)^2 / (sum_{i<j} G_ij^2 * sum_{i<j} d_ij^2)
    - scale-invariant stress of the achieved distances d relative to the
    geodesic ground truth G (B.3, derivation: min_s sum(s*d-G)^2 =>
    s* = sum(Gd)/sum(d^2); substituting back gives
    sum(G^2) - (sum(Gd))^2/sum(d^2), divided by sum(G^2)).

    Invariance: the closed-form optimal scale s* ABSORBS any positive
    scale of d (d->s0*d => s*->s*/s0, but the resulting sum after
    substitution is unchanged - see test_truth_metrics.py, agreement with
    direct numerical minimization over s). Rotation/reflection/translation
    of Y does not change d_ij at all."""
    d_pairs = _pairs(d)
    G_pairs = _pairs(G)
    if d_pairs.shape != G_pairs.shape:
        raise ValueError(f"d and G must have the same shape, got {d.shape} and {G.shape}.")
    sum_dd = float(np.sum(d_pairs * d_pairs))
    sum_GG = float(np.sum(G_pairs * G_pairs))
    sum_Gd = float(np.sum(G_pairs * d_pairs))
    if sum_dd <= 0 or sum_GG <= 0:
        raise ValueError("geodesic_stress_scale_inv: degenerate data (all distances 0).")
    return float(1.0 - (sum_Gd ** 2) / (sum_GG * sum_dd))


def geodesic_pearson(d: np.ndarray, G: np.ndarray) -> float:
    """r_G = Pearson(G_ij, d_ij) over all pairs i<j (B.3). Invariance:
    same as `centroid_pearson`."""
    return _safe_corr(pearsonr, _pairs(d), _pairs(G))


def geodesic_lre(d: np.ndarray, G: np.ndarray) -> float:
    """LRE_G = log_ratio_error(d_ij, G_ij) over all pairs i<j, median
    (B.3). Invariance: same as `log_ratio_error`."""
    return log_ratio_error(_pairs(d), _pairs(G))


def aspect_ratio_error(Y: np.ndarray, T: np.ndarray) -> float:
    """AE = |log AR(Y) - log AR(T)|, AR(Z) = sqrt(lambda1(cov Z)/lambda2(cov Z))
    (B.3: aspect ratio error; lambda1>=lambda2 are the eigenvalues of the
    covariance matrix of Z, Z=(n,2)).

    Invariance: under Z->sRZ+t, cov(Z') = s^2 * R cov(Z) R^T holds. Since R
    is orthogonal, R cov(Z) R^T has the SAME eigenvalues as cov(Z) (only a
    different eigenvector basis), so lambda_i(cov Z') = s^2 * lambda_i(cov Z) -
    in the ratio lambda1'/lambda2' the s^2 cancels exactly, AR(Z') = AR(Z).
    The translation t does not affect the covariance at all. This also
    holds for reflections (det R = -1 does not change the eigenvalues of
    the symmetric matrix R M R^T)."""
    Y = np.asarray(Y, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    if Y.shape[1] != 2 or T.shape[1] != 2:
        raise ValueError(f"aspect_ratio_error expects (n,2) arrays, got shapes {Y.shape} and {T.shape}.")

    def _ar(Z: np.ndarray) -> float:
        cov = np.cov(Z, rowvar=False)
        eigvals = np.linalg.eigvalsh(cov)
        lam2, lam1 = float(eigvals[0]), float(eigvals[1])  # eigvalsh: ascending
        if lam2 <= 0:
            raise ValueError("aspect_ratio_error: degenerate point cloud (smallest covariance eigenvalue <= 0).")
        return float(np.sqrt(lam1 / lam2))

    return float(abs(np.log(_ar(Y)) - np.log(_ar(T))))


def mds_upper_bound_cpcc(u: np.ndarray, n_components: int = 2) -> float:
    """Upper bound of CPCC attainable by CLASSICAL (Torgerson) MDS directly
    on the ultrametric u (K points) - a reference value for S2, where an
    exact 2D embedding of the ultrametric does not exist in general
    (`oracle_truth` is not defined, B.2 S2). A deterministic closed-form
    solution (double centering + eigendecomposition), no randomness.

    Invariance: does not depend on any Y (only on u), so it is trivially
    invariant - tests only verify determinism (same u -> same result)."""
    u = np.asarray(u, dtype=np.float64)
    K = u.shape[0]
    if u.shape != (K, K):
        raise ValueError(f"u must be a square matrix, got {u.shape}.")
    if n_components < 1 or n_components >= K:
        raise ValueError(f"n_components must be in [1,{K-1}], got {n_components}.")
    u2 = u ** 2
    J = np.eye(K) - np.ones((K, K)) / K
    B = -0.5 * J @ u2 @ J
    B = 0.5 * (B + B.T)  # numerical symmetrization
    eigvals, eigvecs = np.linalg.eigh(B)
    order = np.argsort(eigvals)[::-1][:n_components]
    top_vals = np.clip(eigvals[order], a_min=0.0, a_max=None)
    coords = eigvecs[:, order] * np.sqrt(top_vals)[np.newaxis, :]
    delta = squareform(pdist(coords))
    return cophenetic_pearson(delta, u)
