# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-12
# License: see the LICENSE file in the repository root
"""
K2 (documentation/2026-09-12_plan_smeru_clanku.md) - between-cluster geometry
metrics: how well the projection Y preserves (a) relative distances BETWEEN
classes and (b) the relative SIZE (spread) of each class, relative to the
original distance matrix D.

All functions operate directly on the (n x n) distance matrix D (input) and
d (output, = the Euclidean distance matrix of Y) - NOT on centroids directly
in point space, because D can be a general precomputed distance matrix
(graph, resistance, ...) without an underlying vector space in which a
"centroid" would make sense. The between-class distance is therefore
defined as the MEAN of D_ij over all pairs of points i in class a, j in
class b (a medoid-free variant - see the K2 documentation for a comparison
with the variant computing centroids directly from X, used in the draft
reserse/skripty_20260912_myslitel/thinker_h1b.py for E1 vector data). The
within-class "spread" is the median of within-class pairwise distances
(median, not mean, for robustness to outliers).

Precise definitions/justification will be supplied by `sci-researcher` (R3);
this module implements the formulas exactly per the K2 task specification,
deviations are recorded in the documentation.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

MIN_CLASSES_FOR_GEOMETRY = 3
# Several synthetic datasets (swiss_roll, s_curve, sphere, severed_sphere,
# helix, torus - see src/datasets/synthetic.py) have `y` as a CONTINUOUS
# coordinate on the manifold (position on the roll/angle), not a categorical
# class - np.unique(y) then returns on the order of hundreds of "classes"
# (~n), which is (a) semantically meaningless for "between-cluster" geometry,
# (b) O(n^2) in the number of classes -> prohibitively slow.
# The upper bound distinguishes genuinely categorical labels (max observed
# in E1: olivetti 40, letter/isolet 26) from such continuous pseudo-classes.
MAX_CLASSES_FOR_GEOMETRY = 50

CLUSTER_GEOMETRY_KEYS = [
    "centroid_dist_spearman", "class_spread_spearman",
    "class_spread_lie_factor", "centroid_knn_preservation",
]


def between_class_mean_distance_matrix(D: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (classes, M) where M[a,b] = mean of D_ij over i in class
    classes[a], j in class classes[b] (a != b; the diagonal is NaN - it is
    not "between-class")."""
    classes = np.unique(y)
    k = len(classes)
    idx = [np.where(y == c)[0] for c in classes]
    M = np.full((k, k), np.nan, dtype=np.float64)
    for a in range(k):
        for b in range(k):
            if a == b:
                continue
            M[a, b] = float(D[np.ix_(idx[a], idx[b])].mean())
    return classes, M


def within_class_median_distance(D: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (classes, spread) where spread[a] = median of within-class
    pairwise distances of class classes[a] (NaN for a class with < 2 points)."""
    classes = np.unique(y)
    spread = np.full(len(classes), np.nan, dtype=np.float64)
    for a, c in enumerate(classes):
        idx = np.where(y == c)[0]
        if len(idx) < 2:
            continue
        sub = D[np.ix_(idx, idx)]
        triu = sub[np.triu_indices(len(idx), k=1)]
        spread[a] = float(np.median(triu))
    return classes, spread


def centroid_dist_spearman(D_in: np.ndarray, D_out: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation between the vector of between-class distances
    (upper triangle of the `between_class_mean_distance_matrix`) in the
    input and output. NaN if < 3 classes (< 3 independent class pairs)."""
    classes, M_in = between_class_mean_distance_matrix(D_in, y)
    if len(classes) < MIN_CLASSES_FOR_GEOMETRY:
        return float("nan")
    _, M_out = between_class_mean_distance_matrix(D_out, y)
    triu_idx = np.triu_indices(len(classes), k=1)
    v_in, v_out = M_in[triu_idx], M_out[triu_idx]
    if len(v_in) < 3:
        return float("nan")
    rho, _ = spearmanr(v_in, v_out)
    return float(rho)


def class_spread_spearman(D_in: np.ndarray, D_out: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation between the median within-class distance (per
    class) in the input and output. NaN if < 3 classes have a defined spread."""
    classes, spread_in = within_class_median_distance(D_in, y)
    if len(classes) < MIN_CLASSES_FOR_GEOMETRY:
        return float("nan")
    _, spread_out = within_class_median_distance(D_out, y)
    valid = np.isfinite(spread_in) & np.isfinite(spread_out)
    if valid.sum() < 3:
        return float("nan")
    rho, _ = spearmanr(spread_in[valid], spread_out[valid])
    return float(rho)


def class_spread_lie_factor(D_in: np.ndarray, D_out: np.ndarray, y: np.ndarray) -> float:
    """Median over classes of |log((spread_out/scale_out)/(spread_in/scale_in))|,
    where scale = median of ALL pairwise distances (input resp. output) -
    i.e. the "lie factor" (Tufte) of cluster size after normalizing to the
    global scale. NaN if no class has a defined nonzero spread on both sides."""
    classes, spread_in = within_class_median_distance(D_in, y)
    if len(classes) < MIN_CLASSES_FOR_GEOMETRY:
        return float("nan")
    _, spread_out = within_class_median_distance(D_out, y)

    n_in = D_in.shape[0]
    n_out = D_out.shape[0]
    scale_in = float(np.median(D_in[np.triu_indices(n_in, k=1)]))
    scale_out = float(np.median(D_out[np.triu_indices(n_out, k=1)]))
    if scale_in <= 0 or scale_out <= 0:
        return float("nan")

    ratios = []
    for si, so in zip(spread_in, spread_out):
        if not (np.isfinite(si) and np.isfinite(so)) or si <= 0 or so <= 0:
            continue
        rel_in = si / scale_in
        rel_out = so / scale_out
        ratios.append(abs(float(np.log(rel_out / rel_in))))
    if not ratios:
        return float("nan")
    return float(np.median(ratios))


def centroid_knn_preservation(D_in: np.ndarray, D_out: np.ndarray, y: np.ndarray, k_neighbors: int = 3) -> float:
    """Fraction of classes for which the SET of the `k_neighbors` nearest
    classes (by mean between-class distance) is exactly preserved between
    input and output. Requires at least k_neighbors+1 classes (otherwise
    "k nearest among the rest" is not nontrivial), else NaN."""
    classes, M_in = between_class_mean_distance_matrix(D_in, y)
    n_classes = len(classes)
    if n_classes < k_neighbors + 1:
        return float("nan")
    _, M_out = between_class_mean_distance_matrix(D_out, y)

    preserved = 0
    for a in range(n_classes):
        row_in = M_in[a].copy()
        row_out = M_out[a].copy()
        row_in[a] = np.inf
        row_out[a] = np.inf
        top_in = set(np.argsort(row_in, kind="stable")[:k_neighbors].tolist())
        top_out = set(np.argsort(row_out, kind="stable")[:k_neighbors].tolist())
        if top_in == top_out:
            preserved += 1
    return float(preserved / n_classes)


def cluster_geometry_metrics(D_in: np.ndarray, D_out: np.ndarray, y: np.ndarray | None, k_neighbors: int = 3) -> dict[str, float]:
    """Compute all 4 between-cluster geometry metrics at once. Returns a dict
    with all keys NaN if y is None, or has < MIN_CLASSES_FOR_GEOMETRY or
    > MAX_CLASSES_FOR_GEOMETRY unique values (a fallback value is never
    fabricated - see CLUSTER_GEOMETRY_KEYS, MAX_CLASSES_FOR_GEOMETRY)."""
    if y is None:
        return {k: float("nan") for k in CLUSTER_GEOMETRY_KEYS}
    y = np.asarray(y)
    n_classes = len(np.unique(y))
    if not (MIN_CLASSES_FOR_GEOMETRY <= n_classes <= MAX_CLASSES_FOR_GEOMETRY):
        return {k: float("nan") for k in CLUSTER_GEOMETRY_KEYS}
    return {
        "centroid_dist_spearman": centroid_dist_spearman(D_in, D_out, y),
        "class_spread_spearman": class_spread_spearman(D_in, D_out, y),
        "class_spread_lie_factor": class_spread_lie_factor(D_in, D_out, y),
        "centroid_knn_preservation": centroid_knn_preservation(D_in, D_out, y, k_neighbors=k_neighbors),
    }
