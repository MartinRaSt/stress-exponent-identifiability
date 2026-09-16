# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-09
# License: see the LICENSE file in the repository root
"""Tests for identities of the embedding-quality metrics (src/sammon/metrics.py)."""
from __future__ import annotations

import numpy as np

from src.sammon.metrics import evaluate


def _sample_points(n: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 2))


def test_auc_rnx_is_one_for_identity_embedding() -> None:
    """For Y = X (already 2D), the neighbor order is perfectly preserved, so AUC_RNX = 1."""
    X = _sample_points()
    res = evaluate(X, X, None, "vector")
    assert np.isclose(res["auc_rnx"], 1.0, atol=1e-8)


def test_stress_zero_for_uniformly_scaled_embedding() -> None:
    """The scale-invariant stress must be 0 when Y is just a scaled copy of X."""
    X = _sample_points()
    res = evaluate(X, 3.7 * X, None, "vector")
    assert np.isclose(res["stress_scale_invariant"], 0.0, atol=1e-8)


def test_sammon_stress_zero_for_identity() -> None:
    """The Sammon stress for Y = X must be 0 (no change in distances)."""
    X = _sample_points()
    res = evaluate(X, X, None, "vector")
    assert np.isclose(res["sammon_stress"], 0.0, atol=1e-8)


def test_trustworthiness_continuity_one_for_identity() -> None:
    """Both trustworthiness and continuity must be 1 for perfectly preserved neighborhoods."""
    X = _sample_points()
    res = evaluate(X, X, None, "vector")
    assert np.isclose(res["trustworthiness_k7"], 1.0, atol=1e-8)
    assert np.isclose(res["continuity_k7"], 1.0, atol=1e-8)


def test_evaluate_with_labels_adds_neighborhood_hit_and_silhouette() -> None:
    """When labels y are given, the result must contain neighborhood_hit and silhouette."""
    X = _sample_points()
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, size=X.shape[0])
    res = evaluate(X, X, y, "vector")
    assert "neighborhood_hit_k7" in res
    assert "silhouette" in res
    assert -1.0 <= res["silhouette"] <= 1.0


def test_extended_metrics_identity_embedding() -> None:
    """Task spec test: Y=X (already 2D) must give dCor=1, kNN Jaccard=1, LCMC>0
    (the extended metric set, extended=True)."""
    X = _sample_points()
    res = evaluate(X, X, None, "vector", extended=True)
    assert np.isclose(res["distance_correlation"], 1.0, atol=1e-6)
    assert np.isclose(res["knn_jaccard_k7"], 1.0, atol=1e-8)
    assert res["lcmc_k7"] > 0.0
    assert np.isclose(res["kruskal_stress1"], 0.0, atol=1e-8)
    assert np.isclose(res["shepard_pearson_r"], 1.0, atol=1e-6)
    assert res["q_local"] > 0.0
    assert np.isfinite(res["q_global"])


def test_extended_metrics_with_labels_adds_davies_bouldin_and_dsc() -> None:
    """extended=True with labels must add davies_bouldin and distance_consistency."""
    X = _sample_points()
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, size=X.shape[0])
    res = evaluate(X, X, y, "vector", extended=True)
    assert "davies_bouldin" in res
    assert "distance_consistency" in res
    assert 0.0 <= res["distance_consistency"] <= 1.0
    assert res["davies_bouldin"] >= 0.0


def test_extended_metrics_absent_when_not_requested() -> None:
    """Without extended=True, the extended metrics are not computed (default behavior unchanged)."""
    X = _sample_points()
    res = evaluate(X, X, None, "vector")
    assert "distance_correlation" not in res
    assert "kruskal_stress1" not in res


def _labeled_points(n_per_class: int = 30, n_classes: int = 4, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Generates `n_classes` well-separated clusters of different sizes
    (2D), so the inter-cluster geometry (K2) has a non-trivial structure to preserve."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-10, 10, size=(n_classes, 2))
    scales = np.linspace(0.3, 1.5, n_classes)
    X_parts, y_parts = [], []
    for c in range(n_classes):
        X_parts.append(centers[c] + scales[c] * rng.normal(size=(n_per_class, 2)))
        y_parts.append(np.full(n_per_class, c))
    return np.concatenate(X_parts), np.concatenate(y_parts)


def test_cluster_geometry_identity_gives_rho_one() -> None:
    """K2: for Y=X (identity), both centroid_dist_spearman and
    class_spread_spearman must equal 1 and centroid_knn_preservation 1."""
    X, y = _labeled_points()
    res = evaluate(X, X, y, "vector", extended=True)
    assert np.isclose(res["centroid_dist_spearman"], 1.0, atol=1e-8)
    assert np.isclose(res["class_spread_spearman"], 1.0, atol=1e-8)
    assert np.isclose(res["class_spread_lie_factor"], 0.0, atol=1e-8)
    assert np.isclose(res["centroid_knn_preservation"], 1.0, atol=1e-8)


def test_cluster_geometry_row_permutation_destroys_correlation() -> None:
    """K2: Y = randomly shuffled ROWS of X (same points, but the order no
    longer matches the original class assignment `y`) - from the output's
    point of view the class assignment is essentially random, the
    inter-class/intra-class structure of D_in and D_out no longer correlate
    (rho close to 0)."""
    # 10 classes (instead of 5) - Spearman over just 5 values/10 pairs is
    # too unstable (high variance for small n), 10 classes give a more robust test
    X, y = _labeled_points(n_per_class=25, n_classes=10, seed=1)
    rng = np.random.default_rng(10)
    perm = rng.permutation(X.shape[0])
    Y_scrambled = X[perm]
    res = evaluate(X, Y_scrambled, y, "vector", extended=True)
    assert abs(res["centroid_dist_spearman"]) < 0.5
    assert abs(res["class_spread_spearman"]) < 0.5


def test_cluster_geometry_nan_below_three_classes() -> None:
    """K2: with < 3 classes (or without labels), all 4 metrics must be NaN."""
    X, y = _labeled_points(n_classes=2)
    res = evaluate(X, X, y, "vector", extended=True)
    assert np.isnan(res["centroid_dist_spearman"])
    assert np.isnan(res["class_spread_spearman"])
    assert np.isnan(res["class_spread_lie_factor"])
    assert np.isnan(res["centroid_knn_preservation"])

    res_no_y = evaluate(X, X, None, "vector", extended=True)
    assert np.isnan(res_no_y["centroid_dist_spearman"])
