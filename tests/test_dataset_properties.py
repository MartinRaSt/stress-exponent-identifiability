# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-12
# License: see the LICENSE file in the repository root
"""Tests for the pure numerical K1 functions (src/experiments/dataset_properties.py)."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

from src.experiments.dataset_properties import (
    compute_dataset_properties,
    dist_cv,
    id_mle,
    id_twonn,
    nn_distances_from_D,
    nn_ratio,
    relative_contrast,
)


def _regular_grid(side: int = 20, step: float = 1.0) -> np.ndarray:
    """A regular 2D grid of points (side x side), spacing `step`."""
    xs, ys = np.meshgrid(np.arange(side) * step, np.arange(side) * step)
    return np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)


def _uniform_2d(n: int = 3000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=(n, 2))


def test_nn_ratio_on_regular_grid_matches_known_value() -> None:
    """On a regular grid with spacing 1, the distance to the nearest
    neighbor is exactly 1 for all interior points (the median is still 1,
    because most points are interior) - nn_ratio_k1 = 1 / median(all distances)."""
    X = _regular_grid(side=20, step=1.0)
    D = squareform(pdist(X))
    nn1 = nn_distances_from_D(D, 1)
    assert np.allclose(nn1, 1.0), "all grid points have their nearest neighbor at exactly distance 1"
    median_all = float(np.median(D[np.triu_indices(D.shape[0], k=1)]))
    ratio = nn_ratio(D, 1, median_all)
    assert np.isclose(ratio, 1.0 / median_all, atol=1e-9)


def test_id_twonn_on_uniform_2d_is_approximately_two() -> None:
    """TwoNN (Facco et al. 2017) on uniformly distributed 2D data must give
    an intrinsic-dimension estimate close to 2 (+-0.3, see the K1 task spec)."""
    X = _uniform_2d(n=3000, seed=0)
    D = squareform(pdist(X))
    est = id_twonn(D, fraction=0.9)
    assert abs(est - 2.0) < 0.3, f"TwoNN ID on 2D uniform data expected ~2, got {est}"


def test_id_mle_on_uniform_2d_is_approximately_two() -> None:
    """Levina-Bickel MLE (k=10) on uniform 2D data expected ~2."""
    X = _uniform_2d(n=3000, seed=1)
    D = squareform(pdist(X))
    est = id_mle(D, k=10)
    assert abs(est - 2.0) < 0.5, f"MLE ID on 2D uniform data expected ~2, got {est}"


def test_dist_cv_zero_for_equidistant_points() -> None:
    """Points on a regular simplex (all pairwise distances equal) have CV=0."""
    n = 5
    # a simple simplex: n points mutually at the same distance (a regular
    # simplex in (n-1)-dimensional space) - it suffices to generate the
    # distance matrix directly
    D = np.ones((n, n)) - np.eye(n)
    assert np.isclose(dist_cv(D[np.triu_indices(n, k=1)]), 0.0, atol=1e-12)


def test_relative_contrast_zero_when_all_distances_equal() -> None:
    """When all off-diagonal distances are equal, max=min for every row, so
    relative_contrast=0 (Aggarwal et al. 2001)."""
    n = 6
    D = np.ones((n, n)) - np.eye(n)
    assert np.isclose(relative_contrast(D, eps=1e-9), 0.0, atol=1e-9)


def test_compute_dataset_properties_fails_loud_on_degenerate_input() -> None:
    """All points identical (median of all distances = 0) must raise an
    exception, not return NaN/a substitute value (the project's fail-loud rule)."""
    n = 10
    D = np.zeros((n, n))
    with pytest.raises(ValueError):
        compute_dataset_properties(
            D, None, None, "vector",
            nn_ratio_ks=[1], id_mle_k=2, hubness_k=1, twonn_fraction=0.9, eps=1e-9,
        )


def test_compute_dataset_properties_returns_nan_silhouette_without_labels() -> None:
    """Without labels (y=None), silhouette_input must be NaN, not a fabricated value."""
    X = _uniform_2d(n=200, seed=2)
    D = squareform(pdist(X))
    row = compute_dataset_properties(
        D, X, None, "vector",
        nn_ratio_ks=[1, 5], id_mle_k=10, hubness_k=10, twonn_fraction=0.9, eps=1e-9,
    )
    assert np.isnan(row["silhouette_input"])
    assert np.isnan(row["n_classes"])
    assert row["nn_ratio_k1"] > 0.0
    assert row["nn_ratio_k5"] > 0.0
