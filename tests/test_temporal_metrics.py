# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-09
# License: see the LICENSE file in the repository root
"""Tests for the temporal metrics (src/sammon/temporal_metrics.py): the
smoothness of a linear trajectory = 0, mental map preservation for an
unchanged configuration = 1, stability agreement with
TemporalSammon.stability()."""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform

from src.sammon.temporal import TemporalSammon
from src.sammon.temporal_metrics import mental_map_preservation, stability, trajectory_smoothness


def test_trajectory_smoothness_linear_motion_is_zero() -> None:
    """Exactly linear (constant-velocity) motion has a zero 2nd difference."""
    n = 10
    T = 6
    rng = np.random.default_rng(0)
    Y0 = rng.normal(size=(n, 2))
    velocity = rng.normal(scale=0.3, size=(n, 2))
    Y_list = [Y0 + t * velocity for t in range(T)]
    ids = list(range(n))
    node_ids_list = [ids] * T

    smooth = trajectory_smoothness(Y_list, node_ids_list)
    assert np.isclose(smooth, 0.0, atol=1e-10)


def test_trajectory_smoothness_positive_for_nonlinear_motion() -> None:
    """Non-linear (e.g. quadratically accelerating) motion has a positive smoothness."""
    n = 8
    T = 6
    rng = np.random.default_rng(1)
    Y0 = rng.normal(size=(n, 2))
    accel = rng.normal(scale=0.2, size=(n, 2))
    Y_list = [Y0 + (t ** 2) * accel for t in range(T)]
    ids = list(range(n))
    node_ids_list = [ids] * T

    smooth = trajectory_smoothness(Y_list, node_ids_list)
    assert smooth > 1e-6


def test_mental_map_preservation_unchanged_layout_is_one() -> None:
    """When the layout does not change at all between snapshots, the K
    nearest neighbors are perfectly preserved (mental map preservation = 1)."""
    n = 15
    rng = np.random.default_rng(2)
    Y = rng.normal(size=(n, 2))
    ids = list(range(n))
    Y_list = [Y.copy(), Y.copy(), Y.copy()]
    node_ids_list = [ids, ids, ids]

    pres = mental_map_preservation(Y_list, node_ids_list, k=3)
    assert np.isclose(pres, 1.0, atol=1e-8)


def test_stability_function_matches_temporal_sammon_method() -> None:
    """`temporal_metrics.stability` is the canonical implementation also used
    inside `TemporalSammon.stability()` - it must give an identical result."""
    rng = np.random.default_rng(3)
    n = 20
    X0 = rng.normal(size=(n, 5))
    X1 = X0 + rng.normal(scale=0.05, size=(n, 5))
    D0, D1 = squareform(pdist(X0)), squareform(pdist(X1))
    ids = list(range(n))

    ts = TemporalSammon(n_components=2, max_iter=150, tol=1e-4, seed=0)
    ts.fit([D0, D1], [ids, ids], lam=0.3, alpha=1.0)

    stab_method = ts.stability()
    stab_direct = stability(ts.Y_list_, ts.node_ids_, procrustes_align=True)
    assert np.isclose(stab_method, stab_direct, atol=1e-12)
