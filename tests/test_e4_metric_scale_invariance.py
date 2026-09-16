# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-14
# License: see the LICENSE file in the repository root
"""K12 (E4 dtsne baseline, documentation/2026-09-14_e4_dtsne_baseline.md):
`dtsne_lambda<L>` (src/sammon/dynamic_tsne.py) produces embeddings on a
different global scale than TemporalSammon (SMACOF/SGD), so the reported E4
metrics `stab` (stab_norm) and `qual` (qual_si) MUST be invariant to a
global similarity transformation Y -> c*R*Y + b (c>0 scale, R an orthogonal
rotation/reflection, b a translation), applied IDENTICALLY to every snapshot
of the time series - otherwise the dtsne vs. Sammon family comparison would
be distorted by the choice of output scale, not by the actual embedding
quality/stability.

Tested directly on the canonical functions used by both method families:
- `src.sammon.temporal_metrics.stability` (stab_norm, per-transition disp/L_t)
- `src.sammon.metrics.scale_invariant_stress` (qual_si, with the optimal scale s*)

and additionally end-to-end on real `DynamicTSNE.fit()` output, so the test
also covers the real path (not just synthetic data with the same structure as above).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

from src.sammon.dynamic_tsne import DynamicTSNE
from src.sammon.metrics import scale_invariant_stress
from src.sammon.temporal_metrics import stability

# the transformation used in all tests: scale c!=1, an orthogonal R with
# reflection (det<0, so the test also covers mirroring - Procrustes in
# `stability` also allows reflection, see orthogonal_procrustes_no_scale), translation b
_C = 3.7
_THETA = 0.83
_R = np.array([[np.cos(_THETA), np.sin(_THETA)], [np.sin(_THETA), -np.cos(_THETA)]])  # det = -1
_B = np.array([12.5, -4.2])


def _transform(Y: np.ndarray) -> np.ndarray:
    return _C * (Y @ _R.T) + _B


def _make_dynamic_snapshots(seed: int, n0: int = 18, T: int = 4):
    """A synthetic time series with both departing and new nodes between
    snapshots (so the test covers the real E4 situation, not just the
    trivial case without a change in the node set) - distances from random
    points in R^5 (a valid metric)."""
    rng = np.random.default_rng(seed)
    ids_pool = list(range(n0 + T))  # enough extra ids for incoming nodes
    X = rng.normal(size=(n0 + T, 5))

    list_of_D, node_ids = [], []
    active = list(range(n0))
    for t in range(T):
        if t > 0:
            active = active[1:] + [n0 + t - 1]  # one leaves, one new arrives (sliding window)
            X[active[-1]] = X[active[-2]] + rng.normal(scale=0.3, size=5)  # a new node close to a neighbor
        ids_t = [ids_pool[i] for i in active]
        D_t = squareform(pdist(X[active]))
        list_of_D.append(D_t)
        node_ids.append(ids_t)
    return list_of_D, node_ids


def test_scale_invariant_stress_invariant_to_global_similarity_transform() -> None:
    rng = np.random.default_rng(0)
    n = 25
    D = squareform(pdist(rng.normal(size=(n, 6))))
    Y = rng.normal(size=(n, 2))
    d = squareform(pdist(Y))
    qual_before = scale_invariant_stress(D, d)

    Y_t = _transform(Y)
    d_t = squareform(pdist(Y_t))
    qual_after = scale_invariant_stress(D, d_t)

    assert np.isclose(qual_before, qual_after, rtol=1e-10, atol=1e-12)


def test_stability_invariant_to_global_similarity_transform() -> None:
    list_of_D, node_ids = _make_dynamic_snapshots(seed=1)
    rng = np.random.default_rng(2)
    Y_list = [rng.normal(size=(D_t.shape[0], 2)) for D_t in list_of_D]

    stab_before = stability(Y_list, node_ids, procrustes_align=True)
    Y_list_t = [_transform(Y_t) for Y_t in Y_list]
    stab_after = stability(Y_list_t, node_ids, procrustes_align=True)

    assert np.isclose(stab_before, stab_after, rtol=1e-9, atol=1e-12)


def test_stability_not_invariant_to_per_snapshot_independent_scale() -> None:
    """A control negative case: if each snapshot were scaled by a DIFFERENT
    c_t (not the same global transformation), invariance would not hold -
    confirms that the test above actually verifies a non-trivial property
    (correct per-snapshot normalization via L_t), not just numerical
    agreement due to a bug in the test."""
    list_of_D, node_ids = _make_dynamic_snapshots(seed=3)
    rng = np.random.default_rng(4)
    Y_list = [rng.normal(size=(D_t.shape[0], 2)) for D_t in list_of_D]
    stab_before = stability(Y_list, node_ids, procrustes_align=True)

    scales = [1.0, 5.0, 0.2, 9.0]
    Y_list_scaled = [s * Y_t for s, Y_t in zip(scales, Y_list)]
    stab_scaled = stability(Y_list_scaled, node_ids, procrustes_align=True)

    assert not np.isclose(stab_before, stab_scaled, rtol=1e-6)


@pytest.mark.parametrize("lam_dt", [0.0, 1.0])
def test_dynamic_tsne_stability_and_quality_invariant_end_to_end(lam_dt: float) -> None:
    """End-to-end on a real DynamicTSNE.fit() (small n/small max_iter for
    test speed) - only the OUTPUT (Y_list_) is transformed, the fit is not
    rerun, because t-SNE optimization is not invariant by itself (a
    different initialization/optimizer trajectory for a different data
    scale could converge to a different local minimum) - what is verified
    is the invariance of the METRIC to rescaling/rotating ITS INPUT (the
    embedding), not invariance of the optimization process."""
    dtsne_cfg = {
        "early_exaggeration": 4.0, "early_exaggeration_iter": 20, "momentum_init": 0.5,
        "momentum_final": 0.8, "momentum_switch_iter": 20, "learning_rate_min": 10.0,
        "min_gain": 0.01, "eps_num": 1e-12, "perplexity_tol": 1e-5, "perplexity_max_iter": 50,
        "new_node_k_near": 3, "new_node_jitter_scale": 1e-3, "new_node_random_scale": 0.1,
    }
    list_of_D, node_ids = _make_dynamic_snapshots(seed=5, n0=10, T=3)
    ts = DynamicTSNE(n_components=2, perplexity=3.0, max_iter=40, seed=0, dtsne_cfg=dtsne_cfg)
    ts.fit(list_of_D, node_ids, lam_dt=lam_dt)

    stab_before = ts.stability()
    qual_before = ts.quality()

    ts.Y_list_ = [_transform(Y_t) for Y_t in ts.Y_list_]
    for hist, D_t, Y_t in zip(ts.history_list_, list_of_D, ts.Y_list_):
        hist["stress"] = [scale_invariant_stress(D_t, squareform(pdist(Y_t)))]

    stab_after = ts.stability()
    qual_after = ts.quality()

    assert np.isclose(stab_before, stab_after, rtol=1e-8, atol=1e-10)
    assert np.isclose(qual_before, qual_after, rtol=1e-8, atol=1e-10)
