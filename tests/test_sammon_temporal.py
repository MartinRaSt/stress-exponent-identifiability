# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-09
# License: see the LICENSE file in the repository root
"""Tests for section 10 (the temporal extension, section 6): a large lambda
-> Y_t close to the anchors; lambda=0 == an independent fit of each snapshot
(without anchoring)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

from src.sammon.solvers.smacof import smacof_solve
from src.sammon.temporal import TemporalSammon
from src.sammon.weights import alpha_weights, compute_Z_from_W, estimate_eps_D
from src.sammon.init import init_classical_mds

_PRIMARY_SCHOOL_CSV = Path(__file__).resolve().parents[1] / "src" / "data" / "temporal" / "primaryschool.csv.gz"


def _make_snapshots(seed: int, n: int = 25):
    rng = np.random.default_rng(seed)
    X0 = rng.normal(size=(n, 5))
    X1 = X0 + rng.normal(scale=0.05, size=(n, 5))
    D0 = squareform(pdist(X0))
    D1 = squareform(pdist(X1))
    ids = list(range(n))
    return [D0, D1], [ids, ids]


def test_large_lambda_keeps_Y_close_to_anchors() -> None:
    """lambda -> large: Y_t (for shared nodes) must approach the anchors a_i=y_i(t-1)."""
    list_of_D, ids = _make_snapshots(seed=0)
    ts = TemporalSammon(n_components=2, max_iter=200, tol=1e-6, seed=0)
    ts.fit(list_of_D, ids, lam=1.0e6, alpha=1.0)

    Y0, Y1 = ts.Y_list_
    disp = np.linalg.norm(Y1 - Y0, axis=1)
    assert disp.max() < 1e-3, f"For a large lambda, Y_t should practically coincide with the anchors, max displacement = {disp.max()}."


def test_lambda_zero_matches_independent_fit() -> None:
    """lambda=0: temporal SMACOF on a snapshot t>0 must give the same result
    as an independent (non-temporal) SMACOF fit on D_t with the same init."""
    list_of_D, ids = _make_snapshots(seed=1)
    D0, D1 = list_of_D
    n = D1.shape[0]

    ts = TemporalSammon(n_components=2, max_iter=200, tol=1e-8, seed=0)
    ts.fit(list_of_D, ids, lam=0.0, alpha=1.0)
    Y1_temporal = ts.Y_list_[1]

    # independent fit: same Y0 initialization as in TemporalSammon.fit for
    # t>0 with lam=0 (anchors=0, mask=0) - see src/sammon/temporal.py
    eps_D = estimate_eps_D(D1, k=5, q=0.1, kind="distance")
    W = alpha_weights(D1, 1.0, eps_D)
    Z = compute_Z_from_W(D1, W)
    prev_Y = ts.Y_list_[0]
    Y0_init = prev_Y  # continuing nodes start from the t-1 position (see fit())
    mask = np.zeros(n)
    anchors = np.zeros((n, 2))
    Y1_independent, _ = smacof_solve(
        D1, W, Z, Y0_init, max_iter=200, tol=1e-8, eps_num=1e-9,
        dense_pinv_threshold=5000, cg_max_iter=500, cg_tol=1e-8, reg_rho=1e-8,
        anchors=anchors, mask=mask, lam=0.0, verbose=False,
    )

    np.testing.assert_allclose(Y1_temporal, Y1_independent, rtol=1e-8, atol=1e-8)


def test_stability_monotone_in_lambda() -> None:
    """A larger lambda should give lower (or equal) stability (smaller
    displacement) - a qualitative trade-off check (section 6.5), not an exact
    value."""
    list_of_D, ids = _make_snapshots(seed=2)
    stabs = []
    for lam in [0.0, 1.0, 100.0]:
        ts = TemporalSammon(n_components=2, max_iter=150, tol=1e-4, seed=0)
        ts.fit(list_of_D, ids, lam=lam, alpha=1.0)
        stabs.append(ts.stability())
    assert stabs[0] >= stabs[1] >= stabs[2], f"Stability is not monotone in lambda: {stabs}"


def test_sgd_large_lambda_keeps_Y_close_to_anchors() -> None:
    """Section 6.3 (the SGD anchor step): the same test as for SMACOF (a large
    lambda -> Y_t close to the anchors), but with `solver='sgd'`. The anchor
    step is an explicit (gradient) update `y_i <- y_i - gamma_t*lambda*m_i*(y_i-a_i)`,
    so `gamma_mode='fixed'` is used with a small constant (gamma_fixed*lambda
    < 1 for the stability of the explicit Euler step, see the comment on
    sgd_solve)."""
    list_of_D, ids = _make_snapshots(seed=0)
    ts = TemporalSammon(
        n_components=2, seed=0, solver="sgd", sgd_epochs=200,
        gamma_mode="fixed", gamma_fixed=0.05,
    )
    ts.fit(list_of_D, ids, lam=15.0, alpha=1.0)

    Y0, Y1 = ts.Y_list_
    assert np.isfinite(Y1).all()
    disp = np.linalg.norm(Y1 - Y0, axis=1)
    assert disp.max() < 1e-2, f"For a large lambda, Y_t should approach the anchors, max displacement = {disp.max()}."


def test_sgd_lambda_zero_matches_no_anchor_sgd() -> None:
    """Section 6.3: lam=0 => `sgd_solve` with anchors/mask has no effect (the
    same result as without anchors at all, given the same pair/order seed)."""
    from src.sammon.solvers.sgd import sgd_solve
    from src.sammon.weights import alpha_weights, compute_Z_from_W, estimate_eps_D

    list_of_D, ids = _make_snapshots(seed=1)
    D0, D1 = list_of_D
    n = D1.shape[0]

    ts = TemporalSammon(n_components=2, seed=0, solver="sgd", sgd_epochs=80)
    ts.fit(list_of_D, ids, lam=0.0, alpha=1.0)
    Y1_temporal = ts.Y_list_[1]

    eps_D = estimate_eps_D(D1, k=5, q=0.1, kind="distance")
    Y0_init = ts.Y_list_[0]  # continuing nodes start from the t-1 position (see fit())
    mask = np.zeros(n)
    anchors = np.zeros((n, 2))
    Y1_no_anchor, _ = sgd_solve(
        D1, alpha=1.0, Y0=Y0_init, epochs=80, mu_max=0.5, pairs_per_node=60,
        eps_anneal=0.01, eps_num=1e-9, seed=ts.seed + 1, variant="stabilized",
        anchors=anchors, mask=mask, lam=0.0, verbose=False,
    )
    np.testing.assert_allclose(Y1_temporal, Y1_no_anchor, rtol=1e-10, atol=1e-10)


def test_new_node_initialized_near_neighbors_centroid() -> None:
    """A new node (without a position in t-1) should be initialized at the
    centroid of neighbors that already existed in t-1 (section 6.4) - we
    verify this indirectly: the new node ends up closer to its neighbors than
    to a random distant point."""
    rng = np.random.default_rng(3)
    n = 20
    X0 = rng.normal(size=(n, 5))
    X1 = np.vstack([X0 + rng.normal(scale=0.02, size=(n, 5)), X0[0:1] + 0.01])  # new point close to point 0
    D0 = squareform(pdist(X0))
    D1 = squareform(pdist(X1))
    ids0 = list(range(n))
    ids1 = list(range(n)) + [999]

    ts = TemporalSammon(n_components=2, max_iter=200, tol=1e-6, seed=0)
    ts.fit([D0, D1], [ids0, ids1], lam=0.5, alpha=1.0)
    Y1 = ts.Y_list_[1]

    dist_to_all = np.linalg.norm(Y1[:-1] - Y1[-1], axis=1)
    nearest = np.argmin(dist_to_all)
    assert nearest == 0, f"The new node should end up closest to node 0 (its actual neighbor), but the nearest is {nearest}."


def test_new_node_jitter_breaks_exact_duplicate_init() -> None:
    """K14 (documentation/2026-09-13_k14_uklid_hardening.md): a snapshot with
    very few (in this test exactly 1) continuing nodes must not cause ALL new
    nodes to collapse into an identical Y0 - without jitter, the SGD/Gauss-
    Seidel pairwise update (direction vector Y_i-Y_j) would NEVER split
    exactly coincident coordinates (real incident: invs13_temporal t=15, 13
    nodes, only 1 continuing -> the whole snapshot collapsed into a single
    point, stability() correctly but needlessly raised 'average distance
    <= 0')."""
    rng = np.random.default_rng(7)
    n0, n1 = 12, 13
    D0 = squareform(pdist(rng.normal(size=(n0, 5))))
    D1 = squareform(pdist(rng.normal(size=(n1, 5))))
    ids0 = list(range(n0))
    ids1 = [0] + list(range(n0, n0 + n1 - 1))  # only ONE shared node (id=0)

    ts = TemporalSammon(
        n_components=2, solver="sgd", sgd_epochs=100, seed=0, gamma_mode="fixed", gamma_fixed=0.1,
    )
    ts.fit([D0, D1], [ids0, ids1], lam=1.0, alpha=1.0)
    Y1 = ts.Y_list_[1]
    assert np.isfinite(Y1).all()
    n_unique = len({tuple(row) for row in np.round(Y1, 9)})
    assert n_unique == n1, f"All nodes should end up at distinct coordinates, only {n_unique}/{n1} are unique."
    # stability() used to raise ValueError ("average distance <= 0") because
    # the embedding collapsed into a single point - after the fix it must pass without an exception.
    assert ts.stability() >= 0.0


def test_smacof_solve_raises_when_lam_positive_and_no_anchors() -> None:
    """Fail-loud (documentation/2026-09-11_kontrola_vysledku_plnych_behu.md,
    section 3.1): lam > 0 with mask.sum() == 0 means a singular (V+lam*M) -
    `smacof_solve` must raise ValueError instead of silently continuing with
    a bad result."""
    rng = np.random.default_rng(0)
    n = 15
    D = squareform(pdist(rng.normal(size=(n, 5))))
    eps_D = estimate_eps_D(D, k=5, q=0.1, kind="distance")
    W = alpha_weights(D, 1.0, eps_D)
    Z = compute_Z_from_W(D, W)
    Y0 = init_classical_mds(D, 2, 0)
    mask = np.zeros(n)
    anchors = np.zeros((n, 2))

    with pytest.raises(ValueError, match="mask.sum"):
        smacof_solve(
            D, W, Z, Y0, max_iter=50, tol=1e-6, eps_num=1e-9,
            dense_pinv_threshold=5000, cg_max_iter=500, cg_tol=1e-6, reg_rho=1e-8,
            anchors=anchors, mask=mask, lam=0.1, verbose=False,
        )


def test_temporal_disjoint_snapshot_falls_back_to_lam_zero(caplog) -> None:
    """Section 3.1: a snapshot with NO shared node with the previous one
    (mask.sum()==0) must use lam_t=0 (a WARNING in the log) and give the same
    result as an independent fit with lam=0 (tolerance 1e-9), not a singular
    result (stress 0.35, |Y|~1e7)."""
    rng = np.random.default_rng(4)
    n0, n1 = 12, 14
    D0 = squareform(pdist(rng.normal(size=(n0, 5))))
    D1 = squareform(pdist(rng.normal(size=(n1, 5))))
    ids0 = list(range(n0))
    ids1 = list(range(n0, n0 + n1))  # no id shared with ids0 (disjoint snapshots)

    with caplog.at_level("WARNING", logger="sammon.temporal"):
        ts = TemporalSammon(n_components=2, max_iter=300, tol=1e-8, seed=0)
        ts.fit([D0, D1], [ids0, ids1], lam=0.1, alpha=1.0)
    assert any("mask.sum()==0" in rec.message for rec in caplog.records), "A WARNING about using lam_t=0 was expected."

    Y1_temporal = ts.Y_list_[1]
    stress_temporal = ts.history_list_[1]["stress"][-1]
    assert np.abs(Y1_temporal).max() < 100.0, f"|Y| should not diverge, max|Y|={np.abs(Y1_temporal).max()}."

    # reference independent fit with lam=0, using the SAME Y0 initialization
    # that TemporalSammon.fit uses for a snapshot without continuing nodes
    # (random initialization around 0 with scale=0.1*average distance, see
    # fit()) - only this way can 1e-9 agreement be expected (SMACOF is
    # non-convex, different Y0 = different local minimum)
    eps_D = estimate_eps_D(D1, k=5, q=0.1, kind="distance")
    W = alpha_weights(D1, 1.0, eps_D)
    Z = compute_Z_from_W(D1, W)
    scale = float(D1[~np.eye(n1, dtype=bool)].mean())
    rng_ref = np.random.default_rng(ts.seed + 1)
    Y0_ref = np.stack([rng_ref.normal(scale=0.1 * scale, size=2) for _ in range(n1)])
    mask = np.zeros(n1)
    anchors = np.zeros((n1, 2))
    _, hist_ref = smacof_solve(
        D1, W, Z, Y0_ref, max_iter=300, tol=1e-8, eps_num=1e-9,
        dense_pinv_threshold=5000, cg_max_iter=500, cg_tol=1e-6, reg_rho=1e-8,
        anchors=anchors, mask=mask, lam=0.0, verbose=False,
    )
    assert abs(stress_temporal - hist_ref["stress"][-1]) < 1e-9, (
        f"The stress of a snapshot without shared nodes ({stress_temporal}) should match "
        f"the lam=0 fit with the same Y0 ({hist_ref['stress'][-1]})."
    )


@pytest.mark.skipif(not _PRIMARY_SCHOOL_CSV.exists(), reason="primaryschool.csv.gz is not available locally.")
def test_temporal_real_primary_school_disjoint_snapshots(caplog) -> None:
    """A real scenario from section 3.1: primary_school_temporal,
    time_bin_sec=3600, min_snapshot_nodes=20 - snapshots 9 and 10 have no
    shared node (an overnight gap). `TemporalSammon.fit` with the smacof
    solver for lambda=0.1 must not end with a stress ~0.35/|Y|~1e7 (the
    original bug), but with a stress matching lam=0."""
    from src.experiments.exp4_temporal import _load_snapshots

    list_of_D, ids = _load_snapshots("primary_school_temporal", 3600, 20)
    assert len(set(ids[9]) & set(ids[10])) == 0, "Test assumption: snapshots 9 and 10 have no shared node."

    with caplog.at_level("WARNING", logger="sammon.temporal"):
        ts = TemporalSammon(n_components=2, max_iter=300, tol=1e-4, seed=0)
        ts.fit(list_of_D[9:11], ids[9:11], lam=0.1, alpha=1.0)
    assert any("mask.sum()==0" in rec.message for rec in caplog.records)

    stress_snap10 = ts.history_list_[1]["stress"][-1]
    Y10 = ts.Y_list_[1]
    assert stress_snap10 < 0.1, f"The stress of snapshot 10 should not be singular (~0.35), it is {stress_snap10}."
    assert np.abs(Y10).max() < 100.0, f"|Y| of snapshot 10 should not diverge, max|Y|={np.abs(Y10).max()}."
