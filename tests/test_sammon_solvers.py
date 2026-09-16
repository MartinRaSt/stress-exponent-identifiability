# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-09
# License: see the LICENSE file in the repository root
"""Tests for sections 10.2-10.5: alpha=0 SMACOF vs. sklearn MDS, alpha=1 vs.
the reference Sammon 1969 (sammon_classic), monotonicity of the majorization,
sparse with k=n-1 (no pivots) == the full model."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS

from src.datasets.registry import load_dataset
from src.methods.common import to_distance_matrix
from src.methods.sammon_classic import sammon_mapping
from src.sammon.init import init_random
from src.sammon.solvers.newton import newton_solve
from src.sammon.solvers.smacof import smacof_solve
from src.sammon.solvers.sparse import build_sparse_terms, sparse_smacof_solve
from src.sammon.weights import alpha_weights, compute_Z_from_W, estimate_eps_D


def _random_D(seed: int, n: int = 40, d: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return squareform(pdist(rng.normal(size=(n, d))))


def test_smacof_alpha0_matches_sklearn_mds() -> None:
    """Test 10.2: alpha=0 SMACOF (w_ij=1) == sklearn MDS(precomputed), same init."""
    D = _random_D(seed=0, n=40)
    n = D.shape[0]
    Y0 = init_random(n, 2, seed=1, scale=0.5)

    W = alpha_weights(D, 0.0, eps_D=1.0)
    Z = compute_Z_from_W(D, W)
    Y_mine, _ = smacof_solve(
        D, W, Z, Y0.copy(), max_iter=300, tol=1e-8, eps_num=1e-9,
        dense_pinv_threshold=5000, cg_max_iter=500, cg_tol=1e-8, reg_rho=1e-8, verbose=False,
    )
    d_mine = squareform(pdist(Y_mine))
    sigma_mine = 0.5 * ((D - d_mine) ** 2).sum()

    model = MDS(n_components=2, metric_mds=True, metric="precomputed", n_init=1,
                max_iter=300, eps=1e-8, random_state=0, normalized_stress=False)
    Y_sklearn = model.fit_transform(D, init=Y0.copy())

    rel_diff = abs(sigma_mine - model.stress_) / model.stress_
    assert rel_diff < 1e-3, f"Relative stress difference {rel_diff} >= 1e-3."

    _, _, disparity = procrustes(Y_mine, Y_sklearn)
    scale = float(np.linalg.norm(Y_mine - Y_mine.mean(axis=0)))
    assert disparity < 1e-3 * scale


def test_newton_alpha1_matches_reference_sammon_classic() -> None:
    """Test 10.3: alpha=1 pseudo-Newton == an independent reference implementation
    (src/methods/sammon_classic.py, Sammon 1969) with the SAME initialization.

    n=20 is used (an Iris subset without exact duplicates) - larger subsets
    (n>=50) contain near-coincident points and the fixed-step pseudo-Newton
    is then sensitive to the floating-point order of operations (chaotic
    dynamics, both implementations diverge numerically even with identical
    mathematics) - documented, not an implementation bug.
    """
    ds = load_dataset("iris")
    X = ds.X[:20]
    D = to_distance_matrix(X, "vector")
    n = D.shape[0]
    seed, eps, max_iter, mf, tol = 0, 1e-9, 300, 0.3, 1e-6

    Y_ref, stress_ref = sammon_mapping(D, n_components=2, max_iter=max_iter, alpha=mf, tol=tol, eps=eps, seed=seed, step_halving=False)

    mask = ~np.eye(n, dtype=bool)
    rng = np.random.default_rng(seed)
    mean_scale = D[mask].mean()
    Y0 = rng.normal(loc=0.0, scale=0.1 * mean_scale, size=(n, 2))

    W = alpha_weights(D, 1.0, eps_D=eps)
    Z = compute_Z_from_W(D, W)
    # step_halving=False: this test verifies agreement of the computation
    # core (gradient + diagonal Hessian) with an independent reference
    # implementation, not the new step-halving safeguard (section 3.6, see
    # `test_newton_step_halving_*` for that) - with step_halving=False the
    # code is bit-identical to the original fixed step. Even a single step
    # halving can send the chaotic dynamics of the fixed step to a different
    # local minimum (verified empirically), so this exact numerical test
    # would be unstable with halving enabled.
    Y_mine, history = newton_solve(
        D, W, Z, Y0, max_iter=max_iter, mf=mf, tol=tol, eps_num=eps,
        step_halving=False, max_halvings=10, verbose=False,
    )

    rel_diff = abs(stress_ref[-1] - history["stress"][-1]) / stress_ref[-1]
    assert rel_diff < 1e-3, f"Relative difference of the final stress {rel_diff} >= 1e-3."


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("alpha", [0.0, 1.0, 2.0])
def test_smacof_monotone_decrease(seed: int, alpha: float) -> None:
    """Test 10.4: sigma_{k+1} <= sigma_k + eps_tol for every SMACOF iteration."""
    D = _random_D(seed=seed, n=30)
    eps_D = estimate_eps_D(D, k=5, q=0.1, kind="distance")
    W = alpha_weights(D, alpha, eps_D)
    Z = compute_Z_from_W(D, W)
    Y0 = init_random(D.shape[0], 2, seed=seed + 50, scale=0.5)

    _, history = smacof_solve(
        D, W, Z, Y0, max_iter=100, tol=0.0, eps_num=1e-9,
        dense_pinv_threshold=5000, cg_max_iter=500, cg_tol=1e-10, reg_rho=1e-8, verbose=False,
    )
    sigma = np.array(history["sigma"])
    eps_tol = 1e-10
    diffs = np.diff(sigma)
    assert np.all(diffs <= eps_tol), f"Sigma is not monotone: max increase = {diffs.max()}."


def test_sparse_k_equals_n_minus_1_matches_full_model() -> None:
    """Test 10.5: sparse with k=n-1 (no pivots) reproduces the full dense model."""
    D = _random_D(seed=0, n=60)
    n = D.shape[0]
    alpha = 1.0
    eps_D = estimate_eps_D(D, k=5, q=0.1, kind="distance")
    Y0 = init_random(n, 2, seed=1, scale=0.5)

    W_full = alpha_weights(D, alpha, eps_D)
    Z_full = compute_Z_from_W(D, W_full)
    Y_full, hist_full = smacof_solve(
        D, W_full, Z_full, Y0.copy(), max_iter=200, tol=1e-10, eps_num=1e-9,
        dense_pinv_threshold=5000, cg_max_iter=500, cg_tol=1e-10, reg_rho=1e-8, verbose=False,
    )

    terms = build_sparse_terms(D, alpha, eps_D, n_pivots=0, k_neighbors=n - 1, seed=0, kind="distance")
    Y_sparse, hist_sparse = sparse_smacof_solve(
        terms, Y0.copy(), max_iter=200, tol=1e-10, eps_num=1e-9,
        cg_max_iter=500, cg_tol=1e-10, reg_rho=1e-8, verbose=False,
    )

    rel_stress_diff = abs(hist_full["stress"][-1] - hist_sparse["stress"][-1]) / hist_full["stress"][-1]
    assert rel_stress_diff < 1e-3

    _, _, disparity = procrustes(Y_full, Y_sparse)
    assert disparity < 1e-4


# ---------------------------------------------------------------------------
# Tests for the fix from 2026-09-10 (see
# documentation/2026-09-10_oprava_sparse_cg_a_paralelismus.md): pivot-pivot
# pairs in build_sparse_terms guarantee connectivity of the term graph E, and
# CG with Jacobi preconditioning then converges.
# ---------------------------------------------------------------------------

def _make_blob_dataset(n: int, n_clusters: int, cluster_std: float, seed: int) -> np.ndarray:
    """Synthetic 6 well-separated clusters (make_blobs) for testing the
    connectivity of the term graph (without pivot-pivot pairs the graph was
    disconnected, see the diagnostics in documentation/2026-09-10_...)."""
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(n_samples=n, centers=n_clusters, cluster_std=cluster_std, random_state=seed)
    return X


def test_build_sparse_terms_pivot_pivot_pairs_give_connected_graph() -> None:
    """Test: for well-separated clusters (where the term graph could be
    disconnected without pivot-pivot pairs), the term graph E is always
    connected (1 component) after adding pivot-pivot pairs."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n, n_clusters, cluster_std, seed = 600, 6, 0.3, 0
    n_pivots, k_neighbors = 20, 5
    X = _make_blob_dataset(n, n_clusters, cluster_std, seed)

    eps_D = estimate_eps_D(X, k=5, q=0.1, kind="vector")
    terms = build_sparse_terms(X, alpha=1.0, eps_D=eps_D, n_pivots=n_pivots, k_neighbors=k_neighbors, seed=seed, kind="vector")

    row, col = terms["row"], terms["col"]
    adjacency = coo_matrix((np.ones(len(row)), (row, col)), shape=(n, n))
    n_components, _ = connected_components(adjacency, directed=False)
    assert n_components == 1, f"Term graph E has {n_components} components, expected 1 (pivot-pivot pairs)."


@pytest.mark.parametrize("n_pivots", [0, 1])
def test_build_sparse_terms_small_n_pivots_do_not_break(n_pivots: int) -> None:
    """Test: n_pivots=0 (no pivot pairs) and n_pivots=1 (no pivot-pivot pair,
    nothing to form one with) build without error."""
    D = _random_D(seed=0, n=40)
    eps_D = estimate_eps_D(D, k=5, q=0.1, kind="distance")
    terms = build_sparse_terms(D, alpha=1.0, eps_D=eps_D, n_pivots=n_pivots, k_neighbors=5, seed=0, kind="distance")
    assert len(terms["pivots"]) == n_pivots
    assert len(terms["row"]) > 0


def test_sparse_smacof_solve_converges_on_blob_clusters() -> None:
    """Test: sparse_smacof_solve on data with 6 separated clusters (formerly
    a disconnected term graph -> CG RuntimeError, see the diagnostics)
    completes without an exception, and cg_iterations_last_step are all <
    cg_max_iter (actual convergence, not exhaustion of the iteration budget)."""
    n, n_clusters, cluster_std, seed = 600, 6, 0.3, 0
    n_pivots, k_neighbors = 20, 5
    cg_max_iter = 500
    X = _make_blob_dataset(n, n_clusters, cluster_std, seed)

    eps_D = estimate_eps_D(X, k=5, q=0.1, kind="vector")
    terms = build_sparse_terms(X, alpha=1.0, eps_D=eps_D, n_pivots=n_pivots, k_neighbors=k_neighbors, seed=seed, kind="vector")
    assert terms["n"] == n

    Y0 = init_random(n, 2, seed=1, scale=0.5)
    Y, history = sparse_smacof_solve(
        terms, Y0, max_iter=50, tol=1e-6, eps_num=1e-9,
        cg_max_iter=cg_max_iter, cg_tol=1e-6, reg_rho=1e-8, verbose=False,
    )

    assert history["n_components_term_graph"] == 1
    assert len(history["cg_iterations_last_step"]) > 0
    assert all(it < cg_max_iter for it in history["cg_iterations_last_step"]), (
        f"CG reached cg_max_iter={cg_max_iter} (did not actually converge, it just did not raise an exception): "
        f"{history['cg_iterations_last_step']}"
    )
    assert np.all(np.isfinite(Y))


def _digits_subsample_D(n: int, seed: int = 42) -> np.ndarray:
    from src.datasets.subsample import subsample_dataset

    ds = load_dataset("digits")
    ds = subsample_dataset(ds, n_max=n, random_state=seed)
    return to_distance_matrix(ds.X, "vector")


def test_newton_step_halving_prevents_divergence() -> None:
    """Section 3.6 (documentation/2026-09-11_kontrola_vysledku_plnych_behu.md):
    the original fixed-step pseudo-Newton (Sammon 1969, MF=mf) has no check
    on the stress increase and diverges with a higher MF (repro: digits
    n=200, mf=1.0 - at MF=0.3 it diverges on larger datasets like
    cnae9/isolet, see the doc). step_halving=True must (a) keep individual
    stress increments small (max_halvings is not a guarantee of strict
    monotonicity - with a degenerate direction of the abs-Hessian a small
    residual worsening can remain even after exhausting all halvings, see
    the comment on `newton_solve`) and (b) end with a reasonable final
    stress, unlike step_halving=False, where the stress blows up by several
    orders of magnitude during the run."""
    D = _digits_subsample_D(n=200)
    eps_D = estimate_eps_D(D, k=5, q=0.1, kind="distance")
    W = alpha_weights(D, 1.0, eps_D)
    Z = compute_Z_from_W(D, W)
    Y0 = init_random(D.shape[0], 2, seed=0, scale=0.5)
    mf = 1.0  # increased MF for a fast (small n) reproduction of the divergence, see doc section 3.6

    _, hist_no_halving = newton_solve(
        D, W, Z, Y0.copy(), max_iter=200, mf=mf, tol=1e-6, eps_num=1e-9,
        step_halving=False, max_halvings=10, verbose=False,
    )
    stress_no_halving = np.array(hist_no_halving["stress"])
    assert stress_no_halving.max() > 100.0, (
        f"The expected divergence without step-halving (mf={mf}) did not occur, max stress = {stress_no_halving.max()}."
    )

    Y_halving, hist_halving = newton_solve(
        D, W, Z, Y0.copy(), max_iter=200, mf=mf, tol=1e-6, eps_num=1e-9,
        step_halving=True, max_halvings=10, verbose=False,
    )
    stress_halving = np.array(hist_halving["stress"])
    diffs = np.diff(stress_halving)
    assert diffs.max() < 1e-2, f"The stress increment with step-halving should stay small, max increment = {diffs.max()}."
    assert stress_halving[-1] < 1.0, f"The final stress with step-halving should stay within reasonable bounds (<1.0), it is {stress_halving[-1]}."
    assert np.all(np.isfinite(Y_halving))
    assert "n_halvings" in hist_halving and sum(hist_halving["n_halvings"]) > 0, (
        "At least one step halving was expected (mf=1.0 diverges on this dataset without halving)."
    )


def test_newton_step_halving_monotone_and_bounded_on_mild_data() -> None:
    """Section 3.6: on a "mildly" unstable dataset (iris n=60, default
    MF=0.3), where the original fixed step does NOT diverge catastrophically
    but has pronounced transient stress overshoots (verified: max sigma
    during the run is ~8000x higher than the final one - the "chaotic
    dynamics" mentioned in `test_newton_alpha1_matches_reference_sammon_classic`),
    step_halving=True must (a) keep individual stress increments small (see
    the comment in `test_newton_step_halving_prevents_divergence` - a
    degenerate direction of the abs-Hessian can leave a small residual
    worsening even after exhausting max_halvings), (b) end at a finite,
    reasonably low stress (safely below the actual divergence tested in
    `test_newton_step_halving_prevents_divergence`, where the stress blows
    up by several orders of magnitude to >100). Exact agreement with the
    result without halving (let alone the same local optimum) is NOT
    required - even a single, isolated step halving can send the chaotic
    dynamics of the fixed step to a different (still reasonable) local
    minimum."""
    ds = load_dataset("iris")
    X = ds.X[:60]
    D = to_distance_matrix(X, "vector")
    eps_D = estimate_eps_D(D, k=5, q=0.1, kind="distance")
    W = alpha_weights(D, 1.0, eps_D)
    Z = compute_Z_from_W(D, W)
    Y0 = init_random(D.shape[0], 2, seed=0, scale=0.5)

    Y_new, hist_new = newton_solve(
        D, W, Z, Y0.copy(), max_iter=150, mf=0.3, tol=1e-8, eps_num=1e-9,
        step_halving=True, max_halvings=10, verbose=False,
    )
    stress_new = np.array(hist_new["stress"])
    assert np.diff(stress_new).max() < 1e-2, "The stress increment with step-halving should stay small."
    assert stress_new[-1] < 1.0, f"The final stress with step-halving should stay within reasonable bounds, it is {stress_new[-1]}."
    assert np.all(np.isfinite(Y_new))
