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
Sparse stress model (Ortmann, Klimenta, Brandes 2016), see
reserse/2026-09-09_specifikace_metody.md, section 4: max-min pivots
(Gonzalez farthest-point sampling), kNN terms, aggregated pivot weights, a
COO structure, and their integration into SMACOF (CG over a sparse matrix)
and SGD (sampling only from the set E).
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, cg
from sklearn.neighbors import NearestNeighbors

from src.common.progress import progress_iter
from src.sammon.solvers.sgd import _sgd_epoch_numba
from src.sammon.stress import scale_invariant_stress


def select_pivots(D_or_X: np.ndarray, n_pivots: int, seed: int, kind: str = "distance") -> np.ndarray:
    """Select p pivots via max-min (farthest-point, Gonzalez 2-approximation
    of k-center, section 4.1). Start = a random point (seed), iteratively
    add the point maximizing the minimum distance to the already-selected
    pivots."""
    if kind == "distance":
        D = np.asarray(D_or_X, dtype=np.float64)
        n = D.shape[0]
    elif kind == "vector":
        X = np.asarray(D_or_X, dtype=np.float64)
        n = X.shape[0]
    else:
        raise ValueError(f"Unknown kind='{kind}' for select_pivots.")

    if n_pivots < 1 or n_pivots > n:
        raise ValueError(f"n_pivots={n_pivots} must be in range [1, n={n}].")

    rng = np.random.default_rng(seed)
    pivots = [int(rng.integers(0, n))]
    if kind == "distance":
        min_dist = D[pivots[0]].copy()
    else:
        from sklearn.metrics import pairwise_distances as sk_pdist
        min_dist = sk_pdist(X, X[pivots[0]:pivots[0] + 1]).ravel()

    for _ in range(n_pivots - 1):
        next_pivot = int(np.argmax(min_dist))
        pivots.append(next_pivot)
        if kind == "distance":
            new_dist = D[next_pivot]
        else:
            from sklearn.metrics import pairwise_distances as sk_pdist
            new_dist = sk_pdist(X, X[next_pivot:next_pivot + 1]).ravel()
        min_dist = np.minimum(min_dist, new_dist)

    return np.array(pivots, dtype=np.int64)


def compute_knn(
    D_or_X: np.ndarray, k: int, kind: str, exact_knn_threshold: int, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, distances) of shape (n, k) - the k nearest neighbors
    of each point (excluding itself). Vector data: exact
    `sklearn.neighbors.NearestNeighbors` for n < exact_knn_threshold,
    otherwise approximate `pynndescent`. Distance matrix: direct selection
    of the k smallest values in each row."""
    if kind == "distance":
        D = np.asarray(D_or_X, dtype=np.float64)
        n = D.shape[0]
        if k < 1 or k > n - 1:
            raise ValueError(f"k={k} must be in [1, n-1={n - 1}].")
        D_masked = D.copy()
        np.fill_diagonal(D_masked, np.inf)
        idx_part = np.argpartition(D_masked, kth=k - 1, axis=1)[:, :k]
        row_idx = np.arange(n)[:, None]
        dist_part = D_masked[row_idx, idx_part]
        order = np.argsort(dist_part, axis=1)
        idx_sorted = np.take_along_axis(idx_part, order, axis=1)
        dist_sorted = np.take_along_axis(dist_part, order, axis=1)
        return idx_sorted, dist_sorted

    if kind == "vector":
        X = np.asarray(D_or_X, dtype=np.float64)
        n = X.shape[0]
        if k < 1 or k > n - 1:
            raise ValueError(f"k={k} must be in [1, n-1={n - 1}].")
        if n <= exact_knn_threshold:
            nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(X)
            dist, idx = nn.kneighbors(X)
            return idx[:, 1:], dist[:, 1:]  # column 0 is the point itself (distance 0)
        from pynndescent import NNDescent

        index = NNDescent(X, n_neighbors=k + 1, random_state=seed)
        idx, dist = index.neighbor_graph
        return idx[:, 1:], dist[:, 1:]

    raise ValueError(f"Unknown kind='{kind}' for compute_knn.")


def _assign_to_pivots(D_or_X: np.ndarray, pivots: np.ndarray, kind: str) -> np.ndarray:
    """Assign each point to its nearest pivot (a Voronoi cell), return an
    array of pivot indices (into the `pivots` array, not a global index) of
    length n."""
    if kind == "distance":
        D = np.asarray(D_or_X, dtype=np.float64)
        dist_to_pivots = D[:, pivots]
    else:
        from sklearn.metrics import pairwise_distances as sk_pdist
        X = np.asarray(D_or_X, dtype=np.float64)
        dist_to_pivots = sk_pdist(X, X[pivots])
    return np.argmin(dist_to_pivots, axis=1)


def build_sparse_terms(
    D_or_X: np.ndarray,
    alpha: float,
    eps_D: float,
    n_pivots: int,
    k_neighbors: int,
    seed: int,
    kind: str = "distance",
    exact_knn_threshold: int = 20000,
) -> dict:
    """Build the set of terms E = kNN pairs union pivot pairs union
    pivot-pivot pairs (section 4.1, added 2026-09-10 to guarantee
    connectivity of the term graph - see
    documentation/2026-09-10_oprava_sparse_cg_a_paralelismus.md) in COO
    representation (unsymmetrized, i<j, each pair once).

    Without pivot-pivot pairs, the term graph E (viewed as an undirected
    graph on n vertices, where an edge exists for each pair in E) can be
    DISCONNECTED: kNN edges typically do not connect different pivots'
    Voronoi cells, and pivot-point edges attach each point only to its own
    pivot, so the pivots themselves remain unconnected to each other (if
    the Voronoi cells have no mutual kNN overlap). The Laplacian
    V=diag(row_w)-W of a disconnected graph then has more than one zero
    eigenvalue, which causes CG solving the Guttman step to fail to
    converge (RuntimeError, see section 4.3 diagnostics). Solution: for
    every pair of pivots (l, m), l<m, an exact pair is added (without
    multiplying by the Voronoi cell size, unlike the pivot-point pairs
    above, since this is a single pair of points) with weight
    w_lm = (D_lm+eps_D)^{-alpha} and dist=D_lm; the pivots then form a
    clique and the term graph is always connected (every point has a path
    to its pivot, and the pivots are interconnected).

    Returns a dict with keys 'row', 'col', 'dist' (D_ij), 'weight' (w_ij,
    already multiplied by the Voronoi cell size for pivot pairs), 'n',
    'pivots'. For n_pivots=0, pivot and pivot-pivot terms are omitted (kNN
    only); for n_pivots=1, pivot-pivot terms are omitted (nothing to form
    them from); for k_neighbors=n-1 (and n_pivots=0), the term set matches
    the full model (test 10.5). If a pivot-pivot pair already exists from
    kNN, the original is kept (first write wins, `if key not in pair_weight`).
    """
    if kind == "distance":
        D_full = np.asarray(D_or_X, dtype=np.float64)
        n = D_full.shape[0]
    elif kind == "vector":
        n = np.asarray(D_or_X, dtype=np.float64).shape[0]
        D_full = None
    else:
        raise ValueError(f"Unknown kind='{kind}' for build_sparse_terms.")

    # a safeguard for small datasets (n smaller than the configured
    # n_pivots/k_neighbors, e.g. small graphs like karate club, n=34): clip
    # to the highest valid value instead of a hard error, logged - the
    # sparse approximation then degenerates to (nearly) the full model,
    # which is the correct behavior for small n.
    if k_neighbors > n - 1:
        import logging

        logging.getLogger("sammon.sparse").info(
            "k_neighbors=%d > n-1=%d, clipping to n-1 (small n).", k_neighbors, n - 1,
        )
        k_neighbors = n - 1
    if n_pivots > n - 1:
        import logging

        logging.getLogger("sammon.sparse").info(
            "n_pivots=%d > n-1=%d, clipping to n-1 (small n).", n_pivots, n - 1,
        )
        n_pivots = n - 1

    pair_weight: dict[tuple[int, int], float] = {}
    pair_dist: dict[tuple[int, int], float] = {}

    # --- kNN pairs (exact weights w_ij = (D_ij+eps_D)^{-alpha}) ---
    knn_idx, knn_dist = compute_knn(D_or_X, k_neighbors, kind, exact_knn_threshold, seed)
    for i in range(n):
        for jj in range(knn_idx.shape[1]):
            j = int(knn_idx[i, jj])
            dij = float(knn_dist[i, jj])
            key = (i, j) if i < j else (j, i)
            w = 1.0 if alpha == 0.0 else (dij + eps_D) ** (-alpha)
            if key not in pair_weight:
                pair_weight[key] = w
                pair_dist[key] = dij

    # --- pivot pairs (aggregated weight w_il = s_l * D_il^{-alpha}, section 4.1) ---
    pivots = np.array([], dtype=np.int64)
    if n_pivots > 0:
        pivots = select_pivots(D_or_X, n_pivots, seed, kind)
        assign = _assign_to_pivots(D_or_X, pivots, kind)  # (n,) index into `pivots`
        cell_sizes = np.bincount(assign, minlength=n_pivots)
        for i in range(n):
            l_local = int(assign[i])
            l_global = int(pivots[l_local])
            if l_global == i:
                continue
            if kind == "distance":
                dil = float(D_full[i, l_global])
            else:
                from sklearn.metrics import pairwise_distances as sk_pdist
                dil = float(sk_pdist(np.asarray(D_or_X)[i:i + 1], np.asarray(D_or_X)[l_global:l_global + 1])[0, 0])
            s_l = float(cell_sizes[l_local])
            key = (i, l_global) if i < l_global else (l_global, i)
            w = s_l * (1.0 if alpha == 0.0 else (dil + eps_D) ** (-alpha))
            if key not in pair_weight:
                pair_weight[key] = w
                pair_dist[key] = dil

        # --- pivot-pivot pairs (a guarantee of term graph connectivity,
        # added 2026-09-10, see docstring above) - exact weights, without
        # multiplying by the Voronoi cell size. Distances between pivots
        # for kind="vector" are computed in bulk (a single call to
        # pairwise_distances on the pivot submatrix), not one at a time in
        # a loop (for speed).
        if len(pivots) > 1:
            if kind == "distance":
                D_piv = D_full[np.ix_(pivots, pivots)]
            else:
                from sklearn.metrics import pairwise_distances as sk_pdist
                X_all = np.asarray(D_or_X, dtype=np.float64)
                D_piv = sk_pdist(X_all[pivots], X_all[pivots])
            n_piv = len(pivots)
            for l in range(n_piv):
                g_l = int(pivots[l])
                for m in range(l + 1, n_piv):
                    g_m = int(pivots[m])
                    dlm = float(D_piv[l, m])
                    key = (g_l, g_m) if g_l < g_m else (g_m, g_l)
                    w = 1.0 if alpha == 0.0 else (dlm + eps_D) ** (-alpha)
                    if key not in pair_weight:
                        pair_weight[key] = w
                        pair_dist[key] = dlm

    rows = np.array([k[0] for k in pair_weight.keys()], dtype=np.int64)
    cols = np.array([k[1] for k in pair_weight.keys()], dtype=np.int64)
    weights = np.array(list(pair_weight.values()), dtype=np.float64)
    dists = np.array([pair_dist[k] for k in pair_weight.keys()], dtype=np.float64)

    return {"row": rows, "col": cols, "weight": weights, "dist": dists, "n": n, "pivots": pivots}


def _symmetric_sparse_from_coo(terms: dict) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Build symmetric sparse matrices W and D (n x n) from the i<j COO terms."""
    n = terms["n"]
    row, col, w, d = terms["row"], terms["col"], terms["weight"], terms["dist"]
    W = sp.coo_matrix((np.concatenate([w, w]), (np.concatenate([row, col]), np.concatenate([col, row]))), shape=(n, n)).tocsr()
    Dm = sp.coo_matrix((np.concatenate([d, d]), (np.concatenate([row, col]), np.concatenate([col, row]))), shape=(n, n)).tocsr()
    return W, Dm


def sparse_smacof_solve(
    terms: dict,
    Y0: np.ndarray,
    max_iter: int,
    tol: float,
    eps_num: float,
    cg_max_iter: int,
    cg_tol: float,
    reg_rho: float,
    alpha_for_Z: float | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, dict]:
    """Sparse SMACOF (Guttman majorization) over the terms E from
    `build_sparse_terms` (section 4.2): V and B(Y) are sparse (CSR), the
    Guttman step is solved by CG with an implicit sparse matvec and a
    Jacobi (diagonal) preconditioner (added 2026-09-10, see
    documentation/2026-09-10_oprava_sparse_cg_a_paralelismus.md) - without
    preconditioning, CG on disconnected/ill-conditioned term graphs often
    fails to converge within `cg_max_iter` steps (RuntimeError below).

    The `alpha_for_Z` parameter is no longer used to compute Z (kept in the
    API for callers' backward compatibility) - Z is computed directly from
    the already-built weights `terms['weight']` (which are already
    regularized with +eps_D in `build_sparse_terms`), so it remains
    consistent with the objective actually being minimized even for data
    with exact duplicates (section 4.3)."""
    import logging

    n = terms["n"]
    W, D = _symmetric_sparse_from_coo(terms)
    Y = np.asarray(Y0, dtype=np.float64).copy()
    p = Y.shape[1]

    # Z = sum_{(i,j) in E} w_ij D_ij^2 (each term from E once, consistent
    # with `terms['weight']`/`terms['dist']` already being i<j deduplicated).
    Z = float((terms["weight"] * terms["dist"] ** 2).sum())
    if Z <= 0:
        raise ValueError("Z <= 0 for the sparse model - invalid data (all terms zero?).")

    # Fail-loud diagnostics (not a silent fallback): if the term graph E is
    # disconnected, the Laplacian V=diag(row_w)-W has more than one zero
    # eigenvalue, and CG may fail/converge slowly even with preconditioning
    # - we log a WARNING but do not stop the run (build_sparse_terms already
    # adds pivot-pivot pairs precisely to prevent this under normal
    # circumstances).
    n_components_term_graph, _ = sp.csgraph.connected_components(W, directed=False)
    if n_components_term_graph > 1:
        logging.getLogger("sammon.sparse").warning(
            "The term graph E has %d connected components (1 expected) - CG may "
            "converge slowly or fail. Check build_sparse_terms "
            "(pivot-pivot pairs) and the n_pivots/k_neighbors parameters.",
            n_components_term_graph,
        )

    row_w = np.asarray(W.sum(axis=1)).ravel()
    # Jacobi preconditioning: the diagonal of the matrix (V + reg_rho/n * 11^T) is
    # row_w_i + reg_rho/n (diag(V)=row_w, diag(11^T)=1 for every i).
    precond_M = sp.diags(1.0 / (row_w + reg_rho / n))

    def matvec(x: np.ndarray) -> np.ndarray:
        out = row_w * x - W.dot(x)
        out = out + (reg_rho / n) * np.ones(n) * x.sum()
        return out

    op = LinearOperator((n, n), matvec=matvec, dtype=np.float64)

    W_coo = W.tocoo()
    Wr, Wc, Wv = W_coo.row, W_coo.col, W_coo.data

    stress_hist: list[float] = []
    time_hist: list[float] = []
    t_start = time.perf_counter()

    prev_sigma = None
    last_cg_iters: list[int] = []
    iterator = progress_iter(range(max_iter), desc="sparse_smacof") if verbose else range(max_iter)
    for _ in iterator:
        diff = Y[Wr] - Y[Wc]
        d = np.sqrt((diff ** 2).sum(axis=1))
        d_safe = np.where(d < eps_num, eps_num, d)
        Dij = np.asarray(D[Wr, Wc]).ravel()
        sigma = float((Wv * (Dij - d_safe) ** 2).sum() / 2.0)
        e_alpha = sigma / Z
        stress_hist.append(e_alpha)
        time_hist.append(time.perf_counter() - t_start)

        if prev_sigma is not None and (prev_sigma - sigma) < tol * max(prev_sigma, eps_num):
            break
        prev_sigma = sigma

        b_vals = -Wv * Dij / d_safe
        B = sp.coo_matrix((b_vals, (Wr, Wc)), shape=(n, n)).tocsr()
        b_diag = -np.asarray(B.sum(axis=1)).ravel()
        B = B + sp.diags(b_diag)
        rhs = B.dot(Y)

        Y_new = np.empty_like(Y)
        cg_iters_this_step = []
        for q in range(p):
            n_cg_iter = [0]

            def _cb(xk, _counter=n_cg_iter):  # counts CG iterations (section 4.3 diagnostics)
                _counter[0] += 1

            sol, info = cg(op, rhs[:, q], x0=Y[:, q], rtol=cg_tol, maxiter=cg_max_iter, M=precond_M, callback=_cb)
            if info != 0:
                raise RuntimeError(f"CG (sparse SMACOF) did not converge for column {q} (info={info}).")
            Y_new[:, q] = sol
            cg_iters_this_step.append(n_cg_iter[0])
        Y = Y_new
        last_cg_iters = cg_iters_this_step

    if last_cg_iters:
        residual_last = float(np.linalg.norm(op(Y[:, 0]) - rhs[:, 0]))
    else:
        residual_last = float("nan")
    history = {
        "stress": stress_hist, "time_sec": time_hist, "n_terms": int(len(terms["row"])),
        "cg_iterations_last_step": last_cg_iters,
        "cg_residual_last_step_col0": residual_last,
        "n_components_term_graph": int(n_components_term_graph),
    }
    return Y, history


def sparse_sgd_solve(
    terms: dict,
    Y0: np.ndarray,
    epochs: int,
    mu_max: float,
    eps_anneal: float,
    eps_num: float,
    seed: int,
    alpha_for_Z: float | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, dict]:
    """Sparse SGD: sampling pairs only from the term set E (instead of all
    n(n-1)/2 pairs), the rest of the scheme matches the stabilized SGD
    (section 4.2).

    `alpha_for_Z` is no longer used to compute Z (see `sparse_smacof_solve`)
    - Z is computed directly from `terms['weight']`/`terms['dist']`."""
    n = terms["n"]
    row, col, w = terms["row"], terms["col"], terms["weight"]
    dist = terms["dist"]
    Y = np.asarray(Y0, dtype=np.float64).copy()

    W_full = sp.coo_matrix((np.concatenate([w, w]), (np.concatenate([row, col]), np.concatenate([col, row]))), shape=(n, n)).tocsr()
    D_full = sp.coo_matrix((np.concatenate([dist, dist]), (np.concatenate([row, col]), np.concatenate([col, row]))), shape=(n, n)).tocsr()

    Z = float((w * dist ** 2).sum())
    if Z <= 0:
        raise ValueError("Z <= 0 for the sparse model - invalid data (all terms zero?).")

    w_med = float(np.median(w))
    w_max = float(w.max())
    eta_max = 1.0 / w_med
    eta_min = eps_anneal / w_max

    rng = np.random.default_rng(seed)
    stress_hist: list[float] = []
    sat_ratio_hist: list[float] = []
    time_hist: list[float] = []
    t_start = time.perf_counter()

    # dense representation of weights/distances for fast numba access -
    # acceptable for n suitable for the sparse model (n < 10^5-10^6, see
    # section 4.3); for very large n, a direct sparse array would be needed
    # (out of scope for this demo run).
    W_dense = W_full.toarray()
    D_dense = D_full.toarray()

    iterator = progress_iter(range(1, epochs + 1), desc="sparse_sgd") if verbose else range(1, epochs + 1)
    for t in iterator:
        eta_t = eta_max * (eta_min / eta_max) ** ((t - 1) / (epochs - 1)) if epochs > 1 else eta_max
        epoch_rng = np.random.default_rng(rng.integers(0, 2**63 - 1))
        order = epoch_rng.permutation(len(row))
        i_idx, j_idx = row[order].astype(np.int64), col[order].astype(np.int64)

        n_sat, mus = _sgd_epoch_numba(Y, D_dense, W_dense, i_idx, j_idx, eta_t, mu_max, eps_num)

        stress_hist.append(scale_invariant_stress(D_dense, Y, W_dense, Z, eps_num))
        sat_ratio_hist.append(n_sat / len(i_idx))
        time_hist.append(time.perf_counter() - t_start)

    history = {"stress": stress_hist, "sat_ratio": sat_ratio_hist, "time_sec": time_hist, "n_terms": int(len(row))}
    return Y, history
