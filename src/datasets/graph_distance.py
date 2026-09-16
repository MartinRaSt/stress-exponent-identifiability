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
Conversion of a graph to a distance matrix for methods that work with (n x n)
distances (Sammon, MDS, ...): shortest paths (shortest path / geodesic) and
resistance distance (effective resistance via the Laplacian pseudoinverse).

K8 (documentation/2026-09-12_plan_smeru_clanku.md): for large graphs (n above
a threshold), `resistance_distance` computes the Laplacian pseudoinverse via
a symmetric eigendecomposition (`scipy.linalg.eigh`, or `torch.linalg.eigh`
on GPU in float64) instead of a direct `np.linalg.pinv` - more numerically
stable and faster for large n (see
reserse/2026-09-12_r1_r6_podklady_smeru_ab.md R5, note on the computation
for pgp n=10680). Results are additionally cached to disk
(`cached_distance_matrix`), keyed by a hash of the graph content
(nodes+edges), so they are not recomputed on every experiment resume/rerun.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import networkx as nx
import numpy as np
import scipy.linalg
from scipy.sparse.csgraph import shortest_path

from src.common.config import ensure_dir, get_path


def shortest_path_distance(g: nx.Graph) -> np.ndarray:
    """Return the (n x n) matrix of shortest-path lengths (BFS/Dijkstra) between all nodes.

    Nodes are ordered per `list(g.nodes())`, so the order matches the order
    used elsewhere (e.g. when building y). Requires a connected graph - if
    the graph is not connected, infinite distances are an input-data error
    (fail loud), not something to be silently substituted.
    """
    nodes = list(g.nodes())
    adj = nx.to_scipy_sparse_array(g, nodelist=nodes, weight=None, format="csr")
    D = shortest_path(adj, method="D", directed=False, unweighted=True)
    if not np.isfinite(D).all():
        raise ValueError(
            "Graph is not connected - the shortest-path matrix contains infinity. "
            "Use the largest connected component (e.g. g.subgraph(max(nx.connected_components(g), key=len)))."
        )
    return D.astype(np.float64)


def _laplacian_pinv_eigh(L: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Pseudoinverse (symmetric, PSD) of the Laplacian matrix via eigendecomposition.

    O(n^3) just like `np.linalg.pinv`, but more numerically stable (exploits
    the known symmetric/PSD structure of the Laplacian instead of a general
    SVD) and faster on large n thanks to the specialized symmetric solver
    (`eigh` instead of `svd`). `device='cuda'` uses `torch.linalg.eigh` on
    GPU in float64 (fail-loud if CUDA is unavailable) - accuracy against the
    CPU variant is verified in `tests/test_graph_distance.py` (difference < 1e-8 on karate).
    """
    n = L.shape[0]
    if not np.allclose(L, L.T, atol=1e-8):
        raise ValueError("Laplacian is not symmetric (within tolerance 1e-8) - the input graph is likely corrupted.")

    if device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("_laplacian_pinv_eigh: device='cuda' requested, but CUDA is not available in this environment.")
        L_t = torch.tensor(L, dtype=torch.float64, device="cuda")
        w_t, V_t = torch.linalg.eigh(L_t)
        w = w_t.detach().cpu().numpy()
        tol = float(w.max()) * n * np.finfo(np.float64).eps
        if w.min() < -tol:
            raise ValueError(f"Laplacian has a significantly negative eigenvalue ({w.min():.3e}) - it is not positive semidefinite.")
        mask_t = (w_t > tol)
        inv_w = torch.where(mask_t, 1.0 / w_t, torch.zeros_like(w_t))
        L_pinv_t = (V_t * inv_w) @ V_t.T
        return L_pinv_t.detach().cpu().numpy()

    w, V = scipy.linalg.eigh(L)
    tol = float(w.max()) * n * np.finfo(np.float64).eps
    if w.min() < -tol:
        raise ValueError(f"Laplacian has a significantly negative eigenvalue ({w.min():.3e}) - it is not positive semidefinite.")
    mask = w > tol
    L_pinv = (V[:, mask] * (1.0 / w[mask])) @ V[:, mask].T
    return L_pinv


def resistance_distance(g: nx.Graph, large_n_threshold: int = 2000, device: str = "cpu") -> np.ndarray:
    """Return the (n x n) resistance-distance matrix via the Laplacian pseudoinverse.

    R(i,j) = L+[i,i] + L+[j,j] - 2*L+[i,j], where L+ is the Moore-Penrose
    pseudoinverse of the graph Laplacian (Klein & Randic, 1993).

    For n <= `large_n_threshold`, a direct `np.linalg.pinv` is used (the
    original implementation, already verified by tests on small graphs).
    For n > `large_n_threshold` (K8), `_laplacian_pinv_eigh` is used
    (symmetric eigendecomposition, optionally on GPU via `device='cuda'`) -
    see the K8 spec and reserse/2026-09-12_r1_r6_podklady_smeru_ab.md R5.
    """
    nodes = list(g.nodes())
    n = len(nodes)
    L = nx.laplacian_matrix(g, nodelist=nodes).toarray().astype(np.float64)
    if n <= large_n_threshold:
        L_pinv = np.linalg.pinv(L)
    else:
        L_pinv = _laplacian_pinv_eigh(L, device=device)
    diag = np.diag(L_pinv)
    R = diag[:, None] + diag[None, :] - 2.0 * L_pinv
    np.fill_diagonal(R, 0.0)
    R[R < 0] = 0.0  # numerical noise near zero
    return R


def _graph_content_hash(g: nx.Graph) -> str:
    """Deterministic hash of the graph content (node set + edge set),
    independent of the insertion order into networkx - used as the cache
    key for distance matrices (`cached_distance_matrix`)."""
    nodes = sorted(str(node) for node in g.nodes())
    edges = sorted(tuple(sorted((str(u), str(v)))) for u, v in g.edges())
    payload = json.dumps({"n_nodes": len(nodes), "nodes": nodes, "edges": edges}, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _cache_paths(cache_key: str, distance_metric: str) -> tuple[Path, Path]:
    cache_dir = ensure_dir(get_path("graphs_cache_dir") / "cache")
    npy_path = cache_dir / f"{cache_key}__{distance_metric}.npy"
    hash_path = cache_dir / f"{cache_key}__{distance_metric}.hash.txt"
    return npy_path, hash_path


def cached_distance_matrix(cache_key: str, distance_metric: str, g: nx.Graph, **distance_kwargs) -> np.ndarray:
    """Return the distance matrix of graph `g` (`distance_metric` in
    {'shortest_path', 'resistance'}), with a disk cache at
    `src/data/graphs/cache/<cache_key>__<distance_metric>.npy` (K8).

    The cache is keyed by a hash of the graph content (`_graph_content_hash`),
    stored alongside in a `<...>.hash.txt` file. If the hash does not match
    (the graph changed since the last run - e.g. a different version of
    downloaded data), the cache is silently RECOMPUTED (not data
    fabrication, just a cache miss). If the hash MATCHES but the stored
    matrix has an unexpected shape (a corrupted/truncated file), it is a
    fail-loud error - incorrect data is never returned silently.

    `**distance_kwargs` (e.g. `large_n_threshold`, `device`) are passed
    directly to `resistance_distance` (ignored for `shortest_path`, which
    does not need them).
    """
    if distance_metric not in ("shortest_path", "resistance"):
        raise ValueError(f"Unknown distance_metric='{distance_metric}' (expected 'shortest_path' or 'resistance').")

    npy_path, hash_path = _cache_paths(cache_key, distance_metric)
    current_hash = _graph_content_hash(g)
    n = g.number_of_nodes()

    if npy_path.exists() and hash_path.exists():
        cached_hash = hash_path.read_text(encoding="utf-8").strip()
        if cached_hash == current_hash:
            D = np.load(npy_path)
            if D.shape != (n, n):
                raise RuntimeError(
                    f"Corrupted distance-matrix cache {npy_path}: expected shape ({n},{n}), "
                    f"found shape {D.shape}. Delete the cache files ({npy_path}, {hash_path}) manually and rerun."
                )
            return D

    if distance_metric == "shortest_path":
        D = shortest_path_distance(g)
    else:
        D = resistance_distance(g, **distance_kwargs)

    np.save(npy_path, D)
    hash_path.write_text(current_hash, encoding="utf-8")
    return D
