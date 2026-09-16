# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-12
# License: see the LICENSE file in the repository root
"""K8 (documentation/2026-09-12_plan_smeru_clanku.md): tests for
src/datasets/graph_distance.py - the eigh-based Laplacian pseudoinverse
(large graphs) vs. the original np.linalg.pinv (small graphs), and the
on-disk distance-matrix cache keyed by a hash of the graph's content."""
from __future__ import annotations

import shutil

import networkx as nx
import numpy as np
import pytest
import torch

from src.common.config import ensure_dir, get_path
from src.datasets.graph_distance import (
    cached_distance_matrix,
    resistance_distance,
    shortest_path_distance,
)


def test_resistance_eigh_matches_pinv_on_karate() -> None:
    """K8 acceptance criterion: the eigh-based branch (force
    large_n_threshold=0) differs from the original np.linalg.pinv (force
    large_n_threshold=n) by < 1e-8 on the karate club graph (n=34)."""
    g = nx.karate_club_graph()
    R_pinv = resistance_distance(g, large_n_threshold=10**9)
    R_eigh = resistance_distance(g, large_n_threshold=0)
    assert np.max(np.abs(R_pinv - R_eigh)) < 1e-8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available in this environment")
def test_resistance_eigh_gpu_matches_cpu_on_karate() -> None:
    """K8: the GPU (torch.linalg.eigh, float64) pseudoinverse variant agrees
    with the CPU (scipy.linalg.eigh) one to within 1e-8 on a small graph."""
    g = nx.karate_club_graph()
    R_cpu = resistance_distance(g, large_n_threshold=0, device="cpu")
    R_gpu = resistance_distance(g, large_n_threshold=0, device="cuda")
    assert np.max(np.abs(R_cpu - R_gpu)) < 1e-8


def test_resistance_distance_symmetric_and_nonnegative() -> None:
    """Basic sanity check (both branches): R is symmetric, non-negative, and
    has a zero diagonal."""
    g = nx.les_miserables_graph()
    for large_n_threshold in (10**9, 0):
        R = resistance_distance(g, large_n_threshold=large_n_threshold)
        assert np.allclose(R, R.T)
        assert (R >= 0).all()
        assert np.allclose(np.diag(R), 0.0)


def test_cached_distance_matrix_hits_cache_and_matches_direct(tmp_path) -> None:
    """A second call to `cached_distance_matrix` with the same graph must
    return bit-identical data to a direct computation, and the cache files
    must be created."""
    g = nx.karate_club_graph()
    cache_key = "test_cache_karate_pytest"
    cache_dir = ensure_dir(get_path("graphs_cache_dir") / "cache")
    npy_path = cache_dir / f"{cache_key}__resistance.npy"
    hash_path = cache_dir / f"{cache_key}__resistance.hash.txt"
    if npy_path.exists():
        npy_path.unlink()
    if hash_path.exists():
        hash_path.unlink()
    try:
        D1 = cached_distance_matrix(cache_key, "resistance", g)
        assert npy_path.exists() and hash_path.exists()
        D2 = cached_distance_matrix(cache_key, "resistance", g)
        np.testing.assert_array_equal(D1, D2)
        D_direct = resistance_distance(g)
        np.testing.assert_array_equal(D1, D_direct)
    finally:
        if npy_path.exists():
            npy_path.unlink()
        if hash_path.exists():
            hash_path.unlink()


def test_cached_distance_matrix_invalidates_on_graph_change() -> None:
    """A change to the graph (a different edge) must trigger a recomputation,
    not a silent return of the old cache (a different n can also give a
    different shape, but even with the same n the hash changes and the
    cache is recomputed)."""
    cache_key = "test_cache_invalidate_pytest"
    cache_dir = ensure_dir(get_path("graphs_cache_dir") / "cache")
    npy_path = cache_dir / f"{cache_key}__shortest_path.npy"
    hash_path = cache_dir / f"{cache_key}__shortest_path.hash.txt"
    if npy_path.exists():
        npy_path.unlink()
    if hash_path.exists():
        hash_path.unlink()
    try:
        g1 = nx.path_graph(5)
        D1 = cached_distance_matrix(cache_key, "shortest_path", g1)
        g2 = nx.cycle_graph(5)  # same number of nodes, different edges
        D2 = cached_distance_matrix(cache_key, "shortest_path", g2)
        assert not np.array_equal(D1, D2)
        np.testing.assert_array_equal(D2, shortest_path_distance(g2))
    finally:
        if npy_path.exists():
            npy_path.unlink()
        if hash_path.exists():
            hash_path.unlink()
