# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-09
# License: see the LICENSE file in the repository root
"""Tests for the synthetic loaders: output shape and determinism for the same seed."""
from __future__ import annotations

import numpy as np
import pytest

from src.datasets.registry import load_dataset

SYNTHETIC_NAMES = [
    "swiss_roll", "s_curve", "sphere", "severed_sphere", "helix",
    "torus", "two_moons", "gaussian_clusters", "hierarchical_clusters",
]


@pytest.mark.parametrize("name", SYNTHETIC_NAMES)
def test_synthetic_dataset_shape(name: str) -> None:
    """A synthetic dataset must return a vector dataset with the expected number of samples."""
    ds = load_dataset(name, n_samples=100)
    assert ds.kind == "vector"
    assert ds.X.shape[0] == 100
    assert ds.X.ndim == 2


@pytest.mark.parametrize("name", SYNTHETIC_NAMES)
def test_synthetic_dataset_determinism(name: str) -> None:
    """Two calls with the same random_state must return bit-identical data."""
    ds1 = load_dataset(name, n_samples=50, random_state=123)
    ds2 = load_dataset(name, n_samples=50, random_state=123)
    np.testing.assert_array_equal(ds1.X, ds2.X)


def test_iris_builtin_dataset() -> None:
    """The built-in iris dataset must have the expected shape 150x4 and 3 classes."""
    ds = load_dataset("iris")
    assert ds.X.shape == (150, 4)
    assert ds.n_classes == 3


def test_wine_is_standardized() -> None:
    """Wine is in config.yaml datasets.standardize - every column must have mean~0, std~1."""
    ds = load_dataset("wine")
    assert ds.meta["standardized"] is True
    np.testing.assert_allclose(ds.X.mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(ds.X.std(axis=0), 1.0, atol=1e-5)


def test_digits_not_standardized() -> None:
    """Digits (image data) is not in datasets.standardize and must remain unchanged."""
    ds = load_dataset("digits")
    assert ds.meta.get("standardized", False) is False
    assert ds.X.max() > 1.5  # the original scale is 0-16, standardization would change it


def test_karate_graph_dataset() -> None:
    """The karate club graph must have 34 nodes and binary labels for the two factions."""
    ds = load_dataset("karate")
    assert ds.kind == "graph"
    assert ds.graph.number_of_nodes() == 34
    assert ds.n_classes == 2


# ---------------------------------------------------------------------------
# K8 (documentation/2026-09-12_plan_smeru_clanku.md): larger graphs for E3 -
# n/m/LCC/ground-truth per reserse/2026-09-12_r1_r6_podklady_smeru_ab.md R5.
# These tests actually DOWNLOAD data (small files, see R5) - consistent with
# the already existing pattern in `tests/test_sammon_temporal.py` (SocioPatterns).
# ---------------------------------------------------------------------------

# (name, expected number of nodes after loading, expected community_source)
_K8_GRAPH_SPECS = [
    ("cora", 2708, "ground_truth"),
    ("power_grid", 4941, "louvain"),
    ("facebook_combined", 4039, "louvain"),
    ("ca_grqc", 5241, "louvain"),
    ("pgp", 10680, "louvain"),
]


@pytest.mark.parametrize("name,expected_n,expected_source", _K8_GRAPH_SPECS)
def test_k8_large_graph_loader_shape_and_community_source(name: str, expected_n: int, expected_source: str) -> None:
    """Each new graph must have the expected number of nodes (see R5) and a
    correctly labeled community source (ground_truth for cora, otherwise louvain)."""
    ds = load_dataset(name)
    assert ds.kind == "graph"
    assert ds.graph.number_of_nodes() == expected_n
    assert ds.meta.get("community_source") == expected_source
    assert ds.y is not None
    assert len(ds.y) == expected_n


def test_k8_cora_ground_truth_classes() -> None:
    """Cora has 7 article-topic classes (Sen et al. 2008)."""
    ds = load_dataset("cora")
    assert ds.n_classes == 7


def test_k8_large_graph_lcc_via_exp3_helper() -> None:
    """The LCC via `exp3_graph_layout._to_largest_component` must be <= n
    and > 0 for all new graphs (a sanity check that the E3 pipeline
    processes the graph)."""
    from src.experiments.exp3_graph_layout import _to_largest_component

    for name, expected_n, _source in _K8_GRAPH_SPECS:
        ds = load_dataset(name)
        _, _, n_original, n_lcc = _to_largest_component(ds.graph, ds.y)
        assert n_original == expected_n
        assert 0 < n_lcc <= n_original


def test_k8_louvain_deterministic_with_same_seed() -> None:
    """Louvain community detection with the same seed (from config.yaml
    datasets.graphs.louvain_seed) must be deterministic between two
    independent loads of the same graph."""
    ds1 = load_dataset("power_grid")
    ds2 = load_dataset("power_grid")
    np.testing.assert_array_equal(ds1.y, ds2.y)
