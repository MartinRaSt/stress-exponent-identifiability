# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-09
# License: see the LICENSE file in the repository root
"""Tests for the graph metrics (src/sammon/graph_metrics.py): edge crossings
of a square with diagonals = 1, angular resolution/edge length uniformity on
simple geometric examples, identity-layout neighborhood preservation."""
from __future__ import annotations

import networkx as nx
import numpy as np

from src.sammon.graph_metrics import (
    angular_resolution,
    community_silhouette,
    crossing_angle,
    edge_crossings,
    edge_length_uniformity,
    neighborhood_preservation,
)


def _square_with_diagonals() -> tuple[nx.Graph, np.ndarray]:
    """A square (0,1,2,3 around the perimeter) + both diagonals - exactly 1 crossing."""
    g = nx.Graph()
    g.add_nodes_from([0, 1, 2, 3])
    g.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)])
    Y = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    return g, Y


def test_edge_crossings_square_with_diagonals_equals_one() -> None:
    """A square with both diagonals has exactly 1 edge crossing (the
    perimeter edges do not cross, only the two diagonals in the middle)."""
    g, Y = _square_with_diagonals()
    count, reason = edge_crossings(Y, g, exact_max_n=200)
    assert count == 1.0, f"Expected 1 crossing, got {count} ({reason})."


def test_edge_crossings_skipped_above_exact_max_n() -> None:
    """Above the exact_max_n limit, NaN with a reason is returned, not an approximation."""
    g, Y = _square_with_diagonals()
    count, reason = edge_crossings(Y, g, exact_max_n=2)
    assert np.isnan(count)
    assert "skipped" in reason


def test_crossing_angle_square_diagonals_is_90_degrees() -> None:
    """The diagonals of a unit square cross exactly perpendicularly (90 degrees)."""
    g, Y = _square_with_diagonals()
    angle_dev = crossing_angle(Y, g, exact_max_n=200)
    assert np.isclose(angle_dev, 0.0, atol=1e-8)  # deviation from 90 degrees = 0


def test_angular_resolution_square_perimeter_only() -> None:
    """A square without diagonals: each node has 2 neighbors forming a 90-degree angle."""
    g = nx.cycle_graph(4)
    Y = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    res = angular_resolution(Y, g)
    assert np.isclose(res, 90.0, atol=1e-6)


def test_edge_length_uniformity_regular_polygon_is_zero() -> None:
    """A regular polygon (cycle) has all edges of equal length -> CV=0."""
    n = 6
    g = nx.cycle_graph(n)
    angles = 2 * np.pi * np.arange(n) / n
    Y = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    cv = edge_length_uniformity(Y, g)
    assert np.isclose(cv, 0.0, atol=1e-8)


def test_neighborhood_preservation_identity_layout_is_one() -> None:
    """When the layout is exactly the graph (e.g. a path with unit steps on a
    line), each node has its graph neighbors among its K=degree nearest points."""
    g = nx.path_graph(10)
    Y = np.stack([np.arange(10, dtype=np.float64), np.zeros(10)], axis=1)
    pres = neighborhood_preservation(Y, g)
    assert np.isclose(pres, 1.0, atol=1e-8)


def test_community_silhouette_well_separated_communities() -> None:
    """Two well-separated groups of points must give a high (positive) silhouette."""
    g = nx.Graph()
    g.add_nodes_from(range(6))
    Y = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [10.0, 10.0], [10.1, 10.0], [10.0, 10.1]])
    communities = np.array([0, 0, 0, 1, 1, 1])
    s = community_silhouette(Y, g, communities)
    assert s > 0.9
