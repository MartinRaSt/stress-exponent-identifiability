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
Graph-specific layout quality metrics (rows 18-23 of the metrics table,
reserse/2026-09-09_pseudokod_a_metriky.md PART B): edge crossing count,
crossing angle, angular resolution, edge length uniformity, neighborhood
preservation, and community silhouette.

Node ordering convention: `Y[i]` corresponds to `list(g.nodes())[i]` (the
same convention as `src.datasets.graph_distance.shortest_path_distance` and
`src.methods.common.graph_to_array`), so a given embedding Y can be used
directly with the original graph `g` without further mapping.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from src.sammon.metrics import _neighbor_ranks


def _node_index_map(g: nx.Graph) -> dict:
    """Return the node -> row index mapping in Y (order of `list(g.nodes())`)."""
    return {node: i for i, node in enumerate(g.nodes())}


def _edge_index_array(g: nx.Graph) -> np.ndarray:
    """Return the graph edges as an index array (m, 2) per `_node_index_map`."""
    idx = _node_index_map(g)
    edges = [(idx[u], idx[v]) for u, v in g.edges() if u != v]
    if not edges:
        raise ValueError("Graph has no (non-loop) edges - graph metrics are not defined.")
    return np.asarray(edges, dtype=np.int64)


def _segments_intersect(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> bool:
    """Exact intersection test for two segments (p1,p2) and (p3,p4) via the
    orientation (cross product) of point triples - a standard O(1) segment
    intersection test used in the graph-drawing literature (e.g. Purchase
    1997 aesthetics)."""
    def orient(a, b, c) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    d1 = orient(p3, p4, p1)
    d2 = orient(p3, p4, p2)
    d3 = orient(p1, p2, p3)
    d4 = orient(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def edge_crossings(Y: np.ndarray, g: nx.Graph, exact_max_n: int) -> tuple[float, str]:
    """Edge crossing count (row 18 of the metrics table): an exact O(m^2)
    segment-intersection test over all edge pairs without a shared node.

    Returns (crossing_count, reason). For n > exact_max_n returns
    (NaN, reason) - the exact test is NOT computed for large graphs (O(m^2)
    is prohibitive), no approximation is fabricated. `n` = number of graph
    nodes (Y.shape[0]), see section 7 and config
    `sammon.metrics_extended.graph_crossings_exact_max_n`.
    """
    n = Y.shape[0]
    if n > exact_max_n:
        return float("nan"), f"n={n} > graph_crossings_exact_max_n={exact_max_n}, exact O(m^2) test skipped"

    edges = _edge_index_array(g)
    m = edges.shape[0]
    count = 0
    for a in range(m):
        i1, j1 = edges[a]
        for b in range(a + 1, m):
            i2, j2 = edges[b]
            if len({i1, j1, i2, j2}) < 4:
                continue  # shared node - not counted as a crossing
            if _segments_intersect(Y[i1], Y[j1], Y[i2], Y[j2]):
                count += 1
    return float(count), "exact O(m^2) segment test"


def edge_crossings_with_pairs(Y: np.ndarray, g: nx.Graph, exact_max_n: int) -> tuple[float, str, list]:
    """Like `edge_crossings`, but additionally returns the list of edge
    index pairs that cross (needed by `crossing_angle` to avoid a second
    O(m^2) pass)."""
    n = Y.shape[0]
    if n > exact_max_n:
        return float("nan"), f"n={n} > graph_crossings_exact_max_n={exact_max_n}, exact O(m^2) test skipped", []

    edges = _edge_index_array(g)
    m = edges.shape[0]
    pairs: list[tuple[int, int]] = []
    for a in range(m):
        i1, j1 = edges[a]
        for b in range(a + 1, m):
            i2, j2 = edges[b]
            if len({i1, j1, i2, j2}) < 4:
                continue
            if _segments_intersect(Y[i1], Y[j1], Y[i2], Y[j2]):
                pairs.append((a, b))
    return float(len(pairs)), "exact O(m^2) segment test", pairs


def crossing_angle(Y: np.ndarray, g: nx.Graph, exact_max_n: int) -> float:
    """Mean deviation of the angle between crossing edges from 90 degrees
    (row 19 of the metrics table; closer to 90 degrees = better legibility,
    a standard graph-drawing aesthetic criterion). Requires
    `edge_crossings_with_pairs` (exact test, n<=exact_max_n) - else NaN."""
    edges = _edge_index_array(g)
    _, _, pairs = edge_crossings_with_pairs(Y, g, exact_max_n)
    if not pairs:
        return float("nan")
    deviations = []
    for a, b in pairs:
        i1, j1 = edges[a]
        i2, j2 = edges[b]
        v1 = Y[j1] - Y[i1]
        v2 = Y[j2] - Y[i2]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 <= 0 or n2 <= 0:
            continue
        cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        theta_deg = np.degrees(np.arccos(cos_theta))
        theta_deg = min(theta_deg, 180.0 - theta_deg)  # angle between lines, not oriented vectors
        deviations.append(abs(90.0 - theta_deg))
    if not deviations:
        return float("nan")
    return float(np.mean(deviations))


def angular_resolution(Y: np.ndarray, g: nx.Graph) -> float:
    """Minimum angle between adjacent edges at a node, averaged over nodes
    with degree >= 2 (row 20 of the metrics table; higher = a more legible
    arrangement of edges around the node, a standard graph-drawing
    aesthetic criterion)."""
    idx = _node_index_map(g)
    min_angles = []
    for node in g.nodes():
        i = idx[node]
        neighbors = [idx[nb] for nb in g.neighbors(node) if nb != node]
        if len(neighbors) < 2:
            continue
        vecs = Y[neighbors] - Y[i]
        norms = np.linalg.norm(vecs, axis=1)
        valid = norms > 0
        if valid.sum() < 2:
            continue
        vecs = vecs[valid] / norms[valid, None]
        angles = np.degrees(np.arctan2(vecs[:, 1], vecs[:, 0])) % 360.0
        angles_sorted = np.sort(angles)
        gaps = np.diff(np.concatenate([angles_sorted, angles_sorted[:1] + 360.0]))
        min_angles.append(float(gaps.min()))
    if not min_angles:
        return float("nan")
    return float(np.mean(min_angles))


def edge_length_uniformity(Y: np.ndarray, g: nx.Graph) -> float:
    """Coefficient of variation (CV = std/mean) of edge lengths (row 21 of
    the metrics table; lower = more uniform edge lengths, a standard
    aesthetic criterion)."""
    edges = _edge_index_array(g)
    lengths = np.linalg.norm(Y[edges[:, 0]] - Y[edges[:, 1]], axis=1)
    mean_len = float(lengths.mean())
    if mean_len <= 0:
        raise ValueError("Mean edge length is 0 - edge length uniformity is not defined.")
    return float(lengths.std() / mean_len)


def neighborhood_preservation(Y: np.ndarray, g: nx.Graph) -> float:
    """Fraction of node i's graph neighbors that are among the K=deg(i)
    nearest points in the embedding Y (row 22 of the metrics table; Gansner,
    Koren, North 2004/2005 - "neighborhood preservation" for graph drawing
    by stress majorization). Nodes with degree 0 are excluded from the
    average (undefined)."""
    n = Y.shape[0]
    idx = _node_index_map(g)
    d = np.sqrt(((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1))
    _, order_emb = _neighbor_ranks(d)

    fractions = []
    for node in g.nodes():
        i = idx[node]
        neighbors = {idx[nb] for nb in g.neighbors(node) if nb != node}
        deg = len(neighbors)
        if deg == 0:
            continue
        k = min(deg, n - 1)
        knn_emb = set(order_emb[i, :k].tolist())
        fractions.append(len(neighbors & knn_emb) / deg)
    if not fractions:
        raise ValueError("No node has nonzero degree - neighborhood preservation is not defined.")
    return float(np.mean(fractions))


def community_silhouette(Y: np.ndarray, g: nx.Graph, communities: np.ndarray) -> float:
    """Silhouette score (Rousseeuw 1987, DOI 10.1016/0377-0427(87)90125-7) in
    the Euclidean metric of the embedding Y with labels = communities (row
    23 of the metrics table). `communities` must have the same order as
    `list(g.nodes())`."""
    from sklearn.metrics import silhouette_score

    communities = np.asarray(communities)
    n_classes = len(np.unique(communities))
    n = Y.shape[0]
    if not (2 <= n_classes < n):
        raise ValueError(f"Number of communities ({n_classes}) must be in range [2, n-1={n - 1}].")
    return float(silhouette_score(Y, communities))
