# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-09
# License: see the LICENSE file in the repository root
"""Helper functions shared across method adapters in `src/methods/`."""
from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.spatial.distance import pdist, squareform

from src.common.config import load_config
from src.datasets.graph_distance import shortest_path_distance


def to_distance_matrix(data: np.ndarray | nx.Graph, kind: str) -> np.ndarray:
    """Convert the input data to an (n x n) Euclidean/geodesic distance matrix.

    kind == "vector"   -> Euclidean distance between the rows of X
    kind == "distance" -> data is already D, returned unchanged
    kind == "graph"    -> geodesic distance (shortest paths) in the graph
    """
    if kind == "distance":
        return np.asarray(data, dtype=np.float64)
    if kind == "vector":
        return squareform(pdist(np.asarray(data, dtype=np.float64), metric="euclidean"))
    if kind == "graph":
        return shortest_path_distance(data)
    raise ValueError(f"Unknown kind='{kind}' for conversion to a distance matrix.")


def graph_to_array(g: nx.Graph) -> tuple[np.ndarray, list]:
    """Return the graph's (dense) adjacency matrix and the node order used to build it."""
    nodes = list(g.nodes())
    A = nx.to_numpy_array(g, nodelist=nodes)
    return A, nodes


def method_config(name: str) -> dict:
    """Load the given method's hyperparameters from the `methods.<name>` section of config.yaml."""
    cfg = load_config()
    try:
        return dict(cfg["methods"][name])
    except KeyError as exc:
        raise KeyError(f"config.yaml is missing methods.{name}") from exc


def is_compatible(accepts: str, kind: str) -> bool:
    """Tell whether a method with the given `accepts` can handle a dataset of the given `kind`."""
    if accepts == "both":
        return kind in ("vector", "distance", "graph")
    return accepts == kind


def common_n_components(default: int | None = None) -> int:
    """Return the default output dimensionality from config.yaml methods.common.n_components."""
    cfg = load_config()
    n = cfg["methods"]["common"]["n_components"]
    return int(default) if default is not None else int(n)
