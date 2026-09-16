# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-09
# License: see the LICENSE file in the repository root
"""Adapter: UMAP (umap-learn). Uses metric='precomputed' for distance/graph input."""
from __future__ import annotations

import networkx as nx
import numpy as np
import umap

from src.methods.common import method_config, to_distance_matrix
from src.methods.registry import Method, register_method


def fit_umap(
    data: np.ndarray, kind: str, seed: int, n_components: int, n_neighbors: int,
    min_dist: float, metric: str,
) -> np.ndarray:
    """Shared UMAP fit implementation, shared by `UmapMethod` and `UmapAutoMethod`
    (src/methods/umap_auto.py), so the tuned baseline uses the same code as
    the default 'umap' method. `n_neighbors` is clamped to a valid range (< n)."""
    if kind == "vector":
        X = np.asarray(data, dtype=np.float64)
        n_neighbors_eff = min(n_neighbors, X.shape[0] - 1)
        model = umap.UMAP(
            n_neighbors=n_neighbors_eff,
            min_dist=min_dist,
            metric=metric,
            n_components=n_components,
            random_state=seed,
        )
        return model.fit_transform(X)

    D = to_distance_matrix(data, kind)
    n_neighbors_eff = min(n_neighbors, D.shape[0] - 1)
    model = umap.UMAP(
        n_neighbors=n_neighbors_eff,
        min_dist=min_dist,
        metric="precomputed",
        n_components=n_components,
        random_state=seed,
    )
    return model.fit_transform(D)


class UmapMethod(Method):
    name = "umap"
    accepts = "both"

    def fit_transform(self, data: np.ndarray | nx.Graph, kind: str, seed: int, n_components: int = 2) -> np.ndarray:
        cfg = method_config("umap")
        return fit_umap(
            data, kind, seed, n_components, n_neighbors=cfg["n_neighbors"],
            min_dist=cfg["min_dist"], metric=cfg["metric"],
        )


register_method("umap")(UmapMethod)
