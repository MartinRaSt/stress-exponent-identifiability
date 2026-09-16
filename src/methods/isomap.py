# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-09
# License: see the LICENSE file in the repository root
"""Adapter: Isomap (geodesic distances + classical MDS)."""
from __future__ import annotations

import networkx as nx
import numpy as np
from sklearn.manifold import Isomap

from src.methods.common import method_config, to_distance_matrix
from src.methods.registry import Method, register_method


class IsomapMethod(Method):
    name = "isomap"
    accepts = "both"

    def fit_transform(self, data: np.ndarray | nx.Graph, kind: str, seed: int, n_components: int = 2) -> np.ndarray:
        cfg = method_config("isomap")
        n_neighbors = cfg["n_neighbors"]
        if kind == "vector":
            model = Isomap(n_neighbors=n_neighbors, n_components=n_components)
            return model.fit_transform(np.asarray(data, dtype=np.float64))
        D = to_distance_matrix(data, kind)
        model = Isomap(n_neighbors=n_neighbors, n_components=n_components, metric="precomputed")
        return model.fit_transform(D)


register_method("isomap")(IsomapMethod)
