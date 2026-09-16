# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-09
# License: see the LICENSE file in the repository root
"""Adapter: PaCMAP - works only with point data (uses ANN neighbor search)."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pacmap

from src.methods.common import method_config
from src.methods.registry import Method, register_method


class PacmapMethod(Method):
    name = "pacmap"
    accepts = "vector"

    def fit_transform(self, data: np.ndarray | nx.Graph, kind: str, seed: int, n_components: int = 2) -> np.ndarray:
        if kind != "vector":
            raise ValueError(f"Method '{self.name}' only accepts kind='vector', got '{kind}'.")
        cfg = method_config("pacmap")
        X = np.asarray(data, dtype=np.float64)
        n_neighbors = min(cfg["n_neighbors"], X.shape[0] - 1)
        model = pacmap.PaCMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=seed)
        return model.fit_transform(X)


register_method("pacmap")(PacmapMethod)
