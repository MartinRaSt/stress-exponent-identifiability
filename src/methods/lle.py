# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-09
# License: see the LICENSE file in the repository root
"""Adapter: Locally Linear Embedding (LLE) - requires point data (vector)."""
from __future__ import annotations

import networkx as nx
import numpy as np
from sklearn.manifold import LocallyLinearEmbedding

from src.methods.common import method_config
from src.methods.registry import Method, register_method


class LleMethod(Method):
    name = "lle"
    accepts = "vector"

    def fit_transform(self, data: np.ndarray | nx.Graph, kind: str, seed: int, n_components: int = 2) -> np.ndarray:
        if kind != "vector":
            raise ValueError(f"Method '{self.name}' only accepts kind='vector', got '{kind}'.")
        cfg = method_config("lle")
        model = LocallyLinearEmbedding(
            n_neighbors=cfg["n_neighbors"],
            n_components=n_components,
            method=cfg["method"],
            random_state=seed,
        )
        return model.fit_transform(np.asarray(data, dtype=np.float64))


register_method("lle")(LleMethod)
