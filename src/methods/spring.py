# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-09
# License: see the LICENSE file in the repository root
"""Adapter: Fruchterman-Reingold force-directed layout (networkx spring_layout)."""
from __future__ import annotations

import networkx as nx
import numpy as np

from src.methods.common import method_config
from src.methods.registry import Method, register_method


class SpringMethod(Method):
    name = "spring"
    accepts = "graph"

    def fit_transform(self, data: np.ndarray | nx.Graph, kind: str, seed: int, n_components: int = 2) -> np.ndarray:
        if kind != "graph":
            raise ValueError(f"Method '{self.name}' only accepts kind='graph', got '{kind}'.")
        cfg = method_config("spring")
        nodes = list(data.nodes())
        pos = nx.spring_layout(data, dim=n_components, iterations=cfg["iterations"], seed=seed)
        Y = np.array([pos[n] for n in nodes], dtype=np.float64)
        return Y


register_method("spring")(SpringMethod)
