# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-09
# License: see the LICENSE file in the repository root
"""Adapter: Kamada-Kawai graph layout (networkx). Requires a graph (kind='graph')."""
from __future__ import annotations

import networkx as nx
import numpy as np

from src.methods.registry import Method, register_method


class KamadaKawaiMethod(Method):
    name = "kamada_kawai"
    accepts = "graph"

    def fit_transform(self, data: np.ndarray | nx.Graph, kind: str, seed: int, n_components: int = 2) -> np.ndarray:
        if kind != "graph":
            raise ValueError(f"Method '{self.name}' only accepts kind='graph', got '{kind}'.")
        nodes = list(data.nodes())
        pos = nx.kamada_kawai_layout(data, dim=n_components)
        Y = np.array([pos[n] for n in nodes], dtype=np.float64)
        return Y


register_method("kamada_kawai")(KamadaKawaiMethod)
