# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-09
# License: see the LICENSE file in the repository root
"""Adapter: metric MDS (SMACOF) via sklearn.manifold.MDS on a precomputed distance matrix."""
from __future__ import annotations

import networkx as nx
import numpy as np
from sklearn.manifold import MDS

from src.methods.common import method_config, to_distance_matrix
from src.methods.registry import Method, register_method


class MdsMethod(Method):
    name = "mds"
    accepts = "both"

    def fit_transform(self, data: np.ndarray | nx.Graph, kind: str, seed: int, n_components: int = 2) -> np.ndarray:
        cfg = method_config("mds")
        D = to_distance_matrix(data, kind)
        model = MDS(
            n_components=n_components,
            metric_mds=cfg["metric_mds"],
            metric="precomputed",
            init=cfg["init"],
            max_iter=cfg["max_iter"],
            eps=cfg["eps"],
            n_init=cfg["n_init"],
            random_state=seed,
            normalized_stress=False,
        )
        return model.fit_transform(D)


register_method("mds")(MdsMethod)
