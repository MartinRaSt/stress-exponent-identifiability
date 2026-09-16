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
Adapter: TriMap. Supports both point data and a precomputed distance matrix
(use_dist_matrix parameter). The TriMap library has no own random_state;
determinism is ensured by the global numpy seed set before the call
(see src.common.seeding.set_seed, called before every experiment run).
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import trimap

from src.methods.common import method_config, to_distance_matrix
from src.methods.registry import Method, register_method


class TrimapMethod(Method):
    name = "trimap"
    accepts = "both"

    def fit_transform(self, data: np.ndarray | nx.Graph, kind: str, seed: int, n_components: int = 2) -> np.ndarray:
        cfg = method_config("trimap")
        n_inliers, n_outliers, n_random = cfg["n_inliers"], cfg["n_outliers"], cfg["n_random"]

        if kind == "vector":
            X = np.asarray(data, dtype=np.float64)
            model = trimap.TRIMAP(
                n_dims=n_components, n_inliers=n_inliers, n_outliers=n_outliers, n_random=n_random,
            )
            return model.fit_transform(X)

        D = to_distance_matrix(data, kind)
        model = trimap.TRIMAP(
            n_dims=n_components, n_inliers=n_inliers, n_outliers=n_outliers, n_random=n_random,
            use_dist_matrix=True,
        )
        return model.fit_transform(D)


register_method("trimap")(TrimapMethod)
