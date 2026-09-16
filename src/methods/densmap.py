# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-12
# License: see the LICENSE file in the repository root
"""
K5 (documentation/2026-09-12_plan_smeru_clanku.md) - adapter: DensMAP
(Narayan, Berger, Cho 2021, Nature Biotechnology 39:765-774, DOI
10.1038/s41587-020-00801-7), a UMAP variant explicitly optimized to preserve
LOCAL point density (a Ripley-type per-point radius), implemented directly
in `umap-learn` (`umap.UMAP(densmap=True, ...)`). Shares the
n_neighbors/min_dist/metric hyperparameters with `methods.umap` (config.yaml)
- it is the same base method with an extra objective-function term, not an
independent baseline.

For distance/graph input it uses `metric='precomputed'` (same convention
as `src/methods/umap_.py`).
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from src.methods.common import method_config, to_distance_matrix
from src.methods.registry import Method, register_method


def fit_densmap(
    data: np.ndarray, kind: str, seed: int, n_components: int, n_neighbors: int,
    min_dist: float, metric: str,
) -> np.ndarray:
    """DensMAP fit (umap.UMAP with densmap=True) - same structure as
    `src.methods.umap_.fit_umap`, just with `densmap=True` added."""
    import umap

    if kind == "vector":
        X = np.asarray(data, dtype=np.float64)
        n_neighbors_eff = min(n_neighbors, X.shape[0] - 1)
        model = umap.UMAP(
            n_neighbors=n_neighbors_eff, min_dist=min_dist, metric=metric,
            n_components=n_components, random_state=seed, densmap=True,
        )
        return model.fit_transform(X)

    D = to_distance_matrix(data, kind)
    n_neighbors_eff = min(n_neighbors, D.shape[0] - 1)
    model = umap.UMAP(
        n_neighbors=n_neighbors_eff, min_dist=min_dist, metric="precomputed",
        n_components=n_components, random_state=seed, densmap=True,
    )
    return model.fit_transform(D)


class DensmapMethod(Method):
    name = "densmap"
    accepts = "both"

    def fit_transform(self, data: np.ndarray | nx.Graph, kind: str, seed: int, n_components: int = 2) -> np.ndarray:
        cfg = method_config("densmap")
        return fit_densmap(
            data, kind, seed, n_components, n_neighbors=cfg["n_neighbors"],
            min_dist=cfg["min_dist"], metric=cfg["metric"],
        )


register_method("densmap")(DensmapMethod)
