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
K5 (documentation/2026-09-12_plan_smeru_clanku.md) - adapter: PHATE (Moon,
van Dijk, Wang et al. 2019, Nature Biotechnology 37:1482-1492, DOI
10.1038/s41587-019-0336-3), a diffusion-based non-neighbor/global-preserving
baseline (potential distance on the diffusion operator), installed into `venv`
via `pip install phate` (see `requirements.txt`, regenerated
2026-09-12 - `pip freeze` after installation).

For distance/graph input: `phate.PHATE` cannot directly consume a
precomputed distance matrix (the API expects point data or a 'precomputed'
type that expects affinity/similarity, not distance) - so distance/graph
input is converted to coordinates via classical MDS (`sklearn`), so PHATE
has point data to build its own kNN-graph/diffusion representation
(analogous to how `init='pca'` in `SammonAlpha` falls back to
`init_classical_mds` for non-vector input, see
`src/sammon/estimator.py::_init_Y0`) - PHATE computes its own neighbor
graph from these coordinates anyway, so the classical MDS embedding only
serves as a "raw" input space, not as the final projection.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from src.methods.common import method_config, to_distance_matrix
from src.methods.registry import Method, register_method


def _distance_to_points(D: np.ndarray, seed: int) -> np.ndarray:
    """Classical (metric) MDS embedding of the distance matrix into point
    space (see `src.sammon.init.init_classical_mds` for the same principle),
    used only as input for PHATE (which computes its own kNN/diffusion graph
    from the point coordinates - see the module docstring)."""
    from src.sammon.init import init_classical_mds

    n = D.shape[0]
    n_components = min(n - 1, max(2, n // 2)) if n > 3 else 2
    return init_classical_mds(D, n_components, seed)


class PhateMethod(Method):
    name = "phate"
    accepts = "both"

    def fit_transform(self, data: np.ndarray | nx.Graph, kind: str, seed: int, n_components: int = 2) -> np.ndarray:
        import phate

        cfg = method_config("phate")
        if kind == "vector":
            X = np.asarray(data, dtype=np.float64)
        else:
            D = to_distance_matrix(data, kind)
            X = _distance_to_points(D, seed)

        model = phate.PHATE(
            n_components=n_components, knn=cfg["knn"], decay=cfg["decay"],
            random_state=seed, n_jobs=cfg["n_jobs"], verbose=cfg["verbose"],
        )
        return np.asarray(model.fit_transform(X), dtype=np.float64)


register_method("phate")(PhateMethod)
