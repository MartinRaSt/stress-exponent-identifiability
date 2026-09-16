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
K8 (documentation/2026-09-12_plan_smeru_clanku.md) - adapter: spectral
layout (networkx.spectral_layout, coordinates = the first two nonzero
eigenvectors of the graph Laplacian). Requires a graph (kind='graph'), is
deterministic (no random seed - the Laplacian eigenvectors are unique up
to sign/degeneracy, the `seed` parameter is ignored just like in
`kamada_kawai`). Serves as a "cheap" baseline linking the alpha-Sammon
family (alpha=2 ~ Kamada-Kawai, spectral drawing as another point on the
same stress/spectral graph-drawing axis, see 03_metoda.tex section 3.4).
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from src.methods.registry import Method, register_method


class SpectralMethod(Method):
    name = "spectral"
    accepts = "graph"

    def fit_transform(self, data: np.ndarray | nx.Graph, kind: str, seed: int, n_components: int = 2) -> np.ndarray:
        if kind != "graph":
            raise ValueError(f"Method '{self.name}' only accepts kind='graph', got '{kind}'.")
        nodes = list(data.nodes())
        pos = nx.spectral_layout(data, dim=n_components)
        Y = np.array([pos[n] for n in nodes], dtype=np.float64)
        return Y


register_method("spectral")(SpectralMethod)
