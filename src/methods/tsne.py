# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-09
# License: see the LICENSE file in the repository root
"""Adapter: t-SNE (sklearn). Uses metric='precomputed' for distance/graph input."""
from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
from sklearn.manifold import TSNE

from src.methods.common import method_config, to_distance_matrix
from src.methods.registry import Method, register_method


def fit_tsne(
    data: np.ndarray, kind: str, seed: int, n_components: int, perplexity: int,
    learning_rate: Any, max_iter: int, init: str,
) -> np.ndarray:
    """Shared t-SNE fit implementation (sklearn), shared by `TsneMethod` and
    `TsneAutoMethod` (src/methods/tsne_auto.py), so the tuned baseline uses
    the same code as the default 'tsne' method. `perplexity` is clamped to a
    valid range given the sample count (sklearn requires perplexity < n_samples)."""
    n_samples = np.asarray(data).shape[0] if kind == "vector" else to_distance_matrix(data, kind).shape[0]
    perplexity_eff = min(perplexity, max(5, (n_samples - 1) // 3))

    common_kwargs = dict(
        n_components=n_components,
        perplexity=perplexity_eff,
        learning_rate=learning_rate,
        max_iter=max_iter,
        random_state=seed,
    )
    if kind == "vector":
        model = TSNE(init=init, **common_kwargs)
        return model.fit_transform(np.asarray(data, dtype=np.float64))
    D = to_distance_matrix(data, kind)
    # the precomputed metric does not support 'pca' init, so random init is used
    model = TSNE(metric="precomputed", init="random", **common_kwargs)
    return model.fit_transform(D)


class TsneMethod(Method):
    name = "tsne"
    accepts = "both"

    def fit_transform(self, data: np.ndarray | nx.Graph, kind: str, seed: int, n_components: int = 2) -> np.ndarray:
        cfg = method_config("tsne")
        return fit_tsne(
            data, kind, seed, n_components, perplexity=cfg["perplexity"],
            learning_rate=cfg["learning_rate"], max_iter=cfg["max_iter"], init=cfg["init"],
        )


register_method("tsne")(TsneMethod)
