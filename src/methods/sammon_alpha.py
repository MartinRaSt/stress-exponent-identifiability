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
Adapters of the proposed alpha-Sammon method (`src.sammon.estimator.SammonAlpha`)
into the shared method registry (`src.methods.registry`), so it can be used
in the same experiments as other methods (MDS, t-SNE, UMAP, ...).

Each adapter is a thin wrapper: it loads its own `methods.<name>` config
(alpha, solver, device) and delegates to `SammonAlpha.fit_transform`.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from src.methods.common import method_config
from src.methods.registry import Method, register_method

# WATCH OUT for a circular import: `src.sammon.estimator` (and its
# dependencies such as `src.sammon.metrics`) import `src.methods.common`,
# which is part of this (`src.methods`) package - a top-level `from
# src.sammon.estimator import SammonAlpha` would therefore cause a circular
# import (partial module) on the first import of `src.methods`. Importing
# SammonAlpha is therefore done inside `fit_transform` (lazy), once both
# packages are fully loaded.


def _make_sammon_method(name: str) -> type[Method]:
    """Factory function: creates the adapter class for the given name in `methods.<name>`."""

    class _SammonAlphaMethod(Method):
        accepts = "both"

        def __init__(self) -> None:
            self.last_selected_hyperparam: str = ""

        def fit_transform(self, data: np.ndarray | nx.Graph, kind: str, seed: int, n_components: int = 2) -> np.ndarray:
            from src.sammon.estimator import SammonAlpha

            cfg = method_config(name)
            est = SammonAlpha(
                alpha=cfg["alpha"], solver=cfg["solver"], device=cfg["device"],
                n_components=n_components, seed=seed, verbose=False,
            )
            Y = est.fit_transform(data, kind=kind)
            # est.alpha_ is always the resolved numeric alpha (even for
            # alpha='auto'/'multiscale' input) - see
            # src/sammon/estimator.py::SammonAlpha.fit_transform, docstring.
            # Allows writing the actually-used alpha INTO the CSV (E1/E3
            # column 'selected_hyperparam'), analogous to tsne_auto/umap_auto.
            self.last_selected_hyperparam = f"alpha={est.alpha_}"
            return Y

    _SammonAlphaMethod.name = name
    _SammonAlphaMethod.__name__ = f"SammonAlphaMethod_{name}"
    return _SammonAlphaMethod


_METHOD_NAMES = [
    "sammon_alpha_smacof",
    "sammon_alpha0_smacof",
    "sammon_alpha2_smacof",
    "sammon_alpha_auto",
    "sammon_multiscale",
    "sammon_sgd_stab",
    "sammon_sgd_naive",
    "sammon_sparse_smacof",
    "sammon_newton",
]

for _name in _METHOD_NAMES:
    register_method(_name)(_make_sammon_method(_name))
