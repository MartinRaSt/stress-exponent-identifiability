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
Central registry of dimensionality-reduction / layout methods. Every method
implements the shared interface `Method.fit_transform(X_or_D, kind, seed, n_components)`.
Adding a new method = a new adapter in `src/methods/` + registering it here
(the `register_method` decorator).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

import networkx as nx
import numpy as np

VALID_ACCEPTS = {"vector", "distance", "graph", "both"}


class Method(ABC):
    """Shared interface for all dimensionality-reduction / layout methods.

    The `accepts` attribute specifies what kind of input the method can handle:
        "vector"   - point data X only (n x d)
        "distance" - distance matrix D only (n x n)
        "both"     - both X and D (internally converts X to D if needed)
        "graph"    - networkx graph only (graph layout algorithms)
    """

    name: str = "unnamed_method"
    accepts: str = "vector"

    @abstractmethod
    def fit_transform(
        self,
        data: np.ndarray | nx.Graph,
        kind: str,
        seed: int,
        n_components: int = 2,
    ) -> np.ndarray:
        """Compute and return the embedding (n x n_components)."""
        raise NotImplementedError


_REGISTRY: dict[str, Callable[[], Method]] = {}


def register_method(name: str) -> Callable[[Callable[[], Method]], Callable[[], Method]]:
    """Decorator: registers a method factory function/class under the given name."""

    def decorator(factory: Callable[[], Method]) -> Callable[[], Method]:
        if name in _REGISTRY:
            raise ValueError(f"Method '{name}' is already defined in the registry more than once.")
        _REGISTRY[name] = factory
        return factory

    return decorator


def list_registered_methods() -> list[str]:
    """Return a sorted list of all registered method names."""
    return sorted(_REGISTRY.keys())


def get_method(name: str) -> Method:
    """Return a new method instance looked up by name in the registry."""
    if name not in _REGISTRY:
        available = ", ".join(list_registered_methods())
        raise KeyError(f"Method '{name}' is not in the registry. Available methods: {available}")
    return _REGISTRY[name]()


def _import_all_adapters() -> None:
    """Import the method adapter modules so registration takes place."""
    from src.methods import pca  # noqa: F401
    from src.methods import mds  # noqa: F401
    from src.methods import sammon_classic  # noqa: F401
    from src.methods import isomap  # noqa: F401
    from src.methods import lle  # noqa: F401
    from src.methods import tsne  # noqa: F401
    from src.methods import tsne_auto  # noqa: F401
    from src.methods import umap_  # noqa: F401
    from src.methods import umap_auto  # noqa: F401
    from src.methods import pacmap_  # noqa: F401
    from src.methods import trimap_  # noqa: F401
    from src.methods import kamada_kawai  # noqa: F401
    from src.methods import spring  # noqa: F401
    from src.methods import sammon_alpha  # noqa: F401
    # K5 (documentation/2026-09-12_plan_smeru_clanku.md): new baseline methods
    from src.methods import densmap  # noqa: F401
    from src.methods import phate_  # noqa: F401
    # K8: spectral layout (networkx.spectral_layout) for E3 native_graph_methods
    from src.methods import spectral  # noqa: F401
    # K7: predicted alpha (no tuning, 1 SMACOF run - see alpha_pred_rule.json)
    from src.methods import sammon_alpha_pred  # noqa: F401
