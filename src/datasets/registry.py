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
Central dataset registry: name -> loader function mapping and the shared
`Dataset` data container. A new dataset is added by writing a loader
(see synthetic.py / sklearn_sets.py / graphs.py / temporal.py) and
registering it here under a unique name.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import networkx as nx
import numpy as np

from src.common.config import load_config

VALID_KINDS = {"vector", "distance", "graph", "temporal"}


@dataclass
class Dataset:
    """Shared container for all dataset types used in the project.

    Exactly one of X / D / graph corresponds to `kind`:
        kind == "vector"   -> X (n x d) is filled in, y optional
        kind == "distance" -> D (n x n) is filled in, y optional
        kind == "graph"    -> graph (networkx) is filled in, y optional
        kind == "temporal" -> meta contains time slices, see temporal.py
    """

    name: str
    kind: str
    X: np.ndarray | None = None
    D: np.ndarray | None = None
    graph: nx.Graph | None = None
    y: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in VALID_KINDS:
            raise ValueError(f"Unknown kind='{self.kind}' for dataset '{self.name}'. Expected: {VALID_KINDS}")
        if self.kind == "vector" and self.X is None:
            raise ValueError(f"Dataset '{self.name}' has kind='vector', but X is not filled in.")
        if self.kind == "distance" and self.D is None:
            raise ValueError(f"Dataset '{self.name}' has kind='distance', but D is not filled in.")
        if self.kind == "graph" and self.graph is None:
            raise ValueError(f"Dataset '{self.name}' has kind='graph', but graph is not filled in.")

    @property
    def n_samples(self) -> int:
        if self.X is not None:
            return int(self.X.shape[0])
        if self.D is not None:
            return int(self.D.shape[0])
        if self.graph is not None:
            return int(self.graph.number_of_nodes())
        return 0

    @property
    def n_features(self) -> int | None:
        if self.X is not None:
            return int(self.X.shape[1])
        return None

    @property
    def n_classes(self) -> int | None:
        """Number of classes, if y contains discrete labels (int/bool/object).

        For continuous y (e.g. manifold parametrization such as angle/position
        in synthetic datasets like swiss_roll), returns None - these are not
        classification classes, and `len(unique(y))` would misleadingly claim
        "n_samples classes".
        """
        if self.y is None:
            return None
        if np.asarray(self.y).dtype.kind not in ("i", "u", "b", "O", "U", "S"):
            return None
        return int(len(np.unique(self.y)))


_REGISTRY: dict[str, Callable[..., Dataset]] = {}


def register_dataset(name: str) -> Callable[[Callable[..., Dataset]], Callable[..., Dataset]]:
    """Decorator: registers a loader function under the given name in the registry."""

    def decorator(func: Callable[..., Dataset]) -> Callable[..., Dataset]:
        if name in _REGISTRY:
            raise ValueError(f"Dataset '{name}' is already defined in the registry more than once.")
        _REGISTRY[name] = func
        return func

    return decorator


def list_registered_datasets() -> list[str]:
    """Return a sorted list of all registered dataset names."""
    return sorted(_REGISTRY.keys())


def _standardize_inplace(ds: Dataset) -> None:
    """Z-score standardization (StandardScaler) of the ds.X columns, in-place.

    Columns with zero variance are left at 0 (instead of dividing by zero/NaN -
    StandardScaler handles this case by implicitly setting scale_ to 1 when
    scale_==0, so the centered column stays 0).
    Fail-loud: NaN/Inf in the input or in the standardized output are not
    tolerated - they would indicate a loader bug, not a reason to silently discard.
    """
    from sklearn.preprocessing import StandardScaler

    if ds.X is None:
        raise ValueError(f"Dataset '{ds.name}' has standardize=True, but X is not filled in (kind='{ds.kind}').")
    if not np.all(np.isfinite(ds.X)):
        raise ValueError(f"Dataset '{ds.name}': X contains NaN/Inf before standardization - a loader bug, not fabricating a substitute.")
    X_std = StandardScaler().fit_transform(ds.X.astype(np.float64)).astype(np.float32)
    if not np.all(np.isfinite(X_std)):
        raise ValueError(f"Dataset '{ds.name}': standardization produced NaN/Inf - check the input data.")
    ds.X = X_std
    ds.meta["standardized"] = True


def load_dataset(name: str, **kwargs: Any) -> Dataset:
    """Load a dataset by name from the registry.

    Extra parameters (**kwargs) are passed to the loader and override values
    from config.yaml (e.g. load_dataset("swiss_roll", n_samples=500) for the
    exp0 smoke test).

    After loading, z-score standardization is applied centrally (a single
    place for all experiments incl. E5 and figures) for datasets listed in
    config.yaml: datasets.standardize (see documentation/
    2026-09-11_standardizace_a_ladene_baseline.md). Every dataset has a bool
    in ds.meta['standardized'], so it can be written into the dataset table
    (src/datasets/list_datasets.py).
    """
    if name not in _REGISTRY:
        available = ", ".join(list_registered_datasets())
        raise KeyError(f"Dataset '{name}' is not in the registry. Available datasets: {available}")
    ds = _REGISTRY[name](**kwargs)
    standardize_names = set(load_config()["datasets"].get("standardize", []))
    if name in standardize_names:
        if ds.kind != "vector":
            raise ValueError(
                f"Dataset '{name}' is listed in config.yaml datasets.standardize, but has kind='{ds.kind}' "
                "(standardization is only defined for kind='vector')."
            )
        _standardize_inplace(ds)
    else:
        ds.meta.setdefault("standardized", False)
    return ds


def _import_all_loaders() -> None:
    """Import the loader modules so registration happens via the decorator.

    Call before using `load_dataset`/`list_registered_datasets` if the
    modules have not yet been imported (e.g. when only `registry.py` itself is used).
    """
    from src.datasets import synthetic  # noqa: F401
    from src.datasets import sklearn_sets  # noqa: F401
    from src.datasets import graphs  # noqa: F401
    from src.datasets import temporal  # noqa: F401
    from src.datasets import uci_extra  # noqa: F401 (Q1 step 2, hold-out low_ratio candidates)
