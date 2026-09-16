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
Classical scikit-learn benchmark datasets (bundled + OpenML via
fetch_openml with a local cache). A missing/unavailable OpenML dataset
raises a clear exception - no fabricating substitute data.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.datasets import (
    fetch_olivetti_faces,
    fetch_openml,
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
)
from sklearn.preprocessing import LabelEncoder

from src.common.config import ensure_dir, get_path, load_config
from src.datasets.registry import Dataset, register_dataset

# ---------------------------------------------------------------------------
# Bundled scikit-learn datasets - do not require network access
# ---------------------------------------------------------------------------


@register_dataset("iris")
def load_iris_ds(**_overrides: Any) -> Dataset:
    """Iris (150 x 4, 3 classes), the classic bundled scikit-learn dataset."""
    d = load_iris()
    return Dataset(name="iris", kind="vector", X=d.data.astype(np.float32), y=d.target.astype(np.int64),
                    meta={"source": "sklearn.datasets.load_iris", "citation": "Fisher (1936)"})


@register_dataset("wine")
def load_wine_ds(**_overrides: Any) -> Dataset:
    """Wine (178 x 13, 3 classes), a bundled scikit-learn dataset."""
    d = load_wine()
    return Dataset(name="wine", kind="vector", X=d.data.astype(np.float32), y=d.target.astype(np.int64),
                    meta={"source": "sklearn.datasets.load_wine"})


@register_dataset("digits")
def load_digits_ds(**_overrides: Any) -> Dataset:
    """Digits (1797 x 64, 10 classes), a bundled scikit-learn dataset (8x8 digit images)."""
    d = load_digits()
    return Dataset(name="digits", kind="vector", X=d.data.astype(np.float32), y=d.target.astype(np.int64),
                    meta={"source": "sklearn.datasets.load_digits"})


@register_dataset("breast_cancer")
def load_breast_cancer_ds(**_overrides: Any) -> Dataset:
    """Wisconsin Breast Cancer (569 x 30, 2 classes), a bundled scikit-learn dataset."""
    d = load_breast_cancer()
    return Dataset(name="breast_cancer", kind="vector", X=d.data.astype(np.float32), y=d.target.astype(np.int64),
                    meta={"source": "sklearn.datasets.load_breast_cancer"})


@register_dataset("olivetti")
def load_olivetti_ds(**overrides: Any) -> Dataset:
    """Olivetti faces (400 x 4096, 40 classes), downloaded into the sklearn cache on first use."""
    cfg = load_config()
    random_state = overrides.get("random_state", cfg["datasets"]["openml"]["fetch_random_state"])
    d = fetch_olivetti_faces(random_state=random_state, shuffle=True)
    return Dataset(name="olivetti", kind="vector", X=d.data.astype(np.float32), y=d.target.astype(np.int64),
                    meta={"source": "sklearn.datasets.fetch_olivetti_faces"})


# ---------------------------------------------------------------------------
# OpenML datasets (require network access on first download, then cached)
# ---------------------------------------------------------------------------

_OPENML_KEYS = [
    "mnist_784", "fashion_mnist", "coil20", "optdigits", "pendigits", "satimage",
    "letter", "spambase", "har", "isolet", "cnae9", "usps", "glass", "ionosphere",
    "seeds", "yeast", "vehicle", "segment",
]


def _load_openml(key: str, **_overrides: Any) -> Dataset:
    """Generic loader for a single OpenML dataset defined in config.yaml."""
    cfg = load_config()
    try:
        spec = dict(cfg["datasets"]["openml"]["sets"][key])
    except KeyError as exc:
        raise KeyError(f"config.yaml is missing datasets.openml.sets.{key}") from exc

    cache_dir = ensure_dir(get_path("openml_cache_dir"))
    try:
        bunch = fetch_openml(data_home=str(cache_dir), as_frame=False, parser="auto", **spec)
    except Exception as exc:  # network/HTTP/parsing errors from fetch_openml are heterogeneous
        raise RuntimeError(
            f"Failed to load/download OpenML dataset '{key}' (spec={spec}): {exc}. "
            "No substitute data is generated - check your connection or the spec in config.yaml."
        ) from exc

    X = np.asarray(bunch.data, dtype=np.float32)
    y_raw = np.asarray(bunch.target)
    if y_raw.dtype.kind in ("U", "S", "O"):
        y = LabelEncoder().fit_transform(y_raw).astype(np.int64)
    else:
        y = y_raw.astype(np.int64)

    return Dataset(name=key, kind="vector", X=X, y=y,
                    meta={"source": "sklearn.datasets.fetch_openml", "openml_spec": spec})


def _make_openml_loader(key: str) -> Callable[..., Dataset]:
    """Create a named loader for the given OpenML key (a closure over `key`)."""

    def loader(**overrides: Any) -> Dataset:
        return _load_openml(key, **overrides)

    loader.__name__ = f"load_{key}"
    loader.__doc__ = f"OpenML dataset '{key}', see config.yaml: datasets.openml.sets.{key}."
    return loader


for _key in _OPENML_KEYS:
    register_dataset(_key)(_make_openml_loader(_key))
