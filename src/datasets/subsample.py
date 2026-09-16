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
Stratified (or plain) subsampling of a dataset down to n_max elements, so
large datasets (e.g. MNIST) are usable with O(n^2) methods (Sammon, MDS, ...).
Always seeded from config.yaml / an explicit parameter.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from src.common.config import load_config
from src.datasets.registry import Dataset


def deduplicate_rows(X: np.ndarray, y: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None, int]:
    """Remove exact duplicate rows of X (Q1 step 2, A.4 point 3 -
    reserse/2026-09-14_specifikace_rozsireni_q1.md): `np.unique(X, axis=0,
    return_index=True)` returns the index of the FIRST occurrence of each
    unique value in the ORIGINAL row order (not sorted) - sorting these
    indices preserves the original row order and only removes later duplicates.

    Call BEFORE standardization and before subsampling (see
    `subsample_dataset`) - used only in loaders of NEW datasets registered
    after 2026-09-14 (`src/datasets/uci_extra.py`); existing datasets are
    NOT changed (a change would invalidate the already-computed E1 - see
    specification section 0.3).

    Returns (X_dedup, y_dedup|None, n_duplicates_removed).
    """
    n = X.shape[0]
    if n == 0:
        return X, y, 0
    _, first_idx = np.unique(X, axis=0, return_index=True)
    keep_idx = np.sort(first_idx)
    n_removed = n - keep_idx.shape[0]
    X_dedup = X[keep_idx]
    y_dedup = y[keep_idx] if y is not None else None
    return X_dedup, y_dedup, int(n_removed)


def subsample_dataset(ds: Dataset, n_max: int | None = None, random_state: int | None = None) -> Dataset:
    """Return a subsample of the dataset with at most `n_max` elements.

    If the dataset has labels (y) and the number of classes is >= 2 and
    smaller than n_max, stratified selection is used (StratifiedShuffleSplit),
    otherwise plain random selection. `kind="graph"` needs special handling
    (a subgraph), which is not implemented here - raises an exception.
    """
    cfg = load_config()
    sub_cfg = cfg["datasets"]["subsample"]
    n_max = n_max if n_max is not None else sub_cfg["n_max"]
    random_state = random_state if random_state is not None else sub_cfg["random_state"]

    n = ds.n_samples
    if n <= n_max:
        return ds

    if ds.kind == "graph":
        raise NotImplementedError(
            f"Subsampling a graph dataset '{ds.name}' (kind='graph') is not supported - "
            "it would require subgraph selection (e.g. snowball sampling), which is not implemented here."
        )

    rng = np.random.default_rng(random_state)
    can_stratify = (
        sub_cfg.get("stratify_if_labels", True)
        and ds.y is not None
        and 2 <= len(np.unique(ds.y)) < n_max
    )

    if can_stratify:
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=n_max, random_state=random_state)
        idx, _ = next(splitter.split(np.zeros(n), ds.y))
    else:
        idx = rng.choice(n, size=n_max, replace=False)
    idx = np.sort(idx)

    X = ds.X[idx] if ds.X is not None else None
    D = ds.D[np.ix_(idx, idx)] if ds.D is not None else None
    y = ds.y[idx] if ds.y is not None else None

    meta = dict(ds.meta)
    meta["subsampled_from_n"] = n
    meta["subsample_random_state"] = random_state

    return Dataset(name=f"{ds.name}_sub{n_max}", kind=ds.kind, X=X, D=D, graph=None, y=y, meta=meta)
