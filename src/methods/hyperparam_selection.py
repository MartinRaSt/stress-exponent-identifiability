# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-11
# License: see the LICENSE file in the repository root
"""
A shared mechanism for "fairly tuned" baseline methods (`tsne_auto`,
`umap_auto`) that mirrors `src.sammon.alpha_selection.select_alpha`:
hyperparameter grid-search on a fixed data subsample (same n_val size and
the same index-selection method - `numpy.random.default_rng(seed).choice`,
sorted), evaluation with the independent metric `src.sammon.metrics.evaluate`
(default 'auc_rnx'), and a one-off fit of the best value on the full data.

Without this unification, `sammon_alpha_auto` would be the only method in
E1/E3 tuned via grid-search on the evaluation metric, while t-SNE/UMAP would
run with default hyperparameters - see documentation/
2026-09-11_kontrola_vysledku_plnych_behu.md, section 3.5.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

# WATCH OUT for a circular import (same reason as in src/methods/sammon_alpha.py):
# `src.sammon.metrics` imports `src.methods.common`, which is part of this
# (`src.methods`) package. This module gets imported DURING
# `src.methods.registry._import_all_adapters()` (via `tsne_auto.py`/
# `umap_auto.py`), i.e. before `src.methods` package initialization
# completes - a top-level `from src.sammon.metrics import evaluate` would,
# via the chain import -> src.sammon.graph_metrics -> src.sammon.metrics ->
# src.methods.common -> src.methods.__init__ -> ... -> this module, cause
# `ImportError: cannot import name 'evaluate' from partially initialized
# module` (verified - it broke collection of `tests/test_graph_metrics.py`).
# Importing `evaluate` is therefore lazy (inside `select_hyperparam`).


def subsample_for_selection(data: np.ndarray, kind: str, n_val: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Random subsample of at most `n_val` points, the same mechanism as
    `src.sammon.alpha_selection._subsample_D` (so hyperparameter selection
    for t-SNE/UMAP is comparable to alpha selection for alpha-Sammon).

    kind == "vector"   -> subsample of the rows of X
    kind == "distance" -> subsample of the rows/columns of D (square submatrix)
    """
    arr = np.asarray(data, dtype=np.float64)
    n = arr.shape[0]
    if n <= n_val:
        return arr, np.arange(n)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=n_val, replace=False))
    if kind == "vector":
        return arr[idx], idx
    if kind == "distance":
        return arr[np.ix_(idx, idx)], idx
    raise ValueError(f"subsample_for_selection: unsupported kind='{kind}' (expected 'vector' or 'distance').")


def select_hyperparam(
    data: np.ndarray,
    kind: str,
    grid: list[Any],
    fit_fn: Callable[[np.ndarray, str, Any, int, int], np.ndarray],
    n_val: int,
    selection_seed: int,
    metric: str,
    n_components: int,
) -> tuple[Any, pd.DataFrame]:
    """Generic grid-search: for every value in `grid`, calls `fit_fn(data_sub,
    kind_sel, value, selection_seed, n_components)` on a subsample (see
    `subsample_for_selection`), evaluates `metric` via `evaluate()`, and
    returns (best_value, results_table). Mirrors
    `src.sammon.alpha_selection.select_alpha`.

    `kind == "graph"` is converted to a precomputed distance matrix
    (`to_distance_matrix`, geodesic distance) before subsampling - a subsample
    of the graph (a random subset of nodes as an index array) would not have
    a clear meaning for `fit_fn`, whereas a subsample of its distance matrix
    does (`kind_sel="distance"` for the whole grid-search). The final fit on
    the full data in the adapter (`tsne_auto.py`/`umap_auto.py`) uses the
    original `kind` unchanged.
    """
    from src.sammon.metrics import evaluate

    if kind == "graph":
        from src.methods.common import to_distance_matrix

        data_for_selection = to_distance_matrix(data, "graph")
        kind_sel = "distance"
    else:
        data_for_selection = data
        kind_sel = kind

    data_sub, _ = subsample_for_selection(data_for_selection, kind_sel, n_val, selection_seed)

    rows: list[dict[str, Any]] = []
    for value in grid:
        Y = fit_fn(data_sub, kind_sel, value, selection_seed, n_components)
        metrics = evaluate(data_sub, Y, None, kind_sel)
        rows.append({"value": value, metric: metrics.get(metric, np.nan)})

    table = pd.DataFrame(rows)
    if metric not in table.columns:
        raise KeyError(f"Metric '{metric}' is not in the grid-search results table (available: {list(table.columns)}).")
    best_idx = table[metric].astype(float).idxmax()
    best_value = table.loc[best_idx, "value"]
    return best_value, table
