# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-16
# License: see the LICENSE file in the repository root
"""K16 (2026-09-16, documentation/2026-09-16_e4_hierarchicky_test.md) - a small
shared module for the E4 temporal comparison of our family (`lambda<L>_alpha<A>_<solver>`)
with the dynamic t-SNE baseline (`dtsne_lambda<L>`), so that the pairing rule
is NOT DUPLICATED between `export_numbers.py::_add_exp4_dtsne_baseline_numbers`
(K12a, aggregation over the whole dataset) and `exp4_neighbor_metrics.py`
(K16 primary per-dataset test): for a given dtsne point (one lambda), the
point in our family with the CLOSEST `stab` value (node displacement between
frames) is found - no selection based on the outcome of the metric we are
currently testing.
"""
from __future__ import annotations

import re

import numpy as np

# method names in exp4_temporal_results.csv (see exp4_temporal.py: 'dtsne_lambda{L}'
# for dynamic t-SNE, 'lambda{L}_alpha{A}_{solver}' for our family).
DTSNE_METHOD_RE = re.compile(r"^dtsne_lambda(?P<lam>[0-9.]+)$")
OUR_TEMPORAL_METHOD_RE = re.compile(r"^lambda(?P<lam>[0-9.]+)_alpha(?P<alpha>[0-9.]+)_(?P<solver>smacof|sgd)$")


def nearest_stab_index(target_stab: float, candidate_stab: np.ndarray) -> int:
    """Returns the index of the candidate with the closest 'stab' value
    (smallest |candidate-target|) - the ONLY place with this logic in the
    project (K16).

    On a tie (same absolute distance for >=2 candidates), returns the index
    of the FIRST one in input array order (`np.argmin` is deterministic for
    the first occurrence of the minimum) - no random selection.
    """
    arr = np.asarray(candidate_stab, dtype=np.float64)
    if arr.shape[0] == 0:
        raise ValueError("nearest_stab_index: empty candidate array (candidate_stab) - nothing to pair.")
    return int(np.argmin(np.abs(arr - target_stab)))
