# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-19
# License: see the LICENSE file in the repository root
"""Regression test for validator finding B3 (2026-09-19,
podklady/2026-09-19_validace_EN_supplement.md): `pareto_summary.tex`'s
caption hardcoded "32 datasets" while the table body actually held 51 rows
per method. `n_datasets_total_from_summary` must read the count from the
data (the `n_datasets_total` column of the per-method summary) and fail
loudly if it is not the same for every method (a silent gap in the
dataset x method grid)."""
from __future__ import annotations

import pandas as pd
import pytest

from src.experiments.pareto_analysis import n_datasets_total_from_summary


def test_reads_the_count_from_data_not_a_constant() -> None:
    summary = pd.DataFrame({
        "method": ["a", "b", "c"],
        "n_datasets_total": [51, 51, 51],
    })
    assert n_datasets_total_from_summary(summary) == 51


def test_a_different_total_is_also_picked_up_correctly() -> None:
    """Guards against a hardcoded 51 replacing the old hardcoded 32."""
    summary = pd.DataFrame({
        "method": ["a", "b"],
        "n_datasets_total": [7, 7],
    })
    assert n_datasets_total_from_summary(summary) == 7


def test_inconsistent_counts_fail_loud_instead_of_picking_one() -> None:
    summary = pd.DataFrame({
        "method": ["a", "b", "c"],
        "n_datasets_total": [51, 51, 50],
    })
    with pytest.raises(ValueError, match="not the same for every method"):
        n_datasets_total_from_summary(summary)
