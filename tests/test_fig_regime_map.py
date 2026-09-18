# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-17
# License: see the LICENSE file in the repository root
"""Tests for `src.figures.fig_regime_map._select_label_datasets` (author
feedback 2026-09-17, second review): dataset-name labels used to overlap
each other/the data points in fig_regime_map.pdf. The fix is a greedy,
minimum-log10-spacing filter over the configured candidate list - these
tests pin down its two guarantees: (1) no two accepted labels are closer
than the configured gap, (2) the global x-extremes are tried first."""
from __future__ import annotations

import pandas as pd

from src.figures.fig_regime_map import _select_label_datasets


def _make_merged(nn_ratio_by_dataset: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame({"nn_ratio_k1": nn_ratio_by_dataset}).rename_axis("dataset")


def test_selected_labels_respect_minimum_log_gap() -> None:
    # a, b, c are within 0.05 log10 of each other (would all overlap); d is far away
    merged = _make_merged({"a": 0.10, "b": 0.11, "c": 0.115, "d": 0.50})
    selected = _select_label_datasets(merged, priority_order=["a", "b", "c", "d"], min_log10_gap=0.15)
    assert selected[0] == "a"  # first candidate always wins over later close ones
    assert "d" in selected
    assert not ({"b", "c"} & set(selected))


def test_global_extremes_are_tried_first() -> None:
    merged = _make_merged({"low": 0.01, "mid": 0.10, "high": 0.90})
    # priority_order does not even mention the extremes - they must still appear
    selected = _select_label_datasets(merged, priority_order=["mid"], min_log10_gap=0.05)
    assert "low" in selected
    assert "high" in selected


def test_candidates_not_in_merged_are_skipped_not_fabricated() -> None:
    merged = _make_merged({"a": 0.10, "b": 0.50})
    selected = _select_label_datasets(merged, priority_order=["a", "nonexistent", "b"], min_log10_gap=0.05)
    assert selected == ["a", "b"]


def test_empty_merged_returns_empty_list() -> None:
    assert _select_label_datasets(pd.DataFrame({"nn_ratio_k1": []}), priority_order=["a"], min_log10_gap=0.1) == []
