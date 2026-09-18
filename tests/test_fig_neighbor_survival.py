# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-17
# License: see the LICENSE file in the repository root
"""Tests for `src/figures/fig_neighbor_survival.py`: the fixed policy
category order/highlighting and the "aligned scale, different center"
x-axis logic that lets the two very-different-baseline regime panels share
a comparable axis width (see `_shared_span_xlims`'s docstring)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.experiments.exp13_neighbor_survival import POLICY_LABEL_BEST, POLICY_LABEL_PRED
from src.figures import fig_neighbor_survival as fns


def test_category_order_is_fixed_alphas_then_pred_then_best() -> None:
    order = fns._category_order([1.0, 0.0, 2.0])
    assert order == ["alpha=0 (MDS)", "alpha=1 (Sammon)", "alpha=2 (Kamada-Kawai)", POLICY_LABEL_PRED, POLICY_LABEL_BEST]


def test_category_order_matches_smoke_alpha_table() -> None:
    """--smoke uses a narrower alpha_table (see config_experiments.yaml
    exp13_neighbor_survival.smoke) - the category list must shrink with it,
    never silently keep a category that has no data."""
    order = fns._category_order([0.0, 1.0])
    assert order == ["alpha=0 (MDS)", "alpha=1 (Sammon)", POLICY_LABEL_PRED, POLICY_LABEL_BEST]


def test_style_highlights_pred_and_best_distinctly_from_fixed_and_each_other() -> None:
    style_fixed = fns._style_for_category("alpha=0 (MDS)")
    style_pred = fns._style_for_category(POLICY_LABEL_PRED)
    style_best = fns._style_for_category(POLICY_LABEL_BEST)
    assert style_fixed != style_pred != style_best
    assert len({style_fixed["color"], style_pred["color"], style_best["color"]}) == 3
    assert len({style_fixed["marker"], style_pred["marker"], style_best["marker"]}) == 3


def test_shared_span_xlims_same_width_different_center() -> None:
    """The low_ratio panel (small absolute values, larger relative spread)
    and the mid/high_ratio panel (large absolute values, tiny spread) must
    get the SAME axis width so their spreads are visually comparable, each
    centered on its own data (see the module docstring's rationale)."""
    df = pd.DataFrame({
        "panel": ["low", "low", "low", "high", "high", "high"],
        "stress_scale_invariant": [0.010, 0.015, 0.020, 0.058, 0.059, 0.060],
    })
    xlims = fns._shared_span_xlims(df, "stress_scale_invariant", ["low", "high"])
    width_low = xlims["low"][1] - xlims["low"][0]
    width_high = xlims["high"][1] - xlims["high"][0]
    assert width_low == pytest.approx(width_high)
    # centered on each panel's own (min+max)/2
    assert sum(xlims["low"]) / 2 == pytest.approx(0.015)
    assert sum(xlims["high"]) / 2 == pytest.approx(0.059)


def test_shared_span_xlims_handles_single_value_panel_without_fabricating() -> None:
    """A panel with a single (or all-identical) value must not raise / must
    not invent a spread from nothing - see the fallback branch."""
    df = pd.DataFrame({"panel": ["a", "a", "b"], "metric": [0.5, 0.5, 0.5]})
    xlims = fns._shared_span_xlims(df, "metric", ["a", "b"])
    assert xlims["a"][0] < xlims["a"][1]
    assert xlims["b"][0] < xlims["b"][1]


def test_shared_span_xlims_ignores_nan() -> None:
    df = pd.DataFrame({"panel": ["a", "a", "a"], "metric": [0.01, np.nan, 0.03]})
    xlims = fns._shared_span_xlims(df, "metric", ["a"])
    assert sum(xlims["a"]) / 2 == pytest.approx(0.02)


def test_metric_value_fmt_covers_both_plotted_metrics() -> None:
    assert fns._METRIC_NEIGHBORS in fns._METRIC_VALUE_FMT
    assert fns._METRIC_STRESS in fns._METRIC_VALUE_FMT
