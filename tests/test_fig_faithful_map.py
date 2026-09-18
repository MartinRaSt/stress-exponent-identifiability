# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-17
# License: see the LICENSE file in the repository root
"""Tests for `src/figures/fig_faithful_map.py`: the [-1, 1] rank-correlation
and unbounded-stress "longer bar = better" normalizations used by the
per-panel fidelity strip, and config completeness (every configured
method/dataset has a human-readable label - see the task's requirement to
never show raw config keys in the figure)."""
from __future__ import annotations

import numpy as np
import pytest

from src.experiments.config_experiments import load_experiments_config
from src.figures import fig_faithful_map as ffm


def test_corr_to_fidelity_endpoints_and_midpoint() -> None:
    assert ffm._corr_to_fidelity(1.0) == pytest.approx(1.0)
    assert ffm._corr_to_fidelity(-1.0) == pytest.approx(0.0)
    assert ffm._corr_to_fidelity(0.0) == pytest.approx(0.5)


def test_corr_to_fidelity_nan_stays_nan() -> None:
    assert np.isnan(ffm._corr_to_fidelity(float("nan")))


def test_corr_to_fidelity_is_clipped_to_unit_interval() -> None:
    # Spearman rho is mathematically in [-1, 1], but the function must not
    # blow up / go outside [0, 1] on a slightly out-of-range input.
    assert ffm._corr_to_fidelity(1.2) == pytest.approx(1.0)
    assert ffm._corr_to_fidelity(-1.2) == pytest.approx(0.0)


def test_stress_to_fidelity_zero_stress_is_perfect() -> None:
    assert ffm._stress_to_fidelity(0.0, stress_max=0.2) == pytest.approx(1.0)


def test_stress_to_fidelity_at_cap_is_zero_and_beyond_cap_stays_zero() -> None:
    assert ffm._stress_to_fidelity(0.2, stress_max=0.2) == pytest.approx(0.0)
    assert ffm._stress_to_fidelity(0.5, stress_max=0.2) == pytest.approx(0.0)


def test_stress_to_fidelity_nan_stays_nan() -> None:
    assert np.isnan(ffm._stress_to_fidelity(float("nan"), stress_max=0.2))


def test_fig_faithful_map_config_labels_cover_all_methods_and_datasets() -> None:
    """Fail loud (well, assert) if a method/dataset used by fig_faithful_map
    has no EXPLICIT entry in the shared display_labels.method/dataset
    section (config_experiments.yaml) - the task's core complaint: raw
    config keys must never reach the figure. A missing entry would still
    fall back to a non-raw label (see test_fig_common.py), but
    fig_faithful_map is the approved hero figure and should not rely on the
    generic fallback for its own columns/rows."""
    cfg = load_experiments_config()["fig_faithful_map"]
    methods = cfg["methods"]
    datasets = cfg["datasets"]
    display_labels = load_experiments_config()["display_labels"]
    assert set(methods).issubset(set(display_labels["method"])), "every fig_faithful_map method needs a display_labels.method entry"
    assert set(datasets).issubset(set(display_labels["dataset"])), "every fig_faithful_map dataset needs a display_labels.dataset entry"


def test_fig_faithful_map_config_methods_are_distinguishable() -> None:
    """Regression guard for the reported defect: sammon_alpha0_smacof and
    sammon_alpha_auto being the only two 'MDS vs. tuned' columns (numerically
    identical on 2/3 datasets) - the classical alpha=1 Sammon column must be
    present so the figure is not spending 2 of its columns on one pair."""
    cfg = load_experiments_config()["fig_faithful_map"]
    assert "sammon_alpha_smacof" in cfg["methods"]


def test_stress_bar_max_is_positive() -> None:
    cfg = load_experiments_config()["fig_faithful_map"]
    assert float(cfg["stress_bar_max"]) > 0.0
