# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-14
# License: see the LICENSE file in the repository root
"""Tests for Q1 step 2 (reserse/2026-09-14_specifikace_rozsireni_q1.md,
A.7/A.9 item 13) - `src/experiments/power_analysis_regime.py`.

NOTE ON THE NUMBER IN THE SPEC ("Noether n=26.2 +- 0.1 for input A.7"): a
manual recomputation of the MAD from the 10 Delta values in A.7 (see
documentation/2026-09-14_q1_krok2_datasety.md) gives MAD=0.0348 (the median
of the sorted absolute deviations 0.00485,0.00485,0.00695,0.01245,0.03335,
0.03625,0.04205,0.06255,0.19345,0.20115 -> average of the 5th and 6th value
= 0.0348), robust sigma=1.4826*0.0348=0.05159, NOT 0.0523 as stated in the
spec text (the MAD=0.0353 used there is ~1.4% higher - a minor inaccuracy
in the spec author's manual computation). With this (verified) sigma,
Noether's n_required(robust, alpha=0.025) = 25.6, not 26.2. The test below
verifies OUR OWN consistent computation (numpy MAD + the Noether formula),
not a number copied from the spec - see CLAUDE.md "nothing without a
source, everything regenerable"."""
from __future__ import annotations

import numpy as np
import pytest

from src.experiments.power_analysis_regime import (
    n_required_from_grid,
    noether_n_required,
    noether_power_at_n,
    sign_exact_power_at_n,
    summary_stats,
    t_approx_n_required,
    t_approx_power_at_n,
)

_A7_DELTAS = np.array([-0.0292, 0.0000, 0.0209, 0.0264, 0.0285, 0.0382, 0.0696, 0.0754, 0.2268, 0.2345])


def test_summary_stats_matches_hand_computation() -> None:
    stats = summary_stats(_A7_DELTAS)
    assert stats["median"] == pytest.approx(0.03335, abs=1e-9)
    assert stats["mean"] == pytest.approx(0.06911, abs=1e-4)
    assert stats["sd"] == pytest.approx(0.0903, abs=1e-3)
    assert stats["mad"] == pytest.approx(0.0348, abs=1e-4)
    assert stats["p_plus"] == pytest.approx(0.8, abs=1e-9)


def test_noether_n_required_robust_sigma_a7() -> None:
    """The Noether formula with OUR OWN (verified) robust sigma - see the
    module docstring for an explanation of the deviation from the spec text (26.2)."""
    stats = summary_stats(_A7_DELTAS)
    n_req = noether_n_required(stats["median"], stats["sigma_robust"], alpha_onesided=0.025, power_target=0.8)
    assert n_req == pytest.approx(25.6, abs=0.2)


def test_noether_n_required_full_sd_a7_matches_spec_66() -> None:
    """The full sd (0.0903, independent of the MAD discrepancy above) -> n~66.1 (spec A.7 (a))."""
    stats = summary_stats(_A7_DELTAS)
    n_req = noether_n_required(stats["median"], stats["sd"], alpha_onesided=0.025, power_target=0.8)
    assert n_req == pytest.approx(66.1, abs=0.5)


def test_noether_power_at_n_is_increasing_in_n() -> None:
    mu, sigma = 0.0333, 0.0523
    powers = [noether_power_at_n(mu, sigma, 0.025, n) for n in (10, 20, 30, 50)]
    assert all(np.diff(powers) > 0)


def test_noether_power_at_n_required_equals_target() -> None:
    """The inverted formula must be consistent with the direct one: power_at_n(n_required) ~ power_target."""
    mu, sigma, alpha, target = 0.0333, 0.0523, 0.025, 0.8
    n_req = noether_n_required(mu, sigma, alpha, target)
    power_at_exact_n = noether_power_at_n(mu, sigma, alpha, n_req)
    assert power_at_exact_n == pytest.approx(target, abs=1e-6)


def test_t_approx_n_required_matches_spec_order_of_magnitude() -> None:
    """A.7 (b): d=0.637 (mu/sigma_robust=0.0333/0.0523=0.637) -> n=19.4
    (alpha 0.025). We use OUR OWN mu/sigma (see the module docstring) - the
    value should be comparable in order of magnitude (the difference is
    only from the MAD deviation above)."""
    stats = summary_stats(_A7_DELTAS)
    n_req = t_approx_n_required(stats["median"], stats["sigma_robust"], alpha_onesided=0.025, power_target=0.8)
    assert 15.0 < n_req < 25.0


def test_t_approx_power_at_n_consistent_with_n_required() -> None:
    mu, sigma, alpha, target = 0.0333, 0.0523, 0.025, 0.8
    n_req = t_approx_n_required(mu, sigma, alpha, target)
    assert t_approx_power_at_n(mu, sigma, alpha, n_req) == pytest.approx(target, abs=1e-6)


def test_sign_exact_power_matches_spec_n20_alpha025() -> None:
    """A.7 (c): n=20, p_+=0.8, alpha=0.025 (one-sided) -> critical point
    k>=15 (P_H0=0.0207), power 0.804 (a spec number, exact binomial
    distribution - here there is no dependency on the MAD discrepancy, only on p_+=8/10)."""
    power, k_crit = sign_exact_power_at_n(p_plus=0.8, alpha_onesided=0.025, n=20)
    assert k_crit == 15
    assert power == pytest.approx(0.804, abs=0.01)


def test_sign_exact_power_matches_spec_n15_alpha025() -> None:
    """A.7 (c): n=15, alpha=0.025 -> k>=12 (P_H0=0.0176), power 0.648."""
    power, k_crit = sign_exact_power_at_n(p_plus=0.8, alpha_onesided=0.025, n=15)
    assert k_crit == 12
    assert power == pytest.approx(0.648, abs=0.01)


def test_sign_exact_power_matches_spec_n25_alpha025() -> None:
    """A.7 (c): n=25, alpha=0.025 -> k>=18 (P_H0=0.0216), power 0.891."""
    power, k_crit = sign_exact_power_at_n(p_plus=0.8, alpha_onesided=0.025, n=25)
    assert k_crit == 18
    assert power == pytest.approx(0.891, abs=0.01)


def test_n_required_from_grid_picks_smallest_satisfying() -> None:
    power_by_n = {10: 0.5, 15: 0.7, 20: 0.85, 25: 0.95}
    assert n_required_from_grid([10, 15, 20, 25], power_by_n, 0.8) == 20.0


def test_n_required_from_grid_nan_when_never_reached() -> None:
    power_by_n = {10: 0.1, 15: 0.2}
    assert np.isnan(n_required_from_grid([10, 15], power_by_n, 0.8))
