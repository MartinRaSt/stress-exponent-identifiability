# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-14
# License: see the LICENSE file in the repository root
"""Tests for Q1 step 2 (reserse/2026-09-14_specifikace_rozsireni_q1.md, A.6/A.9
item 13) - the pure statistical functions in `src/experiments/stats_holdout.py`:
Holm on a known example, sign-flip p on n=5 manually verifiable, Cliff's
delta on a small example, deterministic bootstrap, TOST on trivial data."""
from __future__ import annotations

import numpy as np
import pytest

from src.experiments.stats_holdout import (
    bootstrap_ci_paired,
    cliff_delta,
    delta_pair,
    exact_sign_flip_pvalue,
    hodges_lehmann,
    holm_correction,
    mc_sign_flip_pvalue,
    sign_flip_test,
    tost_equivalence,
    wilcoxon_pvalue_safe,
)


def test_holm_correction_matches_textbook_example() -> None:
    """A classic example: p=[0.01,0.02,0.03,0.04,0.05], m=5 ->
    adjusted=[0.05,0.08,0.09,0.09,0.09] (manually verifiable, see the
    `holm_correction` docstring)."""
    p = [0.01, 0.02, 0.03, 0.04, 0.05]
    adj = holm_correction(p)
    assert adj == pytest.approx([0.05, 0.08, 0.09, 0.09, 0.09], abs=1e-9)


def test_holm_correction_preserves_input_order() -> None:
    """The input order != the sorted order - the output must follow the ORIGINAL order."""
    p = [0.03, 0.01, 0.05, 0.02, 0.04]
    adj = holm_correction(p)
    # the smallest p (0.01, index 1) gets the smallest multiplier (5) -> 0.05
    assert adj[1] == pytest.approx(0.05, abs=1e-9)
    # the largest p (0.05, index 2) gets running_max >= 0.09
    assert adj[2] == pytest.approx(0.09, abs=1e-9)


def test_holm_correction_never_decreases_with_rank() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0.001, 0.5, size=10)
    adj = np.array(holm_correction(p.tolist()))
    order = np.argsort(p)
    assert np.all(np.diff(adj[order]) >= -1e-12), "the adjusted p-values must be non-decreasing in order of increasing original p"
    assert np.all(adj <= 1.0)


def test_holm_correction_ignores_nan() -> None:
    p = [0.01, np.nan, 0.02]
    adj = holm_correction(p)
    assert np.isnan(adj[1])
    assert adj[0] == pytest.approx(min(1.0, 2 * 0.01), abs=1e-9)


def test_exact_sign_flip_pvalue_n5_hand_verifiable() -> None:
    """n=5, diffs=[1,1,1,1,-1] (all |diff|=1) - T_obs=sum=3, which
    corresponds to exactly k=4 positive signs out of 5 (2k-5=3 => k=4).
    P(T>=3) under H0 (uniform over 2^5=32 sign patterns) = P(k>=4) =
    (C(5,4)+C(5,5))/32 = (5+1)/32 = 0.1875 (computable by hand from the
    binomial distribution)."""
    diffs = np.array([1.0, 1.0, 1.0, 1.0, -1.0])
    p, n_perm = exact_sign_flip_pvalue(diffs, "greater")
    assert n_perm == 32
    assert p == pytest.approx(6.0 / 32.0, abs=1e-12)


def test_exact_sign_flip_pvalue_two_sided_symmetric() -> None:
    """Symmetry: two-sided p = 2x the one-sided p for T_obs>0 (if < 1)."""
    diffs = np.array([1.0, 1.0, 1.0, 1.0, -1.0])
    p_greater, _ = exact_sign_flip_pvalue(diffs, "greater")
    p_two, _ = exact_sign_flip_pvalue(diffs, "two-sided")
    assert p_two == pytest.approx(2 * p_greater, abs=1e-12)


def test_exact_sign_flip_pvalue_all_positive_is_most_extreme() -> None:
    """All diffs positive with the same magnitude -> the most extreme
    possible pattern (the only one out of 2^n giving the maximal sum) -> p = 1/2^n."""
    diffs = np.full(6, 2.0)
    p, n_perm = exact_sign_flip_pvalue(diffs, "greater")
    assert n_perm == 64
    assert p == pytest.approx(1.0 / 64.0, abs=1e-12)


def test_exact_sign_flip_matches_brute_force_small_n() -> None:
    """The meet-in-the-middle implementation must agree with a naive
    enumeration of 2^n patterns for small n (an independent sanity check)."""
    rng = np.random.default_rng(42)
    diffs = rng.normal(size=7)
    diffs = diffs[diffs != 0]
    n = diffs.shape[0]
    abs_d = np.abs(diffs)
    T_obs = float(np.sum(diffs))
    count_ge = 0
    for mask in range(1 << n):
        signs = np.array([1.0 if (mask >> k) & 1 else -1.0 for k in range(n)])
        if float(np.sum(signs * abs_d)) >= T_obs - 1e-9:
            count_ge += 1
    p_expected = count_ge / (1 << n)
    p_actual, _ = exact_sign_flip_pvalue(diffs, "greater")
    assert p_actual == pytest.approx(p_expected, abs=1e-12)


def test_sign_flip_test_dispatches_exact_vs_mc() -> None:
    diffs = np.array([0.1, 0.2, -0.05, 0.3, 0.15])
    p_exact, n_perm_exact, is_exact = sign_flip_test(diffs, "greater", seed=1, n_perm_mc=1000, exact_max_n=22)
    assert is_exact and n_perm_exact == 32
    p_mc, n_perm_mc, is_exact2 = sign_flip_test(diffs, "greater", seed=1, n_perm_mc=1000, exact_max_n=2)
    assert not is_exact2 and n_perm_mc == 1000
    # MC should be reasonably close to the exact value (small n, but only orientational)
    assert abs(p_mc - p_exact) < 0.3


def test_mc_sign_flip_pvalue_deterministic_for_fixed_seed() -> None:
    diffs = np.array([0.1, -0.2, 0.3, 0.05, -0.1, 0.4])
    p1, _ = mc_sign_flip_pvalue(diffs, "greater", seed=123, n_perm=500)
    p2, _ = mc_sign_flip_pvalue(diffs, "greater", seed=123, n_perm=500)
    assert p1 == p2


def test_hodges_lehmann_matches_manual_walsh_median() -> None:
    diffs = np.array([1.0, 2.0, 3.0])
    # Walsh averages (i<=j): 1,1.5,2,2,2.5,3 -> median = 1.75... let's compute it exactly
    walsh_manual = sorted([(1 + 1) / 2, (1 + 2) / 2, (1 + 3) / 2, (2 + 2) / 2, (2 + 3) / 2, (3 + 3) / 2])
    expected = float(np.median(walsh_manual))
    assert hodges_lehmann(diffs) == pytest.approx(expected, abs=1e-12)


def test_cliff_delta_known_small_example() -> None:
    """a always > b -> delta=+1; a always < b -> delta=-1; identical sets -> delta=0."""
    a = np.array([5.0, 6.0, 7.0])
    b = np.array([1.0, 2.0, 3.0])
    assert cliff_delta(a, b) == pytest.approx(1.0)
    assert cliff_delta(b, a) == pytest.approx(-1.0)
    assert cliff_delta(a, a) == pytest.approx(0.0)


def test_cliff_delta_partial_dominance_hand_computed() -> None:
    """a=[1,3], b=[2,2]: pairs (1,2)-> a<b; (1,2)-> a<b; (3,2)-> a>b; (3,2)-> a>b
    => gt=2, lt=2, delta=(2-2)/4=0."""
    a = np.array([1.0, 3.0])
    b = np.array([2.0, 2.0])
    assert cliff_delta(a, b) == pytest.approx(0.0)


def test_delta_pair_hand_computed() -> None:
    diffs = np.array([1.0, -1.0, 2.0, 0.0, 3.0])  # 3 positive, 1 negative, 1 zero, n=5
    assert delta_pair(diffs) == pytest.approx((3 - 1) / 5)


def test_bootstrap_ci_paired_is_deterministic() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(loc=1.0, size=20)
    b = rng.normal(loc=0.0, size=20)
    stat_fn = lambda x, y: float(np.median(x - y))
    lo1, hi1 = bootstrap_ci_paired(a, b, stat_fn, n_boot=500, seed=999)
    lo2, hi2 = bootstrap_ci_paired(a, b, stat_fn, n_boot=500, seed=999)
    assert (lo1, hi1) == (lo2, hi2)
    assert lo1 <= hi1


def test_bootstrap_ci_paired_different_seed_can_differ() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(loc=1.0, size=20)
    b = rng.normal(loc=0.0, size=20)
    stat_fn = lambda x, y: float(np.median(x - y))
    lo1, hi1 = bootstrap_ci_paired(a, b, stat_fn, n_boot=200, seed=1)
    lo2, hi2 = bootstrap_ci_paired(a, b, stat_fn, n_boot=200, seed=2)
    assert (lo1, hi1) != (lo2, hi2)


def test_tost_equivalence_accepts_when_diffs_near_zero() -> None:
    diffs = np.array([0.001, -0.002, 0.0015, -0.001, 0.0005, -0.0008])
    p1, p2, p_tost = tost_equivalence(diffs, margin=0.01)
    assert p_tost < 0.05, "small differences within margin=0.01 should pass the TOST equivalence"


def test_tost_equivalence_rejects_when_diffs_large() -> None:
    diffs = np.array([0.5, 0.6, 0.55, 0.52, 0.58, 0.51])
    p1, p2, p_tost = tost_equivalence(diffs, margin=0.01)
    assert p_tost > 0.05, "large systematic differences must not pass the TOST equivalence"


def test_wilcoxon_pvalue_safe_nan_below_min_n() -> None:
    assert np.isnan(wilcoxon_pvalue_safe(np.array([0.1, 0.2, 0.3]), "greater", min_n=5))


def test_wilcoxon_pvalue_safe_nan_for_all_zero_diffs() -> None:
    assert np.isnan(wilcoxon_pvalue_safe(np.zeros(10), "greater", min_n=5))
