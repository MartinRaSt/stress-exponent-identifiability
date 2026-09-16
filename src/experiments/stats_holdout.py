# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-14
# License: see the LICENSE file in the repository root
"""
Q1 step 2 (reserse/2026-09-14_specifikace_rozsireni_q1.md, A.6) - pure
statistical functions (no file I/O) shared between
`src/experiments/exp1_holdout_confirmatory.py` and its tests
(tests/test_exp1_holdout_confirmatory.py): Holm correction, exact/MC
sign-flip permutation test, Hodges-Lehmann estimator, unpaired
Cliff's delta, percentile bootstrap CI, and TOST equivalence test.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import ttest_1samp, wilcoxon


def holm_correction(pvalues: list[float] | np.ndarray) -> list[float]:
    """Holm (1979) step-down: sort p_(1)<=...<=p_(m), adjusted
    p_(j) = max_{i<=j} min(1, (m-i+1)*p_(i)) - returns the adjusted p-values
    in the ORIGINAL input order (not sorted). NaN inputs are ignored (they
    are returned as NaN in their place, not counted into m, and do not
    affect the ordering)."""
    p = np.asarray(pvalues, dtype=np.float64)
    m_total = p.shape[0]
    out = np.full(m_total, np.nan)
    valid_idx = np.where(np.isfinite(p))[0]
    m = valid_idx.shape[0]
    if m == 0:
        return out.tolist()
    p_valid = p[valid_idx]
    order = np.argsort(p_valid, kind="stable")
    sorted_p = p_valid[order]
    adjusted_sorted = np.empty(m)
    running_max = 0.0
    for i in range(m):  # i=0-based rank, multiplier = m - i (= m-(i+1)+1)
        val = min(1.0, (m - i) * sorted_p[i])
        running_max = max(running_max, val)
        adjusted_sorted[i] = running_max
    adjusted_valid = np.empty(m)
    adjusted_valid[order] = adjusted_sorted
    out[valid_idx] = adjusted_valid
    return out.tolist()


def _all_subset_signed_sums(arr: np.ndarray) -> np.ndarray:
    """Returns all 2^len(arr) sums of +-arr[0] +-arr[1] ... (vectorized
    doubling, O(2^len(arr)) memory/time - used only for halves with n<=~12)."""
    sums = np.array([0.0])
    for v in arr:
        sums = np.concatenate([sums - v, sums + v])
    return sums


def exact_sign_flip_pvalue(diffs: np.ndarray, alternative: str) -> tuple[float, int]:
    """Exact sign-flip permutation test: under H0, each of the 2^n sign
    patterns of |diff_i| is equally likely (diffs symmetric about 0).
    Statistic T = mean(diffs) - since n is the same for all patterns, the
    ordering of T matches the ordering of the sum, so the sum is used
    directly (faster).

    Meet-in-the-middle (split into 2 halves, O(2^(n/2) log) instead of
    O(2^n)) - enables exact computation up to n~22 (2^11=2048 per half)
    consistent with `exp1_holdout_confirmatory.exact_perm_max_n`.

    Returns (p-value, number_of_permutations=2^n)."""
    d = np.asarray(diffs, dtype=np.float64)
    d = d[d != 0.0]
    n = d.shape[0]
    if n == 0:
        return float("nan"), 0
    abs_d = np.abs(d)
    T_obs_sum = float(np.sum(d))
    half = n // 2
    left_sums = _all_subset_signed_sums(abs_d[:half])
    right_sums = np.sort(_all_subset_signed_sums(abs_d[half:]))
    total = 1 << n

    if alternative == "greater":
        idx = np.searchsorted(right_sums, T_obs_sum - left_sums, side="left")
        count = int(np.sum(len(right_sums) - idx))
    elif alternative == "less":
        idx = np.searchsorted(right_sums, T_obs_sum - left_sums, side="right")
        count = int(np.sum(idx))
    elif alternative == "two-sided":
        c = abs(T_obs_sum)
        idx_ge = np.searchsorted(right_sums, c - left_sums, side="left")
        count_ge = int(np.sum(len(right_sums) - idx_ge))
        count = min(total, 2 * count_ge)
    else:
        raise ValueError(f"Unknown alternative='{alternative}' (expected greater/less/two-sided).")
    return float(count) / float(total), total


def mc_sign_flip_pvalue(diffs: np.ndarray, alternative: str, seed: int, n_perm: int) -> tuple[float, int]:
    """Monte-Carlo approximation of the sign-flip test (for n > exact_perm_max_n):
    p = (#{T_perm compared to T_obs} + 1) / (n_perm + 1) (conservative
    convention, see spec A.6)."""
    d = np.asarray(diffs, dtype=np.float64)
    d = d[d != 0.0]
    n = d.shape[0]
    if n == 0:
        return float("nan"), 0
    abs_d = np.abs(d)
    T_obs = float(np.sum(d))
    rng = np.random.default_rng(seed)
    signs = rng.integers(0, 2, size=(n_perm, n)).astype(np.float64) * 2.0 - 1.0
    T_perm = signs @ abs_d
    if alternative == "greater":
        c = int(np.sum(T_perm >= T_obs))
    elif alternative == "less":
        c = int(np.sum(T_perm <= T_obs))
    elif alternative == "two-sided":
        c = int(np.sum(np.abs(T_perm) >= abs(T_obs)))
    else:
        raise ValueError(f"Unknown alternative='{alternative}' (expected greater/less/two-sided).")
    return float(c + 1) / float(n_perm + 1), n_perm


def sign_flip_test(diffs: np.ndarray, alternative: str, seed: int, n_perm_mc: int, exact_max_n: int) -> tuple[float, int, bool]:
    """Dispatcher: exact enumeration if (number of nonzero diffs) <=
    `exact_max_n`, otherwise Monte Carlo. Returns (p-value, number_of_permutations_used, exact_bool)."""
    d = np.asarray(diffs, dtype=np.float64)
    n_nonzero = int(np.sum(d != 0.0))
    if n_nonzero <= exact_max_n:
        p, n_perm = exact_sign_flip_pvalue(d, alternative)
        return p, n_perm, True
    p, n_perm = mc_sign_flip_pvalue(d, alternative, seed, n_perm_mc)
    return p, n_perm, False


def hodges_lehmann(diffs: np.ndarray) -> float:
    """Hodges-Lehmann location estimator: median of the Walsh averages (d_i+d_j)/2, i<=j."""
    d = np.asarray(diffs, dtype=np.float64)
    n = d.shape[0]
    if n == 0:
        return float("nan")
    i, j = np.triu_indices(n)
    walsh = (d[i] + d[j]) / 2.0
    return float(np.median(walsh))


def cliff_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff (1993), DOI 10.1037/0033-2909.114.3.494 - unpaired delta:
    (#{a_i>b_j} - #{a_i<b_j}) / (n_a*n_b)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return float("nan")
    diff_matrix = a[:, None] - b[None, :]
    gt = np.sum(diff_matrix > 0)
    lt = np.sum(diff_matrix < 0)
    return float((gt - lt) / (a.shape[0] * b.shape[0]))


def delta_pair(diffs: np.ndarray) -> float:
    """Paired dominance delta_pair = (#{diff_i>0} - #{diff_i<0}) / n."""
    d = np.asarray(diffs, dtype=np.float64)
    n = d.shape[0]
    if n == 0:
        return float("nan")
    return float((np.sum(d > 0) - np.sum(d < 0)) / n)


def bootstrap_ci_paired(a: np.ndarray, b: np.ndarray, stat_fn, n_boot: int, seed: int, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI: resampling of INDICES (paired, the same index
    for both a and b - simulates resampling the DATASET) with replacement,
    `stat_fn(a', b')` -> statistic value. Deterministic for a given seed
    (test: `test_bootstrap_ci_paired_is_deterministic`)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = a.shape[0]
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    vals = np.empty(n_boot, dtype=np.float64)
    for k in range(n_boot):
        sel = idx[k]
        vals[k] = stat_fn(a[sel], b[sel])
    lo = float(np.percentile(vals, 100.0 * alpha / 2.0))
    hi = float(np.percentile(vals, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def tost_equivalence(diffs: np.ndarray, margin: float) -> tuple[float, float, float]:
    """Two One-Sided Tests (TOST) equivalence of paired differences against
    the band [-margin, margin] (paired t-test on `diffs`, standard TOST
    construction):
    H1_low:  mean(diffs) < margin   (one-sided t-test, alternative='less')
    H1_high: mean(diffs) > -margin  (one-sided t-test, alternative='greater')
    Equivalence is accepted if BOTH tests reject H0 (p_tost=max(p1,p2) < alpha).
    Returns (p1, p2, p_tost). n<2 -> (nan,nan,nan) (t-test undefined)."""
    d = np.asarray(diffs, dtype=np.float64)
    n = d.shape[0]
    if n < 2 or np.allclose(d, d[0]):
        # n<2 or zero variance - paired t-test is undefined without fabricating variance
        return float("nan"), float("nan"), float("nan")
    res_low = ttest_1samp(d, margin, alternative="less")
    res_high = ttest_1samp(d, -margin, alternative="greater")
    p1, p2 = float(res_low.pvalue), float(res_high.pvalue)
    return p1, p2, max(p1, p2)


def wilcoxon_pvalue_safe(diffs: np.ndarray, alternative: str, min_n: int = 5) -> float:
    """Wilcoxon signed-rank test as a cross-check - NaN if n<min_n or all
    differences are (numerically) zero (degenerate case with no defined
    test, same convention as `exp1_regime_stratified._wilcoxon_pvalue`)."""
    d = np.asarray(diffs, dtype=np.float64)
    if d.shape[0] < min_n:
        return float("nan")
    if np.allclose(d, 0.0, atol=1e-12):
        return float("nan")
    try:
        res = wilcoxon(d, alternative=alternative, zero_method="wilcox")
    except ValueError:
        return float("nan")
    return float(res.pvalue)
