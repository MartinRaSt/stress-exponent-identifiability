# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-13
# License: see the LICENSE file in the repository root
"""K14 (cleanup/hardening): the Nemenyi q_alpha computed via
`scipy.stats.studentized_range` must agree with the original manually
transcribed table (Demsar 2006, Tab. 5) for k<=20 and, in addition, must
also work for k>20 (E1 supplement over 21 methods, E5 combinations), where
the original table ends and caused a KeyError/WARNING (see projectstate.md
2026-09-13 12:50)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.experiments.config_experiments import load_experiments_config
from src.experiments.stats import (
    bootstrap_ci_1d,
    bootstrap_spearman_ci,
    cluster_bootstrap_ols_slope,
    friedman_nemenyi,
    maximal_insignificant_cliques,
    nemenyi_cd,
    nemenyi_q_alpha,
    spearman_permutation_test,
)


def test_nemenyi_q_alpha_matches_legacy_table_for_k_le_20() -> None:
    """The original table in config_experiments.yaml
    (stats.nemenyi_q_alpha_0_05, Demsar 2006 Tab. 5, alpha=0.05) is kept as
    a reference - the formula must agree to within 1e-3."""
    stats_cfg = load_experiments_config()["stats"]
    legacy_table = {int(k): float(v) for k, v in stats_cfg["nemenyi_q_alpha_0_05"].items()}
    assert len(legacy_table) >= 19  # k=2..20

    for k, q_legacy in legacy_table.items():
        q_formula = nemenyi_q_alpha(k, alpha=stats_cfg["alpha"])
        assert abs(q_formula - q_legacy) < 1e-3, f"k={k}: formula={q_formula} vs table={q_legacy}"


@pytest.mark.parametrize("k", [21, 25, 50, 117])
def test_nemenyi_q_alpha_works_beyond_legacy_table(k: int) -> None:
    """The original table ends at k=20 (KeyError) - the formula must also
    work for a larger number of methods (E1 supplement k=21, E5
    combinations k=117) and give a reasonable (finite, positive, increasing
    with k) value."""
    q = nemenyi_q_alpha(k, alpha=0.05)
    assert np.isfinite(q)
    assert q > nemenyi_q_alpha(2, alpha=0.05)


def test_nemenyi_q_alpha_monotone_increasing_in_k() -> None:
    values = [nemenyi_q_alpha(k, alpha=0.05) for k in range(2, 30)]
    assert all(b >= a for a, b in zip(values, values[1:]))


def test_nemenyi_q_alpha_rejects_k_below_2() -> None:
    with pytest.raises(ValueError):
        nemenyi_q_alpha(1, alpha=0.05)


def test_friedman_nemenyi_handles_more_than_20_methods_without_error() -> None:
    """Regression test for the 'q_alpha for k=21 is missing' incident:
    friedman_nemenyi must not raise a KeyError for n_methods>20 (previously:
    a 'skipping Friedman/Nemenyi' WARNING in run_main)."""
    rng = np.random.default_rng(0)
    n_datasets, n_methods = 10, 25
    wide = pd.DataFrame(rng.normal(size=(n_datasets, n_methods)), columns=[f"m{i}" for i in range(n_methods)])
    result = friedman_nemenyi(wide, direction="max", alpha=0.05)
    assert result["n_methods"] == n_methods
    assert np.isfinite(result["cd"])
    assert result["cd"] > 0


def test_nemenyi_cd_uses_computed_q_alpha() -> None:
    q = nemenyi_q_alpha(5, alpha=0.05)
    cd = nemenyi_cd(q, n_methods=5, n_datasets=10)
    assert cd == pytest.approx(q * np.sqrt(5 * 6 / (6.0 * 10)))


def test_maximal_insignificant_cliques_matches_exp1_dr_benchmark_main_example() -> None:
    """Pinned to the real exp1_dr_benchmark_main/auc_rnx numbers (author
    feedback 2026-09-17): 13 methods, CD=2.555 must yield exactly the 4
    maximal cliques found by manual/independent computation, with the top
    method (tsne_auto) isolated (no bar - it differs significantly from
    every other method)."""
    ranks = [
        1.7843137254901962, 4.5588235294117645, 5.294117647058823, 5.911764705882353,
        5.96078431372549, 7.0588235294117645, 7.088235294117647, 7.431372549019608,
        7.549019607843137, 8.03921568627451, 8.196078431372548, 11.049019607843137, 11.07843137254902,
    ]
    cd = 2.55483082833608
    cliques = maximal_insignificant_cliques(ranks, cd)
    assert cliques == [(1, 6), (2, 8), (3, 10), (11, 12)]
    covered = {i for s, e in cliques for i in range(s, e + 1)}
    assert 0 not in covered  # the best method (tsne_auto) is significantly better than all others


def test_maximal_insignificant_cliques_requires_ascending_input() -> None:
    # a single clique spanning everything when all ranks are within cd
    cliques = maximal_insignificant_cliques([1.0, 1.5, 2.0], cd=1.5)
    assert cliques == [(0, 2)]


def test_maximal_insignificant_cliques_no_edges_returns_empty() -> None:
    # ranks far enough apart that no pair is "not significantly different"
    cliques = maximal_insignificant_cliques([1.0, 5.0, 9.0], cd=2.0)
    assert cliques == []


def test_maximal_insignificant_cliques_drops_subset_cliques() -> None:
    # (0,1) and (1,2) are both candidate windows, but if (0,2) also qualifies
    # only the maximal (0,2) clique must be returned
    cliques = maximal_insignificant_cliques([1.0, 1.9, 2.8], cd=2.0)
    assert cliques == [(0, 2)]


def test_bootstrap_spearman_ci_perfect_monotone_relationship() -> None:
    x = np.arange(1, 21, dtype=np.float64)
    y = x**2  # exactly monotonic -> rho=1, the CI should be narrow around 1
    out = bootstrap_spearman_ci(x, y, n_boot=500, seed=0, ci_level=0.95)
    assert out["rho"] == pytest.approx(1.0)
    assert out["n"] == 20
    assert out["ci_lo"] > 0.9
    assert out["ci_hi"] <= 1.0 + 1e-9


def test_bootstrap_spearman_ci_drops_nan_pairs_and_is_deterministic() -> None:
    x = np.array([1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0])
    y = np.array([2.0, 1.0, 4.0, 9.0, np.nan, 5.0, 8.0, 6.0])
    out1 = bootstrap_spearman_ci(x, y, n_boot=200, seed=42, ci_level=0.95)
    out2 = bootstrap_spearman_ci(x, y, n_boot=200, seed=42, ci_level=0.95)
    assert out1["n"] == 6  # 2 NaN pairs dropped
    assert out1 == out2  # deterministic for the same seed


def test_bootstrap_spearman_ci_too_few_points_returns_nan() -> None:
    out = bootstrap_spearman_ci(np.array([1.0, 2.0]), np.array([1.0, 2.0]), n_boot=100, seed=0)
    assert np.isnan(out["rho"])
    assert out["n"] == 2


# ============================================================================
# spearman_permutation_test (used by exp10_identifiability_stats.py H1)
# ============================================================================


def test_spearman_permutation_test_perfect_monotone_gives_small_p() -> None:
    x = np.arange(1, 21, dtype=np.float64)
    y = x**3
    out = spearman_permutation_test(x, y, n_perm=2000, seed=0, alternative="two-sided")
    assert out["rho"] == pytest.approx(1.0)
    assert out["n"] == 20
    assert out["pvalue"] < 0.01  # perfect rank agreement must be extreme under any permutation null


def test_spearman_permutation_test_matches_scipy_rho() -> None:
    from scipy.stats import spearmanr

    rng = np.random.default_rng(1)
    x = rng.normal(size=30)
    y = 0.5 * x + rng.normal(size=30)
    out = spearman_permutation_test(x, y, n_perm=3000, seed=1, alternative="two-sided")
    rho_scipy, _ = spearmanr(x, y)
    assert out["rho"] == pytest.approx(rho_scipy, abs=1e-9)


def test_spearman_permutation_test_null_data_p_not_extreme() -> None:
    """Independent (shuffled) x,y - the permutation p-value should NOT be
    tiny (sanity check that the null distribution is correctly centered)."""
    rng = np.random.default_rng(2)
    x = rng.normal(size=25)
    y = rng.normal(size=25)
    out = spearman_permutation_test(x, y, n_perm=5000, seed=3, alternative="two-sided")
    assert out["pvalue"] > 0.05  # not guaranteed in general, but true for this fixed seed/draw


def test_spearman_permutation_test_deterministic_for_same_seed() -> None:
    rng = np.random.default_rng(4)
    x, y = rng.normal(size=15), rng.normal(size=15)
    out1 = spearman_permutation_test(x, y, n_perm=500, seed=42)
    out2 = spearman_permutation_test(x, y, n_perm=500, seed=42)
    assert out1 == out2


def test_spearman_permutation_test_too_few_points_returns_nan() -> None:
    out = spearman_permutation_test(np.array([1.0, 2.0]), np.array([1.0, 2.0]), n_perm=100, seed=0)
    assert np.isnan(out["rho"])
    assert np.isnan(out["pvalue"])
    assert out["n"] == 2


def test_spearman_permutation_test_drops_nan_pairs() -> None:
    x = np.array([1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0])
    y = np.array([2.0, 1.0, 4.0, 9.0, np.nan, 5.0, 8.0, 6.0])
    out = spearman_permutation_test(x, y, n_perm=200, seed=0)
    assert out["n"] == 6


def test_spearman_permutation_test_constant_x_returns_zero_rho_nan_p() -> None:
    x = np.ones(10)
    y = np.arange(10, dtype=np.float64)
    out = spearman_permutation_test(x, y, n_perm=100, seed=0)
    assert out["rho"] == 0.0
    assert np.isnan(out["pvalue"])


# ============================================================================
# bootstrap_ci_1d / cluster_bootstrap_ols_slope (exp10_identifiability_stats.py H2/regression)
# ============================================================================


def test_bootstrap_ci_1d_narrow_around_constant() -> None:
    values = np.full(50, 3.0)
    lo, hi = bootstrap_ci_1d(values, np.median, n_boot=500, seed=0)
    assert lo == pytest.approx(3.0)
    assert hi == pytest.approx(3.0)


def test_bootstrap_ci_1d_contains_true_median_for_normal_sample() -> None:
    rng = np.random.default_rng(5)
    values = rng.normal(loc=2.0, scale=0.5, size=200)
    lo, hi = bootstrap_ci_1d(values, np.median, n_boot=2000, seed=5)
    assert lo < 2.0 < hi


def test_bootstrap_ci_1d_too_few_points_returns_nan() -> None:
    lo, hi = bootstrap_ci_1d(np.array([1.0]), np.median, n_boot=100, seed=0)
    assert np.isnan(lo)
    assert np.isnan(hi)


def test_cluster_bootstrap_ols_slope_recovers_known_slope() -> None:
    """Synthetic data with a KNOWN slope=2, several rows per cluster
    (dataset) - the point estimate must match exactly (noiseless) and the CI
    must be narrow and contain the true slope."""
    rng = np.random.default_rng(6)
    rows = []
    for cluster_id in range(15):
        x0 = rng.uniform(0.5, 5.0)
        for _rep in range(4):  # several rows per cluster (like several seeds per dataset)
            noise = rng.normal(scale=1e-9)
            rows.append({"cluster": f"c{cluster_id}", "x": x0, "y": 2.0 * x0 + 1.0 + noise})
    df = pd.DataFrame(rows)
    out = cluster_bootstrap_ols_slope(df, "x", "y", "cluster", n_boot=500, seed=7)
    assert out["slope"] == pytest.approx(2.0, abs=1e-4)
    assert out["n_clusters"] == 15
    assert out["n_rows"] == 60
    assert out["ci_lo"] < 2.0 < out["ci_hi"]


def test_cluster_bootstrap_ols_slope_rejects_constant_x() -> None:
    df = pd.DataFrame({"x": [1.0] * 10, "y": np.arange(10, dtype=np.float64), "cluster": [f"c{i}" for i in range(10)]})
    with pytest.raises(ValueError):
        cluster_bootstrap_ols_slope(df, "x", "y", "cluster", n_boot=100, seed=0)


def test_cluster_bootstrap_ols_slope_drops_nonfinite_rows() -> None:
    rng = np.random.default_rng(8)
    rows = []
    for cluster_id in range(10):
        x0 = rng.uniform(1.0, 3.0)
        rows.append({"cluster": f"c{cluster_id}", "x": x0, "y": 3.0 * x0})
    rows.append({"cluster": "bad", "x": np.nan, "y": 1.0})
    df = pd.DataFrame(rows)
    out = cluster_bootstrap_ols_slope(df, "x", "y", "cluster", n_boot=200, seed=0)
    assert out["n_rows"] == 10
    assert out["n_clusters"] == 10
