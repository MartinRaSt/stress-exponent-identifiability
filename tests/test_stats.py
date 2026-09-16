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
from src.experiments.stats import bootstrap_spearman_ci, friedman_nemenyi, nemenyi_cd, nemenyi_q_alpha


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
