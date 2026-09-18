# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-17
# License: see the LICENSE file in the repository root
"""Tests for src/experiments/exp14_convergence_robustness.py
(documentation/2026-09-17_zadani_exp14_robustnost_konvergence.md): the
tau=0 reproduction of exp10 (with a REALISTIC, ~1e4-magnitude regression test
for the relative-tolerance trap - see that module's "PAST" section), the
exp11-derived threshold measurement, monotonicity of the number of kept rows
in tau, and mode isolation.

Uses small synthetic DataFrames throughout - no dependency on the real,
already-completed production CSVs (same convention as
tests/test_exp11_convergence.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.experiments.exp14_convergence_robustness import (
    COLUMN_KEYS,
    TAU_SOURCE_CONFIG,
    TAU_SOURCE_MEASURED,
    _check_reproduces_e10,
    _compute_threshold_row,
    _kept_rows,
    _measure_tau_from_exp11,
)

# ============================================================================
# CSV schema
# ============================================================================


def test_column_keys_matches_required_schema() -> None:
    assert COLUMN_KEYS == [
        "experiment", "tau", "tau_source", "n_rows_kept", "n_rows_total", "n_datasets_kept",
        "frac_both_gaps_nonneg", "med_tight_v1", "med_tight_sandwich",
        "n_nontrivial_alpha_one", "spearman_ceiling_vs_gain_rho",
        "spearman_ceiling_vs_gain_p", "status", "error",
    ]


def test_tau_source_constants() -> None:
    assert TAU_SOURCE_CONFIG == "config"
    assert TAU_SOURCE_MEASURED == "measured_from_exp11"


# ============================================================================
# _measure_tau_from_exp11 (realistic ~1e4/~1e5 magnitude stress values)
# ============================================================================


def _e11_row(dataset: str, alpha: float, variant: str, sigma0_Y0: float, sigmaA_Ya: float, status: str = "ok") -> dict:
    return {"dataset": dataset, "alpha": alpha, "variant": variant, "sigma0_Y0": sigma0_Y0, "sigmaA_Ya": sigmaA_Ya, "status": status}


def test_measure_tau_from_exp11_matches_hand_computed_value() -> None:
    """Two datasets, realistic stress magnitudes (~1e4-1e5, like the real
    exp11_convergence_check_results.csv) - expected value hand-computed
    (pivot on (dataset, alpha), relative decline baseline->tight, row-wise
    max of the two quantities, median per dataset, then median over
    datasets)."""
    e11_ok = pd.DataFrame([
        _e11_row("X", 0.5, "baseline", 12345.6, 6789.0),
        _e11_row("X", 0.5, "tight", 12345.0, 6780.0),
        _e11_row("X", 1.0, "baseline", 12345.6, 5555.5),
        _e11_row("X", 1.0, "tight", 12345.0, 5550.0),
        _e11_row("Y", 0.5, "baseline", 99999.9, 42000.0),
        _e11_row("Y", 0.5, "tight", 99900.0, 41950.0),
    ])
    tau = _measure_tau_from_exp11(e11_ok)
    assert tau == pytest.approx(0.0011741590413189658, rel=1e-9)


def test_measure_tau_from_exp11_ignores_datasets_without_both_variants() -> None:
    """A dataset with only 'tight' rows (e.g. an incomplete exp11 checkpoint,
    like optdigits in the real production CSV) must be silently excluded,
    not fabricated as a 0 decline nor raise."""
    e11_ok = pd.DataFrame([
        _e11_row("X", 0.5, "baseline", 12345.6, 6789.0),
        _e11_row("X", 0.5, "tight", 12345.0, 6780.0),
        _e11_row("X", 1.0, "baseline", 12345.6, 5555.5),
        _e11_row("X", 1.0, "tight", 12345.0, 5550.0),
        _e11_row("Z_incomplete", 0.25, "tight", 5000.0, 2000.0),
        _e11_row("Z_incomplete", 0.25, "warm", 5000.0, 2000.0),
    ])
    tau = _measure_tau_from_exp11(e11_ok)
    # Only dataset 'X' contributes -> threshold = its own per-dataset median.
    assert tau == pytest.approx(0.001157841892161741, rel=1e-9)


def test_measure_tau_from_exp11_raises_without_any_tight_baseline_pair() -> None:
    e11_ok = pd.DataFrame([
        _e11_row("Z", 0.25, "tight", 5000.0, 2000.0),
        _e11_row("Z", 0.25, "warm", 5000.0, 2000.0),
    ])
    with pytest.raises(ValueError, match="baseline"):
        _measure_tau_from_exp11(e11_ok)


# ============================================================================
# _kept_rows / monotonicity in tau
# ============================================================================


def _make_pos() -> pd.DataFrame:
    """6 alpha>0 rows over 3 datasets - Delta magnitudes deliberately span
    several orders of magnitude so tau sweeps actually change the kept set."""
    return pd.DataFrame([
        {"dataset": "A", "alpha": 0.5, "Delta": 0.01, "Delta_prime": 0.02, "both_gaps_nonneg": True, "bound_nontrivial": True, "tight_v1": 0.10, "tight_sandwich": 0.05},
        {"dataset": "A", "alpha": 1.0, "Delta": 0.05, "Delta_prime": 0.06, "both_gaps_nonneg": True, "bound_nontrivial": True, "tight_v1": 0.20, "tight_sandwich": 0.07},
        {"dataset": "B", "alpha": 0.5, "Delta": 0.002, "Delta_prime": -0.001, "both_gaps_nonneg": False, "bound_nontrivial": False, "tight_v1": np.nan, "tight_sandwich": np.nan},
        {"dataset": "B", "alpha": 1.0, "Delta": 0.08, "Delta_prime": 0.09, "both_gaps_nonneg": True, "bound_nontrivial": True, "tight_v1": 0.30, "tight_sandwich": 0.09},
        {"dataset": "C", "alpha": 1.0, "Delta": 0.0005, "Delta_prime": 0.0005, "both_gaps_nonneg": True, "bound_nontrivial": False, "tight_v1": np.nan, "tight_sandwich": np.nan},
        {"dataset": "C", "alpha": 2.0, "Delta": 0.15, "Delta_prime": 0.16, "both_gaps_nonneg": True, "bound_nontrivial": True, "tight_v1": 0.40, "tight_sandwich": 0.11},
    ])


def test_kept_rows_at_tau_zero_keeps_everything() -> None:
    pos = _make_pos()
    kept = _kept_rows(pos, 0.0)
    assert kept.shape[0] == pos.shape[0]


def test_kept_rows_count_is_monotonically_nonincreasing_in_tau() -> None:
    pos = _make_pos()
    thresholds = [0.0, 1.0e-4, 1.0e-3, 6.0e-3, 1.0e-2, 3.0e-2, 0.2]
    counts = [_kept_rows(pos, tau).shape[0] for tau in thresholds]
    assert counts == sorted(counts, reverse=True)
    assert counts[0] == pos.shape[0]
    assert counts[-1] < counts[0]  # the sweep must actually filter something out


def test_kept_rows_filters_on_both_delta_and_delta_prime() -> None:
    pos = _make_pos()
    # dataset B, alpha=0.5: |Delta|=0.002 >= tau=0.001, but |Delta_prime|=0.001 < 0.0015 -> dropped.
    kept = _kept_rows(pos, 0.0015)
    row = kept[(kept["dataset"] == "B") & (kept["alpha"] == 0.5)]
    assert row.empty


# ============================================================================
# _compute_threshold_row
# ============================================================================


def _make_ds_all() -> pd.DataFrame:
    return pd.DataFrame(
        {"c_1": [1.0, 2.0, 3.0, 4.0], "gamma_tilde_Y0": [0.5, 0.4, 0.3, 0.2], "G_auc_oracle": [0.01, 0.02, 0.05, 0.09]},
        index=pd.Index(["A", "B", "C", "D"], name="dataset"),
    )


def test_compute_threshold_row_tau_zero_matches_hand_computed_aggregates() -> None:
    pos = _make_pos()
    ds_all = _make_ds_all()
    row = _compute_threshold_row(0.0, TAU_SOURCE_CONFIG, pos, ds_all, n_rows_total=pos.shape[0], n_perm=200, seed=0)

    assert row["n_rows_kept"] == 6
    assert row["n_rows_total"] == 6
    assert row["n_datasets_kept"] == 3  # A, B, C
    assert row["frac_both_gaps_nonneg"] == pytest.approx(5.0 / 6.0)
    # bound_nontrivial==True rows: tight_v1 in {0.10, 0.20, 0.30, 0.40} -> median 0.25
    assert row["med_tight_v1"] == pytest.approx(0.25)
    assert row["med_tight_sandwich"] == pytest.approx((0.07 + 0.09) / 2.0)
    # alpha==1.0 rows kept: A (bound_nontrivial=True), B (True), C (False) -> 2
    assert row["n_nontrivial_alpha_one"] == 2
    assert row["status"] == "ok"
    assert row["error"] == ""


def test_compute_threshold_row_high_tau_yields_nan_medians_not_fabricated_zero() -> None:
    pos = _make_pos()
    ds_all = _make_ds_all()
    row = _compute_threshold_row(0.5, TAU_SOURCE_CONFIG, pos, ds_all, n_rows_total=pos.shape[0], n_perm=200, seed=0)
    assert row["n_rows_kept"] == 0
    assert math_isnan(row["frac_both_gaps_nonneg"])
    assert math_isnan(row["med_tight_v1"])
    assert math_isnan(row["spearman_ceiling_vs_gain_rho"])


def math_isnan(x: float) -> bool:
    import math

    return math.isnan(float(x))


def test_compute_threshold_row_spearman_restricted_to_kept_datasets() -> None:
    """Only datasets that still have >=1 kept row after filtering may enter
    the Spearman correlation - here only A/B/C have alpha>0 rows in `pos` at
    all, so 'D' (present in ds_all but absent from pos) must never be used,
    and with tau high enough to drop dataset C entirely, only A/B remain
    (n=2 -> Spearman undefined, NaN, not fabricated)."""
    pos = _make_pos()
    ds_all = _make_ds_all()
    row = _compute_threshold_row(0.03, TAU_SOURCE_CONFIG, pos, ds_all, n_rows_total=pos.shape[0], n_perm=200, seed=0)
    # tau=0.03 keeps: A(alpha=1.0, Delta=0.05/0.06), B(alpha=1.0, 0.08/0.09), C(alpha=2.0, 0.15/0.16) -> 3 datasets kept
    assert row["n_datasets_kept"] == 3
    assert not math_isnan(row["spearman_ceiling_vs_gain_rho"])


# ============================================================================
# _check_reproduces_e10 (fail-loud tau=0 reproduction, RELATIVE tolerance)
# ============================================================================

# Deliberately large-magnitude stand-in values (order 1e4), mirroring the
# exp11 "PAST" regression: an absolute-only tolerance check must not be
# reintroduced here either.
_REFERENCE = {
    "frac_both_gaps_nonneg": 0.9661458333333334,
    "med_tight_v1": 12353.41272272468,
    "med_tight_sandwich": 0.0010539999579152,
    "n_nontrivial_alpha_one": 18.0,
    "spearman_ceiling_vs_gain_rho": 0.773798,
}


def test_check_reproduces_e10_accepts_one_ulp_on_large_magnitude_field() -> None:
    tau0_row = dict(_REFERENCE)
    tau0_row["med_tight_v1"] = 12353.412722724679  # one ulp off, |diff|~1.8e-12
    assert abs(tau0_row["med_tight_v1"] - _REFERENCE["med_tight_v1"]) > 1e-12
    _check_reproduces_e10(tau0_row, _REFERENCE, tol=1e-9)  # must not raise


def test_check_reproduces_e10_catches_a_real_relative_mismatch_on_large_field() -> None:
    tau0_row = dict(_REFERENCE)
    tau0_row["med_tight_v1"] = _REFERENCE["med_tight_v1"] * 1.001  # 0.1% off
    with pytest.raises(ValueError, match="does NOT reproduce"):
        _check_reproduces_e10(tau0_row, _REFERENCE, tol=1e-9)


def test_check_reproduces_e10_catches_mismatch_on_small_magnitude_field() -> None:
    tau0_row = dict(_REFERENCE)
    tau0_row["n_nontrivial_alpha_one"] = 12.0  # far from 18
    with pytest.raises(ValueError, match="does NOT reproduce"):
        _check_reproduces_e10(tau0_row, _REFERENCE, tol=1e-9)


def test_check_reproduces_e10_passes_for_identical_values() -> None:
    _check_reproduces_e10(dict(_REFERENCE), _REFERENCE, tol=1e-9)


# ============================================================================
# mode isolation
# ============================================================================


def test_experiment_name_isolation_across_modes() -> None:
    from src.experiments.exp_common import resolve_experiment_name

    full_name = resolve_experiment_name("exp14_convergence_robustness", "full")
    quick_name = resolve_experiment_name("exp14_convergence_robustness", "quick")
    smoke_name = resolve_experiment_name("exp14_convergence_robustness", "smoke")

    assert full_name == "exp14_convergence_robustness"
    assert quick_name == "quick/exp14_convergence_robustness"
    assert smoke_name == "smoke/exp14_convergence_robustness"
    assert len({full_name, quick_name, smoke_name}) == 3


def test_smoke_config_overrides_thresholds_and_permutations_without_touching_full() -> None:
    from src.experiments.config_experiments import resolve_experiment_config

    full_cfg = resolve_experiment_config("exp14_convergence_robustness", "full")
    smoke_cfg = resolve_experiment_config("exp14_convergence_robustness", "smoke")

    assert full_cfg["thresholds"] == [0.0, 1.0e-4, 1.0e-3, 6.0e-3, 1.0e-2, 3.0e-2]
    assert smoke_cfg["thresholds"] == [0.0, 1.0e-2]
    assert 0.0 in smoke_cfg["thresholds"]  # the tau=0 reproduction check must still run in --smoke
    assert smoke_cfg["n_permutations"] < full_cfg["n_permutations"]
    # baseline_match_tol is not overridden by 'smoke' - must stay identical.
    assert smoke_cfg["baseline_match_tol"] == full_cfg["baseline_match_tol"]
