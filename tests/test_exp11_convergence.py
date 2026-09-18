# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-17
# License: see the LICENSE file in the repository root
"""Tests for src/experiments/exp11_convergence_check.py
(documentation/2026-09-17_zadani_exp11_konvergence.md): the CSV schema, the
`delta_vs_baseline` computation, the fail-loud baseline-vs-E10 reproduction
check, and mode isolation (smoke/quick outputs never touch the full paths).

A full end-to-end run needs completed exp6_alpha_curves/exp10_identifiability_check
data (author-run only, per project rule) - these tests instead exercise the
pure/testable pieces directly, plus one fully-mocked `_run_single` call to
check the wiring (dataset -> D/eps_D -> cached embeddings -> veta1_quantities
-> row) without touching any real cache."""
from __future__ import annotations

import numpy as np
import pytest

from src.experiments.exp11_convergence_check import (
    COLUMN_KEYS,
    VARIANTS,
    _check_baseline_match,
    _delta_vs_baseline,
    _method_name_for_alpha,
    _method_name_for_task,
    _run_single,
)

# ============================================================================
# CSV schema
# ============================================================================

_REQUIRED_COLUMNS = [
    "alpha", "variant", "n_samples", "eps_d",
    "sigma0_Y0", "sigma0_Ya", "sigmaA_Y0", "sigmaA_Ya",
    "Delta", "Delta_prime", "both_gaps_nonneg",
    "n_iter_alpha0", "n_iter_alpha", "converged_alpha0", "converged_alpha",
    "delta_vs_baseline",
]


def test_column_keys_matches_required_schema() -> None:
    assert COLUMN_KEYS == _REQUIRED_COLUMNS


def test_variants_are_baseline_tight_warm() -> None:
    assert VARIANTS == ("baseline", "tight", "warm")


def test_method_name_helpers_encode_alpha_and_variant() -> None:
    assert _method_name_for_alpha(1.0) == "alpha1.0"
    assert _method_name_for_task(1.0, "tight") == "alpha1.0_tight"
    assert _method_name_for_task(0.75, "warm") == "alpha0.75_warm"


# ============================================================================
# delta_vs_baseline
# ============================================================================


def test_delta_vs_baseline_is_exactly_zero_for_baseline_variant() -> None:
    # Deliberately pass slightly different values - baseline must return
    # exactly 0.0 by definition, NOT delta_value - delta_ref (which would be
    # a tiny nonzero float here).
    assert _delta_vs_baseline(0.05000001, 0.05, "baseline") == 0.0


def test_delta_vs_baseline_tight_and_warm_compute_the_difference() -> None:
    assert _delta_vs_baseline(0.02, -0.017, "tight") == pytest.approx(0.037)
    assert _delta_vs_baseline(-0.017, -0.017, "warm") == pytest.approx(0.0)


# ============================================================================
# _check_baseline_match (fail-loud reproduction of E10)
# ============================================================================

_REF_ROW = {
    "n_samples": 214, "eps_d": 0.01, "sigma0_Y0": 10.0, "sigma0_Ya": 10.5,
    "sigmaA_Y0": 5.0, "sigmaA_Ya": 4.9, "Delta": 0.05, "Delta_prime": -0.017,
}


def test_check_baseline_match_passes_for_identical_values() -> None:
    computed = dict(_REF_ROW)
    _check_baseline_match(computed, _REF_ROW, tol=1e-12, dataset_name="glass", alpha=1.0)


def test_check_baseline_match_raises_on_mismatch() -> None:
    computed = dict(_REF_ROW)
    computed["Delta"] = _REF_ROW["Delta"] + 1e-6  # far above baseline_match_tol
    with pytest.raises(ValueError, match="does NOT reproduce"):
        _check_baseline_match(computed, _REF_ROW, tol=1e-12, dataset_name="glass", alpha=1.0)


def test_check_baseline_match_lists_every_mismatched_field() -> None:
    computed = dict(_REF_ROW)
    computed["Delta"] = _REF_ROW["Delta"] + 1e-6
    computed["n_samples"] = _REF_ROW["n_samples"] + 1
    with pytest.raises(ValueError) as excinfo:
        _check_baseline_match(computed, _REF_ROW, tol=1e-12, dataset_name="glass", alpha=1.0)
    msg = str(excinfo.value)
    assert "Delta:" in msg
    assert "n_samples:" in msg
    assert "sigma0_Y0:" not in msg  # unaffected field must not be listed


def test_check_baseline_match_tolerates_differences_within_tol() -> None:
    computed = dict(_REF_ROW)
    computed["Delta"] = _REF_ROW["Delta"] + 1e-10
    _check_baseline_match(computed, _REF_ROW, tol=1e-9, dataset_name="glass", alpha=1.0)


# ============================================================================
# _run_single wiring (fully mocked dataset/embeddings - no real cache needed)
# ============================================================================


class _FakeDataset:
    """Minimal stand-in for src.datasets.registry.Dataset (kind='vector')."""

    def __init__(self, X: np.ndarray) -> None:
        self.X = X
        self.y = None
        self.kind = "vector"
        self.name = "fake"

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]


def _make_task(variant: str, alpha: float, e10_ref: dict) -> dict:
    return {
        "dataset_name": "fake_dataset", "method_name": _method_name_for_task(alpha, variant),
        "seed": 0, "alpha": alpha, "variant": variant,
        "n_max": 1000, "subsample_seed": 42,
        "eps_D_k": 1, "eps_D_q": 0.5,
        "exp6_experiment_name": "exp6_alpha_curves", "n_components": 2,
        "sammon_cfg": {
            "eps_num": 1e-9,
            "smacof": {"dense_pinv_threshold": 5000, "cg_max_iter": 500, "cg_tol": 1e-6, "reg_rho": 1e-8},
        },
        "tight_max_iter": 500, "tight_tol": 1e-10,
        "baseline_match_tol": 1e-12, "e10_ref": e10_ref,
        "exp6_max_iter": 300, "baseline_n_iter_alpha0": 42.0, "baseline_n_iter_alpha": 17.0,
    }


@pytest.fixture()
def _fake_data_and_embeddings(monkeypatch):
    """Monkeypatches dataset loading and the E6 embedding cache with a small,
    fully deterministic synthetic point cloud - no real files touched."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 3))
    Y0 = rng.normal(size=(12, 2)) * 0.1  # a small arbitrary starting configuration
    Ya = rng.normal(size=(12, 2)) * 0.1

    monkeypatch.setattr("src.datasets.registry.load_dataset", lambda name: _FakeDataset(X))
    monkeypatch.setattr("src.datasets.subsample.subsample_dataset", lambda ds, n_max, random_state: ds)

    def _fake_load_embedding(key):
        if key.method == _method_name_for_alpha(0.0):
            return Y0.copy()
        return Ya.copy()

    monkeypatch.setattr("src.experiments.exp11_convergence_check.load_embedding", _fake_load_embedding)
    return X, Y0, Ya


def test_run_single_baseline_matches_reference_computed_from_same_inputs(_fake_data_and_embeddings) -> None:
    """The baseline variant must reproduce a reference row computed with the
    EXACT SAME D/eps_D/Y0/Ya - i.e. `_check_baseline_match` must not raise
    and the row must carry the reused cached embeddings' quantities unchanged."""
    from src.methods.common import to_distance_matrix
    from src.sammon.identifiability import veta1_quantities
    from src.sammon.weights import estimate_eps_D

    X, Y0, Ya = _fake_data_and_embeddings
    alpha = 1.0
    D = to_distance_matrix(X, "vector")
    eps_D = estimate_eps_D(D, k=1, q=0.5, kind="distance")
    v1 = veta1_quantities(D, Y0, Ya, alpha, eps_D)
    e10_ref = {
        "n_samples": D.shape[0], "eps_d": eps_D,
        "sigma0_Y0": v1["sigma0_Y0"], "sigma0_Ya": v1["sigma0_Ya"],
        "sigmaA_Y0": v1["sigmaA_Y0"], "sigmaA_Ya": v1["sigmaA_Ya"],
        "Delta": v1["Delta"], "Delta_prime": v1["Delta_prime"],
    }

    task = _make_task("baseline", alpha, e10_ref)
    result = _run_single(task)

    assert result["status"] == "ok", result["error"]
    assert result["metrics"]["Delta"] == pytest.approx(v1["Delta"], abs=1e-12)
    assert result["metrics"]["delta_vs_baseline"] == 0.0
    assert result["metrics"]["n_iter_alpha0"] == 42.0
    assert result["metrics"]["n_iter_alpha"] == 17.0
    assert result["extra"] == {"alpha": alpha, "variant": "baseline"}


def test_run_single_baseline_fails_loud_on_wrong_reference(_fake_data_and_embeddings) -> None:
    """A deliberately wrong E10 reference (as if exp11 had a bug and computed
    something different) must surface as a status='error' row with a clear
    message - never silently accepted."""
    task = _make_task("baseline", 1.0, e10_ref={
        "n_samples": 12, "eps_d": 0.0, "sigma0_Y0": 999.0, "sigma0_Ya": 999.0,
        "sigmaA_Y0": 999.0, "sigmaA_Ya": 999.0, "Delta": 999.0, "Delta_prime": 999.0,
    })
    result = _run_single(task)
    assert result["status"] == "error"
    assert "does NOT reproduce" in result["error"]


def test_run_single_tight_variant_reruns_smacof_and_reports_iterations(_fake_data_and_embeddings) -> None:
    """'tight' does not need `_check_baseline_match` to pass (only 'baseline'
    is checked) - it must reach 'ok' and report real n_iter/converged values
    from its OWN SMACOF refit (not copied from the baseline task)."""
    e10_ref = {
        "n_samples": 12, "eps_d": 0.1,  # placeholder reference - only used for delta_vs_baseline here
        "sigma0_Y0": 1.0, "sigma0_Ya": 1.0, "sigmaA_Y0": 1.0, "sigmaA_Ya": 1.0,
        "Delta": 0.0, "Delta_prime": 0.0,
    }
    task = _make_task("tight", 1.0, e10_ref)
    result = _run_single(task)

    assert result["status"] == "ok", result["error"]
    assert result["metrics"]["n_iter_alpha0"] >= 1
    assert result["metrics"]["n_iter_alpha"] >= 1
    assert isinstance(result["metrics"]["converged_alpha0"], bool)
    assert result["metrics"]["delta_vs_baseline"] == pytest.approx(result["metrics"]["Delta"] - 0.0)


def test_run_single_unknown_variant_is_an_error_row(_fake_data_and_embeddings) -> None:
    task = _make_task("bogus", 1.0, e10_ref={})
    result = _run_single(task)
    assert result["status"] == "error"
    assert "Unknown variant" in result["error"]


# ============================================================================
# mode isolation (smoke/quick outputs never collide with full paths)
# ============================================================================


def test_experiment_name_isolation_across_modes() -> None:
    from src.experiments.exp_common import resolve_experiment_name

    full_name = resolve_experiment_name("exp11_convergence_check", "full")
    quick_name = resolve_experiment_name("exp11_convergence_check", "quick")
    smoke_name = resolve_experiment_name("exp11_convergence_check", "smoke")

    assert full_name == "exp11_convergence_check"
    assert quick_name == "quick/exp11_convergence_check"
    assert smoke_name == "smoke/exp11_convergence_check"
    assert len({full_name, quick_name, smoke_name}) == 3


def test_smoke_config_overrides_targets_and_tight_without_touching_full() -> None:
    """Resolving exp11_convergence_check in 'smoke' mode must yield the
    reduced target list and loosened tight budget from config_experiments.yaml,
    while 'full' keeps the complete production target list untouched."""
    from src.experiments.config_experiments import resolve_experiment_config

    full_cfg = resolve_experiment_config("exp11_convergence_check", "full")
    smoke_cfg = resolve_experiment_config("exp11_convergence_check", "smoke")

    assert len(full_cfg["targets"]) > 1
    assert smoke_cfg["targets"] == [{"dataset": "glass", "alphas": [1.0]}]
    assert smoke_cfg["tight"]["max_iter"] < full_cfg["tight"]["max_iter"]
    assert smoke_cfg["tight"]["tol"] > full_cfg["tight"]["tol"]
    # baseline_match_tol is not overridden by 'smoke' - must stay identical.
    assert smoke_cfg["baseline_match_tol"] == full_cfg["baseline_match_tol"]
# ---------------------------------------------------------------------------
# Regression: baseline_match_tol must be RELATIVE (bug found in the full run,
# 2026-09-17). With an ABSOLUTE 1e-12 bound, a correct computation on stress
# values of magnitude ~1e4 could never pass: one ulp there is already ~2e-12.
# The observed failure was
#   sigma0_Ya: 12353.412722724679 vs 12353.41272272468  (|diff| = 1.819e-12)
# whose relative difference is 1.5e-16, i.e. exactly machine epsilon.
# ---------------------------------------------------------------------------

_REF_ROW_REALISTIC = {
    "n_samples": 214, "eps_d": 0.4444,
    "sigma0_Y0": 12290.0, "sigma0_Ya": 12353.41272272468,
    "sigmaA_Y0": 2609.932382323017, "sigmaA_Ya": 2594.092746017059,
    "Delta": -0.0044248306275118665, "Delta_prime": 0.07916847215743106,
}


def test_baseline_match_accepts_one_ulp_on_large_magnitude_fields() -> None:
    """A one-ulp difference on a ~1e4 stress value must NOT be reported as a bug."""
    computed = dict(_REF_ROW_REALISTIC)
    computed["sigma0_Ya"] = 12353.412722724679  # the value actually observed
    assert abs(computed["sigma0_Ya"] - _REF_ROW_REALISTIC["sigma0_Ya"]) > 1e-12
    _check_baseline_match(computed, _REF_ROW_REALISTIC, tol=1e-9, dataset_name="glass", alpha=0.75)


def test_baseline_match_still_catches_a_real_relative_mismatch() -> None:
    """Relaxing to a relative bound must not blind the check to genuine bugs."""
    computed = dict(_REF_ROW_REALISTIC)
    computed["sigma0_Ya"] = _REF_ROW_REALISTIC["sigma0_Ya"] * 1.001  # 0.1 %
    with pytest.raises(ValueError, match="does NOT reproduce"):
        _check_baseline_match(computed, _REF_ROW_REALISTIC, tol=1e-9, dataset_name="glass", alpha=0.75)


def test_baseline_match_handles_near_zero_fields() -> None:
    """A field that is legitimately ~0 must not blow the relative comparison up."""
    ref = dict(_REF_ROW_REALISTIC)
    ref["Delta"] = 0.0
    computed = dict(ref)
    computed["Delta"] = 1e-15
    _check_baseline_match(computed, ref, tol=1e-9, dataset_name="glass", alpha=0.75)
