# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-18
# License: see the LICENSE file in the repository root
"""
Review round 2 fixes (2026-09-18) in `src/experiments/export_numbers.py`:

1. `_add_exp8_prop2_per_dataset_correlation` - the pooled
   spearmanPredictedFactorVsDeclineRho/P macros infer across 1885 rows that
   come from only 32 datasets (pseudo-replication); the unit of analysis
   must be the dataset, so this group computes the Spearman correlation
   WITHIN each dataset and summarizes it ACROSS datasets.
2. `_add_exp10_negative_delta_survival` - checks whether rows with a
   NEGATIVE Delta/Delta_prime survive the exp14-measured convergence
   filter (|Delta|>=tau AND |Delta_prime|>=tau), since the article claimed
   none do.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.export_numbers import (  # noqa: E402
    MacroCollector,
    _add_exp8_prop2_per_dataset_correlation,
    _add_exp10_negative_delta_survival,
)


class _ListLogger:
    """Minimal stand-in for `src.common.logging_utils.get_logger` recording
    warnings, so a test can assert a group was skipped without a crash (the
    documented `try_block` resilience contract)."""

    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, msg: str, *args) -> None:
        self.warnings.append(msg % args if args else msg)

    def info(self, *a, **k) -> None:
        pass


def _value_of(mc: MacroCollector, name: str) -> str:
    for line in mc.lines:
        m = re.match(rf"\\newcommand\{{\\{name}\}}\{{(.*?)\}}\s*%", line)
        if m:
            value = m.group(1)
            num_match = re.match(r"^\\num\{(.*)\}$", value)
            return num_match.group(1) if num_match else value
    raise AssertionError(f"Macro '{name}' was not generated. Lines:\n" + "".join(mc.lines))


def test_exp8_per_dataset_correlation_uses_dataset_as_unit_of_analysis(tmp_path) -> None:
    """Two datasets contribute a WITHIN-dataset Spearman each (one perfectly
    anti-monotone, one perfectly monotone); a third dataset has only 2
    distinct values of r_near_ratio and must be EXCLUDED (undefined
    correlation), not fabricated. A failed row and a p2_holds==False row
    must also be ignored."""
    rows = []
    # ds_up: perfect positive within-dataset correlation
    for i, alpha in enumerate([0.25, 0.5, 0.75, 1.0]):
        rows.append({"dataset": "ds_up", "alpha": alpha, "seed": 0, "status": "ok", "p2_holds": True,
                     "predicted_factor_r_eps_alpha": 0.1 * (i + 1), "r_near_ratio": 0.2 * (i + 1)})
    # ds_down: perfect negative within-dataset correlation
    for i, alpha in enumerate([0.25, 0.5, 0.75, 1.0]):
        rows.append({"dataset": "ds_down", "alpha": alpha, "seed": 0, "status": "ok", "p2_holds": True,
                     "predicted_factor_r_eps_alpha": 0.1 * (i + 1), "r_near_ratio": 1.0 - 0.2 * (i + 1)})
    # ds_constant: r_near_ratio takes only 2 distinct values -> excluded (undefined rho)
    for i, alpha in enumerate([0.25, 0.5, 0.75, 1.0]):
        rows.append({"dataset": "ds_constant", "alpha": alpha, "seed": 0, "status": "ok", "p2_holds": True,
                     "predicted_factor_r_eps_alpha": 0.1 * (i + 1), "r_near_ratio": 1.0 if i < 2 else 2.0})
    # a failed row and a p2_holds==False row in an otherwise valid dataset - must not leak in
    rows.append({"dataset": "ds_up", "alpha": 1.25, "seed": 0, "status": "error",
                 "predicted_factor_r_eps_alpha": 99.0, "r_near_ratio": 99.0, "p2_holds": False})
    rows.append({"dataset": "ds_up", "alpha": 1.5, "seed": 0, "status": "ok", "p2_holds": False,
                 "predicted_factor_r_eps_alpha": 99.0, "r_near_ratio": 99.0})

    df = pd.DataFrame(rows)
    data_dir = tmp_path
    df.to_csv(data_dir / "exp8_prop2_check_results.csv", index=False)

    mc = MacroCollector(_ListLogger())
    _add_exp8_prop2_per_dataset_correlation(mc, data_dir)

    # only ds_up (rho=+1) and ds_down (rho=-1) are used; median of {1,-1} = 0
    assert float(_value_of(mc, "medExpEightPerDatasetDeclineRho")) == pytest.approx(0.0, abs=1e-9)
    assert _value_of(mc, "numExpEightPerDatasetPositiveDeclineRho") == "1/2"
    # pandas linear-interpolation quantiles of the 2-element series {-1, 1}
    q_low = float(_value_of(mc, "iqrLowExpEightPerDatasetDeclineRho"))
    q_high = float(_value_of(mc, "iqrHighExpEightPerDatasetDeclineRho"))
    assert q_low == pytest.approx(-0.5, abs=1e-2)
    assert q_high == pytest.approx(0.5, abs=1e-2)
    # the Wilcoxon p-value must exist and be a valid probability
    p_value = float(_value_of(mc, "wilcoxonExpEightPerDatasetDeclineRhoP"))
    assert 0.0 <= p_value <= 1.0


def test_exp8_per_dataset_correlation_matches_independent_pandas_computation(tmp_path) -> None:
    """Cross-checks the group against an independently computed reference
    (groupby + scipy.stats.spearmanr/wilcoxon done directly in the test,
    not by re-calling the production helper) on a less trivial dataset mix."""
    rng = np.random.default_rng(0)
    rows = []
    datasets = [f"ds{i}" for i in range(6)]
    for ds in datasets:
        n = 8
        x = np.sort(rng.uniform(0, 1, size=n))
        noise = rng.normal(scale=0.05, size=n)
        y = x + noise  # positively associated but not perfectly
        for alpha, xv, yv in zip(np.linspace(0.25, 3.0, n), x, y):
            rows.append({"dataset": ds, "alpha": alpha, "seed": 0, "status": "ok", "p2_holds": True,
                         "predicted_factor_r_eps_alpha": xv, "r_near_ratio": yv})
    df = pd.DataFrame(rows)
    data_dir = tmp_path
    df.to_csv(data_dir / "exp8_prop2_check_results.csv", index=False)

    mc = MacroCollector(_ListLogger())
    _add_exp8_prop2_per_dataset_correlation(mc, data_dir)

    expected = {}
    for ds, g in df.groupby("dataset"):
        rho, _p = spearmanr(g["predicted_factor_r_eps_alpha"], g["r_near_ratio"])
        expected[ds] = rho
    expected_series = pd.Series(expected)
    expected_stat = wilcoxon(expected_series.to_numpy())

    assert float(_value_of(mc, "medExpEightPerDatasetDeclineRho")) == pytest.approx(expected_series.median(), abs=1e-2)
    assert float(_value_of(mc, "iqrLowExpEightPerDatasetDeclineRho")) == pytest.approx(expected_series.quantile(0.25), abs=1e-2)
    assert float(_value_of(mc, "iqrHighExpEightPerDatasetDeclineRho")) == pytest.approx(expected_series.quantile(0.75), abs=1e-2)
    assert float(_value_of(mc, "wilcoxonExpEightPerDatasetDeclineRhoP")) == pytest.approx(expected_stat.pvalue, rel=1e-1, abs=1e-3)


def test_exp8_per_dataset_correlation_is_skipped_not_fabricated_when_column_missing(tmp_path) -> None:
    df = pd.DataFrame([{"dataset": "a", "alpha": 0.25, "status": "ok", "p2_holds": True}])
    data_dir = tmp_path
    df.to_csv(data_dir / "exp8_prop2_check_results.csv", index=False)

    logger = _ListLogger()
    mc = MacroCollector(logger)
    _add_exp8_prop2_per_dataset_correlation(mc, data_dir)

    assert "medExpEightPerDatasetDeclineRho" not in mc.seen
    assert len(logger.warnings) == 1


def _exp10_row(dataset: str, alpha: float, delta: float, delta_prime: float) -> dict:
    return {"dataset": dataset, "alpha": alpha, "status": "ok", "Delta": delta, "Delta_prime": delta_prime}


def test_exp10_negative_delta_survival_matches_the_exp14_filter_rule(tmp_path) -> None:
    """Two rows have a negative Delta: one with |Delta| and |Delta_prime|
    both >= tau (survives), one with |Delta| < tau (filtered out). One row
    has a negative Delta_prime with |Delta_prime| < tau (filtered out)."""
    tau = 0.01
    e10 = pd.DataFrame([
        _exp10_row("keep_ds", 0.5, delta=-0.02, delta_prime=0.05),   # |Delta|>=tau, |Delta'|>=tau -> survives
        _exp10_row("drop_ds", 0.5, delta=-0.005, delta_prime=0.05),  # |Delta|<tau -> filtered out
        _exp10_row("prime_ds", 0.5, delta=0.02, delta_prime=-0.001),  # |Delta'|<tau -> filtered out
        _exp10_row("alpha_zero_ds", 0.0, delta=-0.5, delta_prime=-0.5),  # alpha==0 excluded entirely
        {"dataset": "err_ds", "alpha": 0.5, "status": "error", "Delta": -0.5, "Delta_prime": -0.5},
    ])
    e14 = pd.DataFrame([
        {"tau": 0.001, "tau_source": "config"},
        {"tau": tau, "tau_source": "measured_from_exp11"},
    ])
    data_dir = tmp_path
    e10.to_csv(data_dir / "exp10_identifiability_check_results.csv", index=False)
    e14.to_csv(data_dir / "exp14_convergence_robustness_results.csv", index=False)

    mc = MacroCollector(_ListLogger())
    _add_exp10_negative_delta_survival(mc, data_dir)

    assert _value_of(mc, "numExpNineNegativeDeltaRowsTotal") == "2"
    assert _value_of(mc, "numExpNineNegativeDeltaRowsSurviving") == "1"
    assert _value_of(mc, "negativeDeltaSurvivingDatasetsList") == "keep\\_ds"

    assert _value_of(mc, "numExpNineNegativeDeltaPrimeRowsTotal") == "1"
    assert _value_of(mc, "numExpNineNegativeDeltaPrimeRowsSurviving") == "0"
    assert _value_of(mc, "negativeDeltaPrimeSurvivingDatasetsList") == ""


def test_exp10_negative_delta_survival_is_skipped_not_fabricated_when_measured_tau_missing(tmp_path) -> None:
    e10 = pd.DataFrame([_exp10_row("a", 0.5, delta=-0.02, delta_prime=0.05)])
    e14 = pd.DataFrame([{"tau": 0.001, "tau_source": "config"}])  # no 'measured_from_exp11' row
    data_dir = tmp_path
    e10.to_csv(data_dir / "exp10_identifiability_check_results.csv", index=False)
    e14.to_csv(data_dir / "exp14_convergence_robustness_results.csv", index=False)

    logger = _ListLogger()
    mc = MacroCollector(logger)
    _add_exp10_negative_delta_survival(mc, data_dir)

    assert "numExpNineNegativeDeltaRowsTotal" not in mc.seen
    assert len(logger.warnings) == 1
