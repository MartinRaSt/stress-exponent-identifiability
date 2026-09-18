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
Regression test for the 2026-09-18 fact-check fix in 04_experimenty.tex:
the article claimed rho_NN (nn_ratio_k1) correlates with the grid-optimum
alpha more strongly than every other dataset property, which does not hold
on the actual data (the ambient dimension `d` correlates more strongly).
Covers `_best_alpha_full_grid_from_exp6` and
`_add_regime_map_property_spearman` (src/experiments/export_numbers.py).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments import export_numbers as en  # noqa: E402
from src.experiments.export_numbers import (  # noqa: E402
    MacroCollector,
    _add_regime_map_property_spearman,
    _best_alpha_full_grid_from_exp6,
)


class _ListLogger:
    """Minimal stand-in for `src.common.logging_utils.get_logger` that just
    records warnings, so a test can assert a group was skipped without a
    crash (the documented `try_block` resilience contract)."""

    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, msg: str, *args) -> None:
        self.warnings.append(msg % args if args else msg)

    def info(self, *a, **k) -> None:
        pass


def _value_of(mc: MacroCollector, name: str) -> str:
    """Extracts a macro's raw value, unwrapping siunitx's `\\num{...}` if
    `_wrap_decimal` applied it (decimal values only, see `export_numbers.py`)."""
    for line in mc.lines:
        m = re.match(rf"\\newcommand\{{\\{name}\}}\{{(.*?)\}}\s*%", line)
        if m:
            value = m.group(1)
            num_match = re.match(r"^\\num\{(.*)\}$", value)
            return num_match.group(1) if num_match else value
    raise AssertionError(f"Macro '{name}' was not generated. Lines:\n" + "".join(mc.lines))


def test_best_alpha_full_grid_from_exp6_matches_median_argmax_and_skips_failed_rows() -> None:
    e6 = pd.DataFrame(
        [
            {"dataset": "a", "alpha": 0.0, "auc_rnx": 0.9, "status": "ok"},
            {"dataset": "a", "alpha": 1.0, "auc_rnx": 0.2, "status": "ok"},
            {"dataset": "b", "alpha": 0.0, "auc_rnx": 0.1, "status": "ok"},
            {"dataset": "b", "alpha": 1.0, "auc_rnx": 0.8, "status": "ok"},
            # a failed row at an otherwise-winning alpha must not win
            {"dataset": "b", "alpha": 2.0, "auc_rnx": 0.99, "status": "error"},
        ]
    )
    best = _best_alpha_full_grid_from_exp6(e6)
    assert best["a"] == pytest.approx(0.0)
    assert best["b"] == pytest.approx(1.0)


def test_regime_map_property_spearman_excludes_rho_nn_from_top_competitor(tmp_path, monkeypatch) -> None:
    """8 datasets with a target that is a perfect increasing sequence, and
    3 candidate properties by design: nn_ratio_k1 is a PERFECT predictor
    (rho=-1), 'd' is the second-strongest (rho about -0.976), 'weak' is
    a poor predictor. The function must (a) not re-emit rho_NN's own
    correlation, (b) report 'd' specifically, and (c) name 'd' (not
    nn_ratio_k1) as the strongest COMPETING property, since nn_ratio_k1 is
    excluded from that comparison by construction."""
    datasets = [f"ds{i}" for i in range(8)]
    target_alpha = list(range(8))  # 0..7, one distinct alpha per dataset
    nn_ratio_k1 = [7, 6, 5, 4, 3, 2, 1, 0]  # perfectly anti-monotone -> rho=-1
    d_col = [7, 6, 5, 4, 3, 1, 2, 0]  # one swap -> rho about -0.976 (2nd strongest)
    weak_col = [3, 0, 6, 1, 4, 7, 2, 5]  # weak relationship

    e6_rows = []
    for ds, alpha in zip(datasets, target_alpha):
        e6_rows.append({"dataset": ds, "alpha": float(alpha), "auc_rnx": 0.9, "status": "ok"})
        other_alpha = (alpha + 1) % 8
        e6_rows.append({"dataset": ds, "alpha": float(other_alpha), "auc_rnx": 0.1, "status": "ok"})
    e6 = pd.DataFrame(e6_rows)

    props = pd.DataFrame(
        {
            "dataset": datasets,
            "kind": ["vector"] * len(datasets),
            "nn_ratio_k1": nn_ratio_k1,
            "d": d_col,
            "weak": weak_col,
        }
    )

    data_dir = tmp_path / "data"
    tables_dir = tmp_path / "tables"
    data_dir.mkdir()
    props.to_csv(data_dir / "dataset_properties.csv", index=False)
    e6.to_csv(data_dir / "exp6_alpha_curves_results.csv", index=False)

    monkeypatch.setattr(
        en,
        "load_experiments_config",
        lambda: {"fig_regime_map": {"property_correlation_columns": ["nn_ratio_k1", "d", "weak"]}},
    )

    mc = MacroCollector(_ListLogger())
    _add_regime_map_property_spearman(mc, data_dir, tables_dir)

    out_csv = tables_dir / "regime_map_property_spearman.csv"
    assert out_csv.exists()
    table = pd.read_csv(out_csv).set_index("property")
    assert table.loc["nn_ratio_k1", "rho"] == pytest.approx(-1.0, abs=1e-9)
    assert table.loc["d", "rho"] == pytest.approx(-0.9761904761904763, abs=1e-9)
    assert int(table.loc["weak", "n"]) == 8

    assert float(_value_of(mc, "spearmanRegimeMapBestAlphaDimRho")) == pytest.approx(-0.976, abs=1e-3)
    assert _value_of(mc, "spearmanRegimeMapBestAlphaTopName") == "d"
    assert float(_value_of(mc, "spearmanRegimeMapBestAlphaTopRho")) == pytest.approx(-0.976, abs=1e-3)
    # rho_NN's own correlation is NOT re-exported by this function (already
    # covered elsewhere by spearmanRegimeMapBestAlphaRho, per the docstring).
    assert "spearmanRegimeMapBestAlphaRho" not in mc.seen


def test_regime_map_property_spearman_is_skipped_not_fabricated_when_a_property_is_missing(tmp_path, monkeypatch) -> None:
    """If a configured property column is absent from dataset_properties.csv,
    the whole group must be SKIPPED with a warning (try_block contract) -
    never silently filled with a placeholder value."""
    datasets = [f"ds{i}" for i in range(4)]
    props = pd.DataFrame({"dataset": datasets, "kind": ["vector"] * 4, "nn_ratio_k1": [1, 2, 3, 4]})
    e6 = pd.DataFrame(
        [{"dataset": ds, "alpha": float(i), "auc_rnx": 0.9, "status": "ok"} for i, ds in enumerate(datasets)]
    )

    data_dir = tmp_path / "data"
    tables_dir = tmp_path / "tables"
    data_dir.mkdir()
    props.to_csv(data_dir / "dataset_properties.csv", index=False)
    e6.to_csv(data_dir / "exp6_alpha_curves_results.csv", index=False)

    monkeypatch.setattr(
        en,
        "load_experiments_config",
        lambda: {"fig_regime_map": {"property_correlation_columns": ["nn_ratio_k1", "d"]}},  # 'd' is missing from props
    )

    logger = _ListLogger()
    mc = MacroCollector(logger)
    _add_regime_map_property_spearman(mc, data_dir, tables_dir)

    assert "spearmanRegimeMapBestAlphaDimRho" not in mc.seen
    assert not (tables_dir / "regime_map_property_spearman.csv").exists()
    assert len(logger.warnings) == 1
