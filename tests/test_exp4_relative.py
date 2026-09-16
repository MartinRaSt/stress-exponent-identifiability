# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-14
# License: see the LICENSE file in the repository root
"""Tests for `src/experiments/exp4_relative.py` (K9, the relative-change
computation moved from fig_temporal_pareto.py - see
documentation/2026-09-14_exp4_relative_poradi.md): small synthetic inputs,
`require_experiment_csv`/`mode_data_dir` monkeypatched directly in the
module (same pattern as tests/test_remove_rows.py - never a real
results/data/*.csv), verification of the relative-change logic and
fail-loud behavior."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.experiments import exp4_relative as er


def _make_df() -> pd.DataFrame:
    """dataset 'dsA', solver 'smacof', alpha=1.0, lam in {0.0, 0.5}, 2 seeds."""
    rows = [
        {"dataset": "dsA", "method": "lambda0.0_alpha1.0_smacof", "seed": 0, "status": "ok", "stab": 10.0, "qual": 1.0},
        {"dataset": "dsA", "method": "lambda0.0_alpha1.0_smacof", "seed": 1, "status": "ok", "stab": 12.0, "qual": 1.1},
        {"dataset": "dsA", "method": "lambda0.5_alpha1.0_smacof", "seed": 0, "status": "ok", "stab": 5.0, "qual": 1.2},
        {"dataset": "dsA", "method": "lambda0.5_alpha1.0_smacof", "seed": 1, "status": "ok", "stab": 7.0, "qual": 1.3},
    ]
    return pd.DataFrame(rows)


def test_parse_method_valid() -> None:
    assert er._parse_method("lambda0.5_alpha1.0_smacof") == (0.5, 1.0, "smacof")


def test_parse_method_invalid_format_fails_loud() -> None:
    with pytest.raises(ValueError):
        er._parse_method("not_a_valid_method_name")


def test_parse_method_dtsne() -> None:
    """Regression 2026-09-16: 'dtsne_lambda0.0' must be recognized as its own
    family (alpha=NaN sentinel, solver='dtsne'), not raise a ValueError - see
    src/experiments/exp4_temporal.py: method_name = f"dtsne_lambda{lam_dt}"."""
    lam, alpha, solver = er._parse_method("dtsne_lambda0.0")
    assert lam == pytest.approx(0.0)
    assert pd.isna(alpha)
    assert solver == "dtsne"

    lam2, alpha2, solver2 = er._parse_method("dtsne_lambda0.5")
    assert lam2 == pytest.approx(0.5)
    assert pd.isna(alpha2)
    assert solver2 == "dtsne"


def _make_mixed_df() -> pd.DataFrame:
    """Same data as _make_df() (the smacof family) PLUS a separate dtsne
    family with its own lambda=0 baseline - a regression test for
    dropna=False groupby (pandas would otherwise silently drop alpha=NaN
    from the groups)."""
    rows = [
        {"dataset": "dsA", "method": "lambda0.0_alpha1.0_smacof", "seed": 0, "status": "ok", "stab": 10.0, "qual": 1.0},
        {"dataset": "dsA", "method": "lambda0.0_alpha1.0_smacof", "seed": 1, "status": "ok", "stab": 12.0, "qual": 1.1},
        {"dataset": "dsA", "method": "lambda0.5_alpha1.0_smacof", "seed": 0, "status": "ok", "stab": 5.0, "qual": 1.2},
        {"dataset": "dsA", "method": "lambda0.5_alpha1.0_smacof", "seed": 1, "status": "ok", "stab": 7.0, "qual": 1.3},
        {"dataset": "dsA", "method": "dtsne_lambda0.0", "seed": 0, "status": "ok", "stab": 20.0, "qual": 2.0},
        {"dataset": "dsA", "method": "dtsne_lambda0.0", "seed": 1, "status": "ok", "stab": 22.0, "qual": 2.2},
        {"dataset": "dsA", "method": "dtsne_lambda0.5", "seed": 0, "status": "ok", "stab": 10.0, "qual": 3.0},
        {"dataset": "dsA", "method": "dtsne_lambda0.5", "seed": 1, "status": "ok", "stab": 12.0, "qual": 3.2},
    ]
    return pd.DataFrame(rows)


def test_compute_relative_change_handles_dtsne_family_with_own_baseline() -> None:
    """dtsne has its own lambda=0 baseline INDEPENDENT of smacof - a
    regression test for a ValueError in _parse_method and for dropna=False
    in groupby (K9 fix 2026-09-16, see the compute_exp4_relative import from
    main.py/exp4_relative.py)."""
    df = _make_mixed_df()
    relative = er._compute_relative_change(df)

    assert set(relative["solver"]) == {"smacof", "dtsne"}
    dtsne_rows = relative[relative["solver"] == "dtsne"]
    assert len(dtsne_rows) == 2
    assert dtsne_rows["alpha"].isna().all()

    dtsne0 = dtsne_rows[dtsne_rows["lam"] == 0.0].iloc[0]
    dtsne5 = dtsne_rows[dtsne_rows["lam"] == 0.5].iloc[0]
    assert dtsne0["stab_median"] == pytest.approx(21.0)  # median(20, 22)
    assert dtsne0["stab_ratio_vs_lambda0"] == pytest.approx(1.0)
    assert dtsne5["stab_median"] == pytest.approx(11.0)  # median(10, 12)
    assert dtsne5["stab_ratio_vs_lambda0"] == pytest.approx(11.0 / 21.0)
    assert dtsne5["qual_ratio_vs_lambda0"] == pytest.approx(3.1 / 2.1)

    # smacof family unchanged (same values as in test_compute_relative_change_ratios_and_pct)
    smacof_rows = relative[relative["solver"] == "smacof"]
    row0 = smacof_rows[smacof_rows["lam"] == 0.0].iloc[0]
    assert row0["stab_median"] == pytest.approx(11.0)


def test_compute_exp4_relative_writes_dtsne_rows_to_smoke_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """compute_exp4_relative over a small sample with both families writes a
    CSV containing solver='dtsne' (exactly the regression from the spec:
    'exp4_relative.csv also contains rows with solver=dtsne')."""
    df = _make_mixed_df()
    monkeypatch.setattr(er, "require_experiment_csv", lambda base_name, mode: df)
    monkeypatch.setattr(er, "mode_data_dir", lambda mode: tmp_path)

    out_path = er.compute_exp4_relative("smoke")
    written = pd.read_csv(out_path)
    assert "dtsne" in set(written["solver"])


def test_compute_relative_change_ratios_and_pct() -> None:
    """Median stab/qual over seeds + ratio/percent vs. the lambda=0 baseline
    (manually computed expected values for the synthetic data in _make_df)."""
    df = _make_df()
    relative = er._compute_relative_change(df)
    assert set(relative["lam"]) == {0.0, 0.5}

    row0 = relative[relative["lam"] == 0.0].iloc[0]
    row5 = relative[relative["lam"] == 0.5].iloc[0]

    assert row0["stab_median"] == pytest.approx(11.0)  # median(10, 12)
    assert row0["qual_median"] == pytest.approx(1.05)  # median(1.0, 1.1)
    assert row0["stab_ratio_vs_lambda0"] == pytest.approx(1.0)
    assert row0["stab_pct_change_vs_lambda0"] == pytest.approx(0.0)

    assert row5["stab_median"] == pytest.approx(6.0)  # median(5, 7)
    assert row5["qual_median"] == pytest.approx(1.25)  # median(1.2, 1.3)
    assert row5["stab_ratio_vs_lambda0"] == pytest.approx(6.0 / 11.0)
    assert row5["qual_ratio_vs_lambda0"] == pytest.approx(1.25 / 1.05)
    assert row5["qual_pct_change_vs_lambda0"] == pytest.approx((1.25 / 1.05 - 1.0) * 100.0)


def test_compute_exp4_relative_writes_csv_to_mode_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """compute_exp4_relative writes the result to the path returned by
    `mode_data_dir` (mode isolation - here monkeypatched to tmp_path so the
    test does not touch real results/data/)."""
    df = _make_df()
    monkeypatch.setattr(er, "require_experiment_csv", lambda base_name, mode: df)
    monkeypatch.setattr(er, "mode_data_dir", lambda mode: tmp_path)

    out_path = er.compute_exp4_relative("smoke")

    assert out_path == tmp_path / "exp4_relative.csv"
    assert out_path.exists()
    written = pd.read_csv(out_path)
    assert len(written) == 2
    assert set(written["lam"]) == {0.0, 0.5}


def test_compute_exp4_relative_empty_result_fails_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    """No combination has a lambda=0 baseline -> ValueError (fail-loud, no
    silent empty CSV), same as the original code in fig_temporal_pareto.py."""
    df = pd.DataFrame([
        {"dataset": "dsA", "method": "lambda0.5_alpha1.0_smacof", "seed": 0, "status": "ok", "stab": 1.0, "qual": 1.0},
    ])
    monkeypatch.setattr(er, "require_experiment_csv", lambda base_name, mode: df)
    with pytest.raises(ValueError):
        er.compute_exp4_relative("smoke")
