# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-19
# License: see the LICENSE file in the repository root
"""
n_max methodology macros (reserse/2026-09-19_obsahove_nalezy_pred_podanim.md,
section 2) in `src/experiments/export_numbers.py`:
`expOneNMaxTrain`, `expOneNMaxHoldout`, `expSixNMax`, `expSevenNMax`,
`expTenNMaxHessian`, `numTrainDatasetsAtCap`, `maxNaturalNTrainBelowCap`.

Covers `_add_n_max_methodology_config_numbers` (pure config_experiments.yaml
values) and `_add_n_max_methodology_dataset_numbers` (dataset_properties.csv
counts), including the fail-loud paths (missing config key, missing
hold-out override, cap mismatch between claimed-equal blocks, a training
dataset missing from dataset_properties.csv) - none of these must fabricate
a substitute value; the group must simply be skipped with a warning, exactly
like every other group in this module (`MacroCollector.try_block`).
"""
from __future__ import annotations

import copy
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.config_experiments import resolve_experiment_config  # noqa: E402
from src.experiments.export_numbers import (  # noqa: E402
    MacroCollector,
    _add_n_max_methodology_config_numbers,
    _add_n_max_methodology_dataset_numbers,
)
from src.common.config import get_project_root  # noqa: E402


class _ListLogger:
    """Minimal stand-in for `src.common.logging_utils.get_logger` (same
    contract as the other export_numbers tests)."""

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


# ---------------------------------------------------------------------------
# config-only group (_add_n_max_methodology_config_numbers)
# ---------------------------------------------------------------------------
def test_config_macros_match_config_experiments_yaml_directly() -> None:
    """Every macro must equal a value read straight from
    config_experiments.yaml - no hardcoded number in the test either;
    this is a live re-derivation, not a copy of the expected numbers."""
    mode = "full"
    e1_cfg = resolve_experiment_config("exp1_dr_benchmark", mode)
    holdout_datasets = set(e1_cfg["datasets_holdout"])
    overrides = e1_cfg["n_max_overrides"]
    holdout_caps = {overrides[name] for name in holdout_datasets}
    assert len(holdout_caps) == 1, "test precondition: repo config must have a single hold-out cap"
    expected_holdout = next(iter(holdout_caps))

    mc = MacroCollector(_ListLogger())
    _add_n_max_methodology_config_numbers(mc, mode)

    assert int(_value_of(mc, "expOneNMaxTrain")) == int(e1_cfg["n_max"])
    assert int(_value_of(mc, "expOneNMaxHoldout")) == int(expected_holdout)
    assert int(_value_of(mc, "expSixNMax")) == int(resolve_experiment_config("exp6_alpha_curves", mode)["n_max"])
    assert int(_value_of(mc, "expSevenNMax")) == int(resolve_experiment_config("exp7_rank_weights", mode)["n_max"])
    assert int(_value_of(mc, "expTenNMaxHessian")) == int(
        resolve_experiment_config("exp10_identifiability_check", mode)["n_max_hessian"]
    )


def test_missing_holdout_override_is_fail_loud(monkeypatch) -> None:
    """A hold-out dataset without an explicit n_max override must raise
    (never silently fall back to the default n_max)."""
    real = resolve_experiment_config

    def fake(experiment_key: str, mode: str):
        cfg = copy.deepcopy(real(experiment_key, mode))
        if experiment_key == "exp1_dr_benchmark":
            first_holdout = cfg["datasets_holdout"][0]
            del cfg["n_max_overrides"][first_holdout]
        return cfg

    monkeypatch.setattr("src.experiments.export_numbers.resolve_experiment_config", fake)

    logger = _ListLogger()
    mc = MacroCollector(logger)
    _add_n_max_methodology_config_numbers(mc, "full")

    # expOneNMaxTrain does not depend on the hold-out overrides at all - it is
    # genuinely correct and stays; only the hold-out-derived macros (and
    # everything computed from them downstream) are withheld.
    assert "expOneNMaxTrain" in mc.seen
    assert "expOneNMaxHoldout" not in mc.seen
    assert "expSixNMax" not in mc.seen
    assert len(logger.warnings) == 1
    assert "n_max_overrides" in logger.warnings[0]


def test_inconsistent_holdout_caps_are_fail_loud(monkeypatch) -> None:
    """If hold-out datasets do not all share the same n_max cap, the claim
    'n_max=expOneNMaxHoldout for the hold-out datasets' would be false -
    must raise, not silently pick one value."""
    real = resolve_experiment_config

    def fake(experiment_key: str, mode: str):
        cfg = copy.deepcopy(real(experiment_key, mode))
        if experiment_key == "exp1_dr_benchmark":
            first_holdout = cfg["datasets_holdout"][0]
            cfg["n_max_overrides"][first_holdout] = cfg["n_max_overrides"][first_holdout] + 1
        return cfg

    monkeypatch.setattr("src.experiments.export_numbers.resolve_experiment_config", fake)

    logger = _ListLogger()
    mc = MacroCollector(logger)
    _add_n_max_methodology_config_numbers(mc, "full")

    assert "expOneNMaxHoldout" not in mc.seen
    assert len(logger.warnings) == 1
    assert "distinct caps" in logger.warnings[0]


def test_exp6_n_max_mismatch_with_holdout_cap_is_fail_loud(monkeypatch) -> None:
    """expSixNMax is asserted equal to expOneNMaxHoldout (both cited as the
    same n_max in the methodology paragraph) - a drift must raise."""
    real = resolve_experiment_config

    def fake(experiment_key: str, mode: str):
        cfg = copy.deepcopy(real(experiment_key, mode))
        if experiment_key == "exp6_alpha_curves":
            cfg["n_max"] = cfg["n_max"] + 1
        return cfg

    monkeypatch.setattr("src.experiments.export_numbers.resolve_experiment_config", fake)

    logger = _ListLogger()
    mc = MacroCollector(logger)
    _add_n_max_methodology_config_numbers(mc, "full")

    assert "expSixNMax" not in mc.seen
    assert len(logger.warnings) == 1
    assert "exp6_alpha_curves.n_max" in logger.warnings[0]


def test_screening_n_max_mismatch_with_holdout_cap_is_fail_loud(monkeypatch) -> None:
    """screen_regime_candidates.n_max is claimed to reuse the same cap as
    the Experiment 1 hold-out datasets (no dedicated macro, but the equality
    is checked so the claim cannot silently go stale)."""
    real = resolve_experiment_config

    def fake(experiment_key: str, mode: str):
        cfg = copy.deepcopy(real(experiment_key, mode))
        if experiment_key == "screen_regime_candidates":
            cfg["n_max"] = cfg["n_max"] + 1
        return cfg

    monkeypatch.setattr("src.experiments.export_numbers.resolve_experiment_config", fake)

    logger = _ListLogger()
    mc = MacroCollector(logger)
    _add_n_max_methodology_config_numbers(mc, "full")

    assert "expSevenNMax" not in mc.seen  # whole block aborted before reaching exp7/exp10
    assert len(logger.warnings) == 1
    assert "screen_regime_candidates.n_max" in logger.warnings[0]


def test_missing_config_key_is_fail_loud(monkeypatch) -> None:
    """A config block that no longer has the expected key must raise
    (KeyError), never default to a hardcoded value."""
    real = resolve_experiment_config

    def fake(experiment_key: str, mode: str):
        cfg = copy.deepcopy(real(experiment_key, mode))
        if experiment_key == "exp7_rank_weights":
            del cfg["n_max"]
        return cfg

    monkeypatch.setattr("src.experiments.export_numbers.resolve_experiment_config", fake)

    logger = _ListLogger()
    mc = MacroCollector(logger)
    _add_n_max_methodology_config_numbers(mc, "full")

    assert "expSevenNMax" not in mc.seen
    assert len(logger.warnings) == 1


# ---------------------------------------------------------------------------
# dataset_properties.csv group (_add_n_max_methodology_dataset_numbers)
# ---------------------------------------------------------------------------
def test_dataset_macros_match_an_independent_recomputation_on_real_data() -> None:
    """Cross-checks numTrainDatasetsAtCap/maxNaturalNTrainBelowCap on the
    real repo data/config against an independent recomputation done in the
    test (groupby/filter written from scratch, not by calling the
    production helper)."""
    data_dir = get_project_root() / "results" / "data"
    props_path = data_dir / "dataset_properties.csv"
    if not props_path.exists():
        pytest.skip(f"Missing input (not fabricating a substitute): {props_path}")

    e1_cfg = resolve_experiment_config("exp1_dr_benchmark", "full")
    n_max = int(e1_cfg["n_max"])
    core = set(e1_cfg["datasets"])

    props = pd.read_csv(props_path)
    train = props[(props["kind"] == "vector") & (props["dataset"].isin(core))]
    assert set(train["dataset"]) == core, "test precondition: dataset_properties.csv must cover every training dataset"

    expected_at_cap = int((train["n"] == n_max).sum())
    expected_max_below_cap = int(train.loc[train["n"] < n_max, "n"].max())

    mc = MacroCollector(_ListLogger())
    _add_n_max_methodology_dataset_numbers(mc, data_dir, "full")

    assert int(_value_of(mc, "numTrainDatasetsAtCap")) == expected_at_cap
    assert int(_value_of(mc, "maxNaturalNTrainBelowCap")) == expected_max_below_cap


def test_dataset_macros_missing_training_dataset_is_fail_loud(tmp_path, monkeypatch) -> None:
    """A training dataset absent from dataset_properties.csv must raise -
    the count would otherwise silently undercount."""
    real = resolve_experiment_config

    def fake(experiment_key: str, mode: str):
        cfg = copy.deepcopy(real(experiment_key, mode))
        if experiment_key == "exp1_dr_benchmark":
            cfg["datasets"] = ["ds_a", "ds_b"]
            cfg["n_max"] = 100
        return cfg

    monkeypatch.setattr("src.experiments.export_numbers.resolve_experiment_config", fake)

    props = pd.DataFrame([{"dataset": "ds_a", "kind": "vector", "n": 100}])
    props.to_csv(tmp_path / "dataset_properties.csv", index=False)

    logger = _ListLogger()
    mc = MacroCollector(logger)
    _add_n_max_methodology_dataset_numbers(mc, tmp_path, "full")

    assert "numTrainDatasetsAtCap" not in mc.seen
    assert len(logger.warnings) == 1
    assert "ds_b" in logger.warnings[0]


def test_dataset_macros_computed_correctly_on_synthetic_data(tmp_path, monkeypatch) -> None:
    """Small, fully controlled synthetic dataset_properties.csv: 2 of 3
    training datasets sit at the cap, the remaining one is below it."""
    real = resolve_experiment_config

    def fake(experiment_key: str, mode: str):
        cfg = copy.deepcopy(real(experiment_key, mode))
        if experiment_key == "exp1_dr_benchmark":
            cfg["datasets"] = ["ds_a", "ds_b", "ds_c"]
            cfg["n_max"] = 100
        return cfg

    monkeypatch.setattr("src.experiments.export_numbers.resolve_experiment_config", fake)

    props = pd.DataFrame([
        {"dataset": "ds_a", "kind": "vector", "n": 100},
        {"dataset": "ds_b", "kind": "vector", "n": 100},
        {"dataset": "ds_c", "kind": "vector", "n": 42},
        {"dataset": "ds_holdout_z", "kind": "vector", "n": 100},  # not in `datasets` (core) - must be ignored
    ])
    props.to_csv(tmp_path / "dataset_properties.csv", index=False)

    mc = MacroCollector(_ListLogger())
    _add_n_max_methodology_dataset_numbers(mc, tmp_path, "full")

    assert int(_value_of(mc, "numTrainDatasetsAtCap")) == 2
    assert int(_value_of(mc, "maxNaturalNTrainBelowCap")) == 42
