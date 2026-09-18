# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-17
# License: see the LICENSE file in the repository root
"""Tests for src/experiments/exp12_alpha_grid_extension.py
(documentation/2026-09-17_zadani_exp12_prodlouzeni_mrizky.md): the CSV
schema (must match exp6_alpha_curves_results.csv plus exactly the two new
columns), the weight_dynamic_range_log10/numerically_reliable computation,
mode isolation (smoke/quick never touch full paths), and checkpoint/resume
via `filter_already_done`.

A full end-to-end run needs a completed exp6_alpha_curves FULL run (fit
hyperparameters are always read from it) - author-run only, per project
rule. These tests instead exercise the pure/testable pieces directly."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.common import checkpoint
from src.common.checkpoint import RunKey, append_result
from src.experiments.exp12_alpha_grid_extension import (
    BASE_EXPERIMENT_NAME,
    EXP6_BASE_NAME,
    build_column_keys,
    compute_weight_dynamic_range_log10,
    is_numerically_reliable,
    _method_name_for_alpha,
)

_EXP6_RESULTS_CSV = Path(__file__).resolve().parents[1] / "results" / "data" / "exp6_alpha_curves_results.csv"

# Columns added automatically by src.common.checkpoint.append_result (from
# the RunKey) and src.experiments.exp_common.run_experiment_grid, i.e. NOT
# part of the 'column_keys' list passed to run_experiment_grid.
_AUTO_COLUMNS = {"experiment", "dataset", "method", "seed", "status", "error", "wall_time_sec"}
_NEW_COLUMNS = {"weight_dynamic_range_log10", "numerically_reliable"}


# ============================================================================
# CSV schema: must match exp6_alpha_curves_results.csv plus exactly the two
# new columns (documentation/2026-09-17_zadani_exp12_prodlouzeni_mrizky.md).
# ============================================================================


def _read_csv_header(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Reference CSV not found: {path}")
    with open(path, "r", encoding="utf-8", newline="") as f:
        header = next(csv.reader(f), None)
    if header is None:
        raise ValueError(f"{path} has no header row.")
    return header


def test_schema_matches_exp6_plus_exactly_two_new_columns() -> None:
    """Every non-auto column of exp6_alpha_curves_results.csv must also be a
    column of exp12 (so rows can be concatenated into one alpha curve), and
    exp12 must add EXACTLY the two documented columns - nothing more,
    nothing less."""
    e6_header = _read_csv_header(_EXP6_RESULTS_CSV)
    e6_own_columns = set(e6_header) - _AUTO_COLUMNS

    e12_columns, eval_kwargs = build_column_keys(n_components=2)
    e12_columns_set = set(e12_columns)

    missing_from_e12 = e6_own_columns - e12_columns_set
    assert not missing_from_e12, f"exp12 is missing exp6 columns: {sorted(missing_from_e12)}"

    extra_in_e12 = e12_columns_set - e6_own_columns
    assert extra_in_e12 == _NEW_COLUMNS, f"exp12 must add exactly {_NEW_COLUMNS}, got extra={sorted(extra_in_e12)}"

    assert eval_kwargs.get("extended") is True


def test_method_name_matches_exp6_convention() -> None:
    """The 'method' column value MUST match exp6_alpha_curves.py's own
    convention (f"alpha{alpha}") - otherwise E6/E12 rows of the same
    (dataset, alpha, seed) would not line up when concatenated."""
    assert _method_name_for_alpha(3.25) == "alpha3.25"
    assert _method_name_for_alpha(6.0) == "alpha6.0"


# ============================================================================
# weight_dynamic_range_log10 / numerically_reliable
# ============================================================================


def test_weight_dynamic_range_log10_matches_manual_formula() -> None:
    # D_max=9, D_min=1 (off-diagonal), eps_D=1.0 -> ratio=(9+1)/(1+1)=5
    D = np.array([
        [0.0, 1.0, 9.0],
        [1.0, 0.0, 4.0],
        [9.0, 4.0, 0.0],
    ])
    eps_D = 1.0
    alpha = 3.0
    expected = alpha * np.log10((9.0 + eps_D) / (1.0 + eps_D))
    got = compute_weight_dynamic_range_log10(D, alpha, eps_D)
    assert got == pytest.approx(expected)
    assert got == pytest.approx(3.0 * np.log10(5.0))


def test_weight_dynamic_range_log10_ignores_the_zero_diagonal() -> None:
    """The diagonal is always 0 and must never be picked up as D_min."""
    D = np.array([
        [0.0, 2.0, 8.0],
        [2.0, 0.0, 5.0],
        [8.0, 5.0, 0.0],
    ])
    got = compute_weight_dynamic_range_log10(D, alpha=1.0, eps_D=0.5)
    expected = 1.0 * np.log10((8.0 + 0.5) / (2.0 + 0.5))
    assert got == pytest.approx(expected)


def test_weight_dynamic_range_log10_scales_linearly_with_alpha() -> None:
    D = np.array([[0.0, 3.0], [3.0, 0.0]])
    v1 = compute_weight_dynamic_range_log10(D, alpha=1.0, eps_D=0.1)
    v4 = compute_weight_dynamic_range_log10(D, alpha=4.0, eps_D=0.1)
    assert v4 == pytest.approx(4.0 * v1)


def test_weight_dynamic_range_log10_rejects_non_positive_eps_d() -> None:
    D = np.array([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="eps_D must be positive"):
        compute_weight_dynamic_range_log10(D, alpha=1.0, eps_D=0.0)


def test_weight_dynamic_range_log10_rejects_non_square_D() -> None:
    with pytest.raises(ValueError, match="square"):
        compute_weight_dynamic_range_log10(np.zeros((2, 3)), alpha=1.0, eps_D=1.0)


def test_is_numerically_reliable_threshold_is_inclusive() -> None:
    assert is_numerically_reliable(14.0, reliability_log10_max=14.0) is True
    assert is_numerically_reliable(14.0000001, reliability_log10_max=14.0) is False
    assert is_numerically_reliable(5.0, reliability_log10_max=14.0) is True


# ============================================================================
# mode isolation (smoke/quick outputs never collide with full paths)
# ============================================================================


def test_experiment_name_isolation_across_modes() -> None:
    from src.experiments.exp_common import resolve_experiment_name

    full_name = resolve_experiment_name(BASE_EXPERIMENT_NAME, "full")
    quick_name = resolve_experiment_name(BASE_EXPERIMENT_NAME, "quick")
    smoke_name = resolve_experiment_name(BASE_EXPERIMENT_NAME, "smoke")

    assert full_name == "exp12_alpha_grid_extension"
    assert quick_name == "quick/exp12_alpha_grid_extension"
    assert smoke_name == "smoke/exp12_alpha_grid_extension"
    assert len({full_name, quick_name, smoke_name}) == 3


def test_smoke_config_restricts_datasets_and_alpha_grid_without_touching_full() -> None:
    from src.experiments.config_experiments import resolve_experiment_config

    full_cfg = resolve_experiment_config(BASE_EXPERIMENT_NAME, "full")
    smoke_cfg = resolve_experiment_config(BASE_EXPERIMENT_NAME, "smoke")

    assert len(full_cfg["datasets"]) == 18
    assert smoke_cfg["datasets"] == ["iris"]
    assert smoke_cfg["alpha_grid"] == [3.25, 3.5]
    assert max(float(a) for a in full_cfg["alpha_grid"]) == pytest.approx(6.0)
    # reliability_log10_max is not overridden by 'smoke' - must stay identical.
    assert smoke_cfg["reliability_log10_max"] == full_cfg["reliability_log10_max"]


def test_exp6_base_name_constant_points_at_exp6_alpha_curves() -> None:
    assert EXP6_BASE_NAME == "exp6_alpha_curves"


# ============================================================================
# checkpoint / resume
# ============================================================================


@pytest.fixture()
def isolated_results_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirects `checkpoint.get_path('results_data_dir'/'embeddings_dir')`
    to a temporary directory so the test never touches the real results/data
    (same pattern as tests/test_checkpoint.py)."""
    data_dir = tmp_path / "results_data"
    emb_dir = tmp_path / "results_data" / "embeddings"

    def _fake_get_path(key: str) -> Path:
        if key == "results_data_dir":
            return data_dir
        if key == "embeddings_dir":
            return emb_dir
        raise KeyError(key)

    monkeypatch.setattr(checkpoint, "get_path", _fake_get_path)
    return data_dir


def test_filter_already_done_skips_previously_completed_runs(isolated_results_dir: Path) -> None:
    """A resumed run must skip a (dataset, method, seed) already present in
    the checkpoint CSV and only compute the remaining tasks - the exact
    mechanism exp12's main() relies on for resume."""
    from src.experiments.exp_common import filter_already_done

    experiment_name = "smoke/" + BASE_EXPERIMENT_NAME
    method_name = _method_name_for_alpha(3.25)

    # Simulate one already-completed run from a previous (interrupted) invocation.
    append_result(
        RunKey(experiment_name, "iris", method_name, 0),
        {"status": "ok", "error": "", "wall_time_sec": 0.1, "auc_rnx": 0.5},
    )

    tasks = [
        {"dataset_name": "iris", "method_name": method_name, "seed": 0},
        {"dataset_name": "iris", "method_name": method_name, "seed": 1},
    ]
    todo, n_done = filter_already_done(experiment_name, tasks)

    assert n_done == 1
    assert todo == [{"dataset_name": "iris", "method_name": method_name, "seed": 1}]


def test_filter_already_done_is_idempotent_on_second_call(isolated_results_dir: Path) -> None:
    """Calling filter_already_done twice with the SAME already-written CSV
    (as a second smoke invocation would) must yield the same result both
    times - no partial/duplicate re-scheduling."""
    from src.experiments.exp_common import filter_already_done

    experiment_name = "smoke/" + BASE_EXPERIMENT_NAME
    method_name = _method_name_for_alpha(3.5)
    append_result(
        RunKey(experiment_name, "iris", method_name, 0),
        {"status": "ok", "error": "", "wall_time_sec": 0.1, "auc_rnx": 0.4},
    )

    tasks = [{"dataset_name": "iris", "method_name": method_name, "seed": 0}]
    todo1, n_done1 = filter_already_done(experiment_name, tasks)
    todo2, n_done2 = filter_already_done(experiment_name, tasks)

    assert todo1 == todo2 == []
    assert n_done1 == n_done2 == 1
