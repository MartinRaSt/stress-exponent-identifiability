# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-13
# License: see the LICENSE file in the repository root
"""Separation of smoke/quick from full outputs (the author's rule
2026-09-13, ~/.claude/CLAUDE.md): mode-aware paths from a single shared
function (src/common/config.py::get_mode_path and friends) and an
integration test that `src.main.run("smoke", no_figures=True)` does NOT
change the mtime of any file in results/tables/*.{csv,tex},
clanek/generated/, results/figures/*.pdf, and clanek/img/*.pdf (a
before/after snapshot). Also converting factor levels to macro names
(export_numbers)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from src.common.config import MODES, get_generated_dir, get_mode_path, get_path, get_project_root, get_tables_dir
from src.experiments.export_numbers import num_to_words

ROOT = get_project_root()


def test_get_mode_path_full_is_root() -> None:
    assert get_mode_path("results_tables_dir", "full") == get_path("results_tables_dir")
    assert get_mode_path("results_data_dir", "full") == get_path("results_data_dir")


@pytest.mark.parametrize("mode", [m for m in MODES if m != "full"])
def test_get_mode_path_non_full_is_subdir(mode: str) -> None:
    for key in ("results_tables_dir", "results_data_dir", "results_figures_dir", "results_logs_dir"):
        assert get_mode_path(key, mode) == get_path(key) / mode


def test_get_mode_path_unknown_mode_fails_loud() -> None:
    with pytest.raises(ValueError):
        get_mode_path("results_tables_dir", "production")


def test_generated_dir_never_in_article_for_test_modes() -> None:
    article = get_path("article_generated_dir")
    assert get_generated_dir("full") == article
    for mode in ("quick", "smoke"):
        out = get_generated_dir(mode)
        assert out == get_tables_dir(mode)
        assert article not in out.parents and out != article


def test_num_to_words() -> None:
    assert num_to_words(0) == "Zero"
    assert num_to_words(0.25) == "ZeroPointTwoFive"
    assert num_to_words("1.0") == "One"
    assert num_to_words(3) == "Three"
    assert num_to_words(10) == "Ten"
    assert num_to_words(1.25) == "OnePointTwoFive"
    for x in (0.25, 1.25, 10, 0.3):
        assert num_to_words(x).isalpha()


def _snapshot() -> dict[str, int]:
    files: list[Path] = []
    tables = get_path("results_tables_dir")
    files += list(tables.glob("*.csv")) + list(tables.glob("*.tex"))
    files += [p for p in get_path("article_generated_dir").glob("*") if p.is_file()]
    figures = get_path("results_figures_dir")
    files += list(figures.glob("*.pdf")) + list(figures.glob("*.csv"))
    files += list((ROOT / "clanek" / "img").glob("*.pdf"))
    return {str(p.relative_to(ROOT)): p.stat().st_mtime_ns for p in files}


def test_smoke_main_does_not_touch_full_outputs() -> None:
    """Integration: main in smoke mode (without figures - fast) writes
    EXCLUSIVELY to results/tables/smoke/ (incl. numbers.tex); the full
    tables, clanek/generated, and results/figures remain unchanged (both
    mtime and the set of files)."""
    smoke_e1 = get_mode_path("results_data_dir", "smoke") / "exp1_dr_benchmark_results.csv"
    if not smoke_e1.exists():
        pytest.skip(f"missing smoke data {smoke_e1} (run src\\run_exp1_dr_benchmark.bat smoke)")
    from src.main import run

    before = _snapshot()
    run("smoke", no_figures=True)
    after = _snapshot()

    assert set(after) == set(before), f"new/deleted files outside the smoke subdirectory: {set(after) ^ set(before)}"
    changed = sorted(k for k in before if before[k] != after[k])
    assert not changed, f"the smoke run changed full outputs: {changed}"

    smoke_tables = get_tables_dir("smoke")
    assert (smoke_tables / "numbers.tex").exists()
    assert (smoke_tables / "exp1_main_table.csv").exists()


# ---------------------------------------------------------------------------
# Q1 step 2 (reserse/2026-09-14_specifikace_rozsireni_q1.md, A.9 item 13) -
# the new scripts screen_regime_candidates.py / power_analysis_regime.py
# must not change any file outside their mode subdirectory.
# ---------------------------------------------------------------------------

def _snapshot_data_and_tables() -> dict[str, int]:
    files: list[Path] = []
    files += list(get_path("results_data_dir").glob("*.csv"))
    files += list(get_path("results_data_dir").glob("*.json"))
    files += list(get_path("results_data_dir").glob("*_DONE.txt"))
    files += list(get_path("results_tables_dir").glob("*.csv"))
    files += list(get_path("results_tables_dir").glob("*.tex"))
    files += [p for p in get_path("article_generated_dir").glob("*") if p.is_file()]
    return {str(p.relative_to(ROOT)): p.stat().st_mtime_ns for p in files}


def test_screen_regime_candidates_smoke_does_not_touch_full_outputs() -> None:
    from src.experiments.screen_regime_candidates import main as screen_main

    before = _snapshot_data_and_tables()
    old_argv = sys.argv
    sys.argv = ["screen_regime_candidates", "--smoke"]
    try:
        screen_main()
    finally:
        sys.argv = old_argv
    after = _snapshot_data_and_tables()
    assert set(after) == set(before), f"screen_regime_candidates --smoke changed the file set outside smoke: {set(after) ^ set(before)}"
    changed = sorted(k for k in before if before[k] != after[k])
    assert not changed, f"screen_regime_candidates --smoke changed full outputs: {changed}"


def test_power_analysis_regime_smoke_does_not_touch_full_outputs() -> None:
    from src.experiments.power_analysis_regime import main as power_main

    before = _snapshot_data_and_tables()
    old_argv = sys.argv
    sys.argv = ["power_analysis_regime", "--smoke"]
    try:
        power_main()
    finally:
        sys.argv = old_argv
    after = _snapshot_data_and_tables()
    assert set(after) == set(before), f"power_analysis_regime --smoke changed the file set outside smoke: {set(after) ^ set(before)}"
    changed = sorted(k for k in before if before[k] != after[k])
    assert not changed, f"power_analysis_regime --smoke changed full outputs: {changed}"
