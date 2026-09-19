# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-19
# License: see the LICENSE file in the repository root
"""Tests for the 2026-09-19 "proportions" fix
(podklady/2026-09-19_FINAL_validace_rukopis.md-style follow-up: after the
font-size fix enlarged every label, several figures' text/legend/caption
chrome ended up dominating the canvas over the actual drawing area).

Two independent things are tested:
  1. `src.figures.fig_common.axes_area_fraction` - a pure geometry
     computation, tested against synthetic figures with a KNOWN axes
     bounding box (no generated PDFs needed, always runs).
  2. `src.figures.check_axes_area.check_all` - a regression test that every
     DAMI main-text figure fixed in this pass stays under the height cap
     (`figures.layout.main_text_max_height_frac_textheight`), analogous to
     `tests/test_check_font_sizes.py`'s font-size regression test. Skipped
     if the referenced PDFs/CSV have not been built locally (author-run
     figure builds) - this test does not fabricate missing data.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from src.common.config import get_mode_path, get_path, load_config
from src.figures.check_axes_area import check_all
from src.figures.fig_common import axes_area_fraction

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_axes_area_fraction_single_axes_matches_its_own_box() -> None:
    """A single Axes covering exactly [0.1, 0.1, 0.8, 0.8] of the figure
    (figure-fraction units, i.e. NOT including its tick labels/title, which
    live outside that box) must report an area fraction of 0.8*0.8=0.64,
    regardless of what is drawn inside it."""
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    ax.plot([0, 1], [0, 1])
    ax.set_title("some long title that would need its own vertical space")
    try:
        assert axes_area_fraction(fig) == pytest.approx(0.64, abs=1e-9)
    finally:
        plt.close(fig)


def test_axes_area_fraction_twin_axes_not_double_counted() -> None:
    """ax.twinx() shares an IDENTICAL bounding box with its host axes -
    counting both would double-count the same drawing area (see
    fig_alpha_curves.py, which uses twinx() for every panel)."""
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_axes((0.15, 0.15, 0.7, 0.7))
    ax.plot([0, 1], [0, 1])
    ax_twin = ax.twinx()
    ax_twin.plot([0, 1], [1, 0])
    try:
        assert axes_area_fraction(fig) == pytest.approx(0.7 * 0.7, abs=1e-9)
    finally:
        plt.close(fig)


def test_axes_area_fraction_two_disjoint_axes_sums_areas() -> None:
    """Two side-by-side Axes contribute their own areas independently (the
    multi-panel case, e.g. fig_graph_layouts.py) - this is what makes a
    single figure-level number equal the area-WEIGHTED average over panels
    (see the axes_area_fraction docstring)."""
    fig = plt.figure(figsize=(4, 3))
    ax1 = fig.add_axes((0.0, 0.0, 0.5, 1.0))
    ax2 = fig.add_axes((0.5, 0.0, 0.3, 1.0))
    ax1.plot([0, 1], [0, 1])
    ax2.plot([0, 1], [0, 1])
    try:
        assert axes_area_fraction(fig) == pytest.approx(0.5 + 0.3, abs=1e-9)
    finally:
        plt.close(fig)


def test_axes_area_fraction_hidden_axes_excluded() -> None:
    """`ax.set_visible(False)` (e.g. an unused panel cell in a grid, see
    fig_graph_layouts.py's `ax.axis('off')` skip branches) must not count
    toward the drawing area - it draws nothing."""
    fig = plt.figure(figsize=(4, 3))
    ax1 = fig.add_axes((0.1, 0.1, 0.5, 0.5))
    ax2 = fig.add_axes((0.6, 0.6, 0.3, 0.3))
    ax2.set_visible(False)
    try:
        assert axes_area_fraction(fig) == pytest.approx(0.5 * 0.5, abs=1e-9)
    finally:
        plt.close(fig)


# --- regression test: main-text figures stay under the height cap ---------

# DAMI main-text matplotlib figures fixed in the 2026-09-19 proportions pass
# (method_overview/study_design are SVG-sourced, no matplotlib Axes, and
# excluded here - see check_axes_area.py's SKIPPED handling for those).
_FIXED_MAIN_TEXT_FIGURES = [
    "fig_faithful_map",
    "fig_graph_layouts",
    "fig_neighbor_survival",
    "fig_neighborhood_problem",
    "fig_regime_map",
]

_REQUIRED_TEX = [
    PROJECT_ROOT / "clanek_en" / "sections" / "01_uvod.tex",
    PROJECT_ROOT / "clanek_en" / "sections" / "05_vysledky.tex",
]

pytestmark = pytest.mark.skipif(
    any(not p.is_file() for p in _REQUIRED_TEX),
    reason="clanek_en/sections/*.tex not found (unexpected repo layout).",
)


def test_main_text_figures_stay_under_the_height_cap() -> None:
    figures_dir = get_mode_path("results_figures_dir", "full")
    img_dir = get_path("results_dir").parent / "clanek" / "img"
    area_csv = figures_dir / "check_axes_area.csv"
    missing_pdfs = [name for name in _FIXED_MAIN_TEXT_FIGURES if not (img_dir / f"{name}.pdf").is_file()]
    if not area_csv.exists() or missing_pdfs:
        pytest.skip(
            f"check_axes_area.csv or figure PDF(s) not built locally (missing: {missing_pdfs}) - "
            "run the corresponding fig_*.py --full."
        )

    cfg = load_config()["figures"]["layout"]
    max_height_pt = float(cfg["main_text_max_height_frac_textheight"]) * float(cfg["dami_textheight_pt"])

    df = check_all(figures_dir, img_dir, float(cfg["min_axes_area_fraction"]), max_height_pt).set_index("figure")

    failures = []
    for name in _FIXED_MAIN_TEXT_FIGURES:
        assert name in df.index, f"{name}: not found in the axes-area report (fig_*.py not (re-)run?)."
        row = df.loc[name]
        assert row["is_main_text"], f"{name}: expected to be a DAMI main-text figure."
        if row["height_status"] == "FAIL":
            failures.append(f"{name}: height_pt={row['height_pt']:.1f} > cap {max_height_pt:.1f}pt")
    assert not failures, "Main-text height-cap regression:\n" + "\n".join(failures)
