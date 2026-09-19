# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-19
# License: see the LICENSE file in the repository root
"""Regression test for the 2026-09-19 font-size fix
(podklady/2026-09-19_FINAL_validace_rukopis.md section V2): the DAMI
main-text figures below must never again regress below the Springer Nature
artwork floor once printed at \\textwidth (372pt) - see
src/figures/check_font_sizes.py for the measurement method.

Scope: the figures actually fixed across the two 2026-09-19 passes -
DAMI main text (`_FIXED_DAMI_FIGURES`) and, in a second pass the same day,
the eight figures embedded ONLY in the elsarticle supplement
(`_FIXED_SUPPLEMENT_FIGURES`: fig_alpha_curves_all_p1/p2,
fig_alpha_gain_by_regime, fig_metric_correlations, fig_pareto_front,
fig_runtime_scaling, fig_sgd_convergence, fig_temporal_pareto - drawn at
WIDTH_SUPPLEMENT_FULL_IN/WIDTH_SUPPLEMENT_THREEQ_IN, see
src/figures/fig_common.py). The method_overview.svg schematic is still
KNOWN to be below the floor - it needs a genuine content/layout redesign,
not a font bump (see its file header comment; owned separately by the
graphic-designer agent) - and is deliberately NOT asserted here, so this
test does not turn red for a pre-existing, tracked, out-of-scope issue.
Skipped if the referenced PDFs have not been built locally (author-run
figure/article builds)."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.figures.check_font_sizes import check_all
from src.common.config import load_config, get_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = get_path("results_dir").parent / "clanek" / "img"

# Figures fixed in the 2026-09-19 DAMI main-text pass.
_FIXED_DAMI_FIGURES = [
    "fig_neighborhood_problem",
    "fig_regime_map",
    "fig_faithful_map",
    "fig_neighbor_survival",
    "fig_graph_layouts",
    "fig_discovery_task",
    "fig_cd_diagram_exp1_dr_benchmark_main_stress_scale_invariant",
    "study_design",
]

# Figures fixed in the 2026-09-19 elsarticle-supplement-only pass (see the
# module docstring above).
_FIXED_SUPPLEMENT_FIGURES = [
    "fig_alpha_curves_all_p1",
    "fig_alpha_curves_all_p2",
    "fig_alpha_gain_by_regime",
    "fig_metric_correlations",
    "fig_pareto_front",
    "fig_runtime_scaling",
    "fig_sgd_convergence",
    "fig_temporal_pareto",
]

_FIXED_FIGURES = _FIXED_DAMI_FIGURES + _FIXED_SUPPLEMENT_FIGURES

_REQUIRED_TEX = [
    PROJECT_ROOT / "clanek_en" / "sections" / "01_uvod.tex",
    PROJECT_ROOT / "clanek_en" / "sections" / "04_experimenty.tex",
    PROJECT_ROOT / "clanek_en" / "sections" / "05_vysledky.tex",
    PROJECT_ROOT / "clanek_en" / "supplement" / "sections" / "s2_alpha_curves.tex",
    PROJECT_ROOT / "clanek_en" / "supplement" / "sections" / "s3_negative_results.tex",
    PROJECT_ROOT / "clanek_en" / "supplement" / "sections" / "s6_full_tables.tex",
    PROJECT_ROOT / "clanek_en" / "supplement" / "sections" / "s_solvers.tex",
]


def _missing_pdfs() -> list[str]:
    return [name for name in _FIXED_FIGURES if not (IMG_DIR / f"{name}.pdf").is_file()]


pytestmark = pytest.mark.skipif(
    any(not p.is_file() for p in _REQUIRED_TEX),
    reason="clanek_en/sections/*.tex not found (unexpected repo layout).",
)


def test_fixed_figures_clear_the_font_size_floor() -> None:
    missing = _missing_pdfs()
    if missing:
        pytest.skip(f"Figure PDF(s) not built locally: {missing} (run the corresponding fig_*.py --full).")

    annotation_min_pt = float(load_config()["figures"]["layout"]["annotation_min_pt"])
    df = check_all(IMG_DIR, annotation_min_pt)
    df = df.set_index("figure")

    failures = []
    for name in _FIXED_FIGURES:
        assert name in df.index, f"{name}: not found by check_all (not referenced via \\includegraphics?)."
        row = df.loc[name]
        if row["status"] != "OK":
            failures.append(
                f"{name}: effective_min_font_pt={row['effective_min_font_pt']:.2f} "
                f"< floor {annotation_min_pt}pt (status={row['status']})"
            )
    assert not failures, "Font-size floor regression:\n" + "\n".join(failures)
