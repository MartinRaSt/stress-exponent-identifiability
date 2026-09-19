# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-19
# License: see the LICENSE file in the repository root
"""Regression test for the Springer pre-submission font-embedding finding
(2026-09-19): clanek/img/method_overview.pdf and clanek/img/study_design.pdf
carried an unembedded /Times-Roman font, even though the SVG source only
uses "DejaVu Sans" glyphs. Root cause: reportlab.graphics.renderPDF.drawToFile
falls back to STATE_DEFAULTS['fontName'] (= the non-embedded standard font
"Times-Roman") for a canvas text-state preamble whenever the Drawing has no
initialFontName set; clanek/img/src/svg2pdf.py now sets it explicitly to the
registered, embedded DejaVu Sans font, which makes reportlab skip that
preamble line entirely.

This test regenerates all three schematic figures from their checked-in SVG
sources into a scratch directory (never touching clanek/img/) and checks,
independently of reportlab, that every font referenced on page 1 has an
embedded font program (FontFile/FontFile2/FontFile3)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SVG_SRC_DIR = PROJECT_ROOT / "clanek" / "img" / "src"
SVG2PDF_PATH = SVG_SRC_DIR / "svg2pdf.py"

SCHEMATIC_SVGS = [
    "method_overview.svg",
    "study_design.svg",
    "graphical_abstract.svg",
]


def _load_svg2pdf():
    """Nacte clanek/img/src/svg2pdf.py jako modul (neni to soucast balicku src/)."""
    spec = importlib.util.spec_from_file_location("svg2pdf", SVG2PDF_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


pytestmark = pytest.mark.skipif(
    not SVG2PDF_PATH.is_file(), reason="clanek/img/src/svg2pdf.py not found"
)


@pytest.mark.parametrize("svg_name", SCHEMATIC_SVGS)
def test_schematic_pdf_has_only_embedded_fonts(tmp_path, svg_name) -> None:
    svg_path = SVG_SRC_DIR / svg_name
    if not svg_path.is_file():
        pytest.skip(f"source SVG not found: {svg_path}")
    svg2pdf = _load_svg2pdf()
    try:
        svg2pdf.register_fonts()
    except FileNotFoundError as exc:
        pytest.skip(f"DejaVu Sans TrueType files not available: {exc}")

    out_pdf = tmp_path / (svg_path.stem + ".pdf")
    w_pt, h_pt = svg2pdf.convert(str(svg_path), str(out_pdf))
    svg2pdf.check_pdf(str(out_pdf), w_pt, h_pt)

    fonts = svg2pdf.check_fonts_embedded(str(out_pdf))
    assert fonts, "expected at least one font resource on page 1"
    base_fonts = {base for _, base, _ in fonts}
    assert not any("Times" in b for b in base_fonts), (
        f"non-embedded standard font leaked into {svg_name}: {base_fonts}"
    )


def test_check_fonts_embedded_rejects_non_embedded_standard_font(tmp_path) -> None:
    """Sanity check that the checker actually fails on a PDF with a genuine
    non-embedded standard font (guards against a checker that always passes)."""
    svg2pdf = _load_svg2pdf()
    from reportlab.pdfgen import canvas

    bad_pdf = tmp_path / "bad_times_roman.pdf"
    c = canvas.Canvas(str(bad_pdf), pagesize=(100, 100))
    c.setFont("Times-Roman", 12)
    c.drawString(10, 50, "not embedded")
    c.save()

    with pytest.raises(RuntimeError, match="non-embedded font"):
        svg2pdf.check_fonts_embedded(str(bad_pdf))
