# Project: Sammon mapping revisited (alpha-Sammon, scalable solver, temporal extension)
# Author: Martin Radvansky, VSB - Technical University of Ostrava
# Contact: martinradvansky@gmail.com
# Created: 2026-09-09
# License: for research use within the project; see projectstate.md
#
# SVG -> PDF converter for hand-editable schematic figures (clanek/img/src/*.svg).
# Pure-Python fallback used because Inkscape is not on PATH on the author's
# machine: svglib (SVG parser) + reportlab (PDF writer).
#
# The figures use the "DejaVu Sans" typeface, which covers Greek letters
# (alpha, lambda, sigma) and math symbols (double bar, minus sign, superscripts)
# that the built-in Type1 Helvetica of reportlab cannot encode. The TrueType
# files are looked up in the Windows font directory (and in a few common
# fallback locations); the fonts are embedded (subset) into the PDF.
#
# Usage:
#   venv\python.exe clanek\img\src\svg2pdf.py <input.svg> <output.pdf>
#
# The script fails loudly (non-zero exit code) when the SVG cannot be parsed,
# when the font files are missing, or when the resulting PDF is not a single
# page with the page size declared in the SVG width/height attributes.

import os
import sys

from reportlab.graphics import renderPDF
from svglib.fonts import register_font
from svglib.svglib import svg2rlg

# Font family name used inside the SVG files (font-family="DejaVu Sans").
FONT_FAMILY = "DejaVu Sans"

# Candidate directories with DejaVu TrueType files, first hit wins.
FONT_DIRS = [
    os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"),
    r"C:\Program Files\Inkscape\share\inkscape\fonts",
    "/usr/share/fonts/truetype/dejavu",
]

# (weight, style) -> file name
FONT_FILES = {
    ("normal", "normal"): "DejaVuSans.ttf",
    ("bold", "normal"): "DejaVuSans-Bold.ttf",
    ("normal", "italic"): "DejaVuSans-Oblique.ttf",
    ("bold", "italic"): "DejaVuSans-BoldOblique.ttf",
}

# Tolerance (in PostScript points) for the page-size check.
PAGE_SIZE_TOL_PT = 0.5


def find_font_dir():
    """Vrati prvni adresar, ve kterem existuji vsechny pozadovane TTF soubory DejaVu Sans."""
    for d in FONT_DIRS:
        if all(os.path.isfile(os.path.join(d, f)) for f in FONT_FILES.values()):
            return d
    raise FileNotFoundError(
        "DejaVu Sans TrueType files not found in any of: " + "; ".join(FONT_DIRS)
    )


def register_fonts():
    """Zaregistruje ctyri rezy DejaVu Sans do svglib/reportlab pod jmenem FONT_FAMILY."""
    font_dir = find_font_dir()
    for (weight, style), file_name in FONT_FILES.items():
        path = os.path.join(font_dir, file_name)
        name, ok = register_font(FONT_FAMILY, path, weight=weight, style=style)
        if not ok:
            raise RuntimeError(f"Failed to register font {path} as {FONT_FAMILY} {weight}/{style}")
    return font_dir


def convert(svg_path, pdf_path):
    """Prevede SVG na jednostrankove PDF a vrati rozmery stranky v bodech (sirka, vyska)."""
    drawing = svg2rlg(svg_path)
    if drawing is None:
        raise RuntimeError(f"svglib could not parse {svg_path}")
    # reportlab.graphics.renderPDF.drawToFile falls back to the module-wide
    # default STATE_DEFAULTS['fontName'] (= reportlab.rl_config.defaultGraphicsFontName,
    # which is the non-embedded standard font "Times-Roman") for the canvas
    # text-state preamble whenever drawing.initialFontName is unset. That
    # preamble font is never used to draw a glyph, but it still ends up as an
    # unembedded /Type1 font in the PDF's font resource dictionary, which
    # Springer's font-embedding check rejects. Setting initialFontName to our
    # registered (embedded, dynamic) TrueType font makes drawToFile skip the
    # preamble Tf line entirely (see reportlab.pdfgen.canvas._make_preamble,
    # which only emits it for non-dynamic fonts).
    drawing.initialFontName = FONT_FAMILY
    renderPDF.drawToFile(drawing, pdf_path, autoSize=1)
    return drawing.width, drawing.height


def check_pdf(pdf_path, expected_w_pt, expected_h_pt):
    """Overi, ze PDF ma jedinou stranku a MediaBox odpovida rozmerum SVG (v bodech)."""
    from pypdf import PdfReader  # independent PDF reader (not reportlab) for the check

    reader = PdfReader(pdf_path)
    n_pages = len(reader.pages)
    if n_pages != 1:
        raise RuntimeError(f"{pdf_path}: expected 1 page, found {n_pages}")
    box = reader.pages[0].mediabox
    w, h = float(box.width), float(box.height)
    if abs(w - expected_w_pt) > PAGE_SIZE_TOL_PT or abs(h - expected_h_pt) > PAGE_SIZE_TOL_PT:
        raise RuntimeError(
            f"{pdf_path}: page size {w:.2f} x {h:.2f} pt differs from SVG size "
            f"{expected_w_pt:.2f} x {expected_h_pt:.2f} pt"
        )
    return w, h


def list_fonts(pdf_path):
    """Vrati seznam (font_name, base_font, embedded_bool) pro vsechny fonty pouzite na strance 1."""
    from pypdf import PdfReader  # independent PDF reader (not reportlab) for the check

    reader = PdfReader(pdf_path)
    resources = reader.pages[0].get("/Resources", {})
    font_dict = resources.get("/Font")
    fonts = []
    if font_dict is None:
        return fonts
    for font_key, font_ref in font_dict.items():
        font_obj = font_ref.get_object()
        base_font = str(font_obj.get("/BaseFont", "?"))
        descriptor = font_obj.get("/FontDescriptor")
        embedded = False
        if descriptor is not None:
            descriptor_obj = descriptor.get_object()
            embedded = any(k in descriptor_obj for k in ("/FontFile", "/FontFile2", "/FontFile3"))
        fonts.append((str(font_key), base_font, embedded))
    return fonts


def check_fonts_embedded(pdf_path):
    """Fail-loud kontrola: kazdy font v /Resources musi mit vlozeny font program (FontFile*)."""
    fonts = list_fonts(pdf_path)
    if not fonts:
        raise RuntimeError(f"{pdf_path}: no /Font resource found on page 1 (unexpected for a text figure)")
    not_embedded = [(name, base) for name, base, embedded in fonts if not embedded]
    if not_embedded:
        listing = ", ".join(f"{name}={base}" for name, base in not_embedded)
        raise RuntimeError(f"{pdf_path}: non-embedded font(s) found: {listing}")
    return fonts


def main(argv):
    """Vstupni bod: argumenty <input.svg> <output.pdf>, vypise souhrn a vrati navratovy kod."""
    if len(argv) != 3:
        print("Usage: svg2pdf.py <input.svg> <output.pdf>", file=sys.stderr)
        return 2
    svg_path, pdf_path = argv[1], argv[2]
    if not os.path.isfile(svg_path):
        print(f"ERROR: input SVG not found: {svg_path}", file=sys.stderr)
        return 1
    font_dir = register_fonts()
    w_pt, h_pt = convert(svg_path, pdf_path)
    w_chk, h_chk = check_pdf(pdf_path, w_pt, h_pt)
    fonts = check_fonts_embedded(pdf_path)
    font_listing = ", ".join(f"{name}:{base} (embedded)" for name, base, _ in fonts)
    print(
        f"OK: {pdf_path}  1 page, {w_chk:.2f} x {h_chk:.2f} pt "
        f"= {w_chk / 72 * 25.4:.1f} x {h_chk / 72 * 25.4:.1f} mm  (fonts: {font_dir})"
    )
    print(f"OK: fonts embedded: {font_listing}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
