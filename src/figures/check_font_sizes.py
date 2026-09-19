# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-19
# License: see the LICENSE file in the repository root
r"""
Fail-loud check of the Springer Nature artwork font-size guideline (~8pt
minimum, never below 7pt) for every article figure, IN THE SIZE IT ACTUALLY
PRINTS AT (podklady/2026-09-19_FINAL_validace_rukopis.md section V2): the
PDF a figure script writes is not necessarily embedded 1:1 - LaTeX rescales
it to whatever `\includegraphics[width=...]` says, so a "7.4pt" fontsize in
matplotlib can still print below 7pt if the PDF is shrunk on the way in.

Method (no manual/eyeballed numbers - everything is measured or parsed):
  1. Parse every `\includegraphics[width=<spec>]{<path>}` in
     clanek_en/sections/*.tex (the DAMI main text) and
     clanek_en/supplement/sections/*.tex (the elsarticle supplement),
     resolving `<spec>` (\textwidth, \columnwidth, 0.75\columnwidth, ...)
     against the two page geometries in config.yaml figures.layout
     (dami_full_width_pt / supplement_textwidth_pt - both measured with a
     pdflatex `\the\textwidth` probe, see the config comments).
  2. For each resolved (figure, target_width_pt) pair, open the actual PDF
     in clanek/img/, read its native MediaBox width, and compute
     scale = target_width_pt / native_width_pt.
  3. Extract every `<size> Tf` font-size operator from the PDF's page-1
     content stream (pypdf) and multiply by `scale` - this is the size the
     glyph prints at once LaTeX places it, independent of what the drawing
     script's own rcParams/fontsize= said.
  4. A figure referenced more than once (different widths) is checked
     against ALL of them - the report keeps the worst (smallest) case.
  5. Sizes that are ~0.7x another observed size in the SAME figure are
     flagged as the conventional mathtext sub/superscript shrink (e.g. the
     "NN" in rho_NN) rather than a body/annotation text violation - this is
     universal scientific-figure typography (matplotlib's default
     DejaVu-based mathtext fontset always shrinks scripts by this fixed
     ratio; there is no public rcParam to change it), not a script bug, and
     is reported separately from real violations.

Exit code 0 iff no figure's smallest NON-subscript glyph is below
`figures.layout.annotation_min_pt` (config.yaml; the Springer floor is 7pt,
the configured target keeps a small safety margin above it). Always prints
one row per figure; always writes the same table as CSV alongside the
figures it checked (results/figures/check_font_sizes.csv) - this script
does not "fix" anything, only measures and reports.

Run: venv\\python.exe -m src.figures.check_font_sizes
or:  src\\check_font_sizes.bat
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import pypdf

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from src.common.config import load_config, get_path  # noqa: E402

# --- 1. parse \includegraphics[width=...]{path} from the article .tex ------

_INCLUDEGRAPHICS_RE = re.compile(r"\\includegraphics\[width=([^\]]+)\]\{([^}]+)\}")

# tex source dir -> (label, base-width-pt config key)
_TEX_SOURCE_DIRS = {
    PROJECT_ROOT / "clanek_en" / "sections": ("DAMI main text", "dami_full_width_pt"),
    PROJECT_ROOT / "clanek_en" / "supplement" / "sections": ("elsarticle supplement", "supplement_textwidth_pt"),
}


def _resolve_width_pt(spec: str, base_width_pt: float) -> float:
    """Resolve a LaTeX width spec (e.g. '\\textwidth', '\\columnwidth',
    '0.75\\columnwidth') to points, given the page's \\textwidth==\\columnwidth
    in points (both are equal in the single-column layouts used here - see
    the config.yaml figures.layout comment). Fail loud on any other unit
    (e.g. a literal 'Ncm'/'Npt') - this project only ever uses fractions of
    \\textwidth/\\columnwidth for figures, so anything else is unexpected
    and should be reviewed by hand, not silently guessed at."""
    spec = spec.strip()
    m = re.fullmatch(r"([0-9.]*)\\(?:textwidth|columnwidth)", spec)
    if not m:
        raise ValueError(f"Unrecognized \\includegraphics width spec {spec!r} - extend _resolve_width_pt.")
    factor = float(m.group(1)) if m.group(1) else 1.0
    return factor * base_width_pt


def _find_includegraphics() -> list[tuple[str, str, float]]:
    """Returns a list of (figure_stem, tex_source_label, target_width_pt) -
    one entry per \\includegraphics occurrence found (a figure referenced
    twice at different widths appears twice)."""
    cfg = load_config()["figures"]["layout"]
    found = []
    for tex_dir, (label, width_key) in _TEX_SOURCE_DIRS.items():
        base_width_pt = float(cfg[width_key])
        for tex_path in sorted(tex_dir.glob("*.tex")):
            text = tex_path.read_text(encoding="utf-8")
            for width_spec, rel_path in _INCLUDEGRAPHICS_RE.findall(text):
                stem = Path(rel_path).stem
                target_width_pt = _resolve_width_pt(width_spec, base_width_pt)
                found.append((stem, f"{label} ({tex_path.name})", target_width_pt))
    return found


# --- 2/3. measure a PDF's native width + every Tf font size on page 1 ------

_TF_RE = re.compile(rb"([\d.]+)\s+Tf")


def _measure_pdf(pdf_path: Path) -> tuple[float, list[float]]:
    """Returns (native_mediabox_width_pt, [all distinct Tf font sizes on page 1])."""
    reader = pypdf.PdfReader(str(pdf_path))
    page = reader.pages[0]
    native_width_pt = float(page.mediabox.width)
    content = page.get_contents()
    if content is None:
        raise ValueError(f"{pdf_path}: page 1 has no content stream (unexpected for a text figure).")
    raw = content.get_data()
    sizes = sorted({float(m) for m in _TF_RE.findall(raw)})
    if not sizes:
        raise ValueError(f"{pdf_path}: no 'Tf' font-size operator found on page 1 (unexpected for a text figure).")
    return native_width_pt, sizes


# --- SVG-sourced PDFs (clanek/img/src/*.svg -> svg2pdf.py) -----------------
#
# reportlab/svglib wrap the whole page in a coordinate-transform ("cm")
# operator that maps the SVG's viewBox (user units) onto the physical page,
# and the "Tf" font-size operator is expressed in THAT pre-transform
# (viewBox) space, not in final page points - unlike matplotlib, which
# always emits "Tf" already in absolute page points (verified: for every
# matplotlib figure here, mediabox-based scale reproduces the intended
# fontsize= exactly). So for an SVG-sourced PDF the (target/mediabox) scale
# used for matplotlib PDFs is wrong; the correct scale is
# target_width_pt / viewbox_width_units (the mediabox width cancels out
# completely - see the two file header comments in method_overview.svg/
# study_design.svg for the equivalent derivation from the drawing side).
_SVG_SOURCE_DIR = PROJECT_ROOT / "clanek" / "img" / "src"
_SVG_VIEWBOX_RE = re.compile(r'viewBox="0 0 ([0-9.]+) [0-9.]+"')


def _svg_viewbox_width(stem: str) -> float | None:
    svg_path = _SVG_SOURCE_DIR / f"{stem}.svg"
    if not svg_path.is_file():
        return None
    m = _SVG_VIEWBOX_RE.search(svg_path.read_text(encoding="utf-8"))
    if not m:
        raise ValueError(f"{svg_path}: no viewBox=\"0 0 W H\" found - extend _SVG_VIEWBOX_RE.")
    return float(m.group(1))


# --- 5. mathtext sub/superscript exception ---------------------------------

# matplotlib's mathtext engine multiplies by exactly 0.7 in float arithmetic
# (every genuine instance measured in this project's figures rounds to
# 0.700 at 2-decimal Tf precision) - a tight tolerance here is deliberate:
# it must NOT swallow a figure's own, unrelated, smaller-but-legitimate
# font tier that merely happens to sit near a 0.7 ratio (e.g. two
# independently-chosen tiers 3.78/5.31 user units in study_design.svg ratio
# to 0.712 - a real ~1.7% gap that a looser tolerance would misclassify).
_SUBSCRIPT_RATIO = 0.7
_SUBSCRIPT_RATIO_TOL = 0.005


def _is_subscript_of_another(candidate_pt: float, other_sizes_pt: list[float]) -> bool:
    """True if `candidate_pt` is ~0.7x (matplotlib's fixed mathtext
    sub/superscript shrink, see the module docstring) one of the other
    printed sizes in the same figure."""
    for other in other_sizes_pt:
        if other <= candidate_pt:
            continue
        ratio = candidate_pt / other
        if abs(ratio - _SUBSCRIPT_RATIO) <= _SUBSCRIPT_RATIO_TOL:
            return True
    return False


def check_all(img_dir: Path, annotation_min_pt: float) -> pd.DataFrame:
    """Builds the full report table (one row per figure that is referenced
    at least once via \\includegraphics)."""
    occurrences = _find_includegraphics()
    by_figure: dict[str, list[tuple[str, float]]] = {}
    for stem, source_label, target_width_pt in occurrences:
        by_figure.setdefault(stem, []).append((source_label, target_width_pt))

    rows = []
    for stem in sorted(by_figure):
        pdf_path = img_dir / f"{stem}.pdf"
        if not pdf_path.is_file():
            rows.append({
                "figure": stem, "sources": "; ".join(s for s, _ in by_figure[stem]),
                "target_width_pt": float("nan"), "native_width_pt": float("nan"), "scale": float("nan"),
                "min_font_pt": float("nan"), "max_font_pt": float("nan"),
                "min_is_subscript_exception": False, "status": "MISSING_PDF",
            })
            continue
        native_width_pt, native_sizes = _measure_pdf(pdf_path)
        svg_viewbox_width = _svg_viewbox_width(stem)

        # worst case across every place this figure is embedded (see docstring point 4)
        worst_scale = None
        worst_target = None
        for _source_label, target_width_pt in by_figure[stem]:
            # see the _SVG_SOURCE_DIR comment above: an SVG-sourced PDF's
            # "Tf" is in viewBox units, not page points, so its scale is
            # target/viewbox_width, not target/mediabox_width.
            scale = target_width_pt / (svg_viewbox_width if svg_viewbox_width is not None else native_width_pt)
            if worst_scale is None or scale < worst_scale:
                worst_scale, worst_target = scale, target_width_pt

        printed_sizes = sorted(s * worst_scale for s in native_sizes)
        min_pt, max_pt = printed_sizes[0], printed_sizes[-1]
        is_subscript = _is_subscript_of_another(min_pt, printed_sizes)
        # Every size is checked against the ORIGINAL set (not a shrinking
        # "remaining" list - that would let one exception justify stripping
        # another, unrelated one by transitivity) - a figure can legitimately
        # have several independent subscript tiers (e.g. a plain-text label
        # at LABEL_FONT_PT and a mathtext label at ANNOTATION_FONT_PT each
        # produce their own, differently-sized, ~0.7x subscript).
        non_subscript_sizes = [s for s in printed_sizes if not _is_subscript_of_another(s, printed_sizes)]
        effective_min_pt = min(non_subscript_sizes) if non_subscript_sizes else min_pt

        status = "OK" if effective_min_pt >= annotation_min_pt else "FAIL"
        rows.append({
            "figure": stem, "sources": "; ".join(s for s, _ in by_figure[stem]),
            "target_width_pt": worst_target, "native_width_pt": native_width_pt, "scale": worst_scale,
            "min_font_pt": min_pt, "max_font_pt": max_pt,
            "min_is_subscript_exception": bool(is_subscript and min_pt < annotation_min_pt),
            "effective_min_font_pt": effective_min_pt,
            "status": status,
        })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--img-dir", type=Path, default=None, help="Override clanek/img/ (for testing).")
    args = parser.parse_args()

    cfg = load_config()["figures"]["layout"]
    annotation_min_pt = float(cfg["annotation_min_pt"])
    img_dir = args.img_dir or (get_path("results_dir").parent / "clanek" / "img")

    df = check_all(img_dir, annotation_min_pt)

    out_csv = get_path("results_dir") / "figures" / "check_font_sizes.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 40)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\nWritten: {out_csv}")

    n_fail = int((df["status"] == "FAIL").sum())
    n_missing = int((df["status"] == "MISSING_PDF").sum())
    if n_missing:
        print(f"\n{n_missing} figure(s) referenced in the article but not found in {img_dir} - see MISSING_PDF rows above.")
    if n_fail:
        print(f"\nFAIL: {n_fail} figure(s) have a non-subscript glyph below the {annotation_min_pt}pt floor when printed.")
        return 1
    print(f"\nOK: every referenced figure's smallest non-subscript glyph is >= {annotation_min_pt}pt when printed.")
    return 0 if n_missing == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
