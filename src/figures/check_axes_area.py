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
Fail-loud check of two "proportions" requirements for every figure produced
by src/figures/fig_*.py (author feedback 2026-09-19, "text overwhelms data"
after the font-size fix enlarged every label):

  1. Axes area: the union of a figure's Axes plotting boxes
     (`ax.get_position()`, i.e. NOT tick labels/titles/axis labels, which
     live outside that box) must cover at least
     `figures.layout.min_axes_area_fraction` of the total canvas area -
     measured LIVE, in the matplotlib Figure object, by
     `src.figures.fig_common.axes_area_fraction()` at the moment each
     fig_*.py script calls `save_figure()`, and recorded into
     `results/figures/[<mode>/]check_axes_area.csv` (one row per figure).
     This script does not re-measure it from the PDF (that would need a
     second, error-prone geometry reconstruction) - it reads that CSV,
     which is why a fig_*.py script MUST be (re-)run before this check
     reflects its current state (fail loud below if the CSV row is
     missing/stale relative to figures.article_figures, not silently
     skipped).
  2. Main-text height: a figure that is \includegraphics-referenced from
     the DAMI main text (clanek_en/sections/*.tex) must print at or below
     `figures.layout.main_text_max_height_frac_textheight` of
     `figures.layout.dami_textheight_pt` - otherwise LaTeX pushes it onto
     its own float page at the end of the document. Measured on the ACTUAL
     PDF's MediaBox height (pypdf) in clanek/img/, exactly like
     check_font_sizes.py measures width - these main-text figures draw 1:1
     at `dami_full_width_pt` (see fig_common.WIDTH_FULL_WIDTH_IN), so native
     height == printed height. Supplement-only figures are NOT subject to
     this check (only the main text risks being pushed to a float page).

SVG-sourced figures (method_overview, graphical_abstract, study_design -
produced by clanek/img/src/svg2pdf.py, not by save_figure) have no
matplotlib Axes and are skipped for check 1 with a note, not a failure.

Run: venv\\python.exe -m src.figures.check_axes_area [--mode full]
or:  src\\check_axes_area.bat
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pypdf

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from src.common.config import get_mode_path, get_path, load_config  # noqa: E402
from src.figures.check_font_sizes import _find_includegraphics, _svg_viewbox_width  # noqa: E402


def _main_text_figure_stems() -> set[str]:
    """Stems (without extension) \\includegraphics-referenced from the DAMI
    main text (clanek_en/sections/*.tex) - the subset subject to the
    main-text height cap (see the module docstring, check 2)."""
    return {
        stem
        for stem, source_label, _ in _find_includegraphics()
        if source_label.startswith("DAMI main text")
    }


def check_all(figures_dir: Path, img_dir: Path, min_axes_area: float, max_height_pt: float) -> pd.DataFrame:
    area_csv = figures_dir / "check_axes_area.csv"
    if not area_csv.exists():
        raise FileNotFoundError(
            f"Missing {area_csv} - run the fig_*.py scripts first (each save_figure() call records "
            "one row); this checker only reads already-generated data, it never fabricates a figure."
        )
    area_df = pd.read_csv(area_csv).set_index("figure")
    main_text_stems = _main_text_figure_stems()

    rows = []
    for name, r in area_df.iterrows():
        area_frac = float(r["axes_area_fraction"])
        area_status = "OK" if area_frac >= min_axes_area else "FAIL"
        is_main_text = name in main_text_stems
        height_status = "N/A (supplement-only)"
        height_pt = float("nan")
        if is_main_text:
            pdf_path = img_dir / f"{name}.pdf"
            if not pdf_path.is_file():
                height_status = "MISSING_PDF"
            else:
                reader = pypdf.PdfReader(str(pdf_path))
                height_pt = float(reader.pages[0].mediabox.height)
                height_status = "OK" if height_pt <= max_height_pt else "FAIL"
        rows.append({
            "figure": name,
            "axes_area_fraction": area_frac,
            "n_axes": int(r["n_axes"]),
            "area_status": area_status,
            "is_main_text": is_main_text,
            "height_pt": height_pt,
            "max_height_pt": max_height_pt if is_main_text else float("nan"),
            "height_status": height_status,
        })

    # SVG-sourced article figures never call save_figure() (see the module
    # docstring) - list them separately, as SKIPPED, not silently ignored.
    from src.figures.fig_common import _article_figures_patterns  # local import, avoids a hard dependency at module load

    for pattern in _article_figures_patterns():
        if pattern.endswith("*"):
            continue
        if pattern in area_df.index:
            continue
        if _svg_viewbox_width(pattern) is not None:
            rows.append({
                "figure": pattern, "axes_area_fraction": float("nan"), "n_axes": 0,
                "area_status": "SKIPPED (SVG-sourced, no matplotlib Axes)",
                "is_main_text": pattern in main_text_stems, "height_pt": float("nan"),
                "max_height_pt": float("nan"), "height_status": "N/A",
            })

    return pd.DataFrame(rows).sort_values("figure").reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", default="full", choices=["full", "quick", "smoke"],
                         help="Which results/figures/[<mode>/] to check (default: full, the article's figures).")
    parser.add_argument("--img-dir", type=Path, default=None, help="Override clanek/img/ (for testing).")
    args = parser.parse_args()

    cfg = load_config()["figures"]["layout"]
    min_axes_area = float(cfg["min_axes_area_fraction"])
    max_height_pt = float(cfg["main_text_max_height_frac_textheight"]) * float(cfg["dami_textheight_pt"])

    figures_dir = get_mode_path("results_figures_dir", args.mode)
    img_dir = args.img_dir or (get_path("results_dir").parent / "clanek" / "img")

    df = check_all(figures_dir, img_dir, min_axes_area, max_height_pt)

    out_csv = figures_dir / "check_axes_area_report.csv"
    df.to_csv(out_csv, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 45)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nWritten: {out_csv}")

    n_area_fail = int((df["area_status"] == "FAIL").sum())
    n_height_fail = int((df["height_status"] == "FAIL").sum())
    n_missing = int((df["height_status"] == "MISSING_PDF").sum())
    if n_missing:
        print(f"\n{n_missing} main-text figure(s) missing from {img_dir} - see MISSING_PDF rows above.")
    if n_area_fail:
        print(f"\nFAIL: {n_area_fail} figure(s) below the axes-area floor ({min_axes_area:.0%} of canvas).")
    if n_height_fail:
        print(f"\nFAIL: {n_height_fail} main-text figure(s) exceed the height cap ({max_height_pt:.1f}pt).")
    if n_area_fail == 0 and n_height_fail == 0 and n_missing == 0:
        print(f"\nOK: every figure's axes area is >= {min_axes_area:.0%} and every main-text figure is <= {max_height_pt:.1f}pt tall.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
