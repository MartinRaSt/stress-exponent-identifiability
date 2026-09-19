# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-09
# License: see the LICENSE file in the repository root
"""
Shared constants and helper functions for all scripts in `src/figures/`
(reserse/2026-09-09_specifikace_metody.md section 14): the Okabe-Ito
colorblind-safe palette, vector PDF (pdf.fonttype=42), column widths per
the Elsevier layout (90mm single-column / 190mm full width), and
`require_csv` - a fail-loud check for the existence of input data (no
fabrication of substitute plots).
"""
from __future__ import annotations

import argparse
import functools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.common.checkpoint import results_csv_path
from src.common.config import ensure_dir, get_mode_path, get_path, load_config
from src.common.display_labels import display_label  # noqa: F401 (re-exported, see below)
from src.experiments.exp_common import add_mode_args, resolve_experiment_name, resolve_mode

# Okabe & Ito (2008) colorblind-safe palette - used across all figures
OKABE_ITO = [
    "#000000", "#E69F00", "#56B4E9", "#009E73",
    "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
]

# Figure widths in inches. WIDTH_SINGLE_COL_IN (90mm) is only ever embedded
# at [width=\columnwidth]/[width=0.75\columnwidth] in the elsarticle
# supplement (\columnwidth=390pt there, measured with a pdflatex probe - see
# figures.layout in config.yaml), which ENLARGES these figures, so it is not
# part of the 2026-09-19 font-size fix and stays a literal constant.
# WIDTH_FULL_WIDTH_IN, in contrast, is read from config.yaml
# (figures.layout.dami_full_width_pt): every full-width figure is embedded at
# [width=\textwidth] in BOTH the DAMI main text (\textwidth=372pt) and the
# elsarticle supplement (\textwidth=390pt); drawing at exactly the smaller
# (DAMI) target means LaTeX no longer shrinks it there (scale=1) and only
# mildly enlarges it in the supplement (390/372=1.05) - see the config
# comment for the full derivation (podklady/2026-09-19_FINAL_validace_rukopis.md V2).
WIDTH_SINGLE_COL_IN = 90.0 / 25.4
_FIGURES_LAYOUT_CFG = load_config()["figures"]["layout"]
WIDTH_FULL_WIDTH_IN = float(_FIGURES_LAYOUT_CFG["dami_full_width_pt"]) / 72.0

# 2026-09-19 supplement font-size fix (analogous to the 2026-09-19 DAMI fix
# above): eight figures are embedded ONLY in the elsarticle supplement
# (clanek_en/supplement/sections/*.tex), never in the DAMI main text
# (clanek_en/sections/*.tex) - fig_alpha_curves_all_p1/p2,
# fig_alpha_gain_by_regime, fig_metric_correlations, fig_pareto_front,
# fig_runtime_scaling, fig_sgd_convergence, fig_temporal_pareto. They were
# still drawn at WIDTH_FULL_WIDTH_IN (372pt, the DAMI target) or at
# WIDTH_SINGLE_COL_IN literally multiplied/left alone, so
# `\includegraphics[width=\columnwidth]`/`[width=0.75\columnwidth]` in the
# supplement (`\columnwidth` == `\textwidth` == 390pt there, single-column
# elsarticle preprint layout - see figures.layout.supplement_textwidth_pt)
# printed them at a non-1 scale (0.72-1.53x observed) on top of already
# small literal fontsize= values, both directions of which independently
# violate the Springer floor. `WIDTH_SUPPLEMENT_FULL_IN` /
# `WIDTH_SUPPLEMENT_THREEQ_IN` draw these eight 1:1 at their actual
# `\columnwidth`/`0.75\columnwidth` embed size, so the supplement's
# `[width=...]` is a no-op (scale=1) exactly like WIDTH_FULL_WIDTH_IN does
# for the DAMI main text.
WIDTH_SUPPLEMENT_FULL_IN = float(_FIGURES_LAYOUT_CFG["supplement_textwidth_pt"]) / 72.0
WIDTH_SUPPLEMENT_THREEQ_IN = 0.75 * WIDTH_SUPPLEMENT_FULL_IN

# DPI of rasterized layers (rasterized=True: graph edges as a LineCollection,
# dense point clouds) - from config.yaml figures.raster_dpi (project rule: 300 dpi)
RASTER_DPI = int(load_config()["figures"]["raster_dpi"])

# Springer Nature artwork font-size floors (config.yaml figures.layout), now
# that WIDTH_FULL_WIDTH_IN above makes [width=\textwidth] a no-op scale (see
# comment there): LABEL_FONT_PT for axis/tick/legend labels, ANNOTATION_FONT_PT
# for the smallest permitted annotation (dense per-panel text, subscripts).
# Every fig_*.py script should pull its fontsize= values from one of these
# two (or a config-driven value derived from them) instead of a bare number.
LABEL_FONT_PT = float(_FIGURES_LAYOUT_CFG["label_min_pt"])
ANNOTATION_FONT_PT = float(_FIGURES_LAYOUT_CFG["annotation_min_pt"])
MIN_FONT_PT = ANNOTATION_FONT_PT

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.size"] = LABEL_FONT_PT


def add_quick_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add --quick/--full/--smoke (default --full) - determines which
    `results/data/` subdirectory input data is read from (see
    `resolve_experiment_name`: the `results/data/` root is reserved for
    `--full`, `--quick`/`--smoke` have their own subdirectories). Figures
    for the article are generated from `--full` data; `--quick`/`--smoke`
    are for quick verification during development."""
    return add_mode_args(parser)


# Current mode of the figure script (set by `parse_fig_mode`). Determines
# where figures are saved: only 'full' may overwrite results/figures/ and
# clanek/img/; 'quick'/'smoke' write to results/figures/<mode>/ and are NOT
# copied into the article (otherwise smoke data would overwrite the
# article's final figures).
_CURRENT_MODE = "full"


def parse_fig_mode(args: argparse.Namespace) -> str:
    """Return the mode ('quick'/'smoke'/'full') from the already-parsed
    arguments and remember it for `save_figure`/`figures_out_dir`."""
    global _CURRENT_MODE
    _CURRENT_MODE = resolve_mode(args)
    return _CURRENT_MODE


def require_csv(path: Path, how_to_generate: str) -> pd.DataFrame:
    """Load a CSV, or raise a clear error with a hint on how to generate the
    missing data (fail loud - no fabrication of substitute/illustrative data)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing input data for the figure: {path}\n"
            f"Generate it before running this script: {how_to_generate}"
        )
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Input CSV {path} exists but is empty - nothing to plot.")
    return df


def mode_data_dir(mode: str) -> Path:
    """Return the `results/data/` subdirectory corresponding to the mode
    (the root for 'full', `quick/`/`smoke/` otherwise) - for inputs that
    are not directly `<experiment>_results.csv` (e.g.
    `exp2_convergence_curves.csv`, `exp4_trajectories.csv`, `stats_*.csv`)."""
    return get_mode_path("results_data_dir", mode)


def require_experiment_csv(base_experiment_name: str, mode: str = "full", how_to_generate: str | None = None) -> pd.DataFrame:
    """Shorthand helper for `results/data/[<mode>/]<base_experiment_name>_results.csv`
    (see `resolve_experiment_name` - the root is only for 'full')."""
    resolved_name = resolve_experiment_name(base_experiment_name, mode)
    path = results_csv_path(resolved_name)
    hint = how_to_generate or f"venv\\python.exe -m src.experiments.{base_experiment_name} --{mode}"
    return require_csv(path, hint)


def figures_out_dir() -> Path:
    return ensure_dir(get_mode_path("results_figures_dir", _CURRENT_MODE))


def article_img_dir() -> Path:
    """`clanek/img/` - a copy of final PDFs for the article (see the rules in projectstate.md)."""
    return ensure_dir(get_path("results_dir").parent / "clanek" / "img")


@functools.lru_cache(maxsize=1)
def _article_figures_patterns() -> tuple[str, ...]:
    """`figures.article_figures` from config.yaml - the whitelist of figure
    names (without extension) that `save_figure` mirrors into clanek/img/
    (author feedback 2026-09-17: orphan PDFs the article never
    \\includegraphics-references must stop accumulating there - see the
    config comment). Fail loud: an empty/missing list is a config error,
    not a silent "copy everything" or "copy nothing"."""
    patterns = load_config().get("figures", {}).get("article_figures")
    if not patterns:
        raise KeyError(
            "Missing or empty 'figures.article_figures' in src/common/config.yaml - "
            "required to decide which figures save_figure() mirrors into clanek/img/."
        )
    return tuple(str(p) for p in patterns)


def _is_article_figure(name: str) -> bool:
    """True if `name` is on the `figures.article_figures` whitelist - exact
    match, or prefix match for entries ending in '*' (e.g. 'fig_cd_diagram_*'
    covers 'fig_cd_diagram_<experiment>_<metric>', one file per
    experiment/metric pair)."""
    for pattern in _article_figures_patterns():
        if pattern.endswith("*"):
            if name.startswith(pattern[:-1]):
                return True
        elif name == pattern:
            return True
    return False


def save_figure(fig, name: str) -> tuple[Path, Path | None]:
    """Save the figure as a vector PDF to results/figures/ (ALWAYS, every
    mode, every figure - nothing is ever lost) AND, only in 'full' mode AND
    only if `name` is on the `figures.article_figures` whitelist, a copy to
    clanek/img/ (author feedback 2026-09-17: clanek/img/ must contain only
    figures the article actually \\includegraphics-references, or figures
    explicitly slated to be wired in - see the config comment; everything
    else stays in results/figures/ only). Returns (results_path,
    article_path | None) - article_path is None both in quick/smoke mode
    and in full mode for a figure not on the whitelist."""
    _record_axes_area(name, fig)
    out_results = figures_out_dir() / f"{name}.pdf"
    fig.savefig(out_results, format="pdf", dpi=RASTER_DPI)
    out_article = None
    if _CURRENT_MODE == "full" and _is_article_figure(name):
        out_article = article_img_dir() / f"{name}.pdf"
        fig.savefig(out_article, format="pdf", dpi=RASTER_DPI)
    plt.close(fig)
    return out_results, out_article


def axes_area_fraction(fig) -> float:
    """Fraction of `fig`'s total canvas area covered by the union of its
    Axes' plotting boxes (`ax.get_position()`, in figure-fraction units -
    i.e. NOT including tick labels, titles, or axis labels, which are drawn
    outside this box). This is the operational, programmatic definition of
    "kreslici plocha" (drawing area) used by
    src/figures/check_axes_area.py (author requirement 2026-09-19: after
    the font-size fix enlarged every label, several figures ended up with
    text/legends/per-panel captions occupying most of the canvas - the
    axes area must dominate). For a multi-panel figure this single number
    IS the average over panels, weighted by each panel's own size (summing
    every panel's box area and dividing by the total canvas area is
    algebraically the area-weighted average of each panel's own
    box-fraction) - no separate per-panel loop is needed.

    Twin axes (`ax.twinx()`/`ax.twiny()`) share an IDENTICAL bounding box
    with their host axes; counting both would double-count the same
    drawing area, so distinct axes are deduplicated by their (rounded)
    bounding box before summing. A hidden axes (`ax.set_visible(False)` or
    `ax.axis('off')` - the latter keeps the axes "visible" as a Python
    object but draws nothing) is excluded via `get_visible()`; `ax.axis('off')`
    panels still reserve a gridspec cell and are intentionally still counted
    here (they occupy canvas area the reader sees as blank, which is exactly
    what this metric should penalize)."""
    # Figures using a layout engine (constrained_layout=True) only apply it
    # during a draw - ax.get_position() beforehand would still return the
    # figure's raw, pre-layout GridSpec positions (equal spacing, no margin
    # adjustment), silently under/over-counting the real printed axes area.
    # draw_without_rendering() runs the layout engine (constrained/tight)
    # without needing a real backend render, so this is accurate for BOTH
    # layout-engine figures and figures using static explicit margins
    # (idempotent for the latter - re-running the, already fixed, GridSpec
    # math changes nothing).
    fig.draw_without_rendering()
    boxes = set()
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        b = ax.get_position()
        boxes.add((round(b.x0, 6), round(b.y0, 6), round(b.x1, 6), round(b.y1, 6)))
    return float(sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in boxes))


def _record_axes_area(name: str, fig) -> None:
    """Append/update `name`'s row in results/figures/[<mode>/]check_axes_area.csv
    (one row per figure, keyed by name - re-running a figure script
    overwrites its own row, never duplicates it). Recorded for EVERY mode
    (quick/smoke/full) into that mode's own output directory (see
    figures_out_dir), consistent with the project's smoke/quick separation
    rule - src/figures/check_axes_area.py reads whichever mode it is
    pointed at."""
    width_in, height_in = fig.get_size_inches()
    row = {
        "figure": name,
        "axes_area_fraction": axes_area_fraction(fig),
        "width_pt": float(width_in) * 72.0,
        "height_pt": float(height_in) * 72.0,
        "n_axes": sum(1 for ax in fig.axes if ax.get_visible()),
    }
    out_csv = figures_out_dir() / "check_axes_area.csv"
    if out_csv.exists():
        df = pd.read_csv(out_csv)
        df = df[df["figure"] != name]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.sort_values("figure").to_csv(out_csv, index=False)


def save_csv_alongside(df: pd.DataFrame, name: str) -> Path:
    """Save a CSV with the underlying data alongside the figure in results/figures/."""
    out = figures_out_dir() / f"{name}.csv"
    df.to_csv(out, index=False)
    return out


# --- display_label: human-readable labels for raw config/CSV identifiers ---
#
# Author feedback 2026-09-17: 11 figure scripts printed raw identifiers
# (e.g. "auc_rnx", "mnist_784", "sammon_alpha_smacof") straight into titles,
# axis labels, ticks and legends - "vypada to blbe". `display_label` is the
# ONE shared conversion point (fig_faithful_map used to keep a private
# method_labels/dataset_labels config block just for itself - now merged
# into `display_labels.<kind>` here, used by every figure script).
#
# Moved to `src/common/display_labels.py` on 2026-09-18 (author feedback:
# LaTeX table generation in `src/experiments/report_tables.py` needs the
# exact same lookup for table body cells, but must NOT import matplotlib) -
# imported at the top of this module and re-exported here unchanged, so
# every existing `from src.figures.fig_common import display_label` keeps
# working without modification.
