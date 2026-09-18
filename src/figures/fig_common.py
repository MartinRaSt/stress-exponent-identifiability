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

# figure widths in inches for the Elsevier-style layout (90mm single-column, 190mm full width)
WIDTH_SINGLE_COL_IN = 90.0 / 25.4
WIDTH_FULL_WIDTH_IN = 190.0 / 25.4

# DPI of rasterized layers (rasterized=True: graph edges as a LineCollection,
# dense point clouds) - from config.yaml figures.raster_dpi (project rule: 300 dpi)
RASTER_DPI = int(load_config()["figures"]["raster_dpi"])
MIN_FONT_PT = 7

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.size"] = max(MIN_FONT_PT, 8)


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
    out_results = figures_out_dir() / f"{name}.pdf"
    fig.savefig(out_results, format="pdf", dpi=RASTER_DPI)
    out_article = None
    if _CURRENT_MODE == "full" and _is_article_figure(name):
        out_article = article_img_dir() / f"{name}.pdf"
        fig.savefig(out_article, format="pdf", dpi=RASTER_DPI)
    plt.close(fig)
    return out_results, out_article


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
