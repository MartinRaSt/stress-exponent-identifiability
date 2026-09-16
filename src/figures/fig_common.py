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
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.common.checkpoint import results_csv_path
from src.common.config import ensure_dir, get_mode_path, get_path, load_config
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


def save_figure(fig, name: str) -> tuple[Path, Path]:
    """Save the figure as a vector PDF to results/figures/ AND (only in
    'full' mode) a copy to clanek/img/ (the same file, two locations
    required by project rules). In quick/smoke mode the PDF goes only to
    results/figures/<mode>/ and the second item of the return pair is None.
    Returns (results_path, article_path | None)."""
    out_results = figures_out_dir() / f"{name}.pdf"
    fig.savefig(out_results, format="pdf", dpi=RASTER_DPI)
    out_article = None
    if _CURRENT_MODE == "full":
        out_article = article_img_dir() / f"{name}.pdf"
        fig.savefig(out_article, format="pdf", dpi=RASTER_DPI)
    plt.close(fig)
    return out_results, out_article


def save_csv_alongside(df: pd.DataFrame, name: str) -> Path:
    """Save a CSV with the underlying data alongside the figure in results/figures/."""
    out = figures_out_dir() / f"{name}.csv"
    df.to_csv(out, index=False)
    return out
