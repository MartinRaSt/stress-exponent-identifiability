# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-12
# License: see the LICENSE file in the repository root
"""
K10 (documentation/2026-09-12_plan_smeru_clanku.md) - the "faithful map"
hero figure: a grid of datasets (rows) x methods (columns) over the
ALREADY SAVED embeddings of exp1_dr_benchmark (a single seed, from config
`fig_faithful_map.seed`) + the original data (the same subsample as E1,
see `exp1_dr_benchmark.SUBSAMPLE_SEED`). No re-running of DR methods.

Point color = class (Okabe-Ito for <= 8 classes, otherwise matplotlib
'tab20' - NOTE: the task spec mentions "Tol discrete rainbow" for > 8
classes, but the exact hex values of the Paul Tol (2021) "discrete
rainbow" swatch are not available in the project as a verified source (no
package with this palette is in venv) - matplotlib's built-in 'tab20' is
used instead (20 qualitatively distinct colors, sufficient for the highest
class count observed in E1 datasets, e.g. mnist_784 has 10 digits) - a
documented deviation, not a silent substitution).

Ellipse = 1-sigma (config `ellipse_n_std`) covariance ellipse of the class
spread IN THE OUTPUT (Y), to show the preservation/distortion of relative
cluster size (see K2/R3 class_spread_lie_factor - same motivation, here
visual).

Below each panel: a compact "fidelity strip" (3 horizontal bars, longer =
better) for centroid_dist_spearman, class_spread_spearman
(results/data/exp1_cluster_geometry_results.csv, K2) and
stress_scale_invariant (results/data/exp1_dr_benchmark_results.csv), all
for the same (dataset, method, seed) - see `_corr_to_fidelity`/
`_stress_to_fidelity` for the [0, 1] "longer bar = better" normalization
(the two rank correlations are in [-1, 1] and stress is an unbounded cost,
so raw values are not directly comparable as bar lengths without it). The
raw value is always printed next to the bar so nothing is hidden by the
normalization.

Optional overlay: point transparency = local stress residual
r_i = sum_j (D_ij - d_ij)^2 / sum_j D_ij^2 (w_ij=1, i.e. alpha=0
weighting, D = the input distance matrix on the same subsample as E1, d =
the Euclidean distance in the output Y WITHOUT optimal scaling s* - see
the K10 task spec formula).

2026-09-17 revision (shorten+strengthen the article): added the classical
alpha=1 Sammon column (`sammon_alpha_smacof`, see the config comment) and
switched all column/row headers from raw config keys to human-readable
labels (now via the SHARED `display_labels.method`/`dataset` section +
`src.figures.fig_common.display_label`, used by every figure script).

2026-09-19 proportions fix (author feedback: "text overwhelms data" - the
row label was clipped ("Hierarchical clusters" printed as "erarchical
clusters") and the below-panel "fidelity strip" of 3 bars + printed values
took almost as much vertical space as the scatter itself, so the axes
(drawing) area was only ~39% of the canvas, see
results/figures/check_axes_area.csv before this fix):
  - the 3-bar fidelity strip (its own sub-axes per panel) is REMOVED; the
    same three raw numbers (centroid rank corr. / spread rank corr. /
    scale-invariant stress) are now a single compact "c/s/e" line drawn
    INSIDE each scatter panel (small semi-transparent box, bottom-left
    corner) - no separate axes, so the scatter keeps the full panel area.
    The bar-chart visualization (longer bar = better) is dropped, not
    just compressed, because at panel width ~74pt (372pt / 5 columns) a
    legible bar+track+value no longer fit even on one row; the LaTeX
    caption (clanek_en/sections/05_vysledky.tex) now carries what the
    bars used to convey ("longer = better") in words instead.
  - row labels (dataset names) are drawn HORIZONTALLY via `fig.text`
    (using each row's actual axes y-center from `ax.get_position()`,
    measured, not guessed) instead of a rotated `ax.set_ylabel` - the
    previous rotated label's height (~1 pt/character along the reading
    direction) exceeded the row's own axes height for the longest label
    ("Hierarchical clusters") and was clipped by the canvas edge; a
    horizontal, `textwrap`-wrapped label only needs canvas WIDTH (the left
    margin), which is unconstrained by row count.
  - the in-image suptitle and the multi-line explanatory caption text are
    both removed (the LaTeX caption command already carries this, see
    05_vysledky.tex) - freeing the top/bottom margins that used to hold
    them so the panel grid itself can be taller (i.e. more legible),
    while still keeping the OVERALL figure height comfortably under the
    `figures.layout.main_text_max_height_frac_textheight` cap (checked by
    `src/figures/check_axes_area.py`).

Run: venv\\python.exe -m src.figures.fig_faithful_map [--quick|--full|--smoke]
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from scipy.spatial.distance import pdist, squareform

from src.common.checkpoint import RunKey, load_embedding
from src.experiments.config_experiments import load_experiments_config
from src.experiments.exp_common import resolve_experiment_name
from src.figures.fig_common import (
    ANNOTATION_FONT_PT,
    LABEL_FONT_PT,
    OKABE_ITO,
    WIDTH_FULL_WIDTH_IN,
    add_quick_arg,
    display_label,
    figures_out_dir,
    parse_fig_mode,
    require_experiment_csv,
    save_csv_alongside,
    save_figure,
)

FIG_NAME = "fig_faithful_map"
BASE_EXPERIMENT_NAME = "exp1_dr_benchmark"

# Layout constants (pure drawing geometry, not analysis parameters - see the
# _FIDELITY_BAR_COLOR precedent this file used to have for the same
# rationale). Explicit margins are used INSTEAD of fig.tight_layout(),
# which does not lay out subgridspec children correctly here (large stray
# whitespace, verified visually). Values re-tuned 2026-09-19 (proportions
# fix, see the module docstring) after the fidelity-strip sub-axes was
# removed: the left margin now only needs to hold a HORIZONTAL (not
# rotated) row label, and top/bottom no longer hold a suptitle/caption.
_GRID_LEFT = 0.15
_GRID_RIGHT = 0.99
_GRID_TOP = 0.91
_GRID_BOTTOM = 0.02
_GRID_HSPACE = 0.10
_GRID_WSPACE = 0.06
# Row (dataset) label: wrapped to this many characters per line so a
# horizontal, non-rotated label fits within the left margin above
# regardless of dataset name length (this is what fixed the 2026-09-19 bug
# where "Hierarchical clusters", rotated 90 degrees, was taller than its
# own row and got clipped by the canvas edge - see the module docstring).
_ROW_LABEL_WRAP_CHARS = 12
_ROW_LABEL_X = 0.01


def _class_color_map(labels: np.ndarray) -> dict:
    """Color per class: Okabe-Ito (Okabe & Ito, 2008) for <= 8 classes,
    otherwise matplotlib 'tab20' (see the module docstring for the
    rationale behind the deviation from the "Tol discrete rainbow" spec)."""
    classes = sorted(np.unique(labels).tolist())
    if len(classes) <= len(OKABE_ITO):
        palette = OKABE_ITO
    else:
        cmap = plt.get_cmap("tab20")
        palette = [cmap(i % 20) for i in range(len(classes))]
    return {c: palette[i % len(palette)] for i, c in enumerate(classes)}


def _covariance_ellipse(points: np.ndarray, n_std: float):
    """Return (center, width, height, angle_deg) of the covariance ellipse
    of the points (n_std multiples of the standard deviation along the
    principal axes), or None if < 3 points or a degenerate covariance."""
    if points.shape[0] < 3:
        return None
    center = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    if not np.all(np.isfinite(cov)):
        return None
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    width, height = 2.0 * n_std * np.sqrt(eigvals)
    angle = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))
    return center, float(width), float(height), angle


def _local_stress_residual(D: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """r_i = sum_j (D_ij - d_ij)^2 / sum_j D_ij^2 (w_ij=1, the K10 task spec
    formula, d without optimal scaling). NaN for points with a zero sum of
    D_ij^2 (an isolated/degenerate point - never fabricated)."""
    d = squareform(pdist(Y))
    num = np.sum((D - d) ** 2, axis=1)
    den = np.sum(D ** 2, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(den > 0, num / den, np.nan)
    return r


def _corr_to_fidelity(rho: float) -> float:
    """Maps a Spearman rank correlation in [-1, 1] ("higher = better
    preservation") to a [0, 1] bar length ("longer = better"), via
    (rho + 1) / 2. NaN in, NaN out (never fabricated)."""
    if not np.isfinite(rho):
        return float("nan")
    return float(np.clip((rho + 1.0) / 2.0, 0.0, 1.0))


def _stress_to_fidelity(stress: float, stress_max: float) -> float:
    """Maps stress_scale_invariant (an unbounded cost, "lower = better") to
    a [0, 1] bar length ("longer = better"), via 1 - stress / stress_max,
    clipped to [0, 1] at stress_max (config `fig_faithful_map.stress_bar_max`
    - any stress above the cap just shows a zero-length bar, the raw value
    printed next to it still shows the true number). NaN in, NaN out."""
    if not np.isfinite(stress) or stress_max <= 0:
        return float("nan")
    return float(np.clip(1.0 - stress / stress_max, 0.0, 1.0))


def _panel_metrics_text(centroid_rho: float, spread_rho: float, stress: float) -> str:
    """Compact one-line replacement (2026-09-19 proportions fix, see the
    module docstring) for the former 3-bar fidelity strip: the same three
    raw values (centroid rank corr. / spread rank corr. / scale-invariant
    stress), in that fixed order, as "c/s/e" (never fabricated - a
    missing/NaN metric prints "n/a" in its slot, exactly like the removed
    bars used to)."""
    def fmt(v: float) -> str:
        return f"{v:.2f}" if np.isfinite(v) else "n/a"
    return f"{fmt(centroid_rho)}/{fmt(spread_rho)}/{fmt(stress)}"


def _load_input_distance(dataset_name: str, n_max_overrides: dict, n_max_default: int) -> tuple[np.ndarray, np.ndarray | None]:
    """Load + subsample a dataset EXACTLY like E1 (SUBSAMPLE_SEED) and
    return (D, y) - D is the (n x n) Euclidean distance matrix of X (all
    K10 datasets are kind='vector')."""
    from src.datasets.registry import load_dataset
    from src.datasets.subsample import subsample_dataset
    from src.experiments.exp1_dr_benchmark import SUBSAMPLE_SEED

    ds = load_dataset(dataset_name)
    n_max = int(n_max_overrides.get(dataset_name, n_max_default))
    ds = subsample_dataset(ds, n_max=n_max, random_state=SUBSAMPLE_SEED)
    if ds.kind != "vector":
        raise ValueError(f"fig_faithful_map expects kind='vector' for dataset '{dataset_name}', got '{ds.kind}'.")
    D = squareform(pdist(np.asarray(ds.X, dtype=np.float64)))
    return D, ds.y


def main() -> None:
    import argparse

    from src.experiments.config_experiments import resolve_experiment_config

    parser = argparse.ArgumentParser(description="K10: faithful map hero figure (datasets x methods, E1 embeddings).")
    add_quick_arg(parser)
    args = parser.parse_args()
    mode = parse_fig_mode(args)

    fm_cfg = load_experiments_config()["fig_faithful_map"]
    datasets: list[str] = fm_cfg["datasets"]
    methods: list[str] = fm_cfg["methods"]
    # Column/row labels come from the shared display_labels.method/dataset
    # section (src.figures.fig_common.display_label) - see the config
    # comment in fig_faithful_map (2026-09-17 merge).
    method_labels: dict[str, str] = {m: display_label(m, "method") for m in methods}
    dataset_labels: dict[str, str] = {d: display_label(d, "dataset") for d in datasets}
    seed = int(fm_cfg["seed"])
    ellipse_n_std = float(fm_cfg["ellipse_n_std"])
    csv_max_points = int(fm_cfg["csv_max_points_per_panel"])
    csv_seed = int(fm_cfg["csv_subsample_seed"])
    stress_bar_max = float(fm_cfg["stress_bar_max"])

    EXPERIMENT_NAME = resolve_experiment_name(BASE_EXPERIMENT_NAME, mode)
    e1_df = require_experiment_csv(BASE_EXPERIMENT_NAME, mode)
    e1_ok = e1_df[(e1_df["status"] == "ok") & (e1_df["seed"] == seed)]

    geometry_df = require_experiment_csv("exp1_cluster_geometry", mode, how_to_generate=f"venv\\python.exe -m src.experiments.exp1_cluster_geometry --{mode}")
    geometry_ok = geometry_df[(geometry_df["status"] == "ok") & (geometry_df["seed"] == seed)]

    e1_cfg = resolve_experiment_config("exp1_dr_benchmark", mode)
    n_max_overrides = e1_cfg.get("n_max_overrides", {})
    n_max_default = int(e1_cfg["n_max"])

    n_rows, n_cols = len(datasets), len(methods)
    panel_width_in = WIDTH_FULL_WIDTH_IN / n_cols
    # 2026-09-19 proportions fix (see the module docstring): each dataset
    # row is now a SINGLE scatter panel (no fidelity-strip sub-axes), sized
    # to keep panels square (honest distance comparison - stretching panels
    # non-uniformly would visually misrepresent the very metric fidelity
    # this figure demonstrates). The 1.14 factor is the measured (not
    # guessed) remaining top-margin overhead for one-line-or-wrapped column
    # titles; see check_axes_area.py / check_font_sizes.py for the
    # after-the-fact verification that this keeps the figure under both the
    # font floor and the main-text height cap.
    fig = plt.figure(figsize=(WIDTH_FULL_WIDTH_IN, panel_width_in * n_rows * 1.06))
    outer_gs = fig.add_gridspec(
        n_rows, n_cols, left=_GRID_LEFT, right=_GRID_RIGHT, top=_GRID_TOP, bottom=_GRID_BOTTOM,
        hspace=_GRID_HSPACE, wspace=_GRID_WSPACE,
    )

    rng_csv = np.random.default_rng(csv_seed)
    csv_rows = []
    missing_panels = []

    for i, dataset_name in enumerate(datasets):
        D_in, y = _load_input_distance(dataset_name, n_max_overrides, n_max_default)
        color_map = _class_color_map(y) if y is not None else None

        for j, method_name in enumerate(methods):
            ax = fig.add_subplot(outer_gs[i, j])
            try:
                Y = load_embedding(RunKey(EXPERIMENT_NAME, dataset_name, method_name, seed))
            except FileNotFoundError:
                ax.axis("off")
                missing_panels.append((dataset_name, method_name))
                continue
            if Y.shape[0] != D_in.shape[0]:
                ax.axis("off")
                missing_panels.append((dataset_name, method_name))
                continue

            residual = _local_stress_residual(D_in, Y)
            finite_res = residual[np.isfinite(residual)]
            if finite_res.size > 0:
                r_lo, r_hi = np.percentile(finite_res, [2, 98])
                r_hi = max(r_hi, r_lo + 1e-12)
                alpha_pt = np.clip((r_hi - np.nan_to_num(residual, nan=r_hi)) / (r_hi - r_lo), 0.15, 1.0)
            else:
                alpha_pt = np.full(Y.shape[0], 1.0)

            if y is not None and color_map is not None:
                for cls, color in color_map.items():
                    mask = y == cls
                    if not mask.any():
                        continue
                    ax.scatter(Y[mask, 0], Y[mask, 1], s=4, c=[color], alpha=alpha_pt[mask], rasterized=True, linewidths=0, zorder=2)
                    ellipse_info = _covariance_ellipse(Y[mask], ellipse_n_std)
                    if ellipse_info is not None:
                        center, width, height, angle = ellipse_info
                        ax.add_patch(Ellipse(center, width, height, angle=angle, fill=False, edgecolor=color, linewidth=0.8, zorder=3))
            else:
                ax.scatter(Y[:, 0], Y[:, 1], s=4, c=OKABE_ITO[0], alpha=alpha_pt, rasterized=True, linewidths=0, zorder=2)

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
            if i == 0:
                # 2026-09-19 (font-size fix): titles longer than 12 chars are
                # wrapped onto 2 lines at " (" - at LABEL_FONT_PT on the
                # printed \textwidth (372pt / 5 columns = 74pt/column),
                # "Sammon (alpha=1)" and "alpha-Sammon (tuned)" collided as
                # one-line titles in adjacent columns.
                title = method_labels[method_name]
                if len(title) > 12 and " (" in title:
                    title = title.replace(" (", "\n(", 1)
                ax.set_title(title, fontsize=LABEL_FONT_PT)
            if j == 0:
                # 2026-09-19 proportions fix (see the module docstring):
                # horizontal, wrapped, MEASURED-position row label instead
                # of a rotated ax.set_ylabel (which clipped for long names).
                pos = ax.get_position()
                fig.text(
                    _ROW_LABEL_X, (pos.y0 + pos.y1) / 2.0,
                    textwrap.fill(dataset_labels[dataset_name], width=_ROW_LABEL_WRAP_CHARS, break_long_words=False),
                    ha="left", va="center", fontsize=LABEL_FONT_PT, linespacing=1.05,
                )

            row_geom = geometry_ok[(geometry_ok["dataset"] == dataset_name) & (geometry_ok["method"] == method_name)]
            row_e1 = e1_ok[(e1_ok["dataset"] == dataset_name) & (e1_ok["method"] == method_name)]
            cds = float(row_geom["centroid_dist_spearman"].iloc[0]) if not row_geom.empty else float("nan")
            css = float(row_geom["class_spread_spearman"].iloc[0]) if not row_geom.empty else float("nan")
            stress = float(row_e1["stress_scale_invariant"].iloc[0]) if not row_e1.empty else float("nan")
            # 2026-09-19 proportions fix (see the module docstring): the
            # 3-bar fidelity strip (its own sub-axes) is replaced by a
            # single compact in-panel line - no extra axes, so the scatter
            # keeps the whole panel.
            ax.text(
                0.03, 0.03, _panel_metrics_text(cds, css, stress),
                transform=ax.transAxes, ha="left", va="bottom", fontsize=ANNOTATION_FONT_PT, zorder=4,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", boxstyle="round,pad=0.15"),
            )

            n_points = Y.shape[0]
            take = min(csv_max_points, n_points)
            idx_csv = np.sort(rng_csv.choice(n_points, size=take, replace=False)) if take < n_points else np.arange(n_points)
            for k in idx_csv:
                csv_rows.append({
                    "dataset": dataset_name, "dataset_label": dataset_labels[dataset_name],
                    "method": method_name, "method_label": method_labels[method_name],
                    "seed": seed, "point_index": int(k),
                    "y0": float(Y[k, 0]), "y1": float(Y[k, 1]),
                    "label": (y[k] if y is not None else ""),
                    "local_stress_residual": float(residual[k]) if np.isfinite(residual[k]) else "",
                    "centroid_dist_spearman": cds, "class_spread_spearman": css, "stress_scale_invariant": stress,
                    "centroid_fidelity": _corr_to_fidelity(cds), "spread_fidelity": _corr_to_fidelity(css),
                    "stress_fidelity": _stress_to_fidelity(stress, stress_bar_max),
                })

    # 2026-09-19 proportions fix (see the module docstring): the in-image
    # suptitle and the multi-line explanatory caption are both removed -
    # the LaTeX \caption (clanek_en/sections/05_vysledky.tex) already
    # carries the figure title and now also explains the in-panel
    # "centroid/spread/stress" triplet, freeing this vertical space for the
    # panel grid itself.
    save_figure(fig, FIG_NAME)
    save_csv_alongside(pd.DataFrame(csv_rows), FIG_NAME)

    if missing_panels:
        print(f"{FIG_NAME}: WARNING, missing panels (embedding not found/shape mismatch): {missing_panels}")
    print(f"{FIG_NAME}: {n_rows} datasets x {n_cols} methods, {len(csv_rows)} rows in CSV.")


if __name__ == "__main__":
    main()
