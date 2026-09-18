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

Run: venv\\python.exe -m src.figures.fig_faithful_map [--quick|--full|--smoke]
"""
from __future__ import annotations

import sys
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


# Fidelity-strip style constants (pure styling, not analysis parameters -
# see fig_faithful_map.stress_bar_max in config_experiments.yaml for the one
# value that DOES affect the analysis, the stress normalization cap).
_FIDELITY_BAR_COLOR = "#0072B2"   # OKABE_ITO[5] (blue)
_FIDELITY_TRACK_COLOR = "#d9d9d9"
_FIDELITY_ROW_LABELS = ["Centroid", "Spread", "Stress"]
_FIDELITY_XLIM = (0.0, 1.38)  # room for the raw-value text past the bar end


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


def _draw_fidelity_strip(ax, centroid_rho: float, spread_rho: float, stress: float, stress_max: float, show_row_labels: bool) -> None:
    """Draws the 3-bar "fidelity strip" (Centroid rank corr., Spread rank
    corr., Stress) below one scatter panel: a light-gray full-length track
    (the [0, 1] normalized scale) plus a colored bar of the normalized
    fidelity, with the RAW metric value printed as text past the bar end
    (see `_corr_to_fidelity`/`_stress_to_fidelity` - a missing/NaN metric
    draws only the track and an "n/a" label, never a fabricated bar)."""
    raw_values = [centroid_rho, spread_rho, stress]
    fidelities = [
        _corr_to_fidelity(centroid_rho),
        _corr_to_fidelity(spread_rho),
        _stress_to_fidelity(stress, stress_max),
    ]
    y_positions = [2, 1, 0]
    for y, raw, fidelity in zip(y_positions, raw_values, fidelities):
        ax.barh(y, 1.0, height=0.62, color=_FIDELITY_TRACK_COLOR, linewidth=0, zorder=1)
        if np.isfinite(fidelity):
            ax.barh(y, fidelity, height=0.62, color=_FIDELITY_BAR_COLOR, linewidth=0, zorder=2)
            text = f"{raw:.2f}" if abs(raw) < 10 else f"{raw:.3g}"
        else:
            text = "n/a"
        ax.text(1.05, y, text, va="center", ha="left", fontsize=4.3, zorder=3)

    ax.set_xlim(*_FIDELITY_XLIM)
    ax.set_ylim(-0.65, 2.65)
    ax.set_xticks([])
    if show_row_labels:
        ax.set_yticks(y_positions)
        ax.set_yticklabels(_FIDELITY_ROW_LABELS, fontsize=4.6)
        ax.tick_params(axis="y", length=0, pad=1.5)
    else:
        ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


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
    # each dataset row = a scatter panel (height ratio 3) + its fidelity
    # strip (height ratio 1) - see _draw_fidelity_strip; the 1.42 factor
    # leaves just enough room for column titles, the row-label ylabel and
    # the inter-row spacing (outer_gs hspace below). Explicit margins
    # (left/right/top/bottom) are used INSTEAD of fig.tight_layout(), which
    # does not lay out subgridspec children correctly (large stray
    # whitespace, verified visually - see the task's PDF-inspection
    # requirement).
    fig = plt.figure(figsize=(WIDTH_FULL_WIDTH_IN, panel_width_in * n_rows * 1.15))
    outer_gs = fig.add_gridspec(n_rows, n_cols, left=0.055, right=0.99, top=0.88, bottom=0.06, hspace=0.12, wspace=0.15)

    rng_csv = np.random.default_rng(csv_seed)
    csv_rows = []
    missing_panels = []

    for i, dataset_name in enumerate(datasets):
        D_in, y = _load_input_distance(dataset_name, n_max_overrides, n_max_default)
        color_map = _class_color_map(y) if y is not None else None

        for j, method_name in enumerate(methods):
            inner_gs = outer_gs[i, j].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.12)
            ax = fig.add_subplot(inner_gs[0])
            ax_m = fig.add_subplot(inner_gs[1])
            try:
                Y = load_embedding(RunKey(EXPERIMENT_NAME, dataset_name, method_name, seed))
            except FileNotFoundError:
                ax.axis("off")
                ax_m.axis("off")
                missing_panels.append((dataset_name, method_name))
                continue
            if Y.shape[0] != D_in.shape[0]:
                ax.axis("off")
                ax_m.axis("off")
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
                ax.set_title(method_labels[method_name], fontsize=6.8)
            if j == 0:
                ax.set_ylabel(dataset_labels[dataset_name], fontsize=6.8)

            row_geom = geometry_ok[(geometry_ok["dataset"] == dataset_name) & (geometry_ok["method"] == method_name)]
            row_e1 = e1_ok[(e1_ok["dataset"] == dataset_name) & (e1_ok["method"] == method_name)]
            cds = float(row_geom["centroid_dist_spearman"].iloc[0]) if not row_geom.empty else float("nan")
            css = float(row_geom["class_spread_spearman"].iloc[0]) if not row_geom.empty else float("nan")
            stress = float(row_e1["stress_scale_invariant"].iloc[0]) if not row_e1.empty else float("nan")
            _draw_fidelity_strip(ax_m, cds, css, stress, stress_bar_max, show_row_labels=(j == 0))

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

    fig.suptitle("Faithful map: class geometry preservation across methods (E1, seed=%d)" % seed, fontsize=8, y=0.985)
    fig.text(
        0.5, 0.012,
        "Fidelity bars (Centroid, Spread: Spearman rank corr.; Stress: scale-invariant, capped at "
        f"{stress_bar_max:g}) - longer bar = more faithful; raw value printed next to each bar.",
        ha="center", va="bottom", fontsize=5.6,
    )
    save_figure(fig, FIG_NAME)
    save_csv_alongside(pd.DataFrame(csv_rows), FIG_NAME)

    if missing_panels:
        print(f"{FIG_NAME}: WARNING, missing panels (embedding not found/shape mismatch): {missing_panels}")
    print(f"{FIG_NAME}: {n_rows} datasets x {n_cols} methods, {len(csv_rows)} rows in CSV.")


if __name__ == "__main__":
    main()
