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

Below each panel: centroid_dist_spearman, class_spread_spearman
(results/data/exp1_cluster_geometry_results.csv, K2) and
stress_scale_invariant (results/data/exp1_dr_benchmark_results.csv), all
for the same (dataset, method, seed).

Optional overlay: point transparency = local stress residual
r_i = sum_j (D_ij - d_ij)^2 / sum_j D_ij^2 (w_ij=1, i.e. alpha=0
weighting, D = the input distance matrix on the same subsample as E1, d =
the Euclidean distance in the output Y WITHOUT optimal scaling s* - see
the K10 task spec formula).

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
    seed = int(fm_cfg["seed"])
    ellipse_n_std = float(fm_cfg["ellipse_n_std"])
    csv_max_points = int(fm_cfg["csv_max_points_per_panel"])
    csv_seed = int(fm_cfg["csv_subsample_seed"])

    EXPERIMENT_NAME = resolve_experiment_name(BASE_EXPERIMENT_NAME, mode)
    e1_df = require_experiment_csv(BASE_EXPERIMENT_NAME, mode)
    e1_ok = e1_df[(e1_df["status"] == "ok") & (e1_df["seed"] == seed)]

    geometry_df = require_experiment_csv("exp1_cluster_geometry", mode, how_to_generate=f"venv\\python.exe -m src.experiments.exp1_cluster_geometry --{mode}")
    geometry_ok = geometry_df[(geometry_df["status"] == "ok") & (geometry_df["seed"] == seed)]

    e1_cfg = resolve_experiment_config("exp1_dr_benchmark", mode)
    n_max_overrides = e1_cfg.get("n_max_overrides", {})
    n_max_default = int(e1_cfg["n_max"])

    n_rows, n_cols = len(datasets), len(methods)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(WIDTH_FULL_WIDTH_IN, WIDTH_FULL_WIDTH_IN / n_cols * n_rows * 1.18), squeeze=False)

    rng_csv = np.random.default_rng(csv_seed)
    csv_rows = []
    missing_panels = []

    for i, dataset_name in enumerate(datasets):
        D_in, y = _load_input_distance(dataset_name, n_max_overrides, n_max_default)
        color_map = _class_color_map(y) if y is not None else None

        for j, method_name in enumerate(methods):
            ax = axes[i][j]
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
                ax.set_title(method_name, fontsize=7)
            if j == 0:
                ax.set_ylabel(dataset_name, fontsize=7)

            row_geom = geometry_ok[(geometry_ok["dataset"] == dataset_name) & (geometry_ok["method"] == method_name)]
            row_e1 = e1_ok[(e1_ok["dataset"] == dataset_name) & (e1_ok["method"] == method_name)]
            cds = float(row_geom["centroid_dist_spearman"].iloc[0]) if not row_geom.empty else float("nan")
            css = float(row_geom["class_spread_spearman"].iloc[0]) if not row_geom.empty else float("nan")
            stress = float(row_e1["stress_scale_invariant"].iloc[0]) if not row_e1.empty else float("nan")
            ax.text(
                0.5, -0.12, f"centroid={cds:.2f}  spread={css:.2f}  stress={stress:.3f}",
                transform=ax.transAxes, ha="center", va="top", fontsize=5,
            )

            n_points = Y.shape[0]
            take = min(csv_max_points, n_points)
            idx_csv = np.sort(rng_csv.choice(n_points, size=take, replace=False)) if take < n_points else np.arange(n_points)
            for k in idx_csv:
                csv_rows.append({
                    "dataset": dataset_name, "method": method_name, "seed": seed, "point_index": int(k),
                    "y0": float(Y[k, 0]), "y1": float(Y[k, 1]),
                    "label": (y[k] if y is not None else ""),
                    "local_stress_residual": float(residual[k]) if np.isfinite(residual[k]) else "",
                    "centroid_dist_spearman": cds, "class_spread_spearman": css, "stress_scale_invariant": stress,
                })

    fig.suptitle("Faithful map: class geometry preservation across methods (E1, seed=%d)" % seed, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, FIG_NAME)
    save_csv_alongside(pd.DataFrame(csv_rows), FIG_NAME)

    if missing_panels:
        print(f"{FIG_NAME}: WARNING, missing panels (embedding not found/shape mismatch): {missing_panels}")
    print(f"{FIG_NAME}: {n_rows} datasets x {n_cols} methods, {len(csv_rows)} rows in CSV.")


if __name__ == "__main__":
    main()
