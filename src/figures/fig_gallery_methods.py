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
Figure 14a - a projection gallery: a grid (rows=datasets, columns=methods)
from `results/data/exp1_dr_benchmark_results.csv` + saved embeddings,
Okabe-Ito class colors, a metric (scale-invariant stress) in the corner of
each panel.

Run: venv\\python.exe -m src.figures.fig_gallery_methods [--quick]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.checkpoint import RunKey, load_embedding
from src.common.config import load_config
from src.experiments.exp_common import resolve_experiment_name
from src.figures.fig_common import (
    OKABE_ITO,
    WIDTH_FULL_WIDTH_IN,
    add_quick_arg,
    display_label,
    parse_fig_mode,
    require_experiment_csv,
    save_csv_alongside,
    save_figure,
)

FIG_NAME = "fig_gallery_methods"
BASE_EXPERIMENT_NAME = "exp1_dr_benchmark"
SUBSAMPLE_SEED = 42  # matches src/experiments/exp1_dr_benchmark.py


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Projection gallery (E1: dataset x method).")
    add_quick_arg(parser)
    args = parser.parse_args()
    mode = parse_fig_mode(args)
    EXPERIMENT_NAME = resolve_experiment_name(BASE_EXPERIMENT_NAME, mode)

    df = require_experiment_csv(BASE_EXPERIMENT_NAME, mode)
    ok = df[df["status"] == "ok"]
    if ok.empty:
        raise ValueError(f"No successful runs in {EXPERIMENT_NAME}_results.csv - nothing to plot.")

    gallery_cfg = load_config()["figures"]["gallery_methods"]
    csv_max_points = int(gallery_cfg["csv_max_points_per_panel"])
    csv_subsample_seed = int(gallery_cfg["csv_subsample_seed"])

    seed = int(ok["seed"].min())
    ok_seed = ok[ok["seed"] == seed]

    datasets = sorted(ok_seed["dataset"].unique())
    methods = sorted(ok_seed["method"].unique())

    from src.datasets.registry import load_dataset
    from src.datasets.subsample import subsample_dataset

    fig, axes = plt.subplots(
        len(datasets), len(methods),
        figsize=(WIDTH_FULL_WIDTH_IN, WIDTH_FULL_WIDTH_IN / max(len(methods), 1) * len(datasets)),
        squeeze=False,
    )
    csv_rows = []
    rng = np.random.default_rng(csv_subsample_seed)

    for i, dataset_name in enumerate(datasets):
        # 2026-09-13 (documentation/2026-09-13_hardening_behu.md): the
        # dataset is loaded ONCE per row (previously reloaded in every
        # panel, i.e. 16x per dataset including standardization/subsampling)
        # and the subsample labels are cached by n (all methods on a given
        # dataset share the same E1 subsample)
        ds_full = load_dataset(dataset_name)
        labels_by_n: dict[int, np.ndarray | None] = {}

        def _labels_for(n_points: int) -> np.ndarray | None:
            if n_points not in labels_by_n:
                ds = ds_full
                if ds.y is not None and ds.n_samples != n_points:
                    ds = subsample_dataset(ds_full, n_max=n_points, random_state=SUBSAMPLE_SEED)
                labels_by_n[n_points] = ds.y
            return labels_by_n[n_points]

        for j, method_name in enumerate(methods):
            ax = axes[i][j]
            row = ok_seed[(ok_seed["dataset"] == dataset_name) & (ok_seed["method"] == method_name)]
            if row.empty:
                ax.axis("off")
                continue
            key = RunKey(EXPERIMENT_NAME, dataset_name, method_name, seed)
            try:
                Y = load_embedding(key)
            except FileNotFoundError:
                ax.axis("off")
                continue

            y = _labels_for(Y.shape[0])

            if y is not None and len(y) == Y.shape[0]:
                # a single scatter per panel with an array of colors (instead of one per class)
                _, label_inverse = np.unique(y, return_inverse=True)
                colors = np.asarray(OKABE_ITO)[label_inverse % len(OKABE_ITO)]
                ax.scatter(Y[:, 0], Y[:, 1], s=3, c=colors, rasterized=True, linewidths=0)
            else:
                ax.scatter(Y[:, 0], Y[:, 1], s=3, c=OKABE_ITO[0], rasterized=True, linewidths=0)

            stress = float(row["stress_scale_invariant"].iloc[0])
            ax.text(0.02, 0.98, f"E={stress:.3f}", transform=ax.transAxes, fontsize=6, va="top", ha="left")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
            if i == 0:
                ax.set_title(display_label(method_name, "method"), fontsize=7)
            if j == 0:
                ax.set_ylabel(display_label(dataset_name, "dataset"), fontsize=7)

            # the accompanying CSV is only for documentation/reproducibility
            # of the underlying points - with 32 datasets x 16 methods x up
            # to n_max=5000 points, a full export would be tens of MB (see
            # documentation/2026-09-11_kontrola_vysledku_plnych_behu.md
            # section 6); the PDF always plots ALL points (rasterized), the
            # CSV only a deterministic subsample of at most `csv_max_points`
            # per panel (config.yaml figures.gallery_methods) - the full
            # embedding remains available in results/data/embeddings/<experiment>/.
            n_pts = Y.shape[0]
            if n_pts > csv_max_points:
                pt_idx = np.sort(rng.choice(n_pts, size=csv_max_points, replace=False))
            else:
                pt_idx = np.arange(n_pts)
            has_labels = y is not None and len(y) == n_pts
            csv_rows.append(pd.DataFrame({
                "dataset": dataset_name, "method": method_name, "seed": seed, "point_index": pt_idx.astype(int),
                "y0": Y[pt_idx, 0], "y1": Y[pt_idx, 1], "label": np.asarray(y)[pt_idx] if has_labels else "",
                "stress_scale_invariant": stress,
                "n_points_total": n_pts, "n_points_in_csv": min(n_pts, csv_max_points),
            }))

    fig.suptitle("Method gallery: dataset (row) x method (column), seed=%d" % seed, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, FIG_NAME)
    save_csv_alongside(pd.concat(csv_rows, ignore_index=True), FIG_NAME)
    print(f"{FIG_NAME}: {len(datasets)} datasets x {len(methods)} methods saved.")


if __name__ == "__main__":
    main()
