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
Figure 14b - "before/after": classic Sammon (sammon_classic) vs. the
proposed alpha-Sammon (sammon_alpha_smacof, alpha=1) on the same dataset
and seed. Top row: the embedding colored by per-point error (RMS of
(D_ij-d_ij) over the point's neighbors). Bottom row: Shepard diagrams
(hexbin density of original vs. projected distances) with the diagonal as
a reference.

Run: venv\\python.exe -m src.figures.fig_before_after [--quick] [--dataset NAME]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.checkpoint import RunKey, load_embedding
from src.experiments.exp_common import resolve_experiment_name
from src.figures.fig_common import WIDTH_FULL_WIDTH_IN, add_quick_arg, parse_fig_mode, require_experiment_csv, save_csv_alongside, save_figure

FIG_NAME = "fig_before_after"
BASE_EXPERIMENT_NAME = "exp1_dr_benchmark"
METHOD_CLASSIC = "sammon_classic"
METHOD_ALPHA = "sammon_alpha_smacof"


def _per_point_error(D: np.ndarray, d: np.ndarray) -> np.ndarray:
    n = D.shape[0]
    mask = ~np.eye(n, dtype=bool)
    sq = (D - d) ** 2
    return np.sqrt((sq * mask).sum(axis=1) / mask.sum(axis=1))


def main() -> None:
    import argparse

    from src.datasets.registry import load_dataset
    from src.datasets.subsample import subsample_dataset
    from src.methods.common import to_distance_matrix

    parser = argparse.ArgumentParser(description="Before/after: sammon_classic vs. sammon_alpha_smacof.")
    add_quick_arg(parser)
    parser.add_argument("--dataset", type=str, default=None, help="a specific dataset (otherwise the first common to both methods)")
    args = parser.parse_args()
    mode = parse_fig_mode(args)
    EXPERIMENT_NAME = resolve_experiment_name(BASE_EXPERIMENT_NAME, mode)

    df = require_experiment_csv(BASE_EXPERIMENT_NAME, mode)
    ok = df[df["status"] == "ok"]
    seed = int(ok["seed"].min())
    ok_seed = ok[ok["seed"] == seed]

    common_datasets = sorted(
        set(ok_seed[ok_seed["method"] == METHOD_CLASSIC]["dataset"]) & set(ok_seed[ok_seed["method"] == METHOD_ALPHA]["dataset"])
    )
    if not common_datasets:
        raise ValueError(
            f"No dataset has a successful run of both methods ('{METHOD_CLASSIC}', '{METHOD_ALPHA}') for seed={seed} "
            f"in {EXPERIMENT_NAME}_results.csv - the figure cannot be plotted."
        )
    dataset_name = args.dataset if args.dataset else common_datasets[0]
    if dataset_name not in common_datasets:
        raise ValueError(f"Dataset '{dataset_name}' has no successful runs of both methods. Available: {common_datasets}")

    Y_classic = load_embedding(RunKey(EXPERIMENT_NAME, dataset_name, METHOD_CLASSIC, seed))
    Y_alpha = load_embedding(RunKey(EXPERIMENT_NAME, dataset_name, METHOD_ALPHA, seed))

    ds = load_dataset(dataset_name)
    if ds.n_samples != Y_classic.shape[0]:
        ds = subsample_dataset(ds, n_max=Y_classic.shape[0], random_state=42)
    D = to_distance_matrix(ds.X, "vector")
    d_classic = to_distance_matrix(Y_classic, "vector")
    d_alpha = to_distance_matrix(Y_alpha, "vector")

    err_classic = _per_point_error(D, d_classic)
    err_alpha = _per_point_error(D, d_alpha)

    n = D.shape[0]
    triu = np.triu_indices(n, k=1)
    D_flat = D[triu]

    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_FULL_WIDTH_IN, WIDTH_FULL_WIDTH_IN * 0.95))

    vmax = max(err_classic.max(), err_alpha.max())
    for ax, Y, err, title in (
        (axes[0][0], Y_classic, err_classic, f"{METHOD_CLASSIC}"),
        (axes[0][1], Y_alpha, err_alpha, f"{METHOD_ALPHA}"),
    ):
        sc = ax.scatter(Y[:, 0], Y[:, 1], c=err, cmap="viridis", vmin=0, vmax=vmax, s=8, rasterized=True, linewidths=0)
        ax.set_title(f"{title} (per-point RMS error)", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(sc, ax=ax, shrink=0.8)

    for ax, d, title in ((axes[1][0], d_classic, METHOD_CLASSIC), (axes[1][1], d_alpha, METHOD_ALPHA)):
        d_flat = d[triu]
        hb = ax.hexbin(D_flat, d_flat, gridsize=40, cmap="viridis", mincnt=1, rasterized=True)
        lim = max(D_flat.max(), d_flat.max())
        ax.plot([0, lim], [0, lim], color="#D55E00", linewidth=0.8, linestyle="--")  # Okabe-Ito "vermillion" as the reference diagonal
        ax.set_xlabel("D_ij (original)", fontsize=7)
        ax.set_ylabel("d_ij (embedded)", fontsize=7)
        ax.set_title(f"Shepard diagram: {title}", fontsize=8)
        fig.colorbar(hb, ax=ax, shrink=0.8)

    fig.suptitle(f"Before/after: {dataset_name} (seed={seed})", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, FIG_NAME)

    csv_rows = []
    for i in range(n):
        csv_rows.append({
            "dataset": dataset_name, "seed": seed, "point_index": i,
            "y0_classic": Y_classic[i, 0], "y1_classic": Y_classic[i, 1], "error_classic": err_classic[i],
            "y0_alpha": Y_alpha[i, 0], "y1_alpha": Y_alpha[i, 1], "error_alpha": err_alpha[i],
        })
    save_csv_alongside(pd.DataFrame(csv_rows), FIG_NAME)
    print(f"{FIG_NAME}: dataset={dataset_name}, n={n}, mean_err_classic={err_classic.mean():.4f}, mean_err_alpha={err_alpha.mean():.4f}")


if __name__ == "__main__":
    main()
