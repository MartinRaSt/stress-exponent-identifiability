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
Figures 14h1-h3 - temporal trajectories: (h1) a 3D space-time cube (x,y,t),
(h2) small multiples of snapshots with alpha-blended older trails, (h3) a
stability-vs-quality trade-off curve over the lambda grid. Input:
`results/data/exp4_trajectories.csv` (h1,h2) and
`results/data/exp4_temporal_results.csv` (h3).

Run: venv\\python.exe -m src.figures.fig_temporal_trajectories [--quick] [--lambda 0.1]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - registers the 3D projection

from src.figures.fig_common import (
    OKABE_ITO,
    WIDTH_FULL_WIDTH_IN,
    WIDTH_SINGLE_COL_IN,
    add_quick_arg,
    mode_data_dir,
    parse_fig_mode,
    require_csv,
    require_experiment_csv,
    save_csv_alongside,
    save_figure,
)

BASE_EXPERIMENT_NAME = "exp4_temporal"


def _make_space_time_and_small_multiples(traj: pd.DataFrame, dataset: str, lam: float, alpha: float, solver: str) -> None:
    sub = traj[
        (traj["dataset"] == dataset) & (traj["lambda"] == lam) & (traj["alpha"] == alpha) & (traj["solver"] == solver)
    ].sort_values(["t", "node"])
    if sub.empty:
        raise ValueError(
            f"exp4_trajectories.csv contains no rows for dataset={dataset}, lambda={lam}, alpha={alpha}, solver={solver}."
        )
    ts = sorted(sub["t"].unique())

    # h1: 3D space-time cube
    fig1 = plt.figure(figsize=(WIDTH_FULL_WIDTH_IN, WIDTH_FULL_WIDTH_IN * 0.7))
    ax1 = fig1.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("viridis")
    for t in ts:
        st = sub[sub["t"] == t]
        color = cmap(t / max(ts))
        ax1.scatter(st["x"], st["y"], st["t"], s=4, color=color, rasterized=True, linewidths=0)
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("t (snapshot)")
    ax1.set_title(f"Space-time cube: {dataset} (lambda={lam}, alpha={alpha}, {solver})", fontsize=8)
    fig1.tight_layout()
    save_figure(fig1, f"fig_temporal_trajectories_h1_{dataset}_lam{lam}_{solver}")

    # h2: small multiples with alpha-blending of the last `trail` trails
    trail = min(5, len(ts))
    n_cols = min(6, len(ts))
    n_rows = int(np.ceil(len(ts) / n_cols))
    fig2, axes = plt.subplots(n_rows, n_cols, figsize=(WIDTH_FULL_WIDTH_IN, WIDTH_FULL_WIDTH_IN / n_cols * n_rows), squeeze=False)
    xlim = (sub["x"].min(), sub["x"].max())
    ylim = (sub["y"].min(), sub["y"].max())
    for idx, t in enumerate(ts):
        ax = axes[idx // n_cols][idx % n_cols]
        for back in range(trail, -1, -1):
            t_back = t - back
            if t_back not in ts:
                continue
            frame = sub[sub["t"] == t_back]
            alpha_blend = 1.0 - back / (trail + 1)
            ax.scatter(frame["x"], frame["y"], s=3, color=OKABE_ITO[5], alpha=max(0.08, alpha_blend), rasterized=True, linewidths=0)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"t={t}", fontsize=6)
    for idx in range(len(ts), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")
    fig2.suptitle(f"Temporal small multiples with trailing alpha-blend: {dataset} (lambda={lam}, alpha={alpha}, {solver})", fontsize=9)
    fig2.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig2, f"fig_temporal_trajectories_h2_{dataset}_lam{lam}_{solver}")

    save_csv_alongside(sub, f"fig_temporal_trajectories_h1h2_{dataset}_lam{lam}_{solver}")


def _make_tradeoff(results: pd.DataFrame, dataset: str) -> None:
    ok = results[(results["status"] == "ok") & (results["dataset"] == dataset)]
    if ok.empty:
        raise ValueError(f"No successful runs in exp4_temporal_results.csv for dataset={dataset} (trade-off curve).")
    ok = ok.copy()
    extracted = ok["method"].str.extract(r"^lambda([\d.]+)_alpha([\d.]+)_(\w+)$")
    ok["lam"] = extracted[0].astype(float)
    ok["alpha"] = extracted[1].astype(float)
    ok["solver"] = extracted[2]
    agg = ok.groupby(["lam", "alpha", "solver"])[["stab", "qual"]].median().reset_index().sort_values("lam")

    fig, ax = plt.subplots(figsize=(WIDTH_SINGLE_COL_IN, WIDTH_SINGLE_COL_IN * 0.9))
    for i, ((alpha_val, solver_val), grp) in enumerate(agg.groupby(["alpha", "solver"])):
        ax.plot(grp["stab"], grp["qual"], marker="o", markersize=3, linewidth=1.0, color=OKABE_ITO[(1 + i) % len(OKABE_ITO)], label=f"alpha={alpha_val:g}, {solver_val}")
        for _, r in grp.iterrows():
            ax.annotate(f"{r['lam']:g}", (r["stab"], r["qual"]), fontsize=5, xytext=(2, 2), textcoords="offset points")
    baseline = agg[agg["lam"] == 0.0]
    if not baseline.empty:
        ax.scatter(baseline["stab"], baseline["qual"], color=OKABE_ITO[6], marker="*", s=40, zorder=5, label="baseline (lambda=0)")
    ax.set_xlabel("stability (lower = more stable)")
    ax.set_ylabel("quality: median E_alpha^scale-inv")
    ax.set_title(f"Stability-quality trade-off across lambda: {dataset}", fontsize=8)
    ax.legend(fontsize=6)
    fig.tight_layout()
    save_figure(fig, f"fig_temporal_trajectories_h3_tradeoff_{dataset}")
    save_csv_alongside(agg, f"fig_temporal_trajectories_h3_tradeoff_{dataset}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Temporal trajectories: space-time cube, small multiples, trade-off.")
    add_quick_arg(parser)
    parser.add_argument("--dataset", type=str, default=None, help="restrict to a single dataset (default: all present in the data)")
    parser.add_argument("--lambda", dest="lam", type=float, default=None, help="lambda for h1/h2 (default: default_lambda from config.yaml)")
    parser.add_argument("--alpha", type=float, default=None, help="alpha for h1/h2 (default: the first available)")
    parser.add_argument("--solver", type=str, default="smacof", help="TemporalSammon solver for h1/h2 (default 'smacof')")
    args = parser.parse_args()
    mode = parse_fig_mode(args)

    traj_path = mode_data_dir(mode) / "exp4_trajectories.csv"
    traj = require_csv(traj_path, f"venv\\python.exe -m src.experiments.exp4_temporal --{mode}")
    results = require_experiment_csv(BASE_EXPERIMENT_NAME, mode)

    if args.dataset is not None:
        datasets = [args.dataset]
    else:
        # without --dataset, EVERY dataset present in exp4_trajectories.csv
        # is plotted (intersected with exp4_temporal_results.csv, so h3
        # trade-off has data) - each dataset gets its own figure files
        # (the filename includes the dataset)
        datasets = sorted(set(traj["dataset"].unique()) & set(results["dataset"].unique()))
        if not datasets:
            raise ValueError("exp4_trajectories.csv and exp4_temporal_results.csv have no dataset in common.")

    for dataset in datasets:
        traj_ds = traj[traj["dataset"] == dataset]
        if traj_ds.empty:
            raise ValueError(f"exp4_trajectories.csv contains no rows for dataset={dataset}.")

        lam = args.lam
        if lam is None:
            from src.common.config import load_config

            default_lam = load_config()["sammon"]["temporal"]["default_lambda"]
            lam = default_lam if default_lam in traj_ds["lambda"].unique() else float(traj_ds["lambda"].unique()[0])
        alpha_val = args.alpha if args.alpha is not None else float(traj_ds["alpha"].unique()[0])
        solver_val = args.solver if args.solver in traj_ds["solver"].unique() else str(traj_ds["solver"].unique()[0])

        _make_space_time_and_small_multiples(traj, dataset, lam, alpha_val, solver_val)
        _make_tradeoff(results, dataset)
        print(f"fig_temporal_trajectories: dataset={dataset}, h1/h2 for lambda={lam}, alpha={alpha_val}; h3 trade-off done.")


if __name__ == "__main__":
    main()
