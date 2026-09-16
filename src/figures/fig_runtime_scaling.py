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
Figure 14f - log-log runtime vs. n: `results/data/exp2_scaling_results.csv`
(E2 part b), one curve per (solver, device), CPU/GPU speedup as an
annotation where both are available for the same n.

Run: venv\\python.exe -m src.figures.fig_runtime_scaling [--quick]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import pandas as pd

from src.experiments.exp_common import resolve_experiment_name
from src.figures.fig_common import OKABE_ITO, WIDTH_SINGLE_COL_IN, add_quick_arg, parse_fig_mode, require_experiment_csv, save_csv_alongside, save_figure

FIG_NAME = "fig_runtime_scaling"
BASE_EXPERIMENT_NAME = "exp2_scaling"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Log-log runtime vs. n for the E2b solvers.")
    add_quick_arg(parser)
    args = parser.parse_args()
    mode = parse_fig_mode(args)
    EXPERIMENT_NAME = resolve_experiment_name(BASE_EXPERIMENT_NAME, mode)

    df = require_experiment_csv(BASE_EXPERIMENT_NAME, mode, f"venv\\python.exe -m src.experiments.exp2_solver_scaling --{mode}")
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        raise ValueError(f"No successful runs in {EXPERIMENT_NAME}_results.csv.")

    ok["n"] = ok["dataset"].str.extract(r"_n(\d+)$").astype(int)
    med = ok.groupby(["n", "method"])["wall_time_sec"].median().reset_index()

    fig, ax = plt.subplots(figsize=(WIDTH_SINGLE_COL_IN, WIDTH_SINGLE_COL_IN * 0.85))
    for i, method_name in enumerate(sorted(med["method"].unique())):
        s = med[med["method"] == method_name].sort_values("n")
        ax.plot(s["n"], s["wall_time_sec"], marker="o", markersize=3, linewidth=1.0, label=method_name, color=OKABE_ITO[i % len(OKABE_ITO)])

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("wall-clock time [s] (median)")
    ax.legend(fontsize=5.5, ncol=1)
    ax.set_title("Runtime scaling (E2b, gaussian_clusters)", fontsize=8)

    fig.tight_layout()
    save_figure(fig, FIG_NAME)
    save_csv_alongside(med, FIG_NAME)
    print(f"{FIG_NAME}: {med['method'].nunique()} solvers/devices, n in {sorted(med['n'].unique())}.")


if __name__ == "__main__":
    main()
