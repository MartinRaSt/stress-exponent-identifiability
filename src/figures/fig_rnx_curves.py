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
Figure 14c - R_NX(K) curves with AUC in the legend: for a single dataset
(E1) and all available methods (seed=0), using saved embeddings +
`src.sammon.metrics._coranking_matrix`/`_rnx_curve` (internal functions,
only READ - src/sammon/ is managed concurrently by another agent).

Run: venv\\python.exe -m src.figures.fig_rnx_curves [--quick] [--dataset NAME]
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
from src.figures.fig_common import OKABE_ITO, WIDTH_SINGLE_COL_IN, add_quick_arg, parse_fig_mode, require_experiment_csv, save_csv_alongside, save_figure

FIG_NAME = "fig_rnx_curves"
BASE_EXPERIMENT_NAME = "exp1_dr_benchmark"


def main() -> None:
    import argparse

    from src.datasets.registry import load_dataset
    from src.datasets.subsample import subsample_dataset
    from src.methods.common import to_distance_matrix
    from src.sammon.metrics import _coranking_matrix, _neighbor_ranks, _rnx_curve

    parser = argparse.ArgumentParser(description="R_NX(K) curves for all methods on a single dataset (E1).")
    add_quick_arg(parser)
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()
    mode = parse_fig_mode(args)
    EXPERIMENT_NAME = resolve_experiment_name(BASE_EXPERIMENT_NAME, mode)

    df = require_experiment_csv(BASE_EXPERIMENT_NAME, mode)
    ok = df[df["status"] == "ok"]
    seed = int(ok["seed"].min())
    ok_seed = ok[ok["seed"] == seed]

    dataset_name = args.dataset if args.dataset else sorted(ok_seed["dataset"].unique())[0]
    sub = ok_seed[ok_seed["dataset"] == dataset_name]
    if sub.empty:
        raise ValueError(f"No successful run for dataset='{dataset_name}', seed={seed}.")

    ds = load_dataset(dataset_name)

    fig, ax = plt.subplots(figsize=(WIDTH_SINGLE_COL_IN, WIDTH_SINGLE_COL_IN * 0.9))
    csv_rows = []
    for i, (_, row) in enumerate(sub.iterrows()):
        method_name = row["method"]
        try:
            Y = load_embedding(RunKey(EXPERIMENT_NAME, dataset_name, method_name, seed))
        except FileNotFoundError:
            continue
        ds_sub = ds if ds.n_samples == Y.shape[0] else subsample_dataset(ds, n_max=Y.shape[0], random_state=42)
        D = to_distance_matrix(ds_sub.X, "vector")
        d = to_distance_matrix(Y, "vector")
        n = D.shape[0]
        rank_orig, _ = _neighbor_ranks(D)
        rank_emb, _ = _neighbor_ranks(d)
        Q = _coranking_matrix(rank_orig, rank_emb, n)
        rnx = _rnx_curve(Q, n)
        ks = np.arange(1, len(rnx) + 1)
        auc = float(row["auc_rnx"])
        ax.plot(ks, rnx, label=f"{method_name} (AUC={auc:.3f})", color=OKABE_ITO[i % len(OKABE_ITO)], linewidth=1.0)
        for k, r in zip(ks, rnx):
            csv_rows.append({"dataset": dataset_name, "method": method_name, "seed": seed, "K": int(k), "R_NX": r})

    ax.set_xscale("log")
    ax.set_xlabel("K")
    ax.set_ylabel("R_NX(K)")
    ax.set_title(f"Co-ranking R_NX(K): {dataset_name} (seed={seed})", fontsize=8)
    ax.legend(fontsize=5.5, ncol=1)
    fig.tight_layout()
    save_figure(fig, FIG_NAME)
    save_csv_alongside(pd.DataFrame(csv_rows), FIG_NAME)
    print(f"{FIG_NAME}: dataset={dataset_name}, {sub.shape[0]} methods.")


if __name__ == "__main__":
    main()
