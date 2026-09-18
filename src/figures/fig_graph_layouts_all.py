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
K11 (documentation/2026-09-12_plan_smeru_clanku.md): SUPPLEMENT version
(the original figure 14i, renamed from `fig_graph_layouts.py` - the main
article text now uses the focused `fig_graph_layouts.py`, see its
docstring). A grid of ALL graph datasets (rows) x ALL methods (columns)
present in `results/data/exp3_graph_layout_results.csv` (E3) with thin
edges and nodes colored by community/class - after K8 the grid can be
large (12 graphs x up to 14 methods), so it is kept for the supplement only.

Run: venv\\python.exe -m src.figures.fig_graph_layouts_all [--quick|--full|--smoke]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from src.common.checkpoint import RunKey, load_embedding
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

FIG_NAME = "fig_graph_layouts_all"
BASE_EXPERIMENT_NAME = "exp3_graph_layout"


def main() -> None:
    import argparse

    from src.datasets.registry import load_dataset

    parser = argparse.ArgumentParser(description="Graph layouts with edges and communities (E3).")
    add_quick_arg(parser)
    parser.add_argument("--distance-metric", type=str, default="shortest_path")
    args = parser.parse_args()
    mode = parse_fig_mode(args)
    EXPERIMENT_NAME = resolve_experiment_name(BASE_EXPERIMENT_NAME, mode)

    df = require_experiment_csv(BASE_EXPERIMENT_NAME, mode)
    ok = df[df["status"] == "ok"]
    seed = int(ok["seed"].min())
    ok_seed = ok[ok["seed"] == seed]

    # native (graph directly) + distance-based with the requested metric, one panel per (dataset, method)
    native = ok_seed[~ok_seed["dataset"].str.contains("__", regex=False)]
    dist = ok_seed[ok_seed["dataset"].str.contains(f"__{args.distance_metric}", regex=False)]
    panels_df = pd.concat([native, dist], ignore_index=True)
    if panels_df.empty:
        raise ValueError(f"No successful run for E3 (distance_metric={args.distance_metric}, seed={seed}).")

    base_graphs = sorted(panels_df["base_graph"].unique())
    methods = sorted(panels_df["method"].unique())

    fig, axes = plt.subplots(
        len(base_graphs), len(methods),
        figsize=(WIDTH_FULL_WIDTH_IN, WIDTH_FULL_WIDTH_IN / max(len(methods), 1) * len(base_graphs)),
        squeeze=False,
    )
    csv_rows = []

    for i, base_graph in enumerate(base_graphs):
        ds = load_dataset(base_graph)
        nodes = list(ds.graph.nodes())
        node_idx = {n: k for k, n in enumerate(nodes)}
        edges = np.asarray([(node_idx[u], node_idx[v]) for u, v in ds.graph.edges() if u != v], dtype=np.int64)
        y_labels = ds.y
        if y_labels is not None:
            # node color by class index in the Okabe-Ito palette (a single
            # scatter per panel instead of one per class)
            _, label_inverse = np.unique(y_labels, return_inverse=True)
            node_colors = np.asarray(OKABE_ITO)[label_inverse % len(OKABE_ITO)]
        else:
            node_colors = OKABE_ITO[0]

        for j, method_name in enumerate(methods):
            ax = axes[i][j]
            row = panels_df[(panels_df["base_graph"] == base_graph) & (panels_df["method"] == method_name)]
            if row.empty:
                ax.axis("off")
                continue
            dataset_label = row["dataset"].iloc[0]
            try:
                Y = load_embedding(RunKey(EXPERIMENT_NAME, dataset_label, method_name, seed))
            except FileNotFoundError:
                ax.axis("off")
                continue
            if Y.shape[0] != len(nodes):
                ax.axis("off")
                continue

            # 2026-09-13 (documentation/2026-09-13_hardening_behu.md): all
            # edges of a panel as a SINGLE LineCollection rasterized as a
            # whole (RASTER_DPI from config figures.raster_dpi) - the
            # original ax.plot per edge with rasterized=True created tens
            # of thousands of separate raster images per panel (pgp: 24k
            # edges) -> ~810 s per figure
            if len(edges) > 0:
                segments = Y[edges]  # (m, 2, 2): [edge, endpoint, coordinates]
                lc = LineCollection(segments, colors="#888888", linewidths=0.25, alpha=0.5, zorder=1)
                lc.set_rasterized(True)
                ax.add_collection(lc)
            ax.scatter(Y[:, 0], Y[:, 1], s=6, c=node_colors, zorder=2, rasterized=True, linewidths=0)
            ax.autoscale_view()

            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
            if i == 0:
                ax.set_title(display_label(method_name, "method"), fontsize=7)
            if j == 0:
                ax.set_ylabel(display_label(base_graph, "dataset"), fontsize=7)

            csv_rows.append(pd.DataFrame({
                "base_graph": base_graph, "method": method_name, "seed": seed,
                "node_index": np.arange(Y.shape[0]), "y0": Y[:, 0], "y1": Y[:, 1],
                "label": y_labels if y_labels is not None else "",
            }))

    fig.suptitle(
        f"Graph layouts: dataset (row) x method (column), "
        f"distance={display_label(args.distance_metric, 'distance_metric')}, seed={seed}",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, FIG_NAME)
    save_csv_alongside(pd.concat(csv_rows, ignore_index=True), FIG_NAME)
    print(f"{FIG_NAME}: {len(base_graphs)} graphs x {len(methods)} methods.")


if __name__ == "__main__":
    main()
