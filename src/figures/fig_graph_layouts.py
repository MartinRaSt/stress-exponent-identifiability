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
K11 (documentation/2026-09-12_plan_smeru_clanku.md) - a focused graph-layout
figure for the MAIN text of the article (a replacement of the original
`fig_graph_layouts.py`, which is renamed to `fig_graph_layouts_all.py` for
the supplement - all 12 methods x all E3 graphs, see its docstring).

Grid: rows = (graph, distance metric) from config `fig_graph_layouts_focus`
(default 2 graphs - polbooks, football - x 2 distances = 4 rows; 'cora'
(K8) is appended as the 5th/6th row pair ONLY if all required E3 embeddings
already exist for it - otherwise it is skipped with a WARNING, see the K11
task spec "after K8, optionally cora if embeddings exist - otherwise skip
with a log"), columns = 5 methods (`sammon_alpha0_smacof`,
`sammon_alpha_auto`, `tsne_auto`, `umap_auto`, `spring` - the last is
graph-native, i.e. independent of the distance metric of a given row, but
shown in both rows for grid consistency). Edges are drawn in gray (alpha
0.3, from the GRAPH, not from the embedding), node color = community/class
(`ds.y`, if present). Below each panel: stress_scale_invariant and auc_rnx
(E3, seed from the first available seed of `distance_metrics` - see `_pick_seed`).

Run: venv\\python.exe -m src.figures.fig_graph_layouts [--quick|--full|--smoke]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.checkpoint import RunKey, load_embedding
from src.experiments.config_experiments import load_experiments_config
from src.experiments.exp_common import resolve_experiment_name
from src.figures.fig_common import (
    OKABE_ITO,
    WIDTH_FULL_WIDTH_IN,
    add_quick_arg,
    parse_fig_mode,
    require_experiment_csv,
    save_csv_alongside,
    save_figure,
)

FIG_NAME = "fig_graph_layouts"
BASE_EXPERIMENT_NAME = "exp3_graph_layout"

# methods with no dependence on the distance metric (computed directly on
# the graph - see exp3_graph_layout.py native_graph_methods)
_NATIVE_METHODS = {"kamada_kawai", "spring", "spectral"}


def _dataset_label(graph: str, method_name: str, distance_metric: str) -> str:
    return graph if method_name in _NATIVE_METHODS else f"{graph}__{distance_metric}"


def _panel_available(df_ok: pd.DataFrame, exp_name: str, graph: str, method_name: str, distance_metric: str, seed: int, n_nodes: int) -> bool:
    """True if a successful row exists in the CSV AND a saved embedding
    exists for the given panel AND the number of embedding rows matches
    the number of graph nodes (`n_nodes` = number of nodes in `ds.graph`,
    see `main`). Without this third condition, a graph whose embedding
    only covers the largest connected component (LCC != the full graph,
    e.g. 'cora') would end up in `graphs` and reserve grid rows that would
    then be plotted empty in the main loop (`Y.shape[0] != len(nodes)` ->
    `ax.axis('off')`) - an empty bottom third of the figure (see
    documentation/2026-09-13_w5_vysledky_diskuse.md)."""
    dataset_label = _dataset_label(graph, method_name, distance_metric)
    row = df_ok[(df_ok["dataset"] == dataset_label) & (df_ok["method"] == method_name) & (df_ok["seed"] == seed)]
    if row.empty:
        return False
    try:
        Y = load_embedding(RunKey(exp_name, dataset_label, method_name, seed))
    except FileNotFoundError:
        return False
    return Y.shape[0] == n_nodes


def main() -> None:
    import argparse

    from src.datasets.registry import load_dataset

    parser = argparse.ArgumentParser(description="K11: focused graph-layout figure (main text).")
    add_quick_arg(parser)
    args = parser.parse_args()
    mode = parse_fig_mode(args)
    EXPERIMENT_NAME = resolve_experiment_name(BASE_EXPERIMENT_NAME, mode)

    fg_cfg = load_experiments_config()["fig_graph_layouts_focus"]
    candidate_graphs: list[str] = list(fg_cfg["graphs"])
    distance_metrics: list[str] = list(fg_cfg["distance_metrics"])
    methods: list[str] = list(fg_cfg["methods"])

    df = require_experiment_csv(BASE_EXPERIMENT_NAME, mode)
    ok = df[df["status"] == "ok"]
    if ok.empty:
        raise ValueError(f"No successful run in {BASE_EXPERIMENT_NAME} (mode={mode}).")
    seed = int(ok["seed"].min())
    ok_seed = ok[ok["seed"] == seed]

    # K11/W5: load ALL candidate graphs up front (we need the node count
    # `len(nodes)` to verify the embedding covers the whole graph, not just
    # the LCC - see `_panel_available`), then use 'cora' (and any other
    # candidate beyond the first 2 mandatory graphs from the config) only
    # if all required panels (method x distance metric) already have a
    # saved embedding with the CORRECT number of rows - otherwise skipped
    # with a log (no reserved but empty row in the grid).
    graph_cache: dict[str, tuple[list, list[tuple[int, int]], np.ndarray | None]] = {}
    for graph in candidate_graphs:
        ds = load_dataset(graph)
        nodes = list(ds.graph.nodes())
        node_idx = {n: k for k, n in enumerate(nodes)}
        edges = [(node_idx[u], node_idx[v]) for u, v in ds.graph.edges() if u != v]
        graph_cache[graph] = (nodes, edges, ds.y)

    graphs: list[str] = []
    for graph in candidate_graphs:
        n_nodes = len(graph_cache[graph][0])
        needed = [(dm, m) for dm in distance_metrics for m in methods]
        available = all(_panel_available(ok_seed, EXPERIMENT_NAME, graph, m, dm, seed, n_nodes) for dm, m in needed)
        if available:
            graphs.append(graph)
        else:
            print(f"{FIG_NAME}: graph '{graph}' skipped - one or more E3 embeddings for seed={seed} are missing/mismatched (mode={mode}).")
            del graph_cache[graph]

    if len(graphs) < 2:
        raise ValueError(
            f"{FIG_NAME}: only {len(graphs)} usable graph(s) available out of candidates {candidate_graphs} "
            f"(mode={mode}, seed={seed}) - run src\\run_exp3_graph_layout.bat {mode} first."
        )

    row_keys = [(graph, dm) for graph in graphs for dm in distance_metrics]
    n_rows, n_cols = len(row_keys), len(methods)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(WIDTH_FULL_WIDTH_IN, WIDTH_FULL_WIDTH_IN / n_cols * n_rows * 1.15),
        squeeze=False,
    )
    csv_rows = []

    for i, (graph, distance_metric) in enumerate(row_keys):
        nodes, edges, y_labels = graph_cache[graph]

        for j, method_name in enumerate(methods):
            ax = axes[i][j]
            dataset_label = _dataset_label(graph, method_name, distance_metric)
            row = ok_seed[(ok_seed["dataset"] == dataset_label) & (ok_seed["method"] == method_name)]
            if row.empty:
                ax.axis("off")
                continue
            try:
                Y = load_embedding(RunKey(EXPERIMENT_NAME, dataset_label, method_name, seed))
            except FileNotFoundError:
                ax.axis("off")
                continue
            if Y.shape[0] != len(nodes):
                ax.axis("off")
                continue

            for u, v in edges:
                ax.plot([Y[u, 0], Y[v, 0]], [Y[u, 1], Y[v, 1]], color="#888888", linewidth=0.3, alpha=0.3, zorder=1, rasterized=True)
            if y_labels is not None:
                uniq = np.unique(y_labels)
                for lab_idx, lab in enumerate(uniq):
                    mask = y_labels == lab
                    ax.scatter(Y[mask, 0], Y[mask, 1], s=7, c=OKABE_ITO[lab_idx % len(OKABE_ITO)], zorder=2, rasterized=True, linewidths=0)
            else:
                ax.scatter(Y[:, 0], Y[:, 1], s=7, c=OKABE_ITO[0], zorder=2, rasterized=True, linewidths=0)

            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
            if i == 0:
                ax.set_title(method_name, fontsize=7)
            if j == 0:
                ax.set_ylabel(f"{graph}\n({distance_metric})", fontsize=6.5)

            stress = float(row["stress_scale_invariant"].iloc[0])
            auc_rnx = float(row["auc_rnx"].iloc[0])
            ax.text(0.5, -0.10, f"stress={stress:.3f}  auc_rnx={auc_rnx:.3f}", transform=ax.transAxes, ha="center", va="top", fontsize=5)

            for k in range(Y.shape[0]):
                csv_rows.append({
                    "base_graph": graph, "distance_metric": distance_metric, "method": method_name, "seed": seed,
                    "node_index": k, "y0": float(Y[k, 0]), "y1": float(Y[k, 1]),
                    "label": (y_labels[k] if y_labels is not None else ""),
                    "stress_scale_invariant": stress, "auc_rnx": auc_rnx,
                })

    fig.suptitle(f"Graph layouts: graph x distance (row) vs. method (column), seed={seed}", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, FIG_NAME)
    save_csv_alongside(pd.DataFrame(csv_rows), FIG_NAME)
    print(f"{FIG_NAME}: {len(graphs)} graphs x {len(distance_metrics)} distances x {len(methods)} methods.")


if __name__ == "__main__":
    main()
