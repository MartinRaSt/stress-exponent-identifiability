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
with a log"), columns = 5 methods (`sammon_alpha_auto`, `tsne_auto`,
`umap_auto`, `spectral`, `spring` - the last two are graph-native, i.e.
independent of the distance metric of a given row, but shown in both rows
for grid consistency). Edges are drawn in gray (alpha 0.3, from the GRAPH,
not from the embedding), node color = community/class (`ds.y`, if present).
Below each panel: stress_scale_invariant and auc_rnx (E3, seed from the
first available seed of `distance_metrics` - see `_pick_seed`); the
`sammon_alpha_auto` panel additionally prints the tuned $\\alpha^*$ actually
selected for that row (`selected_hyperparam`, see `_alpha_star_text`) -
this paper's rule picks alpha*=0 (i.e. the MDS baseline) in most of the
default rows, so the annotation tells the reader directly, in-panel, when
"tuned" and "alpha=0" are the same layout, instead of leaving two
visually-identical columns unexplained (2026-09-18 fix, reviewer VADA 2b).

2026-09-19 proportions fix, round 1 (author feedback: "text overwhelms
data" - each panel used to be a small square with three lines of text
(stress, AUC_RNX, alpha*) printed BELOW it, taking almost as much vertical
space as the layout itself, see results/figures/check_axes_area.csv before
this fix, axes area ~45% of the canvas): panels are no longer forced
square: this figure's own caption (05_vysledky.tex) states the reader
should read off "whether same-colored nodes form a separated or an
overlapping region", a topological, aspect-independent statement (unlike
fig_faithful_map, whose whole point is literal distance fidelity, where a
non-square panel would misrepresent the figure's own claim) - so a shorter
(still square-ISH, not stretched to a sliver) panel does not mislead here,
and is what keeps the 4-row grid under
`figures.layout.main_text_max_height_frac_textheight` (checked by
src/figures/check_axes_area.py). The in-image suptitle is also removed (the
LaTeX caption already states the figure's content and seed).

2026-09-19 proportions fix, round 2 (author feedback, having seen the
round-1 typeset PDF: "cisla uvnitr grafu musi byt mensi, jsou obrovska" -
the round-1 fix moved the per-panel numbers from BELOW each panel to an
INSIDE overlay, which fixed the axes-area ratio but the numbers, at the
same ANNOTATION_FONT_PT as everything else, visually dominated the small
(~53pt) panel interior). Per the author's own first-choice fix ("cisla z
kreslici plochy uplne odstranit a presunout do... male tabulky"), the
in-panel numbers are REMOVED entirely; the exact same stress/AUC_RNX/alpha*
values (computed identically, from the same in-memory rows used to draw
the panels - not re-derived, so they cannot drift from what the figure
shows) are now written to a small companion table,
`results/tables/[<mode>/]fig_graph_layouts_metrics.tex/.csv`
(`_write_metrics_table`) for the article text to input next to the
figure.

2026-09-18 fix (reviewer VADA 2a): this figure used to also show
`sammon_alpha0_smacof` (alpha=0/MDS) as a column next to `sammon_alpha_auto`
(the tuned rule) - verified in results/data/exp3_graph_layout_results.csv
that the tuned rule selects alpha*=0 for football/shortest_path,
football/resistance and polbooks/resistance (3 of the 4 default rows), so
those two columns were BIT-IDENTICAL saved embeddings (`np.array_equal`) in
3 of 4 rows - a quarter of the panel grid carried no extra information.
`sammon_alpha0_smacof` was replaced by `spectral` (see
`fig_graph_layouts_focus.methods` in config_experiments.yaml for the full
rationale) - a native layout already computed by exp3_graph_layout (no new
experiment run) and discussed in the article (03_metoda.tex, 05_vysledky.tex,
06_diskuse.tex) but previously missing from this main-text figure.

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
from src.common.config import get_tables_dir
from src.experiments.config_experiments import load_experiments_config
from src.experiments.exp_common import resolve_experiment_name
from src.experiments.report_tables import write_booktabs_tex
from src.figures.fig_common import (
    LABEL_FONT_PT,
    OKABE_ITO,
    WIDTH_FULL_WIDTH_IN,
    add_quick_arg,
    display_label,
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

# Panel height as a fraction of panel width (2026-09-19 proportions fix,
# see the module docstring) - < 1 on purpose (see the docstring for why a
# non-square panel does not mislead in THIS figure), tuned so a 4-row grid
# (2 graphs x 2 distances) plus one line of column titles stays under
# figures.layout.main_text_max_height_frac_textheight (checked by
# src/figures/check_axes_area.py).
_PANEL_ASPECT = 0.80

# 2026-09-18 fix (VADA 2b): the tuned-alpha method whose panel gets the
# extra "alpha*=..." annotation (see `_alpha_star_text`) - this is the only
# method in `fig_graph_layouts_focus.methods` whose weight exponent is
# chosen per-row rather than fixed, so it is the only one where a reader
# needs to be told, in-panel, which fixed-alpha layout it happens to match.
_ALPHA_TUNED_METHOD = "sammon_alpha_auto"


def _dataset_label(graph: str, method_name: str, distance_metric: str) -> str:
    return graph if method_name in _NATIVE_METHODS else f"{graph}__{distance_metric}"


def _alpha_star_text(selected_hyperparam) -> str | None:
    """`None` (draw nothing) if `selected_hyperparam` is missing/NaN
    (native methods, e.g. 'spring'/'spectral', have no tuned hyperparameter)
    or does not look like this experiment's own 'alpha=<value>' encoding
    (see exp3_graph_layout.py) - otherwise the tuned alpha as
    "$\\alpha^*$=<value>", e.g. "$\\alpha^*$=0.0", so a reader immediately
    sees WHICH fixed-alpha layout the tuned rule picked for this row,
    instead of having to infer it from two visually-identical columns
    (VADA 2, reviewer 2026-09-18)."""
    if selected_hyperparam is None or (isinstance(selected_hyperparam, float) and not np.isfinite(selected_hyperparam)):
        return None
    text = str(selected_hyperparam)
    if not text.startswith("alpha="):
        return None
    return r"$\alpha^*$=" + text[len("alpha="):]


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


def _write_metrics_table(metrics_rows: list[dict], mode: str) -> Path:
    """results/tables/[<mode>/]fig_graph_layouts_metrics.tex/.csv (2026-09-19
    proportions fix, round 2, see the module docstring): one row per panel
    of fig_graph_layouts.pdf (graph x distance x method), columns
    stress_scale_invariant / auc_rnx / alpha_star - the exact numbers the
    panel-internal text used to show, now here instead."""
    df = pd.DataFrame(metrics_rows)
    tables_dir = get_tables_dir(mode)
    df.to_csv(tables_dir / "fig_graph_layouts_metrics.csv", index=False)
    out_path = tables_dir / "fig_graph_layouts_metrics.tex"
    write_booktabs_tex(
        df, out_path,
        # The table is typeset in the supplement, which is compiled on its own:
        # a cross-document reference to a main-text label prints as "??".
        caption=r"Graph layouts of the main text (Results, Section ``C: Graphs "
                r"and resistance distance''): scale-invariant stress, "
                r"AUC$_{RNX}$, and the tuned exponent $\alpha^{*}$ "
                r"(only defined for the $\alpha$-Sammon column) per panel.",
        label="tab:graph_layouts_metrics",
        float_format={"stress_scale_invariant": "%.3f", "auc_rnx": "%.3f", "__default__": "%.3f"},
        value_labels={"dataset": "dataset", "distance_metric": "distance_metric", "method": "method"},
        comment_lines=[
            "source: results/figures/fig_graph_layouts.csv (same in-memory values used to draw fig_graph_layouts.pdf)",
            "alpha_star is blank for methods without a tuned hyperparameter (all but the alpha-Sammon column)",
        ],
    )
    return out_path


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
    # 2026-09-19 proportions fix (see the module docstring): the per-panel
    # text block that used to sit BELOW each panel is now an in-panel
    # overlay (no reserved space at all), and panels are drawn as a mild
    # rectangle (PANEL_ASPECT < 1, height < width) rather than forced
    # square - both free the row height that used to go to text/whitespace,
    # keeping the whole grid under the main-text height cap even at 4 rows.
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(WIDTH_FULL_WIDTH_IN, WIDTH_FULL_WIDTH_IN / n_cols * n_rows * _PANEL_ASPECT),
        squeeze=False, constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.03, hspace=0.05)
    csv_rows = []
    metrics_rows = []

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
                ax.set_title(display_label(method_name, "method"), fontsize=LABEL_FONT_PT)
            if j == 0:
                # 2026-09-19 proportions fix (see the module docstring):
                # horizontal (not rotated) row label - constrained_layout
                # grows the left margin to fit it automatically, and a
                # horizontal label cannot suffer the rotated-label
                # canvas-edge clipping seen in fig_faithful_map.
                ax.set_ylabel(
                    f"{display_label(graph, 'dataset')}\n({display_label(distance_metric, 'distance_metric')})",
                    fontsize=LABEL_FONT_PT, rotation=0, ha="right", va="center", labelpad=6,
                )

            stress = float(row["stress_scale_invariant"].iloc[0])
            auc_rnx = float(row["auc_rnx"].iloc[0])
            selected_hyperparam = row["selected_hyperparam"].iloc[0] if "selected_hyperparam" in row.columns else None
            # 2026-09-19 proportions fix, round 2 (see the module docstring):
            # no more in-panel text at all - the exact same stress/AUC_RNX,
            # plus alpha* for the tuned method, go to `metrics_rows` instead,
            # written to a companion table by `_write_metrics_table` below.
            alpha_star_text = _alpha_star_text(selected_hyperparam) if method_name == _ALPHA_TUNED_METHOD else None
            metrics_rows.append({
                "dataset": graph, "distance_metric": distance_metric, "method": method_name,
                "stress_scale_invariant": stress, "auc_rnx": auc_rnx,
                "alpha_star": (alpha_star_text.split("=", 1)[1] if alpha_star_text is not None else ""),
            })

            for k in range(Y.shape[0]):
                csv_rows.append({
                    "base_graph": graph, "distance_metric": distance_metric, "method": method_name, "seed": seed,
                    "node_index": k, "y0": float(Y[k, 0]), "y1": float(Y[k, 1]),
                    "label": (y_labels[k] if y_labels is not None else ""),
                    "stress_scale_invariant": stress, "auc_rnx": auc_rnx,
                    "selected_hyperparam": ("" if selected_hyperparam is None or (isinstance(selected_hyperparam, float) and not np.isfinite(selected_hyperparam)) else str(selected_hyperparam)),
                })

    # 2026-09-19 proportions fix (see the module docstring): no in-image
    # suptitle - the LaTeX caption (clanek_en/sections/05_vysledky.tex)
    # already states the figure's content and seed.
    save_figure(fig, FIG_NAME)
    save_csv_alongside(pd.DataFrame(csv_rows), FIG_NAME)
    metrics_table_path = _write_metrics_table(metrics_rows, mode)
    print(f"{FIG_NAME}: {len(graphs)} graphs x {len(distance_metrics)} distances x {len(methods)} methods.")
    print(f"{FIG_NAME}: metrics table written to {metrics_table_path}.")


if __name__ == "__main__":
    main()
