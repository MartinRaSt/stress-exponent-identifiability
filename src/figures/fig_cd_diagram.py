# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-09
# Rewritten: 2026-09-17 (author feedback - see the module docstring below)
# License: see the LICENSE file in the repository root
"""
Figure 14e - a proper Demsar (2006, JMLR 7:1-30, Fig. 1) critical-difference
diagram: the mean rank of methods over datasets (1 = best, on a horizontal
axis), with one bar per MAXIMAL clique of methods that are NOT significantly
different under the Nemenyi post-hoc test (|avg_rank_i - avg_rank_j| < CD),
from `results/data/stats_<experiment>_<metric>.csv` (src/experiments/stats.py).

Author feedback 2026-09-17: the previous version was not a CD diagram - it
drew one bar per pairwise insignificant comparison (30+ stacked bars instead
of a handful of maximal cliques), method labels used raw config identifiers
with underscores and sat on top of the bars, and the title had a raw
'chi2'. This rewrite:
  - draws exactly one horizontal bar per maximal clique
    (`stats.maximal_insignificant_cliques`), never a bar per pair;
  - fans method labels out to the left (better half) / right (worse half)
    of the axis, connected by an L-shaped line to their point on the axis -
    labels never overlap the clique bars, which live in their own row band
    below the axis;
  - uses `fig_common.display_label` for every method name (adds the alpha
    symbol, drops underscores);
  - spells out the title in words with mathtext $\\chi^2$, no raw identifiers.

Run: venv\\python.exe -m src.figures.fig_cd_diagram [--quick] [--experiment exp1_dr_benchmark_main] [--metric auc_rnx]
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import pandas as pd

from src.experiments.config_experiments import load_experiments_config
from src.experiments.stats import maximal_insignificant_cliques
from src.figures.fig_common import (
    OKABE_ITO,
    WIDTH_SUPPLEMENT_FULL_IN,
    add_quick_arg,
    display_label,
    mode_data_dir,
    parse_fig_mode,
    require_csv,
    save_csv_alongside,
    save_figure,
)

FIG_NAME = "fig_cd_diagram"


def _measure_text_width_in(fig: plt.Figure, text: str, fontsize_pt: float) -> float:
    """Rendered width (inches) of `text` at `fontsize_pt`, via a throwaway
    Text artist drawn on `fig` with matplotlib's own Agg renderer (correctly
    handles mathtext, e.g. the alpha symbol in method labels, unlike a raw
    glyph-width estimate) - author feedback 2026-09-19 (second review): the
    rank axis was squeezed by a FIXED left/right margin guess that did not
    track the actual label text. Used once per side (the longest label on
    that side) to size fig_cd_diagram.label_margin_padding_in-padded
    margins from real content instead of a hand-tuned constant."""
    probe = fig.text(0.0, 0.0, text, fontsize=fontsize_pt)
    fig.canvas.draw()
    bbox = probe.get_window_extent(renderer=fig.canvas.get_renderer())
    probe.remove()
    return float(bbox.width) / fig.dpi


def _label_rows(n_methods: int) -> list[int]:
    """Row index (0 = closest to the axis) for each method's label, fanning
    the better-ranked half out to the left and the worse-ranked half out to
    the right (Demsar 2006, Fig. 1 layout): within each half, the method
    CLOSEST to its side's axis edge gets the shortest connector (row 0), so
    the L-shaped connector lines never cross one another."""
    n_left = math.ceil(n_methods / 2)
    rows = []
    for i in range(n_methods):
        rows.append(i if i < n_left else (n_methods - 1 - i))
    return rows


def _draw_cd_diagram(
    ax: plt.Axes,
    methods: list[str],
    ranks: list[float],
    cd: float,
    cliques: list[tuple[int, int]],
    cfg: dict,
    total_height_in: float,
    width_in: float,
) -> None:
    """Draw the diagram on an axes spanning the full figure ([0,0,1,1], both
    x and y in [0,1]) - all positions below are computed once in INCHES
    (from `cfg`, see config_experiments.yaml fig_cd_diagram) and converted
    to figure fractions, so the figure height set by the caller and the
    positions drawn here always agree."""
    n = len(methods)
    axis_color = OKABE_ITO[cfg["axis_color_index"]]
    bar_color = OKABE_ITO[cfg["clique_bar_color_index"]]
    cd_ref_color = OKABE_ITO[cfg["cd_reference_color_index"]]

    x_left = cfg["left_margin_in"] / width_in
    x_right = 1.0 - cfg["right_margin_in"] / width_in
    rank_lo, rank_hi = 1.0, float(n)  # average ranks always live in [1, n_methods]

    def rankpos(r: float) -> float:
        return x_left + (r - rank_lo) / (rank_hi - rank_lo) * (x_right - x_left)

    def y_of(depth_in: float) -> float:
        return 1.0 - depth_in / total_height_in

    # depth order from the axis downward: axis -> clique bars -> method
    # labels (author feedback 2026-09-17: cliques must sit right under the
    # axis, NOT below the label fan-out, so the reader can directly see
    # which axis points a bar spans without the label clutter in between).
    d_axis = cfg["top_margin_in"]
    rows = _label_rows(n)
    n_left = math.ceil(n / 2)
    total_label_rows = max(n_left, n - n_left)
    n_bars = len(cliques)
    d_bars_start = d_axis + cfg["axis_tick_gap_in"]
    d_labels_start = d_bars_start + n_bars * cfg["clique_row_height_in"] + (cfg["clique_gap_in"] if n_bars > 0 else 0.0)
    d_label_row = [d_labels_start + (row + 0.5) * cfg["label_row_height_in"] for row in range(total_label_rows)]

    # axis line + integer rank ticks (average ranks always lie in [1, n_methods])
    y_axis = y_of(d_axis)
    ax.plot([rankpos(rank_lo), rankpos(rank_hi)], [y_axis, y_axis], color=axis_color, linewidth=cfg["axis_linewidth"], solid_capstyle="butt", zorder=2)
    tick_len = cfg["tick_length_in"] / total_height_in
    # 2026-09-19 (author feedback, second review): a tick MARK is drawn at
    # every integer rank, but the tick NUMBER is only printed every
    # tick_label_stride-th rank (always including 1 and n) - at n>=10 the
    # 2-digit numbers printed at every rank ran into each other
    # ("910111213"); this keeps every rank visually marked while giving
    # printed numbers room to breathe.
    stride = int(cfg["tick_label_stride"])
    for tick in range(1, n + 1):
        x_tick = rankpos(float(tick))
        ax.plot([x_tick, x_tick], [y_axis, y_axis + tick_len], color=axis_color, linewidth=cfg["tick_linewidth"], zorder=2)
        if tick == 1 or tick == n or (tick - 1) % stride == 0:
            ax.text(x_tick, y_axis + tick_len * 1.5, str(tick), ha="center", va="bottom", fontsize=cfg["tick_fontsize_pt"], color=axis_color)

    # CD reference bar (a short ruler of length CD anchored at rank 1), so
    # the reader can compare it by eye against any pair of points below
    d_cd_ref = cfg["cd_reference_depth_in"]
    y_cd = y_of(d_cd_ref)
    x0_cd, x1_cd = rankpos(rank_lo), rankpos(rank_lo + cd)
    ax.plot([x0_cd, x1_cd], [y_cd, y_cd], color=cd_ref_color, linewidth=cfg["cd_reference_linewidth"], solid_capstyle="butt", zorder=3)
    for x_end in (x0_cd, x1_cd):
        ax.plot([x_end, x_end], [y_cd - tick_len, y_cd + tick_len], color=cd_ref_color, linewidth=cfg["cd_reference_linewidth"], zorder=3)
    ax.text(x1_cd + cfg["label_gap_in"] / width_in, y_cd, f"CD = {cd:.3f}", ha="left", va="center", fontsize=cfg["tick_fontsize_pt"], color=cd_ref_color)

    # one point per method on the axis + an L-shaped connector fanning out
    # to its label on the left (better half) or right (worse half) side
    for i, (name, rank) in enumerate(zip(methods, ranks)):
        x_point = rankpos(rank)
        ax.scatter([x_point], [y_axis], s=cfg["marker_size_pt"], color=axis_color, zorder=4)
        y_label = y_of(d_label_row[rows[i]])
        on_left = i < n_left
        x_text = x_left - cfg["label_gap_in"] / width_in if on_left else x_right + cfg["label_gap_in"] / width_in
        ax.plot([x_point, x_point], [y_axis, y_label], color=axis_color, linewidth=cfg["connector_linewidth"], zorder=1)
        ax.plot([x_point, x_text], [y_label, y_label], color=axis_color, linewidth=cfg["connector_linewidth"], zorder=1)
        ax.text(
            x_text, y_label, f"{display_label(name, 'method')} ({rank:.2f})",
            ha="right" if on_left else "left", va="center", fontsize=cfg["label_fontsize_pt"],
        )

    # one bar per MAXIMAL clique of mutually not-significantly-different
    # methods - short vertical end ticks make even a very narrow bar (e.g.
    # two methods with almost identical rank) visually read as a capped
    # span rather than an ambiguous stray dot
    end_tick_len = 0.6 * tick_len
    for k, (s, e) in enumerate(cliques):
        y_bar = y_of(d_bars_start + (k + 0.5) * cfg["clique_row_height_in"])
        x_s, x_e = rankpos(ranks[s]), rankpos(ranks[e])
        ax.plot([x_s, x_e], [y_bar, y_bar], color=bar_color, linewidth=cfg["clique_bar_linewidth"], solid_capstyle="round", zorder=3)
        for x_end in (x_s, x_e):
            ax.plot([x_end, x_end], [y_bar - end_tick_len, y_bar + end_tick_len], color=bar_color, linewidth=cfg["tick_linewidth"], zorder=3)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Critical difference diagram (Friedman/Nemenyi, Demsar 2006).")
    add_quick_arg(parser)
    # default: Friedman/Nemenyi only over report.main_methods (suffix "_main",
    # src/experiments/report_tables.py::write_exp1_main_table, K12b) - k <= 20,
    # so q_alpha exists; over all 21 E1 methods the stats file is not generated
    parser.add_argument("--experiment", type=str, default="exp1_dr_benchmark_main")
    parser.add_argument("--metric", type=str, default="auc_rnx")
    args = parser.parse_args()
    mode = parse_fig_mode(args)

    stats_path = mode_data_dir(mode) / f"stats_{args.experiment}_{args.metric}.csv"
    df = require_csv(stats_path, f"venv\\python.exe -m src.experiments.stats --{mode} (requires {args.experiment}_results.csv)")
    df = df.sort_values("avg_rank").reset_index(drop=True)

    cd = float(df["cd"].iloc[0])
    methods = df["method"].tolist()
    ranks = df["avg_rank"].tolist()
    n_methods = len(methods)
    chi2 = float(df["friedman_chi2"].iloc[0])
    pval = float(df["friedman_pvalue"].iloc[0])
    kendall_w = float(df["kendall_w"].iloc[0])
    n_datasets = int(df["n_datasets"].iloc[0])

    cliques = maximal_insignificant_cliques(ranks, cd)
    n_bars = len(cliques)

    cfg = dict(load_experiments_config()["fig_cd_diagram"])
    n_left = math.ceil(n_methods / 2)
    total_label_rows = max(n_left, n_methods - n_left)
    width_in = WIDTH_SUPPLEMENT_FULL_IN

    # 2026-09-19 (author feedback, second review): left_margin_in/right_margin_in
    # are measured from the ACTUAL label text of THIS run (see
    # _measure_text_width_in and the config block comment), not a fixed
    # guess - this is what lets the rank axis use the width the method
    # names of a given experiment/metric do not need, instead of the same
    # fixed margin regardless of how long the longest label happens to be.
    probe_fig = plt.figure(figsize=(width_in, 1.0))
    label_texts = [f"{display_label(name, 'method')} ({rank:.2f})" for name, rank in zip(methods, ranks)]
    left_texts, right_texts = label_texts[:n_left], label_texts[n_left:]
    padding_in = float(cfg["label_margin_padding_in"]) + float(cfg["label_gap_in"])
    left_margin_in = max((_measure_text_width_in(probe_fig, t, cfg["label_fontsize_pt"]) for t in left_texts), default=0.0) + padding_in
    right_margin_in = max((_measure_text_width_in(probe_fig, t, cfg["label_fontsize_pt"]) for t in right_texts), default=0.0) + padding_in
    plt.close(probe_fig)
    if left_margin_in + right_margin_in >= 0.9 * width_in:
        raise ValueError(
            f"fig_cd_diagram: measured label margins ({left_margin_in:.2f}in + {right_margin_in:.2f}in) leave no "
            f"usable width for the rank axis out of {width_in:.2f}in - method names are too long for this figure width."
        )
    cfg["left_margin_in"] = left_margin_in
    cfg["right_margin_in"] = right_margin_in

    # axis -> clique bars -> method labels -> bottom margin (see _draw_cd_diagram)
    total_height_in = (
        cfg["top_margin_in"] + cfg["axis_tick_gap_in"]
        + (n_bars * cfg["clique_row_height_in"] + cfg["clique_gap_in"] if n_bars > 0 else 0.0)
        + total_label_rows * cfg["label_row_height_in"] + cfg["bottom_margin_in"]
    )

    fig = plt.figure(figsize=(width_in, total_height_in))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    _draw_cd_diagram(ax, methods, ranks, cd, cliques, cfg, total_height_in, width_in)

    experiment_label = display_label(args.experiment, "experiment")
    metric_label = display_label(args.metric, "metric")
    title_y = 1.0 - cfg["title_depth_in"] / total_height_in
    ax.text(
        0.5, title_y,
        f"Critical difference diagram: {experiment_label}, {metric_label}\n"
        rf"Friedman $\chi^2$={chi2:.2f}, $p$={pval:.3g}, Kendall $W$={kendall_w:.3f}, $N$={n_datasets} datasets",
        ha="center", va="top", fontsize=cfg["title_fontsize_pt"], transform=ax.transAxes,
    )

    out_name = f"{FIG_NAME}_{args.experiment}_{args.metric}"
    save_figure(fig, out_name)

    clique_col = []
    for i in range(n_methods):
        members = [k for k, (s, e) in enumerate(cliques) if s <= i <= e]
        clique_col.append(";".join(str(k) for k in members))
    df_out = df.copy()
    df_out["clique_id"] = clique_col
    save_csv_alongside(df_out, out_name)

    cliques_df = pd.DataFrame([
        {
            "clique_id": k, "n_members": e - s + 1, "rank_min": ranks[s], "rank_max": ranks[e],
            "methods": ";".join(methods[i] for i in range(s, e + 1)),
        }
        for k, (s, e) in enumerate(cliques)
    ])
    save_csv_alongside(cliques_df, f"{out_name}_cliques")

    print(
        f"{FIG_NAME}: {args.experiment}/{args.metric}, {n_methods} methods, CD={cd:.3f}, "
        f"{n_bars} maximal clique bar(s)."
    )


if __name__ == "__main__":
    main()
