# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-17
# License: see the LICENSE file in the repository root
"""
Replaces `results/tables/exp13_neighbor_survival.tex` in the MAIN TEXT (the
table moves to the supplement, see the article-shortening task) with a
single figure carrying the same message: does tuning alpha buy more
surviving nearest neighbors, and at what stress cost - and does that trade
even exist outside the low-ratio (distance-concentrated) regime?

Input: `results/tables/[<mode>/]exp13_neighbor_survival.csv` (the two-panel
policy table already built by `src.experiments.exp13_neighbor_survival`'s
`build_survival_table`/`_write_survival_table` - NOT recomputed here, same
"figure reads an already-generated table" convention as fig_pareto_front.py
reading results/tables/pareto_median_front.csv). Must exist for the current
mode (fail loud otherwise, see `how_to_generate` below).

Layout: a 2x2 grid, rows = regime panel (low_ratio "tuning helps" / pooled
mid+high_ratio "tuning does not help"), columns = metric (neighbors kept
out of k | stress, scale-invariant). Each panel is a horizontal dot plot:
one row per alpha-selection policy (fixed alpha=0/1/2..., the proposed
alpha_pred rule, and the alpha_best oracle - category order and labels
reused verbatim from `src.experiments.exp13_neighbor_survival`, never
duplicated), connected by a thin guide line in that fixed order so a
reader's eye reads the trend top-to-bottom. alpha_pred (this paper's rule)
and alpha_best (the oracle upper bound) are visually highlighted (distinct
marker/color, see `_style_for_category`) - everything else is a neutral
fixed policy. A missing/NaN cell (e.g. a --smoke run's narrower alpha grid, see
exp13's own module docstring) draws no marker and no line segment through
it - never a fabricated value.

The intended read: top row - neighbors_kept AND stress both climb
monotonically down the policy list (a real, quantifiable price paid for
tuning); bottom row - both metrics are visually flat across every policy
(tuning buys nothing there).

2026-09-17 fix (author, second review of this exact figure): the stress
column used to plot RAW stress_scale_invariant with each row's x-axis
independently centered on its own data via `_shared_span_xlims` (still
defined below - a correct, still-tested general "same width, different
center" tool - just no longer used for THIS column) - low_ratio's absolute
stress (~0.011-0.017) and mid/high_ratio's (~0.056-0.062) are on completely
different baselines, so "same window WIDTH, different center" still let
the bottom row's ~0.0015 absolute wobble fill its panel just as dramatically
as the top row's real ~0.005 swing, contradicting the figure's own message
("tuning costs stress up top, costs nothing down here"). The stress column
now plots `stress_pct_vs_alpha0` = 100*(stress - stress at alpha=0)/(stress
at alpha=0), i.e. relative change from the alpha=0 (MDS) baseline WITHIN
each row - both rows then share the literal SAME x-axis range (not just the
same width), so a genuinely flat bottom row shows as a flat row, not an
optically-scaled illusion.

2026-09-18 fix (author, third review - VADA 1): the point LABEL used to
print the absolute `stress_scale_invariant` value (e.g. "0.0163") right
next to a marker positioned on the `stress_pct_vs_alpha0` (%) axis (e.g.
~44), so the number printed at a point and the axis it sat on were two
different quantities with no unit shown - a reader had to guess that
"0.0163" was not itself a percent. Fixed by making the label show the SAME
quantity as the axis position: `stress_pct_vs_alpha0` formatted as a signed
percent (e.g. "+44.1%"), see `_METRIC_VALUE_FMT[_METRIC_STRESS_PCT]`. The
absolute `stress_scale_invariant` value (and its alpha=0 baseline) is not
lost - both are still written to the companion CSV
(`results/figures/fig_neighbor_survival.csv`) for readers who want the raw
numbers; only the ON-PLOT label text changed. Percent (not absolute stress)
was kept for both axis and label because the figure's whole point is
comparing the two regimes' PRICE OF TUNING on one shared scale despite
their very different absolute stress baselines (~0.01 vs ~0.06) - relative
change is the quantity that is actually comparable across panels, so it is
also the more informative number to print at each point. Also: y-tick/
legend/panel-ylabel text for the regime panel names and the alpha_pred/
alpha_best policy constants (which contain raw underscores, e.g.
"mid_or_high_ratio (tuning does not help)") now goes through the shared
`display_label(..., kind="regime"|"policy")` (see fig_common.py) - the
underlying CSV `panel`/`policy` STRING VALUES used for filtering are
untouched (still exactly PANEL_LOW_RATIO/PANEL_REST/POLICY_LABEL_PRED/
POLICY_LABEL_BEST from exp13_neighbor_survival.py), only the rendered text
changed. Point labels also got a size-aware offset (see `_label_gap_pt`) so
a big highlighted marker (the alpha_best star) no longer sits under its own
number.

Output: results/figures/[<mode>/]fig_neighbor_survival.pdf + a .csv with
the exact plotted (panel, policy, metric, value, stress_pct_vs_alpha0,
stress_alpha0_baseline) rows.

Run: venv\\python.exe -m src.figures.fig_neighbor_survival [--quick|--full|--smoke]
or: src\\run_fig_neighbor_survival.bat [quick|full|smoke]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.experiments.config_experiments import resolve_experiment_config
from src.experiments.exp13_neighbor_survival import (
    BASE_EXPERIMENT_NAME as EXP13_BASE_NAME,
    PANEL_LOW_RATIO,
    PANEL_REST,
    POLICY_LABEL_BEST,
    POLICY_LABEL_PRED,
    policy_label,
)
from src.figures.fig_common import (
    OKABE_ITO,
    WIDTH_FULL_WIDTH_IN,
    add_quick_arg,
    display_label,
    get_mode_path,
    parse_fig_mode,
    require_csv,
    save_csv_alongside,
    save_figure,
)

FIG_NAME = "fig_neighbor_survival"

# Row order (top to bottom) = panel order (fixed, matches
# exp13_neighbor_survival.build_survival_table's own panel dict order).
_PANEL_ORDER = [PANEL_LOW_RATIO, PANEL_REST]

# Column metrics: (results column, axis label built with the actual k).
_METRIC_NEIGHBORS = "neighbors_kept"
_METRIC_STRESS = "stress_scale_invariant"
# 2026-09-17 fix: the x-AXIS POSITION for the stress column is the relative
# change vs. the alpha=0 baseline (see `_add_stress_pct_column`) - the
# ABSOLUTE stress_scale_invariant value (_METRIC_STRESS) is still what gets
# printed as the point label (see `_draw_dot_panel`'s label_col).
_METRIC_STRESS_PCT = "stress_pct_vs_alpha0"
# 2026-09-18 fix (VADA 1): the stress point LABEL is now the same quantity
# as its axis position (stress_pct_vs_alpha0, a signed percent) - see the
# module docstring's 2026-09-18 note. _METRIC_STRESS's own format string is
# kept only because `_METRIC_VALUE_FMT` is also indexed by column name
# elsewhere; it is no longer used as an on-plot label.
_METRIC_VALUE_FMT = {_METRIC_NEIGHBORS: "{:.2f}", _METRIC_STRESS: "{:.4f}", _METRIC_STRESS_PCT: "{:+.1f}%"}

# Category visual style: neutral for fixed-alpha policies, highlighted for
# the proposed rule (alpha_pred) and the oracle (alpha_best) - colors from
# the Okabe-Ito colorblind-safe palette (fig_common.OKABE_ITO).
_STYLE_FIXED = {"color": OKABE_ITO[5], "marker": "o", "size": 20, "edgecolor": "none", "zorder": 3}
_STYLE_PRED = {"color": OKABE_ITO[1], "marker": "D", "size": 40, "edgecolor": "black", "zorder": 4}
_STYLE_BEST = {"color": OKABE_ITO[6], "marker": "*", "size": 90, "edgecolor": "black", "zorder": 4}
_LINE_COLOR = "#999999"


def _category_order(alpha_table: list[float]) -> list[str]:
    """Fixed top-to-bottom category order for the y-axis: the fixed-alpha
    policies in increasing alpha order, then alpha_pred (this paper's
    rule), then alpha_best (the oracle) - labels via
    `exp13_neighbor_survival.policy_label`, never re-typed here, so the
    figure's category strings always match the table's `policy` column
    exactly (whatever `exp13_neighbor_survival.alpha_table` currently is,
    e.g. the narrower --smoke grid)."""
    return [policy_label(a) for a in sorted(alpha_table)] + [POLICY_LABEL_PRED, POLICY_LABEL_BEST]


def _style_for_category(category: str) -> dict:
    """Marker style for one y-axis category: highlighted for the proposed
    rule/oracle, neutral for every fixed-alpha policy (see the module
    docstring's "visually highlighted" requirement)."""
    if category == POLICY_LABEL_PRED:
        return _STYLE_PRED
    if category == POLICY_LABEL_BEST:
        return _STYLE_BEST
    return _STYLE_FIXED


# Stress panels: the two regimes have very different absolute baselines
# (low_ratio ~0.01-0.02, mid/high_ratio ~0.05-0.06 in production data), so a
# single SHARED x-range would squeeze the low_ratio panel's (larger,
# relative-to-baseline) variation into an unreadable sliver. Instead both
# rows get the SAME x-axis WIDTH (an "aligned-scale, different-center"
# small-multiples convention), each centered on its own row's data - this
# is what makes "top row visibly moves, bottom row visibly does not" an
# honest, comparable visual (a tight per-row autoscale would instead
# artificially blow up the bottom row's tiny absolute spread to fill its
# own panel). `_STRESS_XLIM_PAD_FRAC` is a styling margin (as in
# fig_pareto_front.py's pad_x/pad_y), not an analysis parameter.
_STRESS_XLIM_PAD_FRAC = 0.5


def _shared_span_xlims(df: pd.DataFrame, metric_col: str, panel_order: list[str], pad_frac: float = _STRESS_XLIM_PAD_FRAC) -> dict[str, tuple[float, float]]:
    """{panel: (lo, hi)} for `metric_col`: every panel gets the SAME
    x-axis width (the largest per-panel data span over `panel_order`, times
    `1 + pad_frac`), each centered on that panel's own (min+max)/2 - see the
    comment above `_STRESS_XLIM_PAD_FRAC`. A panel with <= 1 finite value
    falls back to a span derived from its own center (never a fabricated
    spread)."""
    centers: dict[str, float] = {}
    spans: dict[str, float] = {}
    for panel in panel_order:
        values = df.loc[df["panel"] == panel, metric_col].dropna().to_numpy(dtype=float)
        if values.size == 0:
            centers[panel], spans[panel] = 0.0, 0.0
            continue
        lo, hi = float(values.min()), float(values.max())
        centers[panel], spans[panel] = (lo + hi) / 2.0, hi - lo

    max_span = max(spans.values(), default=0.0)
    if max_span <= 0.0:
        # every panel is a single repeated value (or empty) - fall back to
        # a small span relative to the largest center so the axis is not
        # degenerate, still never fabricating a value.
        max_center = max((abs(c) for c in centers.values()), default=1.0)
        max_span = max_center * 0.1 if max_center > 0 else 1.0

    half_width = max_span * (1.0 + pad_frac) / 2.0
    return {panel: (centers[panel] - half_width, centers[panel] + half_width) for panel in panel_order}


def _global_shared_xlim(df: pd.DataFrame, metric_col: str, panel_order: list[str], pad_frac: float = _STRESS_XLIM_PAD_FRAC) -> dict[str, tuple[float, float]]:
    """One LITERAL shared (lo, hi) range for `metric_col` over ALL panels
    combined (the same tuple returned for every panel) - unlike
    `_shared_span_xlims` (same WIDTH, different center per panel), this is
    for a metric that is ALREADY on a common scale across panels (e.g. a
    percent change from a per-panel baseline, see `_add_stress_pct_column`)
    and must therefore be shown on one truly common axis, so a flat row
    reads as flat rather than being rescaled to fill its own panel. A panel
    with <= 1 finite value across the whole figure falls back to a small
    span around 0 (never a fabricated spread)."""
    values = df.loc[df["panel"].isin(panel_order), metric_col].dropna().to_numpy(dtype=float)
    if values.size == 0:
        lo, hi = -1.0, 1.0
    else:
        lo, hi = float(values.min()), float(values.max())
        span = hi - lo
        if span <= 0.0:
            span = max(abs(lo), abs(hi), 1.0) * 0.1
        pad = span * pad_frac
        lo, hi = lo - pad, hi + pad
    return {panel: (lo, hi) for panel in panel_order}


def _add_stress_pct_column(df: pd.DataFrame, panel_order: list[str], baseline_policy: str) -> pd.DataFrame:
    """Add `stress_pct_vs_alpha0` = 100*(stress - baseline)/baseline and
    `stress_alpha0_baseline`, where `baseline` is this row's PANEL's own
    `stress_scale_invariant` at `policy == baseline_policy` (policy_label(0.0),
    i.e. the alpha=0/MDS row - see the module docstring's 2026-09-17 fix).
    Fail loud if a panel has no finite, positive baseline (never divides by
    a fabricated/zero value)."""
    out = df.copy()
    baseline_by_panel: dict[str, float] = {}
    for panel in panel_order:
        row = df[(df["panel"] == panel) & (df["policy"] == baseline_policy)]
        if row.empty or not np.isfinite(row[_METRIC_STRESS].iloc[0]) or float(row[_METRIC_STRESS].iloc[0]) <= 0.0:
            raise ValueError(
                f"Panel '{panel}' has no finite, positive '{baseline_policy}' {_METRIC_STRESS} baseline in "
                "exp13_neighbor_survival.csv - cannot compute the relative stress-vs-alpha0 axis."
            )
        baseline_by_panel[panel] = float(row[_METRIC_STRESS].iloc[0])
    out["stress_alpha0_baseline"] = out["panel"].map(baseline_by_panel)
    out[_METRIC_STRESS_PCT] = (out[_METRIC_STRESS] - out["stress_alpha0_baseline"]) / out["stress_alpha0_baseline"] * 100.0
    return out


# Point-label placement: the annotation must clear the marker's own radius
# (a fixed small offset overlapped the largest markers - e.g. "4.42" sat on
# top of the alpha_best star) - `_label_gap_pt` adds a constant clearance
# ON TOP OF the marker's radius (derived from its scatter `size` in points^2,
# a pure geometry fact, not a tunable analysis parameter) so every marker
# gets a offset proportional to its own footprint.
_LABEL_GAP_PT = 3.0


def _label_dx_pt(style: dict) -> float:
    """Horizontal clearance (points) from a marker's center to where its
    label should start: the marker's own radius (scatter `size` is an area
    in points^2) plus `_LABEL_GAP_PT`."""
    return float(np.sqrt(style["size"] / np.pi)) + _LABEL_GAP_PT


def _draw_dot_panel(
    ax, panel_df: pd.DataFrame, x_col: str, label_col: str, label_fmt: str,
    category_order: list[str], xlim: tuple[float, float] | None,
) -> None:
    """Horizontal dot plot for one (panel, metric) cell: one row per policy
    category (fixed y order, see `_category_order`), connected by a thin
    guide line through the FINITE points only (a missing policy - e.g. a
    --smoke grid gap - breaks the line there instead of interpolating
    through a fabricated value). The marker's x POSITION comes from `x_col`;
    the printed number next to it comes from `label_col` (formatted with
    `label_fmt`) - `x_col` and `label_col` are the SAME column for both
    metrics (neighbors_kept, and stress_pct_vs_alpha0 since the 2026-09-18
    fix, see the module docstring), so the plotted position and the printed
    number always refer to the same quantity."""
    n_cat = len(category_order)
    y_positions = {category: n_cat - 1 - i for i, category in enumerate(category_order)}  # first category on top

    by_x = panel_df.set_index("policy")[x_col] if not panel_df.empty else pd.Series(dtype=float)
    by_label = panel_df.set_index("policy")[label_col] if not panel_df.empty else pd.Series(dtype=float)
    xs_line: list[float] = []
    ys_line: list[float] = []
    for category in category_order:
        y = y_positions[category]
        x = float(by_x.get(category, float("nan")))
        if not np.isfinite(x):
            continue
        xs_line.append(x)
        ys_line.append(y)
    ax.plot(xs_line, ys_line, "-", color=_LINE_COLOR, linewidth=0.9, zorder=2)

    for category in category_order:
        y = y_positions[category]
        x = float(by_x.get(category, float("nan")))
        style = _style_for_category(category)
        if not np.isfinite(x):
            ax.text(0.02, y, "n/a", transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=5.2, color="#888888")
            continue
        ax.scatter(
            [x], [y], s=style["size"], c=style["color"], marker=style["marker"],
            edgecolors=style["edgecolor"], linewidths=0.7, zorder=style["zorder"],
        )
        label_value = float(by_label.get(category, float("nan")))
        label_text = label_fmt.format(label_value) if np.isfinite(label_value) else "n/a"
        ax.annotate(label_text, (x, y), xytext=(_label_dx_pt(style), 0), textcoords="offset points", va="center", fontsize=5.2)

    ax.set_yticks([y_positions[c] for c in category_order])
    ax.set_yticklabels([display_label(c, "policy") for c in category_order], fontsize=5.6)
    ax.set_ylim(-0.7, n_cat - 0.3)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.tick_params(axis="x", labelsize=5.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Neighbor survival vs. metric cost by alpha-selection policy, by regime (replaces the exp13 main-text table).")
    add_quick_arg(parser)
    args = parser.parse_args()
    mode = parse_fig_mode(args)

    exp13_cfg = resolve_experiment_config(EXP13_BASE_NAME, mode)
    alpha_table = [float(a) for a in exp13_cfg["alpha_table"]]
    category_order = _category_order(alpha_table)

    tables_dir = get_mode_path("results_tables_dir", mode)
    table_path = tables_dir / "exp13_neighbor_survival.csv"
    df = require_csv(table_path, f"venv\\python.exe -m src.experiments.exp13_neighbor_survival --{mode}")

    k_values = sorted(df["k"].unique())
    if len(k_values) != 1:
        raise ValueError(f"{table_path}: expected a single k value (this figure shows one k), found {k_values}.")
    k = int(k_values[0])

    missing_panels = [p for p in _PANEL_ORDER if p not in set(df["panel"])]
    if missing_panels:
        raise ValueError(f"{table_path}: missing expected panel(s) {missing_panels} (found {sorted(df['panel'].unique())}).")

    # 2026-09-17 fix: stress is plotted as % change vs. the alpha=0 (MDS)
    # baseline WITHIN each panel (see the module docstring and
    # `_add_stress_pct_column`) so the two regimes - very different absolute
    # stress baselines - become directly, honestly comparable on ONE shared
    # axis instead of two independently-scaled ones.
    df = _add_stress_pct_column(df, _PANEL_ORDER, baseline_policy=policy_label(0.0))

    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_FULL_WIDTH_IN, WIDTH_FULL_WIDTH_IN * 0.85), squeeze=False)

    # neighbors_kept: one SHARED (0, k) scale (a natural, physically bounded
    # axis - both regimes are directly comparable on it). stress: ONE
    # literal shared range too, now that it is a %-of-baseline metric (see
    # `_global_shared_xlim`) - a genuinely flat row reads as flat.
    stress_xlims = _global_shared_xlim(df, _METRIC_STRESS_PCT, _PANEL_ORDER)
    # 2026-09-18 fix (VADA 1): label_col == x_col for BOTH metrics now, so
    # the printed number always matches the axis the marker sits on (see
    # `_draw_dot_panel`'s docstring and the module docstring's 2026-09-18 note).
    metric_columns = [
        (_METRIC_NEIGHBORS, _METRIC_NEIGHBORS, _METRIC_VALUE_FMT[_METRIC_NEIGHBORS], f"Neighbors kept (out of k={k})", {p: (0.0, k * 1.12) for p in _PANEL_ORDER}),
        (_METRIC_STRESS_PCT, _METRIC_STRESS_PCT, _METRIC_VALUE_FMT[_METRIC_STRESS_PCT], r"Stress vs. $\alpha=0$" + "\n(%, higher = worse)", stress_xlims),
    ]

    for row, panel_name in enumerate(_PANEL_ORDER):
        panel_df = df[df["panel"] == panel_name]
        n_datasets = int(panel_df["n_datasets"].max()) if not panel_df.empty and panel_df["n_datasets"].notna().any() else 0
        for col, (x_col, label_col, label_fmt, metric_label, xlim_by_panel) in enumerate(metric_columns):
            ax = axes[row][col]
            _draw_dot_panel(ax, panel_df, x_col, label_col, label_fmt, category_order, xlim_by_panel[panel_name])
            if x_col == _METRIC_NEIGHBORS:
                ax.axvline(k, color="#bbbbbb", linewidth=0.8, linestyle="--", zorder=1)
            elif x_col == _METRIC_STRESS_PCT:
                ax.axvline(0.0, color="#bbbbbb", linewidth=0.8, linestyle="--", zorder=1)
            if row == 0:
                ax.set_title(metric_label, fontsize=7)
            elif row == len(_PANEL_ORDER) - 1:
                ax.set_xlabel(metric_label, fontsize=6.5)
            if col == 0:
                ax.set_ylabel(f"{display_label(panel_name, 'regime')}\n(n={n_datasets} datasets)", fontsize=6.5)

    legend_handles = [
        Line2D([0], [0], marker=_STYLE_FIXED["marker"], color="none", markerfacecolor=_STYLE_FIXED["color"], markersize=5, label=r"fixed $\alpha$ (0/1/2/...)"),
        Line2D([0], [0], marker=_STYLE_PRED["marker"], color="none", markerfacecolor=_STYLE_PRED["color"], markeredgecolor="black", markersize=6, label=display_label(POLICY_LABEL_PRED, "policy")),
        Line2D([0], [0], marker=_STYLE_BEST["marker"], color="none", markerfacecolor=_STYLE_BEST["color"], markeredgecolor="black", markersize=9, label=display_label(POLICY_LABEL_BEST, "policy")),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=6, frameon=False, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle("Neighbor survival vs. metric cost, by alpha-selection policy and distance-concentration regime", fontsize=8, y=0.985)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))

    save_figure(fig, FIG_NAME)
    save_csv_alongside(df, FIG_NAME)

    print(f"{FIG_NAME}: {len(_PANEL_ORDER)} panels x {len(metric_columns)} metrics, k={k}, {len(category_order)} policies, {len(df)} table rows.")


if __name__ == "__main__":
    main()
