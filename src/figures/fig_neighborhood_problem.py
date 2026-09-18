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
Flagship figure of the article (author 2026-09-17): ONE picture that shows,
at a glance, the core problem this paper is about - metric methods keep
distances but shred neighborhoods. Three panels, the SAME dataset, the SAME
probe point, examined under three different 2D embeddings:

  1. MDS (alpha=0)          - keeps distances, neighborhoods broken
  2. alpha-Sammon (tuned)   - neighborhoods restored, at a stress cost
  3. t-SNE (default)        - best neighborhoods, worst geometry

Dataset: 'helix' (config `fig_neighborhood_problem.dataset`), chosen over
's_curve' after rendering both with the edge overlay used by earlier
drafts (see the FOURTH-review note below for why edges were dropped
entirely): helix's ring/spiral projections are more symmetric across all
three panels and make the MDS-tangle vs. tuned-order contrast visually
starker; s_curve's numeric trade-off is slightly more monotonic but the
picture itself is less striking.

FOURTH REVIEW (author 2026-09-17) - WHY THE FIRST THREE DESIGNS FAILED, this
time MEASURED, not guessed. On the full helix/alpha=0 cache
(`results/data/embeddings/exp6_alpha_curves/helix__alpha0.0__0.npy`, k=7):
neighbors kept = 2.17/7, but a true neighbor's RANK in the embedding's own
order is a median of 13 (not "off the map" - just outside the top 7), and
its EMBEDDING DISTANCE is a median 3.8x / 90th-percentile 10.5x the
embedding's own local point spacing. In other words: MDS does not throw
true neighbors far away - it lets FOREIGN points squeeze in between them
and the probe ("crowding out", not "ejection"). Every earlier edge-drawing
attempt (1: all n*k edges thin/transparent - invisible; 2: only edges
"long" relative to local spacing - the true-neighbor edges are ALWAYS
short at panel scale, by the measurement above, so nothing long enough to
single out ever existed; 3: n_probes=4 full-panel rays - short edges
again invisible at panel scale) was trying to show a large-scale effect
that does not exist. The fix: STOP drawing edges altogether and instead
ZOOM into a small window around the probe, where "3.8x-10x the local
spacing" is actually a big, legible distance, and mark WHICH nearby points
are true neighbors and which are intruders.

ONE PROBE, THREE ROLES, ONE ZOOM WINDOW PER PANEL. `n_probes=1` (config -
4 probes would have needed 4 zooms and overwhelmed the picture, per author
feedback). The SAME probe row index and the SAME original-space k=7
neighbor set are used in all three panels; only the embedding coordinates
(and therefore which points end up near the probe) differ. Inside each
panel's zoom inset (`_local_spacing` / `_zoom_roles` / `main`'s per-panel
loop), every point near the probe is one of:
  (a) the probe itself - biggest marker, black outline (`zoom_probe_size`);
  (b) a TRUE neighbor - one of the probe's k=7 ORIGINAL-space nearest
      neighbors (kept or not) - single Okabe-Ito blue (`_TRUE_NEIGHBOR_COLOR`);
  (c) an INTRUDER - one of the probe's k=7 EMBEDDING nearest neighbors that
      is NOT a true neighbor - Okabe-Ito vermillion (`_INTRUDER_COLOR`);
  everything else keeps the regular curve-position color (viridis), small.
The zoom window is a disk of radius `zoom_radius_local_spacing_multiple`
(config, =10.0) times `_local_spacing` = the MEDIAN nearest-neighbor
distance among ALL points in THIS PANEL'S OWN embedding coordinates - a
multiplicative, per-panel radius, because the panels' absolute coordinate
scales differ hugely (t-SNE's coordinate units are not MDS's), but "10x
the local spacing" means the same relative thing in all three. This design
makes the failure visible directly: in the MDS panel the zoom window is
full of vermillion intruders crowding out a couple of blue true neighbors
near the edge; in the tuned alpha-Sammon panel blue true neighbors
dominate close to the probe. A grey rectangle in each main panel + a
`mark_inset` connector (`mpl_toolkits.axes_grid1.inset_locator`) shows
exactly where the zoom comes from.

Points are colored by the dataset's own continuous curve parameter
(`src/datasets/synthetic.py`'s `y` - theta for helix, t for s_curve), a
SINGLE shared color scale across all three panels, so a reader can see
directly that two similarly colored (= originally nearby) points end up
close together in a "good" panel and far apart in a "bad" one. This
curve-position coloring is untouched by the redesign above and applies to
every point except the probe/true-neighbor/intruder roles inside the zoom
insets (which use their own fixed marker colors so the three roles are
told apart from each other and from the ordinary cloud).

Below each panel: `neighbors kept: X.X / 7` and `stress: Y.YYYY` -
`neighbors_kept` = mean count of the k=7 original-space nearest neighbors
that are ALSO among the k=7 nearest neighbors in the embedding
(`src.sammon.metrics.knn_overlap_count`, the exact same definition and k as
`exp13_neighbor_survival.py`'s headline number); `stress_scale_invariant`
(`src.sammon.metrics.scale_invariant_stress`) is UNWEIGHTED (alpha=0
weighting under the hood, regardless of which alpha produced the
embedding), so it is directly comparable across all three panels/alphas -
verified by the author (see the task's data-provenance section). BOTH
numbers are computed over ALL n points (not just the probe point - the
probe and its zoom window are a single-point illustration, the numbers
below each panel are the full-dataset summary; this is called out
explicitly in the caption so the two are never conflated). BOTH are
recomputed HERE directly from the cached embeddings (never looked up from
exp13_neighbor_survival_results.csv), so this figure has no cross-experiment
provenance dependency on exp13 having been (re-)run in the same mode - it
only needs the embeddings themselves.

HONESTY (task requirement - do not soften this in any rewrite): this figure
must NOT read as "our method wins". t-SNE (panel 3) is included on purpose
because it also solves the neighborhood problem - better than the tuned
alpha-Sammon, in fact - but destroys the metric geometry entirely. The
message is that alpha is a KNOB between two goals (faithful global
geometry vs. preserved local neighborhoods), not a claim of superiority.

Suggested caption (English, for the article - tex-writer should adapt
wording to fit the surrounding text, but must preserve this framing):
    "The trade-off at the heart of this paper: distances vs. neighborhoods.
    All three panels embed the same 1000-point helix (points colored by
    position along the curve). One probe point (bold black outline) is
    examined with a zoomed inset (radius = 10x the panel's own local point
    spacing): blue markers are the probe's k=7 TRUE nearest neighbors from
    the original 3D space (same 7 points in every panel); vermillion
    markers are INTRUDERS - points that are among the probe's k=7 nearest
    neighbors in THIS embedding but are not true neighbors at all. MDS
    (alpha=0, left) minimizes distance distortion, but its zoom window is
    crowded with intruders that pushed the true (blue) neighbors toward the
    window's edge - true neighbors are not thrown far away, they are
    crowded out locally. Tuning alpha (center) restores most neighborhoods -
    the zoom window is now dominated by true (blue) neighbors - at a large
    stress cost. t-SNE (right) preserves neighborhoods best but abandons
    metric geometry (stress roughly 4x higher than MDS). The 'neighbors
    kept'/'stress' figures below each panel are full-dataset averages (all
    ~1000 points), not just the single highlighted probe. Weighting the
    stress function by distance is a KNOB between these two goals, not a
    method that wins on both."

Inputs (already-cached embeddings, NO new DR run):
  - `results/data/[<mode>/]embeddings/exp6_alpha_curves/` (MDS panel,
    always; tuned panel, when `tuned_alpha_experiment` == 'exp6_alpha_curves')
  - `results/data/[<mode>/]embeddings/exp12_alpha_grid_extension/` (tuned
    panel, only if `tuned_alpha_experiment` is set to
    'exp12_alpha_grid_extension' - NOT the production config: the tuned
    panel must show a value from the article's own alpha_auto search grid
    {0, 0.25, ..., 3} (`numAlphaGridPoints`=13, subsec:alpha_selection), so
    `config_experiments.yaml.fig_neighborhood_problem.tuned_alpha` is 3.0
    (the largest grid point) with `tuned_alpha_experiment: exp6_alpha_curves`.
    Reviewer note 2026-09-18: helix's oracle alpha_best=3.5 in
    exp13_neighbor_survival_results.csv is off-grid and was used here
    before this fix; verified that alpha=3.0 shows the same phenomenon
    (neighbors_kept 5.49/7 vs. 5.53/7 at 3.5, both up from 2.17/7 at
    alpha=0) so no information is lost by staying on-grid.)
  - `results/data/[<mode>/]embeddings/exp1_dr_benchmark/` (t-SNE panel)
All three must already exist for the configured `--quick/--full/--smoke`
mode (fail loud otherwise - `load_embedding`'s own FileNotFoundError).
NOTE: exp1_dr_benchmark.quick/exp6_alpha_curves.quick do not include
'helix' or 's_curve' at all (see config_experiments.yaml), so `--quick` is
NOT usable for this figure out of the box - only `--smoke` (which covers
every dataset, with a narrower alpha grid and `tsne_auto` instead of
`tsne` - see `fig_neighborhood_problem.smoke` in the config) and `--full`
(the production run this figure is meant for).

Output: results/figures/[<mode>/]fig_neighborhood_problem.pdf + a .csv with
the exact plotted (panel, point, summary) rows.

Run: venv\\python.exe -m src.figures.fig_neighborhood_problem [--quick|--full|--smoke]
or: src\\run_fig_neighborhood_problem.bat [quick|full|smoke]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from src.common.checkpoint import RunKey, load_embedding
from src.datasets.registry import load_dataset
from src.datasets.subsample import subsample_dataset
from src.experiments.config_experiments import resolve_experiment_config
from src.experiments.exp1_dr_benchmark import SUBSAMPLE_SEED
from src.experiments.exp_common import resolve_experiment_name
from src.methods.common import to_distance_matrix
from src.sammon.metrics import _neighbor_ranks, knn_overlap_count, scale_invariant_stress
from src.figures.fig_common import (
    OKABE_ITO,
    WIDTH_FULL_WIDTH_IN,
    add_quick_arg,
    display_label,
    parse_fig_mode,
    save_csv_alongside,
    save_figure,
)

FIG_NAME = "fig_neighborhood_problem"

PANEL_MDS = "mds"
PANEL_TUNED = "tuned"
PANEL_TSNE = "tsne"
PANEL_ORDER = [PANEL_MDS, PANEL_TUNED, PANEL_TSNE]

ROLE_REGULAR = "regular"
ROLE_PROBE = "probe"
ROLE_TRUE_NEIGHBOR = "true_neighbor"
ROLE_INTRUDER = "intruder"

# Fixed roles inside the zoom inset (task requirement: one saturated
# Okabe-Ito color per role, distinct from the viridis curve-position scale
# used by the ordinary point cloud). Index 5 = blue (true neighbors, the
# "good" role), index 6 = vermillion (intruders, the "bad" role that
# crowds true neighbors out) - chosen for maximum contrast against each
# other and against the mid-range viridis greens/yellows.
_TRUE_NEIGHBOR_COLOR = OKABE_ITO[5]
_INTRUDER_COLOR = OKABE_ITO[6]


def _alpha_method_name(alpha: float) -> str:
    """Same f"alpha{alpha}" convention as `exp6_alpha_curves._method_name_for_alpha`/
    `exp12_alpha_grid_extension` - the RunKey used to load a cached
    embedding MUST match the on-disk file name exactly."""
    return f"alpha{float(alpha)}"


def _select_probe_indices(y_color: np.ndarray, n_probes: int) -> np.ndarray:
    """Deterministic (NO randomness, NO config seed needed) selection of
    `n_probes` "probe" point indices, evenly spaced along the dataset's own
    continuous curve parameter `y_color` (theta for helix, t for s_curve):
    points are sorted by `y_color`, then `n_probes` positions evenly spaced
    over that sorted order (`np.linspace(0, n-1, n_probes)`, rounded) are
    picked. Special case `n_probes == 1`: "evenly spaced" degenerates to a
    single position at the MEDIAN of the sorted order (not an endpoint),
    since a lone probe at a curve endpoint would sit in a boundary region
    with fewer true neighbors on one side, for reasons unrelated to the
    method being illustrated. This is the SAME set of original-dataset row
    indices regardless of which panel/embedding is being drawn, so all
    three panels highlight exactly the same point(s) (see module
    docstring's "ONE PROBE, THREE ROLES" section)."""
    if n_probes < 1:
        raise ValueError(f"n_probes must be >= 1, got n_probes={n_probes}.")
    n = len(y_color)
    if n_probes > n:
        raise ValueError(f"n_probes={n_probes} exceeds the number of points n={n}.")
    order_by_curve = np.argsort(y_color, kind="stable")
    if n_probes == 1:
        positions = np.array([round((n - 1) / 2.0)], dtype=int)
    else:
        positions = np.round(np.linspace(0, n - 1, n_probes)).astype(int)
    probes = order_by_curve[positions]
    return np.unique(probes)  # np.unique also sorts - fine, order is not meaningful downstream


def _probe_colors(n_probes: int) -> list[str]:
    """One Okabe-Ito color per probe, reused cyclically if `n_probes`
    exceeds the palette size. Index 0 (black) is reserved for the probe
    marker OUTLINE (see `main`'s scatter call) and is skipped here so a
    probe's fill color never matches its own outline."""
    palette = OKABE_ITO[1:]
    return [palette[i % len(palette)] for i in range(n_probes)]


def _local_spacing(d_emb: np.ndarray) -> float:
    """Median nearest-neighbor distance among ALL points in THIS PANEL'S
    OWN embedding coordinates - the "local spacing" scale used to size the
    zoom window (module docstring's FOURTH-REVIEW measurement: true
    neighbors sit a median 3.8x / 90th-percentile 10.5x this distance from
    the probe). Computed the same way for every panel so a MULTIPLICATIVE
    zoom radius (`zoom_radius_local_spacing_multiple` in config) means the
    same relative thing in every panel even though the panels' absolute
    coordinate scales differ hugely (e.g. t-SNE vs. metric MDS)."""
    masked = np.array(d_emb, dtype=np.float64, copy=True)
    np.fill_diagonal(masked, np.inf)
    nn_dist = masked.min(axis=1)
    return float(np.median(nn_dist))


def _zoom_roles(order_orig: np.ndarray, order_emb: np.ndarray, probe_idx: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """(true_neighbors, intruders) row-index arrays for `probe_idx`:
    true_neighbors = ALL k of the probe's ORIGINAL-space nearest neighbors
    (kept and broken alike - role (b) in the module docstring); intruders =
    points among the probe's k EMBEDDING nearest neighbors that are NOT
    also original-space neighbors (role (c) - the points that crowd true
    neighbors out of the embedding's k-neighborhood). The two arrays are
    disjoint by construction (intruders explicitly excludes true-neighbor
    indices)."""
    true_neighbors = order_orig[probe_idx, :k]
    embedding_neighbors = order_emb[probe_idx, :k]
    intruders = embedding_neighbors[~np.isin(embedding_neighbors, true_neighbors)]
    return true_neighbors, intruders


def _true_neighbor_intrusion_stats(
    order_orig: np.ndarray, rank_emb: np.ndarray, d_emb: np.ndarray, k: int, local_spacing: float,
) -> tuple[float, float, float]:
    """(median_rank, median_distance_ratio, p90_distance_ratio) of a TRUE
    original-space neighbor inside the embedding's own order, aggregated
    over ALL n points and their k true neighbors alike (n*k pairs total) -
    the full-dataset "intrusion, not displacement" measurement referenced
    by the article caption (`clanek/sections/01_uvod.tex`,
    fig:neighborhood_problem) and by the module docstring's FOURTH-REVIEW
    note. For each point i and each of its k true original-space neighbors
    j = order_orig[i, k']: `rank_emb[i, j]` is j's rank (1 = nearest) from
    i's perspective in THIS panel's embedding, and
    `d_emb[i, j] / local_spacing` is j's embedding distance from i, in units
    of this panel's own median nearest-neighbor spacing (the same
    normalization `_local_spacing` uses for the zoom-window radius, so the
    two numbers in the caption and the zoom window mean the same relative
    thing). median_distance_ratio/p90_distance_ratio are the median/90th
    percentile of that ratio."""
    n = order_orig.shape[0]
    rows = np.arange(n)[:, None]
    true_neighbors_all = order_orig[:, :k]  # (n, k)
    ranks_of_true = rank_emb[rows, true_neighbors_all]
    dists_of_true = d_emb[rows, true_neighbors_all]
    ratios = dists_of_true / local_spacing
    return float(np.median(ranks_of_true)), float(np.median(ratios)), float(np.percentile(ratios, 90))


def _panel_metrics(D: np.ndarray, order_orig: np.ndarray, d_emb: np.ndarray, k: int) -> tuple[float, float]:
    """(neighbors_kept, stress_scale_invariant) computed DIRECTLY from the
    embedding's own distance matrix `d_emb` - the same functions/
    definitions as `exp13_neighbor_survival.py` (`knn_overlap_count`) and
    `exp6_alpha_curves.py` (`scale_invariant_stress`), recomputed here
    rather than looked up from exp13's output CSV (see the module
    docstring's "Below each panel" paragraph)."""
    _, order_emb = _neighbor_ranks(d_emb)
    neighbors_kept = knn_overlap_count(order_orig, order_emb, k)
    stress = scale_invariant_stress(D, d_emb)
    return neighbors_kept, stress


def _load_original_space(dataset_name: str, n_max: int, subsample_seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(D, order_orig, y_color) for `dataset_name` - D = the original
    Euclidean distance matrix (EXACTLY the exp6_alpha_curves/exp1_dr_benchmark
    loading pipeline: `load_dataset` + `subsample_dataset` with the shared
    `SUBSAMPLE_SEED`), order_orig = `_neighbor_ranks(D)`'s neighbor order,
    y_color = the dataset's own continuous curve parameter (theta for
    helix, t for s_curve - `src/datasets/synthetic.py`) used for point
    color."""
    ds = load_dataset(dataset_name)
    ds = subsample_dataset(ds, n_max=n_max, random_state=subsample_seed)
    if ds.kind != "vector":
        raise ValueError(f"fig_neighborhood_problem expects kind='vector' for dataset '{dataset_name}', got '{ds.kind}'.")
    if ds.y is None:
        raise ValueError(f"Dataset '{dataset_name}' has no continuous label y to color points by curve position.")
    D = np.asarray(to_distance_matrix(np.asarray(ds.X, dtype=np.float64), "vector"), dtype=np.float64)
    _, order_orig = _neighbor_ranks(D)
    return D, order_orig, np.asarray(ds.y, dtype=np.float64)


def _panel_title(panel: str, tuned_alpha: float, tsne_method: str) -> str:
    """Panel title text - reuses the SHARED `display_label` vocabulary
    (never a new ad-hoc string): panel 1's label matches
    `sammon_alpha0_smacof` ("MDS ($\\alpha=0$)"), panel 2's matches
    `sammon_alpha_auto` ("$\\alpha$-Sammon (tuned)" - the exact phrase
    required by the task spec, alpha rendered as mathtext per author feedback
    2026-09-17), panel 3's matches whichever t-SNE variant this mode
    actually uses (`tsne` full / `tsne_auto` smoke)."""
    if panel == PANEL_MDS:
        return display_label("sammon_alpha0_smacof", "method")
    if panel == PANEL_TUNED:
        return f"{display_label('sammon_alpha_auto', 'method')}\n($\\alpha$={tuned_alpha:g})"
    if panel == PANEL_TSNE:
        return display_label(tsne_method, "method")
    raise ValueError(f"Unknown panel='{panel}'.")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Flagship figure: metric methods keep distances but shred neighborhoods "
        "(MDS vs. tuned alpha-Sammon vs. t-SNE, with a zoomed probe-point inset)."
    )
    add_quick_arg(parser)
    args = parser.parse_args()
    mode = parse_fig_mode(args)

    cfg = resolve_experiment_config(FIG_NAME, mode)
    dataset_name = str(cfg["dataset"])
    k = int(cfg["k_neighbors"])
    seed = int(cfg["seed"])
    mds_alpha = float(cfg["mds_alpha"])
    tuned_alpha = float(cfg["tuned_alpha"])
    tuned_alpha_experiment = str(cfg["tuned_alpha_experiment"])
    tsne_method = str(cfg["tsne_method"])
    n_probes = int(cfg["n_probes"])
    probe_marker_size = float(cfg["probe_marker_size"])
    probe_marker_edgewidth = float(cfg["probe_marker_edgewidth"])
    point_size = float(cfg["point_size"])
    point_cmap = str(cfg["point_cmap"])
    zoom_radius_multiple = float(cfg["zoom_radius_local_spacing_multiple"])
    zoom_inset_size_pct = float(cfg["zoom_inset_size_pct"])
    zoom_inset_loc = str(cfg["zoom_inset_loc"])
    zoom_mark_inset_loc1 = int(cfg["zoom_mark_inset_loc1"])
    zoom_mark_inset_loc2 = int(cfg["zoom_mark_inset_loc2"])
    zoom_point_size = float(cfg["zoom_point_size"])
    zoom_true_neighbor_size = float(cfg["zoom_true_neighbor_size"])
    zoom_intruder_size = float(cfg["zoom_intruder_size"])
    zoom_probe_size = float(cfg["zoom_probe_size"])
    zoom_marker_edgewidth = float(cfg["zoom_marker_edgewidth"])

    # n_max is taken from exp6_alpha_curves' OWN (mode-resolved) config -
    # subsample_dataset is a no-op whenever n_native <= n_max (see
    # src/datasets/subsample.py), which holds for every mode this figure
    # supports (helix/s_curve have 1000 native points; exp6's n_max is
    # 2000 full / 150 smoke - both >= 1000... except smoke, where 150 < 1000
    # and it DOES subsample; exp1_dr_benchmark's smoke n_max is ALSO 150
    # with the SAME SUBSAMPLE_SEED, so the two caches still agree on
    # exactly which 150 points and in which order - verified by the
    # Y.shape[0] check below, never silently trusted).
    exp6_cfg = resolve_experiment_config("exp6_alpha_curves", mode)
    n_max = int(exp6_cfg["n_max"])

    exp6_name = resolve_experiment_name("exp6_alpha_curves", mode)
    exp1_name = resolve_experiment_name("exp1_dr_benchmark", mode)
    tuned_exp_name = resolve_experiment_name(tuned_alpha_experiment, mode)

    D, order_orig, y_color = _load_original_space(dataset_name, n_max, SUBSAMPLE_SEED)
    n_points = D.shape[0]

    panel_keys = {
        PANEL_MDS: RunKey(exp6_name, dataset_name, _alpha_method_name(mds_alpha), seed),
        PANEL_TUNED: RunKey(tuned_exp_name, dataset_name, _alpha_method_name(tuned_alpha), seed),
        PANEL_TSNE: RunKey(exp1_name, dataset_name, tsne_method, seed),
    }

    # Deterministic probe selection (NO RNG - see _select_probe_indices):
    # SAME probe index in all three panels, that is the entire point of the
    # comparison. n_probes=1 (config) - see the module docstring's FOURTH
    # REVIEW note on why a single probe + zoom replaced the earlier 4-probe,
    # whole-panel-edges design.
    probes = _select_probe_indices(y_color, n_probes)
    probe_idx = int(probes[0])
    probe_color = _probe_colors(len(probes))[0]
    is_probe = np.zeros(n_points, dtype=bool)
    is_probe[probes] = True

    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_FULL_WIDTH_IN, WIDTH_FULL_WIDTH_IN / 3.0 + 1.30))
    csv_rows: list[dict] = []
    scatter_mappable = None

    for ax, panel in zip(axes, PANEL_ORDER):
        key = panel_keys[panel]
        Y = np.asarray(load_embedding(key), dtype=np.float64)
        if Y.shape[0] != n_points:
            raise ValueError(
                f"Cached embedding for panel='{panel}' ({key.as_tuple()}) has {Y.shape[0]} rows, "
                f"expected {n_points} (n_max/subsample_seed mismatch between the source experiments)."
            )

        d_emb = np.asarray(to_distance_matrix(Y, "vector"), dtype=np.float64)
        # neighbors_kept/stress are FULL-DATASET summaries (all n_points),
        # NOT restricted to the probe point - see the module docstring's
        # "Below each panel" paragraph.
        neighbors_kept, stress = _panel_metrics(D, order_orig, d_emb, k)

        # Zoom-window role classification (module docstring's "ONE PROBE,
        # THREE ROLES" section): recompute this panel's OWN embedding
        # neighbor order (order_emb) to find which nearby points are true
        # neighbors vs. intruders in THIS panel specifically.
        rank_emb, order_emb = _neighbor_ranks(d_emb)
        true_neighbors, intruders = _zoom_roles(order_orig, order_emb, probe_idx, k)
        local_spacing = _local_spacing(d_emb)
        zoom_radius = zoom_radius_multiple * local_spacing
        center = Y[probe_idx]

        # Full-dataset "intrusion, not displacement" stats (module docstring's
        # FOURTH-REVIEW note, article caption fig:neighborhood_problem):
        # computed for every panel so the CSV documents all three, even
        # though the caption currently quotes only the MDS (alpha=0) panel.
        true_neighbor_median_rank, true_neighbor_median_dist_ratio, true_neighbor_p90_dist_ratio = (
            _true_neighbor_intrusion_stats(order_orig, rank_emb, d_emb, k, local_spacing)
        )

        role = np.full(n_points, ROLE_REGULAR, dtype=object)
        role[true_neighbors] = ROLE_TRUE_NEIGHBOR
        role[intruders] = ROLE_INTRUDER
        role[probe_idx] = ROLE_PROBE

        # Main panel: ALL points colored by curve position (viridis),
        # unchanged from before, plus the probe marked so the reader can
        # find where the zoom rectangle below comes from.
        scatter_mappable = ax.scatter(
            Y[~is_probe, 0], Y[~is_probe, 1], c=y_color[~is_probe], cmap=point_cmap,
            s=point_size, linewidths=0.0, zorder=1, rasterized=True,
        )
        ax.scatter(
            Y[probes, 0], Y[probes, 1], c=probe_color, s=probe_marker_size,
            edgecolors="black", linewidths=probe_marker_edgewidth, zorder=3, rasterized=True,
        )

        span = Y.max(axis=0) - Y.min(axis=0)
        pad = 0.06 * np.maximum(span, 1e-9)
        ax.set_xlim(float(Y[:, 0].min() - pad[0]), float(Y[:, 0].max() + pad[0]))
        ax.set_ylim(float(Y[:, 1].min() - pad[1]), float(Y[:, 1].max() + pad[1]))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(_panel_title(panel, tuned_alpha, tsne_method), fontsize=9)
        ax.set_xlabel(f"neighbors kept: {neighbors_kept:.1f} / {k}\nstress: {stress:.4f}", fontsize=7.5, labelpad=6)

        # Zoom inset: a window of radius `zoom_radius` around the probe, in
        # THIS PANEL'S OWN coordinates, with the three roles marked.
        axins = inset_axes(
            ax, width=f"{zoom_inset_size_pct:g}%", height=f"{zoom_inset_size_pct:g}%",
            loc=zoom_inset_loc, borderpad=0.6,
        )
        axins.scatter(
            Y[:, 0], Y[:, 1], c=y_color, cmap=point_cmap, s=zoom_point_size,
            linewidths=0.0, zorder=1, rasterized=True,
        )
        axins.scatter(
            Y[true_neighbors, 0], Y[true_neighbors, 1], c=_TRUE_NEIGHBOR_COLOR, marker="D",
            s=zoom_true_neighbor_size, edgecolors="black", linewidths=zoom_marker_edgewidth, zorder=3,
        )
        axins.scatter(
            Y[intruders, 0], Y[intruders, 1], c=_INTRUDER_COLOR, marker="s",
            s=zoom_intruder_size, edgecolors="black", linewidths=zoom_marker_edgewidth, zorder=3,
        )
        axins.scatter(
            Y[probe_idx, 0], Y[probe_idx, 1], c=probe_color, marker="o",
            s=zoom_probe_size, edgecolors="black", linewidths=probe_marker_edgewidth, zorder=4,
        )
        axins.set_xlim(center[0] - zoom_radius, center[0] + zoom_radius)
        axins.set_ylim(center[1] - zoom_radius, center[1] + zoom_radius)
        axins.set_aspect("equal", adjustable="box")
        axins.set_xticks([])
        axins.set_yticks([])
        for spine in axins.spines.values():
            spine.set_edgecolor("0.3")
            spine.set_linewidth(0.8)
        mark_inset(ax, axins, loc1=zoom_mark_inset_loc1, loc2=zoom_mark_inset_loc2, fc="none", ec="0.3", linewidth=0.6)

        for xi, yi, ci, ri in zip(Y[:, 0], Y[:, 1], y_color, role):
            csv_rows.append({
                "record_type": "point", "panel": panel, "x": xi, "y": yi, "color_value": ci, "role": ri,
                "neighbors_kept": np.nan, "stress_scale_invariant": np.nan,
                "local_spacing": np.nan, "zoom_radius": np.nan,
                "true_neighbor_median_rank": np.nan, "true_neighbor_median_distance_ratio": np.nan,
                "true_neighbor_p90_distance_ratio": np.nan,
            })
        csv_rows.append({
            "record_type": "summary", "panel": panel, "x": np.nan, "y": np.nan, "color_value": np.nan, "role": np.nan,
            "neighbors_kept": neighbors_kept, "stress_scale_invariant": stress,
            "local_spacing": local_spacing, "zoom_radius": zoom_radius,
            "true_neighbor_median_rank": true_neighbor_median_rank,
            "true_neighbor_median_distance_ratio": true_neighbor_median_dist_ratio,
            "true_neighbor_p90_distance_ratio": true_neighbor_p90_dist_ratio,
        })

    fig.subplots_adjust(left=0.03, right=0.90, top=0.83, bottom=0.28, wspace=0.10)

    cbar_ax = fig.add_axes((0.92, 0.30, 0.015, 0.42))
    cbar = fig.colorbar(scatter_mappable, cax=cbar_ax)
    cbar.set_label("Position along the curve (a.u.)", fontsize=7.5)
    cbar.ax.tick_params(labelsize=6.5)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=probe_color, markeredgecolor="black",
               markersize=7.5, label="probe point"),
        Line2D([0], [0], marker="D", linestyle="", markerfacecolor=_TRUE_NEIGHBOR_COLOR, markeredgecolor="black",
               markersize=6.5, label=f"true neighbor (original-space, k={k})"),
        Line2D([0], [0], marker="s", linestyle="", markerfacecolor=_INTRUDER_COLOR, markeredgecolor="black",
               markersize=6.5, label="intruder (embedding-only neighbor)"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=len(legend_handles), frameon=False, fontsize=7.5,
        bbox_to_anchor=(0.5, 0.09), columnspacing=1.2, handletextpad=0.4,
    )
    fig.text(
        0.5, 0.005,
        f"zoom window radius = {zoom_radius_multiple:g}x this panel's own local point spacing, same probe point in all 3 panels\n"
        "'neighbors kept'/'stress' below each panel are full-dataset averages, not the probe alone",
        ha="center", va="bottom", fontsize=6.8, linespacing=1.4,
    )

    fig.suptitle(
        f"Original-space k={k} nearest neighbors near one probe point on the {display_label(dataset_name, 'dataset')} dataset",
        fontsize=10, y=0.97,
    )

    save_figure(fig, FIG_NAME)
    save_csv_alongside(pd.DataFrame(csv_rows), FIG_NAME)


if __name__ == "__main__":
    main()
