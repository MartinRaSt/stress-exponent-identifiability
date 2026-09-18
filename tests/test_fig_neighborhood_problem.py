# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-17
# License: see the LICENSE file in the repository root
"""Tests for `src/figures/fig_neighborhood_problem.py`: the pure geometry/
styling helpers (probe selection, probe colors, edge segment construction,
alpha-method naming, panel titles) - no plotting, no disk I/O, so these run
without any cached embeddings present."""
from __future__ import annotations

import numpy as np
import pytest

from src.figures import fig_neighborhood_problem as fnp
from src.figures.fig_common import OKABE_ITO


def test_alpha_method_name_matches_exp6_alpha_curves_convention() -> None:
    assert fnp._alpha_method_name(0.0) == "alpha0.0"
    assert fnp._alpha_method_name(3.5) == "alpha3.5"
    assert fnp._alpha_method_name(1) == "alpha1.0"


def test_select_probe_indices_is_deterministic_and_evenly_spaced() -> None:
    # y_color already sorted 0..9 -> sorted order == identity.
    y = np.arange(10.0)
    probes = fnp._select_probe_indices(y, n_probes=4)
    expected = np.unique(np.round(np.linspace(0, 9, 4)).astype(int))
    np.testing.assert_array_equal(probes, expected)
    # deterministic: calling twice gives the identical result (no RNG).
    np.testing.assert_array_equal(probes, fnp._select_probe_indices(y, n_probes=4))


def test_select_probe_indices_uses_curve_order_not_row_order() -> None:
    # row order is reversed relative to curve position -> probes must be
    # picked by the SORTED curve value, not raw row index.
    y = np.array([9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    probes = fnp._select_probe_indices(y, n_probes=3)
    # sorted-by-y order is exactly the reverse of row index -> endpoints are
    # row 9 (smallest y) and row 0 (largest y).
    assert 9 in probes
    assert 0 in probes


def test_select_probe_indices_rejects_too_many_probes() -> None:
    with pytest.raises(ValueError, match="exceeds the number of points"):
        fnp._select_probe_indices(np.arange(3.0), n_probes=5)


def test_select_probe_indices_rejects_non_positive_count() -> None:
    with pytest.raises(ValueError, match="n_probes must be >= 1"):
        fnp._select_probe_indices(np.arange(5.0), n_probes=0)


def test_select_probe_indices_single_probe_picks_curve_median_not_endpoint() -> None:
    # n_probes=1 must NOT reuse the linspace(0, n-1, 1) formula (which would
    # pick position 0, a curve ENDPOINT) - it must pick the MEDIAN sorted
    # position (see the FOURTH-review note in _select_probe_indices).
    y = np.arange(11.0)  # sorted order == identity, median position = 5
    probes = fnp._select_probe_indices(y, n_probes=1)
    assert probes.shape == (1,)
    assert probes[0] == 5
    # deterministic: calling twice gives the identical result (no RNG).
    np.testing.assert_array_equal(probes, fnp._select_probe_indices(y, n_probes=1))


def test_probe_colors_are_okabe_ito_and_skip_black() -> None:
    colors = fnp._probe_colors(4)
    assert len(colors) == 4
    assert len(set(colors)) == 4
    assert OKABE_ITO[0] not in colors  # black reserved for the marker outline


def test_probe_colors_cycle_when_more_probes_than_palette_entries() -> None:
    n = len(OKABE_ITO)  # more probes than (palette - 1) forces a repeat
    colors = fnp._probe_colors(n)
    assert colors[0] == colors[len(OKABE_ITO) - 1]


def test_local_spacing_is_median_nearest_neighbor_distance() -> None:
    # 4 points on a line at 0, 1, 3, 6 -> nearest-neighbor distances are
    # 1 (pt0->pt1), 1 (pt1->pt0), 2 (pt2->pt1 or pt2->pt3... here pt2's
    # nearest is pt3 at distance 3, pt1 at distance 2 -> nearest=2), 3 (pt3->pt2).
    Y = np.array([[0.0], [1.0], [3.0], [6.0]])
    d_emb = np.abs(Y - Y.T)
    # nearest-neighbor distances per row: pt0->1 (d=1), pt1->0 (d=1),
    # pt2->1 (d=2), pt3->2 (d=3) -> sorted [1, 1, 2, 3] -> median 1.5
    assert fnp._local_spacing(d_emb) == pytest.approx(1.5)


def test_zoom_roles_true_neighbors_and_intruders_are_disjoint_and_correct() -> None:
    # probe=0; original-space true neighbors (k=2): rows [1, 2].
    order_orig = np.array([
        [1, 2, 3],
        [0, 2, 3],
        [0, 1, 3],
        [0, 1, 2],
    ])
    # embedding neighbors (k=2) for probe 0: rows [2, 3] -> 2 is a TRUE
    # neighbor (kept), 3 is an INTRUDER (not a true neighbor).
    order_emb = np.array([
        [2, 3, 1],
        [0, 2, 3],
        [0, 1, 3],
        [0, 1, 2],
    ])
    true_neighbors, intruders = fnp._zoom_roles(order_orig, order_emb, probe_idx=0, k=2)
    np.testing.assert_array_equal(np.sort(true_neighbors), np.array([1, 2]))
    np.testing.assert_array_equal(intruders, np.array([3]))
    assert set(true_neighbors.tolist()).isdisjoint(set(intruders.tolist()))


def test_panel_metrics_perfect_embedding_has_zero_stress_and_full_neighbors() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 2))
    from src.methods.common import to_distance_matrix
    from src.sammon.metrics import _neighbor_ranks

    D = to_distance_matrix(X, "vector")
    _, order_orig = _neighbor_ranks(D)
    neighbors_kept, stress = fnp._panel_metrics(D, order_orig, D, k=5)
    assert neighbors_kept == pytest.approx(5.0)
    assert stress == pytest.approx(0.0, abs=1e-10)


def test_panel_title_mds_and_tuned_and_tsne() -> None:
    # author feedback 2026-09-17 (symbol unification): "alpha=" -> mathtext $\alpha$
    assert fnp._panel_title(fnp.PANEL_MDS, tuned_alpha=3.5, tsne_method="tsne") == "MDS ($\\alpha=0$)"
    tuned_title = fnp._panel_title(fnp.PANEL_TUNED, tuned_alpha=3.5, tsne_method="tsne")
    assert "$\\alpha$-Sammon (tuned)" in tuned_title
    assert "$\\alpha$=3.5" in tuned_title
    assert fnp._panel_title(fnp.PANEL_TSNE, tuned_alpha=3.5, tsne_method="tsne") == "t-SNE (default)"


def test_panel_title_unknown_panel_raises() -> None:
    with pytest.raises(ValueError, match="Unknown panel"):
        fnp._panel_title("bogus", tuned_alpha=1.0, tsne_method="tsne")


def test_panel_order_has_exactly_three_panels_in_the_documented_order() -> None:
    assert fnp.PANEL_ORDER == [fnp.PANEL_MDS, fnp.PANEL_TUNED, fnp.PANEL_TSNE]
