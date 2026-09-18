# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-17
# License: see the LICENSE file in the repository root
"""Tests for `src.figures.fig_cd_diagram._label_rows` (author feedback
2026-09-17: the previous CD diagram drew a bar per pairwise insignificant
comparison instead of per maximal clique, and method labels overlapped the
bars). `_label_rows` assigns each method's label to a row (fanning the
better-ranked half left, the worse-ranked half right, Demsar 2006 Fig. 1
layout) - these tests pin down that both halves fan out symmetrically and
no two labels on the SAME side ever collide on the same row."""
from __future__ import annotations

from src.figures.fig_cd_diagram import _label_rows


def test_best_and_worst_method_sit_closest_to_the_axis() -> None:
    # n=6: left half = indices 0,1,2 (best ranks); right half = indices 3,4,5
    rows = _label_rows(6)
    assert rows[0] == 0  # best-ranked method: shortest connector on the left
    assert rows[5] == 0  # worst-ranked method: shortest connector on the right


def test_rows_fan_out_monotonically_within_each_side() -> None:
    rows = _label_rows(7)
    n_left = 4  # ceil(7/2)
    left_rows = rows[:n_left]
    right_rows = rows[n_left:]
    assert left_rows == sorted(left_rows)  # increasing distance from the axis
    assert right_rows == sorted(right_rows, reverse=True)  # increasing distance, read right-to-left


def test_no_row_collisions_within_a_side() -> None:
    for n in (2, 3, 4, 5, 8, 13, 20):
        rows = _label_rows(n)
        n_left = -(-n // 2)  # ceil
        assert len(set(rows[:n_left])) == n_left
        assert len(set(rows[n_left:])) == n - n_left


def test_odd_method_count_splits_larger_half_left() -> None:
    # Demsar's convention: the extra method (odd n) goes to the better-ranked half
    rows = _label_rows(5)
    assert max(rows[:3]) == 2  # left half has 3 methods -> rows 0,1,2
    assert max(rows[3:]) == 1  # right half has 2 methods -> rows 0,1
