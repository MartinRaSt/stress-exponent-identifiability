# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-18
# License: see the LICENSE file in the repository root
"""
E15 "downstream discovery task" (editor of DAMI: "there is not a single task
in the whole manuscript where a better map leads to a better FINDING").

Two questions an analyst actually asks while looking at a 2D map, answered
BOTH from the ground truth (input distance matrix D_in) and from the map
(output distance matrix D_out of the 2D embedding Y):

  Q1 `nearest_class_pair`  - "Which two classes are closest to each other?"
      argmin over unordered class pairs (a, b) of the MEAN pairwise distance
      between class a and class b (same definition as
      `src.sammon.cluster_geometry.centroid_dist_spearman`'s underlying
      quantity - between_class_mean_distance_matrix).
  Q2 `most_dispersed_class` - "Which class is the most spread out?"
      argmax over classes of the median within-class pairwise distance
      (same definition as `src.sammon.cluster_geometry.class_spread_spearman`'s
      underlying quantity - within_class_median_distance).

Both questions are deliberately built on the SAME distance aggregates already
used for the K2 cluster-geometry metrics (`src/sammon/cluster_geometry.py`)
- no new notion of "distance between classes"/"class spread" is introduced,
so the "hard" (argmin/argmax) and "soft" (rank correlation) versions of a
question are directly comparable: the soft score of Q1 IS
`centroid_dist_spearman`, the soft score of Q2 IS `class_spread_spearman`
(reused, never recomputed differently).

The "hard" analyst answer is 0/1 correct: does the argmin/argmax computed
from D_out (the map) match the one computed from D_in (the truth)? On an
EXACT tie the deterministic (stable-sort, lowest-index) choice is used as
the single canonical truth/prediction - no randomization, so the score is
reproducible; `tie_in_truth`/`tie_in_prediction` are recorded separately so
a tie is visible in the output rather than silently hidden inside a coin
flip.

Class-count bounds and definitions of "insufficient"/"excessive classes"
are the SAME as `cluster_geometry` (`MIN_CLASSES_FOR_GEOMETRY`,
`MAX_CLASSES_FOR_GEOMETRY`) - reused directly, not duplicated, so a dataset
is included/excluded from E15 for exactly the same reason as from K2.
"""
from __future__ import annotations

import numpy as np

from src.sammon.cluster_geometry import (
    MAX_CLASSES_FOR_GEOMETRY,
    MIN_CLASSES_FOR_GEOMETRY,
    between_class_mean_distance_matrix,
    centroid_dist_spearman,
    class_spread_spearman,
    within_class_median_distance,
)

QUESTION_NEAREST_PAIR = "nearest_class_pair"
QUESTION_MOST_DISPERSED = "most_dispersed_class"
DISCOVERY_QUESTIONS = (QUESTION_NEAREST_PAIR, QUESTION_MOST_DISPERSED)

# Row schema returned by `discovery_task_rows` for EACH question (in
# addition to 'question' itself) - the canonical column list the experiment
# script uses to build the CSV schema (`check_results_schema`).
DISCOVERY_TASK_KEYS = [
    "n_classes", "note", "n_candidates",
    "true_answer", "pred_answer", "correct",
    "soft_rho", "tie_in_truth", "tie_in_prediction",
]

NOTE_INSUFFICIENT_CLASSES = "insufficient_classes_or_no_labels"
NOTE_EXCESSIVE_CLASSES = "excessive_classes_likely_continuous_label"
NOTE_INSUFFICIENT_SPREAD_CANDIDATES = "fewer_than_2_classes_with_2plus_points"


def _pick_extremum(values: np.ndarray, tie_relative_tolerance: float, kind: str) -> tuple[int, bool]:
    """Return (index_of_extremum, is_tied) over `values` (1D, no NaN).
    `kind`='min' or 'max'. The extremum is chosen deterministically (stable
    sort, lowest original index among tied values) so the result never
    depends on run-to-run non-determinism. `is_tied`=True if the runner-up
    value is within `tie_relative_tolerance` (relative to the extremum's
    magnitude) of the extremum - purely informational, does NOT change which
    index is returned."""
    if values.shape[0] == 0:
        raise ValueError("_pick_extremum: empty `values` array.")
    signed = values if kind == "min" else -values
    order = np.argsort(signed, kind="stable")
    best_idx = int(order[0])
    tied = False
    if values.shape[0] > 1:
        best_val = float(signed[order[0]])
        second_val = float(signed[order[1]])
        denom = max(abs(best_val), 1e-12)
        tied = bool(abs(second_val - best_val) <= tie_relative_tolerance * denom)
    return best_idx, tied


def _nearest_pair_answer(D: np.ndarray, y: np.ndarray, classes: np.ndarray, tie_relative_tolerance: float) -> tuple[str, int, bool]:
    """Return (answer, n_candidates, tie) for Q1 on distance matrix `D`.
    `answer` = 'classA|classB' with classA/classB in the SAME order as
    `classes` (ascending, from `np.unique`) - so the identical unordered
    pair always serializes to the identical string on both the true and the
    predicted side, and `correct` can be scored by exact string equality."""
    _, M = between_class_mean_distance_matrix(D, y)
    k = len(classes)
    iu = np.triu_indices(k, k=1)
    vals = M[iu]
    best, tied = _pick_extremum(vals, tie_relative_tolerance, "min")
    a_idx, b_idx = int(iu[0][best]), int(iu[1][best])
    answer = f"{classes[a_idx]}|{classes[b_idx]}"
    return answer, int(vals.shape[0]), tied


def _most_dispersed_answer(
    D: np.ndarray, y: np.ndarray, classes: np.ndarray, valid_mask: np.ndarray, tie_relative_tolerance: float,
) -> tuple[str, bool]:
    """Return (answer, tie) for Q2 on distance matrix `D`, restricted to
    `valid_mask` (classes with >= 2 points, precomputed ONCE from `y` since
    it does not depend on D - see the caller)."""
    _, spread = within_class_median_distance(D, y)
    vals = spread[valid_mask]
    idx_map = np.where(valid_mask)[0]
    best_local, tied = _pick_extremum(vals, tie_relative_tolerance, "max")
    best_idx = int(idx_map[best_local])
    return f"{classes[best_idx]}", tied


def _nan_row(n_classes: int, note: str) -> dict:
    row = {k: float("nan") for k in DISCOVERY_TASK_KEYS}
    row["n_classes"] = n_classes
    row["note"] = note
    return row


def discovery_task_rows(D_in: np.ndarray, D_out: np.ndarray, y: np.ndarray | None, tie_relative_tolerance: float) -> list[dict]:
    """Compute both E15 questions at once. Returns a list of exactly 2 dicts
    (one per `DISCOVERY_QUESTIONS`, in that order), each with the keys
    'question' + `DISCOVERY_TASK_KEYS`.

    A dataset that does not qualify (no labels / < MIN_CLASSES_FOR_GEOMETRY /
    > MAX_CLASSES_FOR_GEOMETRY, exactly the K2 cluster-geometry thresholds)
    gets NaN metric rows with an explanatory `note` for BOTH questions - the
    row is still emitted (never silently dropped), so the CSV always has a
    complete dataset x method x seed x question grid to join against."""
    if y is None:
        n_classes = 0
    else:
        y = np.asarray(y)
        n_classes = int(len(np.unique(y)))

    if not (MIN_CLASSES_FOR_GEOMETRY <= n_classes <= MAX_CLASSES_FOR_GEOMETRY):
        note = NOTE_INSUFFICIENT_CLASSES if n_classes < MIN_CLASSES_FOR_GEOMETRY else NOTE_EXCESSIVE_CLASSES
        return [{"question": q, **_nan_row(n_classes, note)} for q in DISCOVERY_QUESTIONS]

    classes = np.unique(y)
    rows: list[dict] = []

    # --- Q1: nearest_class_pair ---
    true_answer, n_candidates, tie_truth = _nearest_pair_answer(D_in, y, classes, tie_relative_tolerance)
    pred_answer, _, tie_pred = _nearest_pair_answer(D_out, y, classes, tie_relative_tolerance)
    soft_rho = centroid_dist_spearman(D_in, D_out, y)
    rows.append({
        "question": QUESTION_NEAREST_PAIR, "n_classes": n_classes, "note": "",
        "n_candidates": n_candidates, "true_answer": true_answer, "pred_answer": pred_answer,
        "correct": int(true_answer == pred_answer), "soft_rho": soft_rho,
        "tie_in_truth": tie_truth, "tie_in_prediction": tie_pred,
    })

    # --- Q2: most_dispersed_class --- (validity of a class = >=2 points,
    # depends only on y, NOT on D, so it is identical on the true/pred side)
    _, counts = np.unique(y, return_counts=True)
    valid_mask = counts >= 2
    if int(valid_mask.sum()) < 2:
        rows.append({"question": QUESTION_MOST_DISPERSED, **_nan_row(n_classes, NOTE_INSUFFICIENT_SPREAD_CANDIDATES)})
    else:
        true_answer2, tie_truth2 = _most_dispersed_answer(D_in, y, classes, valid_mask, tie_relative_tolerance)
        pred_answer2, tie_pred2 = _most_dispersed_answer(D_out, y, classes, valid_mask, tie_relative_tolerance)
        soft_rho2 = class_spread_spearman(D_in, D_out, y)
        rows.append({
            "question": QUESTION_MOST_DISPERSED, "n_classes": n_classes, "note": "",
            "n_candidates": int(valid_mask.sum()), "true_answer": true_answer2, "pred_answer": pred_answer2,
            "correct": int(true_answer2 == pred_answer2), "soft_rho": soft_rho2,
            "tie_in_truth": tie_truth2, "tie_in_prediction": tie_pred2,
        })

    return rows
