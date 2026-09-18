# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-17
# License: see the LICENSE file in the repository root
"""Tests for src/experiments/exp13_neighbor_survival.py
(documentation/2026-09-17_zadani_exp13_preziti_sousedu.md): the CSV schema,
the two-panel policy table aggregation (including the "never fabricate a
missing alpha_pred/alpha_best row" rule), mode isolation (smoke/quick
outputs never touch the full paths), and checkpoint/resume.

A full end-to-end run needs completed dataset_properties/exp6_alpha_curves
data (author-run only, per project rule) - these tests instead exercise the
pure/testable pieces directly, plus one fully-mocked `_run_single` call to
check the wiring (dataset -> cached embedding -> order_orig/order_emb ->
knn_overlap_count -> row) without touching any real cache."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.experiments.exp13_neighbor_survival import (
    ALPHA_BEST_SOURCE_EXP6,
    ALPHA_BEST_SOURCE_EXP12,
    BASE_EXPERIMENT_NAME,
    COLUMN_KEYS,
    DATASET_SET_CORE,
    DATASET_SET_HOLDOUT,
    EXP6_BASE_NAME,
    EXP12_BASE_NAME,
    PANEL_LOW_RATIO,
    PANEL_REST,
    POLICY_LABEL_BEST,
    POLICY_LABEL_PRED,
    _collect_dataset_alphas,
    _dataset_order_orig,
    _merge_alpha_median_auc,
    _method_name_for_alpha,
    _method_name_for_task,
    _pick_alpha_best,
    _run_single,
    _select_per_dataset_alpha,
    _try_load_e12_source,
    build_survival_table,
    dataset_set_label,
    policy_label,
)

_LOGGER = logging.getLogger("test_exp13_neighbor_survival")

# ============================================================================
# CSV schema
# ============================================================================

_REQUIRED_COLUMNS = [
    "alpha", "k", "n_samples", "regime", "nn_ratio_k1", "alpha_best_source", "dataset_set",
    "neighbors_kept", "neighbors_kept_frac", "knn_jaccard_reference",
    "stress_scale_invariant", "shepard_spearman_rho", "q_global",
]


def test_column_keys_matches_required_schema() -> None:
    assert COLUMN_KEYS == _REQUIRED_COLUMNS


# ============================================================================
# --datasets all|core|holdout (task 2026-09-17: E13 must cover the same 51
# datasets as exp6_alpha_curves_results.csv/numExpSixDatasets by default, not
# just the 32-dataset 'core' list)
# ============================================================================


def test_dataset_set_label_core_vs_holdout() -> None:
    holdout = {"banknote", "twin_peaks"}
    assert dataset_set_label("banknote", holdout) == DATASET_SET_HOLDOUT
    assert dataset_set_label("twin_peaks", holdout) == DATASET_SET_HOLDOUT
    assert dataset_set_label("iris", holdout) == DATASET_SET_CORE


def test_dataset_set_label_empty_holdout_is_always_core() -> None:
    assert dataset_set_label("iris", set()) == DATASET_SET_CORE


def test_add_dataset_scope_arg_default_is_all_51_datasets() -> None:
    """The module docstring/task decision: exp13's default `--datasets` scope
    is 'all' (core+holdout), NOT 'core' only - unlike leaving it unspecified
    previously silently meaning 'only the 32 core datasets' (exp6_cfg["datasets"])."""
    import argparse

    from src.experiments.exp_common import add_dataset_scope_arg, resolve_dataset_scope

    parser = argparse.ArgumentParser()
    add_dataset_scope_arg(parser)
    args = parser.parse_args([])
    assert args.datasets == "all"

    core = [f"core{i}" for i in range(32)]
    holdout = [f"holdout{i}" for i in range(19)]
    scope = resolve_dataset_scope(args, core, holdout)
    assert len(scope) == 51
    assert scope[:32] == core
    assert scope[32:] == holdout


def test_exp6_base_name_constant_points_at_exp6_alpha_curves() -> None:
    assert EXP6_BASE_NAME == "exp6_alpha_curves"


def test_exp12_base_name_constant_points_at_exp12_alpha_grid_extension() -> None:
    assert EXP12_BASE_NAME == "exp12_alpha_grid_extension"


def test_method_name_helpers_encode_alpha_and_k() -> None:
    assert _method_name_for_alpha(1.0) == "alpha1.0"
    assert _method_name_for_task(1.0, 7) == "alpha1.0_k7"
    assert _method_name_for_task(0.75, 5) == "alpha0.75_k5"


def test_policy_label_uses_known_names_and_falls_back_generically() -> None:
    assert policy_label(0.0) == "alpha=0 (MDS)"
    assert policy_label(1.0) == "alpha=1 (Sammon)"
    assert policy_label(2.0) == "alpha=2 (Kamada-Kawai)"
    assert policy_label(0.75) == "alpha=0.75"


# ============================================================================
# _select_per_dataset_alpha
# ============================================================================


def test_select_per_dataset_alpha_picks_matching_alpha_per_dataset() -> None:
    df = pd.DataFrame({
        "dataset": ["a", "a", "b", "b"],
        "alpha": [0.0, 2.5, 0.0, 1.0],
        "neighbors_kept": [1.0, 4.0, 9.0, 10.0],
    })
    out = _select_per_dataset_alpha(df, {"a": 2.5, "b": 1.0})
    assert set(out["dataset"]) == {"a", "b"}
    assert out.set_index("dataset")["neighbors_kept"].to_dict() == {"a": 4.0, "b": 10.0}


def test_select_per_dataset_alpha_never_fabricates_a_missing_row() -> None:
    """A dataset whose requested alpha has no matching row (e.g. a --smoke
    grid gap) must be ABSENT from the result, not filled with a fallback."""
    df = pd.DataFrame({"dataset": ["a"], "alpha": [0.0], "neighbors_kept": [1.0]})
    out = _select_per_dataset_alpha(df, {"a": 0.0, "b": 0.75})
    assert list(out["dataset"]) == ["a"]


# ============================================================================
# alpha_best oracle over the combined exp6+exp12 grid (task 2026-09-17)
# ============================================================================


def test_pick_alpha_best_picks_highest_auc_regardless_of_source() -> None:
    auc_by_alpha = {0.0: (0.5, "exp6"), 3.0: (0.6, "exp6"), 4.5: (0.7, "exp12")}
    alpha_best, auc_best, source_best = _pick_alpha_best(auc_by_alpha)
    assert alpha_best == pytest.approx(4.5)
    assert auc_best == pytest.approx(0.7)
    assert source_best == "exp12"


def test_pick_alpha_best_stays_in_exp6_when_no_exp12_alpha_wins() -> None:
    auc_by_alpha = {0.0: (0.5, "exp6"), 1.0: (0.9, "exp6"), 3.5: (0.6, "exp12")}
    alpha_best, _auc_best, source_best = _pick_alpha_best(auc_by_alpha)
    assert alpha_best == pytest.approx(1.0)
    assert source_best == "exp6"


def test_merge_alpha_median_auc_unions_disjoint_grids_and_tags_source() -> None:
    e6 = {"a": {0.0: 0.5, 1.0: 0.6}, "b": {0.0: 0.9}}
    e12 = {"a": {3.5: 0.7}}
    merged = _merge_alpha_median_auc(e6, e12)

    assert merged["a"] == {0.0: (0.5, ALPHA_BEST_SOURCE_EXP6), 1.0: (0.6, ALPHA_BEST_SOURCE_EXP6), 3.5: (0.7, ALPHA_BEST_SOURCE_EXP12)}
    # dataset 'b' is absent from exp12 entirely - stays exp6-only, not fabricated
    assert merged["b"] == {0.0: (0.9, ALPHA_BEST_SOURCE_EXP6)}


def test_merge_alpha_median_auc_none_e12_behaves_like_exp6_only() -> None:
    e6 = {"a": {0.0: 0.5, 1.0: 0.6}}
    merged = _merge_alpha_median_auc(e6, None)
    assert merged == {"a": {0.0: (0.5, ALPHA_BEST_SOURCE_EXP6), 1.0: (0.6, ALPHA_BEST_SOURCE_EXP6)}}


def test_merge_alpha_median_auc_raises_on_overlapping_alpha() -> None:
    """The exp6/exp12 alpha grids are supposed to be disjoint by construction
    (exp12 only extends alpha beyond exp6's cap) - an overlap is a
    configuration bug, not something to silently resolve by picking one."""
    e6 = {"a": {3.0: 0.5}}
    e12 = {"a": {3.0: 0.9}}
    with pytest.raises(ValueError, match="present in BOTH"):
        _merge_alpha_median_auc(e6, e12)


def test_try_load_e12_source_returns_none_and_warns_when_csv_missing(tmp_path, monkeypatch, caplog) -> None:
    from src.experiments import exp13_neighbor_survival as mod

    monkeypatch.setattr(mod, "results_csv_path", lambda experiment_name: tmp_path / f"{experiment_name}_results.csv")
    with caplog.at_level(logging.WARNING, logger=_LOGGER.name):
        median_auc, rows = _try_load_e12_source("exp12_alpha_grid_extension", canonical_seed=0, k_values=[7], logger=_LOGGER)

    assert median_auc is None
    assert rows == {}
    assert any("exp12_alpha_grid_extension" in r.message for r in caplog.records)


def test_try_load_e12_source_loads_median_auc_and_rows_when_csv_present(tmp_path, monkeypatch) -> None:
    from src.experiments import exp13_neighbor_survival as mod

    csv_path = tmp_path / "exp12_alpha_grid_extension_results.csv"
    df = pd.DataFrame([
        {
            "dataset": "a", "alpha": 3.5, "seed": 0, "status": "ok", "auc_rnx": 0.8,
            "stress_scale_invariant": 0.1, "shepard_spearman_rho": 0.9, "q_global": 0.7, "knn_jaccard_k7": 0.6,
        },
        {
            "dataset": "a", "alpha": 3.5, "seed": 1, "status": "ok", "auc_rnx": 0.6,
            "stress_scale_invariant": 0.2, "shepard_spearman_rho": 0.8, "q_global": 0.6, "knn_jaccard_k7": 0.5,
        },
    ])
    df.to_csv(csv_path, index=False)
    monkeypatch.setattr(mod, "results_csv_path", lambda experiment_name: csv_path)

    median_auc, rows = _try_load_e12_source("exp12_alpha_grid_extension", canonical_seed=0, k_values=[7], logger=_LOGGER)

    assert median_auc == {"a": {3.5: pytest.approx(0.7)}}  # median(0.8, 0.6)
    assert set(rows) == {("a", 3.5)}
    assert rows[("a", 3.5)]["seed"] == 0  # only the canonical seed's row


# ============================================================================
# _collect_dataset_alphas (per-dataset alpha list with an explicit source,
# never an opportunistic exp6/exp12 fallback search)
# ============================================================================


def test_collect_dataset_alphas_dedupes_and_tags_exp6_alphas() -> None:
    info = {"alpha_pred": 2.0, "alpha_best": 2.0, "alpha_best_source": ALPHA_BEST_SOURCE_EXP6}
    out = _collect_dataset_alphas([0.0, 1.0, 2.0], info)
    assert out == [(0.0, ALPHA_BEST_SOURCE_EXP6), (1.0, ALPHA_BEST_SOURCE_EXP6), (2.0, ALPHA_BEST_SOURCE_EXP6)]


def test_collect_dataset_alphas_keeps_exp12_alpha_best_as_a_separate_entry() -> None:
    info = {"alpha_pred": 1.5, "alpha_best": 4.5, "alpha_best_source": ALPHA_BEST_SOURCE_EXP12}
    out = _collect_dataset_alphas([0.0, 1.0, 2.0], info)
    assert out == [
        (0.0, ALPHA_BEST_SOURCE_EXP6), (1.0, ALPHA_BEST_SOURCE_EXP6), (2.0, ALPHA_BEST_SOURCE_EXP6),
        (1.5, ALPHA_BEST_SOURCE_EXP6), (4.5, ALPHA_BEST_SOURCE_EXP12),
    ]


# ============================================================================
# build_survival_table (documentation section 3: two panels x 5 policy rows)
# ============================================================================


def _make_synthetic_results() -> pd.DataFrame:
    """a1/a2 = 'low_ratio' panel, b1/b2 = pooled 'rest' panel; b2 deliberately
    has NO row at its own alpha_pred/alpha_best (0.75) - it must therefore
    be excluded from those two policy rows' median, not fabricated."""
    rows = [
        {"dataset": "a1", "alpha": 0.0, "neighbors_kept": 1.0},
        {"dataset": "a1", "alpha": 1.0, "neighbors_kept": 2.0},
        {"dataset": "a1", "alpha": 2.0, "neighbors_kept": 3.0},
        {"dataset": "a1", "alpha": 2.5, "neighbors_kept": 4.0},
        {"dataset": "a2", "alpha": 0.0, "neighbors_kept": 5.0},
        {"dataset": "a2", "alpha": 1.0, "neighbors_kept": 6.0},
        {"dataset": "a2", "alpha": 2.0, "neighbors_kept": 7.0},
        {"dataset": "a2", "alpha": 2.5, "neighbors_kept": 8.0},
        {"dataset": "b1", "alpha": 0.0, "neighbors_kept": 9.0},
        {"dataset": "b1", "alpha": 1.0, "neighbors_kept": 10.0},
        {"dataset": "b1", "alpha": 2.0, "neighbors_kept": 11.0},
        {"dataset": "b2", "alpha": 0.0, "neighbors_kept": 12.0},
    ]
    df = pd.DataFrame(rows)
    df["k"] = 7
    df["status"] = "ok"
    df["stress_scale_invariant"] = df["neighbors_kept"] * 0.1
    df["shepard_spearman_rho"] = df["neighbors_kept"] * 0.01
    df["q_global"] = df["neighbors_kept"] * 0.001
    return df


_POLICY_INFO = {
    "a1": {"regime": "low_ratio", "alpha_pred": 2.5, "alpha_best": 2.0},
    "a2": {"regime": "low_ratio", "alpha_pred": 2.5, "alpha_best": 1.0},
    "b1": {"regime": "mid_ratio", "alpha_pred": 0.0, "alpha_best": 0.0},
    "b2": {"regime": "high_ratio", "alpha_pred": 0.75, "alpha_best": 0.75},
}


def test_build_survival_table_has_two_panels_and_five_policy_rows_each() -> None:
    df = _make_synthetic_results()
    table = build_survival_table(df, _POLICY_INFO, alpha_table=[0.0, 1.0, 2.0], k=7)

    assert set(table["panel"]) == {PANEL_LOW_RATIO, PANEL_REST}
    for panel in (PANEL_LOW_RATIO, PANEL_REST):
        policies = set(table.loc[table["panel"] == panel, "policy"])
        assert policies == {policy_label(0.0), policy_label(1.0), policy_label(2.0), POLICY_LABEL_PRED, POLICY_LABEL_BEST}


def test_build_survival_table_medians_match_hand_computed_values() -> None:
    df = _make_synthetic_results()
    table = build_survival_table(df, _POLICY_INFO, alpha_table=[0.0, 1.0, 2.0], k=7).set_index(["panel", "policy"])

    # low_ratio panel: a1, a2
    assert table.loc[(PANEL_LOW_RATIO, policy_label(0.0)), "neighbors_kept"] == pytest.approx(3.0)  # median(1,5)
    assert table.loc[(PANEL_LOW_RATIO, policy_label(1.0)), "neighbors_kept"] == pytest.approx(4.0)  # median(2,6)
    assert table.loc[(PANEL_LOW_RATIO, policy_label(2.0)), "neighbors_kept"] == pytest.approx(5.0)  # median(3,7)
    assert table.loc[(PANEL_LOW_RATIO, POLICY_LABEL_PRED), "neighbors_kept"] == pytest.approx(6.0)  # median(a1@2.5=4, a2@2.5=8)
    assert table.loc[(PANEL_LOW_RATIO, POLICY_LABEL_BEST), "neighbors_kept"] == pytest.approx(4.5)  # median(a1@2.0=3, a2@1.0=6)
    assert table.loc[(PANEL_LOW_RATIO, POLICY_LABEL_PRED), "n_datasets"] == 2
    assert table.loc[(PANEL_LOW_RATIO, POLICY_LABEL_BEST), "n_datasets"] == 2

    # rest panel: b1, b2 (b2 only has an alpha=0 row)
    assert table.loc[(PANEL_REST, policy_label(0.0)), "neighbors_kept"] == pytest.approx(10.5)  # median(9,12)
    assert table.loc[(PANEL_REST, policy_label(1.0)), "neighbors_kept"] == pytest.approx(10.0)  # only b1 has alpha=1
    assert table.loc[(PANEL_REST, policy_label(1.0)), "n_datasets"] == 1
    # b1's alpha_pred=alpha_best=0.0 -> reuses its own alpha=0 row (9.0); b2's
    # alpha_pred=alpha_best=0.75 has NO matching row -> excluded, not fabricated.
    assert table.loc[(PANEL_REST, POLICY_LABEL_PRED), "neighbors_kept"] == pytest.approx(9.0)
    assert table.loc[(PANEL_REST, POLICY_LABEL_PRED), "n_datasets"] == 1
    assert table.loc[(PANEL_REST, POLICY_LABEL_BEST), "neighbors_kept"] == pytest.approx(9.0)
    assert table.loc[(PANEL_REST, POLICY_LABEL_BEST), "n_datasets"] == 1


def test_build_survival_table_filters_to_requested_k_and_ok_status() -> None:
    df = _make_synthetic_results()
    extra = df.iloc[[0]].copy()
    extra["k"] = 5
    extra["neighbors_kept"] = 999.0
    bad = df.iloc[[0]].copy()
    bad["status"] = "error"
    bad["neighbors_kept"] = -999.0
    df_mixed = pd.concat([df, extra, bad], ignore_index=True)

    table = build_survival_table(df_mixed, _POLICY_INFO, alpha_table=[0.0, 1.0, 2.0], k=7)
    assert (table["k"] == 7).all()
    # the k=5 and status='error' rows must never leak into the k=7 medians
    assert 999.0 not in table["neighbors_kept"].values
    assert -999.0 not in table["neighbors_kept"].values


def test_build_survival_table_scope_datasets_excludes_rows_outside_scope() -> None:
    """A stale results CSV may contain rows for a dataset that is NOT part of
    the currently requested --datasets scope (e.g. leftover 'core'-only rows
    from an earlier run, now requesting --datasets=holdout) - `scope_datasets`
    must exclude them from the aggregation entirely, not just from panel
    membership."""
    df = _make_synthetic_results()
    # only restrict to a1/a2 (drop b1/b2 from the low_ratio/rest computation)
    table = build_survival_table(
        df, _POLICY_INFO, alpha_table=[0.0, 1.0, 2.0], k=7, scope_datasets={"a1", "a2"},
    ).set_index("panel")
    # PANEL_REST (b1/b2) has no dataset left in scope -> every one of its rows is empty (n_datasets=0)
    assert (table.loc[[PANEL_REST], "n_datasets"] == 0).all()
    assert table.loc[[PANEL_REST], "neighbors_kept"].isna().all()
    assert (table.loc[[PANEL_LOW_RATIO], "n_datasets"] == 2).all()


def test_build_survival_table_scope_datasets_none_keeps_previous_behavior() -> None:
    """`scope_datasets=None` (the default) must reproduce the exact same
    result as calling without the parameter at all (back-compat for existing
    callers/tests that already pre-filter both `df` and `policy_info`)."""
    df = _make_synthetic_results()
    with_none = build_survival_table(df, _POLICY_INFO, alpha_table=[0.0, 1.0, 2.0], k=7, scope_datasets=None)
    without_arg = build_survival_table(df, _POLICY_INFO, alpha_table=[0.0, 1.0, 2.0], k=7)
    pd.testing.assert_frame_equal(with_none, without_arg)


def test_build_survival_table_scope_datasets_full_set_matches_unfiltered() -> None:
    df = _make_synthetic_results()
    full_scope = set(_POLICY_INFO)
    scoped = build_survival_table(df, _POLICY_INFO, alpha_table=[0.0, 1.0, 2.0], k=7, scope_datasets=full_scope)
    unfiltered = build_survival_table(df, _POLICY_INFO, alpha_table=[0.0, 1.0, 2.0], k=7)
    pd.testing.assert_frame_equal(scoped, unfiltered)


# ============================================================================
# mode isolation (smoke/quick outputs never collide with full paths)
# ============================================================================


def test_experiment_name_isolation_across_modes() -> None:
    from src.experiments.exp_common import resolve_experiment_name

    full_name = resolve_experiment_name(BASE_EXPERIMENT_NAME, "full")
    quick_name = resolve_experiment_name(BASE_EXPERIMENT_NAME, "quick")
    smoke_name = resolve_experiment_name(BASE_EXPERIMENT_NAME, "smoke")

    assert full_name == "exp13_neighbor_survival"
    assert quick_name == "quick/exp13_neighbor_survival"
    assert smoke_name == "smoke/exp13_neighbor_survival"
    assert len({full_name, quick_name, smoke_name}) == 3


def test_smoke_config_narrows_alpha_table_without_touching_full() -> None:
    from src.experiments.config_experiments import resolve_experiment_config

    full_cfg = resolve_experiment_config(BASE_EXPERIMENT_NAME, "full")
    smoke_cfg = resolve_experiment_config(BASE_EXPERIMENT_NAME, "smoke")

    assert full_cfg["alpha_table"] == [0.0, 1.0, 2.0]
    assert smoke_cfg["alpha_table"] == [0.0, 1.0]
    assert smoke_cfg["k_values"] == full_cfg["k_values"] == [7]
    # embedding_seed_index is not overridden by 'smoke' - must stay identical.
    assert smoke_cfg["embedding_seed_index"] == full_cfg["embedding_seed_index"]


# ============================================================================
# _run_single wiring (fully mocked dataset/embedding - no real cache needed)
# ============================================================================


class _FakeDataset:
    """Minimal stand-in for src.datasets.registry.Dataset (kind='vector')."""

    def __init__(self, X: np.ndarray) -> None:
        self.X = X
        self.D = None
        self.y = None
        self.kind = "vector"
        self.name = "fake"

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]


def _make_task(alpha: float, k: int, **overrides) -> dict:
    task = {
        "dataset_name": "fake_dataset", "method_name": _method_name_for_task(alpha, k),
        "seed": 0, "alpha": alpha, "k": k,
        "n_max": 1000, "subsample_seed": 42,
        "embedding_experiment_name": "exp6_alpha_curves",
        "regime": "low_ratio", "nn_ratio_k1": 0.1, "alpha_best_source": "exp6", "dataset_set": DATASET_SET_CORE,
        "knn_jaccard_reference": 0.5,
        "stress_scale_invariant": 0.05, "shepard_spearman_rho": 0.9, "q_global": 0.7,
    }
    task.update(overrides)
    return task


@pytest.fixture()
def _fake_data_and_embedding(monkeypatch):
    """Monkeypatches dataset loading and the E6 embedding cache with a small,
    fully deterministic synthetic point cloud - no real files touched. Also
    clears the module-level `_dataset_order_orig` lru_cache so tests never
    see another test's cached (dataset_name='fake_dataset') entry."""
    _dataset_order_orig.cache_clear()
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 3))
    Y = rng.normal(size=(10, 2))

    monkeypatch.setattr("src.datasets.registry.load_dataset", lambda name: _FakeDataset(X))
    monkeypatch.setattr("src.datasets.subsample.subsample_dataset", lambda ds, n_max, random_state: ds)
    monkeypatch.setattr("src.experiments.exp13_neighbor_survival.load_embedding", lambda key: Y.copy())
    yield X, Y
    _dataset_order_orig.cache_clear()


def test_run_single_computes_neighbors_kept_via_knn_overlap_count(_fake_data_and_embedding) -> None:
    from src.methods.common import to_distance_matrix
    from src.sammon.metrics import _neighbor_ranks, knn_overlap_count

    X, Y = _fake_data_and_embedding
    k = 3
    task = _make_task(alpha=1.0, k=k)
    result = _run_single(task)

    assert result["status"] == "ok", result["error"]

    D = to_distance_matrix(X, "vector")
    _, order_orig = _neighbor_ranks(D)
    d_emb = to_distance_matrix(Y, "vector")
    _, order_emb = _neighbor_ranks(d_emb)
    expected = knn_overlap_count(order_orig, order_emb, k)

    assert result["metrics"]["neighbors_kept"] == pytest.approx(expected)
    assert result["metrics"]["neighbors_kept_frac"] == pytest.approx(expected / k)
    assert result["metrics"]["n_samples"] == X.shape[0]
    assert result["metrics"]["regime"] == "low_ratio"
    assert result["metrics"]["alpha_best_source"] == "exp6"
    assert result["metrics"]["dataset_set"] == DATASET_SET_CORE
    assert result["metrics"]["knn_jaccard_reference"] == 0.5
    assert result["metrics"]["stress_scale_invariant"] == 0.05
    assert result["extra"] == {"alpha": 1.0, "k": k}


def test_run_single_loads_embedding_from_task_embedding_experiment_name(_fake_data_and_embedding, monkeypatch) -> None:
    """The RunKey used to load the cached embedding must come from
    task['embedding_experiment_name'] - the routing that lets an
    exp12_alpha_grid_extension-sourced alpha_best load its embedding from the
    exp12 cache instead of exp6 (2026-09-17 task: no silent exp6 fallback)."""
    seen_keys = []

    def _capture(key):
        seen_keys.append(key)
        return _fake_data_and_embedding[1].copy()

    monkeypatch.setattr("src.experiments.exp13_neighbor_survival.load_embedding", _capture)
    task = _make_task(alpha=4.5, k=3, embedding_experiment_name="exp12_alpha_grid_extension", alpha_best_source="exp12")
    result = _run_single(task)

    assert result["status"] == "ok", result["error"]
    assert len(seen_keys) == 1
    assert seen_keys[0].experiment == "exp12_alpha_grid_extension"
    assert result["metrics"]["alpha_best_source"] == "exp12"


def test_run_single_is_an_error_row_on_embedding_shape_mismatch(_fake_data_and_embedding, monkeypatch) -> None:
    """A cached embedding whose row count does not match the freshly loaded
    dataset (a subsample/n_max mismatch) must fail loud as a status='error'
    row, never silently truncated/padded."""
    monkeypatch.setattr(
        "src.experiments.exp13_neighbor_survival.load_embedding",
        lambda key: np.zeros((3, 2)),  # wrong n_samples (fake dataset has 10 rows)
    )
    task = _make_task(alpha=1.0, k=3)
    result = _run_single(task)
    assert result["status"] == "error"
    assert "mismatch" in result["error"]
    assert result["extra"] == {"alpha": 1.0, "k": 3}


def test_run_single_is_an_error_row_when_embedding_is_missing(_fake_data_and_embedding, monkeypatch) -> None:
    def _raise(key):
        raise FileNotFoundError(f"no embedding for {key}")

    monkeypatch.setattr("src.experiments.exp13_neighbor_survival.load_embedding", _raise)
    task = _make_task(alpha=1.0, k=3)
    result = _run_single(task)
    assert result["status"] == "error"
    assert "FileNotFoundError" in result["error"]


# ============================================================================
# checkpoint / resume
# ============================================================================


@pytest.fixture()
def isolated_results_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirects `checkpoint.get_path('results_data_dir'/'embeddings_dir')`
    to a temporary directory so the test never touches the real results/data
    (same pattern as tests/test_exp12_alpha_grid_extension.py)."""
    from src.common import checkpoint

    data_dir = tmp_path / "results_data"
    emb_dir = tmp_path / "results_data" / "embeddings"

    def _fake_get_path(key: str) -> Path:
        if key == "results_data_dir":
            return data_dir
        if key == "embeddings_dir":
            return emb_dir
        raise KeyError(key)

    monkeypatch.setattr(checkpoint, "get_path", _fake_get_path)
    return data_dir


def test_filter_already_done_skips_previously_completed_runs(isolated_results_dir: Path) -> None:
    """A resumed run must skip a (dataset, method, seed) already present in
    the checkpoint CSV and only compute the remaining tasks - the exact
    mechanism exp13's main() relies on for resume."""
    from src.common.checkpoint import RunKey, append_result
    from src.experiments.exp_common import filter_already_done

    experiment_name = "smoke/" + BASE_EXPERIMENT_NAME
    method_name = _method_name_for_task(1.0, 7)

    # Simulate one already-completed run from a previous (interrupted) invocation.
    append_result(
        RunKey(experiment_name, "iris", method_name, 0),
        {"status": "ok", "error": "", "wall_time_sec": 0.1, "neighbors_kept": 5.0},
    )

    tasks = [
        {"dataset_name": "iris", "method_name": method_name, "seed": 0},
        {"dataset_name": "wine", "method_name": method_name, "seed": 0},
    ]
    todo, n_done = filter_already_done(experiment_name, tasks)

    assert n_done == 1
    assert todo == [{"dataset_name": "wine", "method_name": method_name, "seed": 0}]


def test_filter_already_done_is_idempotent_on_second_call(isolated_results_dir: Path) -> None:
    """Calling filter_already_done twice with the SAME already-written CSV
    (as a second smoke invocation would) must yield the same result both
    times - no partial/duplicate re-scheduling."""
    from src.common.checkpoint import RunKey, append_result
    from src.experiments.exp_common import filter_already_done

    experiment_name = "smoke/" + BASE_EXPERIMENT_NAME
    method_name = _method_name_for_task(0.0, 7)
    append_result(
        RunKey(experiment_name, "iris", method_name, 0),
        {"status": "ok", "error": "", "wall_time_sec": 0.1, "neighbors_kept": 4.0},
    )

    tasks = [{"dataset_name": "iris", "method_name": method_name, "seed": 0}]
    todo1, n_done1 = filter_already_done(experiment_name, tasks)
    todo2, n_done2 = filter_already_done(experiment_name, tasks)

    assert todo1 == todo2 == []
    assert n_done1 == n_done2 == 1
