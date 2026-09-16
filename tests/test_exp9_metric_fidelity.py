# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-14
# License: see the LICENSE file in the repository root
"""Q1 step 3 (B.7 item 9): the schema of exp9_metric_fidelity_results.csv,
smoke-mode isolation (mtime before/after does not change anything outside
results/*/smoke), and `exp9_metric_fidelity_stats.compute_stats` over a
small synthetic CSV (Holm, sign-flip)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.common.config import get_mode_path, get_path, get_project_root
from src.experiments.config_experiments import resolve_experiment_config
from src.experiments.exp9_metric_fidelity import COLUMNS, _TEXT_COLUMNS, build_tasks
from src.experiments.exp9_metric_fidelity_stats import STATS_COLUMNS, compute_stats

ROOT = get_project_root()


def test_build_tasks_smoke_has_all_scenarios_and_oracle_rows() -> None:
    cfg = resolve_experiment_config("exp9_metric_fidelity", "smoke")
    tasks = build_tasks(cfg)
    scenarios = {t["scenario"] for t in tasks}
    assert scenarios == set(cfg["active_scenarios"])
    for scenario in cfg["active_scenarios"]:
        methods = {t["method"] for t in tasks if t["scenario"] == scenario}
        assert "oracle_truth" in methods
        assert set(cfg["methods"]).issubset(methods)


def test_build_tasks_deterministic_seed_derivation() -> None:
    cfg = resolve_experiment_config("exp9_metric_fidelity", "smoke")
    tasks = build_tasks(cfg)
    for t in tasks:
        seed_base = int(cfg["scenarios"][t["scenario"]]["seed_base"])
        assert t["seed_gen"] == seed_base + t["replicate"]
        assert t["seed_method"] == t["replicate"]


def test_columns_text_subset_is_consistent() -> None:
    assert _TEXT_COLUMNS.issubset(set(COLUMNS))
    numeric_cols = set(COLUMNS) - _TEXT_COLUMNS
    assert "centroid_lre" in numeric_cols
    assert "status" in _TEXT_COLUMNS


def _synthetic_results_df(n_replicates: int = 20, seed: int = 0) -> pd.DataFrame:
    """A small synthetic CSV (2 scenarios x 4 methods x n_replicates) with
    the exp9_metric_fidelity_results.csv schema - for testing compute_stats
    without depending on an actual (expensive) DR method run."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    methods = ["sammon_alpha_pred", "tsne_auto", "umap_auto", "densmap"]
    for scenario, metric, base in (("S1", "centroid_lre", 0.05), ("S3", "geodesic_stress_si", 0.02)):
        for method in methods:
            # pred has a systematically LOWER (better) value than the baseline methods (lower is better)
            offset = 0.0 if method == "sammon_alpha_pred" else 0.05
            for r in range(n_replicates):
                val = base + offset + 0.01 * rng.normal()
                row = {c: np.nan for c in COLUMNS}
                row.update({
                    "experiment": "test", "scenario": scenario, "replicate": r, "seed_gen": r, "seed_method": r,
                    "method": method, "status": "ok", "error": "", "is_oracle": False, metric: max(val, 1e-6),
                })
                rows.append(row)
    return pd.DataFrame(rows)


def test_compute_stats_holm_and_direction_on_synthetic_data() -> None:
    cfg = resolve_experiment_config("exp9_metric_fidelity", "full")
    cfg = dict(cfg)
    cfg["active_scenarios"] = ["S1", "S3"]
    cfg["primary_baselines"] = ["tsne_auto", "umap_auto", "densmap"]
    cfg["secondary_baselines"] = []
    cfg["explorational_vs"] = []
    stats_cfg = resolve_experiment_config("exp9_metric_fidelity_stats", "full")

    df = _synthetic_results_df(n_replicates=20, seed=1)

    class _Logger:
        def warning(self, *a, **k):
            pass

    stats_df = compute_stats(df, cfg, stats_cfg, _Logger())
    assert list(stats_df.columns) == STATS_COLUMNS
    assert set(stats_df["family"]) == {"F_B1"}
    # pred is systematically BETTER (lower) than all baselines -> all
    # one-sided tests should reject H0 after Holm (small p)
    assert (stats_df["p_holm"] < 0.05).all()
    assert stats_df["reject_holm"].all()
    assert (stats_df["median_diff"] < 0).all()
    assert (stats_df["delta_pair"] < 0).all()


def test_smoke_run_does_not_touch_full_outputs() -> None:
    """A smoke run (already run by the author/agent before this test) writes
    EXCLUSIVELY to results/data/smoke and results/logs/smoke - an
    integration test comparing the mtime snapshot before/after a REPEATED
    smoke run (should be idempotent/a no-op on already-done tasks, and not
    touch the full data)."""
    smoke_csv = get_mode_path("results_data_dir", "smoke") / "exp9_metric_fidelity_results.csv"
    if not smoke_csv.exists():
        pytest.skip(f"missing smoke data {smoke_csv} (run src\\run_exp9_metric_fidelity.bat smoke)")

    full_dir = get_path("results_data_dir")

    def _snapshot() -> dict[str, float]:
        files = [p for p in full_dir.glob("*.csv") if p.is_file()]
        return {str(p.relative_to(ROOT)): p.stat().st_mtime_ns for p in files}

    before = _snapshot()
    from src.experiments.exp9_metric_fidelity import main

    main(mode="smoke", n_workers=1)
    after = _snapshot()
    assert before == after, "the smoke run changed a file in the results/data root (outside the smoke subdirectory)"
