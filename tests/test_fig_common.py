# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-17
# License: see the LICENSE file in the repository root
"""Tests for `src.figures.fig_common.display_label` - the shared config/CSV
identifier -> human-readable label conversion (author feedback 2026-09-17:
11 figure scripts printed raw identifiers like "auc_rnx"/"mnist_784" into
titles/axis labels/ticks/legends; every figure must now go through this one
function, and an unmapped name must NEVER show a raw underscore)."""
from __future__ import annotations

import matplotlib.pyplot as plt

from src.experiments.config_experiments import load_experiments_config
from src.figures import fig_common
from src.figures.fig_common import _is_article_figure, display_label


def test_explicit_dataset_label_is_used_verbatim() -> None:
    assert display_label("mnist_784", "dataset") == "MNIST (784-d)"


def test_explicit_method_label_is_used_verbatim() -> None:
    # author feedback 2026-09-17 (symbol unification): "alpha=1" -> mathtext $\alpha=1$
    assert display_label("sammon_alpha_smacof", "method") == "Sammon ($\\alpha=1$)"


def test_explicit_metric_label_is_used_verbatim() -> None:
    assert display_label("auc_rnx", "metric") == "AUC$_{RNX}$"
    assert display_label("trustworthiness_k7", "metric") == "Trustworthiness (k=7)"


def test_unknown_name_never_shows_a_raw_underscore() -> None:
    """The core regression guard: even a name with NO config entry at all
    must not reach a figure with a literal underscore in it."""
    for kind in ("dataset", "method", "metric"):
        label = display_label("some_totally_unmapped_identifier_42", kind)
        assert "_" not in label, f"display_label leaked a raw underscore for kind={kind}: {label!r}"


def test_unknown_name_fallback_is_reasonably_capitalized() -> None:
    label = display_label("some_new_dataset", "dataset")
    assert label == "Some New Dataset"


def test_unknown_name_fallback_applies_word_overrides() -> None:
    """A word covered by display_labels.word_overrides (e.g. an acronym)
    must use the override's exact casing even in an otherwise-unmapped name."""
    label = display_label("umap_extra_variant", "method")
    assert label.startswith("UMAP")
    assert "_" not in label


def test_unknown_name_fallback_handles_k_number_pattern() -> None:
    """'k7' -> 'k=7' generic pattern (neighbourhood-size suffix used by
    several metrics), even for a metric name with no explicit config entry."""
    label = display_label("some_new_metric_k12", "metric")
    assert "k=12" in label
    assert "_" not in label


def test_unknown_kind_falls_back_without_raising() -> None:
    """A kind with no section at all in display_labels (e.g. a typo) must
    not raise - it simply has no explicit mappings, so every name in it
    goes through the same fallback."""
    label = display_label("some_name", "totally_unknown_kind")
    assert label == "Some Name"


def test_is_article_figure_exact_match() -> None:
    assert _is_article_figure("fig_faithful_map") is True
    assert _is_article_figure("fig_faithful_map_extra") is False


def test_is_article_figure_prefix_wildcard() -> None:
    # figures.article_figures has a "fig_cd_diagram_*" entry (one PDF per
    # experiment/metric pair, see src/figures/fig_cd_diagram.py).
    assert _is_article_figure("fig_cd_diagram_exp1_dr_benchmark_auc_rnx") is True
    assert _is_article_figure("fig_cd_diagram") is False


def test_is_article_figure_rejects_orphan_names() -> None:
    """Regression guard for the 2026-09-17 orphan-PDF cleanup: a figure name
    not on the whitelist must not be treated as an article figure."""
    assert _is_article_figure("fig_temporal_trajectories_email_eu_core_temporal_dtsne") is False


def test_save_figure_full_mode_copies_only_whitelisted_names(tmp_path, monkeypatch) -> None:
    """End-to-end guard for the 2026-09-17 orphan-PDF fix: in 'full' mode,
    results/figures/ ALWAYS gets the PDF, but clanek/img/ only gets it if
    the name is on figures.article_figures."""
    monkeypatch.setattr(fig_common, "_CURRENT_MODE", "full")
    monkeypatch.setattr(fig_common, "figures_out_dir", lambda: tmp_path / "results_figures")
    monkeypatch.setattr(fig_common, "article_img_dir", lambda: tmp_path / "clanek_img")
    (tmp_path / "results_figures").mkdir()
    (tmp_path / "clanek_img").mkdir()

    fig_whitelisted = plt.figure()
    results_path, article_path = fig_common.save_figure(fig_whitelisted, "fig_faithful_map")
    assert results_path.exists()
    assert article_path is not None and article_path.exists()

    fig_orphan = plt.figure()
    results_path2, article_path2 = fig_common.save_figure(fig_orphan, "fig_some_orphan_not_in_the_article")
    assert results_path2.exists()
    assert article_path2 is None
    assert not (tmp_path / "clanek_img" / "fig_some_orphan_not_in_the_article.pdf").exists()


def test_save_figure_quick_mode_never_copies_to_article_even_if_whitelisted(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(fig_common, "_CURRENT_MODE", "quick")
    monkeypatch.setattr(fig_common, "figures_out_dir", lambda: tmp_path / "results_figures")
    monkeypatch.setattr(fig_common, "article_img_dir", lambda: tmp_path / "clanek_img")
    (tmp_path / "results_figures").mkdir()

    fig = plt.figure()
    results_path, article_path = fig_common.save_figure(fig, "fig_faithful_map")
    assert results_path.exists()
    assert article_path is None


def test_display_labels_config_sections_are_internally_consistent() -> None:
    """Every fig_metric_correlations metric and every report.main_methods
    method has an EXPLICIT display_labels entry (regression guard: these are
    exactly the identifiers the author pointed at as "hell")."""
    cfg = load_experiments_config()
    display_labels = cfg["display_labels"]
    metrics = cfg["fig_metric_correlations"]["metrics"]
    missing_metrics = [m for m in metrics if m not in display_labels["metric"]]
    assert not missing_metrics, f"fig_metric_correlations metrics missing an explicit display_labels.metric entry: {missing_metrics}"

    main_methods = cfg["report"]["main_methods"]
    missing_methods = [m for m in main_methods if m not in display_labels["method"]]
    assert not missing_methods, f"report.main_methods missing an explicit display_labels.method entry: {missing_methods}"
