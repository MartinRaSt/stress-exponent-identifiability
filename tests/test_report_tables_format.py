# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-18
# License: see the LICENSE file in the repository root
"""Regression tests for the 2026-09-18 width fix of `write_booktabs_tex`
(src/experiments/report_tables.py): the header must use short COLUMN_LABELS
instead of raw CSV column names, `<metric>_median`/`<metric>_avg_rank`
pairs must share one spanned header cell, and `_avg_rank` columns must be
rendered with 2 decimals - so the DATA, not the header, determines the
table width."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from src.experiments.report_tables import write_booktabs_tex

_ROW_RE = re.compile(r"^[^%].*\\\\\s*$")


def _tabular_lines(text: str) -> list[str]:
    """Lines strictly between \\toprule and \\bottomrule (header + body)."""
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip() == "\\toprule")
    end = next(i for i, l in enumerate(lines) if l.strip() == "\\bottomrule")
    return lines[start + 1 : end]


def _n_ampersands(line: str) -> int:
    return line.count("&")


def test_median_rank_pair_gets_spanned_header_and_cmidrule(tmp_path):
    df = pd.DataFrame({
        "method": ["a", "b"],
        "auc_rnx_median": [0.9123456, 0.8123456],
        "auc_rnx_avg_rank": [1.333333, 2.0],
        "stress_scale_invariant_median": [0.1, 0.2],
        "stress_scale_invariant_avg_rank": [1.0, 2.0],
    })
    out = tmp_path / "t.tex"
    write_booktabs_tex(df, out, caption="cap", label="tab:t")
    text = out.read_text(encoding="utf-8")

    assert "\\multicolumn{2}{c}{AUC\\textsubscript{RNX}}" in text
    assert "\\multicolumn{2}{c}{Stress}" in text
    assert "\\cmidrule(lr){2-3}" in text
    assert "\\cmidrule(lr){4-5}" in text
    # the cmidrule line is not terminated with a row break
    cmidrule_line = next(l for l in text.splitlines() if "\\cmidrule" in l)
    assert not cmidrule_line.rstrip().endswith("\\\\")


def test_body_row_ampersand_count_matches_column_count(tmp_path):
    # 'm1'/'m2' (not 'a'/'b'): the default `value_labels` auto-mapping
    # (2026-09-18) routes the 'method' column through `display_label`, whose
    # fallback capitalizes a pure-alpha word ('a' -> 'A') but leaves an
    # alphanumeric one untouched ('m1' -> 'm1', same as 'd1'/'d2' below) -
    # this test is about ampersand bookkeeping, not label content.
    df = pd.DataFrame({
        "method": ["m1", "m2"],
        "auc_rnx_median": [0.91, 0.81],
        "auc_rnx_avg_rank": [1.0, 2.0],
        "dataset": ["d1", "d2"],
    })
    out = tmp_path / "t.tex"
    write_booktabs_tex(df, out, caption="cap", label="tab:t")
    text = out.read_text(encoding="utf-8")
    n_cols = df.shape[1]
    lines = _tabular_lines(text)
    body_lines = [l for l in lines if l.strip() and not l.strip().startswith("\\midrule") and "cmidrule" not in l]
    # the last two body_lines are actual data rows (header rows excluded below)
    data_lines = [l for l in body_lines if l.strip() and _ROW_RE.match(l.strip())]
    # keep only genuine data rows (values 'm1'/'d1' etc, not header labels)
    data_rows = [l for l in data_lines if l.strip().startswith("m1 ") or l.strip().startswith("m2 ")]
    assert len(data_rows) == 2
    for row in data_rows:
        assert _n_ampersands(row) == n_cols - 1


def test_header_has_no_bare_underscore_or_math_subscript(tmp_path):
    df = pd.DataFrame({
        "method": ["a"],
        "auc_rnx_median": [0.9],
        "auc_rnx_avg_rank": [1.0],
        "some_unmapped_column_name": [1.0],
    })
    out = tmp_path / "t.tex"
    write_booktabs_tex(df, out, caption="cap", label="tab:t")
    lines = _tabular_lines(out.read_text(encoding="utf-8"))
    # header block = everything up to (and excluding) \midrule
    midrule_idx = next(i for i, l in enumerate(lines) if l.strip() == "\\midrule")
    header_text = "\n".join(lines[:midrule_idx])
    # every remaining underscore must be an escaped \_ (LaTeX control sequence)
    for m in re.finditer(r"_", header_text):
        assert header_text[m.start() - 1] == "\\", f"bare underscore in header: {header_text!r}"
    # no math-mode subscript like $x_{...}$ or $x_y$
    assert not re.search(r"\$[^$]*_[^$]*\$", header_text)


def test_avg_rank_column_uses_two_decimals(tmp_path):
    df = pd.DataFrame({
        "method": ["a", "b"],
        "auc_rnx_median": [0.912345, 0.812345],
        "auc_rnx_avg_rank": [1.333333, 2.0],
    })
    out = tmp_path / "t.tex"
    write_booktabs_tex(df, out, caption="cap", label="tab:t")
    text = out.read_text(encoding="utf-8")
    assert "1.33" in text
    assert "1.333" not in text
    # the median column keeps the default 4-decimal formatting
    assert "0.9123" in text


def test_no_pairs_keeps_single_row_header(tmp_path):
    df = pd.DataFrame({
        "experiment": ["e14"],
        "tau": [0.0],
        "n_rows_kept": [10],
    })
    out = tmp_path / "t.tex"
    write_booktabs_tex(df, out, caption="cap", label="tab:t")
    lines = _tabular_lines(out.read_text(encoding="utf-8"))
    midrule_idx = next(i for i, l in enumerate(lines) if l.strip() == "\\midrule")
    header_lines = lines[:midrule_idx]
    assert len(header_lines) == 1
    assert "\\multicolumn" not in header_lines[0]
    assert "\\cmidrule" not in header_lines[0]
    assert "Experiment" in header_lines[0]
    assert "$\\tau$" in header_lines[0]


def test_column_alignment_l_for_text_r_for_numeric(tmp_path):
    df = pd.DataFrame({
        "method": ["a", "b"],
        "auc_rnx_median": [0.9, 0.8],
    })
    out = tmp_path / "t.tex"
    write_booktabs_tex(df, out, caption="cap", label="tab:t")
    text = out.read_text(encoding="utf-8")
    assert "\\begin{tabular}{lr}" in text


def test_float_format_dict_per_column_with_default(tmp_path):
    df = pd.DataFrame({
        "method": ["a"],
        "wall_time_sec": [1.23456],
    })
    out = tmp_path / "t.tex"
    write_booktabs_tex(
        df, out, caption="cap", label="tab:t",
        float_format={"wall_time_sec": "%.1f", "__default__": "%.4f"},
    )
    text = out.read_text(encoding="utf-8")
    assert "1.2 " in text or "1.2 \\\\" in text
    assert "1.23456" not in text


def test_float_format_string_is_backward_compatible(tmp_path):
    df = pd.DataFrame({
        "method": ["a"],
        "wall_time_sec": [1.23456],
    })
    out = tmp_path / "t.tex"
    write_booktabs_tex(df, out, caption="cap", label="tab:t", float_format="%.2f")
    text = out.read_text(encoding="utf-8")
    assert "1.23" in text


# --- 2026-09-18 follow-up: compositional prefix/suffix affixes -------------


@pytest.mark.parametrize(
    "col,expected",
    [
        ("stress_scale_invariant_iqr", "Stress (IQR)"),
        ("avg_rank_auc_rnx", "AUC\\textsubscript{RNX} rank"),
        ("median_wall_time_sec_k6", "Time [s]"),
        ("iqr_auc_rnx", "AUC\\textsubscript{RNX} (IQR)"),
        ("median_auc_rnx", "AUC\\textsubscript{RNX}"),
        ("auc_at_alpha0", "AUC at $\\alpha{=}0$"),
        ("auc_at_alpha1", "AUC at $\\alpha{=}1$"),
        ("stress_at_alpha0", "Stress at $\\alpha{=}0$"),
        ("spearman_ceiling_vs_gain_rho", "Ceiling vs. gain $\\rho$"),
        ("spearman_ceiling_vs_gain_p", "Ceiling vs. gain $p$"),
        ("gain_vs_alpha0", "Gain vs. $\\alpha{=}0$"),
        ("gain_vs_alpha1", "Gain vs. $\\alpha{=}1$"),
        ("n_datasets_total", "Datasets"),
        ("n_datasets_on_front_auc_stress", "Front (AUC/stress)"),
        ("n_datasets_on_front_qlocal_qglobal", "Front (Q loc/glob)"),
        ("n_datasets_dominated_by_neighbor_method", "Dominated"),
        ("stab_pct_change_median", "Stability [\\%]"),
        ("stab_pct_change_iqr", "Stability [\\%] (IQR)"),
        ("qual_pct_change_median", "Quality [\\%]"),
        ("qual_pct_change_iqr", "Quality [\\%] (IQR)"),
        ("wilcoxon_pvalue", "Wilcoxon $p$"),
        ("n_positive", "$n^{+}$"),
        ("row_type", "Row"),
        ("frac_both_gaps_nonneg", "Both gaps $\\ge 0$"),
        ("med_tight_v1", "Tight. V1"),
        ("med_tight_sandwich", "Tight. sandwich"),
        ("alpha_star_max", "$\\alpha^{*}$"),
        ("alpha_star_constrained", "$\\alpha^{*}$ constr."),
        ("alpha_pred_log_linear", "$\\alpha$ log-lin."),
        ("alpha_pred_two_threshold", "$\\alpha$ 2-thr."),
        ("auc_at_alpha_pred_log_linear", "AUC at $\\alpha$ log-lin."),
        ("auc_at_alpha_pred_two_threshold", "AUC at $\\alpha$ 2-thr."),
        ("auc_at_alpha_auto", "AUC at $\\alpha$ auto"),
        # an unmapped base falls through to display_label(col, "column") ->
        # _fallback_label (2026-09-18: underscores become SPACES, not an
        # escaped-but-still-bare-underscore single word, so the header word
        # wrap in _wrap_header_label has somewhere to break).
        ("totally_unknown_metric_iqr", "Totally Unknown Metric Iqr"),
    ],
)
def test_column_label_composes_known_affixes(col, expected):
    from src.experiments.report_tables import _column_label

    assert _column_label(col) == expected


def test_composed_affix_header_has_no_bare_underscore(tmp_path):
    """A column using a composed affix label must still be free of a bare
    `_` in the header (same guarantee as an exact COLUMN_LABELS match)."""
    df = pd.DataFrame({
        "stress_scale_invariant_iqr": [0.01],
        "spearman_ceiling_vs_gain_rho": [0.5],
    })
    out = tmp_path / "t.tex"
    write_booktabs_tex(df, out, caption="cap", label="tab:t")
    lines = _tabular_lines(out.read_text(encoding="utf-8"))
    midrule_idx = next(i for i, l in enumerate(lines) if l.strip() == "\\midrule")
    header_text = "\n".join(lines[:midrule_idx])
    for m in re.finditer(r"_", header_text):
        assert header_text[m.start() - 1] == "\\", f"bare underscore in header: {header_text!r}"


# --- 2026-09-18 follow-up: drop_constant_cols -------------------------------


def test_drop_constant_cols_removes_truly_constant_column(tmp_path):
    df = pd.DataFrame({
        "experiment": ["exp14_convergence_robustness", "exp14_convergence_robustness"],
        "tau": [0.0, 0.01],
        "status": ["ok", "ok"],
        "error": ["", ""],
    })
    out = tmp_path / "t.tex"
    write_booktabs_tex(df, out, caption="cap", label="tab:t", drop_constant_cols=["experiment", "status", "error"])
    text = out.read_text(encoding="utf-8")
    assert "Experiment" not in text
    assert "Status" not in text
    assert "Error" not in text
    assert "exp14" not in text
    lines = _tabular_lines(text)
    header = lines[0]
    assert _n_ampersands(header) == 0  # only 'tau' left -> single column, no '&'


def test_drop_constant_cols_keeps_non_constant_error_column(tmp_path):
    """A non-constant 'error' column (a real failure happened on some row)
    must NEVER be dropped - fail-loud."""
    df = pd.DataFrame({
        "experiment": ["e14", "e14"],
        "tau": [0.0, 0.01],
        "status": ["ok", "error"],
        "error": ["", "boom: something failed"],
    })
    out = tmp_path / "t.tex"
    write_booktabs_tex(df, out, caption="cap", label="tab:t", drop_constant_cols=["experiment", "status", "error"])
    text = out.read_text(encoding="utf-8")
    assert "Experiment" not in text  # this one IS constant -> dropped
    assert "Status" in text  # non-constant -> kept
    assert "Error" in text  # non-constant -> kept
    assert "boom" in text


def test_drop_constant_cols_csv_untouched(tmp_path):
    """write_booktabs_tex never touches a CSV - drop_constant_cols only
    affects the .tex output, the caller's own CSV write keeps all columns."""
    df = pd.DataFrame({"experiment": ["e14", "e14"], "tau": [0.0, 0.01], "status": ["ok", "ok"]})
    csv_path = tmp_path / "t.csv"
    df.to_csv(csv_path, index=False)
    out = tmp_path / "t.tex"
    write_booktabs_tex(df, out, caption="cap", label="tab:t", drop_constant_cols=["experiment", "status"])
    reloaded = pd.read_csv(csv_path)
    assert list(reloaded.columns) == ["experiment", "tau", "status"]


# --- 2026-09-18 follow-up: value_labels (display_label for BODY cells) -----


def test_value_labels_default_mapping_applies_to_method_column(tmp_path):
    """With no `value_labels` argument, a `method` column is auto-mapped
    through `display_label(..., 'method')` (config_experiments.yaml curated
    text), not left as the raw identifier."""
    df = pd.DataFrame({"method": ["sammon_alpha_smacof"], "auc_rnx_median": [0.9]})
    out = tmp_path / "t.tex"
    write_booktabs_tex(df, out, caption="cap", label="tab:t")
    text = out.read_text(encoding="utf-8")
    assert "Sammon ($\\alpha=1$)" in text
    assert "sammon_alpha_smacof" not in text
    assert "sammon\\_alpha\\_smacof" not in text


def test_value_labels_default_mapping_applies_to_dataset_and_solver_columns(tmp_path):
    df = pd.DataFrame({"dataset": ["mnist_784"], "solver": ["sgd_stab"], "auc_rnx_median": [0.5]})
    out = tmp_path / "t.tex"
    write_booktabs_tex(df, out, caption="cap", label="tab:t")
    text = out.read_text(encoding="utf-8")
    assert "MNIST (784-d)" in text
    assert "SGD (stabilized)" in text
    assert "mnist\\_784" not in text
    assert "sgd\\_stab" not in text


def test_value_labels_empty_dict_disables_default_mapping(tmp_path):
    """Backward compatibility: passing an explicit empty dict restores the
    old raw-value-with-escaped-underscore rendering for every column."""
    df = pd.DataFrame({"method": ["sammon_alpha_smacof"], "auc_rnx_median": [0.9]})
    out = tmp_path / "t.tex"
    write_booktabs_tex(df, out, caption="cap", label="tab:t", value_labels={})
    text = out.read_text(encoding="utf-8")
    assert "sammon\\_alpha\\_smacof" in text
    assert "Sammon ($\\alpha=1$)" not in text


def test_value_labels_explicit_mapping_overrides_column_name(tmp_path):
    """An explicit `value_labels` dict can label a column whose NAME is not
    in the default map (e.g. 'algo' here) - or remap 'method' to a different
    kind entirely."""
    df = pd.DataFrame({"algo": ["pca"], "auc_rnx_median": [0.7]})
    out = tmp_path / "t.tex"
    write_booktabs_tex(df, out, caption="cap", label="tab:t", value_labels={"algo": "method"})
    text = out.read_text(encoding="utf-8")
    lines = _tabular_lines(text)
    midrule_idx = next(i for i, l in enumerate(lines) if l.strip() == "\\midrule")
    body_text = "\n".join(lines[midrule_idx + 1 :])
    assert "PCA" in body_text
    assert "pca" not in body_text  # only the mapped label ('PCA'), not the raw identifier


def test_value_labels_unknown_value_still_uses_fallback_no_raw_underscore(tmp_path):
    """A `method`/`dataset`/... value with NO explicit display_labels config
    entry still goes through `_fallback_label` (word split + capitalize),
    same as figures - never a raw underscore in the table body."""
    df = pd.DataFrame({"method": ["some_new_method_variant"], "auc_rnx_median": [0.1]})
    out = tmp_path / "t.tex"
    write_booktabs_tex(df, out, caption="cap", label="tab:t")
    text = out.read_text(encoding="utf-8")
    assert "some_new_method_variant" not in text
    assert "some\\_new\\_method\\_variant" not in text
    assert "Some New Method Variant" in text


# --- escape_latex_label -----------------------------------------------------


def test_escape_latex_label_escapes_underscore_ampersand_percent_hash():
    from src.experiments.report_tables import escape_latex_label

    assert escape_latex_label("a_b&c%d#e") == "a\\_b\\&c\\%d\\#e"


def test_escape_latex_label_leaves_braced_math_subscript_untouched():
    from src.experiments.report_tables import escape_latex_label

    assert escape_latex_label("$\\rho_{NN}$") == "$\\rho_{NN}$"
    assert escape_latex_label("Sammon ($\\alpha=1$)") == "Sammon ($\\alpha=1$)"


def test_escape_latex_label_raises_on_bare_math_subscript():
    """A math-mode subscript with a bare (unbraced) '_' (e.g. "$x_y$") must
    raise, not be silently reproduced or silently escaped (which would
    corrupt the math)."""
    from src.experiments.report_tables import escape_latex_label

    with pytest.raises(ValueError, match="bare"):
        escape_latex_label("$x_y$")


def test_escape_latex_label_raises_on_unbalanced_math_mode():
    from src.experiments.report_tables import escape_latex_label

    with pytest.raises(ValueError, match="unbalanced"):
        escape_latex_label("$x")


def test_escape_latex_label_converts_known_greek_unicode_letters():
    # chr(codepoint), not a literal glyph, so this source file stays ASCII
    # (project rule: no Unicode special characters in code).
    from src.experiments.report_tables import escape_latex_label

    alpha, rho, epsilon, lam, tau = chr(0x03B1), chr(0x03C1), chr(0x03B5), chr(0x03BB), chr(0x03C4)
    assert escape_latex_label(f"{alpha}-Sammon") == "$\\alpha$-Sammon"  # outside math -> wrapped in $...$
    assert escape_latex_label(f"${rho}_{{NN}}$") == "$\\rho_{NN}$"  # already in math -> bare macro
    assert escape_latex_label(epsilon) == "$\\varepsilon$"
    assert escape_latex_label(lam) == "$\\lambda$"
    assert escape_latex_label(tau) == "$\\tau$"


def test_escape_latex_label_fails_loud_on_unknown_non_ascii():
    """Any non-ASCII character with no known LaTeX equivalent must raise -
    no silent substitution or dropping of the character."""
    from src.experiments.report_tables import escape_latex_label

    with pytest.raises(ValueError, match="Non-ASCII"):
        escape_latex_label("caf" + chr(0xE9))  # 'e' with acute accent (chr(), not a literal glyph)


def test_escape_latex_label_all_configured_method_labels_are_valid():
    """Every explicit config_experiments.yaml display_labels.method entry
    must survive escape_latex_label without raising (regression guard: a
    future author edit to display_labels must stay LaTeX-table-safe)."""
    from src.experiments.config_experiments import load_experiments_config
    from src.experiments.report_tables import escape_latex_label

    methods = load_experiments_config()["display_labels"]["method"]
    for name, label in methods.items():
        escape_latex_label(label)  # must not raise


# --- 2026-09-18 header word-wrap (\thead, author feedback: an underscore
# header is one unbreakable LaTeX word and stays wide even when escaped) ---


def test_wrap_header_label_short_label_stays_unwrapped():
    from src.experiments.report_tables import _wrap_header_label

    assert _wrap_header_label("Method") == "Method"
    assert _wrap_header_label("AUC\\textsubscript{RNX}") == "AUC\\textsubscript{RNX}"


def test_wrap_header_label_long_label_becomes_multirow_thead():
    from src.experiments.report_tables import _wrap_header_label

    wrapped = _wrap_header_label("Front (AUC/stress)")
    assert wrapped.startswith("\\thead{")
    assert wrapped.endswith("}")
    assert "\\\\" in wrapped  # at least one line break


def test_wrap_header_label_never_splits_a_math_span():
    """A `$...$` span with an internal space (e.g. 'Both gaps $\\ge 0$')
    must stay on ONE line of the \\thead - splitting it would leave an
    unbalanced '$' on each side."""
    from src.experiments.report_tables import _wrap_header_label

    wrapped = _wrap_header_label("Both gaps $\\ge 0$")
    for line in wrapped.replace("\\thead{", "").rstrip("}").split("\\\\"):
        assert line.count("$") % 2 == 0


def test_write_booktabs_tex_header_uses_thead_for_a_long_column_name(tmp_path):
    """A column with a long, space-containing label (here an unmapped name,
    so it falls through to the space-separated fallback) must render as a
    multi-row \\thead cell whose LONGEST LINE (the quantity that actually
    drives the rendered column width, unlike total source length, which a
    '\\thead{...\\\\...}' wrapper necessarily increases) is narrower than
    the naive one-line '_'->'\\_' rendering of the same name would have
    been - that one-line rendering is exactly what a plain (non-thead)
    header cell forces the column to accommodate."""
    from src.experiments.report_tables import _table_header_max_line_chars

    df = pd.DataFrame({"n_datasets_dominated_by_neighbor_method_extra_words": [1]})
    out = tmp_path / "t.tex"
    write_booktabs_tex(df, out, caption="cap", label="tab:t")
    text = out.read_text(encoding="utf-8")
    assert "\\thead{" in text
    lines = _tabular_lines(text)
    header_line = lines[0]
    thead_body = header_line[header_line.index("\\thead{") + len("\\thead{") : header_line.rindex("}")]
    longest_line = max(len(part) for part in thead_body.split("\\\\"))
    naive = "n\\_datasets\\_dominated\\_by\\_neighbor\\_method\\_extra\\_words"
    assert longest_line < len(naive)
    assert longest_line <= _table_header_max_line_chars() + 5  # a bit of slack for a single long token/word


def test_display_label_is_identical_between_table_and_figure_code_paths():
    """The table code path (src.common.display_labels.display_label,
    imported by report_tables.py) and the figure code path
    (src.figures.fig_common.display_label, re-exported from the same
    module) must return the exact same string for the same name/kind -
    otherwise a method's name in a table and a figure could disagree."""
    from src.common.display_labels import display_label as table_display_label
    from src.figures.fig_common import display_label as figure_display_label

    for name, kind in [("sammon_alpha_smacof", "method"), ("mnist_784", "dataset"), ("auc_rnx", "metric")]:
        assert table_display_label(name, kind) == figure_display_label(name, kind)
