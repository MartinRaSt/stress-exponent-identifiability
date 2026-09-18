# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-18
# License: see the LICENSE file in the repository root
"""Tests for src/tools/make_submission.py (flat journal-submission package
builder). Fast unit tests exercise the pure path-resolution, name-collision
and command-rewriting logic on synthetic mini .tex trees under tmp_path (no
LaTeX toolchain needed). Two integration tests build the REAL dami_en /
supplement_en packages from the checked-out clanek_en/ tree - these are
skipped when the local (untracked, per-machine) clanek/ and results/tables/
trees are not present, and the one that additionally runs a full LaTeX
compile is skipped when pdflatex is not on PATH."""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from src.tools import make_submission as ms

_HAS_PDFLATEX = shutil.which("pdflatex") is not None
_DAMI_SOURCES_PRESENT = (
    ms.DAMI_SPEC.root_tex.is_file()
    and (ms.PROJECT_ROOT / "clanek" / "img").is_dir()
    and (ms.PROJECT_ROOT / "clanek" / "references.bib").is_file()
)
_SUPPLEMENT_SOURCES_PRESENT = (
    ms.SUPPLEMENT_SPEC.root_tex.is_file()
    and (ms.PROJECT_ROOT / "results" / "tables").is_dir()
)


# --------------------------------------------------------------------------
# FlatNameAssigner
# --------------------------------------------------------------------------


def test_flat_name_assigner_reuses_name_for_same_source(tmp_path: Path) -> None:
    f = tmp_path / "a.tex"
    f.write_text("x", encoding="utf-8")
    assigner = ms.FlatNameAssigner()
    first = assigner.assign(f)
    second = assigner.assign(f)
    assert first == second == "a.tex"


def test_flat_name_assigner_disambiguates_basename_collision(tmp_path: Path) -> None:
    (tmp_path / "sections").mkdir()
    (tmp_path / "supplement_sections").mkdir()
    f1 = tmp_path / "sections" / "results.tex"
    f2 = tmp_path / "supplement_sections" / "results.tex"
    f1.write_text("a", encoding="utf-8")
    f2.write_text("b", encoding="utf-8")
    assigner = ms.FlatNameAssigner()
    name1 = assigner.assign(f1)
    name2 = assigner.assign(f2)
    assert name1 == "results.tex"
    assert name2 == "supplement_sections_results.tex"
    assert name1 != name2


def test_flat_name_assigner_fails_loud_on_unresolvable_collision(tmp_path: Path) -> None:
    # f1 takes "results.tex"; f2 (parent dir "b") takes "b_results.tex". f4
    # has the SAME basename and the SAME parent directory NAME as f2 (but a
    # different absolute path), so both of its candidate names are already
    # taken by other, unrelated source files - must raise, not silently
    # overwrite either one.
    f1 = tmp_path / "a" / "results.tex"
    f2 = tmp_path / "b" / "results.tex"
    f4 = tmp_path / "other" / "b" / "results.tex"
    for f in (f1, f2, f4):
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x", encoding="utf-8")
    assigner = ms.FlatNameAssigner()
    assert assigner.assign(f1) == "results.tex"
    assert assigner.assign(f2) == "b_results.tex"
    with pytest.raises(ms.SubmissionBuildError, match="collision"):
        assigner.assign(f4)


# --------------------------------------------------------------------------
# _resolve_path
# --------------------------------------------------------------------------


def test_resolve_path_tries_candidates_in_order(tmp_path: Path) -> None:
    base_a = tmp_path / "a"
    base_b = tmp_path / "b"
    base_a.mkdir()
    base_b.mkdir()
    target = base_b / "numbers.tex"
    target.write_text("x", encoding="utf-8")
    resolved = ms._resolve_path("numbers.tex", [base_a, base_b])
    assert resolved == target.resolve()


def test_resolve_path_appends_extra_extension(tmp_path: Path) -> None:
    target = tmp_path / "references.bib"
    target.write_text("x", encoding="utf-8")
    resolved = ms._resolve_path("references", [tmp_path], extra_exts=(".bib",))
    assert resolved == target.resolve()


def test_resolve_path_fails_loud_and_lists_tried_paths(tmp_path: Path) -> None:
    with pytest.raises(ms.SubmissionBuildError) as excinfo:
        ms._resolve_path("missing.pdf", [tmp_path])
    assert "missing.pdf" in str(excinfo.value)
    assert str(tmp_path) in str(excinfo.value)


def test_macro_placeholder_detected() -> None:
    assert ms._is_macro_placeholder("#1")
    assert not ms._is_macro_placeholder("sections/01_uvod")


# --------------------------------------------------------------------------
# _rewrite_path_command: regression test for the "the whole \input{...}
# wrapper must survive rewriting, not just the bare argument" bug found
# during development (2026-09-18).
# --------------------------------------------------------------------------


def test_rewrite_input_preserves_command_wrapper(tmp_path: Path) -> None:
    target = tmp_path / "sections" / "01_uvod.tex"
    target.parent.mkdir()
    target.write_text("x", encoding="utf-8")
    assigner = ms.FlatNameAssigner()
    text = r"\input{sections/01_uvod}"
    rewritten = ms._rewrite_path_command(
        text,
        ms._INPUT_RE,
        1,
        lambda ref: ms._resolve_path(ref, [tmp_path], extra_exts=(".tex",)),
        assigner,
        lambda match, flat_name: f"\\input{{{flat_name}}}",
    )
    assert rewritten == r"\input{01_uvod.tex}"


def test_rewrite_input_skips_macro_parameter_placeholder(tmp_path: Path) -> None:
    # \input{#1} appears inside the \rawtableinput macro DEFINITION in
    # supplement.tex; it must never be treated as a file reference.
    assigner = ms.FlatNameAssigner()
    text = r"\newcommand{\rawtableinput}[1]{\input{#1}}"
    rewritten = ms._rewrite_path_command(
        text,
        ms._INPUT_RE,
        1,
        lambda ref: (_ for _ in ()).throw(AssertionError("should not resolve #1")),
        assigner,
        lambda match, flat_name: f"\\input{{{flat_name}}}",
    )
    assert rewritten == text


def test_rewrite_includegraphics_preserves_options(tmp_path: Path) -> None:
    img_dir = tmp_path / "img"
    img_dir.mkdir()
    (img_dir / "fig.pdf").write_text("x", encoding="utf-8")
    assigner = ms.FlatNameAssigner()
    text = r"\includegraphics[width=\textwidth]{../img/fig.pdf}"
    # The reference is written relative to a "master" directory one level
    # below tmp_path, exactly like clanek_en/dami/main_dami.tex resolving
    # "../../clanek/img/x.pdf" relative to its own directory.
    master_dir = tmp_path / "dami"
    master_dir.mkdir()
    rewritten = ms._rewrite_path_command(
        text,
        ms._INCLUDEGRAPHICS_RE,
        2,
        lambda ref: ms._resolve_path(ref, [master_dir]),
        assigner,
        ms._rebuild_wrapped,
    )
    assert rewritten == r"\includegraphics[width=\textwidth]{fig.pdf}"


def test_rewrite_rawtableinput_variants_are_disambiguated(tmp_path: Path) -> None:
    # Regression test for a real bug (2026-09-18): \rawtableinputlong (added
    # alongside \rawtableinput/\rawtableinputwide for longtable-based
    # supplement tables, see clanek_en/supplement/sections/s6_full_tables.tex)
    # was not recognised by any of the three rawtableinput* regexes, so its
    # path was left un-rewritten and the referenced file was never copied
    # into the flat package - a fatal "File not found" only surfaced during
    # the full LaTeX verification compile, not at flattening time.
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    for stem in ("plain", "wide", "long"):
        (tables_dir / f"{stem}.tex").write_text("x", encoding="utf-8")
    assigner = ms.FlatNameAssigner()

    def _rewrite(pattern: re.Pattern[str], text: str) -> str:
        return ms._rewrite_path_command(
            text, pattern, 2,
            lambda ref: ms._resolve_path(ref, [tables_dir]),
            assigner, ms._rebuild_wrapped,
        )

    assert _rewrite(ms._RAWTABLEINPUT_RE, r"\rawtableinput{plain.tex}") == r"\rawtableinput{plain.tex}"
    assert _rewrite(ms._RAWTABLEINPUTWIDE_RE, r"\rawtableinputwide{wide.tex}") == r"\rawtableinputwide{wide.tex}"
    assert _rewrite(ms._RAWTABLEINPUTLONG_RE, r"\rawtableinputlong{long.tex}") == r"\rawtableinputlong{long.tex}"
    # The plain-\rawtableinput pattern must not also match the wide/long
    # variants (would double-process / corrupt them).
    assert _rewrite(ms._RAWTABLEINPUT_RE, r"\rawtableinputwide{wide.tex}") == r"\rawtableinputwide{wide.tex}"
    assert _rewrite(ms._RAWTABLEINPUT_RE, r"\rawtableinputlong{long.tex}") == r"\rawtableinputlong{long.tex}"


def test_rewrite_bibliography_strips_extension(tmp_path: Path) -> None:
    clanek_dir = tmp_path / "clanek"
    clanek_dir.mkdir()
    (clanek_dir / "references.bib").write_text("x", encoding="utf-8")
    assigner = ms.FlatNameAssigner()
    text = r"\bibliography{../clanek/references}"
    master_dir = tmp_path / "dami"
    master_dir.mkdir()
    rewritten = ms._rewrite_path_command(
        text,
        ms._BIBLIOGRAPHY_RE,
        2,
        lambda ref: ms._resolve_path(ref, [master_dir], extra_exts=(".bib",)),
        assigner,
        ms._rebuild_wrapped,
        strip_exts=(".bib",),
    )
    assert rewritten == r"\bibliography{references}"


# --------------------------------------------------------------------------
# External-aux wiring: fail loud if the dependency was not built+verified.
# --------------------------------------------------------------------------


def test_wire_external_aux_fails_loud_without_verified_dependency(tmp_path: Path) -> None:
    result = ms.BuildResult(
        flat_dir=tmp_path,
        root_tex_flat_name="supplement.tex",
        manifest=[],
        needs_external_aux_from="dami",
    )
    with pytest.raises(ms.SubmissionBuildError, match="not built and verified"):
        ms._wire_external_aux(result, tmp_path, verified={})


def test_wire_external_aux_is_noop_when_not_needed(tmp_path: Path) -> None:
    result = ms.BuildResult(
        flat_dir=tmp_path, root_tex_flat_name="main_dami.tex", manifest=[], needs_external_aux_from=None
    )
    ms._wire_external_aux(result, tmp_path, verified={})  # must not raise


# --------------------------------------------------------------------------
# Integration: flatten the REAL project tree (no LaTeX compile - fast).
# --------------------------------------------------------------------------


@pytest.mark.skipif(not _DAMI_SOURCES_PRESENT, reason="clanek/ (img, references.bib) is not present locally.")
def test_build_flat_package_dami_matches_expected_layout(tmp_path: Path) -> None:
    result = ms.build_flat_package(ms.DAMI_SPEC, tmp_path)
    flat_names = {name for name, _ in result.manifest}
    assert "main_dami.tex" in flat_names
    assert "sn-jnl.cls" in flat_names
    assert "sn-basic.bst" in flat_names
    assert "references.bib" in flat_names
    assert "numbers.tex" in flat_names
    for section in ("01_uvod", "02_souvisejici_prace", "03_metoda", "04_experimenty",
                    "05_vysledky", "06_diskuse", "07_zaver"):
        assert f"{section}.tex" in flat_names

    main_text = (result.flat_dir / "main_dami.tex").read_text(encoding="utf-8")
    assert r"\input{01_uvod.tex}" in main_text
    assert r"\InputIfFileExists{numbers.tex}{}{}" in main_text
    assert r"\bibliography{references}" in main_text


def test_build_flat_package_rewires_externaldocument_when_present(tmp_path: Path) -> None:
    # Synthetic root document, deliberately decoupled from the real (and
    # evolving) clanek_en/supplement/supplement.tex: on 2026-09-18 the author
    # removed \externaldocument from the actual supplement (unreliable once
    # the publisher recompiles the main article separately from the
    # checked-in preprint; replaced by hand-written cross-references, see
    # src/validate_supplement_refs.py), so an integration test asserting the
    # real file still contains it would now be testing dead content instead
    # of the rewiring mechanism. This test keeps exercising that mechanism in
    # isolation, in case a future package reintroduces a real cross-document
    # reference.
    root = tmp_path / "doc.tex"
    root.write_text(
        "\\documentclass{article}\n"
        "\\usepackage{xr}\n"
        "\\externaldocument[M-]{../main}\n"
        "\\begin{document}x\\end{document}\n",
        encoding="utf-8",
    )
    spec = ms.DocumentSpec(
        key="synthetic_supplement",
        root_tex=root,
        master_dir=tmp_path,
        cwd_dir=tmp_path,
        reference_pdf=tmp_path / "doc.pdf",
        output_subdir_name="synthetic_supplement",
    )
    result = ms.build_flat_package(spec, tmp_path / "out")
    assert result.needs_external_aux_from == "dami"
    text = (result.flat_dir / "doc.tex").read_text(encoding="utf-8")
    assert f"\\externaldocument[M-]{{{ms.DAMI_SPEC.root_tex.stem}}}" in text
    # The rewritten \externaldocument command itself no longer targets the
    # original relative path.
    assert r"\externaldocument[M-]{../main}" not in text


# --------------------------------------------------------------------------
# Full integration: real packages + mandatory LaTeX verification, matching
# what src/run_make_submission.bat actually does. Skipped unless pdflatex is
# available and both reference PDFs already exist (author-run once via
# clanek_en/dami/build_dami.bat and clanek_en/supplement/compile.bat).
# --------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_PDFLATEX, reason="pdflatex is not on PATH.")
@pytest.mark.skipif(
    not (_DAMI_SOURCES_PRESENT and _SUPPLEMENT_SOURCES_PRESENT
         and ms.DAMI_SPEC.reference_pdf.is_file() and ms.SUPPLEMENT_SPEC.reference_pdf.is_file()),
    reason="Reference PDFs (clanek_en/dami/main_dami.pdf, clanek_en/supplement/supplement.pdf) "
    "are not present locally; compile them once first.",
)
def test_run_builds_and_verifies_both_packages_end_to_end(tmp_path: Path) -> None:
    verified = ms.run(tmp_path, "both", skip_verify=False)
    assert set(verified) == {"dami", "supplement"}
    assert verified["dami"].pages == verified["dami"].reference_pages
    assert verified["supplement"].pages == verified["supplement"].reference_pages
    # The packages land under a directory named after the target journal, so
    # a later submission elsewhere gets its own directory instead of
    # overwriting this one.
    journal_dir = tmp_path / "dami"
    assert (journal_dir / "manuscript_en" / "main_dami.tex").is_file()
    assert (journal_dir / "supplement_en" / "supplement.tex").is_file()
    stamp = (journal_dir / ms.JOURNAL_STAMP_NAME).read_text(encoding="utf-8")
    assert stamp.splitlines()[0] == "journal: dami"
    assert "Data Mining and Knowledge Discovery" in stamp


# --------------------------------------------------------------------------
# Journal identity: one directory per target journal.
# --------------------------------------------------------------------------


def test_clean_verification_artifacts_keeps_sources_and_bbl(tmp_path: Path) -> None:
    for name in ("main_dami.tex", "main_dami.bbl", "main_dami.pdf", "references.bib",
                 "main_dami.aux", "main_dami.log", "main_dami.blg", "main_dami.out"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    removed = ms._clean_verification_artifacts(tmp_path, keep=frozenset())
    assert set(removed) == {"main_dami.aux", "main_dami.log", "main_dami.blg", "main_dami.out"}
    left = {p.name for p in tmp_path.iterdir()}
    assert left == {"main_dami.tex", "main_dami.bbl", "main_dami.pdf", "references.bib"}


def test_clean_verification_artifacts_keeps_a_borrowed_aux(tmp_path: Path) -> None:
    (tmp_path / "main_dami.aux").write_text("x", encoding="utf-8")
    (tmp_path / "supplement.aux").write_text("x", encoding="utf-8")
    removed = ms._clean_verification_artifacts(tmp_path, keep=frozenset({"main_dami.aux"}))
    assert removed == ["supplement.aux"]
    assert (tmp_path / "main_dami.aux").is_file()


def test_claim_journal_dir_creates_stamped_directory(tmp_path: Path) -> None:
    journal = ms.JOURNALS["dami"]
    journal_dir = ms._claim_journal_dir(tmp_path, journal)
    assert journal_dir == tmp_path / "dami"
    assert journal_dir.is_dir()


def test_claim_journal_dir_refuses_a_directory_stamped_for_another_journal(
    tmp_path: Path,
) -> None:
    (tmp_path / "dami").mkdir()
    (tmp_path / "dami" / ms.JOURNAL_STAMP_NAME).write_text(
        "journal: someother\n", encoding="utf-8"
    )
    with pytest.raises(ms.SubmissionBuildError, match="someother"):
        ms._claim_journal_dir(tmp_path, ms.JOURNALS["dami"])


def test_journal_stamp_lists_the_upload_map(tmp_path: Path) -> None:
    journal = ms.JOURNALS["dami"]
    ms._write_journal_stamp(tmp_path, journal, {})
    stamp = (tmp_path / ms.JOURNAL_STAMP_NAME).read_text(encoding="utf-8")
    assert "https://dami.edmgr.com" in stamp
    assert "Regular Paper" in stamp
    for path_in_package, slot in journal.upload_slots:
        assert path_in_package in stamp
        assert slot in stamp
    assert "NOT VERIFIED" in stamp
