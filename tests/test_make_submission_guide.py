# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-19
# License: see the LICENSE file in the repository root
"""Tests for the suggested-reviewer source used by
src/tools/make_submission_guide.py. The reviewer list is the one part of the
cover letter that is neither extracted from the manuscript nor passed on the
command line, so it has to fail loud when the file is missing or an entry is
incomplete: the journal requires an institutional e-mail (or another way to
verify the person) for every suggested reviewer."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.tools import make_submission_guide as mg

REPO_ROOT = Path(__file__).resolve().parents[1]
REVIEWERS_YAML = REPO_ROOT / "dami_submission" / "reviewers.yaml"

COMPLETE_ENTRY = {
    "name": "Jane Doe",
    "affiliation": "Example University, Elsewhere",
    "email": "jane.doe@example.edu",
    "orcid": "0000-0002-1825-0097",
    "reason": "works on the same loss family",
}


def _write_yaml(path: Path, entries: list[dict[str, str]]) -> None:
    lines = ["reviewers:"]
    for entry in entries:
        first = True
        for key, value in entry.items():
            prefix = "  - " if first else "    "
            lines.append(f'{prefix}{key}: "{value}"')
            first = False
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_missing_file_fails_loud(tmp_path: Path) -> None:
    with pytest.raises(mg.SubmissionGuideError, match="not found"):
        mg.load_reviewers(tmp_path / "reviewers.yaml")


def test_empty_list_fails_loud(tmp_path: Path) -> None:
    path = tmp_path / "reviewers.yaml"
    path.write_text("reviewers: []\n", encoding="utf-8")
    with pytest.raises(mg.SubmissionGuideError, match="No 'reviewers' entries"):
        mg.load_reviewers(path)


@pytest.mark.parametrize("missing_field", sorted(COMPLETE_ENTRY))
def test_incomplete_entry_names_the_missing_field(
    tmp_path: Path, missing_field: str
) -> None:
    entry = {k: v for k, v in COMPLETE_ENTRY.items() if k != missing_field}
    path = tmp_path / "reviewers.yaml"
    _write_yaml(path, [entry])
    with pytest.raises(mg.SubmissionGuideError, match=missing_field):
        mg.load_reviewers(path)


def test_complete_entry_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "reviewers.yaml"
    _write_yaml(path, [COMPLETE_ENTRY])
    loaded = mg.load_reviewers(path)
    assert len(loaded) == 1
    assert loaded[0]["email"] == COMPLETE_ENTRY["email"]


def test_repository_reviewer_list_is_complete_and_distinct() -> None:
    """The checked-in list must satisfy what the journal asks of suggested
    reviewers: everyone identifiable, and no two from the same institution."""
    reviewers = mg.load_reviewers(REVIEWERS_YAML)
    assert len(reviewers) >= 3
    emails = [r["email"] for r in reviewers]
    assert len(set(emails)) == len(emails)
    affiliations = [r["affiliation"] for r in reviewers]
    assert len(set(affiliations)) == len(affiliations)
    for reviewer in reviewers:
        assert "@" in reviewer["email"]


# --------------------------------------------------------------------------
# Cover letter as PDF
# --------------------------------------------------------------------------

MINIMAL_LETTER = (
    "# Cover letter\n\n> provenance note\n\n---\n\nDear Editor,\n\n"
    "The exponent D_ij^(-alpha) is 100% of the point & the rest.\n\n"
    "We suggest the following reviewers:\n\n"
    "- Jane Doe, Example University; jane.doe@example.edu; ORCID X - reason.\n\n"
    "On behalf of both authors,\n\nJane Author\nExample University\n"
    "jane.author@example.edu\n"
)


def test_tex_escaping_covers_special_characters() -> None:
    tex = mg.build_cover_letter_tex(MINIMAL_LETTER)
    assert r"D\_ij\textasciicircum{}(-alpha)" in tex
    assert r"100\% of the point \& the rest" in tex


def test_provenance_note_is_not_typeset() -> None:
    tex = mg.build_cover_letter_tex(MINIMAL_LETTER)
    assert "provenance note" not in tex
    assert "Dear Editor," in tex


def test_bullets_become_itemize() -> None:
    tex = mg.build_cover_letter_tex(MINIMAL_LETTER)
    assert r"\begin{itemize}" in tex
    assert "Jane Doe, Example University" in tex


def test_signature_block_breaks_by_line() -> None:
    tex = mg.build_cover_letter_tex(MINIMAL_LETTER)
    signature = tex[tex.index("On behalf of") :]
    assert signature.count(r"\\") >= 3


def test_markdown_without_separator_fails_loud() -> None:
    with pytest.raises(mg.SubmissionGuideError, match="separator"):
        mg.build_cover_letter_tex("Dear Editor,\n\nno separator here\n")
