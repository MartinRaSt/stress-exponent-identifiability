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
Generates the two DAMI submission texts (Springer Nature SNAPP), ON REQUEST ONLY
(the author explicitly asked that this NOT run automatically before the
manuscript is final and the Zenodo DOI exists - see dami_submission/
00_POSTUP_SUBMISSION.md):

  dami_submission/01_TEXTY_DO_FORMULARE.md
      Every text field the author copy-pastes into the SNAPP submission web
      form (title, abstract, keywords, authors, affiliation, funding,
      declarations), as PLAIN TEXT ready to paste - no LaTeX markup, no
      unresolved macros.
  dami_submission/02_COVER_LETTER.md
      The cover letter to the editor, plain text.

Both are derived from the TYPESET manuscript, never hand-typed, so nothing in
them can silently drift from the submitted PDF:

  - Title, abstract, keywords, author names/emails/ORCID, affiliation and the
    Statements-and-Declarations paragraphs are parsed out of the .tex source
    clanek_en/dami/main_dami.tex (the ORCID values live only in a comment
    there, since the manuscript body itself must not carry them - see that
    file). The .tex is used instead of the .pdf because pypdf's text
    extraction breaks up the math typesetting (subscripts/superscripts lose
    their grouping, e.g. "N= 26" or "w ij =D -alpha ij") - see the module
    docstring discussion and the comparison run during development.
  - Every number-macro used inside the abstract (e.g. \\numExpOneDatasets,
    \\pooledAucGainVsAlphaZeroMedian) is expanded from
    clanek/generated/numbers.tex, which is itself generated from the
    experiment CSVs by src/experiments/export_numbers.py - so every number
    in the submission texts still traces back to a source, per CLAUDE.md's
    "no number without a source" rule.
  - Remaining inline LaTeX math ($...$, subscripts, superscripts, \\alpha,
    \\times, \\%, \\num{...}) is rewritten into plain-text notation
    (e.g. "w_ij = D_ij^(-alpha)"), never guessed: if anything is left that
    this script does not recognize, it is wrapped in a visible
    "<<< ZKONTROLUJ: ... >>>" marker instead of being silently dropped or
    approximated (fail-loud/no-fabrication rule).

The Zenodo DOI is a MANDATORY command-line argument: the cover letter cites it
as the archived data+code record, and a cover letter that silently kept a
placeholder there would be worse than one that refuses to be generated.

Run (from anywhere):
    venv\\python.exe -m src.tools.make_submission_guide 10.5281/zenodo.XXXXXXX
or: src\\run_make_submission_guide.bat 10.5281/zenodo.XXXXXXX
    src/run_make_submission_guide.sh 10.5281/zenodo.XXXXXXX

CLI:
    zenodo_doi        Required. The DOI of the Zenodo archive backing this
                       submission, e.g. 10.5281/zenodo.1234567 (no URL
                       prefix - the script builds https://doi.org/<doi>).
    --output-dir PATH  Where to write the two .md files (default:
                       <repo>/dami_submission).
    --main-tex PATH    Source manuscript (default:
                       <repo>/clanek_en/dami/main_dami.tex).
    --numbers-tex PATH Generated number macros (default:
                       <repo>/clanek/generated/numbers.tex).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAIN_TEX = PROJECT_ROOT / "clanek_en" / "dami" / "main_dami.tex"
DEFAULT_NUMBERS_TEX = PROJECT_ROOT / "clanek" / "generated" / "numbers.tex"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dami_submission"

# DAMI's own submission guidelines (recorded in
# podklady/2026-09-16_dami_pokyny_a_kontrolni_seznam.md): abstract must be
# 150-250 words. Not a modelling parameter, so it stays a named constant here
# rather than moving to a YAML config (mirrors FORBIDDEN_LOG_MARKERS in
# src/tools/make_submission.py for the same kind of fixed, documented rule).
DAMI_ABSTRACT_WORD_MIN = 150
DAMI_ABSTRACT_WORD_MAX = 250

ZKONTROLUJ_OPEN = "<<< ZKONTROLUJ: "
ZKONTROLUJ_CLOSE = " >>>"

# DOI syntax per the DOI Handbook (10.<registrant>/<suffix>); loose on
# purpose, this only guards against an obviously-wrong or placeholder value.
_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")


class SubmissionGuideError(RuntimeError):
    """Fail-loud error: a field could not be located in the manuscript
    source, a macro used in it has no definition, or a required CLI argument
    is missing/malformed. Never caught silently."""


# --------------------------------------------------------------------------
# Generic balanced-brace / LaTeX-command extraction (no LaTeX parser
# dependency needed for the handful of commands used in the front matter).
# --------------------------------------------------------------------------


def _extract_balanced(text: str, open_idx: int) -> tuple[str, int]:
    """Given the index of an opening '{', returns (inner content, index just
    past the matching closing '}'). Fails loud on unbalanced braces."""
    if text[open_idx] != "{":
        raise SubmissionGuideError(
            f"_extract_balanced called at index {open_idx}, which is not '{{'."
        )
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1 : i], i + 1
    raise SubmissionGuideError(
        f"Unbalanced braces starting at index {open_idx} while extracting "
        "a LaTeX command argument."
    )


def _find_and_extract(text: str, command: str, *, occurrence: int = 0) -> str:
    """Finds the `occurrence`-th (0-based) \\command[...]{...} or
    \\command{...} in text and returns the content of the final {...} group,
    skipping over any leading [...] optional argument. `command` may contain
    a literal leading '*' (e.g. "affil*"). Fails loud if not found."""
    pattern = re.compile(r"\\" + re.escape(command) + r"(?![A-Za-z@])")
    matches = list(pattern.finditer(text))
    if occurrence >= len(matches):
        raise SubmissionGuideError(
            f"Could not find occurrence {occurrence} of \\{command}{{...}} "
            "in the manuscript source."
        )
    idx = matches[occurrence].end()
    while idx < len(text) and text[idx] == "[":
        close = text.find("]", idx)
        if close == -1:
            raise SubmissionGuideError(
                f"Unclosed '[' after \\{command} at index {idx}."
            )
        idx = close + 1
    while idx < len(text) and text[idx] in " \t\r\n":
        idx += 1
    if idx >= len(text) or text[idx] != "{":
        raise SubmissionGuideError(
            f"Expected '{{' after \\{command}, found "
            f"{text[idx:idx + 20]!r} instead."
        )
    content, _ = _extract_balanced(text, idx)
    return content


def _extract_subsection(text: str, heading: str) -> str:
    """Returns the body text of `\\subsection*{heading}...` up to the next
    `\\subsection*{`, `\\bibliography{` or `\\end{document}`. Fails loud if
    the heading is not found."""
    pattern = re.compile(
        r"\\subsection\*\{" + re.escape(heading) + r"\}(.*?)"
        r"(?=\\subsection\*\{|\\bibliography\{|\\end\{document\})",
        re.DOTALL,
    )
    match = pattern.search(text)
    if match is None:
        raise SubmissionGuideError(
            f"Subsection '{heading}' not found in the manuscript source."
        )
    return match.group(1).strip()


# --------------------------------------------------------------------------
# clanek/generated/numbers.tex -> {macro name: plain value}
# --------------------------------------------------------------------------

_NEWCOMMAND_RE = re.compile(
    r"\\newcommand\{\\(\w+)\}\{((?:[^{}]|\{[^{}]*\})*)\}"
)
_NUM_WRAPPER_RE = re.compile(r"^\\num\{([^{}]*)\}$")


def load_number_macros(numbers_tex: Path) -> dict[str, str]:
    """Parses `\\newcommand{\\name}{value}` lines from numbers.tex, stripping
    the siunitx `\\num{...}` wrapper so the value is a bare, plain-text
    number. Fails loud if the file is missing or contains no macros at all
    (rather than silently returning an empty rule set)."""
    if not numbers_tex.is_file():
        raise SubmissionGuideError(
            f"Generated numbers file not found: {numbers_tex}\n"
            "Run src/run_main.bat (or .sh) first to produce it."
        )
    text = numbers_tex.read_text(encoding="utf-8")
    macros: dict[str, str] = {}
    for match in _NEWCOMMAND_RE.finditer(text):
        name, raw_value = match.group(1), match.group(2)
        num_match = _NUM_WRAPPER_RE.match(raw_value)
        macros[name] = num_match.group(1) if num_match else raw_value
    if not macros:
        raise SubmissionGuideError(
            f"No \\newcommand macros parsed out of {numbers_tex} - "
            "check the file was generated correctly."
        )
    return macros


def _expand_macros(text: str, macros: dict[str, str]) -> str:
    """Replaces every \\macroName or \\macroName{} occurring in `text` with
    its value from `macros`. Names are tried longest-first so e.g.
    \\numExpOneDatasetsPareto is not clobbered by a match on the shorter
    \\numExpOneDatasets. Macros not referenced in `text` are ignored; a macro
    referenced in `text` but absent from `macros` is left untouched and will
    be caught by the residual-LaTeX check downstream (fail-visible, not
    fail-silent)."""
    if not macros:
        return text
    names = sorted(macros.keys(), key=len, reverse=True)
    pattern = re.compile(
        r"\\(" + "|".join(re.escape(n) for n in names) + r")(\{\})?(?![A-Za-z])"
    )
    return pattern.sub(lambda m: macros[m.group(1)], text)


# --------------------------------------------------------------------------
# LaTeX math / typography -> plain text
# --------------------------------------------------------------------------


def _format_superscript(body: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9]", body):
        return "^" + body
    return "^(" + body + ")"


def _latex_math_to_plain(text: str) -> str:
    """Rewrites the handful of LaTeX math/typography constructs used in the
    manuscript front matter into plain text readable in a web-form field
    (no LaTeX, no bare macros). Only handles constructs actually observed in
    this manuscript; anything else survives untouched and is caught by
    `_flag_residual_latex` instead of being guessed at."""
    # siunitx \num{...} left over from macro expansion (e.g. \pooledAuc...
    # expands to "\num{0.0280}") -> bare number.
    text = re.sub(r"\\num\{([^{}]*)\}", r"\1", text)
    # \href{url}{label} -> the label alone: the form takes plain text, and
    # the ORCID note prints the identifier as its own label anyway.
    text = re.sub(r"\\href\{[^{}]*\}\{([^{}]*)\}", r"\1", text)
    # \url{...} and \texttt{...} carry no typesetting meaning in plain text.
    text = re.sub(r"\\url\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\texttt\{([^{}]*)\}", r"\1", text)
    # Inline math delimiters carry no information once the content itself
    # has been converted to plain text.
    text = text.replace("$", "")
    text = text.replace(r"\alpha", "alpha")
    text = text.replace(r"\times", " x ")
    text = re.sub(r"\^\{([^{}]*)\}", lambda m: _format_superscript(m.group(1)), text)
    text = re.sub(
        r"\^([A-Za-z0-9])(?![A-Za-z0-9{])",
        lambda m: _format_superscript(m.group(1)),
        text,
    )
    text = re.sub(r"_\{([^{}]*)\}", r"_\1", text)
    text = re.sub(r"_([A-Za-z0-9])(?![A-Za-z0-9{])", r"_\1", text)
    text = text.replace(r"\%", "%")
    text = text.replace(r"\,", " ")
    text = text.replace("~", " ")
    text = text.replace("--", "-")
    text = re.sub(r"[ \t]*\n[ \t]*", " ", text)
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text


# A LaTeX line comment: an unescaped '%' to end of line. Author-only working
# notes (e.g. the "%% ORCID ..." comments, or the notes explaining why the
# supplement's \ref is a manual number) must never leak into a field pasted
# into the submission form. The one deliberate exception - reading the ORCID
# out of its comment - is done by a dedicated regex on the raw text before
# this stripping is applied (see _AUTHOR_RE).
_COMMENT_RE = re.compile(r"(?<!\\)%.*")


def _strip_latex_comments(text: str) -> str:
    return "\n".join(_COMMENT_RE.sub("", line) for line in text.split("\n"))


# Anything matching this after `_latex_math_to_plain` is a construct the
# converter does not know about - flagged, never guessed.
_RESIDUAL_LATEX_RE = re.compile(r"\\[A-Za-z]+|\{|\}|\$|\?\?")


def _flag_residual_latex(label: str, text: str, warnings: list[str]) -> str:
    """Wraps `text` in a visible <<< ZKONTROLUJ >>> marker and records a
    console warning if any unconverted LaTeX/markup remains. Returns `text`
    unchanged (with the marker prefix) rather than raising, so the rest of
    the document can still be generated for manual review."""
    found = sorted(set(_RESIDUAL_LATEX_RE.findall(text)))
    if not found:
        return text
    warnings.append(f"{label}: unconverted markup left in text: {found}")
    marker = (
        f"{ZKONTROLUJ_OPEN}unconverted LaTeX {found} - fix the converter "
        f"or edit by hand{ZKONTROLUJ_CLOSE}"
    )
    return f"{marker}\n{text}"


def _plain(text: str, macros: dict[str, str], label: str, warnings: list[str]) -> str:
    """Full pipeline: strip LaTeX line comments, expand number macros,
    convert remaining LaTeX math to plain text, flag anything left over."""
    text = _strip_latex_comments(text)
    text = _expand_macros(text, macros)
    text = _latex_math_to_plain(text)
    return _flag_residual_latex(label, text, warnings)


# --------------------------------------------------------------------------
# Front-matter extraction
# --------------------------------------------------------------------------

# The front matter carries the ORCIDs inside \equalcont (the class's own
# \orcid macro needs Orcidlogo.eps, which MiKTeX does not ship, and its
# absence pushes the title block onto page two). They are therefore read
# from that note, in author order.
# The front matter carries the ORCIDs inside the equal-contribution note:
# the class's own orcid macro needs Orcidlogo.eps, which MiKTeX does not
# ship, and its absence pushes the title block onto page two.
_AUTHOR_RE = re.compile(
    r"\\author(\*?)\[[^\]]*\]\{\\fnm\{([^{}]*)\}\s*\\sur\{([^{}]*)\}\}\s*"
    r"\\email\{([^{}]*)\}\s*"
    r"\\equalcont\{(.*?)\}\s*$",
    re.MULTILINE,
)

_ORCID_RE = re.compile(r"ORCID[^:]*:[^0-9]*(\d{4}-\d{4}-\d{4}-\d{3}[\dX])")

class Author:
    def __init__(self, corresponding: bool, given: str, family: str, email: str,
                 equal_contribution_note: str, orcid: str) -> None:
        self.corresponding = corresponding
        self.given = given
        self.family = family.replace("~", " ")
        self.email = email
        self.equal_contribution_note = equal_contribution_note
        self.orcid = orcid

    @property
    def full_name(self) -> str:
        return f"{self.given} {self.family}"


def extract_authors(tex: str) -> list[Author]:
    matches = list(_AUTHOR_RE.finditer(tex))
    if not matches:
        raise SubmissionGuideError(
            "No \\author{...}\\email{...}\\equalcont{...} block found in the "
            "manuscript source - the front-matter layout may have changed; "
            "update _AUTHOR_RE."
        )
    # One ORCID per author, in the order the authors are declared; the note
    # is repeated for every author, so the identifiers are read once.
    orcids = _ORCID_RE.findall(matches[0].group(5))
    if len(orcids) != len(matches):
        raise SubmissionGuideError(
            f"Found {len(matches)} author(s) but {len(orcids)} ORCID(s) in "
            "the equal-contribution note; every author needs one (the "
            "journal asks for it in the submission form)."
        )
    return [
        Author(
            corresponding=(m.group(1) == "*"),
            given=m.group(2),
            family=m.group(3),
            email=m.group(4),
            equal_contribution_note=m.group(5),
            orcid=orcid,
        )
        for m, orcid in zip(matches, orcids)
    ]


_AFFIL_RE = re.compile(
    r"\\affil\*?\[[^\]]*\]\{"
    r"\\orgdiv\{([^{}]*)\}, \\orgname\{([^{}]*)\}, "
    r"\\orgaddress\{\\street\{([^{}]*)\}, \\city\{([^{}]*)\}, "
    r"\\postcode\{([^{}]*)\}, \\country\{([^{}]*)\}\}\}"
)


def extract_affiliation_parts(tex: str) -> dict[str, str]:
    match = _AFFIL_RE.search(tex)
    if match is None:
        raise SubmissionGuideError(
            "No \\affil{\\orgdiv{...}, \\orgname{...}, \\orgaddress{...}} "
            "block found - the affiliation markup may have changed; update "
            "_AFFIL_RE."
        )
    keys = ("orgdiv", "orgname", "street", "city", "postcode", "country")
    return {k: v for k, v in zip(keys, match.groups())}


_FUNDING_RE = re.compile(
    r"supported by (.+?)\s*\[grant number ([^\]]+)\]", re.DOTALL
)
_GITHUB_URL_RE = re.compile(r"https://github\.com/\S+?(?=[}\s.,;]|$)")


def extract_funding(funding_paragraph_raw: str) -> tuple[str, str]:
    funding_paragraph_raw = _strip_latex_comments(funding_paragraph_raw)
    match = _FUNDING_RE.search(funding_paragraph_raw)
    if match is None:
        raise SubmissionGuideError(
            "Could not find 'supported by ... [grant number ...]' in the "
            "Funding statement - the wording may have changed; update "
            "_FUNDING_RE or edit the funder/grant fields by hand."
        )
    funder = _latex_math_to_plain(match.group(1))
    grant_number = _latex_math_to_plain(match.group(2))
    return funder, grant_number


def extract_repository_url(tex: str) -> str:
    match = _GITHUB_URL_RE.search(tex)
    if match is None:
        raise SubmissionGuideError(
            "No https://github.com/... URL found in the manuscript source "
            "(expected in the Code availability statement)."
        )
    return match.group(0)


# --------------------------------------------------------------------------
# Document assembly
# --------------------------------------------------------------------------


def _word_count(text: str) -> int:
    return len(text.split())


def build_form_texts_markdown(
    *, main_tex_path: Path, numbers_tex_path: Path
) -> tuple[str, list[str]]:
    """Builds the content of 01_TEXTY_DO_FORMULARE.md. Returns
    (markdown_text, warnings)."""
    if not main_tex_path.is_file():
        raise SubmissionGuideError(f"Manuscript source not found: {main_tex_path}")
    tex = main_tex_path.read_text(encoding="utf-8")
    macros = load_number_macros(numbers_tex_path)
    warnings: list[str] = []

    # \title[short form]{full title}: the optional [...] is skipped by
    # _find_and_extract, so the mandatory {...} it returns is already the
    # full, real title (not the short running head).
    title_raw = _find_and_extract(tex, "title")
    title = _plain(title_raw, macros, "Nazev", warnings)

    abstract_raw = _find_and_extract(tex, "abstract")
    abstract = _plain(abstract_raw, macros, "Abstrakt", warnings)
    word_count = _word_count(abstract)
    in_limit = DAMI_ABSTRACT_WORD_MIN <= word_count <= DAMI_ABSTRACT_WORD_MAX
    if not in_limit:
        print(
            f"[make_submission_guide] WARNING: abstract is {word_count} words, "
            f"outside the DAMI limit of {DAMI_ABSTRACT_WORD_MIN}-"
            f"{DAMI_ABSTRACT_WORD_MAX} words.",
            file=sys.stderr,
        )
        warnings.append(
            f"Abstrakt: {word_count} slov mimo limit DAMI "
            f"{DAMI_ABSTRACT_WORD_MIN}-{DAMI_ABSTRACT_WORD_MAX}"
        )

    keywords_raw = _find_and_extract(tex, "keywords")
    keywords_plain = _plain(keywords_raw, macros, "Klicova slova", warnings)
    keywords = [k.strip() for k in keywords_plain.split(",") if k.strip()]

    authors = extract_authors(tex)
    affil_parts = extract_affiliation_parts(tex)
    affiliation = _plain(
        ", ".join(
            [
                affil_parts["orgdiv"],
                affil_parts["orgname"],
                affil_parts["street"],
                affil_parts["city"],
                f'{affil_parts["postcode"]}, {affil_parts["country"]}',
            ]
        ),
        macros,
        "Afiliace",
        warnings,
    )

    funding_para_raw = _extract_subsection(tex, "Funding")
    funder_raw, grant_number_raw = extract_funding(funding_para_raw)
    funder = _plain(funder_raw, macros, "Financovani (poskytovatel)", warnings)
    grant_number = _plain(grant_number_raw, macros, "Financovani (cislo grantu)", warnings)
    funding_statement = _plain(funding_para_raw, macros, "Financovani (plne zneni)", warnings)

    competing_interests = _plain(
        _extract_subsection(tex, "Competing Interests"), macros,
        "Competing Interests", warnings,
    )
    data_availability = _plain(
        _extract_subsection(tex, "Data availability"), macros,
        "Data availability", warnings,
    )
    code_availability = _plain(
        _extract_subsection(tex, "Code availability"), macros,
        "Code availability", warnings,
    )
    # Springer asks for these three in the form even when they do not apply;
    # leaving them out is a common reason for a submission to come back.
    ethics_approval = _plain(
        _extract_subsection(tex, "Ethics approval"), macros,
        "Ethics approval", warnings,
    )
    consent_participate = _plain(
        _extract_subsection(tex, "Consent to participate"), macros,
        "Consent to participate", warnings,
    )
    consent_publication = _plain(
        _extract_subsection(tex, "Consent for publication"), macros,
        "Consent for publication", warnings,
    )
    author_contributions_raw = _extract_subsection(tex, "Author contributions")
    author_contributions_raw = re.sub(
        r"\\textbf\{([^{}]*)\}", r"\1", author_contributions_raw
    )
    author_contributions = _plain(
        author_contributions_raw, macros, "Author contributions", warnings
    )

    timestamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: list[str] = []
    lines.append("# Texty do formulare SNAPP (DAMI)")
    lines.append("")
    lines.append(
        f"> AUTOGENEROVANO skriptem `src/tools/make_submission_guide.py` "
        f"dne {timestamp} ze zdroju:"
    )
    lines.append(f"> - `{main_tex_path.relative_to(PROJECT_ROOT).as_posix()}`")
    lines.append(f"> - `{numbers_tex_path.relative_to(PROJECT_ROOT).as_posix()}`")
    lines.append(
        "> NEUPRAVUJ RUCNE - zmen zdroj (rukopis) a spust skript znovu. "
        "Kazdy text nize je PROSTY TEXT pripraveny ke zkopirovani do "
        "formulare (zadny LaTeX)."
    )
    if warnings:
        lines.append(">")
        lines.append(
            f"> **POZOR: {len(warnings)} polozka/y potrebuje rucni kontrolu "
            "(viz <<< ZKONTROLUJ >>> znacky nize).**"
        )
    lines.append("")

    lines.append("## Nazev")
    lines.append(f"*(zdroj: `\\title` v {main_tex_path.name})*")
    lines.append("")
    lines.append(title)
    lines.append("")

    lines.append("## Abstrakt")
    status = "OK" if in_limit else "MIMO LIMIT"
    lines.append(
        f"*(zdroj: `\\abstract` v {main_tex_path.name}; pocet slov: "
        f"{word_count}, limit DAMI {DAMI_ABSTRACT_WORD_MIN}-"
        f"{DAMI_ABSTRACT_WORD_MAX} slov -> {status})*"
    )
    lines.append("")
    lines.append(abstract)
    lines.append("")

    lines.append("## Klicova slova")
    lines.append(f"*(zdroj: `\\keywords` v {main_tex_path.name})*")
    lines.append("")
    for kw in keywords:
        lines.append(f"- {kw}")
    lines.append("")

    lines.append("## Autori")
    lines.append(
        f"*(zdroj: `\\author`/`\\email`/`\\equalcont` v "
        f"{main_tex_path.name})*"
    )
    lines.append("")
    for i, author in enumerate(authors, start=1):
        role = " (corresponding author)" if author.corresponding else ""
        lines.append(f"### {i}. {author.full_name}{role}")
        lines.append(f"- Email: {author.email}")
        lines.append(f"- ORCID: {author.orcid}")
        # The note carries the ORCIDs as typeset LaTeX; the form takes
        # plain text only.
        note = _plain(
            author.equal_contribution_note, macros,
            f"Poznamka autora {i}", warnings,
        )
        lines.append(f"- Poznamka: {note}")
        lines.append("")

    lines.append("## Afiliace")
    lines.append(f"*(zdroj: `\\affil` v {main_tex_path.name}, stejna pro oba autory)*")
    lines.append("")
    lines.append(affiliation)
    lines.append("")

    lines.append("## Financovani")
    lines.append(
        f"*(zdroj: sekce Funding, `\\subsection*{{Funding}}` v {main_tex_path.name})*"
    )
    lines.append("")
    lines.append(f"- Poskytovatel (funder): {funder}")
    lines.append(f"- Cislo grantu (grant number): {grant_number}")
    lines.append("- Plne zneni (pokud formular chce jedno textove pole):")
    lines.append("")
    lines.append(funding_statement)
    lines.append("")

    lines.append("## Prohlaseni")
    lines.append(f"*(zdroj: sekce Statements and Declarations v {main_tex_path.name})*")
    lines.append("")
    lines.append("### Competing Interests")
    lines.append(competing_interests)
    lines.append("")
    lines.append("### Data availability")
    lines.append(data_availability)
    lines.append("")
    lines.append("### Code availability")
    lines.append(code_availability)
    lines.append("")
    lines.append("### Ethics approval")
    lines.append(ethics_approval)
    lines.append("")
    lines.append("### Consent to participate")
    lines.append(consent_participate)
    lines.append("")
    lines.append("### Consent for publication")
    lines.append(consent_publication)
    lines.append("")
    lines.append("### Author contributions")
    lines.append(author_contributions)
    lines.append("")

    lines.append(
        "*(Prohlaseni o pouziti AI: viz `dami_submission/03_PROHLASENI_O_AI.md` "
        "- ten text je hotovy a negeneruje se odsud.)*"
    )
    lines.append("")

    return "\n".join(lines), warnings


# Fields every reviewer entry must carry; a missing one is an error rather
# than a silently incomplete cover letter (the journal requires an
# institutional e-mail or another way to verify the person's identity).
_REVIEWER_FIELDS = ("name", "affiliation", "email", "orcid", "reason")


def load_reviewers(path: Path) -> list[dict[str, str]]:
    """Reads the suggested reviewers from reviewers.yaml. The file is the
    single machine-readable source; 04_NAVRZENI_RECENZENTI.md documents where
    each entry was verified."""
    if not path.is_file():
        raise SubmissionGuideError(
            f"Suggested reviewers file not found: {path}. It is the source of "
            "the reviewer paragraph in the cover letter."
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = data.get("reviewers")
    if not entries:
        raise SubmissionGuideError(f"No 'reviewers' entries in {path}.")
    for i, entry in enumerate(entries, start=1):
        missing = [f for f in _REVIEWER_FIELDS if not str(entry.get(f, "")).strip()]
        if missing:
            raise SubmissionGuideError(
                f"Reviewer #{i} in {path} is missing: {', '.join(missing)}."
            )
    return entries


def build_cover_letter_markdown(
    *,
    main_tex_path: Path,
    numbers_tex_path: Path,
    zenodo_doi: str,
    reviewers_path: Path,
) -> tuple[str, list[str]]:
    """Builds the content of 02_COVER_LETTER.md. The connecting prose is a
    static template (author's request: written by the tool's author, not
    extracted verbatim); every factual detail inserted into it (title,
    author, affiliation, dataset/method counts, repository URL, DOI) is
    pulled from the manuscript or from the mandatory --zenodo-doi argument."""
    tex = main_tex_path.read_text(encoding="utf-8")
    macros = load_number_macros(numbers_tex_path)
    reviewers = load_reviewers(reviewers_path)
    warnings: list[str] = []

    title_raw = _find_and_extract(tex, "title")
    title = _plain(title_raw, macros, "Cover letter / title", warnings)

    authors = extract_authors(tex)
    corresponding = next((a for a in authors if a.corresponding), authors[0])

    affil_parts = extract_affiliation_parts(tex)
    affiliation_short = _plain(
        f'{affil_parts["orgname"]}, {affil_parts["orgdiv"]}',
        macros, "Cover letter / affiliation", warnings,
    )

    repo_url = extract_repository_url(tex)
    doi_url = f"https://doi.org/{zenodo_doi}"

    n_datasets = macros.get("numExpOneDatasets")
    n_methods = macros.get("numExpOneMethods")
    n_graphs = macros.get("numExpThreeGraphs")
    n_temporal = macros.get("numExpFourDatasets")
    missing_counts = [
        name
        for name, value in (
            ("numExpOneDatasets", n_datasets),
            ("numExpOneMethods", n_methods),
            ("numExpThreeGraphs", n_graphs),
            ("numExpFourDatasets", n_temporal),
        )
        if value is None
    ]
    if missing_counts:
        raise SubmissionGuideError(
            f"numbers.tex is missing macro(s) {missing_counts} needed for "
            "the cover letter - regenerate it with src/run_main.bat/.sh."
        )

    timestamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: list[str] = []
    lines.append("# Cover letter (DAMI, Springer Nature SNAPP)")
    lines.append("")
    lines.append(
        f"> AUTOGENEROVANO skriptem `src/tools/make_submission_guide.py` "
        f"dne {timestamp}. Text nize je hotovy prosty text ke zkopirovani "
        "do pole cover letteru; NEUPRAVUJ RUCNE - uprav sablonu ve skriptu "
        "a spust ho znovu."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("Dear Editor,")
    lines.append("")
    lines.append(
        f'Please consider our manuscript "{title}" for publication in Data '
        "Mining and Knowledge Discovery as a Regular Paper."
    )
    lines.append("")
    lines.append(
        "Sammon's projection and related multidimensional scaling methods "
        "weight pairwise stress by D_ij^(-alpha); the exponent alpha has so "
        "far been chosen by hand, by convention, or by a repeated tuning "
        "search. We show that as the relative spread of distances vanishes, "
        "the weighted and unweighted stress coincide at the optimum, so "
        "alpha becomes statistically unidentifiable and alpha=0 is then a "
        "safe default. From this identifiability criterion we derive a rule "
        "that sets alpha from a single input statistic without any tuning "
        f"run, and we verify the theory, the rule, a GPU solver, and "
        f"extensions to graphs and temporal networks across {n_datasets} "
        f"datasets, {n_methods} dimensionality-reduction methods, {n_graphs} "
        f"graphs and {n_temporal} temporal networks."
    )
    lines.append("")
    lines.append(
        "We believe the manuscript fits Data Mining and Knowledge Discovery "
        "because it addresses a widely used but so far unexamined design "
        "choice in metric dimensionality reduction and network "
        "visualization, backed by a formal identifiability result, a large "
        "empirical study with pre-registered hold-out validation, and a "
        "scalable GPU solver."
    )
    lines.append("")
    lines.append(
        "The family of weighted stresses w_ij = D_ij^(-alpha) is itself not "
        "new. What is new is a criterion for WHEN tuning this exponent is "
        "worthwhile at all, and a rule that sets it without any tuning run "
        "in the regime where tuning does not help."
    )
    lines.append("")
    lines.append(
        "All data and code needed to reproduce every number, table and "
        f"figure in the manuscript are publicly available: the source code "
        f"is on GitHub at {repo_url} (MIT license), and the results archive "
        f"behind this submission is deposited on Zenodo under DOI "
        f"{zenodo_doi} ({doi_url})."
    )
    lines.append("")
    lines.append(
        "This manuscript reports original work. It is not under review, in "
        "press, or published elsewhere, in whole or in part, and it is not "
        "under simultaneous consideration at any other journal."
    )
    lines.append("")
    lines.append(
        "For completeness we disclose that the manuscript builds on two "
        "earlier conference papers by the authors, both cited in the text: "
        "Network Layout Visualization Based on Sammon's Projection "
        "(INCoS 2013, doi:10.1109/INCoS.2013.43) and Visualization of "
        "Social Network Dynamics using Sammon's Projection (CASoN 2013, "
        "doi:10.1109/CASoN.2013.6622600). Those papers applied Sammon's "
        "projection to network layout and to network dynamics. The present "
        "manuscript shares that starting point but is otherwise new work: "
        "the identifiability result for the stress exponent, the rule that "
        "sets the exponent from a single input statistic, the GPU solver "
        "and the empirical study are not part of either conference paper."
    )
    lines.append("")
    lines.append(
        "The use of AI-assisted tools during the preparation of this "
        "manuscript is declared in the manuscript itself, in the Methods "
        "section and in full under Statements and Declarations."
    )
    lines.append("")
    lines.append(
        "We suggest the following reviewers. None of them has co-authored "
        "with either of us, none is affiliated with our institution, and "
        "no two of them have co-authored with each other:"
    )
    lines.append("")
    for reviewer in reviewers:
        lines.append(
            f"- {reviewer['name']}, {reviewer['affiliation']}; "
            f"{reviewer['email']}; ORCID {reviewer['orcid']} - "
            f"{reviewer['reason']}."
        )
    lines.append("")
    lines.append("On behalf of both authors,")
    lines.append("")
    lines.append(corresponding.full_name)
    lines.append(affiliation_short)
    lines.append(corresponding.email)
    lines.append("")

    return "\n".join(lines), warnings


# --------------------------------------------------------------------------
# Cover letter as PDF (the submission form takes plain text, but a typeset
# copy is what gets attached or kept on file)
# --------------------------------------------------------------------------

# The ten characters that must not reach pdflatex unescaped. The backslash is
# handled first via a placeholder, otherwise it would escape the escapes.
_TEX_SPECIALS = (
    ("&", "\\&"),
    ("%", "\\%"),
    ("$", "\\$"),
    ("#", "\\#"),
    ("_", "\\_"),
    ("{", "\\{"),
    ("}", "\\}"),
    ("~", "\\textasciitilde{}"),
    ("^", "\\textasciicircum{}"),
)


def _tex_escape(text: str) -> str:
    """Escapes LaTeX special characters. The letter carries D_ij^(-alpha),
    e-mail addresses and URLs, so this is not optional. The backslash goes
    through a placeholder so that the escapes are not escaped again."""
    placeholder = "@BACKSLASH@"
    text = text.replace(chr(92), placeholder)
    for char, replacement in _TEX_SPECIALS:
        text = text.replace(char, replacement)
    return text.replace(placeholder, chr(92) + "textbackslash{}")


_SIGNOFF_PREFIX = "On behalf of"



def build_cover_letter_tex(markdown: str) -> str:
    """Turns the generated cover-letter markdown into a standalone LaTeX
    letter. The markdown heading and the provenance note above the first
    '---' separator are dropped: they address the author, not the editor."""
    separator = chr(10) + "---" + chr(10)
    _, found, body = markdown.partition(separator)
    if not found:
        raise SubmissionGuideError(
            "Cover letter markdown has no '---' separator; cannot separate "
            "the provenance note from the letter itself."
        )

    bs = chr(92)
    out: list[str] = []
    bullets: list[str] = []
    in_signature = False

    def flush_bullets() -> None:
        if not bullets:
            return
        out.append(bs + "begin{itemize}[leftmargin=1.2em,itemsep=2pt,topsep=4pt]")
        for item in bullets:
            out.append(bs + "item " + item)
        out.append(bs + "end{itemize}")
        bullets.clear()

    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            flush_bullets()
            out.append("")
            continue
        if line.startswith("- "):
            bullets.append(_tex_escape(line[2:]))
            continue
        flush_bullets()
        escaped = _tex_escape(line)
        if line.startswith(_SIGNOFF_PREFIX):
            in_signature = True
        # Name, affiliation and e-mail must break by line, not flow.
        out.append(escaped + (' ' + bs + bs if in_signature else ''))
    flush_bullets()

    preamble = [
        bs + "documentclass[11pt,a4paper]{article}",
        bs + "usepackage[T1]{fontenc}",
        bs + "usepackage[utf8]{inputenc}",
        bs + "usepackage[margin=25mm]{geometry}",
        bs + "usepackage{enumitem}",
        bs + "usepackage{parskip}",
        bs + "pagestyle{empty}",
        bs + "begin{document}",
    ]
    tail = [bs + "end{document}", ""]
    return chr(10).join(preamble + [""] + out + tail)


def compile_cover_letter_pdf(tex_path: Path) -> Path:
    """Compiles the letter with pdflatex (two passes are unnecessary: no
    references, no bibliography). Fails loud when pdflatex is missing or the
    compile does not produce a PDF."""
    if shutil.which("pdflatex") is None:
        raise SubmissionGuideError(
            "pdflatex not found on PATH; cannot build the cover-letter PDF. "
            "Rerun with --skip-pdf to write only the markdown."
        )
    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=tex_path.parent,
        capture_output=True,
        text=True,
    )
    pdf_path = tex_path.with_suffix(".pdf")
    if result.returncode != 0 or not pdf_path.is_file():
        log_path = tex_path.with_suffix(".log")
        raise SubmissionGuideError(
            f"pdflatex failed for {tex_path.name} (exit {result.returncode}); "
            f"see {log_path}."
        )
    for suffix in (".aux", ".log", ".out"):
        by_product = tex_path.with_suffix(suffix)
        if by_product.is_file():
            by_product.unlink()
    return pdf_path


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "zenodo_doi",
        help="DOI of the Zenodo results archive, e.g. 10.5281/zenodo.1234567 "
        "(mandatory - the cover letter must not ship with a placeholder).",
    )
    parser.add_argument(
        "--skip-pdf",
        action="store_true",
        help="Write only the markdown files; do not typeset the cover-letter PDF (use when pdflatex is not available).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Where to write the two .md files (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--main-tex", type=Path, default=DEFAULT_MAIN_TEX,
        help=f"Manuscript source (default: {DEFAULT_MAIN_TEX}).",
    )
    parser.add_argument(
        "--numbers-tex", type=Path, default=DEFAULT_NUMBERS_TEX,
        help=f"Generated number macros (default: {DEFAULT_NUMBERS_TEX}).",
    )
    args = parser.parse_args(argv)

    if not _DOI_RE.match(args.zenodo_doi):
        print(
            "[make_submission_guide] FAILED: "
            f"'{args.zenodo_doi}' does not look like a DOI. Expected the "
            "form 10.<registrant>/<suffix>, e.g. 10.5281/zenodo.1234567 "
            "(no 'https://doi.org/' prefix).",
            file=sys.stderr,
        )
        return 1

    try:
        form_texts, form_warnings = build_form_texts_markdown(
            main_tex_path=args.main_tex, numbers_tex_path=args.numbers_tex
        )
        cover_letter, letter_warnings = build_cover_letter_markdown(
            main_tex_path=args.main_tex,
            numbers_tex_path=args.numbers_tex,
            zenodo_doi=args.zenodo_doi,
            reviewers_path=args.output_dir / "reviewers.yaml",
        )
    except SubmissionGuideError as exc:
        print(f"[make_submission_guide] FAILED: {exc}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    form_path = args.output_dir / "01_TEXTY_DO_FORMULARE.md"
    letter_path = args.output_dir / "02_COVER_LETTER.md"
    form_path.write_text(form_texts, encoding="utf-8")
    letter_path.write_text(cover_letter, encoding="utf-8")

    print(f"[make_submission_guide] Wrote {form_path}")
    print(f"[make_submission_guide] Wrote {letter_path}")

    if not args.skip_pdf:
        letter_tex_path = args.output_dir / "02_COVER_LETTER.tex"
        letter_tex_path.write_text(
            build_cover_letter_tex(cover_letter), encoding="utf-8"
        )
        try:
            letter_pdf_path = compile_cover_letter_pdf(letter_tex_path)
        except SubmissionGuideError as exc:
            print(f"[make_submission_guide] FAILED: {exc}", file=sys.stderr)
            return 1
        print(f"[make_submission_guide] Wrote {letter_tex_path}")
        print(f"[make_submission_guide] Wrote {letter_pdf_path}")

    all_warnings = form_warnings + letter_warnings
    if all_warnings:
        print(
            f"[make_submission_guide] {len(all_warnings)} item(s) need manual "
            "review before submission (see <<< ZKONTROLUJ >>> markers):",
            file=sys.stderr,
        )
        for w in all_warnings:
            print(f"  - {w}", file=sys.stderr)
    else:
        print("[make_submission_guide] No <<< ZKONTROLUJ >>> markers - clean generation.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
