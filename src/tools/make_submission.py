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
Builds a FLAT (single-directory, no subfolders) journal-submission package
from the split `clanek_en/` tree (`clanek_en/dami/`, `clanek_en/sections/`,
`clanek_en/supplement/`, which in turn reaches into `../clanek/img/`,
`../clanek/generated/numbers.tex`, `../clanek/references.bib` and
`../../results/tables/`). Many submission systems (Springer's among them)
require every source file for one manuscript in a single directory with no
relative paths, which the split tree violates.

Every package is built for ONE named target journal and lands under a
directory named after it, so that a later submission to a different journal
cannot be confused with this one:

  submission/dami/manuscript_en/   main article, Springer Nature sn-jnl class
  submission/dami/supplement_en/   supplement, elsarticle class
  submission/dami/README.txt       what goes where in the submission form

The journal is chosen with --journal (see JOURNALS below); the directory
also carries a JOURNAL.txt stamp, and the build refuses to write into a
directory stamped for a different journal.

Two packages are produced, each self-contained:

  manuscript_en  The main article (Springer Nature sn-jnl class), from
                 clanek_en/dami/main_dami.tex.
  supplement_en  The supplement (elsarticle class, published separately by
                 Springer), from clanek_en/supplement/supplement.tex. Its
                 \\externaldocument cross-references into the main text are
                 rewired to the FRESH, just-verified main_dami.aux produced
                 while building manuscript_en in the same run (never to the stale
                 checked-in .aux, and never to the separate preprint
                 clanek_en/main.tex) - see _wire_external_aux().

For each package this script:
  1. Walks the \\input/\\InputIfFileExists tree from the root .tex file and
     copies every reachable section file flat, renaming on collision.
  2. Rewrites every \\input, \\InputIfFileExists, \\includegraphics,
     \\graphicspath, \\rawtableinput(wide/long), \\bibliography,
     \\externaldocument, \\documentclass and \\bibliographystyle path/name
     argument to the flat filename actually copied.
  3. Copies only the figures and generated tables that are actually
     referenced (found by scanning the .tex sources), never the whole
     clanek/img/ or results/tables/ directory.
  4. Copies the document class and bibliography style files (project-local
     first, e.g. clanek_en/dami/sn-jnl.cls, else resolved via `kpsewhich`
     for standard classes such as elsarticle).
  5. Writes an English README.txt with the file list and the compile
     command.
  6. MANDATORY verification: compiles the flat package in the target
     directory itself (pdflatex + bibtex + pdflatex x2, mirroring
     clanek_en/dami/build_dami.bat / clanek_en/supplement/compile.bat) and
     compares the page count and the presence of "undefined"/"Overfull"/
     "invalid in math mode"/"Author undefined" markers in the log against
     the existing reference PDF (clanek_en/dami/main_dami.pdf resp.
     clanek_en/supplement/supplement.pdf). Any mismatch is a hard failure -
     a package that does not verify is worse than no package.

Run (from anywhere, builds both packages under submission/<journal>/):
    venv\\python.exe -m src.tools.make_submission
or: src\\run_make_submission.bat   /   src/run_make_submission.sh

CLI:
    --journal KEY        target journal, one of JOURNALS (default: dami).
                          Decides the output directory name and the contents
                          of README.txt / JOURNAL.txt.
    --output-root PATH   base directory holding the per-journal directories
                          (default: <repo>/submission)
    --variant {dami,supplement,both}  which package(s) to produce (default:
                          both; "supplement" still builds+verifies manuscript_en
                          first, because supplement_en's cross-references
                          depend on its freshly verified main_dami.aux)
    --skip-verify         skip the mandatory LaTeX compile-and-check step.
                          Debugging aid ONLY (fast iteration on path
                          rewriting) - never pass this in the .bat/.sh
                          wrappers; a package built this way is unverified
                          and must not be submitted.
"""
from __future__ import annotations

import argparse
import dataclasses
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLANEK_EN = PROJECT_ROOT / "clanek_en"

# Log markers that must never appear in a submission-ready compile (fail-loud
# checklist from the task). Checked case-sensitively, matching how they are
# emitted by pdflatex/hyperref/natbib.
FORBIDDEN_LOG_MARKERS = (
    "undefined",
    "Overfull",
    "invalid in math mode",
    "Author undefined",
)

# The one cross-document reference relationship in this project: the
# supplement's \externaldocument always points at the main (DAMI) article,
# regardless of which package is being built. See _wire_external_aux().
EXTERNAL_DOCUMENT_DEPENDENCY_KEY = "dami"


class SubmissionBuildError(RuntimeError):
    """Fail-loud error: a reference could not be resolved, a verification
    check failed, or a name collision could not be disambiguated. Never
    caught silently - the caller is expected to abort the whole build."""


@dataclasses.dataclass(frozen=True)
class DocumentSpec:
    """Where one flat package's root document lives before flattening.

    extra_style_files: files a class needs that never appear as a literal
    \\documentclass/\\bibliographystyle argument in the .tex source, so a
    text scan cannot discover them. Concretely: sn-jnl.cls picks its BibTeX
    style from the class OPTION "sn-basic" in
    `\\documentclass[pdflatex,sn-basic]{sn-jnl}` (there is no explicit
    `\\bibliographystyle{sn-basic}` call - see the comment on
    \\bibliography in main_dami.tex and clanek_en/dami/build_dami.bat, which
    for the same reason points BSTINPUTS at clanek_en/dami/ explicitly).
    """

    key: str
    root_tex: Path
    master_dir: Path
    cwd_dir: Path
    reference_pdf: Path
    output_subdir_name: str
    extra_style_files: tuple[str, ...] = ()


DAMI_SPEC = DocumentSpec(
    key="dami",
    root_tex=CLANEK_EN / "dami" / "main_dami.tex",
    master_dir=CLANEK_EN / "dami",
    cwd_dir=CLANEK_EN,
    reference_pdf=CLANEK_EN / "dami" / "main_dami.pdf",
    output_subdir_name="manuscript_en",
    extra_style_files=("sn-basic.bst",),
)
SUPPLEMENT_SPEC = DocumentSpec(
    key="supplement",
    root_tex=CLANEK_EN / "supplement" / "supplement.tex",
    master_dir=CLANEK_EN / "supplement",
    cwd_dir=CLANEK_EN / "supplement",
    reference_pdf=CLANEK_EN / "supplement" / "supplement.pdf",
    output_subdir_name="supplement_en",
)
SPECS_BY_KEY = {DAMI_SPEC.key: DAMI_SPEC, SUPPLEMENT_SPEC.key: SUPPLEMENT_SPEC}


@dataclasses.dataclass(frozen=True)
class JournalProfile:
    """Identity of the journal a package is built for.

    Exists so that a package prepared for one journal can never be mistaken
    for a package prepared for another: `key` names the output directory,
    and the same key is written into a JOURNAL.txt stamp that the build
    checks before overwriting anything.
    """

    key: str
    name: str
    publisher: str
    submission_system: str
    article_type: str
    manuscript_class: str
    upload_slots: tuple[tuple[str, str], ...]


JOURNALS: dict[str, JournalProfile] = {
    "dami": JournalProfile(
        key="dami",
        name="Data Mining and Knowledge Discovery",
        publisher="Springer",
        submission_system="https://submission.springernature.com/new-submission/10618/3",
        article_type="Regular Paper",
        manuscript_class="sn-jnl (option sn-basic)",
        upload_slots=(
            ("manuscript_en.zip", "Manuscript (LaTeX sources; Snapp compiles them)"),
            ("supplement_en.zip", "Online Resource 1 (supplementary material)"),
        ),
    ),
}

JOURNAL_STAMP_NAME = "JOURNAL.txt"

# --------------------------------------------------------------------------
# LaTeX command patterns with a resolvable path/name argument. Each entry is
# (regex, group index of the path argument). The path argument is captured
# without surrounding braces so it can be substituted back in verbatim.
# --------------------------------------------------------------------------
_INPUT_RE = re.compile(r"\\input\{([^{}]+)\}")
_INPUTIFEXISTS_RE = re.compile(
    r"\\InputIfFileExists\{([^{}]+)\}(\{[^{}]*\}\{[^{}]*\})"
)
_INCLUDEGRAPHICS_RE = re.compile(
    r"(\\includegraphics(?:\[[^\]]*\])?\{)([^{}]+)(\})"
)
# "rawtableinput" must not swallow "rawtableinputwide"/"rawtableinputlong" -
# negative lookahead (three sibling macros defined in supplement.tex: plain,
# wide (landscape, resizebox-fit), long (own longtable environment, e.g. for
# the "Float too large" fix in list_datasets.py/report_tables.py tables)).
_RAWTABLEINPUTWIDE_RE = re.compile(r"(\\rawtableinputwide\{)([^{}]+)(\})")
_RAWTABLEINPUTLONG_RE = re.compile(r"(\\rawtableinputlong\{)([^{}]+)(\})")
_RAWTABLEINPUT_RE = re.compile(r"(\\rawtableinput(?!wide|long)\*?\{)([^{}]+)(\})")
_BIBLIOGRAPHY_RE = re.compile(r"(\\bibliography\{)([^{}]+)(\})")
_EXTERNALDOCUMENT_RE = re.compile(
    r"(\\externaldocument(?:\[[^\]]*\])?\{)([^{}]+)(\})"
)
_GRAPHICSPATH_RE = re.compile(r"\\graphicspath\{\{[^{}]*\}\}")
_DOCUMENTCLASS_RE = re.compile(r"(\\documentclass(?:\[[^\]]*\])?\{)([^{}]+)(\})")
_BIBLIOGRAPHYSTYLE_RE = re.compile(r"(\\bibliographystyle\{)([^{}]+)(\})")


def _is_macro_placeholder(ref: str) -> bool:
    """True for a LaTeX macro-parameter placeholder such as '#1' - occurs in
    the \\rawtableinput/\\rawtableinputwide macro *definitions* themselves
    (`\\input{#1}`), never in an actual file reference. Must never be treated
    as a path to resolve."""
    return "#" in ref


class FlatNameAssigner:
    """Assigns a unique, collision-free flat basename to every distinct
    source path pulled into one package. Two different source files that
    happen to share a basename get the parent directory name prefixed; a
    third-level collision is a hard error (fail loud, not a silent
    overwrite)."""

    def __init__(self) -> None:
        self._name_to_source: dict[str, Path] = {}
        self._source_to_name: dict[Path, str] = {}

    def assign(self, source: Path, *, preferred_name: str | None = None) -> str:
        source = source.resolve()
        if source in self._source_to_name:
            return self._source_to_name[source]
        base = preferred_name or source.name
        candidates = [base, f"{source.parent.name}_{base}"]
        for candidate in candidates:
            existing = self._name_to_source.get(candidate)
            if existing is None or existing == source:
                self._name_to_source[candidate] = source
                self._source_to_name[source] = candidate
                return candidate
        raise SubmissionBuildError(
            f"Filename collision that could not be disambiguated: "
            f"'{source}' and '{self._name_to_source[candidates[-1]]}' both "
            f"map to '{candidates[-1]}' in the flat package."
        )

    def name_of(self, source: Path) -> str | None:
        return self._source_to_name.get(source.resolve())

    def items(self) -> "list[tuple[Path, str]]":
        return list(self._source_to_name.items())


def _resolve_path(
    ref: str,
    candidate_dirs: list[Path],
    *,
    extra_exts: tuple[str, ...] = (),
) -> Path:
    """Resolves a relative LaTeX path argument against a prioritized list of
    base directories (mirrors kpathsea trying the current working directory,
    then the master file's directory, in that order - see the "DULEZITE k
    cestam" comment in clanek_en/dami/main_dami.tex). Fails loud, listing
    every path tried, rather than silently skipping a missing input."""
    names = [ref] + [ref + ext for ext in extra_exts if not ref.endswith(ext)]
    tried: list[Path] = []
    seen_dirs: list[Path] = []
    for base in candidate_dirs:
        base = base.resolve()
        if base in seen_dirs:
            continue
        seen_dirs.append(base)
        for name in names:
            candidate = (base / name).resolve()
            tried.append(candidate)
            if candidate.is_file():
                return candidate
    tried_str = "\n  ".join(str(t) for t in tried)
    raise SubmissionBuildError(
        f"Could not resolve reference '{ref}'. Tried:\n  {tried_str}"
    )


def _kpsewhich(name: str) -> Path | None:
    """Resolves a standard, installation-provided LaTeX file (class/style)
    via `kpsewhich`. Returns None if not found or if kpsewhich itself is not
    on PATH; never raises for a plain "not found" (the caller decides
    whether that is fatal)."""
    try:
        result = subprocess.run(
            ["kpsewhich", name],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    for line in result.stdout.splitlines():
        line = line.strip()
        if line and Path(line).is_file():
            return Path(line)
    return None


def _resolve_style_file(
    name: str, ext: str, candidate_dirs: list[Path]
) -> Path:
    """Resolves a \\documentclass/\\bibliographystyle NAME (no extension) to
    a file: project-local copy first (e.g. clanek_en/dami/sn-jnl.cls, a
    Springer template that is not part of a generic TeX install), else the
    installation-provided file via kpsewhich (e.g. elsarticle.cls)."""
    try:
        return _resolve_path(name, candidate_dirs, extra_exts=(ext,))
    except SubmissionBuildError:
        pass
    found = _kpsewhich(name if name.endswith(ext) else name + ext)
    if found is not None:
        return found
    raise SubmissionBuildError(
        f"Could not resolve style file '{name}{ext}': not found next to the "
        f"document ({[str(d) for d in candidate_dirs]}) and `kpsewhich "
        f"{name}{ext}` returned nothing. Install the missing LaTeX package "
        f"or check the class/bibliographystyle name."
    )


@dataclasses.dataclass
class BuildResult:
    flat_dir: Path
    root_tex_flat_name: str
    manifest: list[tuple[str, Path]]  # (flat name, original source path)
    needs_external_aux_from: str | None  # key into SPECS_BY_KEY, or None


def _discover_tex_tree(
    root_tex: Path, search_dirs: list[Path]
) -> "dict[Path, str]":
    """Breadth-first walk of \\input/\\InputIfFileExists, returning an
    insertion-ordered {absolute source path: raw text} map with root_tex
    first. Fails loud on any missing file."""
    root_tex = root_tex.resolve()
    if not root_tex.is_file():
        raise SubmissionBuildError(f"Root document not found: {root_tex}")
    registry: dict[Path, str] = {}
    queue: list[Path] = [root_tex]
    while queue:
        path = queue.pop(0)
        if path in registry:
            continue
        text = path.read_text(encoding="utf-8")
        registry[path] = text
        refs = [m.group(1) for m in _INPUT_RE.finditer(text)]
        refs += [m.group(1) for m in _INPUTIFEXISTS_RE.finditer(text)]
        for ref in refs:
            if _is_macro_placeholder(ref):
                continue
            resolved = _resolve_path(
                ref, [path.parent] + search_dirs, extra_exts=(".tex",)
            )
            queue.append(resolved)
    return registry


def _rewrite_path_command(
    text: str,
    pattern: re.Pattern[str],
    ref_group: int,
    resolver,
    assigner: FlatNameAssigner,
    rebuild,
    *,
    strip_exts: tuple[str, ...] = (),
) -> str:
    """Generic path-argument rewriter: for every regex match, resolves the
    reference captured in group `ref_group` to a real file, assigns it a
    flat name (dropping any extension listed in strip_exts, matching how
    \\bibliography/\\bibliographystyle/\\externaldocument are conventionally
    written without one), and hands the match plus the flat name to
    `rebuild(match, flat_name)` to reconstruct the FULL replacement text
    (so the surrounding command syntax, e.g. `\\input{...}`, is never lost -
    the caller, not this function, owns that syntax)."""

    def _sub(match: re.Match[str]) -> str:
        ref = match.group(ref_group)
        if _is_macro_placeholder(ref):
            return match.group(0)
        source = resolver(ref)
        flat_name = assigner.assign(source)
        for ext in strip_exts:
            if flat_name.endswith(ext):
                flat_name = flat_name[: -len(ext)]
                break
        return rebuild(match, flat_name)

    return pattern.sub(_sub, text)


def _rebuild_wrapped(match: re.Match[str], flat_name: str) -> str:
    """rebuild() for 3-group patterns of the shape (prefix)(ref)(suffix)."""
    return f"{match.group(1)}{flat_name}{match.group(3)}"


def _parse_graphicspath_dir(text: str, master_dir: Path) -> Path | None:
    """Extracts the single directory from \\graphicspath{{DIR}} (as used
    throughout clanek_en) and resolves it against master_dir, so that bare
    \\includegraphics{name.pdf} calls (no directory component) can still be
    resolved."""
    match = re.search(r"\\graphicspath\{\{([^{}]+)\}\}", text)
    if match is None:
        return None
    return (master_dir / match.group(1)).resolve()


def build_flat_package(spec: DocumentSpec, output_root: Path) -> BuildResult:
    """Flattens one document (dami or supplement) into
    output_root/<spec.output_subdir_name>/, rewriting every path reference
    and copying every file it needs. Raises SubmissionBuildError on any
    unresolved reference or name collision."""
    flat_dir = output_root / spec.output_subdir_name
    if flat_dir.exists():
        shutil.rmtree(flat_dir)
    flat_dir.mkdir(parents=True)

    search_dirs = [spec.master_dir, spec.cwd_dir]
    tex_registry = _discover_tex_tree(spec.root_tex, search_dirs)

    assigner = FlatNameAssigner()
    # Root document keeps its original name (needed so the compile command
    # and cross-package \externaldocument reference stay predictable).
    root_flat_name = assigner.assign(
        spec.root_tex, preferred_name=spec.root_tex.name
    )
    for source_path in tex_registry:
        assigner.assign(source_path)

    graphicspath_dir = _parse_graphicspath_dir(
        tex_registry[spec.root_tex], spec.master_dir
    )
    image_search_dirs = [spec.master_dir, spec.cwd_dir]
    if graphicspath_dir is not None:
        image_search_dirs = [graphicspath_dir] + image_search_dirs

    needs_external_aux_from: str | None = None
    rewritten: dict[Path, str] = {}
    for source_path, text in tex_registry.items():
        file_search_dirs = [source_path.parent] + search_dirs

        text = _rewrite_path_command(
            text,
            _INPUT_RE,
            1,
            lambda ref, d=file_search_dirs: _resolve_path(
                ref, d, extra_exts=(".tex",)
            ),
            assigner,
            lambda match, flat_name: f"\\input{{{flat_name}}}",
        )
        text = _rewrite_path_command(
            text,
            _INPUTIFEXISTS_RE,
            1,
            lambda ref, d=file_search_dirs: _resolve_path(ref, d, extra_exts=(".tex",)),
            assigner,
            lambda match, flat_name: f"\\InputIfFileExists{{{flat_name}}}{match.group(2)}",
        )
        text = _rewrite_path_command(
            text,
            _INCLUDEGRAPHICS_RE,
            2,
            lambda ref, d=(file_search_dirs + image_search_dirs): _resolve_path(ref, d),
            assigner,
            _rebuild_wrapped,
        )
        text = _rewrite_path_command(
            text,
            _RAWTABLEINPUTWIDE_RE,
            2,
            lambda ref, d=file_search_dirs: _resolve_path(ref, d),
            assigner,
            _rebuild_wrapped,
        )
        text = _rewrite_path_command(
            text,
            _RAWTABLEINPUTLONG_RE,
            2,
            lambda ref, d=file_search_dirs: _resolve_path(ref, d),
            assigner,
            _rebuild_wrapped,
        )
        text = _rewrite_path_command(
            text,
            _RAWTABLEINPUT_RE,
            2,
            lambda ref, d=file_search_dirs: _resolve_path(ref, d),
            assigner,
            _rebuild_wrapped,
        )
        text = _rewrite_path_command(
            text,
            _BIBLIOGRAPHY_RE,
            2,
            lambda ref, d=file_search_dirs: _resolve_path(ref, d, extra_exts=(".bib",)),
            assigner,
            _rebuild_wrapped,
            strip_exts=(".bib",),
        )
        text = _rewrite_path_command(
            text,
            _DOCUMENTCLASS_RE,
            2,
            lambda ref, d=[spec.master_dir]: _resolve_style_file(ref, ".cls", d),
            assigner,
            _rebuild_wrapped,
            strip_exts=(".cls",),
        )
        text = _rewrite_path_command(
            text,
            _BIBLIOGRAPHYSTYLE_RE,
            2,
            lambda ref, d=[spec.master_dir]: _resolve_style_file(ref, ".bst", d),
            assigner,
            _rebuild_wrapped,
            strip_exts=(".bst",),
        )

        if _EXTERNALDOCUMENT_RE.search(text) is not None:
            needs_external_aux_from = _wire_external_aux_target()
            dep_spec = SPECS_BY_KEY[needs_external_aux_from]
            target_stem = dep_spec.root_tex.stem
            text = _EXTERNALDOCUMENT_RE.sub(
                lambda match, stem=target_stem: f"{match.group(1)}{stem}{match.group(3)}",
                text,
            )

        # \graphicspath is now meaningless (everything is flat) - neutralize
        # it instead of leaving a dangling relative directory.
        text = _GRAPHICSPATH_RE.sub(r"\\graphicspath{{./}}", text)

        rewritten[source_path] = text

    # extra_style_files (e.g. sn-basic.bst) are needed by the class but never
    # appear as a literal argument in the .tex source - see DocumentSpec.
    for style_file in spec.extra_style_files:
        stem, ext = Path(style_file).stem, Path(style_file).suffix
        resolved = _resolve_style_file(stem, ext, [spec.master_dir])
        assigner.assign(resolved, preferred_name=style_file)

    manifest: list[tuple[str, Path]] = []
    for source_path, text in rewritten.items():
        flat_name = assigner.name_of(source_path)
        assert flat_name is not None
        (flat_dir / flat_name).write_text(text, encoding="utf-8")
        manifest.append((flat_name, source_path))

    # Copy every non-tex asset that got a flat name assigned above (images,
    # tables, .bib, .cls, .bst) but has not been written yet.
    tex_sources = set(tex_registry)
    for source_path, flat_name in assigner.items():
        if source_path in tex_sources:
            continue
        dest = flat_dir / flat_name
        shutil.copy2(source_path, dest)
        manifest.append((flat_name, source_path))

    manifest.sort(key=lambda item: item[0])
    return BuildResult(
        flat_dir=flat_dir,
        root_tex_flat_name=root_flat_name,
        manifest=manifest,
        needs_external_aux_from=needs_external_aux_from,
    )


def _wire_external_aux_target() -> str:
    """The project has exactly one cross-document reference relationship:
    the supplement's \\externaldocument always targets the main (DAMI)
    article. Encoded as a function (not a bare constant) so a future second
    relationship fails loud here instead of silently reusing this one."""
    return EXTERNAL_DOCUMENT_DEPENDENCY_KEY


# --------------------------------------------------------------------------
# Verification: compile the flat package and compare it against the
# existing reference PDF (mandatory - see module docstring point 6).
# --------------------------------------------------------------------------


@dataclasses.dataclass
class VerificationResult:
    label: str
    pages: int
    reference_pages: int
    log_path: Path


def _is_miktex() -> bool:
    try:
        result = subprocess.run(
            ["pdflatex", "--version"], capture_output=True, text=True, timeout=20
        )
    except FileNotFoundError as exc:
        raise SubmissionBuildError(
            "pdflatex is not on PATH - cannot run the mandatory verification "
            "compile. Install MiKTeX/TeX Live or pass --skip-verify only for "
            "debugging (never for an actual submission build)."
        ) from exc
    return "MiKTeX" in result.stdout


def _run_pdflatex(flat_dir: Path, tex_name: str, *, extra_args: list[str]) -> Path:
    args = ["pdflatex", "-interaction=nonstopmode"] + extra_args + [tex_name]
    subprocess.run(args, cwd=flat_dir, capture_output=True, text=True, check=False)
    log_path = flat_dir / (Path(tex_name).stem + ".log")
    if not log_path.is_file() or "Output written on" not in log_path.read_text(
        encoding="utf-8", errors="replace"
    ):
        raise SubmissionBuildError(
            f"pdflatex failed to produce output for {tex_name} in {flat_dir}. "
            f"See {log_path} for details."
        )
    return log_path


def _run_bibtex(flat_dir: Path, job_stem: str) -> None:
    subprocess.run(
        ["bibtex", job_stem], cwd=flat_dir, capture_output=True, text=True, check=False
    )
    bbl_path = flat_dir / f"{job_stem}.bbl"
    blg_path = flat_dir / f"{job_stem}.blg"
    if not bbl_path.is_file():
        blg_text = blg_path.read_text(encoding="utf-8", errors="replace") if blg_path.is_file() else "(no .blg)"
        raise SubmissionBuildError(
            f"bibtex failed for {job_stem} in {flat_dir}:\n{blg_text}"
        )


def compile_and_verify(
    flat_dir: Path, tex_name: str, reference_pdf: Path, label: str
) -> VerificationResult:
    """Runs pdflatex + bibtex + pdflatex x2 (mirroring build_dami.bat /
    supplement/compile.bat) inside flat_dir and fails loud if the result
    diverges from the checked-in reference PDF: different page count, or any
    forbidden marker (undefined/Overfull/invalid in math mode/Author
    undefined) in the final log."""
    try:
        import pypdf
    except ImportError as exc:
        raise SubmissionBuildError(
            "The 'pypdf' package is required for submission verification "
            "(page-count comparison). Install it with "
            "`venv\\python.exe -m pip install pypdf`."
        ) from exc

    if not reference_pdf.is_file():
        raise SubmissionBuildError(
            f"Reference PDF not found: {reference_pdf}. Compile it first "
            f"(clanek_en/{('dami/build_dami' if label == 'dami_en' else 'supplement/compile')}"
            f".bat) so this script has something to verify against."
        )

    extra_args = ["-disable-installer"] if _is_miktex() else []
    job_stem = Path(tex_name).stem

    _run_pdflatex(flat_dir, tex_name, extra_args=extra_args)
    _run_bibtex(flat_dir, job_stem)
    _run_pdflatex(flat_dir, tex_name, extra_args=extra_args)
    log_path = _run_pdflatex(flat_dir, tex_name, extra_args=extra_args)

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    found_markers = [m for m in FORBIDDEN_LOG_MARKERS if m in log_text]
    if found_markers:
        raise SubmissionBuildError(
            f"[{label}] verification compile log contains forbidden marker(s) "
            f"{found_markers}: see {log_path}"
        )

    pdf_path = flat_dir / f"{job_stem}.pdf"
    if not pdf_path.is_file():
        raise SubmissionBuildError(f"[{label}] no PDF produced: expected {pdf_path}")

    pages = len(pypdf.PdfReader(str(pdf_path)).pages)
    reference_pages = len(pypdf.PdfReader(str(reference_pdf)).pages)
    if pages != reference_pages:
        raise SubmissionBuildError(
            f"[{label}] page count mismatch: flat package has {pages} pages, "
            f"reference {reference_pdf} has {reference_pages}. The flattening "
            f"changed the typeset content - do not submit this package."
        )
    return VerificationResult(
        label=label, pages=pages, reference_pages=reference_pages, log_path=log_path
    )


# By-products of the mandatory verification compile. They are regenerable
# and are NOT source files, so they must not travel to the publisher: the
# submission form asks for "all source files" of a directory, and a stray
# .log or .aux in that upload is at best noise and at worst a reviewer
# reading our local paths. The .bbl is deliberately NOT in this list -
# Springer wants it, because the editorial system does not run BibTeX.
# .spl is an elsarticle by-product that travels with the sources; it has
# no business in an archive handed to the publisher.
VERIFICATION_ARTIFACT_SUFFIXES = (
    ".aux", ".log", ".blg", ".out", ".spl", ".synctex.gz",
)


def _clean_verification_artifacts(flat_dir: Path, *, keep: frozenset[str]) -> list[str]:
    """Deletes the compile by-products left in a finished package, keeping
    the named files (an .aux another package needs for \\externaldocument).
    Returns the names removed, for the run log."""
    removed = []
    for path in sorted(flat_dir.iterdir()):
        if not path.is_file() or path.name in keep:
            continue
        if path.name.endswith(VERIFICATION_ARTIFACT_SUFFIXES):
            path.unlink()
            removed.append(path.name)
    return removed


def _claim_journal_dir(output_root: Path, journal: JournalProfile) -> Path:
    """Returns (creating it if needed) the per-journal directory and makes
    sure it does not already hold a package built for a different journal.

    The stamp is what keeps two submission rounds apart: if the next
    iteration targets another journal, it gets its own directory, and any
    attempt to overwrite this one is refused instead of silently mixing
    files from two different manuscript layouts."""
    journal_dir = output_root / journal.key
    stamp = journal_dir / JOURNAL_STAMP_NAME
    if stamp.is_file():
        first_line = stamp.read_text(encoding="utf-8").splitlines()[0].strip()
        found_key = first_line.split(":", 1)[-1].strip() if ":" in first_line else first_line
        if found_key != journal.key:
            raise SubmissionBuildError(
                f"{journal_dir} already holds a package built for journal "
                f"'{found_key}', not '{journal.key}'. Refusing to overwrite - "
                "delete that directory or pick a different --output-root."
            )
    journal_dir.mkdir(parents=True, exist_ok=True)
    return journal_dir


def _warn_about_legacy_layout(output_root: Path) -> None:
    """Before 2026-09-18 the packages lived directly in submission/ with no
    journal directory. Those leftovers are not regenerated by this script,
    so say so loudly instead of leaving two competing copies around."""
    legacy = [
        name
        for name in ("dami_en", "supplement_en")
        if (output_root / name).is_dir()
    ]
    if legacy:
        print(
            "[make_submission] WARNING: found packages in the old layout "
            f"({', '.join(output_root.joinpath(n).as_posix() for n in legacy)}). "
            "They are stale and are NOT rebuilt - delete them so the only "
            "packages left are the per-journal ones."
        )


def _write_journal_stamp(
    journal_dir: Path, journal: JournalProfile, verified: dict[str, VerificationResult]
) -> None:
    """Writes the journal identity plus the submission-form map, so the
    directory says on its own what it is for without consulting the docs."""
    lines = [
        f"journal: {journal.key}",
        f"name: {journal.name} ({journal.publisher})",
        f"submission system: {journal.submission_system}",
        f"article type: {journal.article_type}",
        f"manuscript class: {journal.manuscript_class}",
        "",
        "Upload map (what to hand the submission system):",
    ]
    for path_in_package, slot in journal.upload_slots:
        lines.append(f"  {path_in_package}  ->  {slot}")
    if verified:
        lines += ["", "Verified page counts of this build:"]
        for label, verification in sorted(verified.items()):
            lines.append(f"  {label}: {verification.pages} pages")
    else:
        lines += ["", "NOT VERIFIED (built with --skip-verify) - do not submit."]
    lines += [
        "",
        "Generated by src/tools/make_submission.py - regenerate rather than edit.",
        "A package for a different journal gets its own directory next to this one.",
    ]
    (journal_dir / JOURNAL_STAMP_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_readme(result: BuildResult, spec: DocumentSpec, journal: JournalProfile) -> None:
    lines = [
        f"Flat submission package: {spec.output_subdir_name}",
        f"Prepared for: {journal.name} ({journal.publisher}), "
        f"{journal.article_type}",
        "Generated by src/tools/make_submission.py - do not edit by hand;",
        "regenerate instead (see src/run_make_submission.bat / .sh).",
        "",
        "To compile (pdflatex + bibtex + pdflatex x2), run from this directory:",
        f"  pdflatex -interaction=nonstopmode {result.root_tex_flat_name}",
        f"  bibtex {Path(result.root_tex_flat_name).stem}",
        f"  pdflatex -interaction=nonstopmode {result.root_tex_flat_name}",
        f"  pdflatex -interaction=nonstopmode {result.root_tex_flat_name}",
        "",
        "Files in this package:",
    ]
    for flat_name, source_path in result.manifest:
        try:
            rel_source = source_path.relative_to(PROJECT_ROOT)
        except ValueError:
            rel_source = source_path
        lines.append(f"  {flat_name}  (from {rel_source.as_posix()})")
    if result.needs_external_aux_from is not None:
        dep_spec = SPECS_BY_KEY[result.needs_external_aux_from]
        lines += [
            "",
            f"  {dep_spec.root_tex.stem}.aux  (copied from the verified "
            f"{dep_spec.output_subdir_name} build; required by \\externaldocument "
            "for cross-references into the main text)",
        ]
    (result.flat_dir / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _wire_external_aux(
    dependent_result: BuildResult, output_root: Path, verified: dict[str, VerificationResult]
) -> None:
    """Copies the freshly-verified dami_en/main_dami.aux into a package that
    needs it (currently only supplement_en). Fails loud if the dependency
    was not actually built and verified in this run."""
    dep_key = dependent_result.needs_external_aux_from
    if dep_key is None:
        return
    if dep_key not in verified:
        raise SubmissionBuildError(
            f"'{dependent_result.flat_dir.name}' cross-references the "
            f"'{dep_key}' package via \\externaldocument, but that package "
            "was not built and verified in this run. Run with "
            "--variant both (or at least build+verify the dependency first) "
            "- refusing to ship a package with a broken cross-reference."
        )
    dep_spec = SPECS_BY_KEY[dep_key]
    dep_flat_dir = output_root / dep_spec.output_subdir_name
    src_aux = dep_flat_dir / f"{dep_spec.root_tex.stem}.aux"
    if not src_aux.is_file():
        raise SubmissionBuildError(
            f"Expected aux file not found after verifying '{dep_key}': {src_aux}"
        )
    shutil.copy2(src_aux, dependent_result.flat_dir / src_aux.name)


def run(
    output_root: Path,
    variant: str,
    skip_verify: bool,
    journal: JournalProfile | None = None,
) -> dict[str, VerificationResult]:
    """Orchestrates one or both packages for one target journal. 'supplement'
    implies building manuscript_en too (see module docstring) because
    supplement_en's cross-references depend on its freshly verified aux."""
    journal = journal or JOURNALS["dami"]
    keys_to_build = {
        "dami": ["dami"],
        "supplement": ["dami", "supplement"],
        "both": ["dami", "supplement"],
    }[variant]

    output_root.mkdir(parents=True, exist_ok=True)
    _warn_about_legacy_layout(output_root)
    output_root = _claim_journal_dir(output_root, journal)
    print(f"[make_submission] Target journal: {journal.name} -> {output_root}")
    results: dict[str, BuildResult] = {}
    verified: dict[str, VerificationResult] = {}
    for key in keys_to_build:
        spec = SPECS_BY_KEY[key]
        print(f"[make_submission] Flattening '{key}' into {output_root / spec.output_subdir_name} ...")
        result = build_flat_package(spec, output_root)
        results[key] = result
        if result.needs_external_aux_from is not None:
            _wire_external_aux(result, output_root, verified)
        _write_readme(result, spec, journal)
        print(f"[make_submission] '{key}': {len(result.manifest)} files copied.")
        if skip_verify:
            print(f"[make_submission] WARNING: --skip-verify set, '{key}' is UNVERIFIED - do not submit it.")
            continue
        print(f"[make_submission] Verifying '{key}' with a full LaTeX compile ...")
        verification = compile_and_verify(
            result.flat_dir, result.root_tex_flat_name, spec.reference_pdf, spec.output_subdir_name
        )
        verified[key] = verification
        print(
            f"[make_submission] '{key}' OK: {verification.pages} pages "
            f"(reference {verification.reference_pages}), no forbidden log markers."
        )
    # Only now, with every package built and verified (and any cross-package
    # .aux already copied where it was needed), is it safe to throw the
    # compile by-products away.
    for key, result in results.items():
        # A package that cross-references another one keeps the borrowed
        # .aux - it is an input of this package, not a by-product of it.
        keep = frozenset(
            [f"{SPECS_BY_KEY[result.needs_external_aux_from].root_tex.stem}.aux"]
            if result.needs_external_aux_from is not None
            else []
        )
        removed = _clean_verification_artifacts(result.flat_dir, keep=keep)
        if removed:
            print(f"[make_submission] '{key}': removed compile by-products ({', '.join(removed)}).")
    _write_journal_stamp(output_root, journal, verified)
    return verified


def write_package_zip(package_dir: Path) -> Path:
    """Zips one flat package directory for Springer Nature SNAPP, which takes
    the LaTeX sources as a single archive and compiles them itself. Entries are
    stored at the root of the archive (no wrapping directory): SNAPP looks for
    the main .tex there. Files are added in sorted order so that two runs over
    the same directory produce the same archive layout."""
    if not package_dir.is_dir():
        raise SubmissionBuildError(
            f"Cannot zip {package_dir}: the package directory does not exist. "
            "Build the package first."
        )
    zip_path = package_dir.with_suffix(".zip")
    members = sorted(
        path
        for path in package_dir.iterdir()
        if path.is_file() and path.suffix.lower() != ".zip"
    )
    if not members:
        raise SubmissionBuildError(f"Cannot zip {package_dir}: no files in it.")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for member in members:
            archive.write(member, arcname=member.name)
    print(
        f"[make_submission] Wrote {zip_path} ({len(members)} files, "
        f"{zip_path.stat().st_size // 1024} kB)."
    )
    return zip_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--journal",
        choices=sorted(JOURNALS),
        default="dami",
        help="Target journal; names the output directory and the JOURNAL.txt "
        "stamp so packages for different journals cannot be mixed up "
        "(default: dami).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "submission",
        help="Base directory holding the per-journal directories "
        "(default: <repo>/submission, so the packages land in "
        "<repo>/submission/<journal>/).",
    )
    parser.add_argument(
        "--variant",
        choices=["dami", "supplement", "both"],
        default="both",
        help="Which package(s) to build (default: both).",
    )
    parser.add_argument(
        "--no-zip",
        action="store_true",
        help="Do not write the .zip archives next to the package directories. "
        "The archives are what Springer Nature SNAPP takes as the upload, so "
        "skip them only when building for a system that wants loose files.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip the mandatory compile-and-check step. Debugging only - "
        "never use for an actual submission build.",
    )
    args = parser.parse_args(argv)

    try:
        verified = run(
            args.output_root, args.variant, args.skip_verify, JOURNALS[args.journal]
        )
    except SubmissionBuildError as exc:
        print(f"[make_submission] FAILED: {exc}", file=sys.stderr)
        return 1

    if not args.no_zip:
        journal_dir = args.output_root / args.journal
        try:
            for label in verified:
                write_package_zip(journal_dir / SPECS_BY_KEY[label].output_subdir_name)
        except SubmissionBuildError as exc:
            print(f"[make_submission] FAILED: {exc}", file=sys.stderr)
            return 1

    print("[make_submission] Done.")
    for label, verification in verified.items():
        print(f"  {label}: {verification.pages} pages (reference {verification.reference_pages}) - OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
