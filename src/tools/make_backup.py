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
Project backup in two profiles, so that an archive never carries what a script
can rebuild (author request 2026-09-18: "backup of the sources only, nothing
that can be regenerated, so the archive does not get too big").

  sources  Everything a human wrote: code, tests, article text, BibTeX, the
           SVG sources of the hand-drawn figures, documentation, reviews and
           the git history. EXCLUDES the venv, the 2.8 GB dataset/graph cache
           in src/data, and every generated artefact (results/, figure PDFs,
           LaTeX build files). `clanek/generated/numbers.tex` IS kept - it is
           generated, but it is tiny and pins the numbers used in the text.

  data     The measured experiment CSVs under results/data only. They are
           "regenerable" in principle, but only by re-running experiments that
           take days, so they are archived separately rather than dropped.
           EXCLUDES embeddings (295 MB), smoke/quick output and the dated
           backup_e4_*/archive_*/.bak_* copies.

Run: venv\\python.exe -m src.tools.make_backup <sources|data> <out.zip>
or:  src\\run_backup.bat        (writes both archives next to the project)
"""
from __future__ import annotations

import fnmatch
import os
import sys
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Never archived, in any profile.
_SKIP_PREFIXES_COMMON = (
    "venv",
    "src/data",  # dataset and graph cache - re-downloadable, 2.8 GB
)
_SKIP_DIR_NAMES = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}

# Rebuilt by src/make_figures.bat, src/run_main.bat and the LaTeX compile.
_SKIP_PREFIXES_SOURCES = _SKIP_PREFIXES_COMMON + ("results",)

_SKIP_PREFIXES_DATA = _SKIP_PREFIXES_COMMON + (
    "results/data/embeddings",
    "results/data/quick",
    "results/data/smoke",
    "results/figures",
    "results/tables",
    "results/logs",
)

# Build artefacts. Figure PDFs live in clanek/img and are covered by '*.pdf';
# the SVG sources next to them in clanek/img/src are kept.
_SKIP_FILE_PATTERNS = (
    "*.pdf", "*.aux", "*.log", "*.out", "*.fls", "*.fdb_latexmk", "*.bbl",
    "*.blg", "*.spl", "*.synctex.gz", "*.toc", "*.lof", "*.lot", "*.pyc",
    "*.npy", "*.npz", "*.png", "*.jpg", "*.jpeg",
)
# Sources that match a pattern above and must survive anyway.
_KEEP_ALWAYS = (
    "konference/*",  # the author's two 2013 papers this project builds on
    "casopisy/*",    # journal templates and author guidelines
)
# Dated "just in case" copies - the live file next to them is the source.
_STALE_PATTERNS = ("*.bak_*", "*backup_e4_*", "*archive_2*")


def _rel(path: str) -> str:
    """Repository-relative path with forward slashes."""
    return os.path.relpath(path, ROOT).replace(os.sep, "/")


def _skip_dir(rel: str, prefixes: tuple[str, ...], drop_stale: bool) -> bool:
    """True if the whole directory is left out of the archive."""
    name = os.path.basename(rel)
    if name in _SKIP_DIR_NAMES:
        return True
    if drop_stale and any(fnmatch.fnmatch(name, p) for p in _STALE_PATTERNS):
        return True
    return any(rel == p or rel.startswith(p + "/") for p in prefixes)


def _keep_file(rel: str, drop_stale: bool) -> bool:
    """True if this file belongs in the archive."""
    if any(fnmatch.fnmatch(rel, pat) for pat in _KEEP_ALWAYS):
        return True
    name = os.path.basename(rel)
    if drop_stale and any(fnmatch.fnmatch(name, p) for p in _STALE_PATTERNS):
        return False
    return not any(fnmatch.fnmatch(name, pat) for pat in _SKIP_FILE_PATTERNS)


def build_archive(profile: str, out_path: str) -> tuple[int, int]:
    """Writes the zip for `profile`; returns (file count, uncompressed bytes)."""
    if profile == "sources":
        prefixes, drop_stale, include_only = _SKIP_PREFIXES_SOURCES, False, None
    elif profile == "data":
        prefixes, drop_stale, include_only = _SKIP_PREFIXES_DATA, True, "results/data/"
    else:
        raise ValueError("profile must be 'sources' or 'data', got %r" % profile)

    n_files = 0
    total = 0
    by_top: dict[str, int] = {}
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for dirpath, dirnames, filenames in os.walk(ROOT):
            rel_dir = _rel(dirpath)
            rel_dir = "" if rel_dir == "." else rel_dir
            dirnames[:] = [
                d for d in dirnames
                if not _skip_dir((rel_dir + "/" + d).lstrip("/"), prefixes, drop_stale)
            ]
            for fn in filenames:
                rel = (rel_dir + "/" + fn).lstrip("/")
                if include_only is not None and not rel.startswith(include_only):
                    continue
                if not _keep_file(rel, drop_stale):
                    continue
                full = os.path.join(dirpath, fn)
                try:
                    size = os.path.getsize(full)
                except OSError:  # a file vanished between walk and stat
                    continue
                z.write(full, "Sammon/" + rel)
                n_files += 1
                total += size
                top = rel.split("/")[0] if "/" in rel else "(root files)"
                by_top[top] = by_top.get(top, 0) + size

    print("profile: %s" % profile)
    print("files: %d, uncompressed: %.1f MB" % (n_files, total / 1048576.0))
    print("archive: %s (%.1f MB)" % (out_path, os.path.getsize(out_path) / 1048576.0))
    for top, size in sorted(by_top.items(), key=lambda kv: -kv[1]):
        print("  %-18s %8.1f MB" % (top, size / 1048576.0))
    return n_files, total


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print(__doc__)
        return 2
    profile, out_path = args
    if profile not in ("sources", "data"):
        print("ERROR: profile must be 'sources' or 'data', got %r" % profile)
        return 2
    n_files, _ = build_archive(profile, out_path)
    if n_files == 0:
        print("ERROR: the archive is empty - check the skip lists.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
