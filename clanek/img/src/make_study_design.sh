#!/usr/bin/env bash
set -euo pipefail
# Project: Sammon mapping revisited (alpha-Sammon, scalable solver, temporal extension)
# Author: Martin Radvansky, VSB - Technical University of Ostrava
# Contact: martinradvansky@gmail.com
# Created: 2026-09-17
# License: for research use within the project; see projectstate.md
#
# Regenerates clanek/img/study_design.pdf (process diagram of the STUDY
# DESIGN: core/hold-out split, embedding grid, metrics, rule fitting,
# confirmatory test) from the hand-editable source
# clanek/img/src/study_design.svg.
# Converter: svglib + reportlab inside the project conda env (venv), driven by
# clanek/img/src/svg2pdf.py (Inkscape is not on PATH on the author's machine).
# Usage: make_study_design.sh   (from any directory; paths are resolved
#        relative to this file: <repo>/clanek/img/src/)

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SRC_DIR/../../.." && pwd)"
PYTHON="${PYTHON:-$REPO/venv/bin/python}"
SVG="$SRC_DIR/study_design.svg"
PDF="$SRC_DIR/../study_design.pdf"

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python interpreter not found: \"$PYTHON\"" >&2
    echo "Set the PYTHON environment variable to the project's venv Python." >&2
    exit 1
fi
if [ ! -e "$SVG" ]; then
    echo "ERROR: SVG source not found: \"$SVG\"" >&2
    exit 1
fi

echo "[study_design] converting SVG to PDF ..."
export PYTHONIOENCODING=utf-8
if ! "$PYTHON" "$SRC_DIR/svg2pdf.py" "$SVG" "$PDF"; then
    echo "ERROR: SVG to PDF conversion failed for study_design.svg" >&2
    exit 1
fi
echo "[study_design] done: \"$PDF\""
exit 0
