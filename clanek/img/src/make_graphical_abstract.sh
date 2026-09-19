#!/usr/bin/env bash
set -euo pipefail
# Project: Sammon mapping revisited (alpha-Sammon, scalable solver, temporal extension)
# Author: Martin Radvansky, VSB - Technical University of Ostrava
# Contact: martinradvansky@gmail.com
# Created: 2026-09-17
# License: for research use within the project; see projectstate.md
#
# Regenerates clanek/img/graphical_abstract.pdf (standalone one-glance graphical
# abstract: rho_NN decides for free whether tuning alpha is worth it) from the
# hand-editable source clanek/img/src/graphical_abstract.svg.
# Converter: svglib + reportlab inside the project conda env (venv), driven by
# clanek/img/src/svg2pdf.py (Inkscape is not on PATH on the author's machine).
# Usage: make_graphical_abstract.sh   (from any directory; paths are resolved
#        relative to this file: <repo>/clanek/img/src/)

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SRC_DIR/../../.." && pwd)"
PYTHON="${PYTHON:-$REPO/venv/bin/python}"
SVG="$SRC_DIR/graphical_abstract.svg"
PDF="$SRC_DIR/../graphical_abstract.pdf"

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python interpreter not found: \"$PYTHON\"" >&2
    echo "Set the PYTHON environment variable to the project's venv Python." >&2
    exit 1
fi
if [ ! -e "$SVG" ]; then
    echo "ERROR: SVG source not found: \"$SVG\"" >&2
    exit 1
fi

echo "[graphical_abstract] converting SVG to PDF ..."
export PYTHONIOENCODING=utf-8
if ! "$PYTHON" "$SRC_DIR/svg2pdf.py" "$SVG" "$PDF"; then
    echo "ERROR: SVG to PDF conversion failed for graphical_abstract.svg" >&2
    exit 1
fi
echo "[graphical_abstract] done: \"$PDF\""
exit 0
