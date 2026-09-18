#!/usr/bin/env bash
set -euo pipefail
# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-17
# License: see the LICENSE file in the repository root
#
# Pre-submission check: verifies that every \supref{N}/\suprefs{N}{M} in
# clanek/sections/*.tex points at a section that actually exists in
# clanek/supplement/supplement.tex, and prints the title of the target
# section next to each reference for a human sanity check. Exit code 1 on
# any failure. See src/validate_supplement_refs.py.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: project venv Python interpreter not found: $PYTHON" >&2
    echo "Set the PYTHON environment variable to the project's venv Python path, e.g.:" >&2
    echo "  PYTHON=/path/to/venv/bin/python $0" >&2
    exit 1
fi

"$PYTHON" -m src.validate_supplement_refs
