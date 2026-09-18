#!/usr/bin/env bash
set -euo pipefail
# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-18
# License: see the LICENSE file in the repository root
#
# Generates dami_submission/01_TEXTY_DO_FORMULARE.md and
# dami_submission/02_COVER_LETTER.md from the CURRENT
# clanek_en/dami/main_dami.tex and clanek/generated/numbers.tex - see
# src/tools/make_submission_guide.py for the full description. Run this ONLY
# once the manuscript is final (author's explicit instruction: these two
# files are deliberately left empty in dami_submission/ until then, see
# dami_submission/00_POSTUP_SUBMISSION.md).
#
# Requires: clanek_en/dami/main_dami.tex up to date and
# clanek/generated/numbers.tex already generated (src/run_main.sh).
#
# Usage: src/run_make_submission_guide.sh <ZENODO-DOI> [output-dir]
#   e.g.: src/run_make_submission_guide.sh 10.5281/zenodo.1234567

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Disable computer standby/hibernation for the duration of the run (systemd-inhibit, see
# src/common/no_sleep.sh) - author requirement.
source "$SCRIPT_DIR/common/no_sleep.sh"
sammon_no_sleep_reexec "$0" "$@"

PYTHON="${PYTHON:-venv/python.exe}"
[ -x "$PYTHON" ] || PYTHON="venv/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: project venv Python interpreter not found: $PYTHON" >&2
    exit 1
fi

ZENODO_DOI="${1:-}"
OUTDIR="${2:-}"

if [ -z "$ZENODO_DOI" ]; then
    echo "[run_make_submission_guide] ERROR: missing required argument ZENODO-DOI." >&2
    echo "Usage: src/run_make_submission_guide.sh <ZENODO-DOI> [output-dir]" >&2
    exit 1
fi

if [ -z "$OUTDIR" ]; then
    "$PYTHON" -m src.tools.make_submission_guide "$ZENODO_DOI"
else
    "$PYTHON" -m src.tools.make_submission_guide "$ZENODO_DOI" --output-dir "$OUTDIR"
fi
