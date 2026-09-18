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
# Builds the FLAT (single-directory, no subfolders) journal-submission
# packages required by Springer/DAMI from the split clanek_en/ tree - see
# src/tools/make_submission.py for the full description. Writes
# submission/dami_en/ (main article, sn-jnl class) and
# submission/supplement_en/ (elsarticle supplement, cross-referenced into
# the main text) and VERIFIES both with a full LaTeX compile against the
# existing reference PDFs (clanek_en/dami/main_dami.pdf,
# clanek_en/supplement/supplement.pdf) - page count and absence of
# "undefined"/"Overfull"/"invalid in math mode"/"Author undefined" in the
# log. Fails loud (non-zero exit) on any mismatch; the packages must not be
# submitted if this script reports an error.
#
# Requires: clanek_en/dami/main_dami.pdf and clanek_en/supplement/supplement.pdf
# already compiled once (build_dami.sh / compile.sh) as the reference to
# check against, and clanek/generated/numbers.tex already generated
# (src/run_main.sh).
#
# Usage: src/run_make_submission.sh [dami|supplement|both] [output-root]
#   (default: both, submission/ next to this project)

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

VARIANT="${1:-both}"
OUTROOT="${2:-}"

if [ -z "$OUTROOT" ]; then
    "$PYTHON" -m src.tools.make_submission --variant "$VARIANT"
else
    "$PYTHON" -m src.tools.make_submission --variant "$VARIANT" --output-root "$OUTROOT"
fi
