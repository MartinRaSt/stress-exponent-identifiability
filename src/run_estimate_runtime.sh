#!/usr/bin/env bash
set -euo pipefail
# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-16
# License: see the LICENSE file in the repository root
#
# Estimate the length of a full experiment run from the MEASURED times (wall_time_sec) in
# an already existing results CSV. Parameter: experiment name (default
# exp4_temporal), optionally a list of worker counts.
# Example: src/run_estimate_runtime.sh exp4_temporal 1,8,12
# Just reads a CSV (seconds) - sleep inhibition is not needed here.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: project venv Python interpreter not found: $PYTHON" >&2
    echo "Set the PYTHON environment variable to the project's venv Python path, e.g.:" >&2
    echo "  PYTHON=/path/to/venv/bin/python $0 $*" >&2
    exit 1
fi

EXPNAME="${1:-exp4_temporal}"
WORKERS="${2:-}"

if [ -n "$WORKERS" ]; then
    "$PYTHON" -m src.tools.estimate_runtime_from_csv "$EXPNAME" --workers "$WORKERS"
else
    "$PYTHON" -m src.tools.estimate_runtime_from_csv "$EXPNAME"
fi
