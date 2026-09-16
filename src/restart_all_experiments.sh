#!/usr/bin/env bash
set -euo pipefail
# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-10
# License: see the LICENSE file in the repository root
#
# Restart full runs after a code fix: (1) removes error rows (status=error)
# from the already existing CSVs in results/data so they get recomputed, (2) runs
# all experiments E1-E5 (resume - already completed combinations are skipped).
# Before running, STOP any currently running run_all_experiments.sh!

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Disable computer standby/hibernation for the duration of the run (systemd-inhibit, see
# src/common/no_sleep.sh) - author requirement.
source "$ROOT/src/common/no_sleep.sh"
sammon_no_sleep_reexec "$0" "$@"

PYTHON="${PYTHON:-$ROOT/venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: project venv Python interpreter not found: $PYTHON" >&2
    echo "Set the PYTHON environment variable to the project's venv Python path, e.g.:" >&2
    echo "  PYTHON=/path/to/venv/bin/python $0 $*" >&2
    exit 1
fi

for f in "$ROOT"/results/data/exp*_results.csv; do
    [ -e "$f" ] || continue
    set +e
    "$PYTHON" -m src.experiments.clean_error_rows "$f"
    rc=$?
    set -e
    if [ "$rc" -ne 0 ]; then
        echo "WARNING: clean_error_rows failed for $f (rc=$rc)" >&2
    fi
done

exec "$ROOT/src/run_all_experiments.sh" full
