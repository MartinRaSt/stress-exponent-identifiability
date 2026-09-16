#!/usr/bin/env bash
set -euo pipefail
# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-12
# License: see the LICENSE file in the repository root
#
# Preparation for a rerun after reviewing the 2026-09-12 results (author decision):
#  - E3: delete sammon_alpha_auto rows (recompute the selected_hyperparam column)
#  - E2: schema of exp2_scaling_results.csv already fixed on 2026-09-12 (column
#    stress_sparse_terms), here just a check; E2 then resumes on its own.
# Parameter "dry" = print only, no writes.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: project venv Python interpreter not found: $PYTHON" >&2
    echo "Set the PYTHON environment variable to the project's venv Python path, e.g.:" >&2
    echo "  PYTHON=/path/to/venv/bin/python $0 $*" >&2
    exit 1
fi

ARG1="${1:-}"
DRY=""
if [ "${ARG1,,}" = "dry" ]; then
    DRY="--dry-run"
fi

echo "=== E3: sammon_alpha_auto (recompute selected_hyperparam) ==="
if [ -n "$DRY" ]; then
    RM_OK=0
    "$PYTHON" -m src.experiments.remove_rows --csv results/data/exp3_graph_layout_results.csv --method sammon_alpha_auto "$DRY" || RM_OK=$?
else
    RM_OK=0
    "$PYTHON" -m src.experiments.remove_rows --csv results/data/exp3_graph_layout_results.csv --method sammon_alpha_auto || RM_OK=$?
fi
if [ "$RM_OK" -ne 0 ]; then
    echo "=== ERROR while preparing the rerun, do not run anything else ===" >&2
    exit 1
fi

echo "=== E2: schema check for exp2_scaling_results.csv ==="
SCHEMA_OK=0
"$PYTHON" -c "import csv,sys; h=next(csv.reader(open(r'results/data/exp2_scaling_results.csv',encoding='utf-8'))); sys.exit(0 if 'stress_sparse_terms' in h else 1)" || SCHEMA_OK=$?
if [ "$SCHEMA_OK" -ne 0 ]; then
    echo "ERROR: exp2_scaling_results.csv is missing the stress_sparse_terms column - see documentation/2026-09-12_kontrola_vysledku_prebehu.md section 2" >&2
    echo "=== ERROR while preparing the rerun, do not run anything else ===" >&2
    exit 1
fi
echo "  OK"

if [ -z "$DRY" ]; then
    echo "=== E2/E3 DONE markers deleted ==="
    rm -f results/data/exp2_solver_scaling_DONE.txt results/data/exp3_graph_layout_DONE.txt 2>/dev/null || true
fi
echo "=== preparation done ==="
exit 0
