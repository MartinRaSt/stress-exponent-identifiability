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
# One command for the rerun after the 2026-09-12 review: preparation -> E2 (2-3 h)
# -> E3 alpha_auto (~1 h) -> run_main (statistics, tables, figures).
# Sleep is disabled by each called run_*.sh itself (no_sleep.sh).
# WARNING: run only after the previous run_main.sh (figures) has finished.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source "$ROOT/src/common/no_sleep.sh"
sammon_no_sleep_reexec "$0" "$@"

RCP=""
if "$ROOT/src/prepare_rerun_20260912.sh"; then RCP=0; else RCP=$?; fi

if [ "$RCP" = "0" ]; then
    RC2=""; RC3=""; RCM=""
    if "$ROOT/src/run_exp2_solver_scaling.sh" full; then RC2=0; else RC2=$?; fi
    if "$ROOT/src/run_exp3_graph_layout.sh" full; then RC3=0; else RC3=$?; fi
    if "$ROOT/src/run_main.sh" full; then RCM=0; else RCM=$?; fi

    echo "============================================================"
    echo "  E2 solver_scaling exit=$RC2"
    echo "  E3 graph_layout   exit=$RC3"
    echo "  main              exit=$RCM"
    echo "============================================================"
    cat "results/data/exp2_solver_scaling_DONE.txt" 2>/dev/null || true
    cat "results/data/exp3_graph_layout_DONE.txt" 2>/dev/null || true
fi
