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
# One command for the whole phase S1 (documentation/2026-09-12_plan_smeru_clanku.md):
#   1. dataset_properties (K1, incl. correlation check with alpha)
#   2. exp1_cluster_geometry (K2, over the saved E1 embeddings)
#   3. pareto_analysis (K3)
#   4. make_figures (all figures incl. the new K3/K4/K9)
#   5. export_numbers (K12a -> clanek/generated/numbers.tex; quick|smoke -> results/tables/<mode>/)
# Steps 1-3 stop on error (later steps depend on them); steps 4-5 always
# run and the exit codes are printed at the end. Sleep is disabled by
# each called run_*.sh itself (no_sleep.sh), here additionally for the whole chain.
# Parameter: full (default) | quick | smoke

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source "$ROOT/src/common/no_sleep.sh"
sammon_no_sleep_reexec "$0" "$@"

MODE="${1:-full}"
LOGDIR="results/logs"
mkdir -p "$LOGDIR"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] S1 start (mode=$MODE)" >> "$LOGDIR/run_s1_all.log"

RC1=""; RC2=""; RC3=""; RC4=""; RC5=""

echo "=== 1/5 dataset_properties ==="
if "$ROOT/src/run_dataset_properties.sh" "$MODE"; then
    RC1=0
else
    RC1=$?
    echo "ERROR: dataset_properties finished with code $RC1 - stopping."
fi

if [ "$RC1" = "0" ]; then
    echo "=== 2/5 exp1_cluster_geometry ==="
    if "$ROOT/src/run_exp1_cluster_geometry.sh" "$MODE"; then
        RC2=0
    else
        RC2=$?
        echo "ERROR: exp1_cluster_geometry finished with code $RC2 - stopping."
    fi
fi

if [ "$RC1" = "0" ] && [ "$RC2" = "0" ]; then
    echo "=== 3/5 pareto_analysis ==="
    if "$ROOT/src/run_pareto_analysis.sh" "$MODE"; then
        RC3=0
    else
        RC3=$?
        echo "ERROR: pareto_analysis finished with code $RC3 - stopping."
    fi
fi

echo "=== 4/5 make_figures ==="
if "$ROOT/src/make_figures.sh" "$MODE"; then RC4=0; else RC4=$?; fi

echo "=== 5/5 export_numbers ==="
if "$ROOT/src/run_export_numbers.sh" "$MODE"; then RC5=0; else RC5=$?; fi

echo "============================================================"
echo "  1 dataset_properties     exit=$RC1"
echo "  2 exp1_cluster_geometry  exit=$RC2"
echo "  3 pareto_analysis        exit=$RC3"
echo "  4 make_figures           exit=$RC4"
echo "  5 export_numbers         exit=$RC5"
echo "============================================================"
cat "results/data/exp1_cluster_geometry_DONE.txt" 2>/dev/null || true
echo "[$(date '+%Y-%m-%d %H:%M:%S')] S1 end rc=$RC1/$RC2/$RC3/$RC4/$RC5" >> "$LOGDIR/run_s1_all.log"
