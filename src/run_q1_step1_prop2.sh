#!/usr/bin/env bash
set -euo pipefail
# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-14
# License: see the LICENSE file in the repository root
#
# Q1-step1 (documentation/2026-09-14_exp8_prop2_check.md): one command for
# the whole step - post-hoc empirical test of Proposition 2 over the already finished
# exp6_alpha_curves embeddings:
#   1. exp8_prop2_check   (src/experiments/exp8_prop2_check.py)
#   2. fig_prop2_tightness           (tightness of the bounds vs. alpha/rho_NN)
#   3. fig_prop2_alpha_prediction    (alpha_bound theory vs. alpha_star empirical)
#   4. export_numbers (K12a -> clanek/generated/numbers.tex; quick|smoke ->
#      results/tables/<mode>/numbers.tex)
# ASSUMPTION: exp6_alpha_curves has already run IN THE SAME mode (otherwise exp8
# fails fail-loud on missing .npy embeddings) - run
# src/run_exp6_alpha_curves.sh MODE first if it has not run yet.
# Steps 1-3 stop on error (later steps depend on them); step 4 always
# runs and the exit codes are printed at the end. Sleep is disabled by each
# called run_*.sh itself (no_sleep.sh), here additionally for the whole chain.
# Parameter: full (default) | quick | smoke

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source "$ROOT/src/common/no_sleep.sh"
sammon_no_sleep_reexec "$0" "$@"

MODE="${1:-full}"
LOGDIR="results/logs"
mkdir -p "$LOGDIR"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Q1-step1 start (mode=$MODE)" >> "$LOGDIR/run_q1_step1_prop2.log"

RC1=""; RC2=""; RC3=""; RC4=""

echo "=== 1/4 exp8_prop2_check ==="
if "$ROOT/src/run_exp8_prop2_check.sh" "$MODE"; then
    RC1=0
else
    RC1=$?
    echo "ERROR: exp8_prop2_check finished with code $RC1 - stopping."
fi

if [ "$RC1" = "0" ]; then
    echo "=== 2/4 fig_prop2_tightness ==="
    if "$ROOT/src/run_fig_prop2_tightness.sh" "$MODE"; then
        RC2=0
    else
        RC2=$?
        echo "ERROR: fig_prop2_tightness finished with code $RC2 - stopping."
    fi
fi

if [ "$RC1" = "0" ] && [ "$RC2" = "0" ]; then
    echo "=== 3/4 fig_prop2_alpha_prediction ==="
    if "$ROOT/src/run_fig_prop2_alpha_prediction.sh" "$MODE"; then
        RC3=0
    else
        RC3=$?
        echo "ERROR: fig_prop2_alpha_prediction finished with code $RC3 - stopping."
    fi
fi

echo "=== 4/4 export_numbers ==="
if "$ROOT/src/run_export_numbers.sh" "$MODE"; then RC4=0; else RC4=$?; fi

echo "============================================================"
echo "Q1-step1 summary (mode=$MODE):"
echo "  1/4 exp8_prop2_check           RC=$RC1"
echo "  2/4 fig_prop2_tightness        RC=$RC2"
echo "  3/4 fig_prop2_alpha_prediction RC=$RC3"
echo "  4/4 export_numbers             RC=$RC4"
echo "============================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Q1-step1 end (mode=$MODE) RC1=$RC1 RC2=$RC2 RC3=$RC3 RC4=$RC4" >> "$LOGDIR/run_q1_step1_prop2.log"

if [ "$RC1" != "0" ]; then exit "$RC1"; fi
if [ "$RC2" != "0" ]; then exit "$RC2"; fi
if [ "$RC3" != "0" ]; then exit "$RC3"; fi
exit "$RC4"
