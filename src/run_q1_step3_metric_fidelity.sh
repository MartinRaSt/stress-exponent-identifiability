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
# Q1-step3 (reserse/2026-09-14_specifikace_rozsireni_q1.md PART B, B.7 item 8)
# - one command for the whole metric fidelity task with known ground truth:
#   1. run_exp9_metric_fidelity        (grid of scenario x replicate x method)
#   2. run_exp9_metric_fidelity_stats  (families F_B1/F_B2/F_B3, Holm)
#   3. run_fig_metric_fidelity_gallery (gallery of r=0 embeddings)
#   4. run_fig_metric_fidelity_boxes   (boxplots of the primary metric)
#   5. run_main --no-figures           (updates numbers.tex incl. the mf* macros)
# The order is mandatory (each next step reads the output of the previous one); fail-stop -
# steps 2-5 only run after step 1 has completed successfully (otherwise nonsensical
# empty/missing inputs). Independent of run_q1_step2_datasets.sh (different
# CSVs, different embeddings), but do NOT run CONCURRENTLY (limit of 8 workers x 1 BLAS
# thread, see config.yaml parallel.*).
# Parameter: quick|full|smoke (default full).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source "$ROOT/src/common/no_sleep.sh"
sammon_no_sleep_reexec "$0" "$@"

MODE="${1:-full}"
LOGDIR="results/logs"
mkdir -p "$LOGDIR"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Q1-step3 start (mode=$MODE)" >> "$LOGDIR/run_q1_step3_metric_fidelity.log"

RC1=""; RC2=""; RC3=""; RC4=""; RC5=""

echo "=== 1/5 run_exp9_metric_fidelity ==="
if "$ROOT/src/run_exp9_metric_fidelity.sh" "$MODE"; then
    RC1=0
else
    RC1=$?
    echo "ERROR: run_exp9_metric_fidelity finished with code $RC1 - stopping."
fi

if [ "$RC1" = "0" ]; then
    echo "=== 2/5 run_exp9_metric_fidelity_stats ==="
    if "$ROOT/src/run_exp9_metric_fidelity_stats.sh" "$MODE"; then
        RC2=0
    else
        RC2=$?
        echo "ERROR: run_exp9_metric_fidelity_stats finished with code $RC2 - stopping."
    fi
fi

if [ "$RC1" = "0" ] && [ "$RC2" = "0" ]; then
    echo "=== 3/5 run_fig_metric_fidelity_gallery ==="
    if "$ROOT/src/run_fig_metric_fidelity_gallery.sh" "$MODE"; then
        RC3=0
    else
        RC3=$?
        echo "ERROR: run_fig_metric_fidelity_gallery finished with code $RC3 - stopping."
    fi
fi

if [ "$RC1" = "0" ] && [ "$RC2" = "0" ] && [ "$RC3" = "0" ]; then
    echo "=== 4/5 run_fig_metric_fidelity_boxes ==="
    if "$ROOT/src/run_fig_metric_fidelity_boxes.sh" "$MODE"; then
        RC4=0
    else
        RC4=$?
        echo "ERROR: run_fig_metric_fidelity_boxes finished with code $RC4 - stopping."
    fi
fi

echo "=== 5/5 run_main --no-figures (numbers.tex incl. mf* macros) ==="
if "$ROOT/src/run_main.sh" "$MODE" --no-figures; then RC5=0; else RC5=$?; fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Q1-step3 end (mode=$MODE) RC1=$RC1 RC2=$RC2 RC3=$RC3 RC4=$RC4 RC5=$RC5" >> "$LOGDIR/run_q1_step3_metric_fidelity.log"
echo "Summary: exp9=$RC1 stats=$RC2 gallery=$RC3 boxes=$RC4 main=$RC5"

if [ "$RC1" != "0" ]; then exit "$RC1"; fi
if [ "$RC2" != "0" ]; then exit "$RC2"; fi
if [ "$RC3" != "0" ]; then exit "$RC3"; fi
if [ "$RC4" != "0" ]; then exit "$RC4"; fi
exit "$RC5"
