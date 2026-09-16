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
# One command for the whole phase S2 (documentation/2026-09-12_plan_smeru_clanku.md):
#   1. run_exp6_alpha_curves   (K6 - alpha curves on the 32 E1 datasets)
#   2. run_fit_alpha_rule      (K7 - derivation + LOO validation of alpha_pred, requires 1 and K1 dataset_properties)
#   3. run_exp1_dr_benchmark   (K5/K7 - resume, adds densmap/phate/sammon_alpha_pred)
#   4. run_exp3_graph_layout   (K7/K8 - resume, adds sammon_alpha_pred + new K8 graphs/methods)
#   5. run_exp1_cluster_geometry (K2 - resume, new rows over the new embeddings from step 3)
#   6. run_main                (statistics, tables, pareto, figures, export_numbers)
# The order is mandatory (see plan section 4): exp6 -> fit_alpha_rule -> exp1 -> exp3,
# because exp1/exp3 use the sammon_alpha_pred method, which requires an already
# finished results/data/alpha_pred_rule.json (only in full mode - see
# src/experiments/fit_alpha_rule.py). Steps 1-4 are fail-stop (each next step
# depends on the previous one - without fit_alpha_rule, exp1/exp3 would just record
# sammon_alpha_pred as status=error, which is not fatal but needlessly
# clutters the results); steps 5-6 always run, with an exit code summary at the
# end. Sleep is disabled by each called run_*.sh itself (no_sleep.sh),
# here additionally for the whole chain. Parameter: full (default) | quick | smoke.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source "$ROOT/src/common/no_sleep.sh"
sammon_no_sleep_reexec "$0" "$@"

MODE="${1:-full}"
LOGDIR="results/logs"
mkdir -p "$LOGDIR"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] S2 start (mode=$MODE)" >> "$LOGDIR/run_s2_all.log"

RC1=""; RC2=""; RC3=""; RC4=""; RC5=""; RC6=""

echo "=== 1/6 run_exp6_alpha_curves ==="
if "$ROOT/src/run_exp6_alpha_curves.sh" "$MODE"; then
    RC1=0
else
    RC1=$?
    echo "ERROR: run_exp6_alpha_curves finished with code $RC1 - stopping."
fi

if [ "$RC1" = "0" ]; then
    echo "=== 2/6 run_fit_alpha_rule ==="
    if "$ROOT/src/run_fit_alpha_rule.sh" "$MODE"; then
        RC2=0
    else
        RC2=$?
        echo "ERROR: run_fit_alpha_rule finished with code $RC2 - stopping."
    fi
fi

if [ "$RC1" = "0" ] && [ "$RC2" = "0" ]; then
    echo "=== 3/6 run_exp1_dr_benchmark (resume - densmap/phate/sammon_alpha_pred) ==="
    if "$ROOT/src/run_exp1_dr_benchmark.sh" "$MODE"; then
        RC3=0
    else
        RC3=$?
        echo "ERROR: run_exp1_dr_benchmark finished with code $RC3 - stopping."
    fi
fi

if [ "$RC1" = "0" ] && [ "$RC2" = "0" ] && [ "$RC3" = "0" ]; then
    echo "=== 4/6 run_exp3_graph_layout (resume - sammon_alpha_pred + K8 additions) ==="
    if "$ROOT/src/run_exp3_graph_layout.sh" "$MODE"; then
        RC4=0
    else
        RC4=$?
        echo "ERROR: run_exp3_graph_layout finished with code $RC4 - stopping."
    fi
fi

echo "=== 5/6 run_exp1_cluster_geometry (resume) ==="
if "$ROOT/src/run_exp1_cluster_geometry.sh" "$MODE"; then RC5=0; else RC5=$?; fi

echo "=== 6/6 run_main (statistics, tables, pareto, figures, export_numbers) ==="
if "$ROOT/src/run_main.sh" "$MODE"; then RC6=0; else RC6=$?; fi

echo "============================================================"
echo "  1 run_exp6_alpha_curves      exit=$RC1"
echo "  2 run_fit_alpha_rule         exit=$RC2"
echo "  3 run_exp1_dr_benchmark      exit=$RC3"
echo "  4 run_exp3_graph_layout      exit=$RC4"
echo "  5 run_exp1_cluster_geometry  exit=$RC5"
echo "  6 run_main                   exit=$RC6"
echo "============================================================"
cat "results/data/exp6_alpha_curves_DONE.txt" 2>/dev/null || true
cat "results/data/exp1_dr_benchmark_DONE.txt" 2>/dev/null || true
cat "results/data/exp3_graph_layout_DONE.txt" 2>/dev/null || true
echo "[$(date '+%Y-%m-%d %H:%M:%S')] S2 end rc=$RC1/$RC2/$RC3/$RC4/$RC5/$RC6" >> "$LOGDIR/run_s2_all.log"
