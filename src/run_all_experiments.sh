#!/usr/bin/env bash
set -euo pipefail
# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-09
# License: see the LICENSE file in the repository root
#
# Runs E1-E5 sequentially. Parameter: quick|full|smoke (default full).
# Continues even if one of the experiments fails (each individual
# dataset/method/seed run is already handled inside the experiment with try/except - this
# adds extra protection against a crash of the WHOLE experiment, e.g. a wrong configuration).
# At the end prints a summary (exit code of each experiment + content of the DONE file).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Disable computer standby/hibernation for the duration of the run (systemd-inhibit, see
# src/common/no_sleep.sh) - author requirement. Each individual run_expN_*.sh
# also sets the inhibition itself (nesting is safe - see the comment in no_sleep.sh).
source "$ROOT/src/common/no_sleep.sh"
sammon_no_sleep_reexec "$0" "$@"

# Limit BLAS threads to 1 per process (see config.yaml parallel.blas_threads_per_worker)
# - otherwise each process would use more OpenBLAS/MKL threads and the CPU would be
# overloaded (author's limit, see config.yaml parallel.n_workers). Each
# called sub-script (run_expN_*.sh) also sets these variables itself,
# this is just an extra layer of safety on direct invocation.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

MODE="${1:-full}"

echo "============================================================"
echo "Running all experiments E1-E5 in mode: $MODE"
echo "============================================================"

# Order from shortest to longest (changed 2026-09-11 at the author's request):
# E5 (minutes) -> E4 -> E3 -> E2 (hours) -> E1 (longest, overnight).
# Every experiment has resume (only computes combinations missing from the CSV).
if "$ROOT/src/run_exp5_ablation.sh" "$MODE"; then RC5=0; else RC5=$?; fi
if "$ROOT/src/run_exp4_temporal.sh" "$MODE"; then RC4=0; else RC4=$?; fi
if "$ROOT/src/run_exp3_graph_layout.sh" "$MODE"; then RC3=0; else RC3=$?; fi
if "$ROOT/src/run_exp2_solver_scaling.sh" "$MODE"; then RC2=0; else RC2=$?; fi
if "$ROOT/src/run_exp1_dr_benchmark.sh" "$MODE"; then RC1=0; else RC1=$?; fi

echo "============================================================"
echo "Run summary (mode $MODE):"
echo "  E1 dr_benchmark   exit=$RC1"
echo "  E2 solver_scaling exit=$RC2"
echo "  E3 graph_layout   exit=$RC3"
echo "  E4 temporal       exit=$RC4"
echo "  E5 ablation       exit=$RC5"
echo "============================================================"
echo "DONE files (see results/data/ or results/data/smoke/ for --smoke):"
if [ "$MODE" = "smoke" ]; then
    cat "results/data/smoke/exp1_dr_benchmark_DONE.txt" 2>/dev/null || true
    cat "results/data/smoke/exp2_solver_scaling_DONE.txt" 2>/dev/null || true
    cat "results/data/smoke/exp3_graph_layout_DONE.txt" 2>/dev/null || true
    cat "results/data/smoke/exp4_temporal_DONE.txt" 2>/dev/null || true
    cat "results/data/smoke/exp5_ablation_DONE.txt" 2>/dev/null || true
else
    cat "results/data/exp1_dr_benchmark_DONE.txt" 2>/dev/null || true
    cat "results/data/exp2_solver_scaling_DONE.txt" 2>/dev/null || true
    cat "results/data/exp3_graph_layout_DONE.txt" 2>/dev/null || true
    cat "results/data/exp4_temporal_DONE.txt" 2>/dev/null || true
    cat "results/data/exp5_ablation_DONE.txt" 2>/dev/null || true
fi
echo "============================================================"
