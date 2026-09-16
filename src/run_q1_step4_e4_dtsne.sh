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
# Q1 extension, step 4 (project-q1-extensions-20260914): external dynamic
# baseline "dynamic t-SNE" (Rauber, Falcao, Telea 2016, DOI
# 10.2312/eurovisshort.20161164) for E4 - see
# documentation/2026-09-14_e4_dtsne_baseline.md.
#
# Runs the SAME script and writes to the SAME results CSV as run_exp4_temporal.sh
# (src/experiments/exp4_temporal.py) - the dtsne_lambda<L> methods run in the
# same run/pipeline as the SMACOF/SGD Sammon family; thanks to the checkpoint
# (results/data/exp4_temporal_results.csv), already completed combinations
# (including an earlier Sammon/SGD family run) are skipped and only the
# missing ones are computed (typically the dtsne_lambda<L> rows added by this task).
# This script is just a named convenience for this specific task -
# there is no separate "dtsne only" switch in the script (see its docstring).
#
# Parameter: smoke|quick|full (default full).
#   smoke - a few frames/iterations, a small lambda_dt grid, for a quick check
#           of run/resume/schema (see VERIFICATION in the task documentation).
#   quick - a medium run, for an ETA estimate before the full run.
#   full  - a complete run over all 7 datasets x all lambda_dt seeds.
#
# The author runs the FULL run (full) manually - agents may only run smoke/quick.

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

# Limit BLAS threads to 1 per process (see config.yaml parallel.blas_threads_per_worker)
# - otherwise each process would use more OpenBLAS/MKL threads and the CPU would be
# overloaded (author's limit, see config.yaml parallel.n_workers).
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

MODE="${1:-full}"

"$PYTHON" -m src.experiments.exp4_temporal --"$MODE"
