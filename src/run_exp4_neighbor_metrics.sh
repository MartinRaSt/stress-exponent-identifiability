#!/usr/bin/env bash
set -euo pipefail
# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-15
# License: see the LICENSE file in the repository root
#
# K15: post-hoc neighborhood preservation metrics (trustworthiness/continuity/
# kNN Jaccard) over exp4_trajectories.csv (src/experiments/exp4_neighbor_metrics.py).
# Requires a completed run of src/run_exp4_temporal.sh (MODE) - see
# documentation/2026-09-15_e4_metrika_sousedstvi.md.
# K16 (2026-09-16): the same script additionally writes a PRIMARY confirmatory test
# (exp4_neighbor_primary_stats.csv/exp4_neighbor_primary_pairs.csv, unit
# = dataset) alongside the original DESCRIPTIVE test (exp4_neighbor_stats.csv,
# column test_role, without a Holm correction across the whole family) - see
# documentation/2026-09-16_e4_hierarchicky_test.md.
# Parameter: quick|full|smoke (default full). Resumable.

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

"$PYTHON" -m src.experiments.exp4_neighbor_metrics --"$MODE"
