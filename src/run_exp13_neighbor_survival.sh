#!/usr/bin/env bash
set -euo pipefail
# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-17
# License: see the LICENSE file in the repository root
#
# Runs exp13 (how many of the k nearest neighbors survive the 2D projection,
# documentation/2026-09-17_zadani_exp13_preziti_sousedu.md,
# src/experiments/exp13_neighbor_survival.py) ->
# results/data/[<mode>/]exp13_neighbor_survival_results.csv + _DONE.txt +
# results/tables/[<mode>/]exp13_neighbor_survival.csv/.tex.
# NO new DR computation - only loads cached exp6_alpha_curves embeddings.
# Requires COMPLETED dataset_properties.py AND exp6_alpha_curves runs in the
# SAME mode (otherwise fail-loud on missing inputs). Parameters:
#   run_exp13_neighbor_survival.sh [quick|full|smoke] [--datasets all|core|holdout]
# --datasets (default 'all' = 51 datasets, core+holdout) is passed through unchanged.

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
# additional parameters (--datasets all|core|holdout) are passed through
EXTRA=("${@:2}")

"$PYTHON" -m src.experiments.exp13_neighbor_survival --"$MODE" "${EXTRA[@]}"
