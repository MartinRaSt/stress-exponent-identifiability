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
# Runs exp14 (robustness of the exp10 identifiability claims to the
# exp11-measured SMACOF convergence residual,
# documentation/2026-09-17_zadani_exp14_robustnost_konvergence.md,
# src/experiments/exp14_convergence_robustness.py) ->
# results/data/[<mode>/]exp14_convergence_robustness_results.csv +
# results/tables/[<mode>/]exp14_convergence_robustness.csv/.tex.
# Requires COMPLETED exp10_identifiability_check AND exp11_convergence_check
# AND exp10_identifiability_stats FULL runs (read regardless of the mode
# given here - see the module docstring; fail-loud on missing inputs).
# No new embedding/solver run - pure filtering/recompute, takes seconds.
# Parameter: quick|full|smoke (default full).

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

"$PYTHON" -m src.experiments.exp14_convergence_robustness --"$MODE"
