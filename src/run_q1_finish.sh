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
# Runs the REMAINING steps 9-11 of the run_q1_step2_datasets.sh chain, which did
# not run due to the failure of step 8 (alternative='tost' in sign_flip_test,
# fixed 2026-09-14). Assumes the following are DONE:
#   - run_q1_step2_datasets.sh full  (steps 1-7)
#   - exp1_holdout_confirmatory      (step 8, recomputed after the fix)
#   - run_q1_step3_metric_fidelity.sh full
#
# All of these are fast analyses and figures over already finished CSVs - no new DR
# computation, on the order of minutes. Once finished, all data and all macros
# in clanek/generated/numbers.tex are up to date.
#
# Parameter: smoke|quick|full (default full).
# Can be run from anywhere (even from the src folder) - the script switches itself
# to the project root.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-full}"

LOGDIR="results/logs"
mkdir -p "$LOGDIR"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_q1_finish start (mode=$MODE)" >> "$LOGDIR/run_q1_finish.log"

# Disable computer standby/hibernation for the duration of the run (systemd-inhibit, see src/common/no_sleep.sh).
source "$ROOT/src/common/no_sleep.sh"
sammon_no_sleep_reexec "$0" "$@"

PYTHON="${PYTHON:-$ROOT/venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: project venv Python interpreter not found: $PYTHON" >&2
    echo "Set the PYTHON environment variable to the project's venv Python path, e.g.:" >&2
    echo "  PYTHON=/path/to/venv/bin/python $0 $*" >&2
    exit 1
fi

# Limit BLAS threads to 1 per process (config.yaml parallel.blas_threads_per_worker).
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

RC1=""; RC2=""; RC3=""; RC4=""

echo "=== 1/4 power_analysis_regime ==="
if "$PYTHON" -m src.experiments.power_analysis_regime --"$MODE"; then
    RC1=0
else
    RC1=$?
    echo "ERROR: power_analysis_regime finished with code $RC1 - stopping."
fi

if [ "$RC1" = "0" ]; then
    echo "=== 2/4 fig_holdout_paired ==="
    if "$PYTHON" -m src.figures.fig_holdout_paired --"$MODE"; then
        RC2=0
    else
        RC2=$?
        echo "ERROR: fig_holdout_paired finished with code $RC2 - stopping."
    fi
fi

if [ "$RC1" = "0" ] && [ "$RC2" = "0" ]; then
    echo "=== 3/4 fig_regime_map ==="
    if "$PYTHON" -m src.figures.fig_regime_map --"$MODE"; then
        RC3=0
    else
        RC3=$?
        echo "ERROR: fig_regime_map finished with code $RC3 - stopping."
    fi
fi

echo "=== 4/4 run_main.sh $MODE --no-figures ==="
if "$ROOT/src/run_main.sh" "$MODE" --no-figures; then RC4=0; else RC4=$?; fi

echo "============================================================"
echo "run_q1_finish (mode=$MODE) summary:"
echo "  1 power_analysis_regime  : $RC1"
echo "  2 fig_holdout_paired     : $RC2"
echo "  3 fig_regime_map         : $RC3"
echo "  4 run_main --no-figures  : $RC4"
echo "============================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_q1_finish done (RC=$RC1/$RC2/$RC3/$RC4)" >> "$LOGDIR/run_q1_finish.log"
