#!/usr/bin/env bash
set -euo pipefail
# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-13
# License: see the LICENSE file in the repository root
#
# Probe of GPU memory/time for SMACOF at the real scale of E3 (pgp, n=10680,
# float64) - see src/sammon/gpu_memory_probe.py and
# documentation/2026-09-13_gpu_pametova_optimalizace.md. Output:
# results/data/quick/gpu_memory_probe.csv + results/logs/gpu_memory_probe_*.log.
# Parameters are passed through (e.g. --runs 0:guttman 1:pinv 1:resident_cg:inexact --max-iter 30);
# without parameters, sammon.gpu_memory_probe from config.yaml is used.
#
# WARNING (Linux/macOS): requires a CUDA GPU (torch cu126 on the project machine,
# RTX 3070 Ti) - on a machine without an NVIDIA GPU the script fails fail-loud.

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

export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

"$PYTHON" -m src.sammon.gpu_memory_probe "$@"
