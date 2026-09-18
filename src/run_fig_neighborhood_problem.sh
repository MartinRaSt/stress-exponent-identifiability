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
# fig_neighborhood_problem: flagship 3-panel figure (MDS / tuned
# alpha-Sammon / t-SNE, original-space k-NN edges) -> results/figures/
# [<mode>/]fig_neighborhood_problem.pdf/.csv. Requires
# run_exp6_alpha_curves.sh, run_exp12_alpha_grid_extension.sh and
# run_exp1_dr_benchmark.sh to have already completed in the same mode.
# NOTE: --quick has no 'helix'/'s_curve' data in exp1_dr_benchmark.quick/
# exp6_alpha_curves.quick - only smoke/full are usable (see
# fig_neighborhood_problem.py's module docstring). Parameter: quick|full|smoke.

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

MODE="${1:-full}"

"$PYTHON" -m src.figures.fig_neighborhood_problem --"$MODE"
