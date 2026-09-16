#!/usr/bin/env bash
set -euo pipefail
# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-16
# License: see the LICENSE file in the repository root
#
# fig_metric_correlations: Spearman correlations of embedding quality metrics over
# the E1 results (all methods, stress family, neighborhood methods) ->
# results/figures/[<mode>/]fig_metric_correlations.pdf/.csv + _clusters.csv.
# Basis for selecting the main table's columns (Bae et al. 2025, DOI
# 10.1109/VIS60296.2025.00014). Fast aggregation over already finished E1 data.
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

MODE="${1:-full}"

"$PYTHON" -m src.figures.fig_metric_correlations --"$MODE"
