#!/usr/bin/env bash
# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-19
# License: see the LICENSE file in the repository root
#
# Checks two "proportions" requirements (author feedback 2026-09-19, "text
# overwhelms data" after the font-size fix enlarged every label):
#   1. every figure's axes (drawing) area is >= figures.layout.
#      min_axes_area_fraction of its own canvas (results/figures/[<mode>/]
#      check_axes_area.csv, recorded live by save_figure() - run the
#      relevant fig_*.py --full first if a figure is missing/stale);
#   2. every DAMI main-text figure prints at or below figures.layout.
#      main_text_max_height_frac_textheight of \textheight (measured on the
#      actual PDF in clanek/img/).
# Prints one row per figure and writes the same table to
# results/figures/[<mode>/]check_axes_area_report.csv. Does not build/
# regenerate any figure or article PDF itself.
# Usage: check_axes_area.sh   (run from anywhere; cds into the repo root)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
venv/python.exe -m src.figures.check_axes_area
