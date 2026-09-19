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
# Measures the PRINTED font size (after LaTeX's [width=...] rescale) of
# every figure \includegraphics-referenced in clanek_en/sections/*.tex and
# clanek_en/supplement/sections/*.tex, and fails (non-zero exit code) if any
# non-subscript glyph is below the Springer Nature artwork floor
# (config.yaml figures.layout.annotation_min_pt). Prints one row per figure
# and writes the same table to results/figures/check_font_sizes.csv. Does
# not build/regenerate any figure or article PDF - run the relevant
# fig_*.py --full and the article build scripts first if a PDF is missing
# or stale.
# Usage: check_font_sizes.sh   (run from anywhere; cds into the repo root)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
venv/python.exe -m src.figures.check_font_sizes
