@echo off
REM Project: Sammon mapping revisited - adaptive weighting, a scalable solver
REM          and a temporal extension (alpha-Sammon)
REM Authors: Martin Radvansky <martin.radvansky@vsb.cz>
REM          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
REM Affiliation: Department of Computer Science, Faculty of Electrical Engineering
REM              and Computer Science, VSB - Technical University of Ostrava,
REM              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
REM Created: 2026-09-19
REM License: see the LICENSE file in the repository root
REM
REM Measures the PRINTED font size (after LaTeX's [width=...] rescale) of
REM every figure \includegraphics-referenced in clanek_en/sections/*.tex and
REM clanek_en/supplement/sections/*.tex, and fails (non-zero exit code) if
REM any non-subscript glyph is below the Springer Nature artwork floor
REM (config.yaml figures.layout.annotation_min_pt). Prints one row per
REM figure and writes the same table to
REM results/figures/check_font_sizes.csv. Does not build/regenerate any
REM figure or article PDF - run the relevant fig_*.py --full and the
REM article build scripts first if a PDF is missing or stale.
REM Usage: check_font_sizes.bat   (from any directory)
setlocal
cd /d "%~dp0.."
"venv\python.exe" -m src.figures.check_font_sizes
exit /b %ERRORLEVEL%
