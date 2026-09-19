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
REM Checks two "proportions" requirements (author feedback 2026-09-19,
REM "text overwhelms data" after the font-size fix enlarged every label):
REM   1. every figure's axes (drawing) area is >= figures.layout.
REM      min_axes_area_fraction of its own canvas (results/figures/
REM      [<mode>/]check_axes_area.csv, recorded live by save_figure() -
REM      run the relevant fig_*.py --full first if a figure is missing/stale);
REM   2. every DAMI main-text figure prints at or below
REM      figures.layout.main_text_max_height_frac_textheight of \textheight
REM      (measured on the actual PDF in clanek/img/).
REM Prints one row per figure and writes the same table to
REM results/figures/[<mode>/]check_axes_area_report.csv. Does not build/
REM regenerate any figure or article PDF itself.
REM Usage: check_axes_area.bat   (from any directory)
setlocal
cd /d "%~dp0.."
"venv\python.exe" -m src.figures.check_axes_area
exit /b %ERRORLEVEL%
