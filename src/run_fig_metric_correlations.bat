@echo off
REM Project: Sammon mapping revisited - adaptive weighting, a scalable solver
REM          and a temporal extension (alpha-Sammon)
REM Authors: Martin Radvansky <martin.radvansky@vsb.cz>
REM          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
REM Affiliation: Department of Computer Science, Faculty of Electrical Engineering
REM              and Computer Science, VSB - Technical University of Ostrava,
REM              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
REM Created: 2026-09-16
REM License: see the LICENSE file in the repository root
REM
REM fig_metric_correlations: Spearman correlations of embedding quality metrics over
REM the E1 results (all methods, stress family, neighborhood methods) ->
REM results/figures/[<mode>/]fig_metric_correlations.pdf/.csv + _clusters.csv.
REM Basis for selecting the main table's columns (Bae et al. 2025, DOI
REM 10.1109/VIS60296.2025.00014). Fast aggregation over already finished E1 data.
REM Parameter: quick|full|smoke (default full).
setlocal
cd /d "%~dp0.."
call "%~dp0common\no_sleep_on.bat"
set MODE=%1
if "%MODE%"=="" set MODE=full
"venv\python.exe" -m src.figures.fig_metric_correlations --%MODE%
set "EXIT_CODE=%ERRORLEVEL%"
call "%~dp0common\no_sleep_off.bat"
exit /b %EXIT_CODE%
