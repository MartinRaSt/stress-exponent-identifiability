@echo off
REM Project: Sammon mapping revisited - adaptive weighting, a scalable solver
REM          and a temporal extension (alpha-Sammon)
REM Authors: Martin Radvansky <martin.radvansky@vsb.cz>
REM          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
REM Affiliation: Department of Computer Science, Faculty of Electrical Engineering
REM              and Computer Science, VSB - Technical University of Ostrava,
REM              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
REM Created: 2026-09-17
REM License: see the LICENSE file in the repository root
REM
REM fig_neighbor_survival: 2x2 dot-plot grid (regime x metric) replacing the
REM exp13_neighbor_survival main-text table -> results/figures/[<mode>/]
REM fig_neighbor_survival.pdf/.csv. Requires
REM run_exp13_neighbor_survival.bat to have already completed in the same
REM mode (reads results/tables/[<mode>/]exp13_neighbor_survival.csv).
REM Parameter: quick|full|smoke.
setlocal
cd /d "%~dp0.."
call "%~dp0common\no_sleep_on.bat"
set MODE=%1
if "%MODE%"=="" set MODE=full
"venv\python.exe" -m src.figures.fig_neighbor_survival --%MODE%
set "EXIT_CODE=%ERRORLEVEL%"
call "%~dp0common\no_sleep_off.bat"
exit /b %EXIT_CODE%
