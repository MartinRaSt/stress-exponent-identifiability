@echo off
REM Project: Sammon mapping revisited - adaptive weighting, a scalable solver
REM          and a temporal extension (alpha-Sammon)
REM Authors: Martin Radvansky <martin.radvansky@vsb.cz>
REM          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
REM Affiliation: Department of Computer Science, Faculty of Electrical Engineering
REM              and Computer Science, VSB - Technical University of Ostrava,
REM              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
REM Created: 2026-09-14
REM License: see the LICENSE file in the repository root
REM
REM fig_metric_fidelity_boxes: boxplots of the primary metric (S1/S2/S3) across
REM replicates, method x scenario, with an oracle_truth line -> results/figures/
REM [<mode>/]fig_metric_fidelity_boxes.pdf/.csv. Requires
REM run_exp9_metric_fidelity.bat to have already completed in the same mode. Parameter: quick|full|smoke.
setlocal
cd /d "%~dp0.."
call "%~dp0common\no_sleep_on.bat"
set MODE=%1
if "%MODE%"=="" set MODE=full
"venv\python.exe" -m src.figures.fig_metric_fidelity_boxes --%MODE%
set "EXIT_CODE=%ERRORLEVEL%"
call "%~dp0common\no_sleep_off.bat"
exit /b %EXIT_CODE%
