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
REM Q1-step3 - statistics (families F_B1/F_B2/F_B3, B.5) over the already finished
REM exp9_metric_fidelity_results.csv -> results/tables/[<mode>/]
REM exp9_metric_fidelity_stats.csv + .tex. Fast run (a few seconds).
REM Requires run_exp9_metric_fidelity.bat to have already completed in the same mode.
REM Parameter: quick|full|smoke (default full).
setlocal
cd /d "%~dp0.."
call "%~dp0common\no_sleep_on.bat"
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
set MODE=%1
if "%MODE%"=="" set MODE=full
"venv\python.exe" -m src.experiments.exp9_metric_fidelity_stats --%MODE%
set "EXIT_CODE=%ERRORLEVEL%"
call "%~dp0common\no_sleep_off.bat"
exit /b %EXIT_CODE%
