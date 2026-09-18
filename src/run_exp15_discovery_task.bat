@echo off
REM Project: Sammon mapping revisited - adaptive weighting, a scalable solver
REM          and a temporal extension (alpha-Sammon)
REM Authors: Martin Radvansky <martin.radvansky@vsb.cz>
REM          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
REM Affiliation: Department of Computer Science, Faculty of Electrical Engineering
REM              and Computer Science, VSB - Technical University of Ostrava,
REM              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
REM Created: 2026-09-18
REM License: see the LICENSE file in the repository root
REM
REM E15: downstream discovery task over the ALREADY SAVED exp1_dr_benchmark
REM embeddings (src/experiments/exp15_discovery_task.py) - NO re-run of DR methods.
REM Parameter: quick|full|smoke (default full). Requires exp1_dr_benchmark
REM to have already run in the same mode.
setlocal
cd /d "%~dp0.."
REM Disable computer standby/hibernation for the duration of the run (restored at the
REM end, even after an error) - author requirement, see src/common/no_sleep_on.bat.
call "%~dp0common\no_sleep_on.bat"
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
set MODE=%1
if "%MODE%"=="" set MODE=full
"venv\python.exe" -m src.experiments.exp15_discovery_task --%MODE%
set "EXIT_CODE=%ERRORLEVEL%"
call "%~dp0common\no_sleep_off.bat"
exit /b %EXIT_CODE%
