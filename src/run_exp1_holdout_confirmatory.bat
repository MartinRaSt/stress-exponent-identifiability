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
REM Q1 step 2 (A.6): confirmatory/exploratory analysis of the regime L hold-out
REM datasets (src/experiments/exp1_holdout_confirmatory.py) ->
REM results/tables/[<mode>/]exp1_holdout_confirmatory.csv/.tex. Requires
REM dataset_properties.py, exp1_dr_benchmark.py (core+holdout) and
REM screen_regime_candidates.py to have already run in the SAME mode (exp6_alpha_curves.py
REM optional, only for F_A6 - otherwise a warning in the log, the rest of the table
REM is still generated). Parameter: quick|full|smoke (default full).
setlocal
cd /d "%~dp0.."
call "%~dp0common\no_sleep_on.bat"
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
set MODE=%1
if "%MODE%"=="" set MODE=full
"venv\python.exe" -m src.experiments.exp1_holdout_confirmatory --%MODE%
set "EXIT_CODE=%ERRORLEVEL%"
call "%~dp0common\no_sleep_off.bat"
exit /b %EXIT_CODE%
