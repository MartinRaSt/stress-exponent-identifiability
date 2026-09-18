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
REM exp10_identifiability_stats: analyses left out of
REM exp10_identifiability_check.py (reserse/2026-09-17_zostreni_propozice2.md
REM section 10, items 2-6): per-dataset table (results/tables/[<mode>/]
REM exp10_identifiability_table.csv/.tex) + regression/H1/H2/H3/quadratic-law
REM stats (results/data/[<mode>/]exp10_identifiability_stats.csv). Fast run
REM (a few seconds/minutes, no new DR/embedding computation). Requires
REM run_exp10_identifiability_check.bat (and, for the regression,
REM run_exp8_prop2_check.bat) to have already completed in the same mode.
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
"venv\python.exe" -m src.experiments.exp10_identifiability_stats --%MODE%
set "EXIT_CODE=%ERRORLEVEL%"
call "%~dp0common\no_sleep_off.bat"
exit /b %EXIT_CODE%
