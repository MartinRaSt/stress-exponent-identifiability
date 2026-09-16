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
REM Q1 step 2 (A.7): power analysis of the confirmatory H1 test (regime L)
REM (src/experiments/power_analysis_regime.py) ->
REM results/tables/[<mode>/]power_analysis_regime.csv. Parameter:
REM quick|full|smoke (default full) - the only difference between modes is n_boot for the
REM Monte Carlo (see config_experiments.yaml power_analysis_regime); the input
REM data (10 observed Delta) are ALWAYS the same (independent of other
REM experiments - no dependency on a completed E1/E6).
setlocal
cd /d "%~dp0.."
call "%~dp0common\no_sleep_on.bat"
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
set MODE=%1
if "%MODE%"=="" set MODE=full
"venv\python.exe" -m src.experiments.power_analysis_regime --%MODE%
set "EXIT_CODE=%ERRORLEVEL%"
call "%~dp0common\no_sleep_off.bat"
exit /b %EXIT_CODE%
