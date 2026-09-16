@echo off
REM Project: Sammon mapping revisited - adaptive weighting, a scalable solver
REM          and a temporal extension (alpha-Sammon)
REM Authors: Martin Radvansky <martin.radvansky@vsb.cz>
REM          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
REM Affiliation: Department of Computer Science, Faculty of Electrical Engineering
REM              and Computer Science, VSB - Technical University of Ostrava,
REM              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
REM Created: 2026-09-12
REM License: see the LICENSE file in the repository root
REM
REM K1: computes properties of the E1+E3 input datasets
REM (src/experiments/dataset_properties.py) -> results/data/[<mode>/]
REM dataset_properties.csv. Parameter: quick|full|smoke (default full).
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
REM --check-correlation requires exp1_dr_benchmark to have already run in the same
REM mode (results/data/[<mode>/]exp1_dr_benchmark_results.csv) - in full mode
REM it additionally computes and prints Spearman(nn_ratio_k1, best alpha
REM from {0,1,2}); in quick/smoke mode it is skipped (data may not exist).
set CHECK_CORR=
if "%MODE%"=="full" set CHECK_CORR=--check-correlation
"venv\python.exe" -m src.experiments.dataset_properties --%MODE% %CHECK_CORR%
set "EXIT_CODE=%ERRORLEVEL%"
call "%~dp0common\no_sleep_off.bat"
exit /b %EXIT_CODE%
