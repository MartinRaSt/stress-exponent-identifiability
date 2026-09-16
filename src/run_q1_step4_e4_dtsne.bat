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
REM Q1 extension, step 4 (project-q1-extensions-20260914): external dynamic
REM baseline "dynamic t-SNE" (Rauber, Falcao, Telea 2016, DOI
REM 10.2312/eurovisshort.20161164) for E4 - see
REM documentation/2026-09-14_e4_dtsne_baseline.md.
REM
REM Runs the SAME script and writes to the SAME results CSV as run_exp4_temporal.bat
REM (src/experiments/exp4_temporal.py) - the dtsne_lambda<L> methods run in the
REM same run/pipeline as the SMACOF/SGD Sammon family; thanks to the checkpoint
REM (results/data/exp4_temporal_results.csv), already completed combinations
REM (including an earlier Sammon/SGD family run) are skipped and only the
REM missing ones are computed (typically the dtsne_lambda<L> rows added by this task).
REM This .bat is just a named convenience for this specific task -
REM there is no separate "dtsne only" switch in the script (see its docstring).
REM
REM Parameter: smoke|quick|full (default full).
REM   smoke - a few frames/iterations, a small lambda_dt grid, for a quick check
REM           of run/resume/schema (see VERIFICATION in the task documentation).
REM   quick - a medium run, for an ETA estimate before the full run.
REM   full  - a complete run over all 7 datasets x all lambda_dt seeds.
REM
REM The author runs the FULL run (full) manually - agents may only run smoke/quick.
setlocal
cd /d "%~dp0.."

REM Disable computer standby/hibernation for the duration of the run (restored at the
REM end, even after an error) - author requirement, see src/common/no_sleep_on.bat.
call "%~dp0common\no_sleep_on.bat"

REM Limit BLAS threads to 1 per process (see config.yaml parallel.blas_threads_per_worker).
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1

set MODE=%1
if "%MODE%"=="" set MODE=full

"venv\python.exe" -m src.experiments.exp4_temporal --%MODE%
set "EXIT_CODE=%ERRORLEVEL%"

call "%~dp0common\no_sleep_off.bat"
exit /b %EXIT_CODE%
