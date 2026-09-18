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
REM Runs exp13 (how many of the k nearest neighbors survive the 2D projection,
REM documentation/2026-09-17_zadani_exp13_preziti_sousedu.md,
REM src/experiments/exp13_neighbor_survival.py) ->
REM results/data/[<mode>/]exp13_neighbor_survival_results.csv + _DONE.txt +
REM results/tables/[<mode>/]exp13_neighbor_survival.csv/.tex.
REM NO new DR computation - only loads cached exp6_alpha_curves embeddings.
REM Requires COMPLETED dataset_properties.py AND exp6_alpha_curves runs in the
REM SAME mode (otherwise fail-loud on missing inputs). Parameters:
REM   run_exp13_neighbor_survival.bat [quick|full|smoke] [--datasets all|core|holdout]
REM --datasets (default 'all' = 51 datasets, core+holdout) is passed through unchanged.
setlocal
cd /d "%~dp0.."
REM Disable computer standby/hibernation for the duration of the run (restored at the
REM end, even after an error) - author requirement, see src/common/no_sleep_on.bat.
call "%~dp0common\no_sleep_on.bat"
REM Limit BLAS threads to 1 per process (see config.yaml parallel.blas_threads_per_worker)
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
set MODE=%1
if "%MODE%"=="" set MODE=full
REM additional parameters (--datasets all|core|holdout) are passed through
set EXTRA=%2 %3
"venv\python.exe" -m src.experiments.exp13_neighbor_survival --%MODE% %EXTRA%
set "EXIT_CODE=%ERRORLEVEL%"
call "%~dp0common\no_sleep_off.bat"
exit /b %EXIT_CODE%
