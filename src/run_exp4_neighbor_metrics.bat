@echo off
REM Project: Sammon mapping revisited - adaptive weighting, a scalable solver
REM          and a temporal extension (alpha-Sammon)
REM Authors: Martin Radvansky <martin.radvansky@vsb.cz>
REM          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
REM Affiliation: Department of Computer Science, Faculty of Electrical Engineering
REM              and Computer Science, VSB - Technical University of Ostrava,
REM              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
REM Created: 2026-09-15
REM License: see the LICENSE file in the repository root
REM
REM K15: post-hoc neighborhood preservation metrics (trustworthiness/continuity/
REM kNN Jaccard) over exp4_trajectories.csv (src/experiments/exp4_neighbor_metrics.py).
REM Requires a completed run of src\run_exp4_temporal.bat (%MODE%) - see
REM documentation/2026-09-15_e4_metrika_sousedstvi.md.
REM K16 (2026-09-16): the same script additionally writes a PRIMARY confirmatory test
REM (exp4_neighbor_primary_stats.csv/exp4_neighbor_primary_pairs.csv, unit
REM = dataset) alongside the original DESCRIPTIVE test (exp4_neighbor_stats.csv,
REM column test_role, without a Holm correction across the whole family) - see
REM documentation/2026-09-16_e4_hierarchicky_test.md.
REM Parameter: quick|full|smoke (default full). Resumable.
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
"venv\python.exe" -m src.experiments.exp4_neighbor_metrics --%MODE%
set "EXIT_CODE=%ERRORLEVEL%"
call "%~dp0common\no_sleep_off.bat"
exit /b %EXIT_CODE%
