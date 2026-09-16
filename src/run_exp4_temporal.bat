@echo off
REM Project: Sammon mapping revisited - adaptive weighting, a scalable solver
REM          and a temporal extension (alpha-Sammon)
REM Authors: Martin Radvansky <martin.radvansky@vsb.cz>
REM          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
REM Affiliation: Department of Computer Science, Faculty of Electrical Engineering
REM              and Computer Science, VSB - Technical University of Ostrava,
REM              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
REM Created: 2026-09-09
REM License: see the LICENSE file in the repository root
REM
REM Runs E4 (temporal alpha-Sammon, src/experiments/exp4_temporal.py).
REM Parameter: quick|full|smoke (default full). Resumable, incl. the permutation
REM test (exp4_stats.csv) and trajectories (exp4_trajectories.csv).
REM
REM 2026-09-16: E4 runs IN PARALLEL over (dataset, method, seed) on
REM parallel.n_workers workers (previously sequential). Estimate of the full grid from
REM measured times: 4.36 h sequential -> approx. 30 min on 12 workers
REM (srcun_estimate_runtime.bat exp4_temporal).
REM The BLAS thread setting below is therefore REQUIRED, not just recommended.
setlocal
cd /d "%~dp0.."
REM Disable computer standby/hibernation for the duration of the run (restored at the
REM end, even after an error) - author requirement, see src/common/no_sleep_on.bat.
call "%~dp0common\no_sleep_on.bat"
REM Limit BLAS threads to 1 per process (see config.yaml parallel.blas_threads_per_worker)
REM - with parallel.n_workers=12, each process would otherwise use more OpenBLAS/MKL
REM threads and the CPU would be overloaded (author's limit: 12 workers x 1 BLAS
REM thread, 16 workers overheated the CPU - see config.yaml parallel.n_workers).
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
