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
REM Q1-step3 (reserse/2026-09-14_specifikace_rozsireni_q1.md PART B) -
REM metric fidelity task with known ground truth: grid of scenario (S1/S2/S3) x
REM replicate x method -> results/data/[<mode>/]exp9_metric_fidelity_results.csv
REM (+ embeddings/truth npz) + _DONE.txt. Resumable (own checkpoint
REM keyed by (scenario,replicate,method), see exp9_metric_fidelity.py).
REM Parameter: quick|full|smoke (default full).
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
"venv\python.exe" -m src.experiments.exp9_metric_fidelity --%MODE%
set "EXIT_CODE=%ERRORLEVEL%"
call "%~dp0common\no_sleep_off.bat"
exit /b %EXIT_CODE%
