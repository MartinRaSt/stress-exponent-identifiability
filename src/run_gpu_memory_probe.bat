@echo off
REM Project: Sammon mapping revisited - adaptive weighting, a scalable solver
REM          and a temporal extension (alpha-Sammon)
REM Authors: Martin Radvansky <martin.radvansky@vsb.cz>
REM          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
REM Affiliation: Department of Computer Science, Faculty of Electrical Engineering
REM              and Computer Science, VSB - Technical University of Ostrava,
REM              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
REM Created: 2026-09-13
REM License: see the LICENSE file in the repository root
REM
REM Probe of GPU memory/time for SMACOF at the real scale of E3 (pgp, n=10680,
REM float64) - see src/sammon/gpu_memory_probe.py and
REM documentation/2026-09-13_gpu_pametova_optimalizace.md. Output:
REM results/data/quick/gpu_memory_probe.csv + results/logs/gpu_memory_probe_*.log.
REM Parameters are passed through (e.g. --runs 0:guttman 1:pinv 1:resident_cg:inexact --max-iter 30);
REM without parameters, sammon.gpu_memory_probe from config.yaml is used.
setlocal
cd /d "%~dp0.."
REM Disable computer standby/hibernation for the duration of the run (restored at the
REM end, even after an error) - author requirement, see src/common/no_sleep_on.bat.
call "%~dp0common\no_sleep_on.bat"
set OPENBLAS_NUM_THREADS=8
set OMP_NUM_THREADS=8
set MKL_NUM_THREADS=8
set NUMEXPR_NUM_THREADS=8
"venv\python.exe" -m src.sammon.gpu_memory_probe %*
set "EXIT_CODE=%ERRORLEVEL%"
call "%~dp0common\no_sleep_off.bat"
exit /b %EXIT_CODE%
