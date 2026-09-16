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
REM Probe of init_classical_mds timing (full eigh vs. iterative eigsh) on the cached
REM pgp distance matrix (n=10680) - see src/sammon/init_probe.py and
REM documentation/2026-09-13_hardening_behu.md. Output:
REM results/data/quick/init_probe.csv + results/logs/init_probe_*.log.
REM Parameters are passed through (e.g. --methods eigsh --threads 1 8); without
REM parameters, sammon.init_probe from config.yaml is used. The number of BLAS threads is
REM controlled by the script itself via threadpoolctl (env variables are not needed here).
setlocal
cd /d "%~dp0.."
REM Disable computer standby/hibernation for the duration of the run (restored at the
REM end, even after an error) - author requirement, see src/common/no_sleep_on.bat.
call "%~dp0common\no_sleep_on.bat"
"venv\python.exe" -m src.sammon.init_probe %*
set "EXIT_CODE=%ERRORLEVEL%"
call "%~dp0common\no_sleep_off.bat"
exit /b %EXIT_CODE%
