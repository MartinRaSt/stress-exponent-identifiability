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
REM K14: detection/repair of rows damaged by an interrupted write (hard reset during
REM append_result) in results/data/<experiment>_results.csv - see
REM src/experiments/repair_csv.py. Example:
REM   src\run_repair_csv.bat --csv results/data/exp7_rank_weights_results.csv
REM   src\run_repair_csv.bat --csv results/data/exp7_rank_weights_results.csv --repair
setlocal
cd /d "%~dp0.."
REM Disable computer standby/hibernation for the duration of the run (restored at the
REM end, even after an error) - author requirement, see src/common/no_sleep_on.bat.
call "%~dp0common\no_sleep_on.bat"
"venv\python.exe" -m src.experiments.repair_csv %*
set "EXIT_CODE=%ERRORLEVEL%"
call "%~dp0common\no_sleep_off.bat"
exit /b %EXIT_CODE%
