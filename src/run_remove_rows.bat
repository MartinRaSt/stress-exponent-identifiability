@echo off
REM Project: Sammon mapping revisited - adaptive weighting, a scalable solver
REM          and a temporal extension (alpha-Sammon)
REM Authors: Martin Radvansky <martin.radvansky@vsb.cz>
REM          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
REM Affiliation: Department of Computer Science, Faculty of Electrical Engineering
REM              and Computer Science, VSB - Technical University of Ostrava,
REM              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
REM Created: 2026-09-11
REM License: see the LICENSE file in the repository root
REM
REM Passes all arguments to src/experiments/remove_rows.py (selective
REM removal of rows from the experiment results CSV + the corresponding embeddings,
REM for a rerun after a method/dataset fix). Example:
REM   src\run_remove_rows.bat --csv results/data/exp1_dr_benchmark_results.csv --dataset wine --dry-run
setlocal
cd /d "%~dp0.."
REM Disable computer standby/hibernation for the duration of the run (restored at the
REM end, even after an error) - author requirement, see src/common/no_sleep_on.bat.
call "%~dp0common\no_sleep_on.bat"
"venv\python.exe" -m src.experiments.remove_rows %*
set "EXIT_CODE=%ERRORLEVEL%"
call "%~dp0common\no_sleep_off.bat"
exit /b %EXIT_CODE%
