@echo off
REM Project: Sammon mapping revisited - adaptive weighting, a scalable solver
REM          and a temporal extension (alpha-Sammon)
REM Authors: Martin Radvansky <martin.radvansky@vsb.cz>
REM          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
REM Affiliation: Department of Computer Science, Faculty of Electrical Engineering
REM              and Computer Science, VSB - Technical University of Ostrava,
REM              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
REM Created: 2026-09-10
REM License: see the LICENSE file in the repository root
REM
REM Restart full runs after a code fix: (1) removes error rows (status=error)
REM from the already existing CSVs in results\data so they get recomputed, (2) runs
REM all experiments E1-E5 (resume - already completed combinations are skipped).
REM Before running, STOP any currently running run_all_experiments.bat!
setlocal
cd /d "%~dp0.."
REM Disable computer standby/hibernation for the duration of the run (restored at the
REM end, even after an error) - author requirement, see src/common/no_sleep_on.bat.
call "%~dp0common\no_sleep_on.bat"
for %%F in (results\data\exp*_results.csv) do (
    venv\python.exe -m src.experiments.clean_error_rows "%%F"
)
call src\run_all_experiments.bat full
set "EXIT_CODE=%ERRORLEVEL%"
call "%~dp0common\no_sleep_off.bat"
exit /b %EXIT_CODE%
