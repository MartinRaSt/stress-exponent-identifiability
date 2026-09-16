@echo off
REM Project: Sammon mapping revisited - adaptive weighting, a scalable solver
REM          and a temporal extension (alpha-Sammon)
REM Authors: Martin Radvansky <martin.radvansky@vsb.cz>
REM          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
REM Affiliation: Department of Computer Science, Faculty of Electrical Engineering
REM              and Computer Science, VSB - Technical University of Ostrava,
REM              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
REM Created: 2026-09-16
REM License: see the LICENSE file in the repository root
REM
REM Estimate the length of a full experiment run from the MEASURED times (wall_time_sec) in
REM an already existing results CSV. Parameter: experiment name (default
REM exp4_temporal), optionally a list of worker counts.
REM Example: src\run_estimate_runtime.bat exp4_temporal 1,8,12
REM Just reads a CSV (seconds) - sleep inhibition is not needed here.
setlocal
cd /d "%~dp0.."
set "EXPNAME=%~1"
if "%EXPNAME%"=="" set "EXPNAME=exp4_temporal"
set "WORKERS=%~2"
REM intentionally without a block if ( ... ) else ( ... ): inside the parentheses
REM cmd treats the comma in the worker list as a separator and the script would fail with
REM "8 was unexpected at this time".
if not defined WORKERS goto :no_list
"venv\python.exe" -m src.tools.estimate_runtime_from_csv %EXPNAME% --workers "%WORKERS%"
goto :done
:no_list
"venv\python.exe" -m src.tools.estimate_runtime_from_csv %EXPNAME%
:done
exit /b %ERRORLEVEL%
