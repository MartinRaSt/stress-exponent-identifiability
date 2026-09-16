@echo off
REM Project: Sammon mapping revisited - adaptive weighting, a scalable solver
REM          and a temporal extension (alpha-Sammon)
REM Authors: Martin Radvansky <martin.radvansky@vsb.cz>
REM          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
REM Affiliation: Department of Computer Science, Faculty of Electrical Engineering
REM              and Computer Science, VSB - Technical University of Ostrava,
REM              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
REM Created: 2026-09-12
REM License: see the LICENSE file in the repository root
REM
REM K12a: generates clanek/generated/numbers.tex (\newcommand macros for
REM all article numbers, see src/experiments/export_numbers.py). Always reads
REM the production ("full") data -> clanek/generated/; parameter quick|smoke reads and writes
REM EXCLUSIVELY results/data/<mode>/ -> results/tables/<mode>/numbers.tex.
setlocal
cd /d "%~dp0.."
call "%~dp0common\no_sleep_on.bat"
set MODE=%1
if "%MODE%"=="" set MODE=full
"venv\python.exe" -m src.experiments.export_numbers --%MODE%
set "EXIT_CODE=%ERRORLEVEL%"
call "%~dp0common\no_sleep_off.bat"
exit /b %EXIT_CODE%
