@echo off
REM Project: Sammon mapping revisited - adaptive weighting, a scalable solver
REM          and a temporal extension (alpha-Sammon)
REM Authors: Martin Radvansky <martin.radvansky@vsb.cz>
REM          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
REM Affiliation: Department of Computer Science, Faculty of Electrical Engineering
REM              and Computer Science, VSB - Technical University of Ostrava,
REM              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
REM Created: 2026-09-17
REM License: see the LICENSE file in the repository root
REM
REM Pre-submission check: verifies that every \supref{N}/\suprefs{N}{M} in
REM clanek/sections/*.tex points at a section that actually exists in
REM clanek/supplement/supplement.tex, and prints the title of the target
REM section next to each reference for a human sanity check. Exit code 1 on
REM any failure. See src/validate_supplement_refs.py.
setlocal
cd /d "%~dp0.."
"venv\python.exe" -m src.validate_supplement_refs
exit /b %ERRORLEVEL%
