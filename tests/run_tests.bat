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
REM Runs the full test suite (tests/test_metrics.py, test_datasets.py, test_methods.py).
setlocal
cd /d "%~dp0.."
"venv\python.exe" -m pytest tests -q
endlocal
