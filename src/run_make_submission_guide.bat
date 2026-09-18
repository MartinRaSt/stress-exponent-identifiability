@echo off
REM Project: Sammon mapping revisited - adaptive weighting, a scalable solver
REM          and a temporal extension (alpha-Sammon)
REM Authors: Martin Radvansky <martin.radvansky@vsb.cz>
REM          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
REM Affiliation: Department of Computer Science, Faculty of Electrical Engineering
REM              and Computer Science, VSB - Technical University of Ostrava,
REM              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
REM Created: 2026-09-18
REM License: see the LICENSE file in the repository root
REM
REM Generates dami_submission/01_TEXTY_DO_FORMULARE.md and
REM dami_submission/02_COVER_LETTER.md from the CURRENT
REM clanek_en/dami/main_dami.tex and clanek/generated/numbers.tex - see
REM src/tools/make_submission_guide.py for the full description. Run this
REM ONLY once the manuscript is final (author's explicit instruction: these
REM two files are deliberately left empty in dami_submission/ until then,
REM see dami_submission/00_POSTUP_SUBMISSION.md).
REM
REM Requires: clanek_en/dami/main_dami.tex up to date and
REM clanek/generated/numbers.tex already generated (src/run_main.bat).
REM
REM Usage: src\run_make_submission_guide.bat <ZENODO-DOI> [output-dir]
REM   e.g.: src\run_make_submission_guide.bat 10.5281/zenodo.1234567
setlocal
cd /d "%~dp0.."
REM Disable computer standby/hibernation for the duration of the run (restored at the
REM end, even after an error) - author requirement, see src/common/no_sleep_on.bat.
call "%~dp0common\no_sleep_on.bat"

set ZENODO_DOI=%1
set OUTDIR=%2

if "%ZENODO_DOI%"=="" (
  echo [run_make_submission_guide] ERROR: missing required argument ZENODO-DOI.
  echo Usage: src\run_make_submission_guide.bat ^<ZENODO-DOI^> [output-dir]
  call "%~dp0common\no_sleep_off.bat"
  exit /b 1
)

if "%OUTDIR%"=="" (
  "venv\python.exe" -m src.tools.make_submission_guide "%ZENODO_DOI%"
) else (
  "venv\python.exe" -m src.tools.make_submission_guide "%ZENODO_DOI%" --output-dir "%OUTDIR%"
)
set "EXIT_CODE=%ERRORLEVEL%"

call "%~dp0common\no_sleep_off.bat"
exit /b %EXIT_CODE%
