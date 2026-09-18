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
REM Builds the FLAT (single-directory, no subfolders) journal-submission
REM packages required by Springer/DAMI from the split clanek_en/ tree - see
REM src/tools/make_submission.py for the full description. Writes
REM submission/dami_en/ (main article, sn-jnl class) and
REM submission/supplement_en/ (elsarticle supplement, cross-referenced into
REM the main text) and VERIFIES both with a full LaTeX compile against the
REM existing reference PDFs (clanek_en/dami/main_dami.pdf,
REM clanek_en/supplement/supplement.pdf) - page count and absence of
REM "undefined"/"Overfull"/"invalid in math mode"/"Author undefined" in the
REM log. Fails loud (non-zero exit) on any mismatch; the packages must not
REM be submitted if this script reports an error.
REM
REM Requires: clanek_en/dami/main_dami.pdf and clanek_en/supplement/supplement.pdf
REM already compiled once (build_dami.bat / compile.bat) as the reference to
REM check against, and clanek/generated/numbers.tex already generated
REM (src/run_main.bat).
REM
REM Usage: src\run_make_submission.bat [dami|supplement|both] [output-root]
REM   (default: both, submission\ next to this project)
setlocal
cd /d "%~dp0.."
REM Disable computer standby/hibernation for the duration of the run (restored at the
REM end, even after an error) - author requirement, see src/common/no_sleep_on.bat.
call "%~dp0common\no_sleep_on.bat"

set VARIANT=%1
if "%VARIANT%"=="" set VARIANT=both
set OUTROOT=%2

if "%OUTROOT%"=="" (
  "venv\python.exe" -m src.tools.make_submission --variant %VARIANT%
) else (
  "venv\python.exe" -m src.tools.make_submission --variant %VARIANT% --output-root "%OUTROOT%"
)
set "EXIT_CODE=%ERRORLEVEL%"

call "%~dp0common\no_sleep_off.bat"
exit /b %EXIT_CODE%
