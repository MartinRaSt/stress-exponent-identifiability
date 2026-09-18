@echo off
REM Project: Sammon mapping revisited - adaptive weighting, a scalable solver
REM          and a temporal extension (alpha-Sammon)
REM Authors: Martin Radvansky <martin.radvansky@vsb.cz>
REM          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
REM Created: 2026-09-18
REM License: see the LICENSE file in the repository root
REM
REM Writes two dated archives NEXT TO the project directory (never inside it,
REM so a backup never archives itself):
REM   Sammon_sources_<date>.zip         sources: code, article text, SVG figure
REM                                    sources, docs, git history (a few MB)
REM   Sammon_data_<date>.zip  the measured experiment CSVs only
REM Everything a script can rebuild (figures, tables, PDFs, embeddings, the
REM dataset cache, the venv) is left out - see src/tools/make_backup.py.
REM
REM Usage: src\run_backup.bat            (both archives)
REM        src\run_backup.bat sources    (sources only)
REM        src\run_backup.bat data       (measured data only)

setlocal
cd /d "%~dp0.."

for /f %%d in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd"') do set STAMP=%%d
set OUTDIR=..

set PROFILE=%1
if "%PROFILE%"=="" set PROFILE=both

if /i "%PROFILE%"=="both" goto :both
if /i "%PROFILE%"=="sources" goto :sources
if /i "%PROFILE%"=="data" goto :data
echo ERROR: unknown profile "%PROFILE%" (use: sources ^| data ^| both).
exit /b 2

:both
call :sources
if errorlevel 1 exit /b 1
call :data
exit /b %errorlevel%

:sources
echo === backup: sources ===
"venv\python.exe" -m src.tools.make_backup sources "%OUTDIR%\Sammon_sources_%STAMP%.zip"
if errorlevel 1 (
  echo ERROR: the sources backup failed.
  exit /b 1
)
exit /b 0

:data
echo === backup: measured data ===
"venv\python.exe" -m src.tools.make_backup data "%OUTDIR%\Sammon_data_%STAMP%.zip"
if errorlevel 1 (
  echo ERROR: the data backup failed.
  exit /b 1
)
exit /b 0
