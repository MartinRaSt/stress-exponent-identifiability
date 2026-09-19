@echo off
REM Project: Sammon mapping revisited (alpha-Sammon, scalable solver, temporal extension)
REM Author: Martin Radvansky, VSB - Technical University of Ostrava
REM Contact: martinradvansky@gmail.com
REM Created: 2026-09-17
REM License: for research use within the project; see projectstate.md
REM
REM Regenerates clanek\img\study_design.pdf (process diagram of the STUDY
REM DESIGN: core/hold-out split, embedding grid, metrics, rule fitting,
REM confirmatory test) from the hand-editable source
REM clanek\img\src\study_design.svg.
REM Converter: svglib + reportlab inside the project conda env (venv\python.exe),
REM driven by clanek\img\src\svg2pdf.py (Inkscape is not on PATH on this machine).
REM Usage: make_study_design.bat   (from any directory; paths are resolved
REM        relative to this file: <repo>\clanek\img\src\)
setlocal

set "SRC_DIR=%~dp0"
set "REPO=%SRC_DIR%..\..\.."
set "PY=%REPO%\venv\python.exe"
set "SVG=%SRC_DIR%study_design.svg"
set "PDF=%SRC_DIR%..\study_design.pdf"

if not exist "%PY%" (
  echo ERROR: Python interpreter not found: "%PY%"
  exit /b 1
)
if not exist "%SVG%" (
  echo ERROR: SVG source not found: "%SVG%"
  exit /b 1
)

echo [study_design] converting SVG to PDF ...
set PYTHONIOENCODING=utf-8
"%PY%" "%SRC_DIR%svg2pdf.py" "%SVG%" "%PDF%"
if errorlevel 1 (
  echo ERROR: SVG to PDF conversion failed for study_design.svg
  exit /b 1
)
echo [study_design] done: "%PDF%"
exit /b 0
