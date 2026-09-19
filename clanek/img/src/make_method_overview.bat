@echo off
REM Project: Sammon mapping revisited (alpha-Sammon, scalable solver, temporal extension)
REM Author: Martin Radvansky, VSB - Technical University of Ostrava
REM Contact: martinradvansky@gmail.com
REM Created: 2026-09-09
REM License: for research use within the project; see projectstate.md
REM
REM Regenerates clanek\img\method_overview.pdf (graphical abstract / conceptual scheme
REM of the method) from the hand-editable source clanek\img\src\method_overview.svg.
REM Converter: svglib + reportlab inside the project conda env (venv\python.exe),
REM driven by clanek\img\src\svg2pdf.py (Inkscape is not on PATH on this machine).
REM Usage: make_method_overview.bat   (from any directory; paths are resolved relative
REM        to this file: <repo>\clanek\img\src\)
setlocal

set "SRC_DIR=%~dp0"
set "REPO=%SRC_DIR%..\..\.."
set "PY=%REPO%\venv\python.exe"
set "SVG=%SRC_DIR%method_overview.svg"
set "PDF=%SRC_DIR%..\method_overview.pdf"

if not exist "%PY%" (
  echo ERROR: Python interpreter not found: "%PY%"
  exit /b 1
)
if not exist "%SVG%" (
  echo ERROR: SVG source not found: "%SVG%"
  exit /b 1
)

echo [method_overview] converting SVG to PDF ...
set PYTHONIOENCODING=utf-8
"%PY%" "%SRC_DIR%svg2pdf.py" "%SVG%" "%PDF%"
if errorlevel 1 (
  echo ERROR: SVG to PDF conversion failed for method_overview.svg
  exit /b 1
)
echo [method_overview] done: "%PDF%"
exit /b 0
