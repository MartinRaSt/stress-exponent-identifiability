@echo off
REM Project: Sammon mapping revisited (alpha-Sammon, scalable solver, temporal extension)
REM Author: Martin Radvansky, VSB - Technical University of Ostrava
REM Contact: martinradvansky@gmail.com
REM Created: 2026-09-09
REM License: for research use within the project; see projectstate.md
REM
REM Regenerates ALL article figures in clanek\img\ by calling the per-figure
REM build scripts. Schematic (SVG) figures live in clanek\img\src\make_<name>.bat;
REM data-derived figures (produced by python-coder) are appended below as they
REM are added, one line per figure, in the same call pattern.
REM Stops at the first failing figure (non-zero exit code) so that errors are
REM never silently skipped.
REM Usage: build_figures.bat   (from any directory)
setlocal

set "IMG_DIR=%~dp0"
set "SRC_DIR=%IMG_DIR%src\"

echo ========================================
echo Building article figures
echo ========================================

REM --- schematic figures (hand-editable SVG -> PDF) -------------------------
call "%SRC_DIR%make_method_overview.bat"
if errorlevel 1 goto :fail

call "%SRC_DIR%make_graphical_abstract.bat"
if errorlevel 1 goto :fail

call "%SRC_DIR%make_study_design.bat"
if errorlevel 1 goto :fail

REM --- data-derived figures (add here: call "<script>.bat" + errorlevel check) ---

echo ========================================
echo All figures built successfully.
echo ========================================
exit /b 0

:fail
echo ========================================
echo Figure build FAILED. See messages above.
echo ========================================
exit /b 1
