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
REM One command for the rerun after the 2026-09-12 review: preparation -> E2 (2-3 h)
REM -> E3 alpha_auto (~1 h) -> run_main (statistics, tables, figures).
REM Sleep is disabled by each called run_*.bat itself (no_sleep_on/off).
REM WARNING: run only after the previous run_main.bat (figures) has finished.
setlocal
cd /d "%~dp0.."
call "%~dp0common\no_sleep_on.bat"

call src\prepare_rerun_20260912.bat
if errorlevel 1 goto :end

call src\run_exp2_solver_scaling.bat full
set RC2=%ERRORLEVEL%
call src\run_exp3_graph_layout.bat full
set RC3=%ERRORLEVEL%
call src\run_main.bat full
set RCM=%ERRORLEVEL%

echo ============================================================
echo   E2 solver_scaling exit=%RC2%
echo   E3 graph_layout   exit=%RC3%
echo   main              exit=%RCM%
echo ============================================================
type results\data\exp2_solver_scaling_DONE.txt 2>nul
type results\data\exp3_graph_layout_DONE.txt 2>nul

:end
call "%~dp0common\no_sleep_off.bat"
endlocal
