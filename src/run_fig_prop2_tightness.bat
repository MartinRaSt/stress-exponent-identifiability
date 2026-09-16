@echo off
REM Project: Sammon mapping revisited - adaptive weighting, a scalable solver
REM          and a temporal extension (alpha-Sammon)
REM Authors: Martin Radvansky <martin.radvansky@vsb.cz>
REM          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
REM Affiliation: Department of Computer Science, Faculty of Electrical Engineering
REM              and Computer Science, VSB - Technical University of Ostrava,
REM              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
REM Created: 2026-09-14
REM License: see the LICENSE file in the repository root
REM
REM fig_prop2_tightness: tightness of the Proposition 2 bounds (tight_a/tight_b) vs.
REM alpha, color=rho_NN -> results/figures/[<mode>/]fig_prop2_tightness.pdf/.csv.
REM Fast aggregation over already finished exp8_prop2_check data, no new computation.
REM Parameter: quick|full|smoke (default full).
setlocal
cd /d "%~dp0.."
call "%~dp0common\no_sleep_on.bat"
set MODE=%1
if "%MODE%"=="" set MODE=full
"venv\python.exe" -m src.figures.fig_prop2_tightness --%MODE%
set "EXIT_CODE=%ERRORLEVEL%"
call "%~dp0common\no_sleep_off.bat"
exit /b %EXIT_CODE%
