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
REM fig_prop2_alpha_prediction: does distance concentration (alpha_bound,
REM Proposition 2) predict the empirical optimum of alpha (alpha_star, exp6_alpha_optimum.csv)?
REM -> results/figures/[<mode>/]fig_prop2_alpha_prediction.pdf/.csv +
REM _spearman.csv. Requires exp8_prop2_check AND exp6_alpha_curves
REM (table exp6_alpha_optimum.csv) to have already completed in the same mode. Parameter: quick|full|smoke.
setlocal
cd /d "%~dp0.."
call "%~dp0common\no_sleep_on.bat"
set MODE=%1
if "%MODE%"=="" set MODE=full
"venv\python.exe" -m src.figures.fig_prop2_alpha_prediction --%MODE%
set "EXIT_CODE=%ERRORLEVEL%"
call "%~dp0common\no_sleep_off.bat"
exit /b %EXIT_CODE%
