@echo off
REM Project: Sammon mapping revisited - adaptive weighting, a scalable solver
REM          and a temporal extension (alpha-Sammon)
REM Authors: Martin Radvansky <martin.radvansky@vsb.cz>
REM          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
REM Affiliation: Department of Computer Science, Faculty of Electrical Engineering
REM              and Computer Science, VSB - Technical University of Ostrava,
REM              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
REM Created: 2026-09-13
REM License: see the LICENSE file in the repository root
REM
REM Stratified E1 analysis by distance-concentration regime (alpha
REM matters only in the concentrated-distance regime) - no re-run of DR methods,
REM fast aggregation over the already computed exp1_dr_benchmark_results.csv +
REM dataset_properties.csv + alpha_pred_rule.json (on the order of seconds) ->
REM results/tables/[<mode>/]exp1_regime_stratified.csv/.tex.
REM Parameter: quick|full|smoke (default full).
setlocal
cd /d "%~dp0.."
REM Disable computer standby/hibernation for the duration of the run (restored at the
REM end, even after an error) - author requirement, see src/common/no_sleep_on.bat.
call "%~dp0common\no_sleep_on.bat"
set MODE=%1
if "%MODE%"=="" set MODE=full
"venv\python.exe" -m src.experiments.exp1_regime_stratified --%MODE%
set "EXIT_CODE=%ERRORLEVEL%"
call "%~dp0common\no_sleep_off.bat"
exit /b %EXIT_CODE%
