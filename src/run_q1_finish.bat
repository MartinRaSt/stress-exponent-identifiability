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
REM Runs the REMAINING steps 9-11 of the run_q1_step2_datasets.bat chain, which did
REM not run due to the failure of step 8 (alternative='tost' in sign_flip_test,
REM fixed 2026-09-14). Assumes the following are DONE:
REM   - run_q1_step2_datasets.bat full  (steps 1-7)
REM   - exp1_holdout_confirmatory       (step 8, recomputed after the fix)
REM   - run_q1_step3_metric_fidelity.bat full
REM
REM All of these are fast analyses and figures over already finished CSVs - no new DR
REM computation, on the order of minutes. Once finished, all data and all macros
REM in clanek/generated/numbers.tex are up to date.
REM
REM Parameter: smoke|quick|full (default full).
REM Can be run from anywhere (even from the src folder) - the script switches itself
REM to the project root.
setlocal
cd /d "%~dp0.."

set MODE=%1
if "%MODE%"=="" set MODE=full

set LOGDIR=results\logs
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
echo [%DATE% %TIME%] run_q1_finish start ^(mode=%MODE%^) >> "%LOGDIR%\run_q1_finish.log"

REM Disable computer standby/hibernation for the duration of the run; restored at the end,
REM even after an error.
call "%~dp0common\no_sleep_on.bat"

REM Limit BLAS threads to 1 per process (config.yaml parallel.blas_threads_per_worker).
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1

echo === 1/4 power_analysis_regime ===
"venv\python.exe" -m src.experiments.power_analysis_regime --%MODE%
set RC1=%ERRORLEVEL%
if not "%RC1%"=="0" (
    echo ERROR: power_analysis_regime finished with code %RC1% - stopping.
    goto :summary
)

echo === 2/4 fig_holdout_paired ===
"venv\python.exe" -m src.figures.fig_holdout_paired --%MODE%
set RC2=%ERRORLEVEL%
if not "%RC2%"=="0" (
    echo ERROR: fig_holdout_paired finished with code %RC2% - stopping.
    goto :summary
)

echo === 3/4 fig_regime_map ===
"venv\python.exe" -m src.figures.fig_regime_map --%MODE%
set RC3=%ERRORLEVEL%
if not "%RC3%"=="0" (
    echo ERROR: fig_regime_map finished with code %RC3% - stopping.
    goto :summary
)

echo === 4/4 run_main.bat %MODE% --no-figures ===
call src\run_main.bat %MODE% --no-figures
set RC4=%ERRORLEVEL%

:summary
echo ============================================================
echo run_q1_finish ^(mode=%MODE%^) summary:
echo   1 power_analysis_regime  : %RC1%
echo   2 fig_holdout_paired     : %RC2%
echo   3 fig_regime_map         : %RC3%
echo   4 run_main --no-figures  : %RC4%
echo ============================================================
echo [%DATE% %TIME%] run_q1_finish done ^(RC=%RC1%/%RC2%/%RC3%/%RC4%^) >> "%LOGDIR%\run_q1_finish.log"
call "%~dp0common\no_sleep_off.bat"
endlocal
