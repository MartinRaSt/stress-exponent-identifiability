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
REM Q1-step1 (documentation/2026-09-14_exp8_prop2_check.md): one command for
REM the whole step - post-hoc empirical test of Proposition 2 over the already finished
REM exp6_alpha_curves embeddings:
REM   1. exp8_prop2_check   (src/experiments/exp8_prop2_check.py)
REM   2. fig_prop2_tightness           (tightness of the bounds vs. alpha/rho_NN)
REM   3. fig_prop2_alpha_prediction    (alpha_bound theory vs. alpha_star empirical)
REM   4. export_numbers (K12a -> clanek/generated/numbers.tex; quick|smoke ->
REM      results/tables/<mode>/numbers.tex)
REM ASSUMPTION: exp6_alpha_curves has already run IN THE SAME mode (otherwise exp8
REM fails fail-loud on missing .npy embeddings) - run
REM src\run_exp6_alpha_curves.bat %MODE% first if it has not run yet.
REM Steps 1-3 stop on error (later steps depend on them); step 4 always
REM runs and the exit codes are printed at the end. Sleep is disabled by each
REM called run_*.bat itself (no_sleep_on/off), here additionally for the whole chain.
REM Parameter: full (default) | quick | smoke
setlocal
cd /d "%~dp0.."
call "%~dp0common\no_sleep_on.bat"
set MODE=%1
if "%MODE%"=="" set MODE=full
set LOGDIR=results\logs
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
echo [%DATE% %TIME%] Q1-step1 start (mode=%MODE%) >> "%LOGDIR%\run_q1_step1_prop2.log"

echo === 1/4 exp8_prop2_check ===
call src\run_exp8_prop2_check.bat %MODE%
set RC1=%ERRORLEVEL%
if not "%RC1%"=="0" (
    echo ERROR: exp8_prop2_check finished with code %RC1% - stopping.
    goto :summary
)

echo === 2/4 fig_prop2_tightness ===
call src\run_fig_prop2_tightness.bat %MODE%
set RC2=%ERRORLEVEL%
if not "%RC2%"=="0" (
    echo ERROR: fig_prop2_tightness finished with code %RC2% - stopping.
    goto :summary
)

echo === 3/4 fig_prop2_alpha_prediction ===
call src\run_fig_prop2_alpha_prediction.bat %MODE%
set RC3=%ERRORLEVEL%
if not "%RC3%"=="0" (
    echo ERROR: fig_prop2_alpha_prediction finished with code %RC3% - stopping.
    goto :summary
)

echo === 4/4 export_numbers ===
call src\run_export_numbers.bat %MODE%
set RC4=%ERRORLEVEL%

:summary
echo ============================================================
echo Q1-step1 summary (mode=%MODE%):
echo   1/4 exp8_prop2_check           RC=%RC1%
echo   2/4 fig_prop2_tightness        RC=%RC2%
echo   3/4 fig_prop2_alpha_prediction RC=%RC3%
echo   4/4 export_numbers             RC=%RC4%
echo ============================================================
echo [%DATE% %TIME%] Q1-step1 end (mode=%MODE%) RC1=%RC1% RC2=%RC2% RC3=%RC3% RC4=%RC4% >> "%LOGDIR%\run_q1_step1_prop2.log"

call "%~dp0common\no_sleep_off.bat"

if not "%RC1%"=="0" exit /b %RC1%
if not "%RC2%"=="0" exit /b %RC2%
if not "%RC3%"=="0" exit /b %RC3%
exit /b %RC4%
