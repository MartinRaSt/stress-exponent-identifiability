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
REM Q1-step3 (reserse/2026-09-14_specifikace_rozsireni_q1.md PART B, B.7 item 8)
REM - one command for the whole metric fidelity task with known ground truth:
REM   1. run_exp9_metric_fidelity        (grid of scenario x replicate x method)
REM   2. run_exp9_metric_fidelity_stats  (families F_B1/F_B2/F_B3, Holm)
REM   3. run_fig_metric_fidelity_gallery (gallery of r=0 embeddings)
REM   4. run_fig_metric_fidelity_boxes   (boxplots of the primary metric)
REM   5. run_main --no-figures           (updates numbers.tex incl. the mf* macros)
REM The order is mandatory (each next step reads the output of the previous one); fail-stop -
REM steps 2-5 only run after step 1 has completed successfully (otherwise nonsensical
REM empty/missing inputs). Independent of run_q1_step2_datasets.bat (different
REM CSVs, different embeddings), but do NOT run CONCURRENTLY (limit of 8 workers x 1 BLAS
REM thread, see config.yaml parallel.*).
REM Parameter: quick|full|smoke (default full).
setlocal
cd /d "%~dp0.."
call "%~dp0common\no_sleep_on.bat"
set MODE=%1
if "%MODE%"=="" set MODE=full
set LOGDIR=results\logs
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
echo [%DATE% %TIME%] Q1-step3 start (mode=%MODE%) >> "%LOGDIR%\run_q1_step3_metric_fidelity.log"

echo === 1/5 run_exp9_metric_fidelity ===
call src\run_exp9_metric_fidelity.bat %MODE%
set RC1=%ERRORLEVEL%
if not "%RC1%"=="0" (
    echo ERROR: run_exp9_metric_fidelity finished with code %RC1% - stopping.
    goto :summary
)

echo === 2/5 run_exp9_metric_fidelity_stats ===
call src\run_exp9_metric_fidelity_stats.bat %MODE%
set RC2=%ERRORLEVEL%
if not "%RC2%"=="0" (
    echo ERROR: run_exp9_metric_fidelity_stats finished with code %RC2% - stopping.
    goto :summary
)

echo === 3/5 run_fig_metric_fidelity_gallery ===
call src\run_fig_metric_fidelity_gallery.bat %MODE%
set RC3=%ERRORLEVEL%
if not "%RC3%"=="0" (
    echo ERROR: run_fig_metric_fidelity_gallery finished with code %RC3% - stopping.
    goto :summary
)

echo === 4/5 run_fig_metric_fidelity_boxes ===
call src\run_fig_metric_fidelity_boxes.bat %MODE%
set RC4=%ERRORLEVEL%
if not "%RC4%"=="0" (
    echo ERROR: run_fig_metric_fidelity_boxes finished with code %RC4% - stopping.
    goto :summary
)

echo === 5/5 run_main --no-figures (numbers.tex incl. mf* macros) ===
call src\run_main.bat %MODE% --no-figures
set RC5=%ERRORLEVEL%

:summary
echo [%DATE% %TIME%] Q1-step3 end (mode=%MODE%) RC1=%RC1% RC2=%RC2% RC3=%RC3% RC4=%RC4% RC5=%RC5% >> "%LOGDIR%\run_q1_step3_metric_fidelity.log"
echo Summary: exp9=%RC1% stats=%RC2% gallery=%RC3% boxes=%RC4% main=%RC5%
call "%~dp0common\no_sleep_off.bat"
if not "%RC1%"=="0" exit /b %RC1%
if not "%RC2%"=="0" exit /b %RC2%
if not "%RC3%"=="0" exit /b %RC3%
if not "%RC4%"=="0" exit /b %RC4%
exit /b %RC5%
