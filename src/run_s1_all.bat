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
REM One command for the whole phase S1 (documentation/2026-09-12_plan_smeru_clanku.md):
REM   1. dataset_properties (K1, incl. correlation check with alpha)
REM   2. exp1_cluster_geometry (K2, over the saved E1 embeddings)
REM   3. pareto_analysis (K3)
REM   4. make_figures (all figures incl. the new K3/K4/K9)
REM   5. export_numbers (K12a -> clanek/generated/numbers.tex; quick|smoke -> results/tables/<mode>/)
REM Steps 1-3 stop on error (later steps depend on them); steps 4-5 always
REM run and the exit codes are printed at the end. Sleep is disabled by
REM each called run_*.bat itself (no_sleep_on/off), here additionally for the whole chain.
REM Parameter: full (default) | quick | smoke
setlocal
cd /d "%~dp0.."
call "%~dp0common\no_sleep_on.bat"
set MODE=%1
if "%MODE%"=="" set MODE=full
set LOGDIR=results\logs
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
echo [%DATE% %TIME%] S1 start (mode=%MODE%) >> "%LOGDIR%\run_s1_all.log"

echo === 1/5 dataset_properties ===
call src\run_dataset_properties.bat %MODE%
set RC1=%ERRORLEVEL%
if not "%RC1%"=="0" (
    echo ERROR: dataset_properties finished with code %RC1% - stopping.
    goto :summary
)

echo === 2/5 exp1_cluster_geometry ===
call src\run_exp1_cluster_geometry.bat %MODE%
set RC2=%ERRORLEVEL%
if not "%RC2%"=="0" (
    echo ERROR: exp1_cluster_geometry finished with code %RC2% - stopping.
    goto :summary
)

echo === 3/5 pareto_analysis ===
call src\run_pareto_analysis.bat %MODE%
set RC3=%ERRORLEVEL%
if not "%RC3%"=="0" (
    echo ERROR: pareto_analysis finished with code %RC3% - stopping.
    goto :summary
)

echo === 4/5 make_figures ===
call src\make_figures.bat %MODE%
set RC4=%ERRORLEVEL%

echo === 5/5 export_numbers ===
call src\run_export_numbers.bat %MODE%
set RC5=%ERRORLEVEL%

:summary
echo ============================================================
echo   1 dataset_properties     exit=%RC1%
echo   2 exp1_cluster_geometry  exit=%RC2%
echo   3 pareto_analysis        exit=%RC3%
echo   4 make_figures           exit=%RC4%
echo   5 export_numbers         exit=%RC5%
echo ============================================================
type results\data\exp1_cluster_geometry_DONE.txt 2>nul
echo [%DATE% %TIME%] S1 end rc=%RC1%/%RC2%/%RC3%/%RC4%/%RC5% >> "%LOGDIR%\run_s1_all.log"
call "%~dp0common\no_sleep_off.bat"
endlocal
