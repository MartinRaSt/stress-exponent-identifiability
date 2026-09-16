@echo off
REM Project: Sammon mapping revisited - adaptive weighting, a scalable solver
REM          and a temporal extension (alpha-Sammon)
REM Authors: Martin Radvansky <martin.radvansky@vsb.cz>
REM          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
REM Affiliation: Department of Computer Science, Faculty of Electrical Engineering
REM              and Computer Science, VSB - Technical University of Ostrava,
REM              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
REM Created: 2026-09-15
REM License: see the LICENSE file in the repository root
REM
REM ONE SCRIPT FOR THE WHOLE UNATTENDED RUN (K15/K16, 2026-09-15).
REM Runs in this order:
REM   0. (full only) backup and MOVE OUT the full E4 outputs into results\data\backup_e4_<stamp>\
REM      - nothing is deleted, only moved; forces a full recompute of E4 with the new
REM        neighborhood preservation metric columns (trust_k*, cont_k*, jacc_k*) for all 3 seeds
REM        and with the corrected permutation p-value (1+count)/(1+B).
REM   1. src\run_exp4_temporal.bat            - full E4 refit (~70 min, resumable)
REM   2. src\run_exp4_neighbor_metrics.bat    - neighborhood metrics over trajectories (minutes)
REM   3. src\run_main.bat --no-figures        - statistics, tables, clanek\generated\numbers.tex
REM   4. src\run_main.bat --figures-only      - figures
REM
REM PARAMETER: full (default) | smoke | quick
REM   smoke/quick does NOT move any full data and only writes to results\...\<mode>\.
REM
REM RESTART AFTER A CRASH: the script creates the marker results\data\exp4_refit_started.flag.
REM   If it exists, step 0 is SKIPPED - a repeated run thus continues where the
REM   run left off (resume from the checkpoint) instead of discarding rows already computed.
REM   The marker is deleted only after the whole chain has completed successfully.
REM
REM COMPUTER SLEEP: disabled and restored by the called run_*.bat scripts (each step separately via
REM   src\common\no_sleep_on.bat / no_sleep_off.bat). This wrapper script deliberately does
REM   NOT call it - a nested call would save the already-disabled state (0) as the
REM   "original" value, and sleep would stay permanently disabled after finishing.
REM
REM Log: results\logs\run_vse_20260915.log (return codes of each step).
setlocal
cd /d "%~dp0.."

set "MODE=%1"
if "%MODE%"=="" set "MODE=full"
if "%MODE%"=="full" goto :mode_ok
if "%MODE%"=="smoke" goto :mode_ok
if "%MODE%"=="quick" goto :mode_ok
echo ERROR: unknown mode "%MODE%". Use: full or smoke or quick
exit /b 2
:mode_ok

set "LOG=results\logs\run_vse_20260915.log"
set "FLAG=results\data\exp4_refit_started.flag"
if not exist "results\logs" mkdir "results\logs"

set "STAMP=%DATE:~6,4%%DATE:~3,2%%DATE:~0,2%_%TIME:~0,2%%TIME:~3,2%"
set "STAMP=%STAMP: =0%"
set "BK=results\data\backup_e4_%STAMP%"

set "RC1=-"
set "RC2=-"
set "RC3=-"
set "RC4=-"

echo. >> "%LOG%"
echo [%DATE% %TIME%] START run_vse_20260915 mode=%MODE% >> "%LOG%"
echo === Sammon: full run, mode=%MODE% - start %DATE% %TIME% ===

REM ---- step 0: backup and move out the full E4 outputs (full only, only on the first run)
if not "%MODE%"=="full" goto :krok1
if not exist "%FLAG%" goto :backup
echo [%DATE% %TIME%] step 0 SKIPPED - marker %FLAG% exists, continuing the interrupted run >> "%LOG%"
echo [0/4] Backup skipped - resuming an interrupted run, continuing via resume.
goto :krok1

:backup
echo [0/4] Backing up and moving out the full E4 outputs to %BK% ...
mkdir "%BK%" 2>nul
move /Y "results\data\exp4_temporal_results.csv" "%BK%\" >nul 2>&1
move /Y "results\data\exp4_temporal_DONE.txt" "%BK%\" >nul 2>&1
move /Y "results\data\exp4_trajectories.csv" "%BK%\" >nul 2>&1
move /Y "results\data\exp4_stats.csv" "%BK%\" >nul 2>&1
move /Y "results\data\exp4_stats_partial.csv" "%BK%\" >nul 2>&1
move /Y "results\data\exp4_relative.csv" "%BK%\" >nul 2>&1
move /Y "results\data\exp4_neighbor_metrics_results.csv" "%BK%\" >nul 2>&1
move /Y "results\data\exp4_neighbor_metrics_DONE.txt" "%BK%\" >nul 2>&1
move /Y "results\data\exp4_neighbor_stats.csv" "%BK%\" >nul 2>&1
if exist "results\data\embeddings\exp4_temporal" move /Y "results\data\embeddings\exp4_temporal" "%BK%\embeddings_exp4_temporal" >nul 2>&1
echo moved out %DATE% %TIME% > "%FLAG%"
echo [%DATE% %TIME%] step 0 done, backup %BK% >> "%LOG%"

:krok1
echo [1/4] E4 temporal experiment, mode=%MODE% - the longest step, roughly an hour for full ...
call "%~dp0run_exp4_temporal.bat" %MODE%
set "RC1=%ERRORLEVEL%"
echo [%DATE% %TIME%] run_exp4_temporal %MODE% rc=%RC1% >> "%LOG%"
if not "%RC1%"=="0" goto :konec

echo [2/4] Neighborhood preservation metrics over trajectories, mode=%MODE% ...
call "%~dp0run_exp4_neighbor_metrics.bat" %MODE%
set "RC2=%ERRORLEVEL%"
echo [%DATE% %TIME%] run_exp4_neighbor_metrics %MODE% rc=%RC2% >> "%LOG%"
if not "%RC2%"=="0" goto :konec

echo [3/4] Statistics, tables and number macros, mode=%MODE% ...
call "%~dp0run_main.bat" %MODE% --no-figures
set "RC3=%ERRORLEVEL%"
echo [%DATE% %TIME%] run_main %MODE% --no-figures rc=%RC3% >> "%LOG%"
if not "%RC3%"=="0" goto :konec

echo [4/4] Figures, mode=%MODE% ...
call "%~dp0run_main.bat" %MODE% --figures-only
set "RC4=%ERRORLEVEL%"
echo [%DATE% %TIME%] run_main %MODE% --figures-only rc=%RC4% >> "%LOG%"
if not "%RC4%"=="0" goto :konec

if "%MODE%"=="full" if exist "%FLAG%" del "%FLAG%" >nul 2>&1

:konec
echo [%DATE% %TIME%] END run_vse_20260915 rc=%RC1%/%RC2%/%RC3%/%RC4% >> "%LOG%"
echo.
echo === End %DATE% %TIME% ===
echo Return codes: E4=%RC1%  neighborhood=%RC2%  tables=%RC3%  figures=%RC4%   -- 0 means OK
echo Log: %LOG%
if "%MODE%"=="full" echo Backup of the original E4 outputs: %BK%
if exist "%FLAG%" echo WARNING: the run did NOT finish completely - after fixing, run the script AGAIN, it will resume the interrupted work via marker %FLAG%
endlocal
exit /b 0
