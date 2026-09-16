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
REM E4 recompute after changing the init of new nodes (K14, jitter in src/sammon/temporal.py)
REM and refresh of the tables, macros and figures:
REM   1. backup of the full E4 outputs (CSV, DONE, embeddings/pertrans cache) into
REM      results/data/backup_e4_<date>/ (nothing is deleted),
REM   2. src\run_exp4_temporal.bat full   (7 datasets, ~70 min, resume from scratch),
REM   3. src\run_main.bat full --no-figures (statistics, tables, numbers.tex, ~1 min),
REM   4. src\run_main.bat full --figures-only (48 figures, ~4 min).
REM Step return codes are written to results/logs/run_e4_prepocet_a_main.log.
REM Run ONLY by the author manually. Sleep is disabled by the called run_*.bat scripts.
setlocal
cd /d "%~dp0.."
set "STAMP=%DATE:~6,4%%DATE:~3,2%%DATE:~0,2%_%TIME:~0,2%%TIME:~3,2%"
set "STAMP=%STAMP: =0%"
set "LOG=results\logs\run_e4_prepocet_a_main.log"
set "BK=results\data\backup_e4_%STAMP%"
echo [%DATE% %TIME%] START run_e4_prepocet_a_main (backup %BK%) >> "%LOG%"

mkdir "%BK%" 2>nul
move /Y "results\data\exp4_temporal_results.csv" "%BK%\" >nul 2>&1
move /Y "results\data\exp4_temporal_DONE.txt" "%BK%\" >nul 2>&1
move /Y "results\data\exp4_relative.csv" "%BK%\" >nul 2>&1
move /Y "results\data\exp4_stats.csv" "%BK%\" >nul 2>&1
move /Y "results\data\exp4_trajectories.csv" "%BK%\" >nul 2>&1
if exist "results\data\embeddings\exp4_temporal" move /Y "results\data\embeddings\exp4_temporal" "%BK%\embeddings_exp4_temporal" >nul 2>&1
echo [%DATE% %TIME%] backup done >> "%LOG%"

call "%~dp0run_exp4_temporal.bat" full
set "RC1=%ERRORLEVEL%"
echo [%DATE% %TIME%] run_exp4_temporal full rc=%RC1% >> "%LOG%"
if not "%RC1%"=="0" goto :end

call "%~dp0run_main.bat" full --no-figures
set "RC2=%ERRORLEVEL%"
echo [%DATE% %TIME%] run_main full --no-figures rc=%RC2% >> "%LOG%"
if not "%RC2%"=="0" goto :end

call "%~dp0run_main.bat" full --figures-only
set "RC3=%ERRORLEVEL%"
echo [%DATE% %TIME%] run_main full --figures-only rc=%RC3% >> "%LOG%"

:end
echo [%DATE% %TIME%] END rc=%RC1%/%RC2%/%RC3% >> "%LOG%"
echo.
echo Done. Return codes: exp4=%RC1% main_tables=%RC2% main_figures=%RC3%  (log: %LOG%)
echo Backup of the original E4 outputs: %BK%
endlocal
exit /b 0
