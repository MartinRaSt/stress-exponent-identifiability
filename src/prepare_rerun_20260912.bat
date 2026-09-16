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
REM Preparation for a rerun after reviewing the 2026-09-12 results (author decision):
REM  - E3: delete sammon_alpha_auto rows (recompute the selected_hyperparam column)
REM  - E2: schema of exp2_scaling_results.csv already fixed on 2026-09-12 (column
REM    stress_sparse_terms), here just a check; E2 then resumes on its own.
REM Parameter "dry" = print only, no writes.
setlocal
cd /d "%~dp0.."
set DRY=
if /I "%1"=="dry" set DRY=--dry-run
set PY="venv\python.exe"
set RM=%PY% -m src.experiments.remove_rows

echo === E3: sammon_alpha_auto (recompute selected_hyperparam) ===
%RM% --csv results\data\exp3_graph_layout_results.csv --method sammon_alpha_auto %DRY%
if errorlevel 1 goto :fail

echo === E2: schema check for exp2_scaling_results.csv ===
%PY% -c "import csv,sys; h=next(csv.reader(open(r'results/data/exp2_scaling_results.csv',encoding='utf-8'))); sys.exit(0 if 'stress_sparse_terms' in h else 1)"
if errorlevel 1 (
    echo ERROR: exp2_scaling_results.csv is missing the stress_sparse_terms column - see documentation\2026-09-12_kontrola_vysledku_prebehu.md section 2
    goto :fail
)
echo   OK

if "%DRY%"=="" (
    echo === E2/E3 DONE markers deleted ===
    del /Q results\data\exp2_solver_scaling_DONE.txt results\data\exp3_graph_layout_DONE.txt 2>nul
)
echo === preparation done ===
endlocal
exit /b 0

:fail
echo === ERROR while preparing the rerun, do not run anything else ===
endlocal
exit /b 1
