@echo off
REM Project: Sammon mapping revisited - adaptive weighting, a scalable solver
REM          and a temporal extension (alpha-Sammon)
REM Authors: Martin Radvansky <martin.radvansky@vsb.cz>
REM          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
REM Affiliation: Department of Computer Science, Faculty of Electrical Engineering
REM              and Computer Science, VSB - Technical University of Ostrava,
REM              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
REM Created: 2026-09-11
REM License: see the LICENSE file in the repository root
REM
REM Preparation for a rerun after the 2026-09-11 fixes (see
REM documentation/2026-09-11_plan_prebehu.md): removes only the rows affected by
REM the fixes from the full CSVs in results/data/ (with a .bak_<timestamp> backup),
REM so that a subsequent `src\run_all_experiments.bat full` computes only the missing
REM combinations (resume). New methods (tsne_auto, umap_auto) and new E4 datasets
REM are added automatically, nothing needs to be deleted for them.
REM Usage: src\prepare_rerun_20260911.bat [dry]   (dry = print only)
setlocal
cd /d "%~dp0.."
set DRY=
if /I "%1"=="dry" set DRY=--dry-run
set PY="venv\python.exe"
set RM=%PY% -m src.experiments.remove_rows

echo === E1: standardized tabular datasets (all methods) ===
%RM% --csv results\data\exp1_dr_benchmark_results.csv --dataset iris,wine,breast_cancer,glass,ionosphere,seeds,yeast,vehicle,segment,satimage,letter,spambase %DRY%
if errorlevel 1 goto :fail
echo === E1: sammon_classic (step-halving) on all datasets ===
%RM% --csv results\data\exp1_dr_benchmark_results.csv --method sammon_classic %DRY%
if errorlevel 1 goto :fail

echo === E2 scaling: sparse solvers (full stress) + GPU SMACOF n=20000 (tiling fix) ===
%RM% --csv results\data\exp2_scaling_results.csv --method sparse_smacof__cpu,sparse_sgd__cpu %DRY%
if errorlevel 1 goto :fail
%RM% --csv results\data\exp2_scaling_results.csv --dataset gaussian_clusters_n20000 --method smacof__cuda %DRY%
if errorlevel 1 goto :fail
echo === E2 sparse gap: whole file (new stress_sparse_terms column) ===
if "%DRY%"=="" (
    if exist results\data\exp2_sparse_results.csv move /Y results\data\exp2_sparse_results.csv results\data\exp2_sparse_results.csv.bak_20260911 >nul
) else (
    echo [dry] would delete results\data\exp2_sparse_results.csv
)

echo === E3: sammon_classic (step-halving) ===
%RM% --csv results\data\exp3_graph_layout_results.csv --method sammon_classic %DRY%
if errorlevel 1 goto :fail

echo === E4: SMACOF rows primary_school (fix for lambda without anchors) ===
%RM% --csv results\data\exp4_temporal_results.csv --dataset primary_school_temporal --method lambda0.0_alpha1.0_smacof,lambda0.01_alpha1.0_smacof,lambda0.03_alpha1.0_smacof,lambda0.1_alpha1.0_smacof,lambda0.3_alpha1.0_smacof,lambda1.0_alpha1.0_smacof,lambda3.0_alpha1.0_smacof,lambda10.0_alpha1.0_smacof %DRY%
if errorlevel 1 goto :fail
if "%DRY%"=="" (
    if exist results\data\exp4_stats.csv move /Y results\data\exp4_stats.csv results\data\exp4_stats.csv.bak_20260911 >nul
    if exist results\data\exp4_trajectories.csv move /Y results\data\exp4_trajectories.csv results\data\exp4_trajectories.csv.bak_20260911 >nul
)

echo === E5: iris (standardization) ===
%RM% --csv results\data\exp5_ablation_results.csv --dataset iris %DRY%
if errorlevel 1 goto :fail

if "%DRY%"=="" (
    echo === DONE markers deleted so run_all_experiments.bat does not show stale state ===
    del /Q results\data\exp1_dr_benchmark_DONE.txt results\data\exp2_solver_scaling_DONE.txt results\data\exp3_graph_layout_DONE.txt results\data\exp4_temporal_DONE.txt results\data\exp5_ablation_DONE.txt 2>nul
)
echo === DONE. Next step: src\run_all_experiments.bat full ===
endlocal
exit /b 0

:fail
echo ERROR while removing rows - DO NOT run the rerun, check the output above.
endlocal
exit /b 1
