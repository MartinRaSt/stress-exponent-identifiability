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
REM One command for the whole phase S2 (documentation/2026-09-12_plan_smeru_clanku.md):
REM   1. run_exp6_alpha_curves   (K6 - alpha curves on the 32 E1 datasets)
REM   2. run_fit_alpha_rule      (K7 - derivation + LOO validation of alpha_pred, requires 1 and K1 dataset_properties)
REM   3. run_exp1_dr_benchmark   (K5/K7 - resume, adds densmap/phate/sammon_alpha_pred)
REM   4. run_exp3_graph_layout   (K7/K8 - resume, adds sammon_alpha_pred + new K8 graphs/methods)
REM   5. run_exp1_cluster_geometry (K2 - resume, new rows over the new embeddings from step 3)
REM   6. run_main                (statistics, tables, pareto, figures, export_numbers)
REM The order is mandatory (see plan section 4): exp6 -> fit_alpha_rule -> exp1 -> exp3,
REM because exp1/exp3 use the sammon_alpha_pred method, which requires an already
REM finished results/data/alpha_pred_rule.json (only in full mode - see
REM src/experiments/fit_alpha_rule.py). Steps 1-4 are fail-stop (each next step
REM depends on the previous one - without fit_alpha_rule, exp1/exp3 would just record
REM sammon_alpha_pred as status=error, which is not fatal but needlessly
REM clutters the results); steps 5-6 always run, with an exit code summary at the
REM end. Sleep is disabled by each called run_*.bat itself (no_sleep_on/off),
REM here additionally for the whole chain. Parameter: full (default) | quick | smoke.
setlocal
cd /d "%~dp0.."
call "%~dp0common\no_sleep_on.bat"
set MODE=%1
if "%MODE%"=="" set MODE=full
set LOGDIR=results\logs
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
echo [%DATE% %TIME%] S2 start (mode=%MODE%) >> "%LOGDIR%\run_s2_all.log"

echo === 1/6 run_exp6_alpha_curves ===
call src\run_exp6_alpha_curves.bat %MODE%
set RC1=%ERRORLEVEL%
if not "%RC1%"=="0" (
    echo ERROR: run_exp6_alpha_curves finished with code %RC1% - stopping.
    goto :summary
)

echo === 2/6 run_fit_alpha_rule ===
call src\run_fit_alpha_rule.bat %MODE%
set RC2=%ERRORLEVEL%
if not "%RC2%"=="0" (
    echo ERROR: run_fit_alpha_rule finished with code %RC2% - stopping.
    goto :summary
)

echo === 3/6 run_exp1_dr_benchmark (resume - densmap/phate/sammon_alpha_pred) ===
call src\run_exp1_dr_benchmark.bat %MODE%
set RC3=%ERRORLEVEL%
if not "%RC3%"=="0" (
    echo ERROR: run_exp1_dr_benchmark finished with code %RC3% - stopping.
    goto :summary
)

echo === 4/6 run_exp3_graph_layout (resume - sammon_alpha_pred + K8 additions) ===
call src\run_exp3_graph_layout.bat %MODE%
set RC4=%ERRORLEVEL%
if not "%RC4%"=="0" (
    echo ERROR: run_exp3_graph_layout finished with code %RC4% - stopping.
    goto :summary
)

echo === 5/6 run_exp1_cluster_geometry (resume) ===
call src\run_exp1_cluster_geometry.bat %MODE%
set RC5=%ERRORLEVEL%

echo === 6/6 run_main (statistics, tables, pareto, figures, export_numbers) ===
call src\run_main.bat %MODE%
set RC6=%ERRORLEVEL%

:summary
echo ============================================================
echo   1 run_exp6_alpha_curves      exit=%RC1%
echo   2 run_fit_alpha_rule         exit=%RC2%
echo   3 run_exp1_dr_benchmark      exit=%RC3%
echo   4 run_exp3_graph_layout      exit=%RC4%
echo   5 run_exp1_cluster_geometry  exit=%RC5%
echo   6 run_main                   exit=%RC6%
echo ============================================================
type results\data\exp6_alpha_curves_DONE.txt 2>nul
type results\data\exp1_dr_benchmark_DONE.txt 2>nul
type results\data\exp3_graph_layout_DONE.txt 2>nul
echo [%DATE% %TIME%] S2 end rc=%RC1%/%RC2%/%RC3%/%RC4%/%RC5%/%RC6% >> "%LOGDIR%\run_s2_all.log"
call "%~dp0common\no_sleep_off.bat"
endlocal
