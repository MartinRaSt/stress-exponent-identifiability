@echo off
REM Project: Sammon mapping revisited - adaptive weighting, a scalable solver
REM          and a temporal extension (alpha-Sammon)
REM Authors: Martin Radvansky <martin.radvansky@vsb.cz>
REM          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
REM Affiliation: Department of Computer Science, Faculty of Electrical Engineering
REM              and Computer Science, VSB - Technical University of Ostrava,
REM              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
REM Created: 2026-09-09
REM License: see the LICENSE file in the repository root
REM
REM Runs E1-E5 sequentially. Parameter: quick|full|smoke (default full).
REM Continues even if one of the experiments fails (each individual
REM dataset/method/seed run is already handled inside the experiment with try/except - this
REM adds extra protection against a crash of the WHOLE experiment, e.g. a wrong configuration).
REM At the end prints a summary (exit code of each experiment + content of the DONE file).
setlocal enabledelayedexpansion
cd /d "%~dp0.."
REM Disable computer standby/hibernation for the duration of the run (restored at the
REM end, even after an error) - author requirement, see src/common/no_sleep_on.bat. The individual
REM run_expN_*.bat scripts also set up the sleep block themselves (nesting is
REM safe - see the comment in no_sleep_on.bat).
call "%~dp0common\no_sleep_on.bat"
REM Limit BLAS threads to 1 per process (see config.yaml parallel.blas_threads_per_worker)
REM - with parallel.n_workers=16, each process would otherwise use more OpenBLAS/MKL
REM threads and the CPU would be overloaded (the author allows max 20 logical
REM processors out of 32 available, see config.yaml parallel.n_workers). Each
REM called sub-script (run_expN_*.bat) also sets these variables itself
REM (its own setlocal), this is just an extra layer of safety on direct invocation.
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
set MODE=%1
if "%MODE%"=="" set MODE=full

echo ============================================================
echo Running all experiments E1-E5 in mode: %MODE%
echo ============================================================

REM Order from shortest to longest (changed 2026-09-11 at the author's request):
REM E5 (minutes) -> E4 -> E3 -> E2 (hours) -> E1 (longest, overnight).
REM Every experiment has resume (only computes combinations missing from the CSV).
call src\run_exp5_ablation.bat %MODE%
set RC5=%ERRORLEVEL%

call src\run_exp4_temporal.bat %MODE%
set RC4=%ERRORLEVEL%

call src\run_exp3_graph_layout.bat %MODE%
set RC3=%ERRORLEVEL%

call src\run_exp2_solver_scaling.bat %MODE%
set RC2=%ERRORLEVEL%

call src\run_exp1_dr_benchmark.bat %MODE%
set RC1=%ERRORLEVEL%

echo ============================================================
echo Run summary (mode %MODE%):
echo   E1 dr_benchmark   exit=%RC1%
echo   E2 solver_scaling exit=%RC2%
echo   E3 graph_layout   exit=%RC3%
echo   E4 temporal       exit=%RC4%
echo   E5 ablation       exit=%RC5%
echo ============================================================
echo DONE files (see results\data\ or results\data\smoke\ for --smoke):
if "%MODE%"=="smoke" (
    type results\data\smoke\exp1_dr_benchmark_DONE.txt 2>nul
    type results\data\smoke\exp2_solver_scaling_DONE.txt 2>nul
    type results\data\smoke\exp3_graph_layout_DONE.txt 2>nul
    type results\data\smoke\exp4_temporal_DONE.txt 2>nul
    type results\data\smoke\exp5_ablation_DONE.txt 2>nul
) else (
    type results\data\exp1_dr_benchmark_DONE.txt 2>nul
    type results\data\exp2_solver_scaling_DONE.txt 2>nul
    type results\data\exp3_graph_layout_DONE.txt 2>nul
    type results\data\exp4_temporal_DONE.txt 2>nul
    type results\data\exp5_ablation_DONE.txt 2>nul
)
echo ============================================================
call "%~dp0common\no_sleep_off.bat"
endlocal
