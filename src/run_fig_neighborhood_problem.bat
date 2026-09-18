@echo off
REM Project: Sammon mapping revisited - adaptive weighting, a scalable solver
REM          and a temporal extension (alpha-Sammon)
REM Authors: Martin Radvansky <martin.radvansky@vsb.cz>
REM          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
REM Affiliation: Department of Computer Science, Faculty of Electrical Engineering
REM              and Computer Science, VSB - Technical University of Ostrava,
REM              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
REM Created: 2026-09-17
REM License: see the LICENSE file in the repository root
REM
REM fig_neighborhood_problem: flagship 3-panel figure (MDS / tuned
REM alpha-Sammon / t-SNE, original-space k-NN edges) -> results/figures/
REM [<mode>/]fig_neighborhood_problem.pdf/.csv. Requires
REM run_exp6_alpha_curves.bat, run_exp12_alpha_grid_extension.bat and
REM run_exp1_dr_benchmark.bat to have already completed in the same mode.
REM NOTE: --quick has no 'helix'/'s_curve' data in exp1_dr_benchmark.quick/
REM exp6_alpha_curves.quick - only smoke/full are usable (see
REM fig_neighborhood_problem.py's module docstring).
REM Parameter: quick|full|smoke.
setlocal
cd /d "%~dp0.."
call "%~dp0common\no_sleep_on.bat"
set MODE=%1
if "%MODE%"=="" set MODE=full
"venv\python.exe" -m src.figures.fig_neighborhood_problem --%MODE%
set "EXIT_CODE=%ERRORLEVEL%"
call "%~dp0common\no_sleep_off.bat"
exit /b %EXIT_CODE%
