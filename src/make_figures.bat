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
REM Runs all 9 data figures in src/figures/ independently (without statistics/
REM tables - see src/run_main.bat for the full pipeline). Parameter: quick|full|smoke
REM (default full - determines which results/data/ subdirectory the inputs are
REM read from, see src/experiments/exp_common.py::resolve_experiment_name). Missing
REM input data for one figure (FileNotFoundError) does not stop the others -
REM each command runs independently, errors are printed to the console.
setlocal
cd /d "%~dp0.."
REM Disable computer standby/hibernation for the duration of the run (restored at the
REM end, even after an error) - author requirement, see src/common/no_sleep_on.bat.
call "%~dp0common\no_sleep_on.bat"
set MODE=%1
if "%MODE%"=="" set MODE=full

echo === fig_gallery_methods ===
"venv\python.exe" -m src.figures.fig_gallery_methods --%MODE%

echo === fig_before_after ===
"venv\python.exe" -m src.figures.fig_before_after --%MODE%

echo === fig_alpha_strip ===
"venv\python.exe" -m src.figures.fig_alpha_strip --%MODE%

echo === fig_sgd_convergence ===
"venv\python.exe" -m src.figures.fig_sgd_convergence --%MODE%

echo === fig_runtime_scaling ===
"venv\python.exe" -m src.figures.fig_runtime_scaling --%MODE%

echo === fig_temporal_trajectories ===
"venv\python.exe" -m src.figures.fig_temporal_trajectories --%MODE%

echo === fig_rnx_curves ===
"venv\python.exe" -m src.figures.fig_rnx_curves --%MODE%

echo === fig_cd_diagram ===
"venv\python.exe" -m src.figures.fig_cd_diagram --%MODE%

REM The main-text CD diagram is the one over `report.main_methods` for
REM scale-invariant stress; the default call above only produces the auc_rnx
REM variant, so the remaining variants used by the paper and the supplement
REM are generated explicitly (everything regenerable in one command).
echo === fig_cd_diagram exp1_dr_benchmark_main/stress_scale_invariant ===
"venv\python.exe" -m src.figures.fig_cd_diagram --%MODE% --experiment exp1_dr_benchmark_main --metric stress_scale_invariant
echo === fig_cd_diagram exp1_dr_benchmark_main/trustworthiness_k7 ===
"venv\python.exe" -m src.figures.fig_cd_diagram --%MODE% --experiment exp1_dr_benchmark_main --metric trustworthiness_k7
echo === fig_cd_diagram exp1_dr_benchmark_main/auc_rnx ===
"venv\python.exe" -m src.figures.fig_cd_diagram --%MODE% --experiment exp1_dr_benchmark_main --metric auc_rnx

REM K11 (documentation/2026-09-12_plan_smeru_clanku.md): focused figure of
REM graph layouts for the main text (polbooks/football/cora x 2 distances
REM x 5 methods) - requires src\run_exp3_graph_layout.bat (E3).
echo === fig_graph_layouts (K11, main text) ===
"venv\python.exe" -m src.figures.fig_graph_layouts --%MODE%

REM K11: supplement variant (all graphs x all E3 methods), original
REM fig_graph_layouts.py renamed.
echo === fig_graph_layouts_all (K11, supplement) ===
"venv\python.exe" -m src.figures.fig_graph_layouts_all --%MODE%

REM K3 (documentation/2026-09-12_plan_smeru_clanku.md): fig_pareto_front reads
REM results/data/[mode/]pareto_membership.csv + results/tables/pareto_median_front.csv
REM (src/experiments/pareto_analysis.py) - must run BEFORE the figure.
echo === pareto_analysis (K3, prerequisite for fig_pareto_front) ===
"venv\python.exe" -m src.experiments.pareto_analysis --%MODE%

echo === fig_pareto_front (K3) ===
"venv\python.exe" -m src.figures.fig_pareto_front --%MODE%

echo === fig_temporal_pareto (K9) ===
"venv\python.exe" -m src.figures.fig_temporal_pareto --%MODE%

REM K4: requires results/data/[mode/]dataset_properties.csv (K1,
REM src/run_dataset_properties.bat) - if missing, the script fails fail-loud and
REM this .bat continues (no command here stops on error).
echo === fig_regime_map (K4) ===
"venv\python.exe" -m src.figures.fig_regime_map --%MODE%

REM K6: requires results/data/[mode/]exp6_alpha_curves_results.csv
REM (src/run_exp6_alpha_curves.bat) - if missing, the script fails fail-loud and
REM this .bat continues.
echo === fig_alpha_curves (K6) ===
"venv\python.exe" -m src.figures.fig_alpha_curves --%MODE%

REM K10 (documentation/2026-09-12_plan_smeru_clanku.md): "faithful map" hero
REM figure - requires exp1_dr_benchmark and exp1_cluster_geometry to have already run.
echo === fig_faithful_map (K10) ===
"venv\python.exe" -m src.figures.fig_faithful_map --%MODE%

REM Q1 step 2 (reserse/2026-09-14_specifikace_rozsireni_q1.md, A.9 item 10) -
REM requires results/data/[mode/]regime_candidates_screen.csv
REM (src/run_screen_regime_candidates.bat) and exp1_dr_benchmark_results.csv with
REM hold-out datasets (src\run_exp1_dr_benchmark.bat <mode> --datasets
REM holdout) - if missing, the script fails fail-loud and this .bat continues.
echo === fig_holdout_paired (Q1 step 2) ===
"venv\python.exe" -m src.figures.fig_holdout_paired --%MODE%

REM Stratification of E1/E6 by distance-concentration regime - requires
REM exp6_alpha_curves and dataset_properties.
echo === fig_alpha_gain_by_regime ===
"venv\python.exe" -m src.figures.fig_alpha_gain_by_regime --%MODE%

REM Tightness of Proposition 2 bounds - requires exp8_prop2_check.
echo === fig_prop2_tightness ===
"venv\python.exe" -m src.figures.fig_prop2_tightness --%MODE%

REM Prediction of R_near decline from concentration - requires exp8_prop2_check.
echo === fig_prop2_alpha_prediction ===
"venv\python.exe" -m src.figures.fig_prop2_alpha_prediction --%MODE%

REM Metric fidelity with known ground truth (E9) - requires exp9_metric_fidelity.
echo === fig_metric_fidelity_boxes ===
"venv\python.exe" -m src.figures.fig_metric_fidelity_boxes --%MODE%

REM Embedding gallery E9 - requires exp9_metric_fidelity and saved embeddings.
echo === fig_metric_fidelity_gallery ===
"venv\python.exe" -m src.figures.fig_metric_fidelity_gallery --%MODE%

REM Correlations of embedding quality metrics - requires exp1_dr_benchmark and
REM exp1_cluster_geometry.
echo === fig_metric_correlations ===
"venv\python.exe" -m src.figures.fig_metric_correlations --%MODE%

REM Veta 3 local quadratic law (identifiability, reserse
REM 2026-09-17_zostreni_propozice2.md section 10) - requires
REM run_exp10_identifiability_check.bat.
echo === fig_identifiability_law ===
"venv\python.exe" -m src.figures.fig_identifiability_law --%MODE%

REM Replaces the exp13_neighbor_survival main-text table (2026-09-17
REM article-shortening task) - requires run_exp13_neighbor_survival.bat.
echo === fig_neighbor_survival ===
"venv\python.exe" -m src.figures.fig_neighbor_survival --%MODE%

REM Flagship figure (2026-09-17): MDS vs. tuned alpha-Sammon vs. t-SNE with
REM original-space k-NN edges on the helix dataset - requires
REM run_exp6_alpha_curves.bat/run_exp12_alpha_grid_extension.bat/
REM run_exp1_dr_benchmark.bat to have already completed in the same mode.
REM NOTE: --quick has no 'helix' data in exp1_dr_benchmark.quick/
REM exp6_alpha_curves.quick - only --smoke/--full are usable (see
REM fig_neighborhood_problem.py's module docstring).
echo === fig_neighborhood_problem ===
"venv\python.exe" -m src.figures.fig_neighborhood_problem --%MODE%

REM E15 downstream discovery task (2026-09-18, DAMI editor objection about a
REM missing "better map -> better finding" task) - requires
REM run_exp15_discovery_task_stats.bat (which itself requires exp15_discovery_task).
echo === fig_discovery_task ===
"venv\python.exe" -m src.figures.fig_discovery_task --%MODE%

echo === done: results\figures\ and clanek\img\ ===
call "%~dp0common\no_sleep_off.bat"
endlocal
