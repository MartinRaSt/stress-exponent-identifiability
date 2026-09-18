#!/usr/bin/env bash
set -euo pipefail
# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-09
# License: see the LICENSE file in the repository root
#
# Runs all 9 data figures in src/figures/ independently (without statistics/
# tables - see src/run_main.sh for the full pipeline). Parameter: quick|full|smoke
# (default full - determines which results/data/ subdirectory the inputs are
# read from, see src/experiments/exp_common.py::resolve_experiment_name). Missing
# input data for one figure (FileNotFoundError) does not stop the others -
# each command runs independently, errors are printed to the console.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Disable computer standby/hibernation for the duration of the run (systemd-inhibit, see
# src/common/no_sleep.sh) - author requirement.
source "$ROOT/src/common/no_sleep.sh"
sammon_no_sleep_reexec "$0" "$@"

PYTHON="${PYTHON:-$ROOT/venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: project venv Python interpreter not found: $PYTHON" >&2
    echo "Set the PYTHON environment variable to the project's venv Python path, e.g.:" >&2
    echo "  PYTHON=/path/to/venv/bin/python $0 $*" >&2
    exit 1
fi

MODE="${1:-full}"

# Run a command, on error just print it and continue (see description above) - temporarily
# disable errexit only for this one command.
run_step() {
    local desc="$1"
    shift
    echo "=== $desc ==="
    set +e
    "$@"
    local rc=$?
    set -e
    if [ "$rc" -ne 0 ]; then
        echo "ERROR: $desc failed (rc=$rc)" >&2
    fi
}

run_step "fig_gallery_methods" "$PYTHON" -m src.figures.fig_gallery_methods --"$MODE"
run_step "fig_before_after" "$PYTHON" -m src.figures.fig_before_after --"$MODE"
run_step "fig_alpha_strip" "$PYTHON" -m src.figures.fig_alpha_strip --"$MODE"
run_step "fig_sgd_convergence" "$PYTHON" -m src.figures.fig_sgd_convergence --"$MODE"
run_step "fig_runtime_scaling" "$PYTHON" -m src.figures.fig_runtime_scaling --"$MODE"
run_step "fig_temporal_trajectories" "$PYTHON" -m src.figures.fig_temporal_trajectories --"$MODE"
run_step "fig_rnx_curves" "$PYTHON" -m src.figures.fig_rnx_curves --"$MODE"
run_step "fig_cd_diagram" "$PYTHON" -m src.figures.fig_cd_diagram --"$MODE"

# The main-text CD diagram is the one over `report.main_methods` for
# scale-invariant stress; the default call above only produces the auc_rnx
# variant, so the remaining variants used by the paper and the supplement
# are generated explicitly (everything regenerable in one command).
for METRIC in stress_scale_invariant trustworthiness_k7 auc_rnx; do
    run_step "fig_cd_diagram exp1_dr_benchmark_main/$METRIC" "$PYTHON" -m src.figures.fig_cd_diagram --"$MODE" --experiment exp1_dr_benchmark_main --metric "$METRIC"
done

# K11 (documentation/2026-09-12_plan_smeru_clanku.md): focused figure of
# graph layouts for the main text (polbooks/football/cora x 2 distances
# x 5 methods) - requires src/run_exp3_graph_layout.sh (E3).
run_step "fig_graph_layouts (K11, main text)" "$PYTHON" -m src.figures.fig_graph_layouts --"$MODE"

# K11: supplement variant (all graphs x all E3 methods), original
# fig_graph_layouts.py renamed.
run_step "fig_graph_layouts_all (K11, supplement)" "$PYTHON" -m src.figures.fig_graph_layouts_all --"$MODE"

# K3 (documentation/2026-09-12_plan_smeru_clanku.md): fig_pareto_front reads
# results/data/[mode/]pareto_membership.csv + results/tables/pareto_median_front.csv
# (src/experiments/pareto_analysis.py) - must run BEFORE the figure.
run_step "pareto_analysis (K3, prerequisite for fig_pareto_front)" "$PYTHON" -m src.experiments.pareto_analysis --"$MODE"

run_step "fig_pareto_front (K3)" "$PYTHON" -m src.figures.fig_pareto_front --"$MODE"

run_step "fig_temporal_pareto (K9)" "$PYTHON" -m src.figures.fig_temporal_pareto --"$MODE"

# K4: requires results/data/[mode/]dataset_properties.csv (K1,
# src/run_dataset_properties.sh) - if missing, the script fails fail-loud and
# this script continues (no command here stops on error).
run_step "fig_regime_map (K4)" "$PYTHON" -m src.figures.fig_regime_map --"$MODE"

# K6: requires results/data/[mode/]exp6_alpha_curves_results.csv
# (src/run_exp6_alpha_curves.sh) - if missing, the script fails fail-loud and
# this script continues.
run_step "fig_alpha_curves (K6)" "$PYTHON" -m src.figures.fig_alpha_curves --"$MODE"

# K10 (documentation/2026-09-12_plan_smeru_clanku.md): "faithful map" hero
# figure - requires exp1_dr_benchmark and exp1_cluster_geometry to have already run.
run_step "fig_faithful_map (K10)" "$PYTHON" -m src.figures.fig_faithful_map --"$MODE"

# Q1 step 2 (reserse/2026-09-14_specifikace_rozsireni_q1.md, A.9 item 10) -
# requires results/data/[mode/]regime_candidates_screen.csv
# (src/run_screen_regime_candidates.sh) and exp1_dr_benchmark_results.csv with
# hold-out datasets (src/run_exp1_dr_benchmark.sh <mode> --datasets
# holdout) - if missing, the script fails fail-loud and this script continues.
run_step "fig_holdout_paired (Q1 step 2)" "$PYTHON" -m src.figures.fig_holdout_paired --"$MODE"

# Stratification of E1/E6 by distance-concentration regime - requires
# exp6_alpha_curves and dataset_properties.
echo "=== fig_alpha_gain_by_regime ==="
"$PYTHON" -m src.figures.fig_alpha_gain_by_regime --"$MODE"
# Tightness of Proposition 2 bounds - requires exp8_prop2_check.
echo "=== fig_prop2_tightness ==="
"$PYTHON" -m src.figures.fig_prop2_tightness --"$MODE"
# Prediction of R_near decline from concentration - requires exp8_prop2_check.
echo "=== fig_prop2_alpha_prediction ==="
"$PYTHON" -m src.figures.fig_prop2_alpha_prediction --"$MODE"
# Metric fidelity with known ground truth (E9) - requires exp9_metric_fidelity.
echo "=== fig_metric_fidelity_boxes ==="
"$PYTHON" -m src.figures.fig_metric_fidelity_boxes --"$MODE"
# Embedding gallery E9 - requires exp9_metric_fidelity and saved embeddings.
echo "=== fig_metric_fidelity_gallery ==="
"$PYTHON" -m src.figures.fig_metric_fidelity_gallery --"$MODE"
# Correlations of embedding quality metrics - requires exp1_dr_benchmark and
# exp1_cluster_geometry.
echo "=== fig_metric_correlations ==="
"$PYTHON" -m src.figures.fig_metric_correlations --"$MODE"

# Veta 3 local quadratic law (identifiability, reserse
# 2026-09-17_zostreni_propozice2.md section 10) - requires
# run_exp10_identifiability_check.sh.
run_step "fig_identifiability_law" "$PYTHON" -m src.figures.fig_identifiability_law --"$MODE"

# Replaces the exp13_neighbor_survival main-text table (2026-09-17
# article-shortening task) - requires run_exp13_neighbor_survival.sh.
run_step "fig_neighbor_survival" "$PYTHON" -m src.figures.fig_neighbor_survival --"$MODE"

# Flagship figure (2026-09-17): MDS vs. tuned alpha-Sammon vs. t-SNE with
# original-space k-NN edges on the helix dataset - requires
# run_exp6_alpha_curves.sh/run_exp12_alpha_grid_extension.sh/
# run_exp1_dr_benchmark.sh to have already completed in the same mode.
# NOTE: --quick has no 'helix' data in exp1_dr_benchmark.quick/
# exp6_alpha_curves.quick - only --smoke/--full are usable (see
# fig_neighborhood_problem.py's module docstring).
run_step "fig_neighborhood_problem" "$PYTHON" -m src.figures.fig_neighborhood_problem --"$MODE"

# E15 downstream discovery task (2026-09-18, DAMI editor objection about a
# missing "better map -> better finding" task) - requires
# run_exp15_discovery_task_stats.sh (which itself requires exp15_discovery_task).
run_step "fig_discovery_task" "$PYTHON" -m src.figures.fig_discovery_task --"$MODE"

echo "=== done: results/figures/ and clanek/img/ ==="
