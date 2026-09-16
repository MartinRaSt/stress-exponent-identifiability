#!/usr/bin/env bash
set -euo pipefail
# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-11
# License: see the LICENSE file in the repository root
#
# Preparation for a rerun after the 2026-09-11 fixes (see
# documentation/2026-09-11_plan_prebehu.md): removes only the rows affected by
# the fixes from the full CSVs in results/data/ (with a .bak_<timestamp> backup),
# so that a subsequent `src/run_all_experiments.sh full` computes only the missing
# combinations (resume). New methods (tsne_auto, umap_auto) and new E4 datasets
# are added automatically, nothing needs to be deleted for them.
# Usage: src/prepare_rerun_20260911.sh [dry]   (dry = print only)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: project venv Python interpreter not found: $PYTHON" >&2
    echo "Set the PYTHON environment variable to the project's venv Python path, e.g.:" >&2
    echo "  PYTHON=/path/to/venv/bin/python $0 $*" >&2
    exit 1
fi

ARG1="${1:-}"
DRY=""
if [ "${ARG1,,}" = "dry" ]; then
    DRY="--dry-run"
fi

run_remove_rows() {
    if [ -n "$DRY" ]; then
        "$PYTHON" -m src.experiments.remove_rows "$@" "$DRY"
    else
        "$PYTHON" -m src.experiments.remove_rows "$@"
    fi
}

fail_and_exit() {
    echo "ERROR while removing rows - DO NOT run the rerun, check the output above." >&2
    exit 1
}

echo "=== E1: standardized tabular datasets (all methods) ==="
run_remove_rows --csv results/data/exp1_dr_benchmark_results.csv --dataset iris,wine,breast_cancer,glass,ionosphere,seeds,yeast,vehicle,segment,satimage,letter,spambase || fail_and_exit
echo "=== E1: sammon_classic (step-halving) on all datasets ==="
run_remove_rows --csv results/data/exp1_dr_benchmark_results.csv --method sammon_classic || fail_and_exit

echo "=== E2 scaling: sparse solvers (full stress) + GPU SMACOF n=20000 (tiling fix) ==="
run_remove_rows --csv results/data/exp2_scaling_results.csv --method sparse_smacof__cpu,sparse_sgd__cpu || fail_and_exit
run_remove_rows --csv results/data/exp2_scaling_results.csv --dataset gaussian_clusters_n20000 --method smacof__cuda || fail_and_exit
echo "=== E2 sparse gap: whole file (new stress_sparse_terms column) ==="
if [ -z "$DRY" ]; then
    if [ -e "results/data/exp2_sparse_results.csv" ]; then
        mv -f "results/data/exp2_sparse_results.csv" "results/data/exp2_sparse_results.csv.bak_20260911"
    fi
else
    echo "[dry] would delete results/data/exp2_sparse_results.csv"
fi

echo "=== E3: sammon_classic (step-halving) ==="
run_remove_rows --csv results/data/exp3_graph_layout_results.csv --method sammon_classic || fail_and_exit

echo "=== E4: SMACOF rows primary_school (fix for lambda without anchors) ==="
run_remove_rows --csv results/data/exp4_temporal_results.csv --dataset primary_school_temporal --method lambda0.0_alpha1.0_smacof,lambda0.01_alpha1.0_smacof,lambda0.03_alpha1.0_smacof,lambda0.1_alpha1.0_smacof,lambda0.3_alpha1.0_smacof,lambda1.0_alpha1.0_smacof,lambda3.0_alpha1.0_smacof,lambda10.0_alpha1.0_smacof || fail_and_exit
if [ -z "$DRY" ]; then
    if [ -e "results/data/exp4_stats.csv" ]; then
        mv -f "results/data/exp4_stats.csv" "results/data/exp4_stats.csv.bak_20260911"
    fi
    if [ -e "results/data/exp4_trajectories.csv" ]; then
        mv -f "results/data/exp4_trajectories.csv" "results/data/exp4_trajectories.csv.bak_20260911"
    fi
fi

echo "=== E5: iris (standardization) ==="
run_remove_rows --csv results/data/exp5_ablation_results.csv --dataset iris || fail_and_exit

if [ -z "$DRY" ]; then
    echo "=== DONE markers deleted so run_all_experiments.sh does not show stale state ==="
    rm -f results/data/exp1_dr_benchmark_DONE.txt results/data/exp2_solver_scaling_DONE.txt \
          results/data/exp3_graph_layout_DONE.txt results/data/exp4_temporal_DONE.txt \
          results/data/exp5_ablation_DONE.txt 2>/dev/null || true
fi
echo "=== DONE. Next step: src/run_all_experiments.sh full ==="
exit 0
