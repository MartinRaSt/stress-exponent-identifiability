# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-09
# License: see the LICENSE file in the repository root
"""
Main connecting script: loads the results of all experiments E1-E7
(results/data/[<mode>/]exp*_results.csv), computes aggregates (median+-IQR
over seeds) and statistics (src/experiments/stats.py: Friedman/Nemenyi),
writes summary tables to results/tables/[<mode>/]*.csv + *.tex (booktabs,
\\input-able in the article), the main-text tables (K12b,
src/experiments/report_tables.py), calls all figures (src/figures/*.py)
and finally the number macros (K12a, src/experiments/export_numbers.py).

MODES (author's rule 2026-09-13, ~/.claude/CLAUDE.md "Separating smoke/quick
from full outputs"): --full writes to results/tables/, results/figures/ and
clanek/generated/; --quick and --smoke write EXCLUSIVELY to
results/tables/<mode>/, results/figures/<mode>/ and
results/tables/<mode>/numbers.tex (paths always go through
src/common/config.py::get_mode_path/get_tables_dir/get_generated_dir).
Test: tests/test_mode_isolation.py.

Flags: --no-figures (statistics + tables + macros only, minutes),
--figures-only (Pareto analysis + figures only).

A missing input CSV (e.g. the experiment has not been run yet) is reported
as a warning and the corresponding table/figure is skipped - substitute
content is never fabricated.

Run: venv\\python.exe -m src.main [--quick|--full|--smoke] [--no-figures|--figures-only]
or: src\\run_main.bat [quick|full|smoke] [--no-figures|--figures-only]
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.common.checkpoint import results_csv_path
from src.common.config import get_generated_dir, get_mode_path, get_tables_dir
from src.common.logging_utils import get_logger, wall_clock
from src.experiments.config_experiments import load_experiments_config
from src.experiments.exp4_relative import compute_exp4_relative
from src.experiments.exp_common import add_mode_args, keep_system_awake, resolve_experiment_name, resolve_mode
from src.experiments.report_tables import (
    write_booktabs_tex,
    write_exp1_main_table,
    write_exp3_group_tables,
    write_exp4_relative_table,
    write_exp5_factorial_summary,
    write_exp7_summary,
)
from src.experiments.stats import _STATS_EXPERIMENTS, aggregate_median_iqr, run_stats_for_experiment

MAIN_NAME = "main"

# metrics shown in the summary tables (median+-IQR); the rest are available
# in the full *_results.csv but would clutter the table
_SUMMARY_METRICS = ["stress_scale_invariant", "auc_rnx", "trustworthiness_k7", "sammon_stress", "wall_time_sec"]

_ALL_BASE_EXPERIMENTS = [
    "exp1_dr_benchmark", "exp2_convergence", "exp2_scaling", "exp2_sparse",
    "exp3_graph_layout", "exp4_temporal", "exp5_ablation",
]


def _summary_table_for_experiment(base_experiment_name: str, mode: str, logger) -> pd.DataFrame | None:
    """Median+-IQR (dataset, method) for the available primary metrics of the given experiment."""
    csv_path = results_csv_path(resolve_experiment_name(base_experiment_name, mode))
    if not csv_path.exists():
        logger.warning("Summary: %s does not exist, skipping.", csv_path)
        return None
    df = pd.read_csv(csv_path)
    available_metrics = [m for m in _SUMMARY_METRICS if m in df.columns]
    if not available_metrics:
        logger.warning("Summary: none of the expected metrics are in %s, skipping.", csv_path)
        return None

    pieces = []
    for metric in available_metrics:
        agg = aggregate_median_iqr(df, metric)
        if agg.empty:
            continue
        agg = agg.rename(columns={"median": f"{metric}_median", "iqr": f"{metric}_iqr"})
        agg = agg.drop(columns=["n_seeds"])
        pieces.append(agg.set_index(["dataset", "method"]))
    if not pieces:
        return None
    merged = pieces[0]
    for p in pieces[1:]:
        merged = merged.join(p, how="outer")
    return merged.reset_index()


def _run_all_stats(mode: str, logger) -> None:
    """Friedman/Nemenyi for E1 (all methods - supplement) and E3 (distance/
    native across all graphs - supplement). Main text: E1 only main_methods
    (report_tables.write_exp1_main_table) and E3 small/large graphs
    (report_tables.write_exp3_group_tables). Since K12c, E5 is not tested
    with Friedman (117 combinations), see write_exp5_factorial_summary."""
    stats_cfg = load_experiments_config()["stats"]
    primary_metrics: dict[str, str] = stats_cfg["primary_metrics"]

    for base_experiment_name in _STATS_EXPERIMENTS:
        csv_path = results_csv_path(resolve_experiment_name(base_experiment_name, mode))
        if not csv_path.exists():
            logger.warning("Statistics: %s does not exist, skipping experiment %s.", csv_path, base_experiment_name)
            continue
        df = pd.read_csv(csv_path)

        if base_experiment_name == "exp3_graph_layout":
            df_native = df[~df["dataset"].str.contains("__", regex=False)]
            df_distance = df[df["dataset"].str.contains("__", regex=False)]
            for metric, direction in primary_metrics.items():
                run_stats_for_experiment(base_experiment_name, metric, direction, logger, df=df_distance, output_suffix="_distance", mode=mode)
                run_stats_for_experiment(base_experiment_name, metric, direction, logger, df=df_native, output_suffix="_native", mode=mode)
        else:
            for metric, direction in primary_metrics.items():
                run_stats_for_experiment(base_experiment_name, metric, direction, logger, df=df, mode=mode)


def _write_tables(mode: str, logger) -> Path:
    tables_dir = get_tables_dir(mode)
    for base_name in _ALL_BASE_EXPERIMENTS:
        summary = _summary_table_for_experiment(base_name, mode, logger)
        if summary is None:
            continue
        out_csv = tables_dir / f"{base_name}_summary.csv"
        summary.to_csv(out_csv, index=False)
        out_tex = tables_dir / f"{base_name}_summary.tex"
        write_booktabs_tex(
            summary, out_tex,
            caption=f"Summary (median $\\pm$ IQR over seeds): {base_name}",
            label=f"tab:{base_name}_summary",
        )
        logger.info("Table written: %s, %s (%d rows).", out_csv, out_tex, summary.shape[0])
    return tables_dir


def _run_exp4_relative(mode: str, logger) -> None:
    """K9: computes results/data/[<mode>/]exp4_relative.csv BEFORE the tables/
    macros (see documentation/2026-09-14_exp4_relative_poradi.md) - previously
    only fig_temporal_pareto.py did this, which does not run at all under
    `--no-figures` and otherwise runs AFTER the tables/macros, so
    `write_exp4_relative_table` and export_numbers.py read stale/missing
    data. A missing exp4_temporal_results.csv (E4 has not run yet in this
    mode) is just logged - it does not stop the rest of main.py (same
    pattern as the other tables)."""
    try:
        compute_exp4_relative(mode, logger)
    except FileNotFoundError as exc:
        logger.warning("exp4_relative (K9) skipped (missing data): %s", exc)
    except Exception as exc:  # intentionally broad - must not stop the rest of tables/macros
        logger.warning("exp4_relative (K9) failed: %s\n%s", exc, traceback.format_exc())


def _write_report_tables(mode: str, logger) -> None:
    """K12b-f: main-text/supplement tables (src/experiments/report_tables.py).
    One failing table does not stop the others (traceback is logged)."""
    for name, fn in [
        ("E1 main table (main_methods) + Friedman/Nemenyi over main_methods only", write_exp1_main_table),
        ("E3 small/large graphs", write_exp3_group_tables),
        ("E4 relative changes vs lambda", write_exp4_relative_table),
        ("E5 factorial summary", write_exp5_factorial_summary),
        ("E7 rank-weighted stress (supplement)", write_exp7_summary),
    ]:
        try:
            fn(mode, logger)
        except Exception as exc:  # intentionally broad - one table must not stop the others
            logger.warning("Table '%s' failed: %s\n%s", name, exc, traceback.format_exc())


def _copy_stats_tables(mode: str, logger) -> None:
    """Copies the already generated results/data/[<mode>/]stats_*.csv into
    results/tables/[<mode>/] as booktabs .tex (methods x primary metrics)."""
    tables_dir = get_tables_dir(mode)
    data_dir = get_mode_path("results_data_dir", mode)
    for p in sorted(data_dir.glob("stats_*.csv")):
        df = pd.read_csv(p)
        out_csv = tables_dir / p.name
        df.to_csv(out_csv, index=False)
        cols = ["method", "avg_rank", "median_across_datasets", "iqr_across_datasets", "cd", "friedman_pvalue", "kendall_w"]
        cols = [c for c in cols if c in df.columns]
        out_tex = tables_dir / p.with_suffix(".tex").name
        write_booktabs_tex(df[cols], out_tex, caption=p.stem.replace("_", " "), label=f"tab:{p.stem}")
        logger.info("Statistics table copied: %s, %s.", out_csv, out_tex)


_FIGURE_MODULES = [
    "fig_gallery_methods", "fig_before_after", "fig_alpha_strip", "fig_sgd_convergence",
    "fig_runtime_scaling", "fig_temporal_trajectories", "fig_rnx_curves", "fig_cd_diagram",
    # K11 (documentation/2026-09-12_plan_smeru_clanku.md): focused figure
    # (main text) + original all-methods variant (supplement, renamed)
    "fig_graph_layouts", "fig_graph_layouts_all",
    # K3/K9/K4 (documentation/2026-09-12_plan_smeru_clanku.md) - require
    # pareto_analysis (see _run_pareto_analysis) resp. dataset_properties.csv
    # (K1, run separately by the author via src/run_dataset_properties.bat)
    "fig_pareto_front", "fig_temporal_pareto", "fig_regime_map",
    # K6 - requires exp6_alpha_curves_results.csv (src/run_exp6_alpha_curves.bat)
    "fig_alpha_curves",
    # K10 - "faithful map" hero figure (exp1_dr_benchmark + exp1_cluster_geometry)
    "fig_faithful_map",
    # Q1 step 2 (reserse/2026-09-14_specifikace_rozsireni_q1.md, A.9 item 10) -
    # requires regime_candidates_screen.csv (src/run_screen_regime_candidates.bat)
    # and exp1_dr_benchmark_results.csv with hold-out datasets
    # (src\run_exp1_dr_benchmark.bat <mode> --datasets holdout); a missing
    # input is just logged (see _run_all_figures), does not stop the other figures.
    "fig_holdout_paired",
]


def _run_module_main(module_path: str, mode: str) -> None:
    """Calls the module's `main()` with argv = [--<mode>] (each script has its own argparse)."""
    import importlib

    mod = importlib.import_module(module_path)
    old_argv = sys.argv
    sys.argv = [module_path.rsplit(".", 1)[-1], f"--{mode}"]
    try:
        mod.main()
    finally:
        sys.argv = old_argv


def _run_all_figures(mode: str, logger) -> None:
    for mod_name in _FIGURE_MODULES:
        try:
            _run_module_main(f"src.figures.{mod_name}", mode)
        except FileNotFoundError as exc:
            logger.warning("Figure %s skipped (missing data): %s", mod_name, exc)
        except Exception as exc:  # intentionally broad - one failed figure must not stop the others
            logger.warning("Figure %s failed: %s\n%s", mod_name, exc, traceback.format_exc())


def _run_pareto_analysis(mode: str, logger) -> None:
    """K3: Pareto analysis over exp1_dr_benchmark_results.csv (must run BEFORE
    fig_pareto_front/fig_regime_map, which read its output). A missing input
    (E1 has not run yet in this mode) is just logged - does not stop the rest of main.py."""
    try:
        _run_module_main("src.experiments.pareto_analysis", mode)
    except FileNotFoundError as exc:
        logger.warning("Pareto analysis (K3) skipped (missing data): %s", exc)
    except Exception as exc:
        logger.warning("Pareto analysis (K3) failed: %s\n%s", exc, traceback.format_exc())


def _run_export_numbers(mode: str, logger) -> None:
    """K12a: number macros. 'full' -> clanek/generated/numbers.tex; 'quick'/
    'smoke' -> results/tables/<mode>/numbers.tex (never into the article), see
    src/common/config.py::get_generated_dir."""
    try:
        from src.experiments.export_numbers import main as export_numbers_main

        export_numbers_main(mode=mode)
    except Exception as exc:
        logger.warning("export_numbers (K12a) failed: %s\n%s", exc, traceback.format_exc())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Main connecting script: aggregation, statistics, tables, figures, number macros.")
    add_mode_args(parser)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--no-figures", action="store_true", help="statistics + tables + number macros only (no figures)")
    group.add_argument("--figures-only", action="store_true", help="Pareto analysis + figures only (no statistics/tables/macros)")
    return parser.parse_args(argv)


def run(mode: str, no_figures: bool = False, figures_only: bool = False, logger=None) -> None:
    """Full main run for the given mode (also callable from tests without argparse)."""
    logger = logger or get_logger(MAIN_NAME, mode=mode)
    logger.info("main: mode=%s, no_figures=%s, figures_only=%s; tables -> %s, macros -> %s",
                mode, no_figures, figures_only, get_tables_dir(mode), get_generated_dir(mode))

    # prevent the computer from sleeping/hibernating during the run (second
    # layer on top of powercfg in src/run_main.bat, see
    # src/experiments/exp_common.py::keep_system_awake)
    with keep_system_awake():
        if not figures_only:
            with wall_clock(logger, "main: statistics (Friedman/Nemenyi)"):
                _run_all_stats(mode, logger)

            with wall_clock(logger, "main: tables"):
                _write_tables(mode, logger)
                _run_exp4_relative(mode, logger)
                _write_report_tables(mode, logger)
                _copy_stats_tables(mode, logger)

        with wall_clock(logger, "main: Pareto analysis (K3)"):
            _run_pareto_analysis(mode, logger)

        if not no_figures:
            with wall_clock(logger, "main: figures"):
                _run_all_figures(mode, logger)

        if not figures_only:
            with wall_clock(logger, "main: number macros (K12a)"):
                _run_export_numbers(mode, logger)

    logger.info("Done. Tables: %s, figures: %s.", get_tables_dir(mode), get_mode_path("results_figures_dir", mode))


def main() -> None:
    args = parse_args()
    run(resolve_mode(args), no_figures=args.no_figures, figures_only=args.figures_only)


if __name__ == "__main__":
    main()
