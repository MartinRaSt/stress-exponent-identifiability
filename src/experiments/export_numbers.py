# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-12
# License: see the LICENSE file in the repository root
"""
K12a (documentation/2026-09-12_plan_smeru_clanku.md) - generates
`numbers.tex` with `\\newcommand` macros for ALL numbers cited by the article
text. Each macro is on its own line with a comment
`% source: <csv>, <column>, <aggregation>` - the writer NEVER types a number
into the text by hand, only references the macro.

MODES (author's rule, 2026-09-13): --full reads results/data/ + results/tables/
and writes clanek/generated/numbers.tex; --quick/--smoke read EXCLUSIVELY
results/data/<mode>/ + results/tables/<mode>/ and write
results/tables/<mode>/numbers.tex (never into the article) - paths via
src/common/config.py::get_mode_path / get_generated_dir.

A missing input CSV/column = the macro (group) is SKIPPED (a substitute
number is never fabricated), with a clear WARNING in the log - the other
macros are still generated normally (same resilience principle as
`src/main.py::_run_all_figures`). Every rerun OVERWRITES the whole
`numbers.tex` (deterministic, no state to resume).

LaTeX \\newcommand names may contain ONLY letters (no digits/underscores) -
see `_macro_line` (fail-loud on violation); numeric factor levels are
converted to words (`num_to_words`: 0.25 -> ZeroPointTwoFive). Numbers are
formatted to 2-3 significant digits (`fmt_sig`), no LaTeX-unsafe characters.

Run: venv\\python.exe -m src.experiments.export_numbers [--quick|--full|--smoke]
or: src\\run_export_numbers.bat [quick|full|smoke]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import re

import numpy as np
import pandas as pd

from src.common.config import get_generated_dir, get_mode_path, load_config
from src.common.logging_utils import get_logger
from src.experiments import exp4_common
from src.experiments.config_experiments import load_experiments_config, resolve_experiment_config
from src.experiments.exp_common import parse_mode_args
from src.experiments.report_tables import (
    EXP1_MAIN_TABLE,
    EXP3_MAIN_TABLE,
    EXP4_RELATIVE_TABLE,
    EXP5_FACTORIAL_SUMMARY,
    EXP5_INIT_EQUIVALENCE,
    EXP5_METRICS,
    add_exp3_groups,
    exp3_large_graph_threshold,
)

MODULE_NAME = "export_numbers"

# method (name in the CSV) -> an ASCII-letter suffix for the macro name (LaTeX
# \newcommand only allows letters, no digits/underscores)
METHOD_MACRO_SUFFIX = {
    "sammon_alpha_auto": "AlphaAuto",
    "sammon_alpha_pred": "AlphaPred",
    "sammon_alpha0_smacof": "AlphaZero",
    "sammon_alpha_smacof": "AlphaOne",
    "sammon_alpha2_smacof": "AlphaTwo",
    "sammon_classic": "SammonClassic",
    "mds": "Mds",
    "pca": "Pca",
    "tsne": "Tsne",
    "tsne_auto": "TsneAuto",
    "umap": "Umap",
    "umap_auto": "UmapAuto",
    "pacmap": "Pacmap",
    "trimap": "Trimap",
    "densmap": "Densmap",
    "phate": "Phate",
    "spectral": "Spectral",
    "spring": "Spring",
    "kamada_kawai": "KamadaKawai",
}

# E1/E3 methods considered the "Sammon/MDS family" (stress/metric methods)
# vs. the "neighbor family" (config report.neighbor_methods) for the win
# counts x/N (see reserse/2026-09-12_proc_sammon_a_smery_clanku.md section 1).
STRESS_FAMILY_METHODS = [
    "pca", "mds", "sammon_classic",
    "sammon_alpha0_smacof", "sammon_alpha_smacof", "sammon_alpha2_smacof", "sammon_alpha_auto", "sammon_alpha_pred",
    "sammon_multiscale", "sammon_sgd_stab", "sammon_sparse_smacof",
]

# (metric, direction, ASCII name for the macro) for E1 method medians
_MEDIAN_METRIC_SPECS = [
    ("shepard_spearman_rho", "max", "Shepard"),
    ("q_global", "max", "Qglobal"),
    ("q_local", "max", "Qlocal"),
    ("stress_scale_invariant", "min", "Stress"),
    ("auc_rnx", "max", "Aucrnx"),
    ("trustworthiness_k7", "max", "Trustworthiness"),
    ("wall_time_sec", "min", "Time"),
]
_MEDIAN_METHODS = [
    "sammon_alpha_auto", "sammon_alpha_pred", "sammon_alpha0_smacof", "sammon_alpha_smacof",
    "tsne_auto", "umap_auto", "pacmap", "trimap", "densmap", "phate",
]
# cluster-geometry metrics (K10) - source exp1_cluster_geometry_results.csv
# (has them for ALL methods, the E1 CSV only for methods added after K5/K7)
_CLUSTER_GEOMETRY_SPECS = [
    ("class_spread_spearman", "max", "ClassSpread"),
    ("centroid_dist_spearman", "max", "CentroidDist"),
]

# (metric, direction, ASCII name) for the win counts Sammon/MDS family vs. neighbor family
_WIN_METRIC_SPECS = [
    ("shepard_pearson_r", "max", "ShepardPearson"),
    # The main table reports Shepard rho as SPEARMAN, so the win count quoted
    # next to it must use the same coefficient; the Pearson variant above is
    # kept because other parts of the text refer to it.
    ("shepard_spearman_rho", "max", "ShepardSpearman"),
    ("kruskal_stress1", "min", "KruskalStressOne"),
    ("stress_scale_invariant", "min", "StressScaleInvariant"),
    ("q_global", "max", "Qglobal"),
    ("auc_rnx", "max", "Aucrnx"),
    ("trustworthiness_k7", "max", "TrustworthinessKSeven"),
]

# (stats file name without extension, ASCII name for the macro) - Kendall W and CD
# ('_main' = Friedman only over report.main_methods, K12b; E5 without Friedman since K12c)
_STATS_FILE_STUBS = [
    ("stats_exp1_dr_benchmark_main_auc_rnx", "ExpOneMainAucrnx"),
    ("stats_exp1_dr_benchmark_main_stress_scale_invariant", "ExpOneMainStress"),
    ("stats_exp1_dr_benchmark_main_trustworthiness_k7", "ExpOneMainTrustworthiness"),
    ("stats_exp3_graph_layout_distance_auc_rnx", "ExpThreeDistanceAucrnx"),
    ("stats_exp3_graph_layout_distance_stress_scale_invariant", "ExpThreeDistanceStress"),
    ("stats_exp3_graph_layout_distance_trustworthiness_k7", "ExpThreeDistanceTrustworthiness"),
    ("stats_exp3_graph_layout_native_auc_rnx", "ExpThreeNativeAucrnx"),
    ("stats_exp3_graph_layout_native_stress_scale_invariant", "ExpThreeNativeStress"),
    ("stats_exp3_graph_layout_native_trustworthiness_k7", "ExpThreeNativeTrustworthiness"),
    ("stats_exp3_graph_layout_distance_small_auc_rnx", "ExpThreeDistanceSmallAucrnx"),
    ("stats_exp3_graph_layout_distance_large_auc_rnx", "ExpThreeDistanceLargeAucrnx"),
    ("stats_exp3_graph_layout_distance_small_stress_scale_invariant", "ExpThreeDistanceSmallStress"),
    ("stats_exp3_graph_layout_distance_large_stress_scale_invariant", "ExpThreeDistanceLargeStress"),
]

# E4 (K9) relative-change lambda values - an ASCII suffix for the macro name
# (the names ZeroThree/One/Three kept due to existing references in the text)
_K9_LAMBDAS = [(0.3, "ZeroThree"), (1.0, "One"), (3.0, "Three"), (10.0, "Ten")]
_K9_SOLVERS = [("smacof", "Smacof"), ("sgd", "Sgd")]

# E4 dtsne baseline (Q1 addendum, 06_diskuse.tex): parses method names from
# exp4_temporal_results.csv, so lambda/solver do not have to be listed by hand.
# K16 (2026-09-16): both the regexes and the pairing rule ("nearest stab") are
# SHARED with the primary test exp4_neighbor_metrics.py via
# src/experiments/exp4_common.py (no duplicate logic).
_DTSNE_METHOD_RE = exp4_common.DTSNE_METHOD_RE
_OUR_TEMPORAL_METHOD_RE = exp4_common.OUR_TEMPORAL_METHOD_RE

# E3: methods for which average ranks and medians per group are exported
_EXP3_METHODS = ["sammon_alpha_auto", "sammon_alpha_pred", "sammon_alpha0_smacof", "sammon_alpha_smacof", "tsne", "tsne_auto", "umap", "umap_auto", "mds"]
_EXP3_NATIVE_METHODS = ["spectral", "spring", "kamada_kawai"]
_EXP3_DISTANCE_SUFFIX = {"resistance": "Resistance", "shortest_path": "ShortestPath"}
_EXP3_GROUP_SUFFIX = {"small": "Small", "large": "Large"}

# E7: method -> ASCII suffix
_EXP7_METHOD_SUFFIX = {
    "dist0": "DistZero", "dist1": "DistOne", "dist2": "DistTwo",
    "rank0.5": "RankZeroPointFive", "rank1": "RankOne", "rank2": "RankTwo", "tsne": "Tsne",
}

# E2 scaling: method (solver__device) -> ASCII suffix
_EXP2_METHOD_SUFFIX = {
    "smacof__cuda": "SmacofCuda", "smacof__cpu": "SmacofCpu",
    "sgd_stab__cpu": "SgdStabCpu", "sgd_stab__cuda": "SgdStabCuda",
    "sparse_smacof__cpu": "SparseSmacofCpu", "sparse_sgd__cpu": "SparseSgdCpu",
}

_DIGIT_WORDS = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
_INT_WORDS = {10: "Ten", 11: "Eleven", 12: "Twelve", 13: "Thirteen", 14: "Fourteen", 15: "Fifteen", 16: "Sixteen", 17: "Seventeen", 18: "Eighteen", 19: "Nineteen", 20: "Twenty"}


def num_to_words(x: float | int | str) -> str:
    """Numeric factor level -> ASCII words for the macro name:
    0 -> Zero, 0.25 -> ZeroPointTwoFive, 1 -> One, 3 -> Three, 10 -> Ten."""
    val = float(x)
    if not np.isfinite(val):
        raise ValueError(f"num_to_words: value is not finite: {x}")
    if val.is_integer() and int(val) in _INT_WORDS:
        return _INT_WORDS[int(val)]
    text = f"{val:g}"
    if "e" in text:
        raise ValueError(f"num_to_words: exponential notation not supported: {x}")
    out = []
    for ch in text:
        if ch == ".":
            out.append("Point")
        elif ch == "-":
            out.append("Minus")
        elif ch in _DIGIT_WORDS:
            out.append(_DIGIT_WORDS[ch])
        else:
            raise ValueError(f"num_to_words: unexpected character '{ch}' in {text}")
    return "".join(out)


# Magnitudes outside this range are written in scientific notation. Inside it,
# fixed decimal notation stays (unchanged formatting of every number already in
# the article); outside it, fixed notation is unusable - a p-value of 2.8e-190
# became 190 decimal places and siunitx rejected it as an invalid number.
SCIENTIFIC_MIN_MAGNITUDE = -6
SCIENTIFIC_MAX_MAGNITUDE = 9


def fmt_sig(x: float, sig: int = 3) -> str:
    """Formats a number to `sig` significant digits without thousands
    separators. Fixed decimal notation is used for ordinary magnitudes and
    scientific notation ('e') for extreme ones; both are parsed by siunitx
    `\\num{...}`, which wraps every decimal value (see `_wrap_decimal`)."""
    if not np.isfinite(x):
        raise ValueError(f"Number is not finite (NaN/Inf) - cannot format for a macro: {x}")
    if x == 0:
        return "0"
    magnitude = math.floor(math.log10(abs(x)))
    if magnitude < SCIENTIFIC_MIN_MAGNITUDE or magnitude > SCIENTIFIC_MAX_MAGNITUDE:
        return f"{x:.{max(sig - 1, 0)}e}"
    decimals = max(sig - 1 - magnitude, 0)
    rounded = round(x, decimals)
    if decimals == 0:
        return f"{rounded:.0f}"
    return f"{rounded:.{decimals}f}"


_DECIMAL_RE = re.compile(r"^[+-]?\d+\.\d+([eE][+-]?\d+)?$")


def _wrap_decimal(value: str) -> str:
    """Wraps a decimal number in `\\num{...}` so siunitx typesets it with a
    decimal COMMA (Czech text; `output-decimal-marker={,}` is set in the
    preamble of both versions of the article and the supplement).

    Integers, fractions ("51/51"), lists of names, and values that already
    contain `\\num` are left unchanged - wrapping them would be incorrect."""
    return f"\\num{{{value}}}" if _DECIMAL_RE.match(value) else value


def _macro_line(name: str, value: str, source: str) -> str:
    """Returns one `\\newcommand` line + a source comment (fail-loud on
    disallowed characters in the macro name - LaTeX only allows letters)."""
    if not name.isalpha():
        raise ValueError(f"Macro name '{name}' contains disallowed characters - \\newcommand requires only letters a-zA-Z.")
    return f"\\newcommand{{\\{name}}}{{{_wrap_decimal(value)}}} % source: {source}\n"


class MacroCollector:
    """Collects (name, value, source) triples and guards against duplicate macro names."""

    def __init__(self, logger) -> None:
        self.logger = logger
        self.lines: list[str] = []
        self.seen: set[str] = set()

    def add(self, name: str, value: Any, source: str, sig: int | None = None) -> None:
        if name in self.seen:
            raise ValueError(f"Macro '{name}' is defined twice - a name collision in export_numbers.py.")
        value_str = fmt_sig(float(value), sig) if sig is not None else str(value)
        self.lines.append(_macro_line(name, value_str, source))
        self.seen.add(name)

    def add_if_finite(self, name: str, value: Any, source: str, sig: int = 3) -> bool:
        """Adds a numeric macro; NaN/Inf (e.g. a method without the metric) is skipped with a WARNING."""
        try:
            val = float(value)
        except (TypeError, ValueError):
            val = float("nan")
        if not np.isfinite(val):
            self.logger.warning("Macro '%s' skipped: value is not finite (%s).", name, source)
            return False
        self.add(name, val, source, sig=sig)
        return True

    def try_block(self, description: str, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as exc:
            self.logger.warning("Macros from group '%s' SKIPPED (missing/incomplete input): %s", description, exc)


def _read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input: {path}")
    return pd.read_csv(path)


def _median_across_seeds(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    ok = df[(df["status"] == "ok") & df[metric].notna()]
    return ok.groupby(["dataset", "method"])[metric].median().unstack("method")


def _family_best_per_dataset(wide: pd.DataFrame, family: list[str], direction: str) -> pd.Series:
    cols = [c for c in family if c in wide.columns]
    if not cols:
        raise KeyError(f"None of the methods in family {family} are among the columns {list(wide.columns)}.")
    return wide[cols].max(axis=1) if direction == "max" else wide[cols].min(axis=1)


def _suffix(method_name: str) -> str:
    if method_name not in METHOD_MACRO_SUFFIX:
        raise KeyError(f"Method '{method_name}' has no ASCII suffix in METHOD_MACRO_SUFFIX.")
    return METHOD_MACRO_SUFFIX[method_name]


# ---------------------------------------------------------------------------
# macro groups
# ---------------------------------------------------------------------------
def _add_counts(mc: MacroCollector, data_dir: Path) -> None:
    def block() -> None:
        e1 = _read_csv_required(data_dir / "exp1_dr_benchmark_results.csv")
        e1_ok = e1[e1["status"] == "ok"]
        mc.add("numExpOneDatasets", e1_ok["dataset"].nunique(), "exp1_dr_benchmark_results.csv, dataset, nunique")
        mc.add("numExpOneMethods", e1_ok["method"].nunique(), "exp1_dr_benchmark_results.csv, method, nunique")
        mc.add("numExpOneSeeds", e1_ok["seed"].nunique(), "exp1_dr_benchmark_results.csv, seed, nunique")
        main_methods = load_experiments_config()["report"]["main_methods"]
        mc.add("numExpOneMainMethods", len(main_methods), "config_experiments.yaml, report.main_methods, count")
    mc.try_block("counts E1", block)

    def block3() -> None:
        e3 = _read_csv_required(data_dir / "exp3_graph_layout_results.csv")
        e3_ok = e3[e3["status"] == "ok"]
        e3_dist = e3_ok[e3_ok["dataset"].str.contains("__", regex=False)]
        mc.add("numExpThreeGraphs", e3_dist["base_graph"].nunique(), "exp3_graph_layout_results.csv, base_graph, nunique (distance-based rows)")
        mc.add("numExpThreeDistanceMetrics", e3_dist["distance_metric"].nunique(), "exp3_graph_layout_results.csv, distance_metric, nunique")
        mc.add("numExpThreeSeeds", e3_ok["seed"].nunique(), "exp3_graph_layout_results.csv, seed, nunique")
        threshold = exp3_large_graph_threshold()
        grouped = add_exp3_groups(e3_dist, threshold)
        mc.add("numExpThreeLargeGraphThresholdNodes", threshold, "config_experiments.yaml, exp3_graph_layout.large_graph_threshold_nodes")
        for group, suffix in _EXP3_GROUP_SUFFIX.items():
            mc.add(
                f"numExpThree{suffix}Graphs", grouped.loc[grouped["size_group"] == group, "base_graph"].nunique(),
                f"exp3_graph_layout_results.csv, base_graph, nunique (size_group={group}, n_nodes_lcc {'<=' if group == 'small' else '>'} {threshold})",
            )
    mc.try_block("counts E3", block3)


def _add_medians(mc: MacroCollector, data_dir: Path) -> None:
    def block() -> None:
        e1 = _read_csv_required(data_dir / "exp1_dr_benchmark_results.csv")
        for metric, _direction, metric_macro in _MEDIAN_METRIC_SPECS:
            if metric not in e1.columns:
                raise KeyError(f"Column '{metric}' is not in exp1_dr_benchmark_results.csv.")
            by_ds_method = _median_across_seeds(e1, metric)
            for method_name in _MEDIAN_METHODS:
                if method_name not in by_ds_method.columns:
                    continue
                mc.add_if_finite(
                    f"med{metric_macro}{_suffix(method_name)}", by_ds_method[method_name].median(),
                    f"exp1_dr_benchmark_results.csv, {metric} (method={method_name}), median over seeds then median over datasets",
                )
    mc.try_block("E1 medians (methods x metrics)", block)

    def block_cg() -> None:
        cg = _read_csv_required(data_dir / "exp1_cluster_geometry_results.csv")
        for metric, _direction, metric_macro in _CLUSTER_GEOMETRY_SPECS:
            if metric not in cg.columns:
                raise KeyError(f"Column '{metric}' is not present in exp1_cluster_geometry_results.csv.")
            by_ds_method = _median_across_seeds(cg, metric)
            for method_name in _MEDIAN_METHODS:
                if method_name not in by_ds_method.columns:
                    continue
                mc.add_if_finite(
                    f"med{metric_macro}{_suffix(method_name)}", by_ds_method[method_name].median(),
                    f"exp1_cluster_geometry_results.csv, {metric} (method={method_name}), median over seeds then median over datasets",
                )
    mc.try_block("cluster-geometry medians (K10, densMAP/PHATE nuance)", block_cg)


def _add_win_counts(mc: MacroCollector, data_dir: Path, neighbor_methods: list[str]) -> None:
    def block() -> None:
        e1 = _read_csv_required(data_dir / "exp1_dr_benchmark_results.csv")
        for metric, direction, metric_macro in _WIN_METRIC_SPECS:
            if metric not in e1.columns:
                mc.logger.warning("Win counts: column '%s' is not in exp1_dr_benchmark_results.csv, skipping.", metric)
                continue
            wide = _median_across_seeds(e1, metric)
            family_best = _family_best_per_dataset(wide, STRESS_FAMILY_METHODS, direction)
            neighbor_best = _family_best_per_dataset(wide, neighbor_methods, direction)
            both = pd.DataFrame({"family": family_best, "neighbor": neighbor_best}).dropna()
            if direction == "max":
                wins = int((both["family"] > both["neighbor"]).sum())
            else:
                wins = int((both["family"] < both["neighbor"]).sum())
            mc.add(
                f"numWins{metric_macro}", f"{wins}/{both.shape[0]}",
                f"exp1_dr_benchmark_results.csv, {metric}, best of stress-family vs. best of neighbor-family per dataset (median over seeds)",
            )
    mc.try_block("win counts Sammon/MDS family vs. neighbor family", block)


def _add_alpha_zero_vs_one(mc: MacroCollector, data_dir: Path) -> None:
    """Paired per-dataset comparison of alpha=0 against Sammon's alpha=1.

    The marginal medians reverse this comparison (see the caption of the main
    table): a method can have the better marginal median and still lose on most
    datasets pairwise, because datasets differ in stress scale. Claims about
    alpha=1 being a poor default must therefore rest on these paired counts.
    """
    def block() -> None:
        e1 = _read_csv_required(data_dir / "exp1_dr_benchmark_results.csv")
        specs = [("stress_scale_invariant", "min", "Stress"), ("auc_rnx", "max", "Aucrnx")]
        for metric, direction, macro_stub in specs:
            if metric not in e1.columns:
                mc.logger.warning("alpha0-vs-alpha1: column '%s' missing, skipping.", metric)
                continue
            wide = _median_across_seeds(e1, metric)
            if "sammon_alpha0_smacof" not in wide.columns or "sammon_alpha_smacof" not in wide.columns:
                mc.logger.warning("alpha0-vs-alpha1: required method columns missing, skipping.")
                return
            both = wide[["sammon_alpha0_smacof", "sammon_alpha_smacof"]].dropna()
            if direction == "min":
                wins = int((both["sammon_alpha0_smacof"] < both["sammon_alpha_smacof"]).sum())
            else:
                wins = int((both["sammon_alpha0_smacof"] > both["sammon_alpha_smacof"]).sum())
            mc.add(
                f"numAlphaZeroBeatsAlphaOne{macro_stub}", f"{wins}/{both.shape[0]}",
                f"exp1_dr_benchmark_results.csv, {metric}, PAIRED per-dataset count where alpha=0 beats alpha=1 (median over seeds); the marginal median reverses this comparison",
            )
    mc.try_block("paired alpha=0 vs alpha=1 (marginal medians reverse it)", block)


def _add_alpha_pred_distribution(mc: MacroCollector, data_dir: Path, mode: str) -> None:
    """How many datasets the rule assigns to each of its three exponents.

    Guards the claim about how often the rule reduces to plain MDS: the text
    used to say "on most datasets", which the distribution contradicts.
    """
    def block() -> None:
        from src.experiments.exp1_regime_stratified import classify_regime  # noqa: F401
        from src.sammon.alpha_predict import load_alpha_pred_rule, predict_alpha

        props_path = data_dir / "dataset_properties.csv"
        props = _read_csv_required(props_path)
        props = props[(props["kind"] == "vector") & props["nn_ratio_k1"].notna()]
        rule = load_alpha_pred_rule()
        picks = [float(predict_alpha(float(v), rule)) for v in props["nn_ratio_k1"]]
        coefficients = rule.get("coefficients", {})
        for name, key in (("Low", "a_low"), ("Mid", "a_mid"), ("High", "a_high")):
            if key not in coefficients:
                mc.logger.warning("alpha_pred distribution: rule has no '%s', skipping.", key)
                continue
            value = float(coefficients[key])
            count = sum(1 for p in picks if abs(p - value) <= 1e-9)
            mc.add(
                f"numAlphaPredPicks{name}", count,
                f"dataset_properties.csv (kind=vector) + alpha_pred_rule.json, number of datasets where the rule returns {key}={value:g}",
            )
        mc.add(
            "numAlphaPredPicksNonzero", sum(1 for p in picks if p > 0.0),
            "dataset_properties.csv (kind=vector) + alpha_pred_rule.json, number of datasets where the rule returns a NONZERO exponent",
        )
    mc.try_block("alpha_pred distribution over datasets", block)


def _add_alpha_pred_on_graphs(mc: MacroCollector, data_dir: Path) -> None:
    """How the rule behaves on graph distances.

    The text claimed the rule returns alpha=0 on graphs because their rho_NN lies
    above the upper threshold - while citing a rho_NN range that extends well
    below it. These macros state what the rule actually does.
    """
    def block() -> None:
        from src.sammon.alpha_predict import load_alpha_pred_rule, predict_alpha

        props = _read_csv_required(data_dir / "dataset_properties.csv")
        graphs = props[(props["kind"] == "graph_distance") & props["nn_ratio_k1"].notna()]
        rule = load_alpha_pred_rule()
        picks = [float(predict_alpha(float(v), rule)) for v in graphs["nn_ratio_k1"]]
        zero = sum(1 for p in picks if p == 0.0)
        mc.add("numAlphaPredGraphsZero", f"{zero}/{len(picks)}",
               "dataset_properties.csv (kind=graph_distance) + alpha_pred_rule.json, graph units where the rule returns alpha=0")
        mc.add("numAlphaPredGraphsNonzero", len(picks) - zero,
               "dataset_properties.csv (kind=graph_distance) + alpha_pred_rule.json, graph units where the rule returns a NONZERO exponent")
    mc.try_block("alpha_pred on graph distances", block)


def _add_high_regime_no_gain(mc: MacroCollector, data_dir: Path) -> None:
    """In the concentrated regime, does even the grid optimum gain anything?

    The rule returns alpha=0 there. The theorem justifies that only through
    CV(D), not through rho_NN (the implication runs the other way - see the
    proposition in the Supplement), so the decision rests on this measurement.
    """
    def block() -> None:
        e10 = _read_csv_required(data_dir / "exp10_identifiability_check_results.csv")
        ok = e10[e10["status"] == "ok"]
        high = ok[ok["stratum"] == "high"]
        gain = high.groupby("dataset")["G_auc_oracle"].first().dropna()
        if gain.empty:
            mc.logger.warning("high-regime gain: no rows, skipping.")
            return
        zero = int((gain <= 0.0).sum())
        mc.add("numHighRegimeZeroGain", f"{zero}/{len(gain)}",
               "exp10_identifiability_check_results.csv, datasets in the high rho_NN stratum where the grid optimum gains nothing over alpha=0 (G_auc_oracle <= 0)")
        mc.add_if_finite("maxHighRegimeOracleGain", float(gain.max()),
                         "exp10_identifiability_check_results.csv, largest G_auc_oracle over the high rho_NN stratum", sig=2)
    mc.try_block("no gain in the concentrated regime", block)


def _add_grid_extension(mc: MacroCollector, data_dir: Path) -> None:
    """Where the optimum lands once the alpha grid is extended past its old cap.

    Combines the exp6 grid (0..3) with the exp12 extension (3.25..6) on the
    datasets exp12 covers, and reports how often the optimum lay beyond the old
    cap - i.e. how much the "grid optimum" reference understated the attainable
    gain on those datasets.
    """
    def block() -> None:
        e6 = _read_csv_required(data_dir / "exp6_alpha_curves_results.csv")
        e12 = _read_csv_required(data_dir / "exp12_alpha_grid_extension_results.csv")
        e6 = e6[e6["status"] == "ok"]
        e12 = e12[e12["status"] == "ok"]
        old_cap = float(e6["alpha"].max())
        merged = pd.concat([e6[["dataset", "alpha", "auc_rnx"]], e12[["dataset", "alpha", "auc_rnx"]]])
        med = merged.groupby(["dataset", "alpha"])["auc_rnx"].median().reset_index()
        covered = set(e12["dataset"].unique())
        beyond, gains, best_alphas = 0, [], []
        for dataset_name, grp in med.groupby("dataset"):
            if dataset_name not in covered:
                continue
            best = grp.loc[grp["auc_rnx"].idxmax()]
            old = grp[grp["alpha"] <= old_cap]
            best_old = old.loc[old["auc_rnx"].idxmax()]
            best_alphas.append(float(best["alpha"]))
            gains.append(float(best["auc_rnx"] - best_old["auc_rnx"]))
            if float(best["alpha"]) > old_cap:
                beyond += 1
        if not best_alphas:
            mc.logger.warning("grid extension: no overlapping datasets, skipping.")
            return
        mc.add("numExpTwelveDatasets", len(best_alphas),
               "exp12_alpha_grid_extension_results.csv, datasets covered by the grid extension")
        mc.add("numExpTwelveBeyondOldCap", f"{beyond}/{len(best_alphas)}",
               f"exp6+exp12 merged, datasets whose argmax median auc_rnx lies above the original cap alpha={old_cap:g}")
        mc.add_if_finite("maxExpTwelveAlphaStar", max(best_alphas),
                         "exp6+exp12 merged, largest argmax alpha over the extended grid", sig=2)
        mc.add_if_finite("maxExpTwelveGain", max(gains),
                         "exp6+exp12 merged, largest auc_rnx gain of the extended grid over the original one", sig=2)
        mc.add_if_finite("medExpTwelveGain", float(pd.Series(gains).median()),
                         "exp6+exp12 merged, median auc_rnx gain of the extended grid over the original one", sig=2)
        mc.add_if_finite("expTwelveGridMax", float(e12["alpha"].max()),
                         "exp12_alpha_grid_extension_results.csv, largest alpha searched in the extension", sig=2)
    mc.try_block("alpha grid extension (exp12)", block)


def _add_stats_kendall_cd(mc: MacroCollector, data_dir: Path) -> None:
    for stub, macro_stub in _STATS_FILE_STUBS:
        def block(stub=stub, macro_stub=macro_stub) -> None:
            path = data_dir / f"{stub}.csv"
            df = _read_csv_required(path)
            if df.empty:
                raise ValueError(f"{path} is empty.")
            mc.add(f"kendallW{macro_stub}", float(df["kendall_w"].iloc[0]), f"{stub}.csv, kendall_w, first row", sig=3)
            mc.add(f"cd{macro_stub}", float(df["cd"].iloc[0]), f"{stub}.csv, cd, first row", sig=3)
            mc.add(f"numDatasets{macro_stub}", int(df["n_datasets"].iloc[0]), f"{stub}.csv, n_datasets, first row")
            mc.add(f"numMethods{macro_stub}", int(df["n_methods"].iloc[0]), f"{stub}.csv, n_methods, first row")
        mc.try_block(f"Kendall W/CD ({stub})", block)


def _add_exp3_group_numbers(mc: MacroCollector, data_dir: Path, tables_dir: Path) -> None:
    """E3 small/large graphs (K12/3): average ranks from the exp3_main_table.csv
    table (report_tables.method_summary) and median auc_rnx/stress per
    distance (resistance/shortest_path) for both large and small graphs."""
    def block_ranks() -> None:
        tab = _read_csv_required(tables_dir / f"{EXP3_MAIN_TABLE}.csv")
        dist = tab[tab["kind"] == "distance"]
        for group, gsuffix in _EXP3_GROUP_SUFFIX.items():
            sub = dist[dist["size_group"] == group].set_index("method")
            for metric, msuffix in [("auc_rnx", "Aucrnx"), ("stress_scale_invariant", "Stress")]:
                col = f"{metric}_avg_rank"
                for method_name in _EXP3_METHODS:
                    if method_name not in sub.index or col not in sub.columns:
                        continue
                    mc.add_if_finite(
                        f"avgRank{msuffix}{gsuffix}{_suffix(method_name)}", sub.loc[method_name, col],
                        f"{EXP3_MAIN_TABLE}.csv, {col} (method={method_name}; unit = graph x distance, {group} graphs; rank within unit among available methods)",
                    )
    mc.try_block("E3 average ranks small/large graphs", block_ranks)

    def block_medians() -> None:
        e3 = _read_csv_required(data_dir / "exp3_graph_layout_results.csv")
        ok = e3[e3["status"] == "ok"]
        grouped = add_exp3_groups(ok, exp3_large_graph_threshold())
        dist = grouped[grouped["kind"] == "distance"]
        for group, gsuffix in _EXP3_GROUP_SUFFIX.items():
            for dmetric, dsuffix in _EXP3_DISTANCE_SUFFIX.items():
                sub = dist[(dist["size_group"] == group) & (dist["distance_metric"] == dmetric)]
                if sub.empty:
                    continue
                for metric, msuffix in [("auc_rnx", "Aucrnx"), ("stress_scale_invariant", "Stress")]:
                    per = sub.groupby(["base_graph", "method"])[metric].median().unstack("method")
                    for method_name in _EXP3_METHODS:
                        if method_name not in per.columns:
                            continue
                        mc.add_if_finite(
                            f"med{msuffix}{gsuffix}{dsuffix}{_suffix(method_name)}", per[method_name].median(),
                            f"exp3_graph_layout_results.csv, {metric} (method={method_name}, distance_metric={dmetric}, {group} graphs), median over seeds then over graphs",
                        )
        native = grouped[grouped["kind"] == "native"]
        for metric, msuffix in [("auc_rnx", "Aucrnx"), ("stress_scale_invariant", "Stress")]:
            per = native.groupby(["base_graph", "method"])[metric].median().unstack("method")
            for method_name in _EXP3_NATIVE_METHODS:
                if method_name not in per.columns:
                    continue
                mc.add_if_finite(
                    f"med{msuffix}Native{_suffix(method_name)}", per[method_name].median(),
                    f"exp3_graph_layout_results.csv, {metric} (method={method_name}, native layout, all graphs), median over seeds then over graphs",
                )
    mc.try_block("E3 medians by distance and group", block_medians)


def _add_pareto_numbers(mc: MacroCollector, tables_dir: Path, data_dir: Path) -> None:
    def block() -> None:
        summary = _read_csv_required(tables_dir / "pareto_summary.csv").set_index("method")
        total = int(summary["n_datasets_total"].iloc[0])
        mc.add("numExpOneDatasetsPareto", total, "pareto_summary.csv, n_datasets_total, first row")
        for method_name in ["sammon_alpha_auto", "sammon_alpha_pred", "sammon_alpha0_smacof", "tsne_auto"]:
            if method_name not in summary.index:
                continue
            mc.add(f"num{_suffix(method_name)}OnFront", int(summary.loc[method_name, "n_datasets_on_front_auc_stress"]), f"pareto_summary.csv, n_datasets_on_front_auc_stress (method={method_name})")
        if "sammon_alpha_auto" in summary.index:
            mc.add(
                "numAlphaAutoDominatedByNeighbor", int(summary.loc["sammon_alpha_auto", "n_datasets_dominated_by_neighbor_method"]),
                "pareto_summary.csv, n_datasets_dominated_by_neighbor_method (method=sammon_alpha_auto)",
            )
    mc.try_block("Pareto (K3)", block)


def _add_speedups(mc: MacroCollector, data_dir: Path) -> None:
    def block_e1() -> None:
        e1 = _read_csv_required(data_dir / "exp1_dr_benchmark_results.csv")
        ok = e1[e1["status"] == "ok"]
        # median over seeds per dataset, then median over datasets (same convention as medTime*)
        med_time = ok.groupby(["dataset", "method"])["wall_time_sec"].median().unstack("method").median()
        pairs = [
            ("speedupClassicVsAlphaSmacof", "sammon_classic", "sammon_alpha_smacof"),
            ("speedupMdsVsAlphaSmacof", "mds", "sammon_alpha_smacof"),
            ("speedupAlphaAutoVsAlphaPred", "sammon_alpha_auto", "sammon_alpha_pred"),
            ("speedupAlphaPredVsAlphaOne", "sammon_alpha_pred", "sammon_alpha_smacof"),
        ]
        for name, a, b in pairs:
            if a not in med_time.index or b not in med_time.index:
                continue
            mc.add(name, float(med_time[a] / med_time[b]), f"exp1_dr_benchmark_results.csv, wall_time_sec, median({a})/median({b}) (median over seeds then datasets)", sig=3)
        if "sammon_alpha_auto" in med_time.index and "sammon_alpha_pred" in med_time.index:
            mc.add(
                "fracTimeAlphaPredVsAlphaAutoPercent", float(med_time["sammon_alpha_pred"] / med_time["sammon_alpha_auto"] * 100.0),
                "exp1_dr_benchmark_results.csv, wall_time_sec, 100*median(sammon_alpha_pred)/median(sammon_alpha_auto)", sig=3,
            )
    mc.try_block("E1 speedups", block_e1)

    def block_e2() -> None:
        e2 = _read_csv_required(data_dir / "exp2_scaling_results.csv")
        n_large = int(load_experiments_config()["exp2_solver_scaling"]["scaling"]["max_n_dense"])
        ok = e2[(e2["status"] == "ok") & e2["dataset"].str.endswith(f"_n{n_large}")]
        if ok.empty:
            raise ValueError(f"exp2_scaling_results.csv has no 'ok' rows for n={n_large} (dataset *_n{n_large}).")
        mc.add("numExpTwoLargeN", n_large, "config_experiments.yaml, exp2_solver_scaling.scaling.max_n_dense")
        med_time = ok.groupby("method")["wall_time_sec"].median()
        med_stress = ok.groupby("method")["stress_scale_invariant"].median()
        for method_name, suffix in _EXP2_METHOD_SUFFIX.items():
            if method_name not in med_time.index:
                continue
            mc.add_if_finite(f"medTimeLargeN{suffix}", med_time[method_name], f"exp2_scaling_results.csv, wall_time_sec, median (method={method_name}, n={n_large})")
            mc.add_if_finite(f"medStressLargeN{suffix}", med_stress[method_name], f"exp2_scaling_results.csv, stress_scale_invariant, median (method={method_name}, n={n_large})")
        for name, a, b in [
            ("speedupSmacofGpuLargeN", "smacof__cpu", "smacof__cuda"),
            ("speedupSmacofGpuVsSgdStabCpuLargeN", "sgd_stab__cpu", "smacof__cuda"),
            ("speedupSmacofCpuVsSgdStabCpuLargeN", "sgd_stab__cpu", "smacof__cpu"),
            ("speedupSparseSmacofVsSmacofCpuLargeN", "smacof__cpu", "sparse_smacof__cpu"),
        ]:
            if a in med_time.index and b in med_time.index:
                mc.add(name, float(med_time[a] / med_time[b]), f"exp2_scaling_results.csv, wall_time_sec, median({a})/median({b}) at n={n_large}", sig=3)
        for name, a in [("relStressSparseSmacofVsSmacofLargeNPercent", "sparse_smacof__cpu"), ("relStressSparseSgdVsSmacofLargeNPercent", "sparse_sgd__cpu"), ("relStressSgdStabVsSmacofLargeNPercent", "sgd_stab__cpu")]:
            if a in med_stress.index and "smacof__cpu" in med_stress.index:
                mc.add(name, float((med_stress[a] / med_stress["smacof__cpu"] - 1.0) * 100.0), f"exp2_scaling_results.csv, stress_scale_invariant, 100*(median({a})/median(smacof__cpu)-1) at n={n_large}", sig=3)
    mc.try_block("E2 scaling (GPU/SGD/sparse at max n)", block_e2)


def _add_k9_relative_changes(mc: MacroCollector, data_dir: Path) -> None:
    def block() -> None:
        rel = _read_csv_required(data_dir / "exp4_relative.csv")
        mc.add("numExpFourDatasets", int(rel["dataset"].nunique()), "exp4_relative.csv, dataset, nunique")
        for solver, ssuffix in _K9_SOLVERS:
            sub_solver = rel[rel["solver"] == solver]
            for lam, lsuffix in _K9_LAMBDAS:
                sub = sub_solver[np.isclose(sub_solver["lam"], lam)]
                if sub.empty:
                    continue
                mc.add(
                    f"relStab{ssuffix}Lambda{lsuffix}", float(sub["stab_pct_change_vs_lambda0"].median()),
                    f"exp4_relative.csv, stab_pct_change_vs_lambda0, median over datasets (solver={solver}, lambda={lam})", sig=3,
                )
                mc.add(
                    f"relQual{ssuffix}Lambda{lsuffix}", float(sub["qual_pct_change_vs_lambda0"].median()),
                    f"exp4_relative.csv, qual_pct_change_vs_lambda0, median over datasets (solver={solver}, lambda={lam})", sig=3,
                )
    mc.try_block("K9 relative changes (E4 temporal, smacof + sgd)", block)


def _add_exp4_dtsne_baseline_numbers(mc: MacroCollector, data_dir: Path) -> None:
    """Q1 addendum (06_diskuse.tex \\todo): comparison of our temporal family
    (`lambda<L>_alpha<A>_<solver>`) with the dynamic t-SNE baseline
    (`dtsne_lambda<L>`) directly from `exp4_temporal_results.csv`. Both
    `stab` (node displacement between snapshots) and `qual` (stress) follow
    the LOWER = BETTER convention (see the caption of Table exp4_relative in
    05_vysledky.tex). Aggregation: median over seeds at (dataset,method),
    then median over datasets - same convention as
    `exp4_relative.py::build_relative_table` (K9).

    WARNING - conflict of interest (see 06_diskuse.tex): `qual` is the
    scale-invariant stress, i.e. DIRECTLY the objective function of our
    method (SMACOF majorization), whereas dtsne optimizes KL divergence -
    comparing on `qual` is therefore systematically in our favor, not a fair
    fight. `exp4_temporal_results.csv` does NOT contain any
    neighborhood-preservation metric (trustworthiness/knn-jaccard/qnx etc.)
    for individual temporal snapshots, so a macro for this (fair) side of
    the comparison CANNOT be generated from existing data - see the WARNING
    in the log and documentation/2026-09-14_makra_doplneni.md (what would
    need to be computed).
    """
    def block() -> None:
        df = _read_csv_required(data_dir / "exp4_temporal_results.csv")
        ok = df[df["status"] == "ok"]
        if ok.empty:
            raise ValueError("exp4_temporal_results.csv has no 'ok' rows.")
        med_seed = ok.groupby(["dataset", "method"])[["stab", "qual"]].median().reset_index()
        med_method = med_seed.groupby("method")[["stab", "qual"]].median()

        dtsne_rows: dict[float, pd.Series] = {}
        our_rows: dict[tuple[float, str], pd.Series] = {}
        for method, row in med_method.iterrows():
            m_dtsne = _DTSNE_METHOD_RE.match(method)
            if m_dtsne:
                dtsne_rows[float(m_dtsne.group("lam"))] = row
                continue
            m_ours = _OUR_TEMPORAL_METHOD_RE.match(method)
            if m_ours:
                our_rows[(float(m_ours.group("lam")), m_ours.group("solver"))] = row
        if not dtsne_rows:
            raise ValueError("exp4_temporal_results.csv contains no 'dtsne_lambda*' method.")
        if not our_rows:
            raise ValueError("exp4_temporal_results.csv contains no method of our family 'lambda*_alpha*_<solver>'.")

        # (a) raw dtsne medians at each lambda it has available
        for lam, row in sorted(dtsne_rows.items()):
            lsuffix = num_to_words(lam)
            mc.add_if_finite(f"medStabDtsneLambda{lsuffix}", row["stab"], f"exp4_temporal_results.csv, stab, median over seeds then datasets (method=dtsne_lambda{lam:g})", sig=3)
            mc.add_if_finite(f"medQualDtsneLambda{lsuffix}", row["qual"], f"exp4_temporal_results.csv, qual, median over seeds then datasets (method=dtsne_lambda{lam:g})", sig=3)

        # (b) the same for our family (both solvers), only at the lambdas that
        # dtsne also has available (a direct pair for comparison in the text)
        dtsne_lambdas = set(dtsne_rows)
        solver_suffix = dict(_K9_SOLVERS)
        for (lam, solver), row in sorted(our_rows.items()):
            if lam not in dtsne_lambdas or solver not in solver_suffix:
                continue
            lsuffix = num_to_words(lam)
            mc.add_if_finite(f"medStab{solver_suffix[solver]}Lambda{lsuffix}", row["stab"], f"exp4_temporal_results.csv, stab, median over seeds then datasets (method=lambda{lam:g}_alpha1.0_{solver})", sig=3)
            mc.add_if_finite(f"medQual{solver_suffix[solver]}Lambda{lsuffix}", row["qual"], f"exp4_temporal_results.csv, qual, median over seeds then datasets (method=lambda{lam:g}_alpha1.0_{solver})", sig=3)

        # (c) dominance on BOTH axes at comparable stability: for each dtsne
        # point (one lambda) find the CLOSEST point of our family by
        # |stab_ours - stab_dtsne| (over ALL our lambda x solver
        # combinations, not just shared lambdas) and check whether it has a
        # lower (better) qual at a comparable (closest possible) stability.
        our_lam_solver = list(our_rows.keys())
        our_stab = np.array([our_rows[k]["stab"] for k in our_lam_solver])
        our_qual = np.array([our_rows[k]["qual"] for k in our_lam_solver])
        stab_diffs_abs: list[float] = []
        qual_gain_pct: list[float] = []
        n_dominated = 0
        for lam, row in dtsne_rows.items():
            idx = exp4_common.nearest_stab_index(row["stab"], our_stab)
            stab_diffs_abs.append(float(abs(our_stab[idx] - row["stab"])))
            qual_gain = float(row["qual"] - our_qual[idx])  # >0 = our variant has a lower (better) qual
            if qual_gain > 0:
                n_dominated += 1
            if row["qual"] > 0:
                qual_gain_pct.append(100.0 * qual_gain / float(row["qual"]))
        n_total = len(dtsne_rows)
        mc.add(
            "numExpFourDtsneDominatedPairs", f"{n_dominated}/{n_total}",
            "exp4_temporal_results.csv, number of dtsne points (each lambda) where the nearest point of our family by |stab_ours-stab_dtsne| has a lower (better) qual - paired across all our lambda x solver combinations",
        )
        mc.add_if_finite(
            "medExpFourDtsneNearestStabDiffAbs", float(np.median(stab_diffs_abs)),
            "exp4_temporal_results.csv, median |stab_ours-stab_dtsne| over the nearest pairs (a measure of pairing stability comparability, smaller = better pair)", sig=2,
        )
        if qual_gain_pct:
            mc.add_if_finite(
                "medExpFourDtsneNearestQualGainPercent", float(np.median(qual_gain_pct)),
                "exp4_temporal_results.csv, median 100*(qual_dtsne-qual_ours)/qual_dtsne over the nearest pairs by stab (positive = our variant has lower stress at a comparable node displacement)", sig=3,
            )
    mc.try_block("E4 dtsne baseline: raw medians + nearest-stability pairing (06_diskuse.tex)", block)


# K16 (2026-09-16): metric (exp4_neighbor_primary_stats.csv column) -> an ASCII
# suffix for the macro name of the E4 primary neighborhood-preservation test
# vs. dtsne (\newcommand only allows letters - K as a number converted to a
# word via num_to_words, same convention as elsewhere in this file; defined
# ONLY HERE, num_to_words must already be defined).
_EXP4_PRIMARY_METRIC_SUFFIX = {
    "trust_k10": "TrustK" + num_to_words(10), "jacc_k10": "JaccK" + num_to_words(10),
    "trust_k5": "TrustK" + num_to_words(5), "jacc_k5": "JaccK" + num_to_words(5),
}


def _add_exp4_neighbor_primary_numbers(mc: MacroCollector, data_dir: Path) -> None:
    """K16 (2026-09-16, the author's spec - "hierarchical test"): the PRIMARY
    confirmatory test of neighborhood preservation (independence unit =
    DATASET) of our temporal family against dynamic t-SNE - source
    `exp4_neighbor_primary_stats.csv` (`exp4_neighbor_metrics.py::_write_primary_stats`,
    the nearest-'stab' pairing shared with (a) above - `exp4_common.py`).

    Sign convention: 'median_diff'/'hl_estimate' = ours - dtsne, POSITIVE =
    our method BETTER (higher trustworthiness/kNN-Jaccard). WARNING (author,
    projectstate.md 2026-09-15/16): per the author's orientational
    computation, ALL 4 primary metrics come out NEGATIVE (our method WORSE
    than dtsne at neighborhood preservation at comparable stability, unlike
    stress, where our method leads) - the value and sign in the generated
    macros ALWAYS come from the CSV, this comment is only an orientational
    direction check, not the source of the number."""
    def block() -> None:
        df = _read_csv_required(data_dir / "exp4_neighbor_primary_stats.csv").set_index("metric")
        if df.empty:
            raise ValueError("exp4_neighbor_primary_stats.csv is empty.")

        # the median |Deltastab| pairing is computed ONCE over all pairs (see
        # _write_primary_stats) - the same value in every row, we verify agreement.
        stab_diffs = df["median_abs_stab_diff"].unique()
        if len(stab_diffs) != 1:
            raise ValueError(
                "exp4_neighbor_primary_stats.csv: the column 'median_abs_stab_diff' should be "
                "the same for all metrics (computed once over all pairs)."
            )
        mc.add_if_finite(
            "medExpFourPrimaryAbsStabDiff", float(stab_diffs[0]),
            "exp4_neighbor_primary_stats.csv, median_abs_stab_diff (a measure of the comparability of pairing our family with dtsne by stab, K16 primary test)", sig=2,
        )

        for metric, suffix in _EXP4_PRIMARY_METRIC_SUFFIX.items():
            if metric not in df.index:
                continue
            row = df.loc[metric]
            mc.add_if_finite(
                f"medExpFourPrimaryDiff{suffix}", row["median_diff"],
                f"exp4_neighbor_primary_stats.csv, median_diff (metric={metric}, ours-minus-dtsne, positive=ours better), median over dtsne lambdas then exact sign-flip over datasets", sig=3,
            )
            mc.add_if_finite(
                f"hlExpFourPrimaryDiff{suffix}", row["hl_estimate"],
                f"exp4_neighbor_primary_stats.csv, hl_estimate (Hodges-Lehmann estimator, metric={metric}, ours-minus-dtsne)", sig=3,
            )
            # Lower/upper bound of the paired percentile bootstrap CI of the median
            # difference (`stats_holdout.bootstrap_ci_paired`, see
            # `_primary_test_for_metric` in exp4_neighbor_metrics.py) - added
            # 2026-09-16 (previously missing, tex-writer had to leave a \todo in
            # 05_vysledky.tex).
            mc.add_if_finite(
                f"ciLowExpFourPrimaryDiff{suffix}", row["ci_low"],
                f"exp4_neighbor_primary_stats.csv, ci_low (paired percentile bootstrap CI lower bound of the median difference, metric={metric}, ours-minus-dtsne)", sig=3,
            )
            mc.add_if_finite(
                f"ciHighExpFourPrimaryDiff{suffix}", row["ci_high"],
                f"exp4_neighbor_primary_stats.csv, ci_high (paired percentile bootstrap CI upper bound of the median difference, metric={metric}, ours-minus-dtsne)", sig=3,
            )
            mc.add_if_finite(
                f"pExpFourPrimary{suffix}", row["p_value"],
                f"exp4_neighbor_primary_stats.csv, p_value (exact sign-flip test over datasets, metric={metric})", sig=3,
            )
            mc.add_if_finite(
                f"pHolmExpFourPrimary{suffix}", row["p_value_holm"],
                f"exp4_neighbor_primary_stats.csv, p_value_holm (Holm-Bonferroni correction ONLY over primary_metrics, metric={metric})", sig=3,
            )
            mc.add(
                f"numExpFourPrimaryPairs{suffix}", int(row["n_datasets"]),
                f"exp4_neighbor_primary_stats.csv, n_datasets (metric={metric}, number of datasets entering the primary test)",
            )
            mc.add(
                f"numExpFourPrimaryWinsOurs{suffix}", int(row["n_wins_ours"]),
                f"exp4_neighbor_primary_stats.csv, n_wins_ours (metric={metric}, number of datasets where our method exceeds dtsne)",
            )
            mc.add(
                f"numExpFourPrimaryWinsDtsne{suffix}", int(row["n_wins_dtsne"]),
                f"exp4_neighbor_primary_stats.csv, n_wins_dtsne (metric={metric}, number of datasets where dtsne exceeds our method)",
            )
        # The number of permutations of the exact sign-flip test is the same
        # for all primary metrics (same number of datasets), so one macro is
        # enough; it also implies the smallest attainable two-sided p-value
        # (2 / number of permutations), which the text cites when discussing
        # the test's power.
        perms = int(df.loc[df["test_role"] == "primary", "n_permutations_exact"].iloc[0])
        mc.add(
            "numExpFourPrimaryPermutations", perms,
            "exp4_neighbor_primary_stats.csv, n_permutations_exact (2^n_datasets, exact sign-flip test)",
        )
        mc.add_if_finite(
            "minPExpFourPrimary", 2.0 / perms,
            "exp4_neighbor_primary_stats.csv, 2/n_permutations_exact - the smallest attainable two-sided p-value of the exact test",
            sig=3,
        )
    mc.try_block("E4 K16 primary neighborhood-preservation test vs. dtsne (unit=dataset, 06_diskuse.tex)", block)


def _add_exp5_factorial(mc: MacroCollector, tables_dir: Path, data_dir: Path) -> None:
    """E5 main effects from exp5_factorial_summary.csv (report_tables.py) +
    a numeric check of agreement between init pca vs. classical_mds."""
    _FACTOR_SUFFIX = {"init": "Init", "eps_D_q": "Epsq", "alpha": "Alpha"}
    _METRIC_SUFFIX = {"auc_rnx": "Aucrnx", "stress_scale_invariant": "Stress"}

    def block() -> None:
        tab = _read_csv_required(tables_dir / f"{EXP5_FACTORIAL_SUMMARY}.csv")
        for _, row in tab.iterrows():
            fsuffix = _FACTOR_SUFFIX[row["factor"]]
            level = str(row["level"])
            lsuffix = level[:1].upper() + level[1:] if row["factor"] == "init" else num_to_words(level)
            if not lsuffix.isalpha():
                raise ValueError(f"Level '{level}' of factor {row['factor']} cannot be converted to a letter suffix.")
            for metric, msuffix in _METRIC_SUFFIX.items():
                mc.add_if_finite(
                    f"medExpFive{msuffix}{fsuffix}{lsuffix}", row[f"median_{metric}"],
                    f"{EXP5_FACTORIAL_SUMMARY}.csv, median_{metric} (factor={row['factor']}, level={level}); median over seeds and other factors per dataset, then over datasets",
                )
        # relative effects: random vs. pca (stress), max eps vs. eps=0 (stress)
        init = tab[tab["factor"] == "init"].set_index("level")
        if {"pca", "random"} <= set(init.index):
            mc.add(
                "relExpFiveStressRandomVsPcaPercent", float((init.loc["random", "median_stress_scale_invariant"] / init.loc["pca", "median_stress_scale_invariant"] - 1.0) * 100.0),
                f"{EXP5_FACTORIAL_SUMMARY}.csv, 100*(median_stress(init=random)/median_stress(init=pca)-1)", sig=3,
            )
            mc.add(
                "diffExpFiveAucrnxRandomVsPca", float(init.loc["random", "median_auc_rnx"] - init.loc["pca", "median_auc_rnx"]),
                f"{EXP5_FACTORIAL_SUMMARY}.csv, median_auc_rnx(init=random) - median_auc_rnx(init=pca)", sig=2,
            )
        eps = tab[tab["factor"] == "eps_D_q"].copy()
        eps["level_f"] = eps["level"].astype(float)
        eps = eps.set_index("level_f").sort_index()
        if eps.shape[0] >= 2 and eps.index[0] == 0.0:
            hi = eps.index[-1]
            mc.add("expFiveEpsqMax", hi, f"{EXP5_FACTORIAL_SUMMARY}.csv, max level of eps_D_q", sig=3)
            mc.add(
                "relExpFiveStressEpsqMaxVsZeroPercent", float((eps.loc[hi, "median_stress_scale_invariant"] / eps.loc[0.0, "median_stress_scale_invariant"] - 1.0) * 100.0),
                f"{EXP5_FACTORIAL_SUMMARY}.csv, 100*(median_stress(eps_D_q={hi})/median_stress(eps_D_q=0)-1)", sig=3,
            )
        alpha = tab[tab["factor"] == "alpha"].copy()
        alpha["level_f"] = alpha["level"].astype(float)
        best = alpha.loc[alpha["median_auc_rnx"].idxmax()]
        mc.add("expFiveAlphaBestAucrnx", float(best["level_f"]), f"{EXP5_FACTORIAL_SUMMARY}.csv, level alpha with max median_auc_rnx", sig=3)
        mc.add("numExpFiveAlphaLevels", int(alpha.shape[0]), f"{EXP5_FACTORIAL_SUMMARY}.csv, number of alpha levels")
    mc.try_block("E5 main effects (factorial summary)", block)

    def block_equiv() -> None:
        eq = _read_csv_required(tables_dir / f"{EXP5_INIT_EQUIVALENCE}.csv")
        for _, row in eq.iterrows():
            msuffix = _METRIC_SUFFIX[row["metric"]]
            mc.add_if_finite(
                f"maxAbsDiffExpFiveInitPcaVsClassicalMds{msuffix}", row["max_abs_diff"],
                f"{EXP5_INIT_EQUIVALENCE}.csv, max_abs_diff ({row['init_a']} vs {row['init_b']}, {row['metric']}, n_pairs={int(row['n_pairs'])})", sig=2,
            )
        if not eq.empty:
            mc.add("numExpFiveInitEquivalencePairs", int(eq["n_pairs"].iloc[0]), f"{EXP5_INIT_EQUIVALENCE}.csv, n_pairs, first row")
    mc.try_block("E5 check of agreement between init pca vs. classical_mds", block_equiv)

    def block_counts() -> None:
        e5 = _read_csv_required(data_dir / "exp5_ablation_results.csv")
        ok = e5[e5["status"] == "ok"]
        mc.add("numExpFiveDatasets", int(ok["dataset"].nunique()), "exp5_ablation_results.csv, dataset, nunique")
        mc.add("numExpFiveRuns", int(ok.shape[0]), "exp5_ablation_results.csv, number of 'ok' rows")
        mc.add("numExpFiveCombinations", int(ok["method"].nunique()), "exp5_ablation_results.csv, method (alpha x init x eps_D_q), nunique")
    mc.try_block("E5 counts", block_counts)


def _add_exp7_numbers(mc: MacroCollector, tables_dir: Path, data_dir: Path) -> None:
    """E7 (supplement S4): median auc_rnx/stress/time from exp7_rank_weights_summary.csv."""
    def block() -> None:
        tab = _read_csv_required(tables_dir / "exp7_rank_weights_summary.csv").set_index("method")
        for method_name, suffix in _EXP7_METHOD_SUFFIX.items():
            if method_name not in tab.index:
                continue
            for col, msuffix in [("auc_rnx_median", "Aucrnx"), ("stress_scale_invariant_median", "Stress"), ("wall_time_sec_median", "Time")]:
                if col in tab.columns:
                    mc.add_if_finite(
                        f"medExpSeven{msuffix}{suffix}", tab.loc[method_name, col],
                        f"exp7_rank_weights_summary.csv, {col} (method={method_name}); median over seeds then datasets",
                    )
        e7 = _read_csv_required(data_dir / "exp7_rank_weights_results.csv")
        ok = e7[e7["status"] == "ok"]
        mc.add("numExpSevenDatasets", int(ok["dataset"].nunique()), "exp7_rank_weights_results.csv, dataset, nunique")
        mc.add("numExpSevenRuns", int(ok.shape[0]), "exp7_rank_weights_results.csv, number of 'ok' rows")
    mc.try_block("E7 rank-weighted stress (supplement)", block)


def _add_exp6_numbers(mc: MacroCollector, tables_dir: Path) -> None:
    """K7 (documentation/2026-09-12_plan_smeru_clanku.md): summary numbers from
    K6 (exp6 alpha curves, `results/tables/exp6_alpha_optimum.csv`) - the
    fraction of datasets with alpha*=0 (concentrated data, alpha
    unidentifiable per R1), the fraction with alpha*>=2 (manifolds), the
    median gain of alpha* vs. alpha=0."""
    def block() -> None:
        df = _read_csv_required(tables_dir / "exp6_alpha_optimum.csv")
        n = int(df.shape[0])
        n_zero = int((df["alpha_star_max"] == 0.0).sum())
        n_high = int((df["alpha_star_max"] >= 2.0).sum())
        n_one = int((df["alpha_star_max"] == 1.0).sum())
        mc.add("numExpSixDatasets", n, "exp6_alpha_optimum.csv, row count")
        mc.add("fracExpSixAlphaZero", f"{n_zero}/{n}", "exp6_alpha_optimum.csv, fraction of rows with alpha_star_max == 0")
        mc.add("fracExpSixAlphaHigh", f"{n_high}/{n}", "exp6_alpha_optimum.csv, fraction of rows with alpha_star_max >= 2")
        mc.add("numExpSixAlphaOneOptimal", f"{n_one}/{n}", "exp6_alpha_optimum.csv, fraction of rows with alpha_star_max == 1")
        mc.add(
            "medExpSixGainVsAlphaZero", float(df["gain_vs_alpha0"].median()),
            "exp6_alpha_optimum.csv, gain_vs_alpha0, median over datasets", sig=3,
        )
        mc.add(
            "medExpSixGainVsAlphaOne", float(df["gain_vs_alpha1"].median()),
            "exp6_alpha_optimum.csv, gain_vs_alpha1, median over datasets (NaN rows dropped by the pandas median)", sig=3,
        )
    mc.try_block("exp6 alpha curves summary (K6/K7)", block)


def _add_alpha_pred_numbers(mc: MacroCollector, data_dir: Path, tables_dir: Path) -> None:
    """K7: LOO numbers of the predicted alpha_pred rule (`results/data/[<mode>/]
    alpha_pred_rule.json`, fit_alpha_rule.py) and the fraction of the
    alpha_auto gain captured (gain_frac); for alpha_pred/alpha_auto timings
    see medTime* and speedupAlphaAutoVsAlphaPred (the E1 speedups group)."""
    _VARIANT_DISPLAY = {"log_linear": "log-linear", "two_threshold": "two-threshold"}

    def block() -> None:
        rule_path = data_dir / "alpha_pred_rule.json"
        if not rule_path.exists():
            raise FileNotFoundError(f"Missing {rule_path}")
        with open(rule_path, "r", encoding="utf-8") as f:
            rule = json.load(f)
        loo = rule["loo_metrics"]
        variant = str(rule["variant"])

        mc.add("AlphaPredVariant", _VARIANT_DISPLAY.get(variant, variant), "alpha_pred_rule.json, variant")
        mc.add("numAlphaPredDatasets", int(loo["n_datasets"]), "alpha_pred_rule.json, loo_metrics.n_datasets")
        for key, name in [
            ("median_auc_at_alpha_pred_chosen", "medAlphaPredAucLoo"),
            ("median_auc_at_alpha_star", "medAlphaPredAucAlphaStar"),
            ("median_auc_at_alpha0", "medAlphaPredAucAlphaZero"),
            ("median_auc_at_alpha1", "medAlphaPredAucAlphaOne"),
            ("median_auc_at_alpha_pred_log_linear", "medAlphaPredAucLogLinear"),
            ("median_auc_at_alpha_pred_two_threshold", "medAlphaPredAucTwoThreshold"),
        ]:
            if key in loo:
                mc.add(name, float(loo[key]), f"alpha_pred_rule.json, loo_metrics.{key}", sig=3)
        gain_frac_median = loo.get("gain_frac_median")
        n_eligible = int(loo.get("n_datasets_gain_frac_eligible", 0))
        if gain_frac_median is not None and np.isfinite(gain_frac_median):
            mc.add(
                "medAlphaPredGainFrac", float(gain_frac_median),
                "alpha_pred_rule.json, loo_metrics.gain_frac_median (fraction of the alpha_auto gain over alpha=0 captured, LOO)", sig=3,
            )
            mc.add(
                "medAlphaPredGainFracPercent", float(gain_frac_median) * 100.0,
                "alpha_pred_rule.json, loo_metrics.gain_frac_median * 100", sig=3,
            )
            if "gain_frac_iqr" in loo and np.isfinite(loo["gain_frac_iqr"]):
                mc.add("iqrAlphaPredGainFrac", float(loo["gain_frac_iqr"]), "alpha_pred_rule.json, loo_metrics.gain_frac_iqr", sig=2)
        if "gain_frac_tau" in loo:
            mc.add("alphaPredGainFracTau", float(loo["gain_frac_tau"]), "alpha_pred_rule.json, loo_metrics.gain_frac_tau (gain threshold for a dataset to be counted)", sig=2)
        mc.add("numAlphaPredGainFracEligible", f"{n_eligible}/{int(loo['n_datasets'])}", "alpha_pred_rule.json, loo_metrics.n_datasets_gain_frac_eligible / n_datasets")
    mc.try_block("alpha_pred LOO summary (K7)", block)

    def block_loo_table() -> None:
        loo = _read_csv_required(tables_dir / "alpha_pred_loo.csv")
        for col, name in [("auc_at_alpha_pred", "AlphaPred"), ("auc_at_alpha_auto", "AlphaAuto"), ("auc_at_alpha_star", "AlphaStar"), ("auc_at_alpha0", "AlphaZero"), ("auc_at_alpha1", "AlphaOne")]:
            if col in loo.columns:
                mc.add_if_finite(f"medLooAuc{name}", loo[col].median(), f"alpha_pred_loo.csv, {col}, median over datasets")
        if {"auc_at_alpha_pred", "auc_at_alpha1"} <= set(loo.columns):
            worse = int((loo["auc_at_alpha_pred"] < loo["auc_at_alpha1"]).sum())
            mc.add("numLooAlphaPredWorseThanAlphaOne", f"{worse}/{loo.shape[0]}", "alpha_pred_loo.csv, number of datasets with auc_at_alpha_pred < auc_at_alpha1")
    mc.try_block("alpha_pred LOO table (medians per variant)", block_loo_table)


def _add_alpha_pred_rule_coefficients(mc: MacroCollector, data_dir: Path) -> None:
    """W5 (03_metoda.tex, eq:alpha_pred): the two thresholds and three levels
    of the 'two_threshold' rule directly from `alpha_pred_rule.json`
    (coefficients.t1/t2/a_low/a_mid/a_high) - see
    `src.sammon.alpha_predict.predict_alpha_two_threshold` (t1, t2 are in
    units of log(nn_ratio_k1))."""
    def block() -> None:
        rule_path = data_dir / "alpha_pred_rule.json"
        if not rule_path.exists():
            raise FileNotFoundError(f"Missing {rule_path}")
        with open(rule_path, "r", encoding="utf-8") as f:
            rule = json.load(f)
        if rule.get("variant") != "two_threshold":
            raise ValueError(
                f"{rule_path}: macros alphaPredThreshold*/AlphaLow/Mid/High require "
                f"variant=='two_threshold', got '{rule.get('variant')}'."
            )
        coef = rule["coefficients"]
        mc.add("alphaPredThresholdOne", float(coef["t1"]), "alpha_pred_rule.json, coefficients.t1 (threshold in log(nn_ratio_k1))", sig=3)
        mc.add("alphaPredThresholdTwo", float(coef["t2"]), "alpha_pred_rule.json, coefficients.t2 (threshold in log(nn_ratio_k1))", sig=3)
        mc.add("alphaPredAlphaLow", float(coef["a_low"]), "alpha_pred_rule.json, coefficients.a_low", sig=3)
        mc.add("alphaPredAlphaMid", float(coef["a_mid"]), "alpha_pred_rule.json, coefficients.a_mid", sig=3)
        mc.add("alphaPredAlphaHigh", float(coef["a_high"]), "alpha_pred_rule.json, coefficients.a_high", sig=3)
    mc.try_block("alpha_pred rule: thresholds and levels (W5, eq:alpha_pred)", block)


def _add_regime_map_spearman(mc: MacroCollector, figures_dir: Path) -> None:
    """W5 (05_vysledky.tex, regime_map figure caption): Spearman correlations
    between rho_NN and (a) the t-SNE lead over alpha_auto in AUC_RNX, (b) the
    best alpha over the article's own full alpha_auto grid
    (`fig_regime_map._best_alpha_full_grid`, E6); only subset='all', as
    cited by the text. Panel key 'best_alpha_full_grid' replaced the
    earlier 'best_alpha012' (reviewer fix 2026-09-18, fig_regime_map.py)."""
    _PANEL_SUFFIX = {"tsne_minus_alpha_auto": "TsneGain", "best_alpha_full_grid": "BestAlpha"}

    def block() -> None:
        df = _read_csv_required(figures_dir / "fig_regime_map_spearman.csv")
        sub = df[df["subset"] == "all"].set_index("panel")
        for panel, suffix in _PANEL_SUFFIX.items():
            if panel not in sub.index:
                continue
            mc.add_if_finite(
                f"spearmanRegimeMap{suffix}Rho", sub.loc[panel, "rho"],
                f"fig_regime_map_spearman.csv, rho (subset=all, panel={panel})", sig=3,
            )
            mc.add_if_finite(
                f"spearmanRegimeMap{suffix}P", sub.loc[panel, "pvalue"],
                f"fig_regime_map_spearman.csv, pvalue (subset=all, panel={panel})", sig=2,
            )
    mc.try_block("Spearman regime map (W5, fig_regime_map)", block)


def _best_alpha_full_grid_from_exp6(e6: pd.DataFrame) -> pd.Series:
    """Best alpha per dataset over the article's own alpha_auto search grid,
    by median-over-seeds auc_rnx - the SAME target as
    `src.figures.fig_regime_map._best_alpha_full_grid` (kept as a private
    duplicate here rather than importing across the experiments/figures
    boundary; any change to the definition must be mirrored in both
    places, checked by `tests/test_stats.py`)."""
    e6_ok = e6[e6["status"] == "ok"]
    med = e6_ok.groupby(["dataset", "alpha"])["auc_rnx"].median()
    wide = med.unstack("alpha")
    return wide.idxmax(axis=1)


def _add_regime_map_property_spearman(mc: MacroCollector, data_dir: Path, tables_dir: Path) -> None:
    """Fact-check fix (author, 2026-09-18): 04_experimenty.tex claimed
    rho_NN (nn_ratio_k1) correlates with the grid-optimum alpha more
    strongly than every other dataset property. Recomputes the Spearman
    correlation of EVERY property in `fig_regime_map.property_correlation_columns`
    (config_experiments.yaml) against the exact same target as the existing
    `spearmanRegimeMapBestAlphaRho` macro (best alpha per dataset over E6's
    own alpha_auto grid, by median-over-seeds auc_rnx, dataset_properties.csv
    kind=='vector', both synthetic and real - subset='all') so the numbers
    are directly comparable. rho_NN itself is NOT re-exported here (already
    `spearmanRegimeMapBestAlphaRho`/`...P`, panel=best_alpha_full_grid,
    subset=all) - only the ambient dimension `d` and the single strongest
    competing property (by |rho|, excluding nn_ratio_k1) are exported, plus
    the full per-property table.

    Writes results/tables/[<mode>/]regime_map_property_spearman.csv
    (columns: property, rho, pvalue, n) - one row per configured property,
    in config order - the source CSV for every macro this function adds."""
    from scipy.stats import spearmanr

    def block() -> None:
        exp_cfg = load_experiments_config()
        try:
            columns: list[str] = list(exp_cfg["fig_regime_map"]["property_correlation_columns"])
        except KeyError as exc:
            raise KeyError("Missing 'fig_regime_map.property_correlation_columns' in config_experiments.yaml.") from exc

        props = _read_csv_required(data_dir / "dataset_properties.csv")
        props = props[props["kind"] == "vector"].copy()
        if props.empty:
            raise ValueError(f"{data_dir / 'dataset_properties.csv'} contains no row with kind=='vector'.")

        e6 = _read_csv_required(data_dir / "exp6_alpha_curves_results.csv")
        best_alpha = _best_alpha_full_grid_from_exp6(e6)

        merged = props.set_index("dataset").join(best_alpha.rename("best_alpha_full_grid"), how="inner")
        if merged.empty:
            raise ValueError("The dataset intersection between dataset_properties.csv and exp6_alpha_curves_results.csv is empty.")

        missing_cols = [c for c in columns if c not in merged.columns]
        if missing_cols:
            raise KeyError(
                f"dataset_properties.csv is missing columns {missing_cols} listed in "
                "'fig_regime_map.property_correlation_columns' (config_experiments.yaml)."
            )

        rows = []
        for col in columns:
            valid = merged[col].notna() & merged["best_alpha_full_grid"].notna()
            n = int(valid.sum())
            if n < 3:
                raise ValueError(f"dataset_properties.csv column '{col}' has fewer than 3 valid (non-NaN) rows (n={n}) - cannot compute Spearman rho.")
            rho, pvalue = spearmanr(merged.loc[valid, col], merged.loc[valid, "best_alpha_full_grid"])
            rows.append({"property": col, "rho": float(rho), "pvalue": float(pvalue), "n": n})
        table = pd.DataFrame(rows)

        out_path = tables_dir / "regime_map_property_spearman.csv"
        tables_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_path, index=False)
        csv_name = out_path.name

        def _macro_name(col: str) -> str:
            # LaTeX \newcommand VALUES may contain underscores (they are
            # plain text, not a macro name) but raw '_' breaks LaTeX text
            # mode - escaped here since the property name itself is quoted
            # verbatim in the macro value.
            return col.replace("_", r"\_")

        row_d = table[table["property"] == "d"]
        if row_d.empty:
            raise ValueError("'d' (ambient dimension) is missing from fig_regime_map.property_correlation_columns.")
        mc.add_if_finite(
            "spearmanRegimeMapBestAlphaDimRho", row_d.iloc[0]["rho"],
            f"{csv_name}, rho (property=d)", sig=3,
        )
        mc.add_if_finite(
            "spearmanRegimeMapBestAlphaDimP", row_d.iloc[0]["pvalue"],
            f"{csv_name}, pvalue (property=d)", sig=2,
        )

        competitors = table[table["property"] != "nn_ratio_k1"].copy()
        if competitors.empty:
            raise ValueError("No competing property left after excluding nn_ratio_k1 - check 'fig_regime_map.property_correlation_columns'.")
        top = competitors.loc[competitors["rho"].abs().idxmax()]
        mc.add("spearmanRegimeMapBestAlphaTopName", _macro_name(str(top["property"])), f"{csv_name}, property with the largest |rho| excluding nn_ratio_k1")
        mc.add_if_finite(
            "spearmanRegimeMapBestAlphaTopRho", top["rho"],
            f"{csv_name}, rho (property={top['property']}, strongest competitor of nn_ratio_k1 by |rho|)", sig=3,
        )
        mc.add_if_finite(
            "spearmanRegimeMapBestAlphaTopP", top["pvalue"],
            f"{csv_name}, pvalue (property={top['property']}, strongest competitor of nn_ratio_k1 by |rho|)", sig=2,
        )
    mc.try_block("Spearman correlations of all dataset properties vs. grid-optimum alpha (fact-check 2026-09-18)", block)


def _add_neighborhood_problem_numbers(mc: MacroCollector, figures_dir: Path) -> None:
    """Flagship intro figure (01_uvod.tex, fig:neighborhood_problem): the
    "intrusion, not displacement" numbers quoted in the caption, computed by
    `fig_neighborhood_problem.py` (`_true_neighbor_intrusion_stats`) over
    ALL n points and their k true original-space neighbors under the MDS
    (alpha=0) panel - median rank of a true neighbor in the embedding's own
    order, and its median embedding distance in units of the panel's own
    local point spacing. NEVER hand-typed in the caption (author reviewer fix
    2026-09-18)."""
    def block() -> None:
        df = _read_csv_required(figures_dir / "fig_neighborhood_problem.csv")
        row = df[(df["record_type"] == "summary") & (df["panel"] == "mds")]
        if len(row) != 1:
            raise ValueError(f"fig_neighborhood_problem.csv: expected exactly 1 summary row for panel='mds', got {len(row)}.")
        r = row.iloc[0]
        mc.add(
            "neighborhoodProblemMdsTrueNeighborMedianRank", int(round(float(r["true_neighbor_median_rank"]))),
            "fig_neighborhood_problem.csv, panel=mds, true_neighbor_median_rank",
        )
        mc.add_if_finite(
            "neighborhoodProblemMdsTrueNeighborMedianDistanceRatio", r["true_neighbor_median_distance_ratio"],
            "fig_neighborhood_problem.csv, panel=mds, true_neighbor_median_distance_ratio (multiple of the panel's own local point spacing)", sig=2,
        )
    mc.try_block("fig_neighborhood_problem intrusion numbers (01_uvod.tex caption)", block)


def _add_regime_stratified(mc: MacroCollector, tables_dir: Path) -> None:
    """W5 (05_vysledky.tex, stratification by concentration regime): the
    number of datasets in the 3 bands and the paired differences of
    AUC_RNX/stress of alpha_pred vs. alpha=0/alpha=1 in the low-rho_NN band
    ('low_ratio', highest distance concentration) + a p-value in the
    transitional 'mid_ratio' band (`exp1_regime_stratified.py`,
    `results/tables/exp1_regime_stratified.csv`)."""
    # Neutral rho_NN band names (author's decision 2026-09-14 - the original
    # "concentrated"/"moderate"/"diffuse" were swapped relative to the
    # distance-concentration theory, see documentation/2026-09-14_prejmenovani_rezimu.md).
    _REGIME_SUFFIX = {"low_ratio": "LowRatio", "mid_ratio": "MidRatio", "high_ratio": "HighRatio"}

    def block() -> None:
        df = _read_csv_required(tables_dir / "exp1_regime_stratified.csv")
        n_by_regime = df.groupby("regime")["n_datasets_regime"].first()
        for regime, suffix in _REGIME_SUFFIX.items():
            if regime not in n_by_regime.index:
                continue
            mc.add(f"Regime{suffix}N", int(n_by_regime[regime]), f"exp1_regime_stratified.csv, n_datasets_regime (regime={regime})")

        paired = df[df["row_type"] == "paired_diff"]

        def _row(regime: str, method: str, metric: str):
            sub = paired[(paired["regime"] == regime) & (paired["method"] == method) & (paired["metric"] == metric)]
            return sub.iloc[0] if not sub.empty else None

        specs = [
            ("low_ratio", "sammon_alpha_pred - sammon_alpha0_smacof", "auc_rnx", "RegimeLowRatioAucGainVsAlphaZero"),
            ("low_ratio", "sammon_alpha_pred - sammon_alpha_smacof", "auc_rnx", "RegimeLowRatioAucGainVsAlphaOne"),
            ("low_ratio", "sammon_alpha_pred - sammon_alpha0_smacof", "stress_scale_invariant", "RegimeLowRatioStressCostVsAlphaZero"),
        ]
        for regime, method, metric, macro_stub in specs:
            row = _row(regime, method, metric)
            if row is None:
                continue
            mc.add_if_finite(
                f"{macro_stub}Median", row["median"],
                f"exp1_regime_stratified.csv, median (regime={regime}, row_type=paired_diff, method={method}, metric={metric})", sig=3,
            )
            mc.add_if_finite(
                f"{macro_stub}Pvalue", row["wilcoxon_pvalue"],
                f"exp1_regime_stratified.csv, wilcoxon_pvalue (regime={regime}, row_type=paired_diff, method={method}, metric={metric})", sig=2,
            )

        row_mod = _row("mid_ratio", "sammon_alpha_pred - sammon_alpha0_smacof", "auc_rnx")
        if row_mod is not None:
            mc.add_if_finite(
                "RegimeMidRatioAucGainVsAlphaZeroPvalue", row_mod["wilcoxon_pvalue"],
                "exp1_regime_stratified.csv, wilcoxon_pvalue (regime=mid_ratio, row_type=paired_diff, method=sammon_alpha_pred - sammon_alpha0_smacof, metric=auc_rnx)", sig=2,
            )
    mc.try_block("E1 stratification by concentration regime (W5)", block)


def _add_graph_nn_ratio_range(mc: MacroCollector, data_dir: Path) -> None:
    """W5 (05_vysledky.tex/06_diskuse.tex): the range of rho_NN (nn_ratio_k1)
    of the E3 graph datasets (kind='graph_distance' in dataset_properties.csv)
    - explains why alpha_pred always chooses alpha=0 on graphs (rho_NN above
    the upper threshold t2)."""
    def block() -> None:
        df = _read_csv_required(data_dir / "dataset_properties.csv")
        graphs = df[df["kind"] == "graph_distance"]
        if graphs.empty:
            raise ValueError("dataset_properties.csv contains no rows with kind=='graph_distance'.")
        lo = fmt_sig(float(graphs["nn_ratio_k1"].min()), sig=2)
        hi = fmt_sig(float(graphs["nn_ratio_k1"].max()), sig=2)
        mc.add("rangeNnRatioGraphsExpThree", f"{lo}--{hi}", "dataset_properties.csv, nn_ratio_k1 (kind=graph_distance), min--max over all graphs and distances")
    mc.try_block("E3 graphs nn_ratio_k1 range (W5)", block)


def _add_umap_grid_edge(mc: MacroCollector, data_dir: Path) -> None:
    """W5 (06_diskuse.tex, baseline tuning): the fraction of E1 runs
    (dataset x seed) where umap_auto selected the edge value of the
    n_neighbors grid (min from config.yaml methods.umap_auto.n_neighbors_grid)."""
    def block() -> None:
        e1 = _read_csv_required(data_dir / "exp1_dr_benchmark_results.csv")
        sub = e1[(e1["method"] == "umap_auto") & (e1["status"] == "ok")]
        if sub.empty:
            raise ValueError("exp1_dr_benchmark_results.csv has no 'ok' rows for method=umap_auto.")
        grid = load_config()["methods"]["umap_auto"]["n_neighbors_grid"]
        edge = min(grid)
        n_edge = int((sub["selected_hyperparam"] == f"n_neighbors={edge}").sum())
        mc.add(
            "numUmapAutoAtGridEdge", f"{n_edge}/{sub.shape[0]}",
            f"exp1_dr_benchmark_results.csv, selected_hyperparam=='n_neighbors={edge}' (method=umap_auto); "
            "edge = min(config.yaml methods.umap_auto.n_neighbors_grid)",
        )
    mc.try_block("UMAP auto at the n_neighbors grid edge (W5)", block)


def _add_multiscale_vs_alpha_auto(mc: MacroCollector, data_dir: Path) -> None:
    """W5 (supplement S3.3): the median difference of sammon_multiscale vs.
    sammon_alpha_auto in AUC_RNX and stress (per-dataset median over seeds,
    then median over datasets) - documents that multiscale weighting does
    not improve over alpha_auto (a negative result)."""
    def block() -> None:
        e1 = _read_csv_required(data_dir / "exp1_dr_benchmark_results.csv")
        for metric, msuffix in [("auc_rnx", "Aucrnx"), ("stress_scale_invariant", "Stress")]:
            if metric not in e1.columns:
                raise KeyError(f"Column '{metric}' is not present in exp1_dr_benchmark_results.csv.")
            wide = _median_across_seeds(e1, metric)
            if "sammon_multiscale" not in wide.columns or "sammon_alpha_auto" not in wide.columns:
                continue
            diff = (wide["sammon_multiscale"] - wide["sammon_alpha_auto"]).dropna()
            if diff.empty:
                continue
            mc.add_if_finite(
                f"medDiff{msuffix}MultiscaleVsAlphaAuto", diff.median(),
                f"exp1_dr_benchmark_results.csv, {metric}, median over datasets of (median_seeds(sammon_multiscale) - median_seeds(sammon_alpha_auto))", sig=2,
            )
    mc.try_block("sammon_multiscale vs. alpha_auto median difference (W5, supplement S3.3)", block)


def _add_exp1_cluster_geometry_dataset_count(mc: MacroCollector, data_dir: Path) -> None:
    """The number of E1 datasets with >=3 classes used for cluster geometry
    (K10) - the threshold MIN_CLASSES_FOR_GEOMETRY=3
    (src/sammon/cluster_geometry.py) is already applied when generating the
    CSV: rows with note=='' have 3<=n_classes<=50 (otherwise
    note='insufficient_classes_or_no_labels'/'excessive_classes_likely_continuous_label',
    see src/experiments/exp1_cluster_geometry.py)."""
    def block() -> None:
        df = _read_csv_required(data_dir / "exp1_cluster_geometry_results.csv")
        ok = df[(df["status"] == "ok") & (df["note"].fillna("") == "")]
        if ok.empty:
            raise ValueError("exp1_cluster_geometry_results.csv has no 'ok' rows with valid geometry (note=='', i.e. >=3 classes).")
        mc.add(
            "numExpOneClusterGeometryDatasets", int(ok["dataset"].nunique()),
            "exp1_cluster_geometry_results.csv, dataset, nunique (status=='ok' & note=='' <=> 3<=n_classes<=50, MIN_CLASSES_FOR_GEOMETRY v src/sammon/cluster_geometry.py)",
        )
    mc.try_block("E1 number of datasets with >=3 classes (K10 cluster geometry)", block)


def _add_exp2_solver_scaling_config_numbers(mc: MacroCollector) -> None:
    """Configuration numbers of the E2 plan (K2/K9) directly from
    config_experiments.yaml - no CSV computation, just citing experiment
    parameters in the text (number of seeds in the scaling run, n sizes for
    the sparse approximation and solver convergence curves)."""
    def block() -> None:
        cfg = load_experiments_config()["exp2_solver_scaling"]
        mc.add(
            "numExpTwoScalingSeeds", int(len(cfg["scaling"]["seeds"])),
            "config_experiments.yaml, exp2_solver_scaling.scaling.seeds, count",
        )
        mc.add(
            "numExpTwoSparseApproxSwissRollN", int(cfg["sparse_approx"]["swiss_roll_n"]),
            "config_experiments.yaml, exp2_solver_scaling.sparse_approx.swiss_roll_n",
        )
        mc.add(
            "numExpTwoSparseApproxDigitsN", int(cfg["sparse_approx"]["digits_n"]),
            "config_experiments.yaml, exp2_solver_scaling.sparse_approx.digits_n",
        )
        mc.add(
            "numExpTwoConvergenceDigitsN", int(cfg["convergence"]["digits_n"]),
            "config_experiments.yaml, exp2_solver_scaling.convergence.digits_n",
        )
        mc.add(
            "numExpTwoConvergenceSwissRollN", int(cfg["convergence"]["swiss_roll_n"]),
            "config_experiments.yaml, exp2_solver_scaling.convergence.swiss_roll_n",
        )
    mc.try_block("E2 configuration numbers (scaling.seeds, sparse_approx/convergence n) directly from config_experiments.yaml", block)


def _add_q1_holdout_numbers(mc: MacroCollector, data_dir: Path, tables_dir: Path) -> None:
    """Q1 step 2 (reserse/2026-09-14_specifikace_rozsireni_q1.md, A.9 item 11) -
    macros for the hold-out candidates of regime L: screening
    (regime_candidates_screen.csv), confirmatory/exploratory analysis
    (exp1_holdout_confirmatory.csv, families F_A1/F_A2/F_A6), and power
    analysis (power_analysis_regime.csv)."""

    def block_screen() -> None:
        df = _read_csv_required(data_dir / "regime_candidates_screen.csv")
        mc.add("numHoldoutCandidates", int(df.shape[0]), "regime_candidates_screen.csv, row count (candidates)")
        n_eligible_l = int(((df["eligible_all"] == True) & (df["regime_frozen"] == "low_ratio")).sum())  # noqa: E712
        mc.add("numHoldoutEligibleRegimeL", n_eligible_l, "regime_candidates_screen.csv, eligible_all & regime_frozen=='low_ratio'")

        # Real vs. synthetic candidates (K/Q1 step 2 addendum 2026-09-14, see
        # the \todo in 05_vysledky.tex "real:synthetic ratio"). The
        # distinction is NOT in a separate 'source' column (that is
        # empty/NaN in the CSV for all rows) - instead the already existing
        # 'requires_download' flag is used (a real dataset is downloaded
        # from OpenML/UCI, see config_experiments.yaml
        # screen_regime_candidates.candidates and
        # src/experiments/screen_regime_candidates.py); synthetic datasets
        # (manifolds, Gaussian clusters) are generated locally without downloading.
        if "requires_download" not in df.columns:
            raise KeyError(
                "regime_candidates_screen.csv is missing the column 'requires_download' needed to "
                "distinguish real/synthetic candidates - add the field to the anchor "
                "config_experiments.yaml::screen_regime_candidates.candidates (field "
                "'requires_download' for each candidate) and regenerate the screening."
            )
        is_real = df["requires_download"] == True  # noqa: E712
        is_synthetic = df["requires_download"] == False  # noqa: E712
        if int((~(is_real | is_synthetic)).sum()) > 0:
            raise ValueError("regime_candidates_screen.csv, requires_download: found an ambiguous/missing value (neither True nor False).")
        mc.add("numHoldoutCandidatesReal", int(is_real.sum()), "regime_candidates_screen.csv, requires_download==True, count (all candidates)")
        mc.add("numHoldoutCandidatesSynthetic", int(is_synthetic.sum()), "regime_candidates_screen.csv, requires_download==False, count (all candidates)")
        is_regime_l = df["regime_frozen"] == "low_ratio"
        mc.add(
            "numHoldoutCandidatesRealRegimeL", int((is_real & is_regime_l).sum()),
            "regime_candidates_screen.csv, requires_download==True & regime_frozen=='low_ratio', count (subset relevant for the power of test F_A2)",
        )
        mc.add(
            "numHoldoutCandidatesSyntheticRegimeL", int((is_synthetic & is_regime_l).sum()),
            "regime_candidates_screen.csv, requires_download==False & regime_frozen=='low_ratio', count (subset relevant for the power of test F_A2)",
        )
    mc.try_block("Q1 step 2: screening of regime-L hold-out candidates", block_screen)

    def block_negative_controls() -> None:
        # Addendum 2026-09-14 (pre-registration change): the count and list
        # of pre-registered control (negative) datasets of regime L
        # (config_experiments.yaml, q1_negative_control_datasets) - originally
        # only `phoneme` (empirically turned out to be in regime L, so it
        # stopped being a control), replaced by `wall_robot`+`dry_bean`
        # (empirically mid_ratio).
        control_datasets = sorted(load_experiments_config()["q1_negative_control_datasets"])
        if not control_datasets:
            raise ValueError("config_experiments.yaml: q1_negative_control_datasets is an empty list.")
        mc.add("numHoldoutControlDatasets", len(control_datasets), "config_experiments.yaml, q1_negative_control_datasets, count")
        escaped = ", ".join(d.replace("_", "\\_") for d in control_datasets)
        mc.add("holdoutControlDatasetsList", escaped, "config_experiments.yaml, q1_negative_control_datasets, list (LaTeX-escaped)")
    mc.try_block("Q1 step 2 addendum 2026-09-14: regime-L control datasets", block_negative_controls)

    def block_confirmatory() -> None:
        df = _read_csv_required(tables_dir / "exp1_holdout_confirmatory.csv")

        def _row(family: str, hyp_id: str) -> pd.Series:
            sub = df[(df["family"] == family) & (df["hypothesis_id"] == hyp_id)]
            if sub.empty:
                raise KeyError(f"exp1_holdout_confirmatory.csv is missing the row family='{family}' hypothesis_id='{hyp_id}'.")
            return sub.iloc[0]

        r_pooled_a0 = _row("F_A1", "H1a")
        mc.add("numRegimeLTotal", int(r_pooled_a0["n"]), "exp1_holdout_confirmatory.csv, F_A1/H1a, n (pooled regime L)")
        mc.add_if_finite("pooledAucGainVsAlphaZeroMedian", r_pooled_a0["median_diff"], "exp1_holdout_confirmatory.csv, F_A1/H1a, median_diff")
        mc.add_if_finite("pooledAucGainVsAlphaZeroPermP", r_pooled_a0["p_perm"], "exp1_holdout_confirmatory.csv, F_A1/H1a, p_perm", sig=2)
        mc.add_if_finite("pooledAucGainVsAlphaZeroHolmP", r_pooled_a0["p_holm"], "exp1_holdout_confirmatory.csv, F_A1/H1a, p_holm", sig=2)
        # Cliff's delta + bootstrap CI for the pooled confirmatory test
        # (K/Q1 step 2 addendum 2026-09-14) - added to the medians/p_holm
        # above, values cited in 01_uvod.tex/05_vysledky.tex/06_diskuse.tex
        # (previously a \todo).
        mc.add_if_finite("pooledCliffDeltaVsAlphaZero", r_pooled_a0["cliff_delta"], "exp1_holdout_confirmatory.csv, F_A1/H1a, cliff_delta", sig=2)
        mc.add_if_finite("pooledCliffDeltaVsAlphaZeroCiLow", r_pooled_a0["cliff_ci_low"], "exp1_holdout_confirmatory.csv, F_A1/H1a, cliff_ci_low", sig=2)
        mc.add_if_finite("pooledCliffDeltaVsAlphaZeroCiHigh", r_pooled_a0["cliff_ci_high"], "exp1_holdout_confirmatory.csv, F_A1/H1a, cliff_ci_high", sig=2)

        r_pooled_a1 = _row("F_A1", "H1b")
        mc.add_if_finite("pooledAucGainVsAlphaOneMedian", r_pooled_a1["median_diff"], "exp1_holdout_confirmatory.csv, F_A1/H1b, median_diff")
        mc.add_if_finite("pooledAucGainVsAlphaOnePermP", r_pooled_a1["p_perm"], "exp1_holdout_confirmatory.csv, F_A1/H1b, p_perm", sig=2)
        mc.add_if_finite("pooledAucGainVsAlphaOneHolmP", r_pooled_a1["p_holm"], "exp1_holdout_confirmatory.csv, F_A1/H1b, p_holm", sig=2)
        mc.add_if_finite("pooledCliffDeltaVsAlphaOne", r_pooled_a1["cliff_delta"], "exp1_holdout_confirmatory.csv, F_A1/H1b, cliff_delta", sig=2)
        mc.add_if_finite("pooledCliffDeltaVsAlphaOneCiLow", r_pooled_a1["cliff_ci_low"], "exp1_holdout_confirmatory.csv, F_A1/H1b, cliff_ci_low", sig=2)
        mc.add_if_finite("pooledCliffDeltaVsAlphaOneCiHigh", r_pooled_a1["cliff_ci_high"], "exp1_holdout_confirmatory.csv, F_A1/H1b, cliff_ci_high", sig=2)

        r_holdout_a0 = _row("F_A2", "H1a")
        mc.add_if_finite("holdoutAucGainVsAlphaZeroMedian", r_holdout_a0["median_diff"], "exp1_holdout_confirmatory.csv, F_A2/H1a, median_diff")
        mc.add_if_finite("holdoutAucGainVsAlphaZeroPermP", r_holdout_a0["p_perm"], "exp1_holdout_confirmatory.csv, F_A2/H1a, p_perm", sig=2)
        mc.add_if_finite("holdoutAucGainVsAlphaZeroHolmP", r_holdout_a0["p_holm"], "exp1_holdout_confirmatory.csv, F_A2/H1a, p_holm", sig=2)
        mc.add_if_finite("holdoutCliffDelta", r_holdout_a0["cliff_delta"], "exp1_holdout_confirmatory.csv, F_A2/H1a, cliff_delta", sig=2)
        mc.add_if_finite("holdoutCliffDeltaCiLow", r_holdout_a0["cliff_ci_low"], "exp1_holdout_confirmatory.csv, F_A2/H1a, cliff_ci_low", sig=2)
        mc.add_if_finite("holdoutCliffDeltaCiHigh", r_holdout_a0["cliff_ci_high"], "exp1_holdout_confirmatory.csv, F_A2/H1a, cliff_ci_high", sig=2)

        # F_A2 against alpha=1 (H1b, secondary/hold-out, n=17=numHoldoutEligibleRegimeL)
        # - added so the text can also cite the NEGATIVE result (the Holm
        # correction does NOT pass on this comparison, see reject_holm==False).
        r_holdout_a1 = _row("F_A2", "H1b")
        mc.add_if_finite("holdoutAucGainVsAlphaOneMedian", r_holdout_a1["median_diff"], "exp1_holdout_confirmatory.csv, F_A2/H1b, median_diff")
        mc.add_if_finite("holdoutAucGainVsAlphaOnePermP", r_holdout_a1["p_perm"], "exp1_holdout_confirmatory.csv, F_A2/H1b, p_perm", sig=3)
        mc.add_if_finite("holdoutAucGainVsAlphaOneHolmP", r_holdout_a1["p_holm"], "exp1_holdout_confirmatory.csv, F_A2/H1b, p_holm (WARNING: reject_holm==False, does NOT pass the correction)", sig=3)
        mc.add_if_finite("holdoutCliffDeltaVsAlphaOne", r_holdout_a1["cliff_delta"], "exp1_holdout_confirmatory.csv, F_A2/H1b, cliff_delta", sig=2)
        mc.add_if_finite("holdoutCliffDeltaVsAlphaOneCiLow", r_holdout_a1["cliff_ci_low"], "exp1_holdout_confirmatory.csv, F_A2/H1b, cliff_ci_low", sig=2)
        mc.add_if_finite("holdoutCliffDeltaVsAlphaOneCiHigh", r_holdout_a1["cliff_ci_high"], "exp1_holdout_confirmatory.csv, F_A2/H1b, cliff_ci_high", sig=2)
        mc.add("holdoutRejectHolmVsAlphaOne", "yes" if bool(r_holdout_a1["reject_holm"]) else "no", "exp1_holdout_confirmatory.csv, F_A2/H1b, reject_holm (passes/does not pass the Holm correction at alpha=0.05, one-sided)")
    mc.try_block("Q1 step 2: confirmatory analysis F_A1/F_A2 (pooled/hold-out regime L)", block_confirmatory)

    def block_rule_validation() -> None:
        df = _read_csv_required(tables_dir / "exp1_holdout_confirmatory.csv")
        sub = df[(df["family"] == "F_A6") & (df["hypothesis_id"] == "rule_validation")]
        if sub.empty:
            raise KeyError("exp1_holdout_confirmatory.csv is missing the row family='F_A6' (requires K6 on hold-out datasets).")
        row = sub.iloc[0]
        mc.add_if_finite("holdoutRuleSpearman", row["cliff_delta"], "exp1_holdout_confirmatory.csv, F_A6, Spearman(alpha_pred,alpha*) (column cliff_delta)", sig=2)
        mc.add_if_finite("holdoutGainFracMedian", row["median_diff"], "exp1_holdout_confirmatory.csv, F_A6, median gain_frac (column median_diff)", sig=2)
    mc.try_block("Q1 step 2: rule validation F_A6 (requires K6 hold-out)", block_rule_validation)

    def block_power() -> None:
        df = _read_csv_required(tables_dir / "power_analysis_regime.csv")

        def _n_required(method: str, scale_type: str, alpha: float) -> float:
            sub = df[(df["method"] == method) & (df["scale_type"] == scale_type) & np.isclose(df["alpha_onesided"], alpha)]
            if sub.empty:
                raise KeyError(f"power_analysis_regime.csv is missing method='{method}' scale_type='{scale_type}' alpha={alpha}.")
            return float(sub["n_required"].iloc[0])

        def _power_at_n(method: str, alpha: float, n: int) -> float:
            sub = df[(df["method"] == method) & np.isclose(df["alpha_onesided"], alpha) & (df["n"] == n)]
            if sub.empty:
                raise KeyError(f"power_analysis_regime.csv is missing method='{method}' alpha={alpha} n={n}.")
            return float(sub["power_at_n"].iloc[0])

        mc.add_if_finite("powerNRequiredNoetherRobust", _n_required("noether", "robust_mad", 0.025), "power_analysis_regime.csv, noether/robust_mad/alpha=0.025, n_required", sig=3)
        mc.add_if_finite("powerNRequiredNoetherFull", _n_required("noether", "sd", 0.025), "power_analysis_regime.csv, noether/sd/alpha=0.025, n_required", sig=3)
        mc.add_if_finite("powerAtTwentyFiveSign", _power_at_n("sign_exact", 0.025, 25), "power_analysis_regime.csv, sign_exact/alpha=0.025/n=25, power_at_n", sig=2)
        mc.add_if_finite("powerAtTwentyFiveMcEmpirical", _power_at_n("mc_empirical", 0.025, 25), "power_analysis_regime.csv, mc_empirical/alpha=0.025/n=25, power_at_n", sig=2)
    mc.try_block("Q1 step 2: power analysis (power_analysis_regime.csv)", block_power)


def _add_exp8_prop2_numbers(mc: MacroCollector, data_dir: Path) -> None:
    """Q1 step 1 (documentation/2026-09-14_exp8_prop2_check.md): the
    empirical test of Proposition 2 (local residual and rho_NN,
    03_metoda.tex). The fraction of rows with p2_holds (a violated
    assumption (P2) is NOT silently discarded - it is always a separate
    macro), the fraction of compliance with both bounds (a)/(b) AMONG
    p2_holds rows (proof check - see analysis item 1 of the K/Q1 step 1
    spec), the median tightness, and the Spearman correlations of
    rho_NN/the predicted factor r_eps^alpha against the empirical decline of
    R_near(Y)/R_near(Ybar) (analysis item 3)."""
    from scipy.stats import spearmanr

    def block_counts() -> None:
        df = _read_csv_required(data_dir / "exp8_prop2_check_results.csv")
        ok = df[df["status"] == "ok"]
        if ok.empty:
            raise ValueError("exp8_prop2_check_results.csv contains no rows with status 'ok'.")
        n_ok = int(ok.shape[0])
        n_error = int((df["status"] == "error").sum())
        holds = ok[ok["p2_holds"] == True]  # noqa: E712
        n_holds = int(holds.shape[0])
        mc.add("numExpEightRowsTotal", int(df.shape[0]), "exp8_prop2_check_results.csv, row count (incl. failed)")
        mc.add("numExpEightRowsError", n_error, "exp8_prop2_check_results.csv, count of rows with status 'error' (fail-loud - only eps_D<=eps_d_zero_tol SIMULTANEOUSLY with D_min=0, see src/sammon/prop2_check.py::evaluate_proposition2, 2026-09-14 fix)")
        mc.add("fracExpEightPropTwoHolds", f"{n_holds}/{n_ok}", "exp8_prop2_check_results.csv, fraction of 'ok' rows with p2_holds==True (with tolerance p2_tolerance_rel/abs)")

        # proof check: AMONG p2_holds rows, R_near(Y) > bound_a/b must NOT
        # hold (except for rounding tolerance matching p2_tolerance_abs) -
        # any counterexample is a bug in the derivation OR the
        # implementation (see K/Q1 step 1 spec item 1), NOT a reason for a
        # silent exclusion.
        tol = 1e-6 * holds["bound_a"].abs().clip(lower=1.0) if not holds.empty else pd.Series(dtype=float)
        viol_a = int((holds["R_near_Y"] > holds["bound_a"] + tol).sum()) if not holds.empty else 0
        viol_b = int((holds["R_near_Y"] > holds["bound_b"] + tol).sum()) if not holds.empty else 0
        mc.add("numExpEightBoundAViolations", f"{viol_a}/{n_holds}", "exp8_prop2_check_results.csv, p2_holds==True rows with R_near_Y > bound_a (numerical tolerance 1e-6 relative)")
        mc.add("numExpEightBoundBViolations", f"{viol_b}/{n_holds}", "exp8_prop2_check_results.csv, p2_holds==True rows with R_near_Y > bound_b (numerical tolerance 1e-6 relative)")

        mc.add_if_finite("medExpEightTightA", holds["tight_a"].median(), "exp8_prop2_check_results.csv, tight_a, median over p2_holds rows (R_near(Y)/bound_a)", sig=2)
        mc.add_if_finite("medExpEightTightB", holds["tight_b"].median(), "exp8_prop2_check_results.csv, tight_b, median over p2_holds rows (R_near(Y)/bound_b)", sig=2)
        # Interquartile range (Q3-Q1) of the tightness - both bounds (a)/(b)
        # are LOOSE (median on the order of 1e-3..1e-4, see
        # medExpEightTightA/B above), this number rounds out the spread: the
        # conclusion is QUALITATIVE/DIRECTIONAL (Proposition 2 held up
        # numerically, but the bounds are not tight), NOT a sharp bound -
        # see documentation/2026-09-14_exp8_prop2_check.md.
        mc.add_if_finite("iqrExpEightTightA", holds["tight_a"].quantile(0.75) - holds["tight_a"].quantile(0.25), "exp8_prop2_check_results.csv, tight_a, interquartile range (Q3-Q1) over p2_holds rows - the bounds are loose, this is a directional/qualitative result, not a sharp bound", sig=2)
        mc.add_if_finite("iqrExpEightTightB", holds["tight_b"].quantile(0.75) - holds["tight_b"].quantile(0.25), "exp8_prop2_check_results.csv, tight_b, interquartile range (Q3-Q1) over p2_holds rows - the bounds are loose, this is a directional/qualitative result, not a sharp bound", sig=2)

        # Distinguishing datasets with exact duplicates (D_ij=0, column
        # has_duplicates - 2026-09-14 fix of a too-strict D_min=0 check,
        # documentation/2026-09-14_exp8_prop2_check.md): the median
        # tightness in BOTH subgroups separately, so it is verifiable that
        # the conclusion about the looseness of the bounds does not stem
        # only from datasets without duplicates (or vice versa).
        if "has_duplicates" in ok.columns:
            n_datasets_dup = int(ok.loc[ok["has_duplicates"] == True, "dataset"].nunique())  # noqa: E712
            n_datasets_total = int(ok["dataset"].nunique())
            mc.add("numExpEightDatasetsWithDuplicates", n_datasets_dup, "exp8_prop2_check_results.csv, number of distinct datasets (among 'ok' rows) with at least one D_ij=0 (has_duplicates==True)")
            mc.add("numExpEightDatasetsTotal", n_datasets_total, "exp8_prop2_check_results.csv, number of distinct datasets among 'ok' rows")

            holds_dup = holds[holds["has_duplicates"] == True]  # noqa: E712
            holds_nodup = holds[holds["has_duplicates"] == False]  # noqa: E712
            mc.add("numExpEightHoldsRowsDup", int(holds_dup.shape[0]), "exp8_prop2_check_results.csv, count of p2_holds==True rows with has_duplicates==True")
            mc.add("numExpEightHoldsRowsNodup", int(holds_nodup.shape[0]), "exp8_prop2_check_results.csv, count of p2_holds==True rows with has_duplicates==False")
            mc.add_if_finite("medExpEightTightADup", holds_dup["tight_a"].median(), "exp8_prop2_check_results.csv, tight_a, median over p2_holds rows with has_duplicates==True (subgroup with exact duplicates)", sig=2)
            mc.add_if_finite("medExpEightTightBDup", holds_dup["tight_b"].median(), "exp8_prop2_check_results.csv, tight_b, median over p2_holds rows with has_duplicates==True (subgroup with exact duplicates)", sig=2)
            mc.add_if_finite("medExpEightTightANodup", holds_nodup["tight_a"].median(), "exp8_prop2_check_results.csv, tight_a, median over p2_holds rows with has_duplicates==False (subgroup without duplicates)", sig=2)
            mc.add_if_finite("medExpEightTightBNodup", holds_nodup["tight_b"].median(), "exp8_prop2_check_results.csv, tight_b, median over p2_holds rows with has_duplicates==False (subgroup without duplicates)", sig=2)
    mc.try_block("exp8 Proposition 2: counts and bound tightness (Q1 step 1, items 1-2)", block_counts)

    def block_correlations() -> None:
        df = _read_csv_required(data_dir / "exp8_prop2_check_results.csv")
        holds = df[(df["status"] == "ok") & (df["p2_holds"] == True)]  # noqa: E712
        if holds.empty:
            raise ValueError("exp8_prop2_check_results.csv has no 'ok' row with p2_holds==True for the correlations.")

        # (a) does the theoretical factor r_eps^alpha agree with the empirical
        # decline of R_near(Y)/R_near(Ybar) over ALL (dataset,alpha,seed) rows?
        pooled = holds[["predicted_factor_r_eps_alpha", "r_near_ratio"]].dropna()
        if pooled.shape[0] >= 3:
            rho, pvalue = spearmanr(pooled["predicted_factor_r_eps_alpha"], pooled["r_near_ratio"])
            mc.add("spearmanPredictedFactorVsDeclineRho", float(rho), "exp8_prop2_check_results.csv, Spearman(predicted_factor_r_eps_alpha, r_near_ratio), pooled over (dataset,alpha,seed), p2_holds==True", sig=3)
            mc.add("spearmanPredictedFactorVsDeclineP", float(pvalue), "exp8_prop2_check_results.csv, Spearman(predicted_factor_r_eps_alpha, r_near_ratio) p-value, pooled", sig=2)
            mc.add("numExpEightPooledCorrelationRows", int(pooled.shape[0]), "exp8_prop2_check_results.csv, number of rows in the pooled correlation predicted_factor_r_eps_alpha vs. r_near_ratio")

        # the same, separately for datasets WITH duplicates and WITHOUT
        # duplicates (has_duplicates, 2026-09-14 fix of a too-strict D_min=0
        # check, documentation/2026-09-14_exp8_prop2_check.md) - verifies
        # that the association of the theoretical factor with the empirical
        # decline does not stem only from one subgroup.
        if "has_duplicates" in holds.columns:
            for suffix2, sub_holds in (("Dup", holds[holds["has_duplicates"] == True]), ("Nodup", holds[holds["has_duplicates"] == False])):  # noqa: E712
                pooled2 = sub_holds[["predicted_factor_r_eps_alpha", "r_near_ratio"]].dropna()
                if pooled2.shape[0] < 3:
                    continue
                rho2, pvalue2 = spearmanr(pooled2["predicted_factor_r_eps_alpha"], pooled2["r_near_ratio"])
                mc.add(f"spearmanPredictedFactorVsDecline{suffix2}Rho", float(rho2), f"exp8_prop2_check_results.csv, Spearman(predicted_factor_r_eps_alpha, r_near_ratio), pooled, p2_holds==True, has_duplicates=={'True' if suffix2 == 'Dup' else 'False'}", sig=3)
                mc.add(f"spearmanPredictedFactorVsDecline{suffix2}P", float(pvalue2), f"exp8_prop2_check_results.csv, Spearman(predicted_factor_r_eps_alpha, r_near_ratio) p-value, has_duplicates=={'True' if suffix2 == 'Dup' else 'False'}", sig=2)
                mc.add(f"numExpEightPooledCorrelationRows{suffix2}", int(pooled2.shape[0]), f"exp8_prop2_check_results.csv, number of rows in the correlation predicted_factor_r_eps_alpha vs. r_near_ratio, has_duplicates=={'True' if suffix2 == 'Dup' else 'False'}")

        # (b) does rho_NN (concentration, per dataset) predict the empirical
        # decline of R_near at fixed alpha? - per-dataset median over seeds,
        # for each alpha from exp8_prop2_check.summary_alphas (K/Q1 step 1 item 3).
        for alpha_value in _EXP8_SUMMARY_ALPHAS:
            sub = holds[np.isclose(holds["alpha"], alpha_value)]
            per_dataset = sub.groupby("dataset")[["rho_nn", "r_near_ratio"]].median().dropna()
            if per_dataset.shape[0] < 3:
                continue
            rho, pvalue = spearmanr(per_dataset["rho_nn"], per_dataset["r_near_ratio"])
            suffix = num_to_words(alpha_value)
            mc.add(f"spearmanRhoNnVsDeclineAlpha{suffix}Rho", float(rho), f"exp8_prop2_check_results.csv, Spearman(rho_nn, r_near_ratio) over datasets at alpha={alpha_value:g} (median over seeds), p2_holds==True", sig=3)
            mc.add(f"spearmanRhoNnVsDeclineAlpha{suffix}P", float(pvalue), f"exp8_prop2_check_results.csv, Spearman(rho_nn, r_near_ratio) p-value at alpha={alpha_value:g}", sig=2)
            mc.add(f"numExpEightDeclineAlpha{suffix}Datasets", int(per_dataset.shape[0]), f"exp8_prop2_check_results.csv, number of datasets in the correlation rho_nn vs. r_near_ratio at alpha={alpha_value:g}")
    mc.try_block("exp8 Proposition 2: correlations rho_NN/r_eps^alpha vs. empirical R_near decline (Q1 step 1, item 3)", block_correlations)


def _add_exp8_prop2_per_dataset_correlation(mc: MacroCollector, data_dir: Path) -> None:
    """Review round 2 fix (2026-09-18), point 1: `spearmanPredictedFactorVsDeclineRho`/
    `...P` in `_add_exp8_prop2_numbers` pool 1885 (dataset,alpha,seed) rows
    that come from only 32 datasets - the unit of analysis must be the
    DATASET (project rule "never infer across pooled rows", see
    exp1_holdout_confirmatory.py), so this group adds the SAME correlation
    computed WITHIN each dataset (across its own alpha/seed rows), then
    summarized ACROSS datasets via a median/IQR and a sign test (Wilcoxon
    signed-rank against zero). The old pooled macros are kept unchanged -
    the supplement may still cite them - this only ADDS the per-dataset
    version for the main text.

    A dataset is included only if BOTH `predicted_factor_r_eps_alpha` and
    `r_near_ratio` take at least 3 distinct values among its p2_holds==True
    rows - otherwise the within-dataset Spearman correlation is undefined
    (e.g. two_moons, where r_near_ratio==1.0 on every row: NaN, silently
    excluded, not fabricated)."""
    from scipy.stats import spearmanr, wilcoxon

    def block() -> None:
        df = _read_csv_required(data_dir / "exp8_prop2_check_results.csv")
        holds = df[(df["status"] == "ok") & (df["p2_holds"] == True)]  # noqa: E712
        if holds.empty:
            raise ValueError("exp8_prop2_check_results.csv has no 'ok' row with p2_holds==True for the per-dataset correlations.")

        rhos: dict[str, float] = {}
        for dataset_name, group in holds.groupby("dataset"):
            pair = group[["predicted_factor_r_eps_alpha", "r_near_ratio"]].dropna()
            if pair["predicted_factor_r_eps_alpha"].nunique() < 3 or pair["r_near_ratio"].nunique() < 3:
                continue
            rho, _pvalue = spearmanr(pair["predicted_factor_r_eps_alpha"], pair["r_near_ratio"])
            if np.isfinite(rho):
                rhos[dataset_name] = float(rho)
        if len(rhos) < 2:
            raise ValueError("exp8_prop2_check_results.csv: fewer than 2 datasets have a defined per-dataset Spearman(predicted_factor_r_eps_alpha, r_near_ratio).")
        rho_series = pd.Series(rhos)
        n_datasets = int(rho_series.shape[0])
        n_positive = int((rho_series > 0.0).sum())

        mc.add_if_finite(
            "medExpEightPerDatasetDeclineRho", rho_series.median(),
            "exp8_prop2_check_results.csv, Spearman(predicted_factor_r_eps_alpha, r_near_ratio) computed WITHIN each dataset (p2_holds==True, datasets with >=3 distinct values of both variables), median over datasets - unit of analysis = dataset (review round 2, point 1)", sig=3,
        )
        mc.add_if_finite(
            "iqrLowExpEightPerDatasetDeclineRho", rho_series.quantile(0.25),
            "exp8_prop2_check_results.csv, per-dataset Spearman(predicted_factor_r_eps_alpha, r_near_ratio), first quartile (Q1) over datasets", sig=3,
        )
        mc.add_if_finite(
            "iqrHighExpEightPerDatasetDeclineRho", rho_series.quantile(0.75),
            "exp8_prop2_check_results.csv, per-dataset Spearman(predicted_factor_r_eps_alpha, r_near_ratio), third quartile (Q3) over datasets", sig=3,
        )
        mc.add(
            "numExpEightPerDatasetPositiveDeclineRho", f"{n_positive}/{n_datasets}",
            "exp8_prop2_check_results.csv, per-dataset Spearman(predicted_factor_r_eps_alpha, r_near_ratio), count of datasets with rho>0 out of datasets with a defined per-dataset correlation",
        )
        stat = wilcoxon(rho_series.to_numpy())
        mc.add_if_finite(
            "wilcoxonExpEightPerDatasetDeclineRhoP", float(stat.pvalue),
            "exp8_prop2_check_results.csv, per-dataset Spearman(predicted_factor_r_eps_alpha, r_near_ratio), two-sided Wilcoxon signed-rank test against zero over datasets (scipy.stats.wilcoxon defaults: zero_method='wilcox', method='auto')", sig=2,
        )
    mc.try_block("exp8 Proposition 2: per-dataset Spearman(predicted_factor,decline), unit of analysis = dataset (review round 2, point 1)", block)


# alphas for which the correlations rho_NN vs. R_near decline are computed
# in numbers.tex (matches config_experiments.yaml exp8_prop2_check.summary_alphas
# - read directly from the config so the number is not duplicated in two places)
_EXP8_SUMMARY_ALPHAS: list[float] = [float(a) for a in load_experiments_config()["exp8_prop2_check"]["summary_alphas"]]


def _camel(snake_name: str) -> str:
    """snake_case -> CamelCase (letters only - LaTeX macros do not allow
    digits/underscores, see the module docstring) - used for metric names in
    Q1 step 3 (exp9_metric_fidelity) macros, so the metric is not
    transcribed by hand in two places."""
    return "".join(part.capitalize() for part in snake_name.split("_"))


# Q1 step 3 (reserse/2026-09-14_specifikace_rozsireni_q1.md, PART B.7 item 7)
# - scenario (repo key S1/S2/S3) -> an ASCII word for the macro name (digits
# not allowed in LaTeX \newcommand names).
_METRIC_FIDELITY_SCENARIO_WORD = {"S1": "SOne", "S2": "STwo", "S3": "SThree"}


def _add_exp9_metric_fidelity_numbers(mc: MacroCollector, data_dir: Path, tables_dir: Path) -> None:
    """Q1 step 3 (B.7 item 7) - metric-fidelity macros with a known truth:
    counts/medians of the primary metric per scenario (S1/S2/S3) and
    baseline (F_B1: primary_baselines), Holm's p and Cliff's delta (+
    bootstrap CI) from `exp9_metric_fidelity_stats.csv`, radius_slope (S1)
    and aspect_ratio_error (S3) for pred vs. an illustrative neighbor
    method, and the price of pred vs. alpha0 (F_B3) on the primary metric of S1."""
    exp_cfg = load_experiments_config()["exp9_metric_fidelity"]
    active_scenarios: list[str] = exp_cfg["active_scenarios"]
    primary_metric: dict[str, str] = exp_cfg["primary_metric"]
    primary_baselines: list[str] = exp_cfg["primary_baselines"]
    pred_method: str = exp_cfg["pred_method"]

    def block_counts_and_medians() -> None:
        df = _read_csv_required(data_dir / "exp9_metric_fidelity_results.csv")
        ok = df[df["status"] == "ok"]
        if ok.empty:
            raise ValueError("exp9_metric_fidelity_results.csv has no 'ok' rows.")
        mc.add("numMetricFidelityReplicates", int(exp_cfg["replicates"]), "config_experiments.yaml exp9_metric_fidelity.replicates (B.2: R replicates per scenario)")
        mc.add("numMetricFidelityScenarios", int(len(active_scenarios)), "config_experiments.yaml exp9_metric_fidelity.active_scenarios, number of active scenarios (S1-S3)")

        for scenario in ("S1", "S2", "S3"):
            if scenario not in active_scenarios:
                continue
            metric = primary_metric[scenario]
            metric_word = _camel(metric)
            scen_word = _METRIC_FIDELITY_SCENARIO_WORD[scenario]
            for method in primary_baselines:
                suffix = _suffix(method)
                vals = ok[(ok["scenario"] == scenario) & (ok["method"] == method)][metric].dropna()
                mc.add_if_finite(
                    f"mf{scen_word}{metric_word}{suffix}Median", vals.median(),
                    f"exp9_metric_fidelity_results.csv, {metric}, median over replicates, scenario={scenario}, method={method}", sig=3,
                )
    mc.try_block("exp9 metric fidelity: counts and medians of the primary metric (F_B1 baselines)", block_counts_and_medians)

    def block_stats() -> None:
        stats_df = _read_csv_required(tables_dir / "exp9_metric_fidelity_stats.csv")
        for scenario in ("S1", "S2", "S3"):
            if scenario not in active_scenarios:
                continue
            scen_word = _METRIC_FIDELITY_SCENARIO_WORD[scenario]
            for method in primary_baselines:
                suffix = _suffix(method)
                row = stats_df[(stats_df["family"] == "F_B1") & (stats_df["scenario"] == scenario) & (stats_df["method_b"] == method)]
                if row.empty:
                    continue
                r = row.iloc[0]
                mc.add_if_finite(f"mf{scen_word}PredVs{suffix}HolmP", r["p_holm"], f"exp9_metric_fidelity_stats.csv, F_B1, scenario={scenario}, {pred_method} vs {method}, p after the Holm correction", sig=2)
                mc.add_if_finite(f"mf{scen_word}PredVs{suffix}CliffDelta", r["cliff_delta"], f"exp9_metric_fidelity_stats.csv, F_B1, scenario={scenario}, {pred_method} vs {method}, Cliff's delta", sig=3)
                mc.add_if_finite(f"mf{scen_word}PredVs{suffix}CliffDeltaCiLow", r["cliff_ci_low"], f"exp9_metric_fidelity_stats.csv, F_B1, scenario={scenario}, {pred_method} vs {method}, Cliff's delta - bootstrap CI lower bound", sig=3)
                mc.add_if_finite(f"mf{scen_word}PredVs{suffix}CliffDeltaCiHigh", r["cliff_ci_high"], f"exp9_metric_fidelity_stats.csv, F_B1, scenario={scenario}, {pred_method} vs {method}, Cliff's delta - bootstrap CI upper bound", sig=3)
    mc.try_block("exp9 metric fidelity: F_B1 Holm's p and Cliff's delta", block_stats)

    def block_radius_and_aspect() -> None:
        df = _read_csv_required(data_dir / "exp9_metric_fidelity_results.csv")
        ok = df[df["status"] == "ok"]
        if "S1" in active_scenarios:
            for method, name in ((pred_method, "Pred"), ("tsne_auto", "TsneAuto")):
                vals = ok[(ok["scenario"] == "S1") & (ok["method"] == method)]["radius_slope"].dropna()
                mc.add_if_finite(
                    f"mfSOneRadiusSlope{name}", vals.median(),
                    f"exp9_metric_fidelity_results.csv, radius_slope, median over replicates, scenario=S1, method={method} "
                    "(beta_r=1 preserves cluster radius ratios, beta_r=0 equalizes sizes - Kobak&Berens 2019)", sig=3,
                )
        if "S3" in active_scenarios:
            for method, name in ((pred_method, "Pred"), ("umap_auto", "UmapAuto")):
                vals = ok[(ok["scenario"] == "S3") & (ok["method"] == method)]["aspect_ratio_error"].dropna()
                mc.add_if_finite(
                    f"mfSThreeAspectError{name}", vals.median(),
                    f"exp9_metric_fidelity_results.csv, aspect_ratio_error, median over replicates, scenario=S3, method={method}", sig=3,
                )
    mc.try_block("exp9 metric fidelity: radius_slope (S1) and aspect_ratio_error (S3), pred vs. an illustrative neighbor method", block_radius_and_aspect)

    def block_pred_vs_alpha_zero() -> None:
        stats_df = _read_csv_required(tables_dir / "exp9_metric_fidelity_stats.csv")
        row = stats_df[(stats_df["family"] == "F_B3") & (stats_df["scenario"] == "S1") & (stats_df["method_b"] == "sammon_alpha0_smacof")]
        if row.empty:
            raise ValueError("exp9_metric_fidelity_stats.csv has no row F_B3 S1 sammon_alpha_pred vs sammon_alpha0_smacof.")
        mc.add_if_finite(
            "mfSOnePredVsAlphaZeroLreMedianDiff", row.iloc[0]["median_diff"],
            "exp9_metric_fidelity_stats.csv, F_B3, S1, centroid_lre, median(pred-alpha0) - the price of local emphasis in terms of global fidelity (B.5 F_B3)", sig=3,
        )
    mc.try_block("exp9 metric fidelity: pred vs. alpha0 (F_B3, the price of local emphasis in terms of global fidelity)", block_pred_vs_alpha_zero)


def _add_exp10_identifiability_numbers(mc: MacroCollector, data_dir: Path, mode: str) -> None:
    """reserse/2026-09-17_zostreni_propozice2.md, section 9.3 - the macro
    list is given there EXACTLY under the names 'ExpNine...' (the reserse
    note calls the experiment 'E9'; it was implemented as
    src/experiments/exp10_identifiability_check.py /
    exp10_identifiability_stats.py to avoid a collision with the
    already-existing exp9_metric_fidelity - the macro NAMES from section 9.3
    are kept UNCHANGED, only the source CSV file names differ from what the
    reserse note calls them, see the per-macro `source:` comments below).

    Sources (both mode-specific, results/data/[<mode>/]):
      exp10_identifiability_check_results.csv - raw per-(dataset,alpha) rows
        (src/experiments/exp10_identifiability_check.py).
      exp10_identifiability_stats.csv - regression/H1/H2/H3/quadratic-law
        (src/experiments/exp10_identifiability_stats.py, long format:
        columns analysis/label/value/ci_low/ci_high/pvalue/n/note).
    A missing/incomplete CSV SKIPS this whole macro group (mc.try_block) -
    exp10 has not necessarily run yet (Q1 extension, 2026-09-17)."""

    def block_raw() -> None:
        df = _read_csv_required(data_dir / "exp10_identifiability_check_results.csv")
        ok = df[df["status"] == "ok"]
        if ok.empty:
            raise ValueError("exp10_identifiability_check_results.csv has no rows with status 'ok'.")

        a1 = ok[np.isclose(ok["alpha"], 1.0)]
        if a1.empty:
            raise ValueError("exp10_identifiability_check_results.csv has no alpha=1.0 rows.")
        mc.add_if_finite("medExpNineCAlphaOne", a1["c_alpha"].median(), "exp10_identifiability_check_results.csv, c_alpha, median over datasets at alpha=1.0 (c_1, Tvrzeni 3)", sig=3)
        mc.add_if_finite("iqrExpNineCAlphaOne", a1["c_alpha"].quantile(0.75) - a1["c_alpha"].quantile(0.25), "exp10_identifiability_check_results.csv, c_alpha, interquartile range (Q3-Q1) over datasets at alpha=1.0", sig=3)
        mc.add_if_finite("medExpNineGammaTildeZero", a1["gamma_tilde_Y0"].median(), "exp10_identifiability_check_results.csv, gamma_tilde_Y0, median over datasets (alpha-independent, read off the alpha=1.0 row - Veta 1 notation gamma~_0)", sig=3)
        mc.add("numExpNineNontrivialAlphaOne", int(a1["bound_nontrivial"].sum()), "exp10_identifiability_check_results.csv, count of datasets with bound_nontrivial==True (c_1*gamma~_1<1) at alpha=1.0, out of " + str(int(a1.shape[0])))

        nontrivial_pos = ok[(ok["alpha"] > 0.0) & (ok["bound_nontrivial"] == True)]  # noqa: E712
        # NOTE naming: the reserse note (section 9.3) spells this macro
        # 'medExpNineTightV1' (V1 = "Veta 1") - LaTeX \newcommand names may
        # only contain LETTERS (see the module docstring/`_macro_line`,
        # project rule, same convention as num_to_words for alpha levels
        # elsewhere in this file), so the digit is spelled out: V1 -> VOne.
        mc.add_if_finite("medExpNineTightVOne", nontrivial_pos["tight_v1"].median(), "exp10_identifiability_check_results.csv, tight_v1 (Delta_prime/bound_v1), median over (dataset,alpha>0) rows with bound_nontrivial==True (reserse 2026-09-17 section 9.3 spells this macro 'medExpNineTightV1'; V1->VOne, LaTeX macro names are letters-only)", sig=2)
        mc.add_if_finite("medExpNineTightSandwich", nontrivial_pos["tight_sandwich"].median(), "exp10_identifiability_check_results.csv, tight_sandwich (Delta_prime/bound_sandwich, Lemma 1), median over the SAME rows as medExpNineTightVOne - for comparison", sig=2)

        pos = ok[ok["alpha"] > 0.0]
        mc.add_if_finite("medExpNineSOne", pos["S1"].median(), "exp10_identifiability_check_results.csv, S1 (E8 slack decomposition, optimization gain vs. PCA init), median over (dataset,alpha>0) rows", sig=2)
        mc.add_if_finite("medExpNineSTwo", pos["S2"].median(), "exp10_identifiability_check_results.csv, S2 (E8 slack decomposition, budget factor N/|P_near|), median over (dataset,alpha>0) rows", sig=2)
        mc.add_if_finite("medExpNineSThree", pos["S3"].median(), "exp10_identifiability_check_results.csv, S3 (E8 slack decomposition, within-block weight heterogeneity), median over (dataset,alpha>0) rows", sig=2)
        mc.add_if_finite("medExpNineSFour", pos["S4"].median(), "exp10_identifiability_check_results.csv, S4 (E8 slack decomposition, bound_b/bound_a), median over (dataset,alpha>0) rows", sig=2)
        # Total slack of bound (a), i.e. the product S1*S2*S3 = 1/tight_a. Taken
        # as the median of the per-row product, NOT as the product of the three
        # medians above (the median is not multiplicative).
        mc.add_if_finite("medExpNineSlackA", (pos["S1"] * pos["S2"] * pos["S3"]).median(), "exp10_identifiability_check_results.csv, S1*S2*S3 = 1/tight_a (total slack of Proposition 2 bound (a)), median of the per-row product over (dataset,alpha>0) rows", sig=2)
        mc.add("numExpNineDatasets", int(ok["dataset"].nunique()), "exp10_identifiability_check_results.csv, number of distinct datasets with a successful identifiability check")
        # The quadratic law (Veta 3) is only testable where the exact Hessian of
        # the unweighted stress was computed; `fracExpNineQuadraticLawHolds` is a
        # fraction of THOSE rows, so the text must also report how many there are.
        tested = pos[~pos["hessian_skipped"].astype(bool)]
        mc.add("numExpNineHessianDatasets", int(tested["dataset"].nunique()), "exp10_identifiability_check_results.csv, number of datasets with hessian_skipped==False for at least one alpha>0 (the only ones on which the quadratic law is testable)")
        mc.add("numExpNineHessianRows", int(tested.shape[0]), "exp10_identifiability_check_results.csv, number of (dataset,alpha>0) rows with hessian_skipped==False (denominator of fracExpNineQuadraticLawHolds)")

        exp10_cfg = resolve_experiment_config("exp10_identifiability_check", mode)
        eta = float(exp10_cfg["eta"])
        certified = pos[pos["cert_lower"] >= eta].groupby("dataset").size()
        # Direct certificate: the FIRST inequality of the dual proposition
        # (sigma_alpha(Y_0)/sigma_alpha(Y~)), i.e. before the block relaxation
        # that `cert_lower` applies. Reported alongside it because the relaxed
        # form never fires on real data - see the limitation in the Discussion.
        direct_pos = pos[pos["cert_lower_direct"] > 0.0]
        mc.add("numExpNineCertDirectRows", int(direct_pos.shape[0]), "exp10_identifiability_check_results.csv, count of (dataset,alpha>0) rows with cert_lower_direct>0 (direct witness certificate of Delta'_alpha>0)")
        mc.add("numExpNineCertDirectTotalRows", int(pos.shape[0]), "exp10_identifiability_check_results.csv, total number of (dataset,alpha>0) rows (denominator of numExpNineCertDirectRows)")
        mc.add("numExpNineCertDirectDatasets", int(direct_pos["dataset"].nunique()), "exp10_identifiability_check_results.csv, number of datasets with cert_lower_direct>0 for at least one alpha>0")
        mc.add_if_finite("medExpNineCertDirect", direct_pos["cert_lower_direct"].median(), "exp10_identifiability_check_results.csv, cert_lower_direct, median over rows where it is positive (certified lower bound on Delta'_alpha)", sig=2)
        mc.add_if_finite("maxExpNineIdentityResidual", ok["identity_residual"].abs().max(), "exp10_identifiability_check_results.csv, identity_residual, maximum absolute value over ALL rows (numerical check of the exchange-rate identity, Lemma 2)", sig=2)
        mc.add("numExpNineCertifiedDatasets", int(certified.shape[0]), f"exp10_identifiability_check_results.csv, count of datasets with cert_lower>=eta (eta={eta:g}, exp10_identifiability_check.eta) for at least one alpha>0 (Veta 2 certificate), out of {int(pos['dataset'].nunique())}")
    def block_alpha_grid() -> None:
        # Size of the alpha grid searched by alpha_auto. This is a property of
        # THIS configuration, not a universal constant - alpha_pred saves the
        # whole grid search, whatever its size - so the article must never
        # state the number by hand (it did: "13 behu SMACOF" in 03_metoda.tex).
        grid = list(resolve_experiment_config("exp6_alpha_curves", mode)["alpha_grid"])
        mc.add("numAlphaGridPoints", len(grid), "config_experiments.yaml, exp6_alpha_curves.alpha_grid, number of grid points searched by alpha_auto")
        mc.add_if_finite("alphaGridStep", (grid[1] - grid[0]) if len(grid) > 1 else float("nan"), "config_experiments.yaml, exp6_alpha_curves.alpha_grid, spacing between consecutive grid points", sig=2)
        mc.add_if_finite("alphaGridMax", max(grid), "config_experiments.yaml, exp6_alpha_curves.alpha_grid, largest alpha searched", sig=2)
    mc.try_block("alpha grid size (alpha_auto reference strategy)", block_alpha_grid)

    mc.try_block("exp10 identifiability: raw per-row medians/counts (reserse 2026-09-17, section 9.3)", block_raw)

    def block_stats() -> None:
        stats = _read_csv_required(data_dir / "exp10_identifiability_stats.csv")

        def _row(analysis: str, label: str):
            r = stats[(stats["analysis"] == analysis) & (stats["label"] == label)]
            return r.iloc[0] if not r.empty else None

        r = _row("regression_slack_vs_pairs", "slope")
        if r is not None:
            mc.add_if_finite("slopeExpNineSlackVsPairs", r["value"], "exp10_identifiability_stats.csv, analysis=regression_slack_vs_pairs/label=slope, OLS slope of log(1/tight_a) ~ log(N_over_n_near) (exp8_prop2_check rows, p2_holds=True)", sig=3)
            mc.add_if_finite("slopeExpNineSlackVsPairsCiLow", r["ci_low"], "exp10_identifiability_stats.csv, analysis=regression_slack_vs_pairs/label=slope, cluster (dataset) bootstrap CI lower bound", sig=3)
            mc.add_if_finite("slopeExpNineSlackVsPairsCiHigh", r["ci_high"], "exp10_identifiability_stats.csv, analysis=regression_slack_vs_pairs/label=slope, cluster (dataset) bootstrap CI upper bound", sig=3)

        r = _row("H1_spearman", "cGammaTildeZero_vs_G_auc_oracle")
        if r is not None:
            mc.add_if_finite("spearmanExpNineCeilingVsGainRho", r["value"], "exp10_identifiability_stats.csv, analysis=H1_spearman/label=cGammaTildeZero_vs_G_auc_oracle, Spearman(c_1*gamma~_0, G_auc_oracle) over datasets", sig=3)
            mc.add_if_finite("spearmanExpNineCeilingVsGainP", r["pvalue"], "exp10_identifiability_stats.csv, analysis=H1_spearman/label=cGammaTildeZero_vs_G_auc_oracle, permutation p-value", sig=2)

        r = _row("H1_spearman", "I0exact_vs_G_auc_oracle")
        if r is not None:
            mc.add_if_finite("spearmanExpNineIZeroVsGainRho", r["value"], "exp10_identifiability_stats.csv, analysis=H1_spearman/label=I0exact_vs_G_auc_oracle, Spearman(I0_exact, G_auc_oracle) over datasets", sig=3)
            mc.add_if_finite("spearmanExpNineIZeroVsGainP", r["pvalue"], "exp10_identifiability_stats.csv, analysis=H1_spearman/label=I0exact_vs_G_auc_oracle, permutation p-value", sig=2)

        r = _row("H2_phi_low", "phi")
        if r is not None:
            mc.add_if_finite("medExpNinePhiLow", r["value"], "exp10_identifiability_stats.csv, analysis=H2_phi_low/label=phi, median(G_pred/G_auc_oracle) over 'low'-stratum datasets", sig=3)
            mc.add_if_finite("medExpNinePhiLowCiLow", r["ci_low"], "exp10_identifiability_stats.csv, analysis=H2_phi_low/label=phi, percentile bootstrap CI lower bound", sig=3)
            mc.add_if_finite("medExpNinePhiLowCiHigh", r["ci_high"], "exp10_identifiability_stats.csv, analysis=H2_phi_low/label=phi, percentile bootstrap CI upper bound", sig=3)

        r = _row("H3_tost_high", "G_pred")
        if r is not None:
            mc.add_if_finite("tostExpNineHighP", r["pvalue"], "exp10_identifiability_stats.csv, analysis=H3_tost_high/label=G_pred, TOST p-value (equivalence of G_pred to 0 within +-delta_eq in the 'high' stratum)", sig=2)

        r = _row("quadratic_law_fraction", "pooled")
        if r is not None:
            mc.add_if_finite("fracExpNineQuadraticLawHolds", r["value"], "exp10_identifiability_stats.csv, analysis=quadratic_law_fraction/label=pooled, fraction of (dataset,alpha>0) rows with quadratic_law_holds==True", sig=2)
    mc.try_block("exp10 identifiability: regression/H1/H2/H3/quadratic-law (reserse 2026-09-17, section 9.3)", block_stats)


def _add_exp10_negative_delta_survival(mc: MacroCollector, data_dir: Path) -> None:
    """Review round 2 fix (2026-09-18), point 3: 06_diskuse.tex claims that
    filtering `exp10_identifiability_check_results.csv` to the
    convergence-driven threshold tau measured by
    `exp14_convergence_robustness.py` (`tau_source=='measured_from_exp11'`)
    reverses no claim. That is true for `Delta_prime` but NOT for `Delta`:
    of the (dataset,alpha>0) rows with a NEGATIVE `Delta`, some survive the
    same |Delta|>=tau AND |Delta_prime|>=tau filter that
    `exp14_convergence_robustness.py::_kept_rows` applies. tau is read from
    `exp14_convergence_robustness_results.csv` (measured row), NEVER
    hardcoded (project rule - no magic numbers)."""

    def block() -> None:
        e10 = _read_csv_required(data_dir / "exp10_identifiability_check_results.csv")
        pos = e10[(e10["status"] == "ok") & (e10["alpha"] > 0.0)]
        if pos.empty:
            raise ValueError("exp10_identifiability_check_results.csv has no status=='ok' & alpha>0 rows.")

        e14 = _read_csv_required(data_dir / "exp14_convergence_robustness_results.csv")
        measured = e14[e14["tau_source"] == "measured_from_exp11"]
        if measured.empty:
            raise ValueError("exp14_convergence_robustness_results.csv has no tau_source=='measured_from_exp11' row.")
        tau = float(measured.iloc[0]["tau"])

        def _report(column: str, macro_stub: str) -> None:
            neg = pos[pos[column] < 0.0]
            survives = (neg["Delta"].abs() >= tau) & (neg["Delta_prime"].abs() >= tau)
            kept = neg[survives]
            mc.add(
                f"numExpNineNegative{macro_stub}RowsTotal", int(neg.shape[0]),
                f"exp10_identifiability_check_results.csv, count of (dataset,alpha>0) rows with {column}<0",
            )
            mc.add(
                f"numExpNineNegative{macro_stub}RowsSurviving", int(kept.shape[0]),
                f"exp10_identifiability_check_results.csv rows with {column}<0, count surviving the exp14-measured filter (exp14_convergence_robustness_results.csv, tau_source=measured_from_exp11, tau={tau:.6g}): |Delta|>=tau AND |Delta_prime|>=tau, same rule as exp14_convergence_robustness.py::_kept_rows",
            )
            dataset_list = sorted(kept["dataset"].unique())
            escaped = ", ".join(str(d).replace("_", "\\_") for d in dataset_list)
            mc.add(
                f"negative{macro_stub}SurvivingDatasetsList", escaped,
                f"exp10_identifiability_check_results.csv, datasets with a {column}<0 row surviving the exp14-measured tau filter (LaTeX-escaped list, empty string if none survive)",
            )

        _report("Delta", "Delta")
        _report("Delta_prime", "DeltaPrime")

    mc.try_block("exp10 identifiability: negative Delta/Delta_prime rows surviving the exp14-measured convergence filter (review round 2, point 3)", block)


def _add_exp13_neighbor_survival_numbers(mc: MacroCollector, tables_dir: Path) -> None:
    """documentation/2026-09-17_zadani_exp13_preziti_sousedu.md - the
    plain-language "how many neighbors survive" table
    (results/tables/[<mode>/]exp13_neighbor_survival.csv, written by
    `src/experiments/exp13_neighbor_survival.py::_write_survival_table`):
    median neighbors kept (out of k) for the alpha=0/alpha=1/alpha_best
    policies in each of the two regime panels, plus the k used - the numbers
    behind the article's plain-language framing of the alpha-vs-metric-cost
    trade-off."""
    from src.experiments.exp13_neighbor_survival import (
        PANEL_LOW_RATIO,
        PANEL_REST,
        POLICY_LABEL_BEST,
        POLICY_LABEL_PRED,
        policy_label,
    )

    def block() -> None:
        df = _read_csv_required(tables_dir / "exp13_neighbor_survival.csv")

        def _row(panel: str, policy: str):
            r = df[(df["panel"] == panel) & (df["policy"] == policy)]
            return r.iloc[0] if not r.empty else None

        k_values = df["k"].dropna().unique()
        if len(k_values) != 1:
            raise ValueError(f"exp13_neighbor_survival.csv is expected to use a single k, got {sorted(k_values)}.")
        mc.add("numNeighborSurvivalK", int(k_values[0]), "exp13_neighbor_survival.csv, k (exp13_neighbor_survival.k_values[0])")

        panels = {"LowRatio": PANEL_LOW_RATIO, "RestRatio": PANEL_REST}
        policies = {"AlphaZero": policy_label(0.0), "AlphaOne": policy_label(1.0), "AlphaPred": POLICY_LABEL_PRED, "AlphaBest": POLICY_LABEL_BEST}
        for panel_word, panel in panels.items():
            for policy_word, policy in policies.items():
                r = _row(panel, policy)
                if r is None:
                    continue
                mc.add_if_finite(
                    f"medNeighborSurvivalKept{policy_word}{panel_word}", r["neighbors_kept"],
                    f"exp13_neighbor_survival.csv, neighbors_kept, panel={panel}, policy={policy}, median over datasets", sig=3,
                )
                mc.add(f"numNeighborSurvivalDatasets{policy_word}{panel_word}", int(r["n_datasets"]), f"exp13_neighbor_survival.csv, n_datasets, panel={panel}, policy={policy}")

    mc.try_block("exp13 neighbor survival: median neighbors kept by alpha policy and regime panel", block)


def _add_exp14_convergence_robustness_numbers(mc: MacroCollector, data_dir: Path) -> None:
    """documentation/2026-09-17_zadani_exp14_robustnost_konvergence.md - the
    row measured from exp11_convergence_check_results.csv
    (`tau_source=='measured_from_exp11'` in
    results/data/[<mode>/]exp14_convergence_robustness_results.csv), i.e. the
    answer to "do the exp10 claims survive filtering out rows below the
    convergence residual". Macro names use a 'robust' prefix and are
    DELIBERATELY DIFFERENT from the original 'ExpNine...' macros of
    `_add_exp10_identifiability_numbers` (those are never overwritten - see
    the E14 task description)."""

    def block() -> None:
        df = _read_csv_required(data_dir / "exp14_convergence_robustness_results.csv")
        measured = df[df["tau_source"] == "measured_from_exp11"]
        if measured.empty:
            raise ValueError("exp14_convergence_robustness_results.csv has no tau_source=='measured_from_exp11' row.")
        r = measured.iloc[0]

        mc.add_if_finite("robustTauMeasured", r["tau"], "exp14_convergence_robustness_results.csv, tau_source=measured_from_exp11, threshold measured from exp11_convergence_check_results.csv (median over datasets of the max relative stress decline baseline->tight)", sig=2)
        if float(r["n_rows_total"]) > 0:
            mc.add_if_finite("robustFracRowsKept", float(r["n_rows_kept"]) / float(r["n_rows_total"]), "exp14_convergence_robustness_results.csv, tau_source=measured_from_exp11, n_rows_kept/n_rows_total (fraction of exp10 alpha>0 rows surviving the measured threshold)", sig=2)
        mc.add("robustNumDatasetsKept", int(r["n_datasets_kept"]), "exp14_convergence_robustness_results.csv, tau_source=measured_from_exp11, n_datasets_kept")
        mc.add_if_finite("robustMedTightVOne", r["med_tight_v1"], "exp14_convergence_robustness_results.csv, tau_source=measured_from_exp11, med_tight_v1 (compare to medExpNineTightVOne at tau=0)", sig=2)
        mc.add("robustNumNontrivialAlphaOne", int(r["n_nontrivial_alpha_one"]), "exp14_convergence_robustness_results.csv, tau_source=measured_from_exp11, n_nontrivial_alpha_one (compare to numExpNineNontrivialAlphaOne at tau=0)")
        mc.add_if_finite("robustSpearmanCeilingVsGainRho", r["spearman_ceiling_vs_gain_rho"], "exp14_convergence_robustness_results.csv, tau_source=measured_from_exp11, spearman_ceiling_vs_gain_rho (compare to spearmanExpNineCeilingVsGainRho at tau=0)", sig=3)
        mc.add_if_finite("robustSpearmanCeilingVsGainP", r["spearman_ceiling_vs_gain_p"], "exp14_convergence_robustness_results.csv, tau_source=measured_from_exp11, spearman_ceiling_vs_gain_p (exp14's OWN permutation seed, not comparable bit-for-bit to spearmanExpNineCeilingVsGainP)", sig=2)

    mc.try_block("exp14 convergence robustness: numbers at the exp11-measured threshold", block)


def _add_exp15_discovery_task_numbers(mc: MacroCollector, data_dir: Path, tables_dir: Path) -> None:
    """E15 downstream discovery task (author request 2026-09-18, motivated
    by a DAMI editor objection that no task in the manuscript shows a
    better map leading to a better FINDING). Dataset inclusion/exclusion
    counts come from results/data/[<mode>/]exp15_discovery_task_results.csv
    ('note' column; reasons defined in src/sammon/discovery_task.py::NOTE_*),
    per-method n_correct/error_rate from
    results/tables/[<mode>/]exp15_discovery_task_summary.csv, and the
    Holm-adjusted McNemar comparisons of exp15_discovery_task_stats.
    reference_method vs every other report.main_methods baseline from
    results/tables/[<mode>/]exp15_discovery_task_pairwise.csv
    (src/experiments/exp15_discovery_task_stats.py). Method list comes from
    report.main_methods (config), never hardcoded.

    Honesty check (author 2026-09-18, after two 'overclaiming' referee
    catches): errRateExpFifteen* for the stress family macros stays at
    32-38%, never near zero, and the pairwise family macros show alpha_pred
    is statistically indistinguishable from the OTHER stress-family
    variants (sammon_alpha0_smacof/_smacof/_auto/2) - only the
    stress-vs-neighbor-family comparisons (t-SNE/UMAP/PaCMAP/TriMap/PHATE)
    reach significance. The win belongs to the stress family, not to the
    alpha_pred rule specifically."""
    from src.sammon.discovery_task import (
        DISCOVERY_QUESTIONS,
        NOTE_EXCESSIVE_CLASSES,
        NOTE_INSUFFICIENT_CLASSES,
        QUESTION_MOST_DISPERSED,
        QUESTION_NEAREST_PAIR,
    )

    question_word = {QUESTION_NEAREST_PAIR: "NearestPair", QUESTION_MOST_DISPERSED: "MostDispersed"}
    exp_cfg = load_experiments_config()
    main_methods: list[str] = exp_cfg["report"]["main_methods"]
    reference_method = exp_cfg["exp15_discovery_task_stats"]["reference_method"]

    def block_counts() -> None:
        df = _read_csv_required(data_dir / "exp15_discovery_task_results.csv")
        notes = df.groupby("dataset")["note"].first()
        mc.add("numExpFifteenDatasetsTotal", int(notes.shape[0]), "exp15_discovery_task_results.csv, number of distinct dataset values")
        mc.add("numExpFifteenDatasetsIncluded", int(notes.isna().sum()), "exp15_discovery_task_results.csv, datasets with note is empty (both questions defined)")
        mc.add("numExpFifteenDatasetsExcluded", int(notes.notna().sum()), "exp15_discovery_task_results.csv, datasets with a non-empty note")
        mc.add(
            "numExpFifteenDatasetsExcludedInsufficientClasses", int((notes == NOTE_INSUFFICIENT_CLASSES).sum()),
            f"exp15_discovery_task_results.csv, note=='{NOTE_INSUFFICIENT_CLASSES}' (src/sammon/discovery_task.py::NOTE_INSUFFICIENT_CLASSES)",
        )
        mc.add(
            "numExpFifteenDatasetsExcludedExcessiveClasses", int((notes == NOTE_EXCESSIVE_CLASSES).sum()),
            f"exp15_discovery_task_results.csv, note=='{NOTE_EXCESSIVE_CLASSES}' (src/sammon/discovery_task.py::NOTE_EXCESSIVE_CLASSES)",
        )

    mc.try_block("exp15 discovery task: dataset inclusion/exclusion counts", block_counts)

    def block_summary() -> None:
        summary = _read_csv_required(tables_dir / "exp15_discovery_task_summary.csv")
        for question in DISCOVERY_QUESTIONS:
            qword = question_word[question]
            for method in main_methods:
                row = summary[(summary["question"] == question) & (summary["method"] == method)]
                if row.empty:
                    continue
                r = row.iloc[0]
                suffix = _suffix(method)
                n_correct = int(round(float(r["n_datasets"]) * (1.0 - float(r["error_rate"]))))
                mc.add(
                    f"numExpFifteenCorrect{suffix}{qword}", n_correct,
                    f"exp15_discovery_task_summary.csv, question={question}, method={method}, round(n_datasets*(1-error_rate))",
                )
                mc.add_if_finite(
                    f"errRateExpFifteen{suffix}{qword}", r["error_rate"],
                    f"exp15_discovery_task_summary.csv, question={question}, method={method}, error_rate", sig=2,
                )

    mc.try_block("exp15 discovery task: per-method n_correct/error_rate (report.main_methods)", block_summary)

    def block_pairwise() -> None:
        pairwise = _read_csv_required(tables_dir / "exp15_discovery_task_pairwise.csv")
        ref_suffix = _suffix(reference_method)
        for question in DISCOVERY_QUESTIONS:
            qword = question_word[question]
            fam = pairwise[pairwise["question"] == question]
            if fam.empty:
                continue
            mc.add(
                f"numExpFifteenBaselinesTotal{qword}", int(len(fam)),
                f"exp15_discovery_task_pairwise.csv, question={question}, number of baseline rows (method_a=={reference_method} vs every other report.main_methods method)",
            )
            mc.add(
                f"numExpFifteenSignificantMcnemarHolm{qword}", int(fam["reject_mcnemar_holm"].fillna(False).sum()),
                f"exp15_discovery_task_pairwise.csv, question={question}, count of reject_mcnemar_holm==True (alpha=exp15_discovery_task_stats.alpha)",
            )
            for baseline_key, baseline_method in (("TsneAuto", "tsne_auto"), ("UmapAuto", "umap_auto")):
                row = fam[fam["method_b"] == baseline_method]
                if row.empty:
                    continue
                r = row.iloc[0]
                mc.add_if_finite(
                    f"pExpFifteenMcnemarHolm{ref_suffix}Vs{baseline_key}{qword}", r["p_mcnemar_holm"],
                    f"exp15_discovery_task_pairwise.csv, question={question}, method_a={reference_method}, method_b={baseline_method}, p_mcnemar_holm", sig=2,
                )

    mc.try_block("exp15 discovery task: Holm-adjusted McNemar p-values (alpha_pred vs t-SNE/UMAP) and significant-baseline counts", block_pairwise)


def build_numbers_tex(mode: str = "full") -> tuple[str, int]:
    """Builds the content of numbers.tex for a given mode. Returns (text, number_of_generated_macros)."""
    logger = get_logger(MODULE_NAME, mode=mode)
    exp_cfg = load_experiments_config()
    neighbor_methods: list[str] = exp_cfg["report"]["neighbor_methods"]

    data_dir = get_mode_path("results_data_dir", mode)
    tables_dir = get_mode_path("results_tables_dir", mode)
    figures_dir = get_mode_path("results_figures_dir", mode)
    logger.info("export_numbers: mode=%s, data=%s, tables=%s, figures=%s", mode, data_dir, tables_dir, figures_dir)

    mc = MacroCollector(logger)
    groups: list[Callable[[], None]] = [
        lambda: _add_counts(mc, data_dir),
        lambda: _add_medians(mc, data_dir),
        lambda: _add_win_counts(mc, data_dir, neighbor_methods),
        lambda: _add_alpha_zero_vs_one(mc, data_dir),
        lambda: _add_alpha_pred_distribution(mc, data_dir, mode),
        lambda: _add_alpha_pred_on_graphs(mc, data_dir),
        lambda: _add_high_regime_no_gain(mc, data_dir),
        lambda: _add_grid_extension(mc, data_dir),
        lambda: _add_stats_kendall_cd(mc, data_dir),
        lambda: _add_exp3_group_numbers(mc, data_dir, tables_dir),
        lambda: _add_pareto_numbers(mc, tables_dir, data_dir),
        lambda: _add_speedups(mc, data_dir),
        lambda: _add_k9_relative_changes(mc, data_dir),
        lambda: _add_exp4_dtsne_baseline_numbers(mc, data_dir),
        lambda: _add_exp4_neighbor_primary_numbers(mc, data_dir),
        lambda: _add_exp5_factorial(mc, tables_dir, data_dir),
        lambda: _add_exp6_numbers(mc, tables_dir),
        lambda: _add_alpha_pred_numbers(mc, data_dir, tables_dir),
        lambda: _add_alpha_pred_rule_coefficients(mc, data_dir),
        lambda: _add_exp7_numbers(mc, tables_dir, data_dir),
        lambda: _add_regime_map_spearman(mc, figures_dir),
        lambda: _add_regime_map_property_spearman(mc, data_dir, tables_dir),
        lambda: _add_neighborhood_problem_numbers(mc, figures_dir),
        lambda: _add_regime_stratified(mc, tables_dir),
        lambda: _add_graph_nn_ratio_range(mc, data_dir),
        lambda: _add_umap_grid_edge(mc, data_dir),
        lambda: _add_multiscale_vs_alpha_auto(mc, data_dir),
        lambda: _add_exp1_cluster_geometry_dataset_count(mc, data_dir),
        lambda: _add_exp2_solver_scaling_config_numbers(mc),
        lambda: _add_q1_holdout_numbers(mc, data_dir, tables_dir),
        lambda: _add_exp8_prop2_numbers(mc, data_dir),
        lambda: _add_exp8_prop2_per_dataset_correlation(mc, data_dir),
        lambda: _add_exp9_metric_fidelity_numbers(mc, data_dir, tables_dir),
        lambda: _add_exp10_identifiability_numbers(mc, data_dir, mode),
        lambda: _add_exp10_negative_delta_survival(mc, data_dir),
        lambda: _add_exp13_neighbor_survival_numbers(mc, tables_dir),
        lambda: _add_exp14_convergence_robustness_numbers(mc, data_dir),
        lambda: _add_exp15_discovery_task_numbers(mc, data_dir, tables_dir),
    ]
    for fn in groups:
        fn()

    header = (
        "% AUTO-GENERATED: src/experiments/export_numbers.py - DO NOT EDIT BY HAND.\n"
        f"% mode: {mode}; data: {data_dir}; tables: {tables_dir}\n"
        "% Every number in the article MUST be cited via one of these macros\n"
        "% (never transcribe a value by hand - see CLAUDE.md 'no number without a source').\n\n"
    )
    text = header + "".join(mc.lines)
    return text, len(mc.lines)


def main(mode: str | None = None) -> Path:
    """Writes numbers.tex: 'full' -> clanek/generated/, otherwise results/tables/<mode>/."""
    if mode is None:
        mode = parse_mode_args("K12a: number macros for the article (numbers.tex).")
    logger = get_logger(MODULE_NAME, mode=mode)
    text, n_macros = build_numbers_tex(mode)

    out_path = get_generated_dir(mode) / "numbers.tex"
    out_path.write_text(text, encoding="utf-8")
    logger.info("Written: %s (%d macros).", out_path, n_macros)
    print(f"export_numbers ({mode}): {out_path} ({n_macros} macros).")
    return out_path


if __name__ == "__main__":
    main()
