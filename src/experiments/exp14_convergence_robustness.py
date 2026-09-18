# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-17
# License: see the LICENSE file in the repository root
"""
E14: robustness of the exp10_identifiability_check claims to under-converged
embeddings (documentation/2026-09-17_zadani_exp14_robustnost_konvergence.md).

exp11_convergence_check found that the exp6_alpha_curves embeddings behind
ALL of exp10_identifiability_check are not fully converged w.r.t. the
precision that |Delta|/|Delta_prime| need: a much tighter SMACOF stopping
criterion still reduces stress by a median 0.6%, which is LARGER than
|Delta| itself on 11 of 12 comparable rows. But those 13 rows were a biased
sample (picked because both_gaps_nonneg was False, i.e. |Delta| smallest);
across all 384 alpha>0 rows of exp10, median |Delta| is 0.036 (~6x that
residual). This experiment asks: do exp10's headline numbers change if rows
with |Delta| below a convergence-driven threshold `tau` are dropped?

NO new embedding or solver run - pure filtering/recomputation over the
ALREADY COMPLETED results/data/exp10_identifiability_check_results.csv and
results/data/exp11_convergence_check_results.csv. Both are ALWAYS read from
their FULL production location, regardless of exp14's own --quick/--smoke/--full
mode (exactly the exp11_convergence_check.py convention for exp6/exp10 - see
that module's docstring): exp14's own mode only restricts its OWN threshold
list and permutation count (`exp14_convergence_robustness.smoke.*` in
config_experiments.yaml), so --smoke still exercises the full computation
(including the fail-loud tau=0 reproduction check) in a few seconds.

For each threshold `tau` (config `exp14_convergence_robustness.thresholds`,
plus one threshold MEASURED from exp11_convergence_check_results.csv, not
chosen by hand - see `_measure_tau_from_exp11`), rows of exp10 with
status=='ok' and alpha>0 are kept iff BOTH |Delta|>=tau and
|Delta_prime|>=tau, and the following exp10 claims are recomputed over the
kept rows only:
  - the count of kept rows/datasets,
  - the fraction of kept rows with both_gaps_nonneg==True,
  - the median tight_v1/tight_sandwich over kept rows with bound_nontrivial==True,
  - the count of datasets with bound_nontrivial==True at alpha==1 (kept rows only),
  - Spearman(c_alpha*gamma_tilde_Y0, G_auc_oracle) + permutation p over the
    DATASETS that still have >=1 kept row (values read from that dataset's
    alpha==1 row of the FULL, unfiltered exp10 CSV - same convention as
    `exp10_identifiability_stats.py::dataset_level_frame`/`analyze_h1_spearman`,
    which this module imports directly rather than re-implementing, so the
    numbers are comparable).

tau=0 keeps every alpha>0 row (|Delta|>=0 is vacuously true) and therefore
MUST exactly reproduce exp10's own numbers (`export_numbers.py`'s
medExpNineTightVOne/medExpNineTightSandwich/numExpNineNontrivialAlphaOne and
exp10_identifiability_stats.csv's H1_spearman/cGammaTildeZero_vs_G_auc_oracle
rho) - checked fail-loud with a RELATIVE tolerance
(`exp14_convergence_robustness.baseline_match_tol`) against those
INDEPENDENTLY computed reference values, see `_check_reproduces_e10` (the
permutation p-value is NOT part of this check: it is inherently seed-
dependent and exp14 uses its OWN seed, not exp10_identifiability_stats.seed).

Output: results/data/[<mode>/]exp14_convergence_robustness_results.csv (one
row per threshold, columns per the module docstring's `COLUMN_KEYS`) +
results/tables/[<mode>/]exp14_convergence_robustness.csv/.tex. No checkpoint/
resume (like exp10_identifiability_stats.py, this recomputes from scratch
every time - the whole run takes seconds, not hours).

Run: venv\\python.exe -m src.experiments.exp14_convergence_robustness [--quick|--full|--smoke]
or: src\\run_exp14_convergence_robustness.bat [quick|full|smoke]
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.common.checkpoint import results_csv_path
from src.common.config import ensure_dir, get_mode_path, get_tables_dir
from src.common.logging_utils import get_logger
from src.experiments.config_experiments import resolve_experiment_config
from src.experiments.exp10_identifiability_stats import dataset_level_frame
from src.experiments.exp_common import keep_system_awake, parse_mode_args, resolve_experiment_name
from src.experiments.report_tables import write_booktabs_tex
from src.experiments.stats import spearman_permutation_test

BASE_EXPERIMENT_NAME = "exp14_convergence_robustness"
EXP10_BASE_NAME = "exp10_identifiability_check"
EXP11_BASE_NAME = "exp11_convergence_check"

# Canonical CSV schema (documentation/2026-09-17_zadani_exp14_robustnost_konvergence.md,
# section "Vystup").
COLUMN_KEYS = [
    "experiment", "tau", "tau_source", "n_rows_kept", "n_rows_total", "n_datasets_kept",
    "frac_both_gaps_nonneg", "med_tight_v1", "med_tight_sandwich",
    "n_nontrivial_alpha_one", "spearman_ceiling_vs_gain_rho",
    "spearman_ceiling_vs_gain_p", "status", "error",
]

TAU_SOURCE_CONFIG = "config"
TAU_SOURCE_MEASURED = "measured_from_exp11"

# Fields checked (RELATIVE difference) by `_check_reproduces_e10` at tau=0 -
# spearman p is deliberately excluded, see the module docstring.
_REPRO_FIELDS = [
    "frac_both_gaps_nonneg", "med_tight_v1", "med_tight_sandwich",
    "n_nontrivial_alpha_one", "spearman_ceiling_vs_gain_rho",
]


def _load_full_ok(base_name: str) -> pd.DataFrame:
    """Loads results/data/<base_name>_results.csv (ALWAYS the FULL production
    file - `resolve_experiment_name(base_name, "full")` is `base_name`
    unchanged, see `src.experiments.exp_common.resolve_experiment_name`),
    filtered to status=='ok'. Fail-loud if missing/empty."""
    path = results_csv_path(base_name)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing input: {path}\nRun first: venv\\python.exe -m src.experiments.{base_name} --full."
        )
    df = pd.read_csv(path)
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        raise ValueError(f"{path} has no rows with status 'ok'.")
    return ok


def _measure_tau_from_exp11(e11_ok: pd.DataFrame) -> float:
    """Measures a convergence-driven threshold from
    results/data/exp11_convergence_check_results.csv (NOT chosen by hand -
    documentation/2026-09-17_zadani_exp14_robustnost_konvergence.md, section
    "Odvozeni doporuceneho prahu z E11"): for each (dataset, alpha) task that
    has BOTH a 'baseline' and a 'tight' row, the relative stress decline

        (sigma0_Y0[baseline] - sigma0_Y0[tight]) / |sigma0_Y0[baseline]|
        (sigmaA_Ya[baseline] - sigmaA_Ya[tight]) / |sigmaA_Ya[baseline]|

    is computed, and the row-wise max of the two is taken (both quantities
    measure the same phenomenon - residual optimization gain from tightening
    the stopping criterion - from a different one of the two configurations
    compared by Delta). sigma0_Y0[tight] does not depend on the task's own
    alpha (the alpha=0 objective is refit identically regardless of which
    alpha the row's Delta concerns) and is, up to solver noise, the same
    value for every alpha row of one dataset - so rows are first reduced to
    ONE number per dataset via the MEDIAN over that dataset's own (alpha)
    rows (same 'median per unit, then over units' convention as
    `src.experiments.report_tables.method_summary`), and the returned
    threshold is the MEDIAN of that per-dataset number over datasets.

    A dataset without at least one (alpha) task carrying both a 'baseline'
    and a 'tight' row (e.g. a still-incomplete exp11 checkpoint) is silently
    excluded - not fabricated - see the two `.dropna(subset=...)` calls
    below."""
    pivot0 = e11_ok.pivot_table(index=["dataset", "alpha"], columns="variant", values="sigma0_Y0", aggfunc="first")
    pivotA = e11_ok.pivot_table(index=["dataset", "alpha"], columns="variant", values="sigmaA_Ya", aggfunc="first")
    for pivot, label in ((pivot0, "sigma0_Y0"), (pivotA, "sigmaA_Ya")):
        if "baseline" not in pivot.columns or "tight" not in pivot.columns:
            raise ValueError(
                f"{results_csv_path(EXP11_BASE_NAME)}: no row has variant=='baseline' or variant=='tight' for "
                f"column '{label}' - cannot measure the E14 threshold."
            )
    both0 = pivot0.dropna(subset=["baseline", "tight"])
    bothA = pivotA.dropna(subset=["baseline", "tight"])
    if both0.empty or bothA.empty:
        raise ValueError(
            f"{results_csv_path(EXP11_BASE_NAME)}: no (dataset, alpha) task has both a 'baseline' and a 'tight' "
            "row for both sigma0_Y0 and sigmaA_Ya - cannot measure the E14 threshold."
        )
    decl0 = (both0["baseline"] - both0["tight"]) / both0["baseline"].abs()
    declA = (bothA["baseline"] - bothA["tight"]) / bothA["baseline"].abs()
    combined = pd.concat([decl0.rename("decl0"), declA.rename("declA")], axis=1).dropna()
    if combined.empty:
        raise ValueError(f"{results_csv_path(EXP11_BASE_NAME)}: no (dataset, alpha) task has both declines defined.")
    row_max = combined[["decl0", "declA"]].max(axis=1)
    per_dataset = row_max.groupby(level=0).median()
    if per_dataset.empty:
        raise ValueError(f"{results_csv_path(EXP11_BASE_NAME)}: no dataset available to measure the E14 threshold.")
    return float(per_dataset.median())


def _kept_rows(pos: pd.DataFrame, tau: float) -> pd.DataFrame:
    """exp10 alpha>0/status=='ok' rows (`pos`) with BOTH |Delta|>=tau AND
    |Delta_prime|>=tau (module docstring)."""
    return pos[(pos["Delta"].abs() >= tau) & (pos["Delta_prime"].abs() >= tau)]


def _compute_threshold_row(
    tau: float, tau_source: str, pos: pd.DataFrame, ds_all: pd.DataFrame, n_rows_total: int, n_perm: int, seed: int,
) -> dict[str, Any]:
    """One output row: filters `pos` at `tau` and recomputes the exp10 claims
    listed in the module docstring over the kept rows/datasets."""
    kept = _kept_rows(pos, tau)
    n_rows_kept = int(kept.shape[0])
    n_datasets_kept = int(kept["dataset"].nunique())
    frac_both_gaps_nonneg = float(kept["both_gaps_nonneg"].mean()) if n_rows_kept > 0 else float("nan")

    nontrivial = kept[kept["bound_nontrivial"] == True]  # noqa: E712
    med_tight_v1 = float(nontrivial["tight_v1"].median()) if not nontrivial.empty else float("nan")
    med_tight_sandwich = float(nontrivial["tight_sandwich"].median()) if not nontrivial.empty else float("nan")

    kept_a1 = kept[np.isclose(kept["alpha"], 1.0)]
    n_nontrivial_alpha_one = int(kept_a1["bound_nontrivial"].sum()) if not kept_a1.empty else 0

    kept_datasets = pd.Index(kept["dataset"].unique())
    ds_kept = ds_all.loc[ds_all.index.intersection(kept_datasets)]
    if ds_kept.shape[0] >= 3:
        x = (ds_kept["c_1"] * ds_kept["gamma_tilde_Y0"]).to_numpy(dtype=np.float64)
        y = ds_kept["G_auc_oracle"].to_numpy(dtype=np.float64)
        spear = spearman_permutation_test(x, y, n_perm=n_perm, seed=seed, alternative="two-sided")
        rho, pvalue = spear["rho"], spear["pvalue"]
    else:
        rho, pvalue = float("nan"), float("nan")

    return {
        "tau": tau, "tau_source": tau_source,
        "n_rows_kept": n_rows_kept, "n_rows_total": n_rows_total, "n_datasets_kept": n_datasets_kept,
        "frac_both_gaps_nonneg": frac_both_gaps_nonneg, "med_tight_v1": med_tight_v1, "med_tight_sandwich": med_tight_sandwich,
        "n_nontrivial_alpha_one": n_nontrivial_alpha_one,
        "spearman_ceiling_vs_gain_rho": rho, "spearman_ceiling_vs_gain_p": pvalue,
        "status": "ok", "error": "",
    }


def _reference_values_from_e10(e10_ok_all: pd.DataFrame) -> dict[str, float]:
    """INDEPENDENTLY computed reference values for the tau=0 reproduction
    check: `frac_both_gaps_nonneg`/`med_tight_v1`/`med_tight_sandwich`/
    `n_nontrivial_alpha_one` are recomputed here directly from the FULL,
    UNFILTERED exp10 CSV with the exact same formulas as
    `src/experiments/export_numbers.py::_add_exp10_identifiability_numbers`
    (medExpNineTightVOne/medExpNineTightSandwich/numExpNineNontrivialAlphaOne);
    `spearman_ceiling_vs_gain_rho` is read from the ALREADY MATERIALIZED
    results/data/exp10_identifiability_stats.csv (a genuinely independent
    computation, run by a different script) rather than recomputed here."""
    pos_all = e10_ok_all[e10_ok_all["alpha"] > 0.0]
    frac_both_gaps_nonneg = float(pos_all["both_gaps_nonneg"].mean())
    nontrivial_all = pos_all[pos_all["bound_nontrivial"] == True]  # noqa: E712
    med_tight_v1 = float(nontrivial_all["tight_v1"].median())
    med_tight_sandwich = float(nontrivial_all["tight_sandwich"].median())
    a1 = e10_ok_all[np.isclose(e10_ok_all["alpha"], 1.0)]
    if a1.empty:
        raise ValueError("exp10_identifiability_check_results.csv has no alpha=1.0 rows - cannot build the E14 reference.")
    n_nontrivial_alpha_one = int(a1["bound_nontrivial"].sum())

    stats_path = get_mode_path("results_data_dir", "full") / "exp10_identifiability_stats.csv"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Missing input: {stats_path}\nRun first: venv\\python.exe -m src.experiments.exp10_identifiability_stats --full."
        )
    stats = pd.read_csv(stats_path)
    row = stats[(stats["analysis"] == "H1_spearman") & (stats["label"] == "cGammaTildeZero_vs_G_auc_oracle")]
    if row.empty:
        raise ValueError(f"{stats_path} has no H1_spearman/cGammaTildeZero_vs_G_auc_oracle row.")
    spearman_rho = float(row.iloc[0]["value"])

    return {
        "frac_both_gaps_nonneg": frac_both_gaps_nonneg,
        "med_tight_v1": med_tight_v1,
        "med_tight_sandwich": med_tight_sandwich,
        "n_nontrivial_alpha_one": float(n_nontrivial_alpha_one),
        "spearman_ceiling_vs_gain_rho": spearman_rho,
    }


def _check_reproduces_e10(tau0_row: dict[str, Any], reference: dict[str, float], tol: float) -> None:
    """Fail-loud check that the tau=0 row (which keeps every alpha>0 row,
    since |Delta|>=0 is vacuously true) reproduces the INDEPENDENTLY computed
    `reference` values within RELATIVE `tol` (also used as an absolute
    floor) - see the module docstring's "PAST" warning: an absolute bound
    chosen for one field's magnitude (e.g. tight_v1 medians ~5e-3) can be
    wrong by orders of magnitude for another. A mismatch means exp14 is
    computing something DIFFERENT from exp10 (a bug in exp14), not a
    legitimate finding."""
    mismatches: list[str] = []
    for field in _REPRO_FIELDS:
        a = float(tau0_row[field])
        b = float(reference[field])
        if not math.isclose(a, b, rel_tol=tol, abs_tol=tol):
            scale = max(abs(a), abs(b), 1.0)
            mismatches.append(
                f"{field}: exp14(tau=0)={a!r} vs reference={b!r} "
                f"(|diff|={abs(a - b):.3e}, relative={abs(a - b) / scale:.3e} > rel_tol={tol:.3e})"
            )
    if mismatches:
        raise ValueError(
            "exp14_convergence_robustness's tau=0 row does NOT reproduce the independently computed exp10 "
            f"reference values within relative baseline_match_tol={tol:.3e} - exp14 is computing something "
            "different from exp10 (bug), not a legitimate finding:\n  " + "\n  ".join(mismatches)
        )


def build_results(cfg: dict[str, Any], logger) -> pd.DataFrame:
    """Loads the FULL exp10/exp11 CSVs, measures the exp11-derived threshold,
    computes one row per threshold (config list + measured), and runs the
    fail-loud tau=0 reproduction check. Returns the results DataFrame
    (columns = `COLUMN_KEYS` minus 'experiment', added by the caller)."""
    thresholds = [float(t) for t in cfg["thresholds"]]
    n_perm = int(cfg["n_permutations"])
    seed = int(cfg["seed"])
    baseline_match_tol = float(cfg["baseline_match_tol"])

    e10_ok = _load_full_ok(EXP10_BASE_NAME)
    e11_ok = _load_full_ok(EXP11_BASE_NAME)

    pos = e10_ok[e10_ok["alpha"] > 0.0].copy()
    if pos.empty:
        raise ValueError(f"{results_csv_path(EXP10_BASE_NAME)} has no status=='ok' rows with alpha>0.")
    n_rows_total = int(pos.shape[0])
    ds_all = dataset_level_frame(e10_ok, reference_alpha=1.0)

    measured_tau = _measure_tau_from_exp11(e11_ok)
    logger.info("exp14: threshold measured from exp11_convergence_check_results.csv: tau=%.4g.", measured_tau)

    rows: list[dict[str, Any]] = []
    tau0_row: dict[str, Any] | None = None
    for tau in thresholds:
        row = _compute_threshold_row(tau, TAU_SOURCE_CONFIG, pos, ds_all, n_rows_total, n_perm, seed)
        rows.append(row)
        if tau == 0.0:
            tau0_row = row
    measured_row = _compute_threshold_row(measured_tau, TAU_SOURCE_MEASURED, pos, ds_all, n_rows_total, n_perm, seed)
    rows.append(measured_row)

    if tau0_row is None:
        raise ValueError(
            "exp14_convergence_robustness.thresholds does not contain 0.0 - the tau=0 reproduction check "
            "(documentation/2026-09-17_zadani_exp14_robustnost_konvergence.md) cannot run."
        )
    reference = _reference_values_from_e10(e10_ok)
    _check_reproduces_e10(tau0_row, reference, baseline_match_tol)
    logger.info("exp14: tau=0 row reproduces the independent exp10 reference values within rel_tol=%.1e.", baseline_match_tol)

    return pd.DataFrame(rows, columns=[c for c in COLUMN_KEYS if c != "experiment"])


def _table_caption(df: pd.DataFrame) -> str:
    """Caption text; the values that are constant across threshold rows are
    stated here instead of repeating them in a column (the table is too wide
    for the supplement text block otherwise). Read from the data, never typed.
    """
    n_total = int(df["n_rows_total"].iloc[0])
    p_perm = float(df["spearman_ceiling_vs_gain_p"].iloc[0])
    return (
        "E14: robustness of the exp10 identifiability claims to the exp11-measured "
        "SMACOF convergence residual (rows filtered to |Delta|>=tau and "
        "|Delta\\_prime|>=tau; tau=0 reproduces exp10 exactly). Every row is taken "
        f"out of {n_total} (dataset, alpha>0) rows in total, and the permutation "
        f"p-value of the ceiling-vs-gain correlation sits at its floor {p_perm:.0e} "
        "on every row."
    )


def write_table(df: pd.DataFrame, mode: str, logger) -> Path:
    tables_dir = get_tables_dir(mode)
    out_csv = tables_dir / f"{BASE_EXPERIMENT_NAME}.csv"
    df.to_csv(out_csv, index=False)
    write_booktabs_tex(
        df, tables_dir / f"{BASE_EXPERIMENT_NAME}.tex",
        caption=_table_caption(df),
        label=f"tab:{BASE_EXPERIMENT_NAME}",
        comment_lines=[
            "source: results/data/exp10_identifiability_check_results.csv (ALWAYS the full production file, see the module docstring) + "
            "results/data/exp11_convergence_check_results.csv (threshold measurement, tau_source=='measured_from_exp11')",
            "tau=0 keeps every alpha>0 row and reproduces exp10_identifiability_check_results.csv/exp10_identifiability_stats.csv exactly (fail-loud check in the script)",
        ],
        # 'experiment'/'status'/'error' are pure run-status redundancy here
        # (single value repeated on every threshold row, widest columns in
        # the table) - dropped from the LaTeX table only if truly constant
        # (write_booktabs_tex's drop_constant_cols); they stay in the CSV,
        # and 'error' would NOT be dropped if any row actually failed.
        # Also dropped: the two columns that carry the SAME value on every
        # threshold row (total row count, permutation p-value at its floor).
        # They are the widest headers in a table that does not fit the
        # supplement text width, so they move into the caption instead.
        drop_constant_cols=["experiment", "status", "error",
                            "n_rows_total", "spearman_ceiling_vs_gain_p"],
    )
    logger.info("Table written: %s (%d rows).", out_csv, df.shape[0])
    return out_csv


def main() -> None:
    mode = parse_mode_args(
        "E14: robustness of the exp10_identifiability_check claims to filtering out rows below an "
        "exp11-measured SMACOF convergence residual (documentation/2026-09-17_zadani_exp14_robustnost_konvergence.md)."
    )
    logger = get_logger(BASE_EXPERIMENT_NAME, mode=mode)
    with keep_system_awake():
        cfg = resolve_experiment_config(BASE_EXPERIMENT_NAME, mode)
        EXPERIMENT_NAME = resolve_experiment_name(BASE_EXPERIMENT_NAME, mode)

        results = build_results(cfg, logger)
        results.insert(0, "experiment", EXPERIMENT_NAME)

        out_csv = results_csv_path(EXPERIMENT_NAME)
        ensure_dir(out_csv.parent)
        results.to_csv(out_csv, index=False)
        logger.info("Results written: %s (%d rows).", out_csv, results.shape[0])

        write_table(results, mode, logger)
        print(f"{BASE_EXPERIMENT_NAME} (mode={mode}): {out_csv} ({results.shape[0]} rows).")


if __name__ == "__main__":
    main()
