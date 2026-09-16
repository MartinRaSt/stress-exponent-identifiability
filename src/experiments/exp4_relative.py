# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-14
# License: see the LICENSE file in the repository root
"""
K9 (documentation/2026-09-12_plan_smeru_clanku.md) - computes the relative
change in stability/quality of temporal regularization vs. the lambda=0
baseline (`results/data/[<mode>/]exp4_relative.csv`).

Originally this CSV was computed ONLY by `src/figures/fig_temporal_pareto.py`.
But in `src/main.py --no-figures` the figures do not run at all, and the
table/number step (`report_tables.write_exp4_relative_table`,
`export_numbers.py` group 'K9 relative change') ran BEFORE the figures even
in the mode with figures - the result was stale/missing data and skipped E4
macros (found 2026-09-14, see
documentation/2026-09-14_exp4_relative_poradi.md). The computation was
therefore moved here into a shared module; both `fig_temporal_pareto.py` and
`src/main.py` call the same function `compute_exp4_relative`.

Run standalone: venv\\python.exe -m src.experiments.exp4_relative [--quick|--full|--smoke]
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.figures.fig_common import mode_data_dir, require_experiment_csv
from src.experiments.exp4_common import DTSNE_METHOD_RE, OUR_TEMPORAL_METHOD_RE

BASE_EXPERIMENT_NAME = "exp4_temporal"

# Kept as a fallback in case OUR_TEMPORAL_METHOD_RE (only
# solver in {smacof, sgd}) does not cover a future solver - matches the
# original pattern before the 2026-09-16 fix (regression: dtsne_lambda<L>
# was not recognized, see documentation).
_METHOD_RE = re.compile(r"^lambda(?P<lam>[0-9.eE+-]+)_alpha(?P<alpha>[0-9.eE+-]+)_(?P<solver>\w+)$")


def _parse_method(method: str) -> tuple[float, float, str]:
    """Parses 'method' into (lam, alpha, solver).

    Two families (see src/experiments/exp4_temporal.py and the shared
    regexes in exp4_common.py, also used by exp4_neighbor_metrics.py):
    - our family 'lambda{lam}_alpha{alpha}_{solver}' (solver=smacof/sgd),
    - the dynamic t-SNE baseline 'dtsne_lambda{lam}' (K12, 2026-09-14) - has
      no alpha weighting, alpha=NaN sentinel, solver='dtsne' for bucketing.

    Fix 2026-09-16 (blocking regression introduced 2026-09-14 when dtsne was
    added): the original code only knew the first pattern and raised
    ValueError on 'dtsne_lambda0.0', which crashed the whole
    compute_exp4_relative and exp4_relative.csv was never generated (see
    results/logs/main_20260915_232925.log)."""
    m_dtsne = DTSNE_METHOD_RE.match(method)
    if m_dtsne is not None:
        return float(m_dtsne.group("lam")), float("nan"), "dtsne"
    m_ours = OUR_TEMPORAL_METHOD_RE.match(method)
    if m_ours is not None:
        return float(m_ours.group("lam")), float(m_ours.group("alpha")), m_ours.group("solver")
    # Fallback to the original generic pattern (any 'solver', not just smacof/sgd) -
    # preserves behavior for any other families beyond smacof/sgd/dtsne.
    m = _METHOD_RE.match(method)
    if m is None:
        raise ValueError(f"Unexpected 'method' format in exp4_temporal_results.csv: '{method}'.")
    return float(m.group("lam")), float(m.group("alpha")), m.group("solver")


def _compute_relative_change(df: pd.DataFrame) -> pd.DataFrame:
    """Median stab/qual over seeds per (dataset, solver, lam, alpha) and the
    relative change (ratio and percent) vs. the lambda=0 baseline in the same
    (dataset, solver, alpha). Logic moved unchanged from
    fig_temporal_pareto.py (2026-09-14)."""
    ok = df[df["status"] == "ok"].copy()
    parsed = ok["method"].apply(_parse_method)
    ok["lam"] = [p[0] for p in parsed]
    ok["alpha"] = [p[1] for p in parsed]
    ok["solver"] = [p[2] for p in parsed]

    # dropna=False: the dtsne family has alpha=NaN sentinel (no alpha
    # weighting) - pandas would otherwise silently drop whole groups with a
    # NaN key (without dropna=False, dtsne would be missing from
    # exp4_relative.csv even though _parse_method already succeeds).
    med = ok.groupby(["dataset", "solver", "alpha", "lam"], dropna=False)[["stab", "qual"]].median().reset_index()

    rows = []
    for (dataset_name, solver, alpha), sub in med.groupby(["dataset", "solver", "alpha"], dropna=False):
        baseline = sub[sub["lam"] == 0.0]
        if baseline.empty:
            continue
        stab0 = float(baseline["stab"].iloc[0])
        qual0 = float(baseline["qual"].iloc[0])
        for _, r in sub.iterrows():
            stab_ratio = float(r["stab"]) / stab0 if stab0 != 0 else np.nan
            qual_ratio = float(r["qual"]) / qual0 if qual0 != 0 else np.nan
            rows.append({
                "dataset": dataset_name, "solver": solver, "alpha": alpha, "lam": r["lam"],
                "stab_median": float(r["stab"]), "qual_median": float(r["qual"]),
                "stab_ratio_vs_lambda0": stab_ratio, "qual_ratio_vs_lambda0": qual_ratio,
                "stab_pct_change_vs_lambda0": (stab_ratio - 1.0) * 100.0 if np.isfinite(stab_ratio) else np.nan,
                "qual_pct_change_vs_lambda0": (qual_ratio - 1.0) * 100.0 if np.isfinite(qual_ratio) else np.nan,
            })
    # Minor fix found during the move (2026-09-14): the original
    # `pd.DataFrame(rows)` without explicit columns returns, for an empty
    # `rows`, a DataFrame WITHOUT columns -> the following
    # `sort_values(["dataset", ...])` raises a confusing KeyError instead of
    # the intended fail-loud ValueError in `compute_exp4_relative`
    # (`if relative.empty: raise ValueError(...)`). An explicit `columns=`
    # ensures the empty result stays a genuinely empty DataFrame with the
    # correct schema - values for the non-empty case are unchanged.
    columns = [
        "dataset", "solver", "alpha", "lam", "stab_median", "qual_median",
        "stab_ratio_vs_lambda0", "qual_ratio_vs_lambda0",
        "stab_pct_change_vs_lambda0", "qual_pct_change_vs_lambda0",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(["dataset", "solver", "lam"]).reset_index(drop=True)


def compute_exp4_relative(mode: str, logger: logging.Logger | None = None) -> Path:
    """Loads `exp4_temporal_results.csv` for the given mode (`require_experiment_csv`
    - the same fail-loud check as in the figure script), computes the relative
    change vs. lambda=0 and writes `results/data/[<mode>/]exp4_relative.csv`
    (mode-aware path via `mode_data_dir`, no manual path assembly).

    Fail-loud: an empty result (no combination with a lambda=0 baseline) is
    an error (ValueError), not a silent skip - same as previously in
    fig_temporal_pareto.py."""
    df = require_experiment_csv(BASE_EXPERIMENT_NAME, mode)
    relative = _compute_relative_change(df)
    if relative.empty:
        raise ValueError(
            f"exp4_temporal_results.csv (mode={mode}) contains no combination with a lambda=0 "
            "baseline - cannot compute the relative change."
        )
    relative_path = mode_data_dir(mode) / "exp4_relative.csv"
    relative.to_csv(relative_path, index=False)
    if logger is not None:
        logger.info("exp4_relative.csv written: %s (%d rows).", relative_path, len(relative))
    return relative_path


def main() -> None:
    from src.experiments.exp_common import parse_mode_args
    from src.common.logging_utils import get_logger

    mode = parse_mode_args("K9: relative change in stability/quality of temporal regularization vs. lambda=0.")
    logger = get_logger("exp4_relative", mode=mode)
    path = compute_exp4_relative(mode, logger)
    print(f"exp4_relative: written to {path}.")


if __name__ == "__main__":
    main()
