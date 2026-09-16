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
Q1 step 2 (reserse/2026-09-14_specifikace_rozsireni_q1.md, part A, step 0) -
screening of candidate hold-out datasets for regime L (low rho_NN): for each
candidate (config_experiments.yaml: screen_regime_candidates.candidates)
loads the dataset, deduplicates, subsamples to n_max=2000
(SUBSAMPLE_SEED=42, same mechanism as E1 -
`src.datasets.subsample.subsample_dataset`), computes
nn_ratio_k1/theta/m/id_twonn (SAME functions as K1,
`src.experiments.dataset_properties`), classifies into a regime using the
FROZEN rule (`results/data/alpha_pred_rule_frozen_20260913.json`, NEVER the
production path, which may be overwritten in the future) and evaluates
eligibility E1-E5 (A.4):
  E1: d >= eligibility.min_d (requires kind='vector' with X, i.e. all new
      datasets - ambient dimension d, NOT intrinsic)
  E2: theta > 0 (after dedup; nn_ratio_k1 must be defined)
  E3: n_used >= eligibility.min_n
  E4: regime_frozen == 'low_ratio' (regime L per the frozen rule)
  E5: the loader is deterministic (two independent calls to `load_dataset`
      return bit-identical X) AND runs without error (caught by the
      exception below)
`eligible_all` = E1 AND E2 AND E3 AND E5 (E4 is NOT a condition for general
"usability" - only for inclusion in the CONFIRMATORY regime L analysis, see
the last paragraph of A.4: the M/H regime is still computed for the F_A3
no-harm check).

Output: results/data/[<mode>/]regime_candidates_screen.csv (schema see A.9,
`_FINAL_COLUMNS` below) + log + DONE file (checkpoint/resume - 1 row = 1
candidate, RunKey with method='screen', seed=0, same mechanism as
`dataset_properties.py`).

Run: venv\\python.exe -m src.experiments.screen_regime_candidates [--quick|--full|--smoke]
or: src\\run_screen_regime_candidates.bat [quick|full|smoke]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.common.checkpoint import results_csv_path
from src.common.logging_utils import get_logger
from src.experiments.config_experiments import resolve_experiment_config
from src.experiments.exp1_regime_stratified import classify_regime, regime_alpha_pred
from src.experiments.exp_common import (
    add_mode_args,
    filter_already_done,
    resolve_experiment_name,
    resolve_mode,
    run_experiment_grid,
    write_done_file,
)

BASE_EXPERIMENT_NAME = "screen_regime_candidates"
METHOD_LABEL = "screen"
SEED_LABEL = 0

COLUMN_KEYS = [
    "source", "requires_download", "n_raw", "n_after_dedup", "n_duplicates_removed", "n_used",
    "d", "n_classes", "standardized", "nn_ratio_k1", "log_nn_ratio_k1", "theta", "m",
    "regime_frozen", "alpha_pred_frozen", "id_twonn",
    "eligible_E1_d", "eligible_E2_theta", "eligible_E3_n", "eligible_E4_regime", "eligible_E5_loader", "eligible_all",
    "priority_rank", "is_control",
]
_FINAL_COLUMNS = ["dataset"] + COLUMN_KEYS


def frozen_rule_path() -> Path:
    from src.common.config import get_path

    return get_path("results_data_dir") / "alpha_pred_rule_frozen_20260913.json"


def load_frozen_rule() -> dict[str, Any]:
    """ALWAYS loads the frozen copy of the rule (A.5) - NEVER the production
    `alpha_pred_rule.json`, which may change on the next `fit_alpha_rule --full`
    run (e.g. once the hold-out datasets are added to E1/E6) - screening must
    give the same classification regardless of when it is run."""
    import json

    path = frozen_rule_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Missing frozen rule: {path}\n"
            "Create it as a one-time copy of the production results/data/alpha_pred_rule.json "
            "(see reserse/2026-09-14_specifikace_rozsireni_q1.md, A.5)."
        )
    with open(path, "r", encoding="utf-8") as f:
        rule = json.load(f)
    if rule.get("variant") != "two_threshold":
        raise ValueError(f"{path}: expected variant 'two_threshold', got '{rule.get('variant')}'.")
    return rule


def _check_deterministic_loader(dataset_key: str) -> bool:
    """E5: loads the dataset TWICE independently and compares X bit-for-bit
    (`np.array_equal`) - detects nondeterministic loaders (missing/wrong
    seed). y is also compared, if present."""
    from src.datasets.registry import load_dataset

    ds1 = load_dataset(dataset_key)
    ds2 = load_dataset(dataset_key)
    if not np.array_equal(ds1.X, ds2.X):
        return False
    if (ds1.y is None) != (ds2.y is None):
        return False
    if ds1.y is not None and not np.array_equal(ds1.y, ds2.y):
        return False
    return True


def _run_single(task: dict[str, Any]) -> dict[str, Any]:
    """One row = one candidate: loading + dedup (already in the loader) +
    subsampling + K1-style properties + classification using the frozen
    rule + E1-E5 eligibility."""
    from scipy.spatial.distance import pdist, squareform

    from src.datasets.registry import load_dataset
    from src.datasets.subsample import subsample_dataset
    from src.experiments.dataset_properties import id_twonn as _id_twonn
    from src.experiments.dataset_properties import nn_distances_from_D as _nn_distances_from_D
    from src.experiments.dataset_properties import nn_ratio as _nn_ratio

    dataset_key = task["dataset_key"]
    n_max = task["n_max"]
    subsample_seed = task["subsample_seed"]
    rule = task["rule"]
    eligibility = task["eligibility"]
    priority_rank = task["priority_rank"]
    is_control = task["is_control"]
    requires_download = task["requires_download"]

    result: dict[str, Any] = {"dataset_name": dataset_key, "method_name": METHOD_LABEL, "seed": SEED_LABEL}
    t0 = time.perf_counter()
    row: dict[str, Any] = {
        "source": task.get("source", ""), "requires_download": requires_download,
        "priority_rank": priority_rank, "is_control": is_control,
    }
    try:
        ds = load_dataset(dataset_key)
        n_after_dedup = ds.n_samples
        n_raw = int(ds.meta.get("n_before_dedup", n_after_dedup))
        n_duplicates_removed = int(ds.meta.get("n_duplicates_removed", 0))
        standardized = bool(ds.meta.get("standardized", False))

        ds_sub = subsample_dataset(ds, n_max=n_max, random_state=subsample_seed)
        n_used = ds_sub.n_samples
        d = int(ds_sub.X.shape[1])
        n_classes = ds_sub.n_classes if ds_sub.n_classes is not None else np.nan

        X = np.asarray(ds_sub.X, dtype=np.float64)
        D = squareform(pdist(X, metric="euclidean"))
        n = D.shape[0]
        triu = D[np.triu_indices(n, k=1)]
        m = float(np.median(triu))
        if m <= 0:
            raise ValueError(f"'{dataset_key}': median of all pairwise distances is 0 (degenerate data after subsampling).")
        theta = float(np.median(_nn_distances_from_D(D, 1)))
        nn_ratio_k1 = _nn_ratio(D, 1, m)

        eligible_e1_d = d >= int(eligibility["min_d"])
        eligible_e2_theta = theta > 0.0
        eligible_e3_n = n_used >= int(eligibility["min_n"])

        if eligible_e2_theta:
            regime_frozen = str(classify_regime(np.array([nn_ratio_k1]), rule)[0])
            alpha_pred_frozen = float(regime_alpha_pred(rule)[regime_frozen])
            log_nn_ratio_k1 = float(np.log(nn_ratio_k1))
        else:
            regime_frozen, alpha_pred_frozen, log_nn_ratio_k1 = "", np.nan, np.nan
        eligible_e4_regime = regime_frozen == "low_ratio"

        try:
            idt = _id_twonn(D, 0.9)
        except Exception:
            idt = np.nan

        eligible_e5_loader = _check_deterministic_loader(dataset_key)
        eligible_all = bool(eligible_e1_d and eligible_e2_theta and eligible_e3_n and eligible_e5_loader)

        row.update({
            "n_raw": n_raw, "n_after_dedup": n_after_dedup, "n_duplicates_removed": n_duplicates_removed,
            "n_used": n_used, "d": d, "n_classes": n_classes, "standardized": standardized,
            "nn_ratio_k1": nn_ratio_k1, "log_nn_ratio_k1": log_nn_ratio_k1, "theta": theta, "m": m,
            "regime_frozen": regime_frozen, "alpha_pred_frozen": alpha_pred_frozen, "id_twonn": idt,
            "eligible_E1_d": eligible_e1_d, "eligible_E2_theta": eligible_e2_theta, "eligible_E3_n": eligible_e3_n,
            "eligible_E4_regime": eligible_e4_regime, "eligible_E5_loader": eligible_e5_loader, "eligible_all": eligible_all,
        })
        result["status"] = "ok"
        result["error"] = ""
    except Exception as exc:  # network/parsing/computation errors for one candidate must not stop the others
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["wall_time_sec"] = time.perf_counter() - t0
    result["embedding"] = None
    result["metrics"] = row
    result["extra"] = {}
    return result


def _write_final_csv(experiment_name: str, logger) -> Path:
    csv_path = results_csv_path(experiment_name)
    df = pd.read_csv(csv_path)
    ok = df[df["status"] == "ok"].copy()
    final = ok[_FINAL_COLUMNS].sort_values("priority_rank").reset_index(drop=True)
    out_path = csv_path.parent / "regime_candidates_screen.csv"
    final.to_csv(out_path, index=False)
    n_eligible_all = int(final["eligible_all"].sum()) if not final.empty else 0
    n_eligible_regime_l = int((final["eligible_all"] & (final["regime_frozen"] == "low_ratio")).sum()) if not final.empty else 0
    logger.info(
        "Final CSV written: %s (%d rows, %d failed omitted; eligible_all=%d, of which regime L=%d).",
        out_path, final.shape[0], (df["status"] != "ok").sum(), n_eligible_all, n_eligible_regime_l,
    )
    print(
        f"screen_regime_candidates: {final.shape[0]} candidates processed, "
        f"{n_eligible_all} eligible (E1-E3,E5), of which {n_eligible_regime_l} in regime L -> {out_path}"
    )
    return out_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Q1 step 2, step 0: screening of candidate hold-out datasets for regime L.")
    add_mode_args(parser)
    args = parser.parse_args()
    mode = resolve_mode(args)

    logger = get_logger(BASE_EXPERIMENT_NAME, mode=mode)
    EXPERIMENT_NAME = resolve_experiment_name(BASE_EXPERIMENT_NAME, mode)

    cfg = resolve_experiment_config("screen_regime_candidates", mode)
    conf_cfg = resolve_experiment_config("exp1_holdout_confirmatory", mode)
    n_max = int(cfg["n_max"])
    subsample_seed = int(cfg["subsample_seed"])
    rule = load_frozen_rule()
    logger.info("Frozen rule loaded from %s: t1=%.6f, t2=%.6f.", frozen_rule_path(), rule["coefficients"]["t1"], rule["coefficients"]["t2"])

    tasks: list[dict[str, Any]] = []
    for cand in cfg["candidates"]:
        tasks.append({
            "dataset_name": cand["key"], "method_name": METHOD_LABEL, "seed": SEED_LABEL,
            "dataset_key": cand["key"], "priority_rank": int(cand["priority_rank"]),
            "is_control": bool(cand["is_control"]), "requires_download": bool(cand["requires_download"]),
            "source": cand.get("source", ""), "n_max": n_max, "subsample_seed": subsample_seed,
            "rule": rule, "eligibility": conf_cfg["eligibility"],
        })

    todo, n_done = filter_already_done(EXPERIMENT_NAME, tasks)
    logger.info("%s (mode=%s): %d candidates already done (skipped), %d new to compute.", EXPERIMENT_NAME, mode, n_done, len(todo))

    # sequential run (sequential=True): the number of candidates is small
    # (<=17) and several loaders download/parse relatively large OpenML data
    # (shuttle 58000x9) - parallelization would just open several concurrent
    # network connections without benefit (unlike E1/E6 with thousands of
    # runs). This also avoids concurrent writes to the shared OpenML/manifest cache.
    n_ok, n_err = run_experiment_grid(EXPERIMENT_NAME, todo, _run_single, COLUMN_KEYS, logger, sequential=True)
    logger.info("Done. Newly successful: %d, newly failed: %d, skipped: %d.", n_ok, n_err, n_done)
    write_done_file(EXPERIMENT_NAME, logger, extra_info={"mode": mode})

    _write_final_csv(EXPERIMENT_NAME, logger)


if __name__ == "__main__":
    main()
