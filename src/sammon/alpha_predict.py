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
K7 (documentation/2026-09-12_plan_smeru_clanku.md) - alpha prediction from
distance concentration (nn_ratio_k1) without tuning on the data: pure
functions (no I/O apart from loading the rule) shared between
`src/experiments/fit_alpha_rule.py` (rule derivation + LOO validation) and
`src/methods/sammon_alpha_pred.py` (method adapter for E1/E3).

`nn_ratio_k1_from_D` uses the SAME functions as K1
(`src.experiments.dataset_properties.nn_distances_from_D`/`nn_ratio` -
imported, NOT copied, so that the prediction in a production run and the
rule derivation from K1 data always compute the same quantity the same
way).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

ALPHA_MIN = 0.0
ALPHA_MAX = 3.0

VALID_VARIANTS = {"log_linear", "two_threshold"}


def nn_ratio_k1_from_D(D: np.ndarray) -> float:
    """nn_ratio_k1 = median distance to the 1st nearest neighbor / median of
    all pairwise distances - EXACTLY the same computation as K1
    (`src.experiments.dataset_properties.nn_ratio(D, k=1, median_all)`)."""
    from src.experiments.dataset_properties import nn_ratio

    n = D.shape[0]
    triu = D[np.triu_indices(n, k=1)]
    median_all = float(np.median(triu))
    if median_all <= 0:
        raise ValueError("Median of all pairwise distances is 0 - nn_ratio_k1 is undefined (degenerate data).")
    return nn_ratio(D, 1, median_all)


def predict_alpha_log_linear(nn_ratio_k1: float, a: float, b: float) -> float:
    """alpha_pred = clip(a + b*ln(nn_ratio_k1), 0, 3) (R6-B log-linear variant)."""
    if nn_ratio_k1 <= 0:
        raise ValueError(f"nn_ratio_k1 must be positive, got {nn_ratio_k1}.")
    raw = a + b * float(np.log(nn_ratio_k1))
    return float(np.clip(raw, ALPHA_MIN, ALPHA_MAX))


def predict_alpha_two_threshold(nn_ratio_k1: float, t1: float, t2: float, a_low: float, a_mid: float, a_high: float) -> float:
    """Piecewise-constant rule over 3 regimes thresholded in log(nn_ratio_k1)
    (R6-B / documentation 2026-09-12_kontrola_vysledku_s1.md - a sharper
    threshold than the linear relation). Thresholds `t1<t2` are in units of
    log(nn_ratio_k1)."""
    if nn_ratio_k1 <= 0:
        raise ValueError(f"nn_ratio_k1 must be positive, got {nn_ratio_k1}.")
    if t1 >= t2:
        raise ValueError(f"t1={t1} must be < t2={t2}.")
    log_nn = float(np.log(nn_ratio_k1))
    if log_nn < t1:
        raw = a_low
    elif log_nn < t2:
        raw = a_mid
    else:
        raw = a_high
    return float(np.clip(raw, ALPHA_MIN, ALPHA_MAX))


def predict_alpha(nn_ratio_k1: float, rule: dict[str, Any]) -> float:
    """Apply an already-derived rule (loaded from JSON, see
    `load_alpha_pred_rule`) to the given nn_ratio_k1. `rule['variant']`
    determines which of the functions above is used."""
    variant = rule.get("variant")
    coef = rule.get("coefficients", {})
    if variant == "log_linear":
        return predict_alpha_log_linear(nn_ratio_k1, a=float(coef["a"]), b=float(coef["b"]))
    if variant == "two_threshold":
        return predict_alpha_two_threshold(
            nn_ratio_k1, t1=float(coef["t1"]), t2=float(coef["t2"]),
            a_low=float(coef["a_low"]), a_mid=float(coef["a_mid"]), a_high=float(coef["a_high"]),
        )
    raise ValueError(f"Unknown rule variant '{variant}' (expected one of {sorted(VALID_VARIANTS)}).")


def default_rule_path() -> Path:
    from src.common.config import get_path

    return get_path("results_data_dir") / "alpha_pred_rule.json"


def load_alpha_pred_rule(path: Path | None = None) -> dict[str, Any]:
    """Load the rule derived by `src/experiments/fit_alpha_rule.py`
    (`results/data/alpha_pred_rule.json`). Fail-loud if the file is missing -
    NO silent fallback to hardcoded coefficients."""
    rule_path = path if path is not None else default_rule_path()
    if not rule_path.exists():
        raise FileNotFoundError(
            f"Missing rule for sammon_alpha_pred: {rule_path}\n"
            "Run first: venv\\python.exe -m src.experiments.fit_alpha_rule "
            "(requires dataset_properties.py AND exp6_alpha_curves.py to have already run)."
        )
    with open(rule_path, "r", encoding="utf-8") as f:
        rule = json.load(f)
    if rule.get("variant") not in VALID_VARIANTS:
        raise ValueError(f"{rule_path} contains an unknown/missing 'variant': {rule.get('variant')!r}.")
    return rule
