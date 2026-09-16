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
Q1 step 2 (reserse/2026-09-14_specifikace_rozsireni_q1.md, A.7) - reproduces
a power analysis for the confirmatory test H1 (F_A1, primary family, the
strictest Holm threshold alpha/2=0.025 in the m=2 family) on 10 observed
paired differences Delta = auc_rnx(pred) - auc_rnx(alpha0) in regime L
(`exp1_dr_benchmark`, the original 9 eligible datasets without two_moons +
segment/hierarchical_clusters/torus/iris/swiss_roll/sphere/s_curve/helix/severed_sphere
- see config_experiments.yaml:
power_analysis_regime.observed_deltas_pred_minus_alpha0, the source of the
numbers is cited directly in the config, NOT copied here in the code).

Four methods for estimating the required n (A.7 (a)-(d)):
  (a) noether  - Noether (1987) DOI 10.1080/01621459.1987.10478478, the
      formula for the Wilcoxon signed-rank test: n = (z_(1-a)+z_(1-b))^2/(3*(p_w-1/2)^2),
      p_w = Phi(sqrt(2)*mu/sigma); 2 scale variants (robust MAD, full sd).
  (b) t_approx - a paired t-test approximation, n=(z_(1-a)+z_(1-b))^2/d^2, d=mu/sigma.
  (c) sign_exact - the exact sign test (binomial distribution, p_+ = the
      fraction of positive Delta), the critical point k for a given n and
      alpha, power = P(K>=k|n,p_+).
  (d) mc_empirical/mc_normal - Monte Carlo (seed from the config, B=n_boot
      replications): the Wilcoxon test with normal approximation and the
      Yates 0.5 correction (scipy `wilcoxon(..., mode='approx', correction=True)`)
      on (i) resampled (with replacement) data from the 10 observed Delta
      ("mc_empirical", the heavy right tail helps Wilcoxon) and (ii) data
      generated from Normal(mu,sigma) for both sigma variants ("mc_normal").

Each method/scale/alpha combination is computed over
`power_analysis_regime.n_grid` (power_at_n for each n) + `n_required` (the
smallest n reaching power_target - analytically for (a)/(b), by search in
n_grid for (c)/(d), NaN if the grid is insufficient - no silent fallback to
a hardcoded number).

Output: results/tables/[<mode>/]power_analysis_regime.csv (schema see A.9).

Run: venv\\python.exe -m src.experiments.power_analysis_regime [--quick|--full|--smoke]
or: src\\run_power_analysis_regime.bat [quick|full|smoke]
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from scipy.stats import norm, wilcoxon

from src.common.config import get_tables_dir
from src.common.logging_utils import get_logger
from src.experiments.config_experiments import resolve_experiment_config
from src.experiments.exp_common import add_mode_args, resolve_mode

MODULE_NAME = "power_analysis_regime"

_CSV_COLUMNS = [
    "method", "effect_median", "scale", "scale_type", "alpha_onesided", "power_target",
    "n", "n_required", "power_at_n", "n_boot", "seed", "source_rows",
]


# ---------------------------------------------------------------------------
# Pure computational functions (testable without I/O - see tests/test_power_analysis_regime.py)
# ---------------------------------------------------------------------------

def summary_stats(deltas: np.ndarray) -> dict[str, float]:
    """mu (median), mean, sd (ddof=1), MAD (median absolute deviation from
    the median), robust sigma (1.4826*MAD - a consistent sigma estimator
    for the normal distribution, Huber 1981), and p_+ (fraction of positive Delta)."""
    d = np.asarray(deltas, dtype=np.float64)
    median = float(np.median(d))
    mad = float(np.median(np.abs(d - median)))
    return {
        "median": median, "mean": float(d.mean()), "sd": float(d.std(ddof=1)),
        "mad": mad, "sigma_robust": 1.4826 * mad, "p_plus": float(np.mean(d > 0)),
    }


def noether_n_required(mu: float, sigma: float, alpha_onesided: float, power_target: float) -> float:
    """Noether (1987): n = (z_(1-a)+z_(1-b))^2 / (3*(p_w-1/2)^2), p_w=Phi(sqrt(2)*mu/sigma)."""
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}.")
    p_w = float(norm.cdf(np.sqrt(2.0) * mu / sigma))
    if p_w <= 0.5:
        return float("inf")
    z_a = float(norm.ppf(1.0 - alpha_onesided))
    z_b = float(norm.ppf(power_target))
    return float((z_a + z_b) ** 2 / (3.0 * (p_w - 0.5) ** 2))


def noether_power_at_n(mu: float, sigma: float, alpha_onesided: float, n: int) -> float:
    """Inverted Noether formula: z_(1-b) = sqrt(3n)*(p_w-1/2) - z_(1-a), power=Phi(z_(1-b))."""
    p_w = float(norm.cdf(np.sqrt(2.0) * mu / sigma))
    z_a = float(norm.ppf(1.0 - alpha_onesided))
    z_b = np.sqrt(3.0 * n) * (p_w - 0.5) - z_a
    return float(norm.cdf(z_b))


def t_approx_n_required(mu: float, sigma: float, alpha_onesided: float, power_target: float) -> float:
    """A paired t-test approximation: n = (z_(1-a)+z_(1-b))^2 / d^2, d=mu/sigma."""
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}.")
    d = mu / sigma
    if d <= 0:
        return float("inf")
    z_a = float(norm.ppf(1.0 - alpha_onesided))
    z_b = float(norm.ppf(power_target))
    return float((z_a + z_b) ** 2 / d ** 2)


def t_approx_power_at_n(mu: float, sigma: float, alpha_onesided: float, n: int) -> float:
    d = mu / sigma
    z_a = float(norm.ppf(1.0 - alpha_onesided))
    z_b = np.sqrt(n) * d - z_a
    return float(norm.cdf(z_b))


def sign_exact_power_at_n(p_plus: float, alpha_onesided: float, n: int) -> tuple[float, int]:
    """Exact sign test: the smallest k for which P(K>=k|n,0.5) <= alpha (the
    one-sided critical point under H0), then power = P(K>=k|n,p_plus) (a
    binomial distribution). Returns (power, critical_point_k)."""
    from scipy.stats import binom

    k_crit = n + 1  # if no k satisfies the condition (n too small), the test never rejects
    for k in range(0, n + 1):
        p_h0 = float(binom.sf(k - 1, n, 0.5))  # P(K>=k|H0)
        if p_h0 <= alpha_onesided:
            k_crit = k
            break
    power = float(binom.sf(k_crit - 1, n, p_plus)) if k_crit <= n else 0.0
    return power, k_crit


def mc_wilcoxon_power(sampler, alpha_onesided_list: list[float], n: int, n_boot: int, seed: int) -> dict[float, float]:
    """Monte Carlo power of the Wilcoxon test (normal approximation, the
    Yates 0.5 correction, one-sided 'greater') for a given sampler
    `sampler(rng, n) -> np.ndarray` (bootstrap from empirical data OR
    Normal(mu,sigma)). Returns {alpha: power} - the p-values are computed
    only ONCE for all alphas (saving computation)."""
    rng = np.random.default_rng(seed)
    pvalues = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sample = sampler(rng, n)
        sample = sample[sample != 0.0]
        if sample.shape[0] < 1 or np.allclose(sample, 0.0):
            pvalues[b] = 1.0
            continue
        try:
            res = wilcoxon(sample, alternative="greater", mode="approx", correction=True, zero_method="wilcox")
            pvalues[b] = float(res.pvalue)
        except ValueError:
            pvalues[b] = 1.0
    return {alpha: float(np.mean(pvalues < alpha)) for alpha in alpha_onesided_list}


def n_required_from_grid(n_grid: list[int], power_at_n: dict[int, float], power_target: float) -> float:
    """The smallest n from `n_grid` for which `power_at_n[n] >= power_target`, otherwise NaN."""
    candidates = sorted(n for n in n_grid if power_at_n.get(n, 0.0) >= power_target)
    return float(candidates[0]) if candidates else float("nan")


# ---------------------------------------------------------------------------
# Orchestration (I/O)
# ---------------------------------------------------------------------------

def build_power_analysis_table(cfg: dict[str, Any]) -> pd.DataFrame:
    deltas = np.array([float(v) for v in cfg["observed_deltas_pred_minus_alpha0"].values()], dtype=np.float64)
    n_datasets_source = deltas.shape[0]
    stats = summary_stats(deltas)
    mu = stats["median"]
    n_grid: list[int] = [int(v) for v in cfg["n_grid"]]
    alpha_list: list[float] = [float(v) for v in cfg["alpha_onesided"]]
    power_target = float(cfg["power_target"])
    n_boot = int(cfg["n_boot"])
    seed = int(cfg["seed"])
    source_rows = f"power_analysis_regime.observed_deltas_pred_minus_alpha0 (n={n_datasets_source}, config_experiments.yaml)"

    rows: list[dict[str, Any]] = []

    for scale_type, sigma in (("robust_mad", stats["sigma_robust"]), ("sd", stats["sd"])):
        for alpha in alpha_list:
            n_req = noether_n_required(mu, sigma, alpha, power_target)
            for n in n_grid:
                power = noether_power_at_n(mu, sigma, alpha, n)
                rows.append({
                    "method": "noether", "effect_median": mu, "scale": sigma, "scale_type": scale_type,
                    "alpha_onesided": alpha, "power_target": power_target, "n": n, "n_required": n_req,
                    "power_at_n": power, "n_boot": np.nan, "seed": np.nan, "source_rows": source_rows,
                })

    for scale_type, sigma in (("robust_mad", stats["sigma_robust"]), ("sd", stats["sd"])):
        for alpha in alpha_list:
            n_req = t_approx_n_required(mu, sigma, alpha, power_target)
            for n in n_grid:
                power = t_approx_power_at_n(mu, sigma, alpha, n)
                rows.append({
                    "method": "t_approx", "effect_median": mu, "scale": sigma, "scale_type": scale_type,
                    "alpha_onesided": alpha, "power_target": power_target, "n": n, "n_required": n_req,
                    "power_at_n": power, "n_boot": np.nan, "seed": np.nan, "source_rows": source_rows,
                })

    for alpha in alpha_list:
        power_by_n: dict[int, float] = {}
        for n in n_grid:
            power, _k = sign_exact_power_at_n(stats["p_plus"], alpha, n)
            power_by_n[n] = power
        n_req = n_required_from_grid(n_grid, power_by_n, power_target)
        for n in n_grid:
            rows.append({
                "method": "sign_exact", "effect_median": mu, "scale": stats["p_plus"], "scale_type": "binomial_p",
                "alpha_onesided": alpha, "power_target": power_target, "n": n, "n_required": n_req,
                "power_at_n": power_by_n[n], "n_boot": np.nan, "seed": np.nan, "source_rows": source_rows,
            })

    def _empirical_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.choice(deltas, size=n, replace=True)

    for n in n_grid:
        power_by_alpha = mc_wilcoxon_power(_empirical_sampler, alpha_list, n, n_boot, seed)
        for alpha in alpha_list:
            rows.append({
                "method": "mc_empirical", "effect_median": mu, "scale": np.nan, "scale_type": "empirical",
                "alpha_onesided": alpha, "power_target": power_target, "n": n, "n_required": np.nan,
                "power_at_n": power_by_alpha[alpha], "n_boot": n_boot, "seed": seed, "source_rows": source_rows,
            })
    for scale_type, sigma in (("robust_mad", stats["sigma_robust"]), ("sd", stats["sd"])):
        def _normal_sampler(rng: np.random.Generator, n: int, _mu=mu, _sigma=sigma) -> np.ndarray:
            return rng.normal(loc=_mu, scale=_sigma, size=n)

        for n in n_grid:
            power_by_alpha = mc_wilcoxon_power(_normal_sampler, alpha_list, n, n_boot, seed)
            for alpha in alpha_list:
                rows.append({
                    "method": "mc_normal", "effect_median": mu, "scale": sigma, "scale_type": scale_type,
                    "alpha_onesided": alpha, "power_target": power_target, "n": n, "n_required": np.nan,
                    "power_at_n": power_by_alpha[alpha], "n_boot": n_boot, "seed": seed, "source_rows": source_rows,
                })

    # n_required for mc_empirical/mc_normal (grid search, like sign_exact) - filled in only after computing all n
    out = pd.DataFrame(rows, columns=_CSV_COLUMNS)
    for method in ("mc_empirical", "mc_normal"):
        for scale_type in out.loc[out["method"] == method, "scale_type"].unique():
            for alpha in alpha_list:
                mask = (out["method"] == method) & (out["scale_type"] == scale_type) & (out["alpha_onesided"] == alpha)
                power_by_n = dict(zip(out.loc[mask, "n"], out.loc[mask, "power_at_n"]))
                n_req = n_required_from_grid(n_grid, power_by_n, power_target)
                out.loc[mask, "n_required"] = n_req

    return out.sort_values(["method", "scale_type", "alpha_onesided", "n"]).reset_index(drop=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Q1 step 2 A.7: power analysis for the confirmatory test H1 (regime L).")
    add_mode_args(parser)
    args = parser.parse_args()
    mode = resolve_mode(args)

    logger = get_logger(MODULE_NAME, mode=mode)
    cfg = resolve_experiment_config("power_analysis_regime", mode)

    table = build_power_analysis_table(cfg)

    tables_dir = get_tables_dir(mode)
    out_csv = tables_dir / "power_analysis_regime.csv"
    table.to_csv(out_csv, index=False)
    logger.info("Written: %s (%d rows).", out_csv, table.shape[0])

    noether_robust_025 = table[(table["method"] == "noether") & (table["scale_type"] == "robust_mad") & (table["alpha_onesided"] == 0.025)]
    n_req = noether_robust_025["n_required"].iloc[0] if not noether_robust_025.empty else float("nan")
    print(f"power_analysis_regime: {table.shape[0]} rows -> {out_csv}; Noether (robust, alpha=0.025) n_required={n_req:.1f}")


if __name__ == "__main__":
    main()
