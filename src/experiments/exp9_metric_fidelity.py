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
Q1 step 3 (reserse/2026-09-14_specifikace_rozsireni_q1.md, PART B) - a
metric-fidelity task with a KNOWN truth: a grid (scenario x replicate x
method), where scenario S1/S2/S3(/S4) has an exactly defined latent geometry
(see `src/datasets/synthetic_truth.py`) and the metrics
(`src/sammon/truth_metrics.py`) measure the ratio/cophenetic/geodesic
agreement of the embedding with this truth - unlike E1
(`exp1_dr_benchmark.py`), which only measures against the INPUT matrix D
without a known truth (see B.1).

Results (incl. embeddings and a compact truth summary) go to
results/data/[<mode>/]exp9_metric_fidelity_results.csv - resumable (its own
checkpoint, not exp_common.run_experiment_grid, because the run key is
(scenario, replicate, method), NOT (dataset, method, seed) - see the B.7
schema for `exp9_metric_fidelity_results.csv`).

Run: venv\\python.exe -m src.experiments.exp9_metric_fidelity [--quick|--full|--smoke]
or: src\\run_exp9_metric_fidelity.bat [quick|full|smoke]
"""
from __future__ import annotations

import csv
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.common.checkpoint import results_csv_path
from src.common.config import ensure_dir, get_path, load_config
from src.common.logging_utils import get_logger, wall_clock
from src.common.parallel import resolve_n_workers, run_parallel_map
from src.common.seeding import set_seed
from src.experiments.config_experiments import resolve_experiment_config
from src.experiments.exp_common import parse_mode_args, resolve_experiment_name

BASE_EXPERIMENT_NAME = "exp9_metric_fidelity"

# B.7: schema of results/data/[<mode>/]exp9_metric_fidelity_results.csv (the
# column order is binding - see the spec). Text columns vs. numeric (NaN
# default) - see `_TEXT_COLUMNS` below this list.
COLUMNS: list[str] = [
    "experiment", "scenario", "replicate", "seed_gen", "seed_method", "method", "n", "d", "n_clusters", "alpha",
    "selected_hyperparam", "init", "nn_ratio_k1", "regime_frozen",
    "centroid_pearson", "centroid_spearman", "centroid_lre", "centroid_logdist", "radius_slope", "radius_lre",
    "cophenetic_pearson", "triplet_accuracy", "pointwise_ultrametric_spearman", "geodesic_stress_si",
    "geodesic_pearson", "geodesic_lre", "aspect_ratio_error", "radius_ratio_error",
    "auc_rnx", "trustworthiness_k7", "continuity_k7", "stress_scale_invariant",
    "is_oracle", "note", "status", "error", "wall_time_sec",
]
_TEXT_COLUMNS = {"experiment", "scenario", "method", "selected_hyperparam", "init", "regime_frozen", "note", "status", "error"}
_KEY_COLUMNS = ("scenario", "replicate", "method")

# scenarios with a defined `oracle_truth` (B.2: S4 has no oracle - the
# "truth" is only a scalar radius ratio, no 2D embedding is evaluated as "truth")
_SCENARIOS_WITH_ORACLE = {"S1", "S2", "S3"}

_ALPHA_RE = re.compile(r"alpha=([-0-9.eE]+)")


def _embeddings_dir(experiment_name: str) -> Path:
    return get_path("embeddings_dir") / experiment_name


def _embedding_filename(scenario: str, replicate: int, method: str) -> str:
    return f"{scenario}__r{replicate}__{method}.npy"


def _truth_filename(scenario: str, replicate: int) -> str:
    return f"{scenario}__r{replicate}__truth.npz"


def build_tasks(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Builds the list of tasks (scenario x replicate x method [+ oracle_truth,
    see `_SCENARIOS_WITH_ORACLE`]) - B.2: generative seed = seed_base(scenario)
    + replicate, method seed = replicate."""
    tasks: list[dict[str, Any]] = []
    replicates = int(cfg["replicates"])
    methods = list(cfg["methods"])
    n_components = int(cfg["n_components"])
    n_pairs = int(cfg["pointwise_ultrametric_n_pairs"])

    for scenario in cfg["active_scenarios"]:
        scenario_cfg = cfg["scenarios"][scenario]
        seed_base = int(scenario_cfg["seed_base"])
        method_names = list(methods)
        if scenario in _SCENARIOS_WITH_ORACLE:
            method_names = method_names + ["oracle_truth"]
        for r in range(replicates):
            seed_gen = seed_base + r
            for method_name in method_names:
                tasks.append({
                    "scenario": scenario, "replicate": r, "method": method_name,
                    "seed_gen": seed_gen, "seed_method": r,
                    "cfg_scenario": scenario_cfg, "n_components": n_components,
                    "pointwise_n_pairs": n_pairs,
                })
    return tasks


def _n_clusters_for_scenario(scenario: str, truth: dict[str, Any]) -> float:
    if scenario == "S1":
        return float(truth["Delta"].shape[0])
    if scenario == "S2":
        return float(truth["u"].shape[0])
    if scenario == "S4":
        return 2.0
    return float("nan")  # S3: a manifold, no clusters


def _extract_alpha(method_name: str, selected_hyperparam: str) -> float:
    """Numeric alpha for sammon_* methods (B.4: 'selected_hyperparam ... and
    alpha for Sammon variants are written to the CSV') - fixed alphas taken
    directly from method_config, dynamic (auto/pred) ones parsed from the
    already-built `selected_hyperparam` (see `src/methods/sammon_alpha.py`,
    `src/methods/sammon_alpha_pred.py`, same format 'alpha=<x>...')."""
    from src.methods.common import method_config

    if method_name in ("sammon_alpha0_smacof", "sammon_alpha_smacof", "sammon_alpha2_smacof"):
        return float(method_config(method_name)["alpha"])
    if method_name in ("sammon_alpha_auto", "sammon_alpha_pred"):
        match = _ALPHA_RE.search(selected_hyperparam)
        return float(match.group(1)) if match else float("nan")
    return float("nan")


def _init_label(method_name: str) -> str:
    """Initialization label (B.4: 'tsne initialization ... because the
    initialization determines the global structure'). The Sammon family
    uses the shared `sammon.init.default` (PCA, see common/config.yaml);
    methods with their own 'init' key in method_config (tsne/tsne_auto)
    report it directly; others have no explicit init concept -> ''."""
    from src.methods.common import method_config

    if method_name.startswith("sammon"):
        return str(load_config()["sammon"]["init"]["default"])
    try:
        cfg = method_config(method_name)
    except KeyError:
        return ""
    return str(cfg.get("init", ""))


def _pointwise_ultrametric_spearman(
    Y: np.ndarray, leaf_of_point: np.ndarray, u: np.ndarray, n_pairs: int, seed: int,
) -> float:
    """S2 secondary metric (B.2): Spearman(u_ij, d_ij) over deterministically
    selected INTER-LEAF point pairs (leaf_of_point[i] != leaf_of_point[j]).
    For small n (quick/smoke), the number of available inter-leaf pairs is
    smaller than the requested `n_pairs` - ALL found pairs are used (no
    fabrication of missing ones), fail-loud only if < 2 pairs are found."""
    from scipy.stats import spearmanr

    n = Y.shape[0]
    rng = np.random.default_rng(seed)
    i = np.empty(0, dtype=np.int64)
    j = np.empty(0, dtype=np.int64)
    for _attempt in range(50):
        remaining = n_pairs - i.shape[0]
        if remaining <= 0:
            break
        batch = max(remaining * 3, 1000)
        cand_i = rng.integers(0, n, size=batch)
        cand_j = rng.integers(0, n, size=batch)
        mask = (cand_i != cand_j) & (leaf_of_point[cand_i] != leaf_of_point[cand_j])
        i = np.concatenate([i, cand_i[mask]])
        j = np.concatenate([j, cand_j[mask]])
    n_eff = min(n_pairs, i.shape[0])
    if n_eff < 2:
        raise RuntimeError("_pointwise_ultrametric_spearman: not enough inter-leaf point pairs (n too small).")
    i, j = i[:n_eff], j[:n_eff]
    u_ij = u[leaf_of_point[i], leaf_of_point[j]]
    d_ij = np.linalg.norm(Y[i] - Y[j], axis=1)
    rho, _ = spearmanr(u_ij, d_ij)
    return float(rho)


def _scenario_metrics(
    scenario: str, Y: np.ndarray, y: np.ndarray | None, truth: dict[str, Any], n_pairs: int, pointwise_seed: int,
) -> dict[str, float]:
    """Computes the metric-fidelity metrics (B.3) relevant to the given
    scenario - other columns remain NaN (undefined for that scenario, B.7
    schema note 'metrics undefined for a scenario = NaN')."""
    from scipy.spatial.distance import pdist, squareform

    from src.sammon import truth_metrics as tm

    m: dict[str, float] = {}
    if scenario == "S1":
        Delta = truth["Delta"]
        K = Delta.shape[0]
        centroids_Y = np.stack([Y[y == k].mean(axis=0) for k in range(K)], axis=0)
        delta = squareform(pdist(centroids_Y))
        m["centroid_pearson"] = tm.centroid_pearson(delta, Delta)
        m["centroid_spearman"] = tm.centroid_spearman(delta, Delta)
        delta_p, Delta_p = tm.upper_triangle_pairs(delta), tm.upper_triangle_pairs(Delta)
        m["centroid_lre"] = tm.log_ratio_error(delta_p, Delta_p)
        m["centroid_logdist"] = tm.log_distortion(delta_p, Delta_p)
        rho_Y = np.array([
            float(np.sqrt(np.mean(np.sum((Y[y == k] - centroids_Y[k]) ** 2, axis=-1)))) for k in range(K)
        ])
        m["radius_slope"] = tm.radius_slope(rho_Y, truth["r"])
        m["radius_lre"] = tm.radius_lre(rho_Y, truth["r"])
    elif scenario == "S2":
        u = truth["u"]
        Delta = truth["Delta"]
        leaf_of_point = truth["leaf_of_point"]
        K = u.shape[0]
        centroids_Y = np.stack([Y[leaf_of_point == k].mean(axis=0) for k in range(K)], axis=0)
        delta = squareform(pdist(centroids_Y))
        m["cophenetic_pearson"] = tm.cophenetic_pearson(delta, u)
        m["triplet_accuracy"] = tm.triplet_accuracy(u, delta)
        m["centroid_pearson"] = tm.centroid_pearson(delta, Delta)
        m["centroid_spearman"] = tm.centroid_spearman(delta, Delta)
        m["centroid_lre"] = tm.log_ratio_error(tm.upper_triangle_pairs(delta), tm.upper_triangle_pairs(Delta))
        m["pointwise_ultrametric_spearman"] = _pointwise_ultrametric_spearman(Y, leaf_of_point, u, n_pairs, pointwise_seed)
    elif scenario == "S3":
        G = truth["G"]
        T = truth["T"]
        d = squareform(pdist(Y))
        m["geodesic_stress_si"] = tm.geodesic_stress_scale_inv(d, G)
        m["geodesic_pearson"] = tm.geodesic_pearson(d, G)
        m["geodesic_lre"] = tm.geodesic_lre(d, G)
        m["aspect_ratio_error"] = tm.aspect_ratio_error(Y, T)
    elif scenario == "S4":
        r_ratio_truth = float(truth["r_ratio"])
        mu0, mu1 = Y[y == 0].mean(axis=0), Y[y == 1].mean(axis=0)
        rho0 = float(np.sqrt(np.mean(np.sum((Y[y == 0] - mu0) ** 2, axis=-1))))
        rho1 = float(np.sqrt(np.mean(np.sum((Y[y == 1] - mu1) ** 2, axis=-1))))
        if rho0 <= 0 or rho1 <= 0 or r_ratio_truth <= 0:
            raise ValueError("S4: non-positive radius/r_ratio - radius_ratio_error is undefined.")
        m["radius_ratio_error"] = float(abs(np.log(rho0 / rho1) - np.log(r_ratio_truth)))
    else:
        raise ValueError(f"Unknown scenario '{scenario}'.")
    return m


def _truth_summary(scenario: str, truth: dict[str, Any], y: np.ndarray | None) -> dict[str, np.ndarray]:
    """A compact truth summary for saving to `<scenario>__r<rep>__truth.npz`
    (B.7 item 3) - LARGE (n,n) matrices (S3 'G') are NOT saved, because they
    are exactly reconstructible from the saved coordinates (G = pdist(T)) -
    deterministically, per the project rule 'everything regenerable'."""
    if scenario == "S1":
        return {
            "centroids_in": truth["centroids_in"], "Delta": truth["Delta"], "r": truth["r"],
            "centers_latent": truth["centers_latent"], "sigma_latent": truth["sigma_latent"],
            "y": y.astype(np.int32), "oracle_Y": truth["oracle_Y"].astype(np.float32),
        }
    if scenario == "S2":
        return {
            "u": truth["u"], "Delta": truth["Delta"], "leaf_of_point": truth["leaf_of_point"].astype(np.int32),
            "tree_levels": truth["tree_levels"],
        }
    if scenario == "S3":
        return {"T": truth["T"].astype(np.float32)}
    if scenario == "S4":
        return {"r_ratio": np.array([truth["r_ratio"]], dtype=np.float64), "y": y.astype(np.int32)}
    raise ValueError(f"Unknown scenario '{scenario}'.")


def _run_single(task: dict[str, Any]) -> dict[str, Any]:
    """Computes a single run (scenario, replicate, method) in a separate
    worker process - generates data + truth (deterministically from
    seed_gen), runs the method (or uses oracle_Y/mds_upper_bound_cpcc for
    'oracle_truth'), computes the metric-fidelity metrics + standard
    E1-style metrics (auc_rnx/trustworthiness_k7/continuity_k7/stress_scale_invariant
    against the INPUT matrix X, secondary context)."""
    from src.datasets.synthetic_truth import SCENARIO_GENERATORS
    from src.experiments.exp1_regime_stratified import classify_regime
    from src.methods.common import method_config, to_distance_matrix
    from src.methods.registry import get_method
    from src.sammon import truth_metrics as tm
    from src.sammon.alpha_predict import load_alpha_pred_rule, nn_ratio_k1_from_D, predict_alpha
    from src.sammon.metrics import evaluate

    scenario = task["scenario"]
    replicate = task["replicate"]
    method_name = task["method"]
    seed_gen = task["seed_gen"]
    seed_method = task["seed_method"]
    scenario_cfg = task["cfg_scenario"]
    n_components = task["n_components"]
    n_pairs = task["pointwise_n_pairs"]

    result: dict[str, Any] = {"scenario": scenario, "replicate": replicate, "method": method_name, "seed_gen": seed_gen, "seed_method": seed_method}
    t0 = time.perf_counter()
    try:
        set_seed(seed_method)
        X, y, truth = SCENARIO_GENERATORS[scenario](seed_gen, scenario_cfg)
        n, d = X.shape

        D = to_distance_matrix(X, "vector")
        nn_ratio_k1 = nn_ratio_k1_from_D(D)
        rule = load_alpha_pred_rule()
        regime_frozen = str(classify_regime(np.array([nn_ratio_k1]), rule)[0])

        is_oracle = method_name == "oracle_truth"
        metrics_row: dict[str, float] = {}
        note = ""
        alpha = float("nan")
        selected_hyperparam = ""
        init_label = ""
        Y: np.ndarray | None
        truth_summary: dict[str, np.ndarray] | None = _truth_summary(scenario, truth, y)

        if is_oracle and scenario == "S2":
            metrics_row["cophenetic_pearson"] = tm.mds_upper_bound_cpcc(truth["u"])
            note = "mds_upper_bound_cpcc (K=16 leaves, classic Torgerson MDS directly on u) - a 2D embedding of the ultrametric generally does not exist (B.2 S2)"
            Y = None
        elif is_oracle:
            Y = truth["oracle_Y"]
            init_label = "oracle_projection" if scenario == "S1" else "oracle_intrinsic_coords"
            metrics_row = _scenario_metrics(scenario, Y, y, truth, n_pairs, seed_gen)
            std_metrics = evaluate(X, Y, y, "vector")
            for k in ("auc_rnx", "trustworthiness_k7", "continuity_k7", "stress_scale_invariant"):
                metrics_row[k] = std_metrics.get(k, float("nan"))
        else:
            method = get_method(method_name)
            Y = method.fit_transform(X, "vector", seed=seed_method, n_components=n_components)
            metrics_row = _scenario_metrics(scenario, Y, y, truth, n_pairs, seed_gen)
            std_metrics = evaluate(X, Y, y, "vector")
            for k in ("auc_rnx", "trustworthiness_k7", "continuity_k7", "stress_scale_invariant"):
                metrics_row[k] = std_metrics.get(k, float("nan"))
            selected_hyperparam = str(getattr(method, "last_selected_hyperparam", ""))
            alpha = _extract_alpha(method_name, selected_hyperparam)
            init_label = _init_label(method_name)

        result["status"] = "ok"
        result["error"] = ""
        result["wall_time_sec"] = time.perf_counter() - t0
        result["n"] = float(n)
        result["d"] = float(d)
        result["n_clusters"] = _n_clusters_for_scenario(scenario, truth)
        result["alpha"] = alpha
        result["selected_hyperparam"] = selected_hyperparam
        result["init"] = init_label
        result["nn_ratio_k1"] = nn_ratio_k1
        result["regime_frozen"] = regime_frozen
        result["is_oracle"] = bool(is_oracle)
        result["note"] = note
        result["metrics"] = metrics_row
        result["embedding"] = Y
        result["truth_summary"] = truth_summary
    except Exception as exc:  # intentionally broad catch - one incompatible combination must not stop the whole run
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["wall_time_sec"] = time.perf_counter() - t0
        for k in ("n", "d", "n_clusters", "alpha", "nn_ratio_k1"):
            result[k] = float("nan")
        result["selected_hyperparam"] = ""
        result["init"] = ""
        result["regime_frozen"] = ""
        result["is_oracle"] = method_name == "oracle_truth"
        result["note"] = ""
        result["metrics"] = {}
        result["embedding"] = None
        result["truth_summary"] = None
    return result


def _check_schema(csv_path: Path, columns: list[str]) -> None:
    """Fail-fast schema check BEFORE workers start (same purpose as
    `exp_common.check_results_schema`, but over this experiment's OWN
    column schema - the run key here is not (dataset,method,seed))."""
    if not csv_path.exists():
        return
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        header = next(csv.reader(f), None)
    if header is None:
        return
    missing = [c for c in columns if c not in header]
    if missing:
        raise ValueError(
            f"The schema of the results CSV {csv_path} does not match the current exp9_metric_fidelity "
            f"schema - missing {missing}. Backup + manual migration before resuming (fail-loud, nothing is added automatically)."
        )


def _read_done_keys(csv_path: Path) -> set[tuple[str, int, str]]:
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path, usecols=list(_KEY_COLUMNS))
    return set(zip(df["scenario"].astype(str), df["replicate"].astype(int), df["method"].astype(str)))


def _append_row(csv_path: Path, row: dict[str, Any], columns: list[str]) -> None:
    ensure_dir(csv_path.parent)
    full_row = {c: row.get(c, np.nan if c not in _TEXT_COLUMNS else "") for c in columns}
    file_exists = csv_path.exists()
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not file_exists:
            writer.writeheader()
        writer.writerow(full_row)
        f.flush()
        os.fsync(f.fileno())  # K14-style safeguard - see src/common/checkpoint.py::append_result


def _save_embedding(experiment_name: str, scenario: str, replicate: int, method: str, Y: np.ndarray | None) -> None:
    if Y is None:
        return
    path = _embeddings_dir(experiment_name) / _embedding_filename(scenario, replicate, method)
    ensure_dir(path.parent)
    np.save(path, Y.astype(np.float32))


def _save_truth_once(experiment_name: str, scenario: str, replicate: int, truth_summary: dict[str, np.ndarray] | None) -> None:
    if truth_summary is None:
        return
    path = _embeddings_dir(experiment_name) / _truth_filename(scenario, replicate)
    if path.exists():
        return
    ensure_dir(path.parent)
    np.savez(path, **truth_summary)


def _write_done_file(experiment_name: str, csv_path: Path, logger, extra_info: dict[str, Any]) -> Path:
    n_rows, n_errors = 0, 0
    error_lines: list[str] = []
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        n_rows = len(df)
        if "status" in df.columns:
            err_df = df[df["status"] == "error"]
            n_errors = len(err_df)
            for _, r in err_df.head(50).iterrows():
                error_lines.append(f"  FAILED: scenario={r.get('scenario')} replicate={r.get('replicate')} method={r.get('method')} -> {r.get('error')}")
            if n_errors > 50:
                error_lines.append(f"  ... and {n_errors - 50} more errors (see full CSV: {csv_path}).")

    done_path = get_path("results_data_dir") / f"{experiment_name}_DONE.txt"
    ensure_dir(done_path.parent)
    lines = [
        f"experiment={experiment_name}",
        f"finished_at={datetime.now().isoformat(timespec='seconds')}",
        f"n_rows_csv={n_rows}",
        f"n_errors={n_errors}",
    ]
    for k, v in extra_info.items():
        lines.append(f"{k}={v}")
    lines.extend(error_lines)
    done_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("DONE file written: %s (rows=%d, errors=%d)", done_path, n_rows, n_errors)
    return done_path


def main(mode: str | None = None, n_workers: int | None = None) -> None:
    """`n_workers=None` (default, CLI): the number of workers from
    `parallel.n_workers` (common/config.yaml) - the same existing override
    mechanism as `exp_common.run_experiment_grid(..., n_workers=...)`. An
    explicit value (only for PROGRAMMATIC calls, not from the CLI) serves
    e.g. for a limited verification during a concurrent heavy run of another
    experiment, without hardcoding anything in the production CLI path."""
    if mode is None:
        mode = parse_mode_args("E9 (Q1 step 3): metric fidelity of embeddings against a known latent truth (scenarios S1-S3).")
    from src.experiments.exp_common import keep_system_awake

    logger = get_logger(BASE_EXPERIMENT_NAME, mode=mode)
    cfg = resolve_experiment_config(BASE_EXPERIMENT_NAME, mode)
    experiment_name = resolve_experiment_name(BASE_EXPERIMENT_NAME, mode)
    csv_path = results_csv_path(experiment_name)
    _check_schema(csv_path, COLUMNS)

    tasks = build_tasks(cfg)
    done_keys = _read_done_keys(csv_path)
    todo = [t for t in tasks if (t["scenario"], t["replicate"], t["method"]) not in done_keys]
    n_done = len(tasks) - len(todo)
    logger.info(
        "%s (mode=%s): %d combinations total, %d already done (skipped), %d new to compute.",
        experiment_name, mode, len(tasks), n_done, len(todo),
    )

    n_ok, n_err = 0, 0
    cfg_parallel = load_config()["parallel"]
    every_below = int(cfg_parallel["progress_log_every_below"])
    every = int(cfg_parallel["progress_log_every"])
    n_workers_resolved = resolve_n_workers(n_workers)

    with keep_system_awake():
        t_start = time.perf_counter()
        wall_sum = 0.0
        n_total_todo = len(todo)
        with wall_clock(logger, f"{experiment_name} ({n_total_todo} runs, n_workers={n_workers_resolved})"):
            for i, result in enumerate(run_parallel_map(_run_single, todo, n_workers=n_workers), start=1):
                metrics = result.pop("metrics", {})
                embedding = result.pop("embedding", None)
                truth_summary = result.pop("truth_summary", None)
                row = {**result, **metrics, "experiment": experiment_name}
                _append_row(csv_path, row, COLUMNS)
                _save_embedding(experiment_name, row["scenario"], row["replicate"], row["method"], embedding)
                _save_truth_once(experiment_name, row["scenario"], row["replicate"], truth_summary)

                if row["status"] == "ok":
                    n_ok += 1
                else:
                    n_err += 1
                    logger.warning(
                        "Run failed: scenario=%s replicate=%s method=%s -> %s",
                        row["scenario"], row["replicate"], row["method"], row["error"],
                    )
                wall_sum += float(row["wall_time_sec"])
                log_now = n_total_todo <= every_below or i == 1 or i == n_total_todo or i % every == 0
                if log_now:
                    mean_task = wall_sum / max(i, 1)
                    eta_sec = mean_task * (n_total_todo - i) / max(n_workers_resolved, 1)
                    logger.info(
                        "[%d/%d] %s r=%s %s %s wall=%.1fs | elapsed=%.0fs | mean/task=%.1fs | ETA=%.0fs (n_workers=%d)",
                        i, n_total_todo, row["scenario"], row["replicate"], row["method"], row["status"],
                        float(row["wall_time_sec"]), time.perf_counter() - t_start, mean_task, eta_sec, n_workers_resolved,
                    )

    logger.info("Done. Newly successful: %d, newly failed: %d, skipped: %d.", n_ok, n_err, n_done)
    _write_done_file(experiment_name, csv_path, logger, extra_info={"mode": mode, "n_workers": n_workers_resolved})


if __name__ == "__main__":
    main()
