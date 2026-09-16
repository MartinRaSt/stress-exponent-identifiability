# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-13
# License: see the LICENSE file in the repository root
"""
A GPU memory and time probe for SMACOF at the real scale of E3
(documentation/2026-09-13_gpu_pametova_optimalizace.md): loads a cached
graph distance matrix (`src/data/graphs/cache/<graph>__<distance>.npy`,
default pgp resistance, n=10680 float64), builds W and the initial
configuration EXACTLY as `src.sammon.estimator.SammonAlpha` does for
kind='distance' (eps_D from `sammon.eps_D`, `alpha_weights`,
`compute_Z_from_W`, init='pca' -> `init_classical_mds`), and runs
`smacof_solve_gpu` (float64, cuda) for the given "alpha:mode:cg" runs. Each
run writes a row to `results/data/quick/gpu_memory_probe.csv` and logs to
`results/logs/gpu_memory_probe_<timestamp>.log`.

Run via: `src\\run_gpu_memory_probe.bat [--runs 0:guttman:config 1:pinv:config ...]
[--max-iter N]` or directly `venv\\python.exe -m src.sammon.gpu_memory_probe ...`.
Default values from `sammon.gpu_memory_probe` in config.yaml. The probe is
intended for short runs (tens of iterations) - the estimate for a full run
(estimator max_iter) is computed from `sec_per_iter`, not measured directly.

Fail-loud: a missing cache, wrong shape/dtype, or mismatch with
`expected_n` is an error (no fallback/synthetic data).
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from src.common.config import ensure_dir, get_path, load_config
from src.common.logging_utils import get_logger
from src.sammon.init import init_classical_mds
from src.sammon.solvers.smacof import GPU_MODES, resolve_gpu_dtype, smacof_solve_gpu
from src.sammon.weights import alpha_weights, compute_Z_from_W, estimate_eps_D

CSV_COLUMNS = [
    "timestamp", "graph", "distance", "n", "dtype", "alpha", "mode", "cg_variant", "solver_path", "rhs_mode",
    "max_iter", "n_iter", "wall_sec", "setup_sec", "sec_per_iter", "cg_iters_total", "inexact_cg",
    "max_memory_allocated_gb", "max_memory_reserved_gb", "n2_gb", "buffer_gb", "sigma_last", "stress_last",
    "est_full_run_sec_300",
]
CG_VARIANTS = ("config", "exact", "inexact")
GB = 1e9


def _csv_path() -> Path:
    return get_path("results_data_dir") / "quick" / "gpu_memory_probe.csv"


def _append_row(row: dict) -> None:
    path = _csv_path()
    ensure_dir(path.parent)
    write_header = not path.exists()
    if not write_header:
        with open(path, "r", encoding="utf-8", newline="") as f:
            header = next(csv.reader(f))
        if header != CSV_COLUMNS:
            raise RuntimeError(
                f"Schema of {path} ({header}) does not match CSV_COLUMNS ({CSV_COLUMNS}) - rename/back up "
                "the old file; migration is not fabricated."
            )
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def parse_run_spec(spec: str) -> tuple[float, str, str]:
    """'alpha:mode[:cg]' -> (alpha, mode, cg_variant); cg in {config, exact, inexact}."""
    parts = spec.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"Run '{spec}' must have the form alpha:mode[:cg] (cg = {CG_VARIANTS}).")
    alpha = float(parts[0])
    mode = parts[1]
    cg_variant = parts[2] if len(parts) == 3 else "config"
    if alpha < 0:
        raise ValueError(f"alpha must be >= 0, got {alpha} in '{spec}'.")
    if mode not in GPU_MODES:
        raise ValueError(f"mode '{mode}' in '{spec}' is not in {GPU_MODES}.")
    if cg_variant not in CG_VARIANTS:
        raise ValueError(f"cg '{cg_variant}' in '{spec}' is not in {CG_VARIANTS}.")
    return alpha, mode, cg_variant


def load_cached_distance(graph: str, distance: str, expected_n: int) -> np.ndarray:
    """Load `<graphs_cache_dir>/cache/<graph>__<distance>.npy` and fail-loud
    check its shape (expected_n x expected_n), dtype float64, symmetry, and
    zero diagonal."""
    path = get_path("graphs_cache_dir") / "cache" / f"{graph}__{distance}.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"Distance matrix cache {path} does not exist - run E3 first (run_exp3_graph_layout.bat), "
            "which creates it. The probe does not fabricate data."
        )
    D = np.load(path)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"{path}: expected a square matrix, got shape {D.shape}.")
    if D.shape[0] != expected_n:
        raise ValueError(f"{path}: n={D.shape[0]} does not match sammon.gpu_memory_probe.expected_n={expected_n}.")
    if D.dtype != np.float64:
        raise ValueError(f"{path}: expected dtype float64, got {D.dtype}.")
    if not np.allclose(np.diagonal(D), 0.0):
        raise ValueError(f"{path}: the distance matrix diagonal is not zero.")
    if not np.allclose(D, D.T, rtol=0.0, atol=1e-9 * float(D.max())):
        raise ValueError(f"{path}: the distance matrix is not symmetric.")
    return D


def run_probe(runs: list[str], max_iter: int, logger) -> list[dict]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("gpu_memory_probe requires CUDA to be available (torch.cuda.is_available()==False).")
    cfg_all = load_config()
    scfg = cfg_all["sammon"]
    pcfg = scfg["gpu_memory_probe"]
    gd_cfg = scfg["gpu_dense"]
    est_cfg = scfg["estimator"]
    graph, distance, expected_n = str(pcfg["graph"]), str(pcfg["distance"]), int(pcfg["expected_n"])
    n_components, seed = int(pcfg["n_components"]), int(pcfg["seed"])
    tol = float(est_cfg["tol"])
    full_iters = int(est_cfg["max_iter"])
    # large E3 graphs run in float64 (exp3_graph_layout.large_graph_gpu_dtype)
    dtype_name = "float64"
    torch_dtype = resolve_gpu_dtype(dtype_name)

    t0 = time.perf_counter()
    D = load_cached_distance(graph, distance, expected_n)
    n = D.shape[0]
    logger.info("Loaded D %s (%s, n=%d) in %.1f s", graph, distance, n, time.perf_counter() - t0)

    t0 = time.perf_counter()
    eps_D = estimate_eps_D(D, k=scfg["eps_D"]["k"], q=scfg["eps_D"]["q"], kind="distance")
    Y0 = init_classical_mds(D, n_components, seed)
    logger.info("eps_D=%.6g, init_classical_mds (same as the estimator for kind='distance') in %.1f s", eps_D, time.perf_counter() - t0)

    rows: list[dict] = []
    alpha_cache: dict[float, tuple[np.ndarray, float]] = {}
    for spec in runs:
        alpha, mode, cg_variant = parse_run_spec(spec)
        if alpha not in alpha_cache:
            W = alpha_weights(D, alpha, eps_D)
            alpha_cache[alpha] = (W, compute_Z_from_W(D, W))
        W, Z = alpha_cache[alpha]
        inexact = bool(gd_cfg["inexact_cg"]) if cg_variant == "config" else (cg_variant == "inexact")

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        logger.info("Run %s: alpha=%g mode=%s inexact_cg=%s max_iter=%d ...", spec, alpha, mode, inexact, max_iter)
        t_run = time.perf_counter()
        try:
            _, hist = smacof_solve_gpu(
                D, W, Z, Y0, max_iter=max_iter, tol=tol, eps_num=scfg["eps_num"],
                tile_rows=gd_cfg["tile_rows"], cg_max_iter=gd_cfg["cg_max_iter"], cg_tol=gd_cfg["cg_tol"],
                reg_rho=gd_cfg["reg_rho"], pinv_max_bytes=gd_cfg["pinv_max_bytes"],
                resident_max_bytes=gd_cfg["resident_max_bytes"], const_w_rtol=gd_cfg["const_w_rtol"],
                inexact_cg=inexact, cg_tol_factor=gd_cfg["cg_tol_factor"], cg_tol_max=gd_cfg["cg_tol_max"],
                cg_check_every=gd_cfg["cg_check_every"], stress_blowup_factor=gd_cfg["stress_blowup_factor"],
                dtype=torch_dtype, device="cuda", mode=mode, verbose=False,
            )
        except Exception as exc:
            logger.error("Run %s FAILED: %s: %s", spec, type(exc).__name__, exc)
            raise
        torch.cuda.synchronize()
        wall = time.perf_counter() - t_run
        n_iter = int(hist["n_iter"])
        iter_sec = (wall - hist["setup_sec"]) / max(n_iter, 1)
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "graph": graph, "distance": distance,
            "n": n, "dtype": dtype_name, "alpha": alpha, "mode": mode, "cg_variant": cg_variant,
            "solver_path": hist["solver_path"], "rhs_mode": hist["rhs_mode"], "max_iter": max_iter,
            "n_iter": n_iter, "wall_sec": round(wall, 3), "setup_sec": round(hist["setup_sec"], 3),
            "sec_per_iter": round(iter_sec, 4), "cg_iters_total": hist["cg_iters_total"], "inexact_cg": inexact,
            "max_memory_allocated_gb": round(torch.cuda.max_memory_allocated() / GB, 3),
            "max_memory_reserved_gb": round(torch.cuda.max_memory_reserved() / GB, 3),
            "n2_gb": round(hist["n2_bytes"] / GB, 3), "buffer_gb": round(hist["buffer_bytes"] / GB, 3),
            "sigma_last": hist["sigma"][-1], "stress_last": hist["stress"][-1],
            "est_full_run_sec_300": round(hist["setup_sec"] + full_iters * iter_sec, 1),
        }
        _append_row(row)
        rows.append(row)
        logger.info(
            "  -> path=%s rhs=%s n_iter=%d wall=%.2fs setup=%.2fs sec/iter=%.4f cg_total=%d "
            "mem alloc=%.2f GB reserved=%.2f GB sigma=%.6g estimated %d iter: %.0f s",
            row["solver_path"], row["rhs_mode"], n_iter, wall, row["setup_sec"], iter_sec, row["cg_iters_total"],
            row["max_memory_allocated_gb"], row["max_memory_reserved_gb"], row["sigma_last"], full_iters,
            row["est_full_run_sec_300"],
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    pcfg = load_config()["sammon"]["gpu_memory_probe"]
    parser = argparse.ArgumentParser(description="GPU memory/time probe for SMACOF at real scale (pgp, float64).")
    parser.add_argument("--runs", nargs="+", default=list(pcfg["runs"]),
                        help="runs 'alpha:mode[:cg]', mode in %s, cg in %s" % (GPU_MODES, CG_VARIANTS))
    parser.add_argument("--max-iter", type=int, default=int(pcfg["max_iter"]))
    args = parser.parse_args(argv)
    if args.max_iter <= 0:
        raise ValueError(f"--max-iter must be positive, got {args.max_iter}.")
    for spec in args.runs:
        parse_run_spec(spec)  # fail-loud validation before expensive data loading

    logger = get_logger("gpu_memory_probe")
    from src.experiments.exp_common import keep_system_awake

    with keep_system_awake():
        rows = run_probe(args.runs, args.max_iter, logger)
    logger.info("Done: %d runs, CSV %s", len(rows), _csv_path())
    return 0


if __name__ == "__main__":
    sys.exit(main())
