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
CPU vs GPU dense SMACOF timing benchmark (reserse/2026-09-09_specifikace_metody.md
sections 12.1, 12.5): for n in `sammon.gpu_dense_timing.n_values`
(gaussian_clusters, a single seed) measures wall-clock time and peak memory
of CPU (`smacof_solve`) and GPU (`smacof_solve_gpu`) at a fixed number of
iterations (tol=0.0 - a timing benchmark, not a convergence run). Output:
`results/data/gpu_dense_timing.csv` (n, solver, device, time_s, stress,
peak_mem_bytes).

Run via: `src/run_gpu_dense_timing.bat` or directly
`venv\\python.exe -m src.sammon.gpu_dense_timing`.

NOTE ON FILE LOCATION: the task spec suggested a script
`src/experiments/exp_gpu_dense_timing.py`, but `src/experiments/` was being
concurrently modified by another agent (per the task spec - "another agent
is working CONCURRENTLY on src/experiments/ ... do NOT touch those"). This
script therefore stays in `src/sammon/` (the scope of this task) and uses
its own simple resumable CSV writer directly to
`results/data/gpu_dense_timing.csv` (not `src.common.checkpoint`, which
expects the (experiment, dataset, method, seed) schema, not
(n, solver, device)). Once `src/experiments/` is free, this script can be
taken over/renamed without changing its logic.
"""
from __future__ import annotations

import csv
import time
import tracemalloc
from pathlib import Path

from src.common.config import ensure_dir, get_path, load_config
from src.common.progress import progress_iter
from src.datasets.registry import load_dataset
from src.methods.common import to_distance_matrix
from src.sammon.device import measure_gpu_memory, resolve_device
from src.sammon.init import init_random
from src.sammon.solvers.smacof import smacof_solve, smacof_solve_gpu
from src.sammon.weights import alpha_weights, compute_Z_from_W, estimate_eps_D

CSV_COLUMNS = ["n", "solver", "device", "time_s", "stress", "peak_mem_bytes"]


def _csv_path() -> Path:
    return get_path("results_data_dir") / "gpu_dense_timing.csv"


def _load_done_keys() -> set[tuple[int, str, str]]:
    """Load already-completed (n, solver, device) combinations to enable resume."""
    path = _csv_path()
    if not path.exists():
        return set()
    done: set[tuple[int, str, str]] = set()
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add((int(row["n"]), row["solver"], row["device"]))
    return done


def _append_row(row: dict) -> None:
    path = _csv_path()
    ensure_dir(path.parent)
    write_header = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _prepare_problem(n: int, dataset_name: str, seed: int, alpha: float):
    """Load/generate a dataset of `n` points and compute D, W, Z, Y0 (seeded)."""
    ds = load_dataset(dataset_name, n_samples=n, random_state=seed)
    X = ds.X
    if X.shape[0] != n:
        raise ValueError(f"Loaded dataset '{dataset_name}' has {X.shape[0]} points, expected n={n}.")
    D = to_distance_matrix(X, "vector")
    eps_D = estimate_eps_D(D, k=5, q=0.1, kind="distance")
    W = alpha_weights(D, alpha, eps_D)
    Z = compute_Z_from_W(D, W)
    Y0 = init_random(n, 2, seed=seed, scale=0.5)
    return D, W, Z, Y0


def _run_cpu(n: int, D, W, Z, Y0, cfg: dict) -> dict:
    tracemalloc.start()
    _, hist = smacof_solve(
        D, W, Z, Y0.copy(), max_iter=cfg["max_iter"], tol=cfg["tol"], eps_num=1e-9,
        dense_pinv_threshold=cfg["dense_pinv_threshold"], cg_max_iter=cfg["cg_max_iter"],
        cg_tol=cfg["cg_tol"], reg_rho=1e-8, verbose=False,
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "n": n, "solver": "smacof", "device": "cpu",
        "time_s": hist["time_sec"][-1], "stress": hist["stress"][-1], "peak_mem_bytes": peak,
    }


def _run_gpu(n: int, D, W, Z, Y0, cfg: dict, gd_cfg: dict) -> dict:
    def _call():
        return smacof_solve_gpu(
            D, W, Z, Y0.copy(), max_iter=cfg["max_iter"], tol=cfg["tol"], eps_num=1e-9,
            tile_rows=gd_cfg["tile_rows"], cg_max_iter=cfg["cg_max_iter"], cg_tol=cfg["cg_tol"],
            reg_rho=1e-8, pinv_max_bytes=gd_cfg["pinv_max_bytes"], resident_max_bytes=gd_cfg["resident_max_bytes"],
            const_w_rtol=gd_cfg["const_w_rtol"], inexact_cg=gd_cfg["inexact_cg"],
            cg_tol_factor=gd_cfg["cg_tol_factor"], cg_tol_max=gd_cfg["cg_tol_max"],
            cg_check_every=gd_cfg["cg_check_every"], stress_blowup_factor=gd_cfg["stress_blowup_factor"],
            device="cuda", verbose=False,
        )

    (_, hist), peak = measure_gpu_memory(_call)
    return {
        "n": n, "solver": "smacof", "device": "cuda",
        "time_s": hist["time_sec"][-1], "stress": hist["stress"][-1], "peak_mem_bytes": peak,
    }


def main() -> None:
    cfg_all = load_config()
    cfg = cfg_all["sammon"]["gpu_dense_timing"]
    gd_cfg = cfg_all["sammon"]["gpu_dense"]

    gpu_available = resolve_device("auto") == "cuda"
    devices = ["cpu", "cuda"] if gpu_available else ["cpu"]
    if not gpu_available:
        print("CUDA is not available on this machine - measuring CPU time only (GPU rows are not fabricated).")

    done = _load_done_keys()
    todo = [(n, dev) for n in cfg["n_values"] for dev in devices if (n, "smacof", dev) not in done]
    print(f"{len(done)} combinations already done (skipped), {len(todo)} new to compute.")

    for n, device in progress_iter(todo, desc="gpu_dense_timing"):
        D, W, Z, Y0 = _prepare_problem(n, cfg["dataset"], cfg["seed"], cfg["alpha"])
        if device == "cpu":
            row = _run_cpu(n, D, W, Z, Y0, cfg)
        else:
            row = _run_gpu(n, D, W, Z, Y0, cfg, gd_cfg)
        _append_row(row)
        print(f"n={n} device={device}: time_s={row['time_s']:.3f} stress={row['stress']:.5f} peak_mem_bytes={row['peak_mem_bytes']}")


if __name__ == "__main__":
    main()
