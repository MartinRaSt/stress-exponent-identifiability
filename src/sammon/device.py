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
Selection of the compute device (CPU/CUDA) and estimation of the maximum
dense n that fits within the GPU memory budget (spec sections 12.1, 12.5).

Determinism on GPU: `torch.use_deterministic_algorithms(True)` makes
operations without a deterministic implementation RAISE AN ERROR
(RuntimeError) instead of silently returning a different result (section
12.5). Therefore this project always uses the CPU path for "correctness"
numbers (agreement tests, article tables); the GPU is used for the
runtime/speedup benchmark, where the residual nondeterminism (atomic
scatter/index_add_) is quantified via stability across seeds (section 7),
not via bitwise agreement.
"""
from __future__ import annotations

from src.common.config import load_config


def resolve_device(device: str = "auto") -> str:
    """Return the actual device used ('cpu' or 'cuda') based on the request
    and CUDA availability. 'auto' selects 'cuda' if available, else 'cpu'."""
    if device not in ("cpu", "cuda", "auto"):
        raise ValueError(f"Unknown device='{device}' (expected 'cpu'/'cuda'/'auto').")
    if device == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        if device == "cuda":
            raise RuntimeError("device='cuda' requested, but torch is not installed.")
        return "cpu"
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested, but torch.cuda.is_available()==False on this machine.")
        return "cuda"
    # auto
    return "cuda" if torch.cuda.is_available() else "cpu"


def max_dense_n(memory_budget_bytes: float | None = None, bytes_per_element: int | None = None,
                 n_resident_dense_matrices: int | None = None) -> int:
    """Estimate the maximum n for dense alpha-Sammon on GPU given a memory
    budget (section 12.1): n_resident_dense_matrices * n^2 * bytes_per_element
    <= memory_budget_bytes.

    Default values are loaded from config.yaml (sammon.device) if not
    passed explicitly.
    """
    cfg = load_config()["sammon"]["device"]
    budget = memory_budget_bytes if memory_budget_bytes is not None else float(cfg["memory_budget_bytes"])
    bpe = bytes_per_element if bytes_per_element is not None else int(cfg["bytes_per_element"])
    n_mats = n_resident_dense_matrices if n_resident_dense_matrices is not None else int(cfg["n_resident_dense_matrices"])

    if budget <= 0 or bpe <= 0 or n_mats <= 0:
        raise ValueError("memory_budget_bytes, bytes_per_element, and n_resident_dense_matrices must be positive.")

    import math

    n_max = math.isqrt(int(budget / (n_mats * bpe)))
    return int(n_max)


def measure_gpu_memory(func, *args, **kwargs) -> tuple[object, int]:
    """Run `func(*args, **kwargs)` on the GPU and measure the peak allocated
    memory via `torch.cuda.max_memory_allocated()` (section 12.5). Returns
    (result, byte count). Requires CUDA to be available."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("measure_gpu_memory requires CUDA to be available.")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    result = func(*args, **kwargs)
    torch.cuda.synchronize()
    peak_bytes = int(torch.cuda.max_memory_allocated())
    return result, peak_bytes
