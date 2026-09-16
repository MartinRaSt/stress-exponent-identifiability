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
Pseudo-Newton solver for general w_ij (a generalization of Sammon 1969 to
alpha-Sammon), see reserse/2026-09-09_specifikace_metody.md, section 1.3.

Update: dy_ik = -MF * (dE/dy_ik) / |d2E/dy_ik^2|, a diagonal Hessian.
Returns Y and `history` (stress per iteration and wall-clock time), so
convergence can be compared with the other solvers (SMACOF, SGD).
"""
from __future__ import annotations

import time

import numpy as np

from src.common.progress import progress_iter
from src.sammon.stress import diag_hessian, gradient, stress_alpha


def newton_solve(
    D: np.ndarray,
    W: np.ndarray,
    Z: float,
    Y0: np.ndarray,
    max_iter: int,
    mf: float,
    tol: float,
    eps_num: float,
    step_halving: bool,
    max_halvings: int,
    verbose: bool = False,
) -> tuple[np.ndarray, dict]:
    """Pseudo-Newton solver for minimizing E_alpha(Y) for fixed D, W, Z.

    Parameters:
        D, W: distance and weight matrices (n x n).
        Z: normalization constant of E_alpha.
        Y0: initial configuration (n x p).
        max_iter, mf, tol, eps_num: hyperparameters (see config.yaml sammon.newton).
        step_halving: if True, when stress increases after a step, the step
            is undone and the "magic factor" mf is halved for further
            attempts in this iteration (and subsequent iterations) - see
            documentation/2026-09-11_kontrola_vysledku_plnych_behu.md,
            section 3.6 - the original fixed step of Sammon 1969 has no
            control over stress increase and diverges on some datasets
            (cnae9, isolet).
        max_halvings: maximum number of step halvings within a single
            iteration before an even worsening step is accepted (prevents
            an infinite loop). Note: step-halving only shortens the step in
            the direction `grad/|hess|` - if this direction (due to abs()
            in the diagonal Hessian approximation) is not actually a
            descent direction at the given point, no number of halvings
            will reduce the stress, and after exhausting `max_halvings` a
            slightly worsening step is accepted (verified empirically - see
            tests/test_sammon_solvers.py::test_newton_step_halving_monotone_and_bounded_on_mild_data).
            The guarantee is therefore not "strict monotonicity at any
            cost", but "no LARGE stress increase in a single step" (unlike
            an unbounded fixed step, where stress can explode by orders of
            magnitude).
        verbose: show a tqdm progress bar with the current stress.

    Returns (Y, history) where history contains 'stress' (a list of floats
    per iteration), 'time_sec' (a list of cumulative wall-clock time), and
    'n_halvings' (a list of the number of step halvings performed in a
    given iteration, 0 if step_halving=False or the step did not worsen
    the stress).
    """
    Y = np.asarray(Y0, dtype=np.float64).copy()
    D = np.asarray(D, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)

    stress_hist: list[float] = []
    time_hist: list[float] = []
    n_halvings_hist: list[int] = []
    t_start = time.perf_counter()

    prev_stress = None
    mf_current = mf
    iterator = progress_iter(range(max_iter), desc="newton") if verbose else range(max_iter)
    for _ in iterator:
        stress = stress_alpha(D, Y, W, Z, eps_num)
        stress_hist.append(stress)
        time_hist.append(time.perf_counter() - t_start)

        if prev_stress is not None and abs(prev_stress - stress) < tol * max(prev_stress, eps_num):
            n_halvings_hist.append(0)
            break
        prev_stress = stress

        grad = gradient(D, Y, W, Z, eps_num)
        hess = diag_hessian(D, Y, W, Z, eps_num)
        hess_safe = np.where(np.abs(hess) < eps_num, np.sign(hess) * eps_num + (hess == 0) * eps_num, hess)
        step = grad / np.abs(hess_safe)

        if step_halving:
            n_halve = 0
            mf_try = mf_current
            Y_new = Y - mf_try * step
            stress_new = stress_alpha(D, Y_new, W, Z, eps_num)
            while not (np.isfinite(stress_new) and stress_new <= stress) and n_halve < max_halvings:
                mf_try = mf_try / 2.0
                n_halve += 1
                Y_new = Y - mf_try * step
                stress_new = stress_alpha(D, Y_new, W, Z, eps_num)
            Y = Y_new
            mf_current = mf_try
            n_halvings_hist.append(n_halve)
        else:
            Y = Y - mf * step
            n_halvings_hist.append(0)

    history = {"stress": stress_hist, "time_sec": time_hist, "n_halvings": n_halvings_hist}
    return Y, history


# ---------------------------------------------------------------------------
# GPU variant (section 12.1). For n <= tile_threshold, the torch branches of
# `src.sammon.stress` are used directly (gradient/diag_hessian/stress_alpha
# already have torch dispatch, see stress.py) - vectorized broadcast
# operations over the full (n,n,p) tensor on the GPU. For n > tile_threshold,
# the gradient/Hessian/sigma are computed in row blocks (D, W streamed from
# CPU) - peak extra memory O(tile_rows*n) instead of O(n^2*p), see config
# `sammon.gpu_dense`.
# ---------------------------------------------------------------------------

def _tiled_grad_hessian_stress_gpu(D_cpu: np.ndarray, W_cpu: np.ndarray, Y, Z: float, eps_num: float,
                                    n: int, tile_rows: int, device, dtype):
    import torch

    from src.sammon.solvers.smacof import _row_blocks

    p = Y.shape[1]
    grad = torch.empty((n, p), device=device, dtype=torch.float64)
    hess = torch.empty((n, p), device=device, dtype=torch.float64)
    sigma_acc = torch.zeros((), device=device, dtype=torch.float64)
    for r0, r1 in _row_blocks(n, tile_rows):
        rows = r1 - r0
        D_block = torch.as_tensor(D_cpu[r0:r1], device=device, dtype=dtype)
        W_block = torch.as_tensor(W_cpu[r0:r1], device=device, dtype=dtype)
        Y_block = Y[r0:r1]
        diff = Y_block[:, None, :] - Y[None, :, :]  # (rows, n, p)
        d = torch.clamp(torch.sqrt((diff ** 2).sum(-1)), min=eps_num)  # (rows, n)
        idx_local = torch.arange(rows, device=device)
        idx_global = idx_local + r0
        mask_block = torch.ones((rows, n), dtype=torch.bool, device=device)
        mask_block[idx_local, idx_global] = False

        coeff = torch.where(mask_block, W_block * (D_block - d) / d, torch.zeros_like(D_block))
        grad_block = -2.0 / Z * (coeff[:, :, None] * diff).sum(dim=1)

        bracket = (1.0 - D_block / d)[:, :, None] + D_block[:, :, None] * diff ** 2 / d[:, :, None] ** 3
        w_masked = torch.where(mask_block, W_block, torch.zeros_like(W_block))
        hess_block = 2.0 / Z * (w_masked[:, :, None] * bracket).sum(dim=1)

        grad[r0:r1] = grad_block.double()
        hess[r0:r1] = hess_block.double()
        sigma_acc = sigma_acc + (mask_block * W_block * (D_block - d) ** 2).sum().double()

    stress = float((sigma_acc / 2.0 / Z).item())
    return grad, hess, stress


def _newton_stress_gpu(use_tiled, D_np, W_np, D_t, W_t, Y, Z, eps_num, n, tile_rows, dev, torch_dtype):
    """Helper function - computes only the stress (without gradient/Hessian)
    for a given Y, used in the step-halving line search (see
    `newton_solve_gpu`)."""
    from src.sammon.stress import stress_alpha

    if use_tiled:
        _, _, stress = _tiled_grad_hessian_stress_gpu(D_np, W_np, Y, Z, eps_num, n, tile_rows, dev, torch_dtype)
        return stress
    return stress_alpha(D_t, Y, W_t, Z, eps_num)


def newton_solve_gpu(
    D: np.ndarray,
    W: np.ndarray,
    Z: float,
    Y0: np.ndarray,
    max_iter: int,
    mf: float,
    tol: float,
    eps_num: float,
    tile_threshold: int,
    tile_rows: int,
    step_halving: bool,
    max_halvings: int,
    device: str = "cuda",
    dtype: "object" = None,
    verbose: bool = False,
) -> tuple[np.ndarray, dict]:
    """GPU pseudo-Newton (sections 1.3, 12.1). Same API/semantics as
    `newton_solve` (including step-halving line search, see the comment
    there - same reason: the original fixed step of Sammon 1969 has no
    control over stress increase). For n <= tile_threshold, uses the torch
    branch of `stress.py` (all of D/W on GPU); for n > tile_threshold,
    tiled processing in row blocks (see `_tiled_grad_hessian_stress_gpu`).
    Returns (Y as numpy float64, history) - history additionally contains
    'solver_path', 'device', and 'n_halvings'.
    """
    import torch

    from src.sammon.stress import diag_hessian, gradient, stress_alpha

    torch_dtype = dtype if dtype is not None else torch.float32
    dev = torch.device(device)
    n = D.shape[0]

    D_np = np.asarray(D, dtype=np.float32)
    W_np = np.asarray(W, dtype=np.float32)
    Y = torch.as_tensor(np.asarray(Y0, dtype=np.float64), device=dev, dtype=torch_dtype).clone()

    use_tiled = n > tile_threshold
    if not use_tiled:
        D_t = torch.as_tensor(D_np, device=dev, dtype=torch_dtype)
        W_t = torch.as_tensor(W_np, device=dev, dtype=torch_dtype)
        solver_path = "gpu_dense"
    else:
        D_t = W_t = None
        solver_path = "gpu_tiled"

    stress_hist: list[float] = []
    time_hist: list[float] = []
    n_halvings_hist: list[int] = []
    t_start = time.perf_counter()

    prev_stress = None
    mf_current = mf
    iterator = progress_iter(range(max_iter), desc=f"newton_{solver_path}") if verbose else range(max_iter)
    for _ in iterator:
        if use_tiled:
            grad, hess, stress = _tiled_grad_hessian_stress_gpu(D_np, W_np, Y, Z, eps_num, n, tile_rows, dev, torch_dtype)
        else:
            stress = stress_alpha(D_t, Y, W_t, Z, eps_num)
            grad = gradient(D_t, Y, W_t, Z, eps_num)
            hess = diag_hessian(D_t, Y, W_t, Z, eps_num)

        stress_hist.append(stress)
        time_hist.append(time.perf_counter() - t_start)

        if prev_stress is not None and abs(prev_stress - stress) < tol * max(prev_stress, eps_num):
            n_halvings_hist.append(0)
            break
        prev_stress = stress

        hess_abs = torch.abs(hess)
        hess_safe = torch.where(hess_abs < eps_num, torch.full_like(hess, eps_num), hess_abs)
        step = grad / hess_safe

        if step_halving:
            n_halve = 0
            mf_try = mf_current
            Y_new = (Y.double() - mf_try * step).to(torch_dtype)
            stress_new = _newton_stress_gpu(use_tiled, D_np, W_np, D_t, W_t, Y_new, Z, eps_num, n, tile_rows, dev, torch_dtype)
            while not (np.isfinite(stress_new) and stress_new <= stress) and n_halve < max_halvings:
                mf_try = mf_try / 2.0
                n_halve += 1
                Y_new = (Y.double() - mf_try * step).to(torch_dtype)
                stress_new = _newton_stress_gpu(use_tiled, D_np, W_np, D_t, W_t, Y_new, Z, eps_num, n, tile_rows, dev, torch_dtype)
            Y = Y_new
            mf_current = mf_try
            n_halvings_hist.append(n_halve)
        else:
            Y = (Y.double() - mf * step).to(torch_dtype)
            n_halvings_hist.append(0)

    history = {
        "stress": stress_hist, "time_sec": time_hist, "solver_path": solver_path,
        "device": device, "n_halvings": n_halvings_hist,
    }
    return Y.detach().cpu().numpy().astype(np.float64), history
