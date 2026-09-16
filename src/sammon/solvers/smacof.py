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
SMACOF majorization for general weights w_ij = D_ij^{-alpha} (the Guttman
transform), see reserse/2026-09-09_specifikace_metody.md, sections 2 and 6.2.

Y_{k+1} = (V + lambda*M)^{-1} (B(Y_k) Y_k + lambda*M*A)

where V is the weighted Laplacian of the weights W (constant across
iterations), B(Y) is the majorization matrix (depends on the current Y_k),
and (lambda, M, A) is an optional temporal anchoring term (section 6.2):
M=diag(m_i), A=the anchor matrix.

For n <= `dense_pinv_threshold`, (V+lambda*M) is inverted once before the
iterations (for lambda=0 we use the pseudoinverse formula for the
Laplacian; for lambda>0 the matrix is positive definite and a direct
inverse can be used). For larger n, the linear system is solved in every
iteration via CG with an implicit matvec (without materializing V),
regularized with `reg_rho/n * 11^T` for lambda=0 (see section 2.3).
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, cg

from src.common.progress import progress_iter
from src.sammon.stress import pairwise_distances, stress_alpha


def _build_B(D: np.ndarray, W: np.ndarray, d: np.ndarray, eps_num: float) -> np.ndarray:
    """Build the majorization matrix B(Y): B_ij = -w_ij D_ij / d_ij (i!=j),
    B_ii = -sum_{j!=i} B_ij (zero row sum)."""
    n = D.shape[0]
    mask = ~np.eye(n, dtype=bool)
    off = np.where(mask, -W * D / d, 0.0)
    B = off.copy()
    np.fill_diagonal(B, -off.sum(axis=1))
    return B


def _precompute_dense_inverse(row_w: np.ndarray, W: np.ndarray, lam: float, mask_anchor: np.ndarray | None, reg_rho: float) -> np.ndarray:
    """Precompute the inverse of (V+lambda*M) for small to medium n.

    lambda=0: uses the pseudoinverse formula for the Laplacian V^+ =
        (V + (1/n)*11^T)^{-1} - (1/n)*11^T (the kernel of V is the constant vector).
    lambda>0: (V+lambda*M) is positive definite (at least one m_i>0), uses
        np.linalg.inv directly.
    """
    n = row_w.shape[0]
    V = np.diag(row_w) - W
    if lam > 0:
        if mask_anchor is None:
            raise ValueError("lam > 0 requires mask_anchor (m_i) to build M.")
        V = V + lam * np.diag(mask_anchor)
        return np.linalg.inv(V)
    ones = np.ones((n, n)) / n
    inv_reg = np.linalg.inv(V + ones)
    return inv_reg - ones


def _make_cg_matvec(row_w: np.ndarray, W: np.ndarray, lam: float, mask_anchor: np.ndarray | None, reg_rho: float):
    """Return the matvec(x) function for the implicit matrix product
    (V+lambda*M)x (plus reg_rho/n*11^T regularization for lambda=0, see
    section 2.3)."""
    n = row_w.shape[0]

    def matvec(x: np.ndarray) -> np.ndarray:
        out = row_w * x - W @ x
        if lam > 0:
            out = out + lam * mask_anchor * x
        else:
            out = out + (reg_rho / n) * np.ones(n) * x.sum()
        return out

    return matvec


def _make_jacobi_preconditioner(row_w: np.ndarray, lam: float, mask_anchor: np.ndarray | None, reg_rho: float, n: int) -> sp.dia_matrix:
    """Jacobi (diagonal) CG preconditioning, added 2026-09-10 (see
    documentation/2026-09-10_oprava_sparse_cg_a_paralelismus.md) - without
    preconditioning, CG on ill-conditioned/near-singular systems (e.g. a
    graph with several weakly connected components) can need orders of
    magnitude more iterations or fail to converge within `cg_max_iter` steps.

    Returns the diagonal of the matrix (V+lambda*M) (plus reg_rho/n*11^T
    regularization for lambda=0), as a sparse diagonal matrix passed to CG
    via the `M=` parameter. diag(V)=row_w, because W always has a zero
    diagonal (see src/sammon/weights.py:alpha_weights)."""
    if lam > 0:
        if mask_anchor is None:
            raise ValueError("lam > 0 requires mask_anchor (m_i) to build the preconditioner.")
        diag_vals = row_w + lam * mask_anchor
    else:
        diag_vals = row_w + reg_rho / n
    return sp.diags(1.0 / diag_vals)


def smacof_solve(
    D: np.ndarray,
    W: np.ndarray,
    Z: float,
    Y0: np.ndarray,
    max_iter: int,
    tol: float,
    eps_num: float,
    dense_pinv_threshold: int,
    cg_max_iter: int,
    cg_tol: float,
    reg_rho: float,
    anchors: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    lam: float = 0.0,
    verbose: bool = False,
) -> tuple[np.ndarray, dict]:
    """SMACOF (Guttman majorization) for general w_ij, with an optional
    temporal anchoring term (section 6.2).

    Parameters:
        D, W: distance and weight matrices (n x n).
        Z: normalization constant of E_alpha.
        Y0: initial configuration (n x p).
        max_iter, tol, eps_num: stopping criteria and numerical stabilization.
        dense_pinv_threshold: below this n threshold, the inverse is
            precomputed; otherwise CG with an implicit matvec is used.
        cg_max_iter, cg_tol: CG solver parameters for large n.
        reg_rho: regularization V+rho/n*11^T for lambda=0 (section 2.3).
        anchors, mask, lam: temporal extension (section 6.2); lam=0 => plain
            SMACOF (default).
        verbose: tqdm progress bar.

    Returns (Y, history): history contains 'stress' (E_alpha per
    iteration), 'sigma' (raw stress before normalization), 'time_sec'
    (cumulative wall-clock time).
    """
    D = np.asarray(D, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    Y = np.asarray(Y0, dtype=np.float64).copy()
    n, p = Y.shape

    if lam > 0 and (anchors is None or mask is None):
        raise ValueError("lam > 0 requires 'anchors' and 'mask' to be provided (temporal anchoring, section 6.2).")
    if lam > 0 and mask is not None and float(np.sum(mask)) == 0.0:
        raise ValueError(
            "lam > 0, but mask.sum() == 0 (no anchoring node) - the system (V+lam*M) is "
            "singular. Call with lam=0.0 if the snapshot has no node in common with the "
            "previous one (see src.sammon.temporal.TemporalSammon.fit)."
        )

    row_w = W.sum(axis=1)
    use_dense = n <= dense_pinv_threshold

    inv_matrix = None
    matvec = None
    precond_M = None
    if use_dense:
        inv_matrix = _precompute_dense_inverse(row_w, W, lam, mask, reg_rho)
    else:
        matvec = _make_cg_matvec(row_w, W, lam, mask, reg_rho)
        precond_M = _make_jacobi_preconditioner(row_w, lam, mask, reg_rho, n)

    stress_hist: list[float] = []
    sigma_hist: list[float] = []
    time_hist: list[float] = []
    t_start = time.perf_counter()

    prev_sigma = None
    n_iter_done = 0
    iterator = progress_iter(range(max_iter), desc="smacof") if verbose else range(max_iter)
    for _ in iterator:
        d = pairwise_distances(Y, eps_num)
        mask_full = ~np.eye(n, dtype=bool)
        sigma = float((mask_full * W * (D - d) ** 2).sum() / 2.0)
        e_alpha = sigma / Z
        stress_hist.append(e_alpha)
        sigma_hist.append(sigma)
        time_hist.append(time.perf_counter() - t_start)
        n_iter_done += 1

        if prev_sigma is not None and (prev_sigma - sigma) < tol * max(prev_sigma, eps_num):
            break
        prev_sigma = sigma

        B = _build_B(D, W, d, eps_num)
        rhs = B @ Y
        if lam > 0:
            rhs = rhs + lam * mask[:, None] * anchors

        if use_dense:
            Y = inv_matrix @ rhs
        else:
            Y_new = np.empty_like(Y)
            op = LinearOperator((n, n), matvec=matvec, dtype=np.float64)
            for q in range(p):
                x0 = Y[:, q]
                sol, info = cg(op, rhs[:, q], x0=x0, rtol=cg_tol, maxiter=cg_max_iter, M=precond_M)
                if info != 0:
                    raise RuntimeError(f"CG did not converge for column {q} (info={info}).")
                Y_new[:, q] = sol
            Y = Y_new

    history = {
        "stress": stress_hist, "sigma": sigma_hist, "time_sec": time_hist,
        "n_iter": n_iter_done, "solver_path": "dense_pinv" if use_dense else "cg",
    }
    return Y, history


# ---------------------------------------------------------------------------
# GPU variant (section 12.1). Reworked 2026-09-13 (documentation/
# 2026-09-13_gpu_pametova_optimalizace.md): the mode is chosen BY BYTES
# (dtype-aware, n^2 * itemsize), not by a fixed n, and the RHS B(Y)Y + sigma
# are ALWAYS computed in row blocks with preallocated buffers
# (`_BlockedRhsSigma`) - no n x n temporary tensor is created within an
# iteration. Modes (`solver_path`):
#   gpu_guttman    : W constant off-diagonal (alpha=0) and lam=0 -> V = c(nI-11^T),
#                    V^+ = (1/(cn))(I - 11^T/n), i.e. the closed form Y = center(B(Y)Y)/(cn);
#                    no CG or factorization, only D (1 n^2 bytes) is resident on the GPU.
#   gpu_dense_pinv : 3 n^2 bytes <= pinv_max_bytes -> the Cholesky factor L L^T =
#                    V + 11^T/n (lam=0; V^+ rhs = L^-T L^-1 rhs - colmean(rhs)), resp.
#                    V + lam*M (lam>0), computed once; then 2 triangular solves
#                    per iteration. Resident D, W, L (3 n^2), the factorization peak
#                    W+V+L = 3 n^2 (D is loaded only after factorization).
#   gpu_cg         : 2 n^2 bytes <= resident_max_bytes -> D, W resident, Jacobi PCG.
#   gpu_tiled_cg   : otherwise -> D, W stay on the CPU (numpy), row blocks are
#                    streamed to the GPU on every RHS and every CG matvec.
# ---------------------------------------------------------------------------

GPU_MODES = ("auto", "guttman", "pinv", "resident_cg", "tiled_cg")


def _row_blocks(n: int, tile_rows: int):
    """Generate (r0, r1) row-block boundaries covering [0, n)."""
    if tile_rows <= 0:
        raise ValueError(f"tile_rows must be positive, got {tile_rows}.")
    r0 = 0
    while r0 < n:
        r1 = min(r0 + tile_rows, n)
        yield r0, r1
        r0 = r1


def _torch_cg(matvec, rhs, x0, tol: float, max_iter: int, precond_diag=None, check_every: int = 1):
    """Batched (P)CG (each RHS column is an independent system with the same
    matvec, solved simultaneously as vectorized operations with a per-column
    alpha/beta step) - see section 2.3. `matvec` takes and returns a tensor
    (n, p). Returns (x, number of CG iterations performed).

    `precond_diag` (2026-09-12, see documentation/2026-09-12_gpu_pcg_a_metriky_velkych_grafu.md):
    optional diagonal of the Jacobi preconditioner (same construction as the
    CPU `_make_jacobi_preconditioner` - diag = row_w + lam*mask_anchor, resp.
    row_w + reg_rho/n for lam=0), shape (n,). If None, unpreconditioned CG
    (M=I). PCG recursion (Shewchuk 1994, sec. B2): alpha = r^T M^{-1} r / p^T A p,
    beta = r_new^T M^{-1} r_new / r^T M^{-1} r, M^{-1} r = r / precond_diag.
    The stopping criterion (relative residual decrease) uses the ACTUAL norm
    ||r|| (not the M-norm), comparable to CPU `scipy.sparse.linalg.cg`.

    `check_every` (2026-09-13): the convergence, finiteness, and degeneracy
    check is performed only every `check_every` iterations (and on the
    last) - each check is a GPU->CPU synchronization, which for small n
    dominates the time of a single CG iteration. Cost: CG may perform up to
    `check_every-1` extra iterations after reaching `tol`, and on numerical
    non-finiteness it returns the last checked (valid) x, i.e. at most
    `check_every` iterations stale.

    Direction degeneracy (2026-09-12 fix, see the document above):
    scale-invariant Rayleigh quotient `|p^T A p| < eps * p^T p` or
    `p^T A p <= 0` ((V+lam*M) is PSD/PD, a negative value is always
    numerical noise - typically float32 on ill-conditioned systems).
    2026-09-13: a degenerate column gets a step of alpha=0 (the column
    "freezes" at its last valid value) instead of the original
    `alpha = rz/eps` for partially degenerate batches, which was a latent
    huge step; if all columns are degenerate, CG terminates."""
    import torch

    if check_every <= 0:
        raise ValueError(f"cg_check_every must be positive, got {check_every}.")
    eps = torch.finfo(rhs.dtype).eps
    x = x0.clone()
    r = rhs - matvec(x)

    def _apply_precond(v):
        if precond_diag is None:
            return v
        return v / precond_diag[:, None]

    z = _apply_precond(r)
    p_dir = z.clone()
    rz_old = (r * z).sum(dim=0)
    rhs_norm = torch.clamp(torch.sqrt((rhs * rhs).sum(dim=0)), min=eps)
    x_last_ok = x
    n_iter = 0
    for it in range(1, max_iter + 1):
        Ap = matvec(p_dir)
        denom = (p_dir * Ap).sum(dim=0)
        p_norm2 = torch.clamp((p_dir * p_dir).sum(dim=0), min=eps)
        degenerate = (torch.abs(denom) < eps * p_norm2) | (denom <= 0)
        denom_safe = torch.where(degenerate, torch.ones_like(denom), denom)
        alpha = torch.where(degenerate, torch.zeros_like(denom), rz_old / denom_safe)
        x = x + alpha[None, :] * p_dir
        r = r - alpha[None, :] * Ap
        r_norm = torch.sqrt((r * r).sum(dim=0))
        z = _apply_precond(r)
        rz_new = (r * z).sum(dim=0)
        beta = rz_new / torch.clamp(rz_old, min=eps)
        p_dir = z + beta[None, :] * p_dir
        rz_old = rz_new
        n_iter = it
        if it % check_every == 0 or it == max_iter:
            # the single GPU->CPU synchronization in this block of iterations
            flags = torch.stack([
                torch.isfinite(x).all(), (r_norm < tol * rhs_norm).all(), degenerate.all(),
            ]).tolist()
            finite, converged, all_degenerate = flags
            if not finite:
                x = x_last_ok  # numerical instability - the last checked valid x
                break
            x_last_ok = x
            if converged or all_degenerate:
                break
    return x, n_iter


def _jacobi_diag_gpu(row_w, lam: float, mask_anchor, reg_rho: float, n: int):
    """GPU counterpart of `_make_jacobi_preconditioner` (CPU) - returns the
    diagonal DIRECTLY (not its reciprocal like the sparse version;
    `_torch_cg` divides directly). diag(V) = row_w (W always has a zero
    diagonal, see src/sammon/weights.py)."""
    if lam > 0:
        if mask_anchor is None:
            raise ValueError("lam > 0 requires mask_anchor (m_i) to build the preconditioner.")
        return row_w + lam * mask_anchor
    return row_w + reg_rho / n


def _cholesky_factor_gpu(W_t, row_w, lam: float, mask_anchor, n: int):
    """Cholesky factor L (L L^T = A, L stored column-major) of the matrix
    A = V + 11^T/n (lam=0: regularization of the Laplacian's kernel as in
    the CPU `_precompute_dense_inverse`, V^+ = A^{-1} - 11^T/n), resp.
    A = V + lam*M (lam>0, positive definite). A is SPD, so Cholesky is
    stable and 2x cheaper than LU; the explicit inverse
    (`torch.cholesky_inverse`/`torch.linalg.inv`) is NOT materialized - it
    would cost another 2 n^2 bytes of peak memory (measured 2026-09-13:
    n=10680 float64 peak 4.56 GB vs. 2.74 GB with just the factor), whereas
    two triangular solves per iteration (`_apply_cholesky_pinv`) cost ~7 ms
    (n=10680 f64) and no extra memory.

    Peak extra memory: V (1 n^2, freed right after factorization) + L (1 n^2).
    The factor is converted to column-major (Fortran) storage so that
    `torch.linalg.solve_triangular` does not need to create a temporary
    n x n copy on every call (measured: with row-major L, +0.91 GB
    transient per iteration)."""
    import torch

    V = W_t.neg()
    V.diagonal().add_(row_w)
    if lam > 0:
        if mask_anchor is None:
            raise ValueError("lam > 0 requires mask_anchor (m_i) to build M.")
        V.diagonal().add_(lam * mask_anchor)
    else:
        V.add_(1.0 / n)
    L, info = torch.linalg.cholesky_ex(V)
    del V
    info_val = int(info.item())
    if info_val != 0:
        raise RuntimeError(
            f"Cholesky factorization (V + regularization) failed (info={info_val}: leading minor "
            f"of order {info_val} is not positive definite) - the weight matrix W is invalid "
            "(negative/zero weights or a disconnected structure). Fail-loud, no silent fallback to LU."
        )
    if not L.mT.is_contiguous():
        L = L.mT.contiguous().mT
    return L


def _apply_cholesky_pinv(L, rhs, lam: float):
    """Y = A^{-1} rhs via two triangular solves; for lam=0, additionally
    - (11^T/n) rhs = - column means of rhs (the Laplacian pseudoinverse, see
    `_cholesky_factor_gpu` and the CPU `_precompute_dense_inverse`)."""
    import torch

    sol = torch.linalg.solve_triangular(L, rhs, upper=False)
    sol = torch.linalg.solve_triangular(L.mT, sol, upper=True)
    if lam == 0:
        sol = sol - rhs.mean(dim=0, keepdim=True)
    return sol


def _dense_matvec_gpu(row_w, W_t, lam: float, mask_anchor, reg_rho: float, n: int):
    """Implicit matvec (V+lambda*M)X over a GPU-resident W. W is symmetric
    (required by CG), so W X = (X^T W)^T - cuBLAS float64 for (n,n)@(n,p)
    with p=2 is an order of magnitude slower than (p,n)@(n,n) (measured
    n=10680 f64: 25.6 ms vs. 1.7 ms, 2026-09-13)."""
    import torch

    def matvec(X):
        WX = torch.matmul(X.T.contiguous(), W_t).T
        out = row_w[:, None] * X - WX
        if lam > 0:
            out = out + lam * mask_anchor[:, None] * X
        else:
            out = out + (reg_rho / n) * X.sum(dim=0, keepdim=True)
        return out

    return matvec


def _tiled_matvec_gpu(W_cpu, row_w, lam: float, mask_anchor, reg_rho: float, n: int,
                      tile_rows: int, device, dtype):
    """Implicit matvec (V+lambda*M)X where W stays on the CPU and row blocks
    (tile_rows x n) are copied into a preallocated GPU buffer - peak extra
    memory O(tile_rows*n). The block-by-X product is done column-by-column
    via torch.mv (a skinny GEMM in cuBLAS float64 is ~10x slower, measured
    2026-09-13)."""
    import torch

    rows_buf = min(tile_rows, n)
    W_buf = torch.empty((rows_buf, n), device=device, dtype=dtype)

    def matvec(X):
        p = X.shape[1]
        XT = X.T.contiguous()
        outT = torch.empty((p, n), device=device, dtype=dtype)
        for r0, r1 in _row_blocks(n, tile_rows):
            rows = r1 - r0
            W_block = W_buf[:rows]
            W_block.copy_(torch.from_numpy(W_cpu[r0:r1]))
            for k in range(p):
                torch.mv(W_block, XT[k], out=outT[k, r0:r1])
        out = row_w[:, None] * X - outT.T
        if lam > 0:
            out = out + lam * mask_anchor[:, None] * X
        else:
            out = out + (reg_rho / n) * X.sum(dim=0, keepdim=True)
        return out

    return matvec


class _BlockedRhsSigma:
    """RHS = B(Y)Y and sigma(Y) = 1/2 sum_{i!=j} w_ij (D_ij - d_ij)^2 in row
    blocks with PREALLOCATED buffers (tile_rows x n) and in-place operations
    - no n x n temporary tensor (2026-09-13, documentation/
    2026-09-13_gpu_pametova_optimalizace.md: the original
    `_dense_rhs_and_sigma_gpu` allocated ~8 n x n tensors per iteration; for
    n=10680 float64 that was an 8-10 GB peak per process -> WDDM paging,
    iterations > 6 min instead of ~0.1 s).

    D/W sources: `streamed=False` -> `D_src`/`W_src` are GPU tensors, the
    block is a slice (a view, no copy); `streamed=True` -> numpy arrays on
    CPU, the block is copied into a preallocated GPU buffer. `w_const` !=
    None means a constant off-diagonal weight (alpha=0) - W is not held at
    all (saves n^2).

    Distances d_ij are computed DIRECTLY from per-component coordinate
    differences (NOT torch.cdist - its matmul mode |x|^2+|y|^2-2xy in
    float32 catastrophically cancels significant digits for nearby points,
    error up to 0.011 at a true distance of 0.0014, see
    documentation/2026-09-11_opravy_resicu.md). sigma is accumulated in
    float64 (`sum(dtype=float64)`) even for float32 data. The product
    B_block @ Y is done column-by-column via torch.mv (see
    `_tiled_matvec_gpu`)."""

    def __init__(self, n: int, p: int, tile_rows: int, device, dtype, eps_num: float,
                 D_src, W_src, w_const: float | None, streamed: bool) -> None:
        import torch

        if tile_rows <= 0:
            raise ValueError(f"tile_rows must be positive, got {tile_rows}.")
        if w_const is None and W_src is None:
            raise ValueError("_BlockedRhsSigma: W_src must be provided when w_const is not set.")
        self.n, self.p, self.tile_rows, self.eps_num = n, p, tile_rows, eps_num
        self.D_src, self.W_src, self.w_const, self.streamed = D_src, W_src, w_const, streamed
        rows = min(tile_rows, n)
        self.buf_d = torch.empty((rows, n), device=device, dtype=dtype)
        self.buf_tmp = torch.empty_like(self.buf_d)
        self.buf_off = torch.empty_like(self.buf_d)
        self.rhs_T = torch.empty((p, n), device=device, dtype=dtype)
        self.idx = torch.arange(rows, device=device)
        self.buf_D = torch.empty_like(self.buf_d) if streamed else None
        self.buf_W = torch.empty_like(self.buf_d) if (streamed and w_const is None) else None
        bufs = [self.buf_d, self.buf_tmp, self.buf_off, self.rhs_T, self.buf_D, self.buf_W]
        self.n_bytes_buffers = int(sum(b.numel() * b.element_size() for b in bufs if b is not None))

    def _fetch(self, r0: int, r1: int):
        """Return (D_block, W_block) for rows [r0, r1); W_block is None when w_const is set."""
        import torch

        rows = r1 - r0
        if self.streamed:
            D_block = self.buf_D[:rows]
            D_block.copy_(torch.from_numpy(self.D_src[r0:r1]))
            W_block = None
            if self.w_const is None:
                W_block = self.buf_W[:rows]
                W_block.copy_(torch.from_numpy(self.W_src[r0:r1]))
            return D_block, W_block
        W_block = None if self.w_const is not None else self.W_src[r0:r1]
        return self.D_src[r0:r1], W_block

    def __call__(self, Y):
        import torch

        n, p = self.n, self.p
        YT = Y.T.contiguous()  # (p, n) - contiguous columns for broadcasting and torch.mv
        sigma_acc = torch.zeros((), device=Y.device, dtype=torch.float64)
        for r0, r1 in _row_blocks(n, self.tile_rows):
            rows = r1 - r0
            D_block, W_block = self._fetch(r0, r1)
            d = self.buf_d[:rows]
            tmp = self.buf_tmp[:rows]
            off = self.buf_off[:rows]
            idx_l = self.idx[:rows]
            idx_g = idx_l + r0

            # d_ij = sqrt(sum_k (Y_ik - Y_jk)^2), component-by-component into the (rows, n) buffer
            d.zero_()
            for k in range(p):
                torch.sub(Y[r0:r1, k:k + 1], YT[k][None, :], out=tmp)
                tmp.mul_(tmp)
                d.add_(tmp)
            d.sqrt_()
            d.clamp_(min=self.eps_num)

            # B_ij = -w_ij D_ij / d_ij (i != j), B_ii = -sum_{j!=i} B_ij
            if W_block is None:
                torch.div(D_block, d, out=off)
                off.mul_(-self.w_const)
            else:
                torch.mul(W_block, D_block, out=off)
                off.div_(d)
                off.neg_()
            off[idx_l, idx_g] = 0.0
            diag = off.sum(dim=1)
            off[idx_l, idx_g] = -diag
            for k in range(p):
                torch.mv(off, YT[k], out=self.rhs_T[k, r0:r1])

            # sigma block: sum_{j!=i} w_ij (D_ij - d_ij)^2
            torch.sub(D_block, d, out=tmp)
            tmp.mul_(tmp)
            if W_block is None:
                tmp.mul_(self.w_const)
            else:
                tmp.mul_(W_block)
            tmp[idx_l, idx_g] = 0.0
            sigma_acc = sigma_acc + tmp.sum(dtype=torch.float64)

        rhs = self.rhs_T.T.contiguous()
        return rhs, float((sigma_acc / 2.0).item())


def _const_offdiag_weight(W_np: np.ndarray, rtol: float, tile_rows: int) -> float | None:
    """Return the constant off-diagonal weight c (= W[0,1]) if
    |w_ij - c| <= rtol*|c| for all i != j, else None (also None for c <= 0,
    when V would be zero/singular, and for rtol < 0 = detection disabled).
    Computed in row blocks on CPU (no n x n copy). For alpha>0, typically
    terminates already in the first block (fast)."""
    n = W_np.shape[0]
    if n < 2 or rtol < 0:
        return None
    c = float(W_np[0, 1])
    if not c > 0.0:
        return None
    tol = rtol * abs(c)
    for r0, r1 in _row_blocks(n, tile_rows):
        dev = np.abs(W_np[r0:r1] - c)
        idx = np.arange(r1 - r0)
        dev[idx, idx + r0] = 0.0
        if float(dev.max()) > tol:
            return None
    return c


def resolve_gpu_dtype(name: str):
    """Convert a string from config.yaml (`sammon.gpu_dense.dtype`, 'float32'
    or 'float64') to a torch dtype - used in `src.sammon.estimator` and
    `src.experiments.exp3_graph_layout._override_gpu_dtype`. Fail-loud for
    an unknown value (no silent fallback to float32)."""
    import torch

    mapping = {"float32": torch.float32, "float64": torch.float64}
    if name not in mapping:
        raise ValueError(f"Unknown sammon.gpu_dense.dtype='{name}' (expected 'float32' or 'float64').")
    return mapping[name]


def select_gpu_mode(n: int, itemsize: int, pinv_max_bytes: float, resident_max_bytes: float,
                    w_const: float | None, lam: float, mode: str) -> tuple[str, bool]:
    """Select the GPU SMACOF mode based on bytes (see the comment at this block).

    Returns (solver_path, resident): resident=True means D (and W) on the
    GPU, False = streaming blocks from CPU. `mode`='auto' decides
    automatically, other values force the mode (`guttman` requires a
    constant W and lam=0)."""
    if mode not in GPU_MODES:
        raise ValueError(f"Unknown GPU mode='{mode}' (expected one of {GPU_MODES}).")
    n2 = float(n) * float(n) * float(itemsize)
    can_guttman = w_const is not None and lam == 0.0
    if mode == "guttman" and not can_guttman:
        raise ValueError(
            "mode='guttman' requires constant off-diagonal weights (alpha=0) and lam=0 - "
            "W is not constant or lam>0."
        )
    if mode == "auto":
        if can_guttman:
            return "gpu_guttman", n2 <= resident_max_bytes
        if 3.0 * n2 <= pinv_max_bytes:
            return "gpu_dense_pinv", True
        if 2.0 * n2 <= resident_max_bytes:
            return "gpu_cg", True
        return "gpu_tiled_cg", False
    if mode == "guttman":
        return "gpu_guttman", n2 <= resident_max_bytes
    if mode == "pinv":
        return "gpu_dense_pinv", True
    if mode == "resident_cg":
        return "gpu_cg", True
    return "gpu_tiled_cg", False


# number of n x n tensors resident on the GPU in each mode (see the comment
# at the GPU block above): guttman only D; pinv D, W, L (the peak during
# factorization W+V+L is also 3 n^2); CG D, W; tiled/streamed none
_N2_TENSORS_PER_PATH = {"gpu_guttman": 1, "gpu_dense_pinv": 3, "gpu_cg": 2, "gpu_tiled_cg": 0}


def blocked_rhs_buffer_bytes(n: int, p: int, tile_rows: int, itemsize: int, streamed: bool, w_const: bool) -> int:
    """Size of the preallocated `_BlockedRhsSigma` buffers in bytes (same
    tensors as in its `__init__`: buf_d, buf_tmp, buf_off (3 x rows x n),
    rhs_T (p x n), plus when streaming buf_D (rows x n), and without a
    constant weight buf_W (rows x n)). Used by `estimate_gpu_peak_bytes`;
    agreement with the actual allocation is tested (tests/test_gpu_preflight.py)."""
    rows = min(int(tile_rows), int(n))
    n_row_bufs = 3 + (1 if streamed else 0) + (1 if (streamed and not w_const) else 0)
    return int(n_row_bufs * rows * n * itemsize + p * n * itemsize)


def estimate_gpu_peak_bytes(n: int, p: int, itemsize: int, tile_rows: int, pinv_max_bytes: float,
                            resident_max_bytes: float, cuda_context_bytes: float,
                            w_const: bool = False, lam: float = 0.0) -> tuple[int, str]:
    """Estimate the peak GPU memory of a SINGLE `smacof_solve_gpu` run
    (bytes) for given n, p, dtype - the mode is chosen with the SAME
    function `select_gpu_mode` as in the solver (no duplicated formulas):
    number of resident n x n tensors per mode (`_N2_TENSORS_PER_PATH`) x
    n^2 x itemsize + `_BlockedRhsSigma` buffers (`blocked_rhs_buffer_bytes`)
    + Y (n x p) + CUDA context/allocator overhead (`cuda_context_bytes`,
    config sammon.gpu_dense.cuda_context_bytes). `w_const=True` models
    alpha=0 (the Guttman form); default False = the worse case (weighted
    stress, alpha>0). Returns (bytes, solver_path). Used by the preflight
    check in exp3_graph_layout.main (documentation/2026-09-13_hardening_behu.md)."""
    if n <= 0 or p <= 0 or itemsize <= 0:
        raise ValueError(f"n={n}, p={p}, itemsize={itemsize} must be positive.")
    solver_path, resident = select_gpu_mode(
        n, itemsize, pinv_max_bytes, resident_max_bytes, w_const=1.0 if w_const else None, lam=lam, mode="auto",
    )
    n2_bytes = int(n) * int(n) * int(itemsize)
    n_resident = _N2_TENSORS_PER_PATH[solver_path] if resident else 0
    buffers = blocked_rhs_buffer_bytes(n, p, tile_rows, itemsize, streamed=not resident, w_const=w_const)
    total = n_resident * n2_bytes + buffers + int(n) * int(p) * int(itemsize) + int(cuda_context_bytes)
    return int(total), solver_path


def smacof_solve_gpu(
    D: np.ndarray,
    W: np.ndarray,
    Z: float,
    Y0: np.ndarray,
    max_iter: int,
    tol: float,
    eps_num: float,
    tile_rows: int,
    cg_max_iter: int,
    cg_tol: float,
    reg_rho: float,
    pinv_max_bytes: float,
    resident_max_bytes: float,
    const_w_rtol: float,
    inexact_cg: bool,
    cg_tol_factor: float,
    cg_tol_max: float,
    cg_check_every: int,
    anchors: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    lam: float = 0.0,
    device: str = "cuda",
    dtype: "object" = None,
    stress_blowup_factor: float = 10.0,
    mode: str = "auto",
    verbose: bool = False,
) -> tuple[np.ndarray, dict]:
    """GPU SMACOF (section 12.1). Same API/semantics as `smacof_solve`;
    modes by bytes, see the comment at this block and config `sammon.gpu_dense`.

    Parameters in addition to the CPU version (all from config `sammon.gpu_dense`):
        tile_rows: size of the row block for RHS/sigma (and streaming).
        pinv_max_bytes: mode 'gpu_dense_pinv' only if 3*n^2*itemsize <= limit.
        resident_max_bytes: D (+W) resident on the GPU only if
            (1 or 2)*n^2*itemsize <= limit, otherwise streaming from CPU
            ('gpu_tiled_cg', resp. streamed guttman).
        const_w_rtol: tolerance for detecting constant off-diagonal weights
            (the closed-form Guttman case); a negative value disables detection.
        inexact_cg, cg_tol_factor, cg_tol_max: inexact majorization -
            relative CG tolerance at the k-th iteration = clip(cg_tol_factor *
            relative sigma decrease in the previous iteration, cg_tol,
            cg_tol_max); cg_tol_max in the first iteration. SMACOF monotonicity
            is preserved: CG starts from the current Y and each of its steps
            decreases the quadratic majorant, so
            sigma(Y_new) <= majorant(Y_new) <= majorant(Y) = sigma(Y).
        cg_check_every: period of synchronized checks in `_torch_cg`.
        mode: 'auto' | 'guttman' | 'pinv' | 'resident_cg' | 'tiled_cg' (force
            a mode for tests and the `src.sammon.gpu_memory_probe` probe).
        dtype: torch dtype (see `resolve_gpu_dtype`), default float32.
        stress_blowup_factor: fail-loud safeguard - RuntimeError if sigma
            grows by more than this factor after a Guttman step (see
            documentation/2026-09-11_opravy_resicu.md).

    Returns (Y as numpy float64, history) - history additionally contains
    'solver_path' in {'gpu_guttman','gpu_dense_pinv','gpu_cg','gpu_tiled_cg'},
    'rhs_mode' in {'resident','streamed'}, 'device', 'setup_sec' (factorization),
    'cg_iters' (per iteration), 'cg_iters_total', 'cg_tol_used', 'n2_bytes'.
    After completion (even after an exception), the GPU cache is freed
    (`torch.cuda.empty_cache`) - ProcessPoolExecutor workers live across
    many jobs."""
    import torch

    try:
        return _smacof_solve_gpu_impl(
            D, W, Z, Y0, max_iter=max_iter, tol=tol, eps_num=eps_num, tile_rows=tile_rows,
            cg_max_iter=cg_max_iter, cg_tol=cg_tol, reg_rho=reg_rho, pinv_max_bytes=pinv_max_bytes,
            resident_max_bytes=resident_max_bytes, const_w_rtol=const_w_rtol, inexact_cg=inexact_cg,
            cg_tol_factor=cg_tol_factor, cg_tol_max=cg_tol_max, cg_check_every=cg_check_every,
            anchors=anchors, mask=mask, lam=lam, device=device, dtype=dtype,
            stress_blowup_factor=stress_blowup_factor, mode=mode, verbose=verbose,
        )
    finally:
        # all GPU tensors are local to _smacof_solve_gpu_impl (freed by refcount
        # on return/exception) - here the allocator cache is returned to the
        # driver (the GPU is shared across several worker processes, see
        # documentation/2026-09-13_gpu_pametova_optimalizace.md)
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.empty_cache()


def _smacof_solve_gpu_impl(
    D, W, Z, Y0, *, max_iter, tol, eps_num, tile_rows, cg_max_iter, cg_tol, reg_rho,
    pinv_max_bytes, resident_max_bytes, const_w_rtol, inexact_cg, cg_tol_factor, cg_tol_max,
    cg_check_every, anchors, mask, lam, device, dtype, stress_blowup_factor, mode, verbose,
) -> tuple[np.ndarray, dict]:
    """Body of `smacof_solve_gpu` (separated out for the try/finally GPU cache release)."""
    import torch

    if lam > 0 and (anchors is None or mask is None):
        raise ValueError("lam > 0 requires 'anchors' and 'mask' to be provided (temporal anchoring, section 6.2).")
    if lam > 0 and mask is not None and float(np.sum(mask)) == 0.0:
        raise ValueError(
            "lam > 0, but mask.sum() == 0 (no anchoring node) - the system (V+lam*M) is "
            "singular. Call with lam=0.0 if the snapshot has no node in common with the "
            "previous one (see src.sammon.temporal.TemporalSammon.fit)."
        )
    if not (0.0 < cg_tol <= cg_tol_max):
        raise ValueError(f"Expected 0 < cg_tol ({cg_tol}) <= cg_tol_max ({cg_tol_max}).")
    if cg_tol_factor <= 0:
        raise ValueError(f"cg_tol_factor must be positive, got {cg_tol_factor}.")

    torch_dtype = dtype if dtype is not None else torch.float32
    dev = torch.device(device)
    # D_np/W_np must have a numpy dtype matching torch_dtype (2026-09-12:
    # previously hardcoded float32 -> silent precision loss when float64 was requested).
    np_dtype = np.float64 if torch_dtype == torch.float64 else np.float32
    D_np = np.ascontiguousarray(np.asarray(D, dtype=np_dtype))
    W_np = np.ascontiguousarray(np.asarray(W, dtype=np_dtype))
    n, p = Y0.shape[0], Y0.shape[1]
    if D_np.shape != (n, n) or W_np.shape != (n, n):
        raise ValueError(f"D {D_np.shape} and W {W_np.shape} must be (n, n) with n={n} per Y0.")
    itemsize = int(torch_dtype.itemsize)

    t_setup = time.perf_counter()
    w_const = _const_offdiag_weight(W_np, const_w_rtol, tile_rows) if lam == 0.0 else None
    solver_path, resident = select_gpu_mode(n, itemsize, pinv_max_bytes, resident_max_bytes, w_const, lam, mode)
    if solver_path != "gpu_guttman":
        w_const = None  # a forced CG/pinv mode works with the full W even for alpha=0

    Y = torch.as_tensor(np.asarray(Y0, dtype=np.float64), device=dev, dtype=torch_dtype).clone()
    mask_t = torch.as_tensor(mask, device=dev, dtype=torch_dtype) if mask is not None else None
    anchors_t = torch.as_tensor(anchors, device=dev, dtype=torch_dtype) if anchors is not None else None

    L = None
    matvec = None
    precond_diag = None
    guttman_scale = None
    if solver_path == "gpu_guttman":
        D_t = torch.as_tensor(D_np, device=dev, dtype=torch_dtype) if resident else None
        rhs_sigma = _BlockedRhsSigma(n, p, tile_rows, dev, torch_dtype, eps_num,
                                     D_src=D_t if resident else D_np, W_src=None,
                                     w_const=w_const, streamed=not resident)
        guttman_scale = 1.0 / (w_const * n)
    elif solver_path == "gpu_dense_pinv":
        # order: W -> factorization (peak W+V+L = 3 n^2) -> only then D
        W_t = torch.as_tensor(W_np, device=dev, dtype=torch_dtype)
        row_w = W_t.sum(dim=1)
        L = _cholesky_factor_gpu(W_t, row_w, lam, mask_t, n)
        D_t = torch.as_tensor(D_np, device=dev, dtype=torch_dtype)
        rhs_sigma = _BlockedRhsSigma(n, p, tile_rows, dev, torch_dtype, eps_num,
                                     D_src=D_t, W_src=W_t, w_const=None, streamed=False)
    elif solver_path == "gpu_cg":
        D_t = torch.as_tensor(D_np, device=dev, dtype=torch_dtype)
        W_t = torch.as_tensor(W_np, device=dev, dtype=torch_dtype)
        row_w = W_t.sum(dim=1)
        matvec = _dense_matvec_gpu(row_w, W_t, lam, mask_t, reg_rho, n)
        precond_diag = _jacobi_diag_gpu(row_w, lam, mask_t, reg_rho, n)
        rhs_sigma = _BlockedRhsSigma(n, p, tile_rows, dev, torch_dtype, eps_num,
                                     D_src=D_t, W_src=W_t, w_const=None, streamed=False)
    else:  # gpu_tiled_cg
        row_w = torch.as_tensor(W_np.sum(axis=1), device=dev, dtype=torch_dtype)
        matvec = _tiled_matvec_gpu(W_np, row_w, lam, mask_t, reg_rho, n, tile_rows, dev, torch_dtype)
        precond_diag = _jacobi_diag_gpu(row_w, lam, mask_t, reg_rho, n)
        rhs_sigma = _BlockedRhsSigma(n, p, tile_rows, dev, torch_dtype, eps_num,
                                     D_src=D_np, W_src=W_np, w_const=None, streamed=True)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    setup_sec = time.perf_counter() - t_setup

    stress_hist: list[float] = []
    sigma_hist: list[float] = []
    time_hist: list[float] = []
    cg_iters_hist: list[int] = []
    cg_tol_hist: list[float] = []
    t_start = time.perf_counter()

    prev_sigma = None
    prev_rel_drop = None
    n_iter_done = 0
    iterator = progress_iter(range(max_iter), desc=f"smacof_{solver_path}") if verbose else range(max_iter)
    for _ in iterator:
        rhs, sigma = rhs_sigma(Y)
        e_alpha = sigma / Z
        if prev_sigma is not None and prev_sigma > eps_num and sigma > stress_blowup_factor * prev_sigma:
            raise RuntimeError(
                f"smacof_solve_gpu (solver_path='{solver_path}'): sigma increased by more than "
                f"{stress_blowup_factor}x between iterations ({prev_sigma:.6g} -> {sigma:.6g}) - "
                "numerical instability of the majorization step, fail-loud instead of silently continuing."
            )
        stress_hist.append(e_alpha)
        sigma_hist.append(sigma)
        time_hist.append(time.perf_counter() - t_start)
        n_iter_done += 1

        if prev_sigma is not None:
            drop = prev_sigma - sigma
            if drop < tol * max(prev_sigma, eps_num):
                break
            prev_rel_drop = drop / max(prev_sigma, eps_num)
        prev_sigma = sigma

        if lam > 0:
            rhs = rhs + lam * mask_t[:, None] * anchors_t

        if solver_path == "gpu_guttman":
            Y = rhs * guttman_scale
            Y = Y - Y.mean(dim=0, keepdim=True)
        elif solver_path == "gpu_dense_pinv":
            Y = _apply_cholesky_pinv(L, rhs, lam)
        else:
            if inexact_cg:
                cg_tol_k = cg_tol_max if prev_rel_drop is None else min(max(cg_tol_factor * prev_rel_drop, cg_tol), cg_tol_max)
            else:
                cg_tol_k = cg_tol
            Y, n_cg = _torch_cg(matvec, rhs, x0=Y, tol=cg_tol_k, max_iter=cg_max_iter,
                                precond_diag=precond_diag, check_every=cg_check_every)
            cg_iters_hist.append(int(n_cg))
            cg_tol_hist.append(float(cg_tol_k))

    history = {
        "stress": stress_hist, "sigma": sigma_hist, "time_sec": time_hist,
        "n_iter": n_iter_done, "solver_path": solver_path, "device": device,
        "rhs_mode": "resident" if resident else "streamed",
        "setup_sec": setup_sec, "cg_iters": cg_iters_hist, "cg_iters_total": int(sum(cg_iters_hist)),
        "cg_tol_used": cg_tol_hist, "n2_bytes": int(n) * int(n) * itemsize,
        "buffer_bytes": rhs_sigma.n_bytes_buffers, "w_const": w_const,
    }
    return Y.detach().cpu().numpy().astype(np.float64), history
