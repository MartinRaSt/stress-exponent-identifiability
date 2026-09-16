# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-09
# License: see the LICENSE file in the repository root
"""Tests for sections 10.6 and 12.4/12.5: GPU vs. CPU agreement. `stress`/`gradient`
(numpy vs. torch, same sample data) -> agreement at float64 precision level;
SGD (stochastic, GPU uses a different but equivalent sampling scheme for
performance reasons, see the test below) -> qualitative convergence check.

2026-09-13 (documentation/2026-09-13_gpu_pametova_optimalizace.md): GPU SMACOF
selects the mode by byte budget (`pinv_max_bytes`, `resident_max_bytes`), has a
closed-form Guttman transform for alpha=0 (`gpu_guttman`), blocked RHS/sigma
without an n x n temporary, and inexact CG - the tests below cover all these parts.

The whole module is skipped if CUDA is not available on the machine."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA is not available on this machine - test_sammon_gpu.py is being skipped.", allow_module_level=True)

from scipy.spatial import procrustes

from src.common.config import load_config
from src.sammon.init import init_random
from src.sammon.solvers.newton import newton_solve, newton_solve_gpu
from src.sammon.solvers.sgd import sgd_solve, sgd_solve_gpu
from src.sammon.solvers.smacof import (
    _BlockedRhsSigma,
    _build_B,
    _torch_cg,
    resolve_gpu_dtype,
    select_gpu_mode,
    smacof_solve,
    smacof_solve_gpu,
)
from src.sammon.stress import gradient as gradient_dispatch
from src.sammon.stress import pairwise_distances, stress_alpha
from src.sammon.weights import alpha_weights, compute_Z_from_W, estimate_eps_D


def _random_D(seed: int, n: int = 40, d: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return squareform(pdist(rng.normal(size=(n, d))))


def _gpu_kwargs(**overrides) -> dict:
    """Parameters of `smacof_solve_gpu` from the config `sammon.gpu_dense` (no
    magic numbers in tests), with the option to override individual keys (e.g.
    forcing a mode via byte limits). Byte limits 'everything fits' = 1e12 (the
    test works with n on the order of hundreds, n^2*8 ~ 1e5-1e6 B), 'nothing
    fits' = 0."""
    gd = load_config()["sammon"]["gpu_dense"]
    kwargs = dict(
        tile_rows=gd["tile_rows"], cg_max_iter=gd["cg_max_iter"], cg_tol=gd["cg_tol"], reg_rho=gd["reg_rho"],
        pinv_max_bytes=gd["pinv_max_bytes"], resident_max_bytes=gd["resident_max_bytes"],
        const_w_rtol=gd["const_w_rtol"], inexact_cg=False, cg_tol_factor=gd["cg_tol_factor"],
        cg_tol_max=gd["cg_tol_max"], cg_check_every=gd["cg_check_every"],
        stress_blowup_factor=gd["stress_blowup_factor"], device="cuda", verbose=False,
    )
    kwargs.update(overrides)
    return kwargs


ALL_FITS = dict(pinv_max_bytes=1e12, resident_max_bytes=1e12)


def test_stress_and_gradient_numpy_vs_torch_cpu_and_cuda() -> None:
    """The numpy and torch (both CPU and CUDA) branches of `stress.py` must give
    identical stress and gradient values for the same input data (float64
    everywhere so the difference stays below numerical noise)."""
    D = _random_D(seed=0, n=25)
    eps_D = estimate_eps_D(D, k=5, q=0.1, kind="distance")
    W = alpha_weights(D, 1.0, eps_D)
    Z = compute_Z_from_W(D, W)
    Y = init_random(D.shape[0], 2, seed=1, scale=0.5)
    eps_num = 1e-9

    e_np = stress_alpha(D, Y, W, Z, eps_num)
    g_np = gradient_dispatch(D, Y, W, Z, eps_num)

    for device in ("cpu", "cuda"):
        D_t = torch.tensor(D, device=device, dtype=torch.float64)
        W_t = torch.tensor(W, device=device, dtype=torch.float64)
        Y_t = torch.tensor(Y, device=device, dtype=torch.float64)
        e_t = stress_alpha(D_t, Y_t, W_t, Z, eps_num)
        g_t = gradient_dispatch(D_t, Y_t, W_t, Z, eps_num).cpu().numpy()

        assert abs(e_np - e_t) < 1e-8, f"device={device}: stress difference {abs(e_np - e_t)}"
        np.testing.assert_allclose(g_np, g_t, atol=1e-8, rtol=1e-6)


def test_sgd_gpu_round_robin_matches_cpu_stabilized() -> None:
    """Test 10.6 (modified, see projectstate.md 2026-09-09 - GPU SGD
    performance): the original requirement "GPU round-robin vs. CPU sequential
    - relative stress tolerance 1e-3" assumed that both paths sample the SAME
    pairs (just in a different processing order). For performance reasons
    (greedy node-disjoint edge coloring over sampled pairs scaled with the
    maximum node degree -> at n~1000 the GPU was measured to be 30-60x SLOWER
    than the CPU) the GPU path now uses a different, but equivalent, sampling
    scheme with a FIXED number of rounds (C random perfect matchings per epoch
    instead of C*n randomly chosen pairs) - see the comment on `sgd_solve_gpu`.
    As a result, CPU and GPU minimize the same objective (E_alpha) but do not
    visit an identical sequence of pairs, so we test QUALITATIVE agreement
    (both substantially reduce stress relative to the initial configuration and
    end up in a comparable range), not bitwise agreement.
    """
    D = _random_D(seed=0, n=50)
    Y0 = init_random(D.shape[0], 2, seed=1, scale=0.5)
    common_kwargs = dict(
        alpha=1.0, epochs=60, mu_max=0.5, pairs_per_node=60, eps_anneal=0.01,
        eps_num=1e-9, seed=0, variant="stabilized",
    )

    Y_cpu, hist_cpu = sgd_solve(D, Y0=Y0.copy(), **common_kwargs, verbose=False)
    Y_gpu, hist_gpu = sgd_solve_gpu(
        D, Y0=Y0.copy(), **common_kwargs, batch_mode="round_robin",
        device="cuda", dtype=torch.float64, log_every=1, verbose=False,
    )

    stress_init = hist_cpu["stress"][0]
    assert hist_cpu["stress"][-1] < 0.6 * stress_init, "CPU SGD did not improve substantially relative to the initial configuration."
    assert hist_gpu["stress"][-1] < 0.6 * stress_init, "GPU SGD did not improve substantially relative to the initial configuration."

    rel_diff = abs(hist_cpu["stress"][-1] - hist_gpu["stress"][-1]) / hist_cpu["stress"][-1]
    assert rel_diff < 0.2, f"Relative difference of the final CPU vs GPU stress {rel_diff} >= 0.2 (different but equivalent sampling scheme)."


@pytest.mark.parametrize("mode", ["pinv", "resident_cg", "tiled_cg"])
def test_smacof_gpu_matches_cpu_dense(mode: str) -> None:
    """Test for section 10.6/12.4: GPU dense SMACOF (all three internal modes for
    alpha>0 - Cholesky V^+, resident PCG, tiled/streamed PCG, forced via the
    `mode` parameter) must give Procrustes RMSE <= 1e-4 relative to CPU SMACOF
    on small n (deterministic path, float32 vs float64 tolerance per section
    12.4). Default dtype float32 (as in E1)."""
    D = _random_D(seed=0, n=80)
    eps_D = estimate_eps_D(D, k=5, q=0.1, kind="distance")
    W = alpha_weights(D, 1.0, eps_D)
    Z = compute_Z_from_W(D, W)
    Y0 = init_random(D.shape[0], 2, seed=1, scale=0.5)

    Y_cpu, _ = smacof_solve(
        D, W, Z, Y0.copy(), max_iter=200, tol=1e-8, eps_num=1e-9,
        dense_pinv_threshold=5000, cg_max_iter=500, cg_tol=1e-8, reg_rho=1e-8, verbose=False,
    )
    Y_gpu, hist_gpu = smacof_solve_gpu(
        D, W, Z, Y0.copy(), max_iter=200, tol=1e-8, eps_num=1e-9,
        **_gpu_kwargs(mode=mode, tile_rows=32, cg_max_iter=500, cg_tol=1e-6, reg_rho=1e-8, **ALL_FITS),
    )

    expected_path = {"pinv": "gpu_dense_pinv", "resident_cg": "gpu_cg", "tiled_cg": "gpu_tiled_cg"}[mode]
    assert hist_gpu["solver_path"] == expected_path
    assert hist_gpu["rhs_mode"] == ("streamed" if mode == "tiled_cg" else "resident")

    _, _, disparity = procrustes(Y_cpu, Y_gpu)
    scale = float(np.linalg.norm(Y_cpu - Y_cpu.mean(axis=0)))
    assert disparity < 1e-4 * scale, f"Procrustes disparity {disparity} >= 1e-4*scale for mode {mode}."


@pytest.mark.parametrize("n", [2000, 3000])
def test_smacof_gpu_tiled_float32_matches_cpu_large_n(n: int) -> None:
    """Regression test for section 3.2/5 (documentation/2026-09-11_kontrola_vysledku_plnych_behu.md
    and 2026-09-11_opravy_resicu.md): `gpu_tiled_cg` in the default float32 used
    `torch.cdist` (matmul-based Euclidean distance), which for nearby points
    catastrophically cancels significant digits in float32 (error up to 0.011
    measured at an actual distance of 0.0014) - the RHS majorization then
    contained the term -W*D/d with an incorrect (occasionally near-zero) d,
    which led to divergence as early as the 1st-2nd Guttman step (stress > 400
    instead of ~0.03) for n>=tile_threshold. The fix (computing the distance
    directly from the difference, now in `_BlockedRhsSigma`) must give the same
    stress as the CPU (float64) within a tolerance of 1e-4, starting from
    n=2000 (the error was observed from n=5000 onward, but it can be reproduced
    at a smaller n as well by forcing `mode='tiled_cg'` - see
    `documentation/2026-09-11_opravy_resicu.md`)."""
    from src.datasets.registry import load_dataset

    ds = load_dataset("gaussian_clusters", n_samples=n)
    D = squareform(pdist(ds.X))
    eps_D = estimate_eps_D(D, k=5, q=0.0, kind="distance")
    W = alpha_weights(D, 1.0, eps_D)
    Z = compute_Z_from_W(D, W)
    Y0 = init_random(D.shape[0], 2, seed=0, scale=0.5)

    Y_cpu, hist_cpu = smacof_solve(
        D, W, Z, Y0.copy(), max_iter=150, tol=1e-4, eps_num=1e-9,
        dense_pinv_threshold=5000, cg_max_iter=500, cg_tol=1e-6, reg_rho=1e-8, verbose=False,
    )

    # tiled (streamed) branch forced via `mode`; default dtype (float32,
    # section 12.1) - this is exactly where the bug originally manifested
    Y_gpu, hist_gpu = smacof_solve_gpu(
        D, W, Z, Y0.copy(), max_iter=150, tol=1e-4, eps_num=1e-9,
        **_gpu_kwargs(mode="tiled_cg", tile_rows=1024, cg_max_iter=500, cg_tol=1e-6, reg_rho=1e-8),
    )
    assert hist_gpu["solver_path"] == "gpu_tiled_cg"
    assert hist_gpu["rhs_mode"] == "streamed"

    stress_cpu = hist_cpu["stress"][-1]
    stress_gpu = hist_gpu["stress"][-1]
    rel_diff = abs(stress_cpu - stress_gpu) / stress_cpu
    assert rel_diff < 1e-4, (
        f"n={n}: gpu_tiled_cg float32 stress {stress_gpu} differs from the CPU float64 stress {stress_cpu} "
        f"by {rel_diff} >= 1e-4."
    )
    assert np.abs(Y_gpu).max() < 100.0, f"n={n}: |Y_gpu| should not diverge, max={np.abs(Y_gpu).max()}."


@pytest.mark.parametrize("solver_path_regime", ["dense", "tiled"])
def test_newton_gpu_converges_like_cpu(solver_path_regime: str) -> None:
    """Test for section 10.6/12.4 for pseudo-Newton: GPU (both dense and tiled
    mode) must substantially reduce stress relative to the initial
    configuration, comparably to CPU. Exact agreement of Y (Procrustes) is NOT
    required - pseudo-Newton with a fixed step is known to be chaotic dynamics
    sensitive to the order of floating-point operations (see
    `tests/test_sammon_solvers.py::test_newton_alpha1_matches_reference_sammon_classic`
    and `documentation/2026-09-09_sammon_core.md`); float32 GPU vs float64 CPU
    can therefore end up at a different but qualitatively comparable point."""
    D = _random_D(seed=0, n=20)
    eps_D = estimate_eps_D(D, k=5, q=0.1, kind="distance")
    W = alpha_weights(D, 1.0, eps_D)
    Z = compute_Z_from_W(D, W)
    Y0 = init_random(D.shape[0], 2, seed=1, scale=0.5)

    # step_halving=False: this test only verifies CPU/GPU agreement of the
    # computation core (section 10.6/12.4), not the step-halving safeguard (see
    # tests/test_sammon_solvers.py::test_newton_step_halving_*) - with a small
    # number of iterations (40) the convergence slowdown from step halving
    # could make the test needlessly noisy.
    Y_cpu, hist_cpu = newton_solve(
        D, W, Z, Y0.copy(), max_iter=40, mf=0.3, tol=1e-8, eps_num=1e-9,
        step_halving=False, max_halvings=10, verbose=False,
    )

    tile_kwargs = {"dense": dict(tile_threshold=12000, tile_rows=32), "tiled": dict(tile_threshold=8, tile_rows=4)}[solver_path_regime]
    Y_gpu, hist_gpu = newton_solve_gpu(
        D, W, Z, Y0.copy(), max_iter=40, mf=0.3, tol=1e-8, eps_num=1e-9,
        step_halving=False, max_halvings=10, device="cuda", verbose=False,
        **tile_kwargs,
    )

    stress_init = hist_cpu["stress"][0]
    assert hist_cpu["stress"][-1] < 0.5 * stress_init, "CPU Newton did not improve substantially relative to the initial configuration."
    assert hist_gpu["stress"][-1] < 0.5 * stress_init, "GPU Newton did not improve substantially relative to the initial configuration."
    assert np.isfinite(Y_gpu).all()


# ---------------------------------------------------------------------------
# 2026-09-12 (documentation/2026-09-12_gpu_pcg_a_metriky_velkych_grafu.md):
# Jacobi preconditioning for GPU CG (_torch_cg precond_diag) and the float64
# GPU branch.
# ---------------------------------------------------------------------------

def test_torch_cg_preconditioned_matches_unpreconditioned_and_direct_solve() -> None:
    """`_torch_cg` with `precond_diag` (Jacobi PCG) must converge to the same
    solution as the direct solver (`torch.linalg.solve`) on a random SPD
    system, and with `precond_diag=None` behave exactly like the original
    (unpreconditioned) CG (regression - see `precond_diag=None` in the
    `_torch_cg` docstring). Since 2026-09-13, `_torch_cg` returns (x,
    iteration_count) and checks convergence only every `check_every`
    iterations - the result must be independent of `check_every` (up to at
    most check_every-1 extra iterations)."""
    torch.manual_seed(0)
    n = 60
    A = torch.randn(n, n, dtype=torch.float64)
    A = A @ A.T + n * torch.eye(n, dtype=torch.float64)  # SPD, well-conditioned
    rhs = torch.randn(n, 3, dtype=torch.float64)
    x0 = torch.zeros(n, 3, dtype=torch.float64)
    diag = torch.diagonal(A).clone()

    def matvec(x):
        return A @ x

    x_true = torch.linalg.solve(A, rhs)
    x_nopre, it_nopre = _torch_cg(matvec, rhs, x0, tol=1e-12, max_iter=300)
    x_pre, it_pre = _torch_cg(matvec, rhs, x0, tol=1e-12, max_iter=300, precond_diag=diag)
    x_pre4, it_pre4 = _torch_cg(matvec, rhs, x0, tol=1e-12, max_iter=300, precond_diag=diag, check_every=4)

    assert (x_nopre - x_true).abs().max().item() < 1e-8, "Unpreconditioned CG does not match the direct solver."
    assert (x_pre - x_true).abs().max().item() < 1e-6, "Preconditioned (Jacobi) CG does not match the direct solver."
    assert (x_pre4 - x_true).abs().max().item() < 1e-6, "PCG with check_every=4 does not match the direct solver."
    assert 0 < it_nopre <= 300 and 0 < it_pre <= 300
    assert it_pre <= it_pre4 <= it_pre + 3, f"check_every=4 may add at most 3 iterations ({it_pre} -> {it_pre4})."


def test_smacof_gpu_cg_weighted_alpha_matches_cpu_float64() -> None:
    """Regression test for K8 finding (documentation/2026-09-12_s2_koder_b.md,
    appendix): GPU CG SMACOF (`solver_path='gpu_cg'`) for WEIGHTED stress
    (alpha>0) was numerically unstable in float32 (RuntimeError stress_blowup)
    - directly verified also for alpha=0 on real graphs (see the document
    above, near-null "translation" mode of the Laplacian with reg_rho=1e-8 ->
    float32 catastrophic cancellation of significant digits). The float64 GPU
    branch (`dtype=torch.float64`, `resolve_gpu_dtype('float64')`) must, on the
    same task (mode 'resident_cg' forced via `mode`), give a stress consistent
    with CPU (float64) at a relative tolerance of 1e-3 (project requirement),
    WITHOUT a stress_blowup exception."""
    from src.datasets.registry import load_dataset

    n = 2500
    alpha = 1.0
    ds = load_dataset("gaussian_clusters", n_samples=n)
    D = squareform(pdist(ds.X))
    eps_D = estimate_eps_D(D, k=5, q=0.1, kind="distance")
    W = alpha_weights(D, alpha, eps_D)
    Z = compute_Z_from_W(D, W)
    Y0 = init_random(D.shape[0], 2, seed=0, scale=0.5)

    Y_cpu, hist_cpu = smacof_solve(
        D, W, Z, Y0.copy(), max_iter=150, tol=1e-4, eps_num=1e-9,
        dense_pinv_threshold=5000, cg_max_iter=500, cg_tol=1e-6, reg_rho=1e-8, verbose=False,
    )

    Y_gpu, hist_gpu = smacof_solve_gpu(
        D, W, Z, Y0.copy(), max_iter=150, tol=1e-4, eps_num=1e-9,
        **_gpu_kwargs(mode="resident_cg", tile_rows=1024, cg_max_iter=500, cg_tol=1e-6, reg_rho=1e-8,
                      dtype=resolve_gpu_dtype("float64"), stress_blowup_factor=10.0),
    )
    assert hist_gpu["solver_path"] == "gpu_cg"

    from src.sammon.metrics import scale_invariant_stress

    d_cpu = pairwise_distances(Y_cpu, 1e-9)
    d_gpu = pairwise_distances(Y_gpu, 1e-9)
    stress_cpu = scale_invariant_stress(D, d_cpu)
    stress_gpu = scale_invariant_stress(D, d_gpu)
    rel_diff = abs(stress_cpu - stress_gpu) / stress_cpu
    assert rel_diff < 1e-3, f"alpha={alpha}: GPU float64 stress {stress_gpu} vs CPU {stress_cpu}, rel_diff={rel_diff} >= 1e-3."


# ---------------------------------------------------------------------------
# 2026-09-13 (documentation/2026-09-13_gpu_pametova_optimalizace.md): blocked
# RHS/sigma, closed-form Guttman transform, byte-based mode selection,
# inexact CG, history.
# ---------------------------------------------------------------------------

def _problem_257(alpha: float):
    """Test problem n=257 (not a multiple of tile_rows=64 -> the last block is shorter)."""
    D = _random_D(seed=0, n=257)
    eps_D = estimate_eps_D(D, k=5, q=0.1, kind="distance")
    W = alpha_weights(D, alpha, eps_D)
    Z = compute_Z_from_W(D, W)
    Y0 = init_random(D.shape[0], 2, seed=1, scale=0.5)
    return D, W, Z, Y0


@pytest.mark.parametrize("dtype_name,rtol", [("float64", 1e-10), ("float32", 1e-4)])
@pytest.mark.parametrize("streamed", [False, True])
def test_blocked_rhs_sigma_matches_numpy_reference(dtype_name: str, rtol: float, streamed: bool) -> None:
    """`_BlockedRhsSigma` (RHS = B(Y)Y and sigma computed in row blocks with
    preallocated buffers, both resident and streamed D/W source) must give the
    same result as the numpy reference `_build_B(D,W,d) @ Y` and
    `sum(mask*W*(D-d)^2)/2` (CPU `smacof_solve`): n=257, tile_rows=64 (the last
    block has 1 row), float64 rtol 1e-10, float32 rtol 1e-4. Also checks the
    `w_const` variant (alpha=0, W is not held resident)."""
    dtype = resolve_gpu_dtype(dtype_name)
    np_dtype = np.float64 if dtype == torch.float64 else np.float32
    dev = torch.device("cuda")
    eps_num = 1e-9
    n, p, tile_rows = 257, 2, 64
    for alpha in (1.0, 0.0):
        D, W, _, Y0 = _problem_257(alpha)
        d = pairwise_distances(Y0, eps_num)
        rhs_ref = _build_B(D, W, d, eps_num) @ Y0
        sigma_ref = float((~np.eye(n, dtype=bool) * W * (D - d) ** 2).sum() / 2.0)
        Y_t = torch.as_tensor(Y0, device=dev, dtype=dtype)
        w_const = 1.0 if alpha == 0.0 else None
        if streamed:
            D_src, W_src = D.astype(np_dtype), (None if w_const else W.astype(np_dtype))
        else:
            D_src = torch.as_tensor(D, device=dev, dtype=dtype)
            W_src = None if w_const else torch.as_tensor(W, device=dev, dtype=dtype)
        blocked = _BlockedRhsSigma(n, p, tile_rows, dev, dtype, eps_num, D_src, W_src, w_const, streamed)
        rhs, sigma = blocked(Y_t)
        rel_rhs = float(np.abs(rhs.cpu().numpy() - rhs_ref).max() / np.abs(rhs_ref).max())
        rel_sigma = abs(sigma - sigma_ref) / sigma_ref
        assert rel_rhs < rtol, f"alpha={alpha} {dtype_name} streamed={streamed}: rhs rel. error {rel_rhs} >= {rtol}"
        assert rel_sigma < rtol, f"alpha={alpha} {dtype_name} streamed={streamed}: sigma rel. error {rel_sigma} >= {rtol}"
        # no n x n buffer: helper buffers are O(tile_rows*n)
        assert blocked.n_bytes_buffers <= 6 * tile_rows * n * dtype.itemsize + p * n * dtype.itemsize


def test_smacof_gpu_guttman_matches_cpu_alpha0() -> None:
    """The closed-form Guttman transform (`gpu_guttman`, alpha=0 => W constant
    off the diagonal, V^+ = (1/(cn))(I - 11^T/n)) must give iterations
    identical to CPU `smacof_solve` with an explicit pseudoinverse (Procrustes
    disparity < 1e-10 * scale in float64), be selected automatically in 'auto'
    mode, and for alpha>0 must NOT be selected (forcing `mode='guttman'` for
    alpha>0 = ValueError)."""
    D, W, Z, Y0 = _problem_257(0.0)
    Y_cpu, hist_cpu = smacof_solve(
        D, W, Z, Y0.copy(), max_iter=200, tol=1e-8, eps_num=1e-9,
        dense_pinv_threshold=5000, cg_max_iter=500, cg_tol=1e-8, reg_rho=1e-8, verbose=False,
    )
    for mode in ("auto", "guttman"):
        Y_gpu, hist = smacof_solve_gpu(
            D, W, Z, Y0.copy(), max_iter=200, tol=1e-8, eps_num=1e-9,
            **_gpu_kwargs(mode=mode, tile_rows=64, dtype=torch.float64, **ALL_FITS),
        )
        assert hist["solver_path"] == "gpu_guttman"
        assert hist["rhs_mode"] == "resident"
        assert hist["cg_iters_total"] == 0
        assert hist["n_iter"] == hist_cpu["n_iter"]
        _, _, disparity = procrustes(Y_cpu, Y_gpu)
        scale = float(np.linalg.norm(Y_cpu - Y_cpu.mean(axis=0)))
        assert disparity < 1e-10 * scale, f"mode={mode}: Procrustes disparity {disparity} >= 1e-10*scale"
        assert abs(hist["stress"][-1] - hist_cpu["stress"][-1]) < 1e-12
    # guttman streamed (D does not fit in resident_max_bytes) - same result
    Y_s, hist_s = smacof_solve_gpu(
        D, W, Z, Y0.copy(), max_iter=200, tol=1e-8, eps_num=1e-9,
        **_gpu_kwargs(mode="auto", tile_rows=64, dtype=torch.float64, pinv_max_bytes=0, resident_max_bytes=0),
    )
    assert hist_s["solver_path"] == "gpu_guttman" and hist_s["rhs_mode"] == "streamed"
    np.testing.assert_allclose(Y_s, Y_gpu, rtol=0, atol=1e-9)

    D1, W1, Z1, _ = _problem_257(1.0)
    with pytest.raises(ValueError):
        smacof_solve_gpu(D1, W1, Z1, Y0.copy(), max_iter=5, tol=1e-8, eps_num=1e-9,
                         **_gpu_kwargs(mode="guttman", tile_rows=64, **ALL_FITS))


def test_smacof_gpu_mode_selection_by_bytes_and_agreement_with_cpu() -> None:
    """Mode selection by byte budget (`select_gpu_mode`, dtype-aware:
    n^2*itemsize): pinv <=> 3 n^2 B <= pinv_max_bytes, resident CG <=> 2 n^2 B
    <= resident_max_bytes, otherwise tiled. All three modes must, on the same
    task (alpha=1, float64), give a result matching CPU `smacof_solve`
    (Procrustes disparity < 1e-8 * scale) and the same iteration count."""
    D, W, Z, Y0 = _problem_257(1.0)
    n = D.shape[0]
    n2_f64 = n * n * 8
    # pure selection logic (without a GPU run), including the dtype-aware boundary
    assert select_gpu_mode(n, 8, 3 * n2_f64, 2 * n2_f64, None, 0.0, "auto") == ("gpu_dense_pinv", True)
    assert select_gpu_mode(n, 8, 3 * n2_f64 - 1, 2 * n2_f64, None, 0.0, "auto") == ("gpu_cg", True)
    assert select_gpu_mode(n, 8, 3 * n2_f64 - 1, 2 * n2_f64 - 1, None, 0.0, "auto") == ("gpu_tiled_cg", False)
    assert select_gpu_mode(n, 4, 3 * n2_f64 // 2, 0, None, 0.0, "auto") == ("gpu_dense_pinv", True)  # float32 = half the bytes
    assert select_gpu_mode(n, 8, 0, n2_f64, 1.0, 0.0, "auto") == ("gpu_guttman", True)
    assert select_gpu_mode(n, 8, 0, n2_f64 - 1, 1.0, 0.0, "auto") == ("gpu_guttman", False)
    assert select_gpu_mode(n, 8, 1e12, 1e12, 1.0, 0.5, "auto") == ("gpu_dense_pinv", True)  # lam>0 -> not guttman
    with pytest.raises(ValueError):
        select_gpu_mode(n, 8, 1e12, 1e12, None, 0.0, "nonsense")

    Y_cpu, hist_cpu = smacof_solve(
        D, W, Z, Y0.copy(), max_iter=200, tol=1e-8, eps_num=1e-9,
        dense_pinv_threshold=5000, cg_max_iter=500, cg_tol=1e-10, reg_rho=1e-8, verbose=False,
    )
    scale = float(np.linalg.norm(Y_cpu - Y_cpu.mean(axis=0)))
    cases = [
        (dict(pinv_max_bytes=3 * n2_f64, resident_max_bytes=2 * n2_f64), "gpu_dense_pinv", "resident"),
        (dict(pinv_max_bytes=3 * n2_f64 - 1, resident_max_bytes=2 * n2_f64), "gpu_cg", "resident"),
        (dict(pinv_max_bytes=0, resident_max_bytes=0), "gpu_tiled_cg", "streamed"),
    ]
    for limits, expected_path, expected_rhs in cases:
        Y_gpu, hist = smacof_solve_gpu(
            D, W, Z, Y0.copy(), max_iter=200, tol=1e-8, eps_num=1e-9,
            **_gpu_kwargs(tile_rows=64, cg_tol=1e-10, dtype=torch.float64, **limits),
        )
        assert hist["solver_path"] == expected_path, (limits, hist["solver_path"])
        assert hist["rhs_mode"] == expected_rhs
        assert hist["n_iter"] == hist_cpu["n_iter"]
        _, _, disparity = procrustes(Y_cpu, Y_gpu)
        assert disparity < 1e-8 * scale, f"{expected_path}: Procrustes disparity {disparity} >= 1e-8*scale"


def test_smacof_gpu_inexact_cg_matches_exact_within_tol() -> None:
    """Inexact majorization (`inexact_cg=True`: the CG tolerance is derived
    from the drop in sigma, bounded by [cg_tol, cg_tol_max]) must end with a
    stress matching the exact variant within `tol`, and with a total CG
    iteration count that is less than or equal to the exact variant; the
    history carries `cg_tol_used` within the bounds."""
    D, W, Z, Y0 = _problem_257(1.0)
    tol = 1e-6
    common = dict(max_iter=300, tol=tol, eps_num=1e-9)
    kw = _gpu_kwargs(mode="resident_cg", tile_rows=64, dtype=torch.float64, cg_tol=1e-8, cg_tol_max=1e-3,
                     cg_tol_factor=0.1, **ALL_FITS)
    _, h_exact = smacof_solve_gpu(D, W, Z, Y0.copy(), **common, **{**kw, "inexact_cg": False})
    _, h_inexact = smacof_solve_gpu(D, W, Z, Y0.copy(), **common, **{**kw, "inexact_cg": True})
    assert h_exact["solver_path"] == h_inexact["solver_path"] == "gpu_cg"
    rel = abs(h_exact["stress"][-1] - h_inexact["stress"][-1]) / h_exact["stress"][-1]
    assert rel < tol, f"inexact vs exact final stress rel. difference {rel} >= tol={tol}"
    assert h_inexact["cg_iters_total"] <= h_exact["cg_iters_total"]
    assert all(1e-8 <= t <= 1e-3 for t in h_inexact["cg_tol_used"])
    assert all(t == 1e-8 for t in h_exact["cg_tol_used"])
    # sigma monotonicity preserved even with inexact CG (CG starts from the current Y)
    sig = np.asarray(h_inexact["sigma"])
    assert np.all(np.diff(sig) <= 1e-12 * sig[:-1])


def test_smacof_gpu_history_contains_new_keys() -> None:
    """history must contain `rhs_mode`, `cg_iters_total`, `cg_iters`,
    `setup_sec`, `n2_bytes` (and the original keys) in all modes."""
    D, W, Z, Y0 = _problem_257(1.0)
    required = {"stress", "sigma", "time_sec", "n_iter", "solver_path", "device", "rhs_mode",
                "cg_iters_total", "cg_iters", "cg_tol_used", "setup_sec", "n2_bytes", "buffer_bytes"}
    for mode in ("pinv", "resident_cg", "tiled_cg"):
        _, hist = smacof_solve_gpu(D, W, Z, Y0.copy(), max_iter=5, tol=1e-8, eps_num=1e-9,
                                   **_gpu_kwargs(mode=mode, tile_rows=64, **ALL_FITS))
        assert required <= set(hist), f"mode={mode}: missing {required - set(hist)}"
        assert hist["n2_bytes"] == D.shape[0] ** 2 * 4  # default float32
        if mode == "pinv":
            assert hist["cg_iters_total"] == 0 and hist["cg_iters"] == []
        else:
            assert hist["cg_iters_total"] == sum(hist["cg_iters"]) > 0
            assert len(hist["cg_iters"]) == hist["n_iter"] - 1 or len(hist["cg_iters"]) == hist["n_iter"]
