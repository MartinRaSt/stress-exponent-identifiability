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
`SammonAlpha` - a unified estimator (scikit-learn-style `fit_transform`)
over all solvers (pseudo-Newton, SMACOF, SGD, sparse SMACOF/SGD) and alpha
choices (a fixed number, 'auto' - grid + R_NX AUC, 'multiscale'), see
reserse/2026-09-09_specifikace_metody.md, sections 1-6.

Precision: float64 on CPU (numpy solvers), float32 on GPU (torch solvers
for device='cuda'), see sections 12.1/12.4 and `sammon.device` in config.yaml.
"""
from __future__ import annotations

import logging

import numpy as np

from src.common.config import load_config
from src.sammon.alpha_selection import multiscale as run_multiscale
from src.sammon.alpha_selection import select_alpha
from src.sammon.device import resolve_device
from src.sammon.init import init_classical_mds, init_pca, init_random
from src.sammon.solvers.newton import newton_solve, newton_solve_gpu
from src.sammon.solvers.sgd import sgd_solve, sgd_solve_gpu
from src.sammon.solvers.smacof import resolve_gpu_dtype, smacof_solve, smacof_solve_gpu
from src.sammon.solvers.sparse import build_sparse_terms, sparse_sgd_solve, sparse_smacof_solve
from src.sammon.weights import alpha_weights, compute_Z_from_W, estimate_eps_D

logger = logging.getLogger("sammon.estimator")

VALID_SOLVERS = {"newton", "smacof", "sgd", "sgd_naive", "sparse_smacof", "sparse_sgd"}
VALID_INITS = {"pca", "random", "classical_mds"}


class SammonAlpha:
    """Alpha-Sammon: E_alpha(Y) = sum w_ij(D_ij-d_ij)^2 / sum w_ij D_ij^2,
    w_ij = D_ij^{-alpha} (regularized with +eps_D), minimized by the chosen
    solver.

    Parameters:
        alpha: 'auto' (grid search + R_NX AUC, section 5.1), 'multiscale'
            (section 5.2), or a fixed nonnegative number.
        solver: 'newton' | 'smacof' | 'sgd' | 'sgd_naive' | 'sparse_smacof'
            | 'sparse_sgd'. 'sgd' uses the stabilized variant (section 3.3,
            a safe default for alpha>0); 'sgd_naive' is the original scheme
            of Zheng 2018 (section 3.1, may diverge for alpha>0 - intended
            for the comparative ablation).
        device: 'cpu' | 'cuda' | 'auto'.
        n_components, init, max_iter, tol, seed, verbose: see config.yaml
            `sammon.estimator` for default values (used when None).

    After `fit_transform`: `self.embedding_`, `self.stress_` (E_alpha^scale-inv
    on the resulting Y), `self.history_` (solver history), `self.alpha_`
    (the resolved alpha - a number, even if the input was 'auto'),
    `self.eps_D_`.
    """

    def __init__(
        self,
        alpha: float | str = 1.0,
        solver: str = "smacof",
        device: str = "auto",
        n_components: int | None = None,
        init: str = "pca",
        max_iter: int | None = None,
        tol: float | None = None,
        seed: int | None = None,
        verbose: bool = False,
    ) -> None:
        if solver not in VALID_SOLVERS:
            raise ValueError(f"Unknown solver='{solver}' (expected one of {sorted(VALID_SOLVERS)}).")
        if init not in VALID_INITS:
            raise ValueError(f"Unknown init='{init}' (expected one of {sorted(VALID_INITS)}).")

        cfg = load_config()["sammon"]["estimator"]
        self.alpha = alpha
        self.solver = solver
        self.device = device
        self.n_components = n_components if n_components is not None else int(cfg["n_components"])
        self.init = init
        self.max_iter = max_iter if max_iter is not None else int(cfg["max_iter"])
        self.tol = tol if tol is not None else float(cfg["tol"])
        self.seed = seed if seed is not None else int(cfg["seed"])
        self.verbose = verbose

    def _init_Y0(self, data, D: np.ndarray, kind: str) -> np.ndarray:
        if self.init == "random":
            scale = float(load_config()["sammon"]["init"]["random_scale"]) * float(D[~np.eye(D.shape[0], dtype=bool)].mean())
            return init_random(D.shape[0], self.n_components, self.seed, scale=scale)
        if self.init == "classical_mds":
            return init_classical_mds(D, self.n_components, self.seed)
        # init == "pca": requires point data; for kind!='vector' we use
        # classical MDS as a natural substitute (not fabrication - a
        # well-defined, documented alternative for inputs without X,
        # section 1.4/2.3)
        if kind == "vector":
            return init_pca(data, self.n_components, self.seed)
        logger.info("init='pca' requires vector data, kind='%s' -> using init_classical_mds instead.", kind)
        return init_classical_mds(D, self.n_components, self.seed)

    def fit_transform(self, data, kind: str = "vector") -> np.ndarray:
        """Compute and return the embedding (n x n_components)."""
        # local import - `src.methods` registers this module as well during
        # its initialization (src.methods.sammon_alpha -> SammonAlpha); a
        # module-level (top-level) import would cause a circular import
        from src.methods.common import to_distance_matrix

        cfg = load_config()["sammon"]
        device_resolved = resolve_device(self.device)
        D = to_distance_matrix(data, kind)
        n = D.shape[0]

        if self.alpha == "multiscale":
            Y, history = run_multiscale(
                D, alphas=cfg["multiscale_alphas"], seed=self.seed, kind="distance",
                n_components=self.n_components, max_iter=self.max_iter, tol=self.tol,
                eps_num=cfg["eps_num"], eps_D_k=cfg["eps_D"]["k"], eps_D_q=cfg["eps_D"]["q"],
                C=cfg["multiscale_C"], dense_pinv_threshold=cfg["smacof"]["dense_pinv_threshold"],
                cg_max_iter=cfg["smacof"]["cg_max_iter"], cg_tol=cfg["smacof"]["cg_tol"],
                reg_rho=cfg["smacof"]["reg_rho"],
            )
            self.alpha_ = "multiscale"
            self.eps_D_ = history["eps_D"]
            self.embedding_, self.history_ = Y, history
            self.stress_ = history["stress"][-1]
            return Y

        if self.alpha == "auto":
            sel_cfg = cfg["alpha_selection"]
            best_alpha, table = select_alpha(
                D, grid=cfg["alpha_grid"], n_val=sel_cfg["n_val"], seed=sel_cfg["seed"],
                kind="distance", metric=sel_cfg["metric"], n_components=self.n_components,
                max_iter=self.max_iter, tol=self.tol, eps_num=cfg["eps_num"],
                eps_D_k=cfg["eps_D"]["k"], eps_D_q=cfg["eps_D"]["q"],
                dense_pinv_threshold=cfg["smacof"]["dense_pinv_threshold"],
                cg_max_iter=cfg["smacof"]["cg_max_iter"], cg_tol=cfg["smacof"]["cg_tol"],
                reg_rho=cfg["smacof"]["reg_rho"],
            )
            alpha = best_alpha
            self.alpha_selection_table_ = table
        else:
            alpha = float(self.alpha)
        self.alpha_ = alpha

        eps_D = estimate_eps_D(D, k=cfg["eps_D"]["k"], q=cfg["eps_D"]["q"], kind="distance")
        self.eps_D_ = eps_D
        Y0 = self._init_Y0(data if kind == "vector" else None, D, kind)

        if self.solver == "newton":
            W = alpha_weights(D, alpha, eps_D)
            Z = compute_Z_from_W(D, W)
            if device_resolved == "cuda":
                gd_cfg = cfg["gpu_dense"]
                Y, history = newton_solve_gpu(
                    D, W, Z, Y0, max_iter=self.max_iter, mf=cfg["newton"]["mf"], tol=self.tol,
                    eps_num=cfg["eps_num"], tile_threshold=gd_cfg["tile_threshold"],
                    tile_rows=gd_cfg["tile_rows"], step_halving=cfg["newton"]["step_halving"],
                    max_halvings=cfg["newton"]["max_halvings"], device="cuda", verbose=self.verbose,
                )
            else:
                Y, history = newton_solve(
                    D, W, Z, Y0, max_iter=self.max_iter, mf=cfg["newton"]["mf"], tol=self.tol,
                    eps_num=cfg["eps_num"], step_halving=cfg["newton"]["step_halving"],
                    max_halvings=cfg["newton"]["max_halvings"], verbose=self.verbose,
                )
        elif self.solver == "smacof":
            W = alpha_weights(D, alpha, eps_D)
            Z = compute_Z_from_W(D, W)
            if device_resolved == "cuda":
                gd_cfg = cfg["gpu_dense"]
                Y, history = smacof_solve_gpu(
                    D, W, Z, Y0, max_iter=self.max_iter, tol=self.tol, eps_num=cfg["eps_num"],
                    tile_rows=gd_cfg["tile_rows"], cg_max_iter=gd_cfg["cg_max_iter"],
                    cg_tol=gd_cfg["cg_tol"], reg_rho=gd_cfg["reg_rho"],
                    pinv_max_bytes=gd_cfg["pinv_max_bytes"], resident_max_bytes=gd_cfg["resident_max_bytes"],
                    const_w_rtol=gd_cfg["const_w_rtol"], inexact_cg=gd_cfg["inexact_cg"],
                    cg_tol_factor=gd_cfg["cg_tol_factor"], cg_tol_max=gd_cfg["cg_tol_max"],
                    cg_check_every=gd_cfg["cg_check_every"],
                    stress_blowup_factor=gd_cfg["stress_blowup_factor"],
                    dtype=resolve_gpu_dtype(gd_cfg["dtype"]), device="cuda", verbose=self.verbose,
                )
            else:
                Y, history = smacof_solve(
                    D, W, Z, Y0, max_iter=self.max_iter, tol=self.tol, eps_num=cfg["eps_num"],
                    dense_pinv_threshold=cfg["smacof"]["dense_pinv_threshold"],
                    cg_max_iter=cfg["smacof"]["cg_max_iter"], cg_tol=cfg["smacof"]["cg_tol"],
                    reg_rho=cfg["smacof"]["reg_rho"], verbose=self.verbose,
                )
        elif self.solver in ("sgd", "sgd_naive"):
            variant = "naive" if self.solver == "sgd_naive" else "stabilized"
            sgd_cfg = cfg["sgd"]
            if device_resolved == "cuda":
                Y, history = sgd_solve_gpu(
                    D, alpha, Y0, epochs=sgd_cfg["epochs"], mu_max=sgd_cfg["mu_max"],
                    pairs_per_node=sgd_cfg["pairs_per_node"], eps_anneal=sgd_cfg["eps_anneal"],
                    eps_num=cfg["eps_num"], seed=self.seed, variant=variant,
                    batch_mode="round_robin", k_eps=sgd_cfg["k_eps"], q_eps=sgd_cfg["q_eps"],
                    device="cuda", log_every=sgd_cfg["log_every"], verbose=self.verbose,
                )
            else:
                Y, history = sgd_solve(
                    D, alpha, Y0, epochs=sgd_cfg["epochs"], mu_max=sgd_cfg["mu_max"],
                    pairs_per_node=sgd_cfg["pairs_per_node"], eps_anneal=sgd_cfg["eps_anneal"],
                    eps_num=cfg["eps_num"], seed=self.seed, variant=variant,
                    k_eps=sgd_cfg["k_eps"], q_eps=sgd_cfg["q_eps"], verbose=self.verbose,
                )
        elif self.solver in ("sparse_smacof", "sparse_sgd"):
            sp_cfg = cfg["sparse"]
            terms = build_sparse_terms(
                D, alpha, eps_D, n_pivots=sp_cfg["n_pivots"], k_neighbors=sp_cfg["k_neighbors"],
                seed=self.seed, kind="distance", exact_knn_threshold=sp_cfg["exact_knn_threshold"],
            )
            if self.solver == "sparse_smacof":
                Y, history = sparse_smacof_solve(
                    terms, Y0, max_iter=self.max_iter, tol=self.tol, eps_num=cfg["eps_num"],
                    cg_max_iter=cfg["smacof"]["cg_max_iter"], cg_tol=cfg["smacof"]["cg_tol"],
                    reg_rho=cfg["smacof"]["reg_rho"], alpha_for_Z=alpha, verbose=self.verbose,
                )
            else:
                sgd_cfg = cfg["sgd"]
                Y, history = sparse_sgd_solve(
                    terms, Y0, epochs=sgd_cfg["epochs"], mu_max=sgd_cfg["mu_max"],
                    eps_anneal=sgd_cfg["eps_anneal"], eps_num=cfg["eps_num"], seed=self.seed,
                    alpha_for_Z=alpha, verbose=self.verbose,
                )
            history["n_terms"] = terms["row"].shape[0]
        else:  # pragma: no cover - already handled in __init__
            raise ValueError(f"Unknown solver='{self.solver}'.")

        self.embedding_ = Y
        self.history_ = history
        self.stress_ = history["stress"][-1]
        return Y
