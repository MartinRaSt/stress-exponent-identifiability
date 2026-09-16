# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-13
# License: see the LICENSE file in the repository root
"""Tests for the VRAM preflight for large E3 graphs (documentation/
2026-09-13_hardening_behu.md, task 2): the peak estimate
`estimate_gpu_peak_bytes` (same mode as `select_gpu_mode`), agreement of the
buffer estimate with the actual `_BlockedRhsSigma` allocation (on CPU
tensors, without CUDA), and `exp3_graph_layout.preflight_gpu_memory` with a
mocked `mem_get_info` (without CUDA) + a real run on this machine, if CUDA
is available."""
from __future__ import annotations

import logging

import numpy as np
import pytest

from src.common.config import load_config
from src.experiments.exp3_graph_layout import preflight_gpu_memory
from src.sammon.solvers.smacof import (
    _BlockedRhsSigma,
    blocked_rhs_buffer_bytes,
    estimate_gpu_peak_bytes,
    select_gpu_mode,
)

torch = pytest.importorskip("torch")

GB = 1e9
# pgp LCC (E3, config sammon.gpu_memory_probe.expected_n) and the RTX 3070 Ti total
# (torch.cuda.mem_get_info()[1] measured 2026-09-13)
N_PGP = int(load_config()["sammon"]["gpu_memory_probe"]["expected_n"])
TOTAL_RTX3070TI = 8_589_410_304


def _gd():
    return load_config()["sammon"]["gpu_dense"]


def test_estimate_matches_select_gpu_mode_and_formula_pgp_float64() -> None:
    gd = _gd()
    itemsize = 8
    total, path = estimate_gpu_peak_bytes(
        N_PGP, 2, itemsize, tile_rows=int(gd["tile_rows"]), pinv_max_bytes=float(gd["pinv_max_bytes"]),
        resident_max_bytes=float(gd["resident_max_bytes"]), cuda_context_bytes=float(gd["cuda_context_bytes"]),
    )
    expected_path, resident = select_gpu_mode(N_PGP, itemsize, gd["pinv_max_bytes"], gd["resident_max_bytes"], None, 0.0, "auto")
    assert path == expected_path == "gpu_cg" and resident
    n2 = N_PGP * N_PGP * itemsize
    bufs = blocked_rhs_buffer_bytes(N_PGP, 2, int(gd["tile_rows"]), itemsize, streamed=False, w_const=False)
    assert total == 2 * n2 + bufs + N_PGP * 2 * itemsize + int(gd["cuda_context_bytes"])
    # order of magnitude: D+W 1.83 GB + buffers 0.26 GB + context 0.3 GB ~ 2.4 GB (see the config comment)
    assert 2.2 * GB < total < 2.6 * GB


def test_estimate_guttman_uses_single_n2() -> None:
    gd = _gd()
    total_w, path_w = estimate_gpu_peak_bytes(N_PGP, 2, 8, int(gd["tile_rows"]), gd["pinv_max_bytes"], gd["resident_max_bytes"], 0.0, w_const=False)
    total_c, path_c = estimate_gpu_peak_bytes(N_PGP, 2, 8, int(gd["tile_rows"]), gd["pinv_max_bytes"], gd["resident_max_bytes"], 0.0, w_const=True)
    assert path_w == "gpu_cg" and path_c == "gpu_guttman"
    assert total_c < total_w
    assert abs((total_w - total_c) - N_PGP * N_PGP * 8) < 1  # the difference is exactly one n^2 (W)


@pytest.mark.parametrize("streamed,w_const", [(False, False), (True, False), (True, True), (False, True)])
def test_buffer_bytes_match_blocked_rhs_sigma_allocation(streamed: bool, w_const: bool) -> None:
    """The buffer estimate must match the class's actual allocation (CPU tensors suffice)."""
    n, p, tile_rows = 700, 2, 256
    D = np.random.default_rng(0).random((n, n))
    src_D = D if streamed else torch.as_tensor(D)
    src_W = None if w_const else (D if streamed else torch.as_tensor(D))
    obj = _BlockedRhsSigma(n, p, tile_rows, torch.device("cpu"), torch.float64, 1e-9,
                           D_src=src_D, W_src=src_W, w_const=1.0 if w_const else None, streamed=streamed)
    assert obj.n_bytes_buffers == blocked_rhs_buffer_bytes(n, p, tile_rows, 8, streamed=streamed, w_const=w_const)


def _mock_mem(total: int):
    return lambda: (total // 2, total)


def test_preflight_passes_for_two_workers_pgp(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("test_preflight")
    with caplog.at_level(logging.INFO, logger="test_preflight"):
        info = preflight_gpu_memory(N_PGP, 2, 2, "float64", reserved_other_bytes=2.4e9, safety=0.9, logger=logger,
                                    mem_get_info=_mock_mem(TOTAL_RTX3070TI))
    assert info["solver_path"] == "gpu_cg"
    assert info["estimate_total_bytes"] <= info["budget_bytes"]
    assert info["recommended_max_workers"] >= 2
    assert any("VRAM preflight" in r.getMessage() for r in caplog.records)


def test_preflight_raises_for_four_workers_pgp_with_recommendation() -> None:
    logger = logging.getLogger("test_preflight")
    with pytest.raises(RuntimeError) as exc:
        preflight_gpu_memory(N_PGP, 2, 4, "float64", reserved_other_bytes=2.4e9, safety=0.9, logger=logger,
                             mem_get_info=_mock_mem(TOTAL_RTX3070TI))
    msg = str(exc.value)
    assert "VRAM preflight failed" in msg
    assert "large_graph_max_workers" in msg
    # recommended count = the one that fits (2 for pgp at 8.59 GB with a 2.4 GB reserve)
    assert "to 2 " in msg


def test_preflight_invalid_args() -> None:
    logger = logging.getLogger("test_preflight")
    with pytest.raises(ValueError):
        preflight_gpu_memory(N_PGP, 2, 0, "float64", 0.0, 0.9, logger, mem_get_info=_mock_mem(TOTAL_RTX3070TI))
    with pytest.raises(ValueError):
        preflight_gpu_memory(N_PGP, 2, 2, "float64", 0.0, 1.5, logger, mem_get_info=_mock_mem(TOTAL_RTX3070TI))
    with pytest.raises(ValueError):
        preflight_gpu_memory(N_PGP, 2, 2, "float16", 0.0, 0.9, logger, mem_get_info=_mock_mem(TOTAL_RTX3070TI))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available - the real preflight is skipped.")
def test_preflight_real_gpu_with_config_values(caplog: pytest.LogCaptureFixture) -> None:
    """A real run with the values from config_experiments.yaml (2 workers, pgp)
    on this machine - must pass (otherwise the full E3 run would fail right at the start)."""
    from src.experiments.config_experiments import resolve_experiment_config

    cfg = resolve_experiment_config("exp3_graph_layout", "full")
    logger = logging.getLogger("test_preflight")
    with caplog.at_level(logging.INFO, logger="test_preflight"):
        info = preflight_gpu_memory(
            N_PGP, int(cfg["n_components"]), int(cfg["large_graph_max_workers"]), str(cfg["large_graph_gpu_dtype"]),
            reserved_other_bytes=float(cfg["gpu_reserved_other_bytes"]), safety=float(cfg["gpu_preflight_safety"]),
            logger=logger,
        )
    assert info["gpu_total_bytes"] > 0
    assert info["estimate_total_bytes"] <= info["budget_bytes"]
