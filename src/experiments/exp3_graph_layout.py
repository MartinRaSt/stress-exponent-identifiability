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
E3 - graph layouts: all graph datasets from the registry (karate,
les_miserables, florentine, dolphins, football, polbooks, email_eu_core),
each converted to a distance matrix in two ways (shortest paths, resistance
distance, src/datasets/graph_distance.py). Methods working with a
precomputed distance matrix (MDS, Sammon/alpha-Sammon family, t-SNE, UMAP)
are run on both distance variants; graph-native layouts (Kamada-Kawai,
spring/Fruchterman-Reingold) are computed directly on the graph (independent
of the distance metric choice). 10 seeds. Results incl. layouts go to
results/data/exp3_graph_layout_results.csv (resumable).

Run: venv\\python.exe -m src.experiments.exp3_graph_layout [--quick|--full]
or: src\\run_exp3_graph_layout.bat [quick|full]
"""
from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.common.checkpoint import RunKey, append_result, is_done
from src.common.logging_utils import get_logger
from src.experiments.config_experiments import resolve_experiment_config
from src.experiments.exp_common import (
    discover_metric_keys,
    filter_already_done,
    parse_mode_args,
    resolve_experiment_name,
    run_experiment_grid,
    write_done_file,
)

BASE_EXPERIMENT_NAME = "exp3_graph_layout"

# distance metric marked in the dataset label in the results (the name of
# the graph loader remains part of the label so results are easy to tell apart)
_DISTANCE_FN_NAME = {"shortest_path": "shortest_path_distance", "resistance": "resistance_distance"}


@contextmanager
def _override_method_device(method_name: str, device: str) -> Iterator[None]:
    """K8 (follow-up task, see documentation/2026-09-12_s2_koder_b.md):
    temporarily overrides `methods.<method_name>.device` in the global
    (cached, `functools.lru_cache`) `common/config.yaml` configuration ONLY
    for the duration of the `method.fit_transform(...)` call in this worker
    process, without changing the global `methods.<method_name>.device`
    (which must remain 'cpu' for E1 timings - see the spec "so the global
    methods.*.device: cpu does not have to change"). The original value is
    ALWAYS restored (even on an exception, `finally`).

    Safe within a `ProcessPoolExecutor`: each worker process handles tasks
    sequentially (one Python interpreter, no threads calling `method_config`
    concurrently), so a temporary mutation of the global cached dict in this
    process does not affect other workers or the main process.

    Does NOT require editing `src/methods/sammon_alpha.py` or
    `src/methods/sammon_alpha_pred.py` (both only read `methods.<name>.device`
    via `method_config()` at the start of their `fit_transform`, see their
    source) - this is therefore the smallest possible intervention that
    works identically for all 5 methods in `large_graph_device_methods`,
    including the method from the second (concurrently working) coder.

    Fail-loud: if 'cuda' is requested but `methods.<method_name>` in
    config.yaml has no 'device' key at all (the method has no GPU
    implementation at all), a clear `KeyError` is raised - no silent no-op override.
    """
    from src.common.config import load_config

    cfg = load_config()
    try:
        method_cfg = cfg["methods"][method_name]
    except KeyError as exc:
        raise KeyError(f"config.yaml methods.{method_name} does not exist - cannot temporarily override 'device'.") from exc
    if "device" not in method_cfg:
        raise KeyError(
            f"config.yaml methods.{method_name} has no 'device' key - this method "
            "likely has no GPU implementation, an override to 'cuda' would be a no-op (not fabricated)."
        )
    original_device = method_cfg["device"]
    method_cfg["device"] = device
    try:
        yield
    finally:
        method_cfg["device"] = original_device


@contextmanager
def _override_gpu_dtype(dtype: str) -> Iterator[None]:
    """2026-09-12 (documentation/2026-09-12_gpu_pcg_a_metriky_velkych_grafu.md):
    temporarily overrides `sammon.gpu_dense.dtype` in the global (cached)
    configuration ONLY for the duration of the `fit_transform` call in this
    worker process - analogous to `_override_method_device`, same safety
    rationale (sequential task processing within a single
    `ProcessPoolExecutor` worker process). The original value is always
    restored (`finally`). Used together with `_override_method_device` for
    large graphs (`large_graph_gpu_dtype` in config_experiments.yaml
    exp3_graph_layout), so the SMACOF family on GPU can run in float64 (more
    precise but slower FP64 on the RTX 3070 Ti) if required by measurements
    for the given method (see the document above, the CPU/GPU
    float32/float64 precision table)."""
    from src.common.config import load_config

    cfg = load_config()
    gd_cfg = cfg["sammon"]["gpu_dense"]
    original_dtype = gd_cfg["dtype"]
    gd_cfg["dtype"] = dtype
    try:
        yield
    finally:
        gd_cfg["dtype"] = original_dtype


def _to_largest_component(g: Any, y: "np.ndarray | None") -> tuple[Any, "np.ndarray | None", int, int]:
    """Restricts the graph to its largest connected component (some real
    graphs, e.g. SNAP email_eu_core, contain isolated nodes - shortest-path
    distance and nx.kamada_kawai_layout/spring_layout are both undefined on
    a disconnected graph, see src/datasets/graph_distance.py). The `y`
    labels are trimmed consistently (same order as `list(lcc.nodes())`).

    Returns (lcc_graph, y_lcc, n_nodes_original, n_nodes_lcc). For an
    already connected graph, returns the input unchanged (n_original == n_lcc)."""
    import networkx as nx

    n_original = g.number_of_nodes()
    if nx.is_connected(g):
        return g, y, n_original, n_original

    largest_nodes = max(nx.connected_components(g), key=len)
    lcc = g.subgraph(largest_nodes).copy()
    n_lcc = lcc.number_of_nodes()

    import logging

    logging.getLogger("sammon.exp3_graph_layout").info(
        "Graph is not connected - restricting to the largest connected component: %d/%d nodes.", n_lcc, n_original,
    )

    y_lcc = None
    if y is not None:
        orig_nodes = list(g.nodes())
        idx_by_node = {node: i for i, node in enumerate(orig_nodes)}
        y_arr = np.asarray(y)
        y_lcc = y_arr[[idx_by_node[node] for node in lcc.nodes()]]

    return lcc, y_lcc, n_original, n_lcc


def _graph_lcc_info(base_name: str) -> tuple[int, int, str]:
    """Returns (n_nodes_original, n_nodes_lcc, community_source) WITHOUT
    computing the distance matrix - just the graph structure (loading + LCC,
    see `_to_largest_component`). Used in `main()` to decide whether a graph
    is "large" (K8: `large_graph_threshold_nodes`) without requiring an
    (expensive) shortest_path/resistance distance computation beforehand."""
    from src.datasets.registry import load_dataset

    ds = load_dataset(base_name)
    _, _, n_original, n_lcc = _to_largest_component(ds.graph, ds.y)
    return n_original, n_lcc, str(ds.meta.get("community_source", ""))


def _build_distance_dataset(
    base_name: str, distance_metric: str, large_n_threshold: int = 2000, resistance_device: str = "cpu",
):
    """Loads the graph `base_name`, restricts it to its largest connected
    component (see `_to_largest_component`) and converts it to
    Dataset(kind='distance') using the given distance metric
    (src/datasets/graph_distance.py).

    K8: the result is cached to disk (`cached_distance_matrix`, keyed by a
    hash of the LCC subgraph's content) and, for n > `large_n_threshold`,
    computes resistance distance via the Laplacian's eigendecomposition
    instead of a direct `pinv` (see `graph_distance.py`, optional
    `resistance_device='cuda'`).

    Returns (Dataset, lcc_graph, n_nodes_original, n_nodes_lcc)."""
    from src.datasets.graph_distance import cached_distance_matrix
    from src.datasets.registry import Dataset, load_dataset

    ds = load_dataset(base_name)
    g_lcc, y_lcc, n_original, n_lcc = _to_largest_component(ds.graph, ds.y)
    D = cached_distance_matrix(
        base_name, distance_metric, g_lcc,
        large_n_threshold=large_n_threshold, device=resistance_device,
    )
    out_ds = Dataset(
        name=f"{base_name}__{distance_metric}", kind="distance", D=D, y=y_lcc,
        meta={**ds.meta, "distance_metric": distance_metric, "base_graph": base_name,
              "n_nodes_original": n_original, "n_nodes_lcc": n_lcc},
    )
    return out_ds, g_lcc, n_original, n_lcc


def _graph_metrics_dict(Y: "np.ndarray", g: Any, y: "np.ndarray | None") -> dict[str, Any]:
    """Computes graph-specific metrics (src/sammon/graph_metrics.py, if it
    exists - imported conditionally, otherwise an empty dict is returned and
    the run continues with only the basic metrics from evaluate(), exactly
    per the E3 spec). Each metric is computed independently in a try/except
    so that one failing (e.g. n>exact_max_n) does not cause the loss of the others."""
    try:
        from src.sammon import graph_metrics as gm
    except ImportError:
        return {}

    from src.common.config import load_config

    cfg_ext = load_config()["sammon"]["metrics_extended"]
    exact_max_n = cfg_ext["graph_crossings_exact_max_n"]

    out: dict[str, Any] = {}
    try:
        out["edge_crossings"], _ = gm.edge_crossings(Y, g, exact_max_n)
    except Exception:
        out["edge_crossings"] = np.nan
    try:
        out["crossing_angle"] = gm.crossing_angle(Y, g, exact_max_n)
    except Exception:
        out["crossing_angle"] = np.nan
    try:
        out["angular_resolution"] = gm.angular_resolution(Y, g)
    except Exception:
        out["angular_resolution"] = np.nan
    try:
        out["edge_length_uniformity"] = gm.edge_length_uniformity(Y, g)
    except Exception:
        out["edge_length_uniformity"] = np.nan
    try:
        out["neighborhood_preservation"] = gm.neighborhood_preservation(Y, g)
    except Exception:
        out["neighborhood_preservation"] = np.nan
    if y is not None:
        try:
            out["community_silhouette"] = gm.community_silhouette(Y, g, y)
        except Exception:
            out["community_silhouette"] = np.nan
    else:
        out["community_silhouette"] = np.nan
    return out


GRAPH_METRIC_KEYS = [
    "edge_crossings", "crossing_angle", "angular_resolution",
    "edge_length_uniformity", "neighborhood_preservation", "community_silhouette",
]


def _run_single_distance(task: dict[str, Any]) -> dict[str, Any]:
    """Runs a method on a precomputed distance matrix (one of distance_metrics)."""
    from src.common.seeding import set_seed
    from src.methods.common import method_config
    from src.methods.registry import get_method
    from src.sammon.metrics import evaluate

    dataset_label = task["dataset_name"]
    method_name = task["method_name"]
    seed = task["seed"]
    n_components = task["n_components"]
    eval_kwargs = task["eval_kwargs"]
    device_override = task.get("device_override")
    gpu_dtype_override = task.get("gpu_dtype_override")

    set_seed(seed)
    result: dict[str, Any] = {"dataset_name": dataset_label, "method_name": method_name, "seed": seed}
    t0 = time.perf_counter()
    try:
        ds, g_lcc, n_original, n_lcc = _build_distance_dataset(
            task["base_name"], task["distance_metric"],
            large_n_threshold=task.get("large_n_threshold", 2000),
            resistance_device=task.get("resistance_device", "cpu"),
        )
        method = get_method(method_name)
        # K8 follow-up task: 'device_override' (set only for large graphs
        # and methods in exp3_graph_layout.large_graph_device_methods, see
        # main()) temporarily overrides methods.<method_name>.device to
        # 'cuda' ONLY for this run, without changing the global config.yaml
        # (E1 stays CPU). 2026-09-12: 'gpu_dtype_override' (config
        # large_graph_gpu_dtype) simultaneously temporarily overrides
        # sammon.gpu_dense.dtype - see
        # documentation/2026-09-12_gpu_pcg_a_metriky_velkych_grafu.md for
        # the rationale behind the float32/float64 choice.
        if device_override:
            with _override_method_device(method_name, device_override), _override_gpu_dtype(gpu_dtype_override or "float32"):
                Y = method.fit_transform(ds.D, "distance", seed=seed, n_components=n_components)
            device_used = device_override
        else:
            Y = method.fit_transform(ds.D, "distance", seed=seed, n_components=n_components)
            try:
                device_used = str(method_config(method_name).get("device", "cpu"))
            except KeyError:
                device_used = "cpu"
        metrics = evaluate(ds.D, Y, ds.y, "distance", **eval_kwargs)
        metrics.update(_graph_metrics_dict(Y, g_lcc, ds.y))
        # see exp1_dr_benchmark.py - same mechanism for recording the
        # selected hyperparameter for methods with a grid search
        # (sammon_alpha_auto, sammon_multiscale, tsne_auto, umap_auto)
        selected_hyperparam = str(getattr(method, "last_selected_hyperparam", ""))

        result["status"] = "ok"
        result["error"] = ""
        result["wall_time_sec"] = time.perf_counter() - t0
        result["embedding"] = Y
        result["metrics"] = metrics
        result["extra"] = {
            "distance_metric": task["distance_metric"], "base_graph": task["base_name"],
            "n_nodes_original": n_original, "n_nodes_lcc": n_lcc, "selected_hyperparam": selected_hyperparam,
            "community_source": str(ds.meta.get("community_source", "")), "device": device_used,
        }
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["wall_time_sec"] = time.perf_counter() - t0
        result["embedding"] = None
        result["metrics"] = {}
        result["extra"] = {
            "distance_metric": task["distance_metric"], "base_graph": task["base_name"],
            "n_nodes_original": np.nan, "n_nodes_lcc": np.nan, "selected_hyperparam": "", "community_source": "",
            "device": device_override or "",
        }
    return result


def _run_single_native(task: dict[str, Any]) -> dict[str, Any]:
    """Runs a graph-native method (Kamada-Kawai, spring) directly on the
    graph - restricted to the largest connected component (see
    `_to_largest_component`), because nx.kamada_kawai_layout/spring_layout
    fail on a disconnected graph (they internally compute with geodesic distances)."""
    from src.common.seeding import set_seed
    from src.datasets.registry import load_dataset
    from src.methods.registry import get_method
    from src.sammon.metrics import evaluate

    base_name = task["base_name"]
    method_name = task["method_name"]
    seed = task["seed"]
    n_components = task["n_components"]
    eval_kwargs = task["eval_kwargs"]

    set_seed(seed)
    result: dict[str, Any] = {"dataset_name": base_name, "method_name": method_name, "seed": seed}
    t0 = time.perf_counter()
    try:
        ds_full = load_dataset(base_name)
        g_lcc, y_lcc, n_original, n_lcc = _to_largest_component(ds_full.graph, ds_full.y)
        method = get_method(method_name)
        Y = method.fit_transform(g_lcc, "graph", seed=seed, n_components=n_components)
        metrics = evaluate(g_lcc, Y, y_lcc, "graph", **eval_kwargs)
        metrics.update(_graph_metrics_dict(Y, g_lcc, y_lcc))

        result["status"] = "ok"
        result["error"] = ""
        result["wall_time_sec"] = time.perf_counter() - t0
        result["embedding"] = Y
        result["metrics"] = metrics
        result["extra"] = {
            "distance_metric": "native", "base_graph": base_name,
            "n_nodes_original": n_original, "n_nodes_lcc": n_lcc, "selected_hyperparam": "",
            "community_source": str(ds_full.meta.get("community_source", "")),
            # K8 follow-up task: native methods (kamada_kawai/spring/spectral)
            # have no GPU implementation - the device override is never
            # applied to them (see main(), large_graph_device_methods
            # contains only the SMACOF family), always 'cpu'.
            "device": "cpu",
        }
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["wall_time_sec"] = time.perf_counter() - t0
        result["embedding"] = None
        result["metrics"] = {}
        result["extra"] = {
            "distance_metric": "native", "base_graph": base_name,
            "n_nodes_original": np.nan, "n_nodes_lcc": np.nan, "selected_hyperparam": "", "community_source": "",
            "device": "cpu",
        }
    return result


def _write_skip_row(
    experiment_name: str, dataset_label: str, method_name: str, seed: int,
    column_keys: list[str], extra: dict[str, Any], logger,
) -> None:
    """K8: writes an explicit row with status 'skipped_max_n' (NOT a silent
    omission from tasks) for the (dataset, method, seed) combination where
    the graph size (n_nodes_lcc) exceeds `method_max_n[method_name]`.
    Idempotent (checks `is_done` before writing, so a resume run does not
    revisit skip-rows already written or other previously completed runs)."""
    key = RunKey(experiment_name, dataset_label, method_name, seed)
    if is_done(key):
        return
    row: dict[str, Any] = {k: np.nan for k in column_keys}
    row.update(extra)
    row["status"] = "skipped_max_n"
    row["error"] = f"n_nodes_lcc > method_max_n['{method_name}'] (see config_experiments.yaml exp3_graph_layout.method_max_n)"
    row["wall_time_sec"] = 0.0
    append_result(key, row)
    logger.info("Skipped (skipped_max_n): dataset=%s method=%s seed=%s.", dataset_label, method_name, seed)


def preflight_gpu_memory(
    n_max: int, n_components: int, n_workers: int, gpu_dtype: str, reserved_other_bytes: float,
    safety: float, logger, mem_get_info=None,
) -> dict[str, Any]:
    """2026-09-13 (documentation/2026-09-13_hardening_behu.md): a VRAM
    preflight check BEFORE launching a batch of large graphs on GPU. The
    estimated peak per worker for the largest graph in the batch
    (`estimate_gpu_peak_bytes` - same formulas/mode as `select_gpu_mode` in
    smacof.py, worst case alpha>0) x `n_workers` +
    `reserved_other_bytes` (dwm/Windows, config
    exp3_graph_layout.gpu_reserved_other_bytes) is compared to `safety` x
    total VRAM (`torch.cuda.mem_get_info()[1]`). Exceeding it = RuntimeError
    with the recommended `large_graph_max_workers` (the max integer that
    fits) - instead of WDDM paging and > 6 min per iteration (a 2026-09-13
    incident, 10 h with no result). Always logs INFO with the numbers.

    `mem_get_info` can be passed in for tests without CUDA (returns (free,
    total) in bytes like torch.cuda.mem_get_info). Returns a dict with the estimate numbers."""
    from src.common.config import load_config
    from src.sammon.solvers.smacof import estimate_gpu_peak_bytes, resolve_gpu_dtype

    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}.")
    if not (0.0 < safety <= 1.0):
        raise ValueError(f"gpu_preflight_safety={safety} must be in (0, 1].")
    if mem_get_info is None:
        import torch

        mem_get_info = torch.cuda.mem_get_info
    gd_cfg = load_config()["sammon"]["gpu_dense"]
    itemsize = int(resolve_gpu_dtype(gpu_dtype).itemsize)
    per_worker, solver_path = estimate_gpu_peak_bytes(
        n_max, n_components, itemsize, tile_rows=int(gd_cfg["tile_rows"]),
        pinv_max_bytes=float(gd_cfg["pinv_max_bytes"]), resident_max_bytes=float(gd_cfg["resident_max_bytes"]),
        cuda_context_bytes=float(gd_cfg["cuda_context_bytes"]), w_const=False, lam=0.0,
    )
    _free, total = mem_get_info()
    total = int(total)
    budget = safety * total
    estimate_total = per_worker * n_workers + float(reserved_other_bytes)
    n_fit = int((budget - float(reserved_other_bytes)) // per_worker)
    info = {
        "n_max": int(n_max), "itemsize": itemsize, "solver_path": solver_path, "per_worker_bytes": int(per_worker),
        "n_workers": int(n_workers), "reserved_other_bytes": float(reserved_other_bytes),
        "estimate_total_bytes": float(estimate_total), "gpu_total_bytes": total, "budget_bytes": float(budget),
        "recommended_max_workers": n_fit,
    }
    logger.info(
        "VRAM preflight (large graphs, n_max=%d, %s, mode %s): estimate %.2f GB/worker x %d workers + %.2f GB other "
        "= %.2f GB vs. budget %.2f GB (%.0f %% of %.2f GB total); max workers that fit: %d.",
        n_max, gpu_dtype, solver_path, per_worker / 1e9, n_workers, reserved_other_bytes / 1e9, estimate_total / 1e9,
        budget / 1e9, safety * 100.0, total / 1e9, n_fit,
    )
    if estimate_total > budget:
        raise RuntimeError(
            f"VRAM preflight failed: the estimated peak {estimate_total / 1e9:.2f} GB "
            f"({per_worker / 1e9:.2f} GB/worker [n={n_max}, {gpu_dtype}, mode {solver_path}] x {n_workers} workers "
            f"+ {reserved_other_bytes / 1e9:.2f} GB other processes) exceeds the budget {budget / 1e9:.2f} GB "
            f"({safety:.2f} x {total / 1e9:.2f} GB total). Reduce exp3_graph_layout.large_graph_max_workers "
            f"to {max(n_fit, 1)} (recommended; max integer that fits: {n_fit}), or increase "
            "sammon.gpu_dense.*_max_bytes toward a more memory-efficient mode / reduce gpu_reserved_other_bytes "
            "per the actual state (nvidia-smi). No silent run with GPU memory paging."
        )
    return info


def main() -> None:
    mode = parse_mode_args("E3: graph layouts (shortest-path/resistance distance) across all graph datasets.")
    logger = get_logger(BASE_EXPERIMENT_NAME, mode=mode)
    cfg = resolve_experiment_config(BASE_EXPERIMENT_NAME, mode)
    EXPERIMENT_NAME = resolve_experiment_name(BASE_EXPERIMENT_NAME, mode)

    from src.methods.common import is_compatible
    from src.methods.registry import get_method

    n_components = cfg["n_components"]
    metric_keys, eval_kwargs = discover_metric_keys(n_components=n_components)
    column_keys = metric_keys + GRAPH_METRIC_KEYS + [
        "distance_metric", "base_graph", "n_nodes_original", "n_nodes_lcc", "selected_hyperparam",
        "community_source", "device",
    ]

    # K8: the size (LCC) of each graph is determined UPFRONT (without
    # computing the distance matrix, see `_graph_lcc_info`), so that the
    # number of seeds (`large_graph_seeds`) and skipping too-demanding
    # methods (`method_max_n`) can be decided before anything runs in the workers.
    large_threshold = int(cfg["large_graph_threshold_nodes"])
    large_graph_seeds = set(cfg["large_graph_seeds"])
    method_max_n: dict[str, int] = cfg.get("method_max_n", {}) or {}
    large_graph_max_workers = int(cfg["large_graph_max_workers"])
    resistance_device = str(cfg.get("resistance_device", "cpu"))
    # K8 follow-up task (documentation/2026-09-12_s2_koder_b.md): device
    # override ONLY for large graphs and ONLY for the SMACOF family of
    # methods (see `_override_method_device`) - the global
    # `methods.*.device` in common/config.yaml remains unchanged (E1 timings
    # must stay CPU).
    large_device = str(cfg.get("large_graph_device", "") or "")
    large_device_methods: set[str] = set(cfg.get("large_graph_device_methods", []) or [])
    # 2026-09-12 (documentation/2026-09-12_gpu_pcg_a_metriky_velkych_grafu.md):
    # per-method dtype override for large graphs on GPU (fail-loud value
    # validation via resolve_gpu_dtype, which is only called inside
    # smacof_solve_gpu - here we just store the string from the config, no
    # silent fallback to another value).
    large_gpu_dtype = str(cfg.get("large_graph_gpu_dtype", "float32") or "float32")

    graph_info: dict[str, tuple[int, int, str]] = {}
    for base_name in cfg["datasets"]:
        graph_info[base_name] = _graph_lcc_info(base_name)
        n_original, n_lcc, community_source = graph_info[base_name]
        logger.info(
            "%s: n_original=%d, n_lcc=%d, community_source=%s, large_graph=%s.",
            base_name, n_original, n_lcc, community_source, n_lcc > large_threshold,
        )

    # Fail-loud: if the config requires CUDA (for large graphs and/or for
    # resistance distance) AND at least one graph exists that would actually
    # use the override (n_lcc > large_threshold), CUDA availability is
    # checked ONCE, at the start - no silent fallback to CPU (per the spec).
    # The check is INTENTIONALLY conditioned on a large graph existing in
    # the current configuration (`cfg["datasets"]`) - `smoke`/`quick` modes
    # typically work only with small graphs (n_lcc <= large_threshold),
    # where the GPU branch is NEVER used (see `resistance_distance`:
    # `device` is only read for n > large_n_threshold), so they would
    # needlessly require a GPU even where the code does not need it at all
    # (bad for portability of the smoke test to machines without a GPU).
    any_large_graph = any(n_lcc > large_threshold for _, n_lcc, _ in graph_info.values())
    if any_large_graph and (large_device == "cuda" or resistance_device == "cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                f"config_experiments.yaml exp3_graph_layout requires CUDA for large graphs "
                f"(large_graph_device={large_device!r}, resistance_device={resistance_device!r}), "
                "but torch.cuda.is_available()==False on this machine. No silent fallback to CPU - "
                "fix the config (set 'cpu') or run on a machine with an available GPU."
            )

    def _effective_seeds(base_name: str) -> list[int]:
        if graph_info[base_name][1] <= large_threshold:
            return cfg["seeds"]
        return [s for s in cfg["seeds"] if s in large_graph_seeds]

    tasks: list[dict[str, Any]] = []
    n_skipped_max_n = 0

    for base_name in cfg["datasets"]:
        n_original, n_lcc, community_source = graph_info[base_name]
        seeds_for_graph = _effective_seeds(base_name)
        is_large_graph = n_lcc > large_threshold
        for distance_metric in cfg["distance_metrics"]:
            dataset_label = f"{base_name}__{distance_metric}"
            for method_name in cfg["distance_methods"]:
                method = get_method(method_name)
                if not is_compatible(method.accepts, "distance"):
                    logger.warning("Method '%s' does not support kind='distance', skipping for '%s'.", method_name, dataset_label)
                    continue
                max_n = method_max_n.get(method_name)
                if max_n is not None and n_lcc > int(max_n):
                    skip_extra = {
                        "distance_metric": distance_metric, "base_graph": base_name,
                        "n_nodes_original": n_original, "n_nodes_lcc": n_lcc,
                        "selected_hyperparam": "", "community_source": community_source, "device": "",
                    }
                    for seed in seeds_for_graph:
                        _write_skip_row(EXPERIMENT_NAME, dataset_label, method_name, seed, column_keys, skip_extra, logger)
                        n_skipped_max_n += 1
                    continue
                # K8 follow-up task: device override only for large graphs
                # AND only for methods in large_graph_device_methods (the
                # SMACOF family incl. sammon_alpha_pred) - see _override_method_device.
                device_override = large_device if (is_large_graph and large_device and method_name in large_device_methods) else None
                gpu_dtype_override = large_gpu_dtype if device_override else None
                for seed in seeds_for_graph:
                    tasks.append({
                        "dataset_name": dataset_label, "method_name": method_name, "seed": seed,
                        "base_name": base_name, "distance_metric": distance_metric,
                        "n_components": n_components, "eval_kwargs": eval_kwargs, "kind": "distance",
                        "large_n_threshold": large_threshold, "resistance_device": resistance_device,
                        "device_override": device_override, "gpu_dtype_override": gpu_dtype_override,
                    })
        for method_name in cfg["native_graph_methods"]:
            method = get_method(method_name)
            if not is_compatible(method.accepts, "graph"):
                logger.warning("Method '%s' does not support kind='graph', skipping for '%s'.", method_name, base_name)
                continue
            max_n = method_max_n.get(method_name)
            if max_n is not None and n_lcc > int(max_n):
                skip_extra = {
                    "distance_metric": "native", "base_graph": base_name,
                    "n_nodes_original": n_original, "n_nodes_lcc": n_lcc,
                    "selected_hyperparam": "", "community_source": community_source, "device": "",
                }
                for seed in seeds_for_graph:
                    _write_skip_row(EXPERIMENT_NAME, base_name, method_name, seed, column_keys, skip_extra, logger)
                    n_skipped_max_n += 1
                continue
            for seed in seeds_for_graph:
                tasks.append({
                    "dataset_name": base_name, "method_name": method_name, "seed": seed,
                    "base_name": base_name, "n_components": n_components, "eval_kwargs": eval_kwargs, "kind": "graph",
                })

    todo, n_done = filter_already_done(EXPERIMENT_NAME, tasks)
    logger.info(
        "%s (mode=%s): %d combinations already done (skipped), %d new to compute, %d explicitly skipped (skipped_max_n).",
        EXPERIMENT_NAME, mode, n_done, len(todo), n_skipped_max_n,
    )

    # K8: tasks on "large" graphs (n_lcc > large_threshold) run with a
    # limited number of workers (`large_graph_max_workers`) due to memory
    # requirements (resistance distance ~0.9 GB per matrix for n~10680) -
    # other tasks run with the full `parallel.n_workers` from common/config.yaml.
    def _is_large_task(t: dict[str, Any]) -> bool:
        return graph_info[t["base_name"]][1] > large_threshold

    todo_distance_normal = [t for t in todo if t["kind"] == "distance" and not _is_large_task(t)]
    todo_distance_large = [t for t in todo if t["kind"] == "distance" and _is_large_task(t)]
    todo_native_normal = [t for t in todo if t["kind"] == "graph" and not _is_large_task(t)]
    todo_native_large = [t for t in todo if t["kind"] == "graph" and _is_large_task(t)]

    n_ok1a, n_err1a = run_experiment_grid(EXPERIMENT_NAME, todo_distance_normal, _run_single_distance, column_keys, logger)
    # 2026-09-13: a VRAM preflight check before a batch of large graphs on
    # GPU (only when CUDA is actually used and the batch is not empty) - see preflight_gpu_memory
    if todo_distance_large and large_device == "cuda":
        n_max_large = max(graph_info[t["base_name"]][1] for t in todo_distance_large)
        preflight_gpu_memory(
            n_max_large, n_components, large_graph_max_workers, large_gpu_dtype,
            reserved_other_bytes=float(cfg["gpu_reserved_other_bytes"]), safety=float(cfg["gpu_preflight_safety"]),
            logger=logger,
        )
    n_ok1b, n_err1b = run_experiment_grid(
        EXPERIMENT_NAME, todo_distance_large, _run_single_distance, column_keys, logger, n_workers=large_graph_max_workers,
    )
    n_ok2a, n_err2a = run_experiment_grid(EXPERIMENT_NAME, todo_native_normal, _run_single_native, column_keys, logger)
    n_ok2b, n_err2b = run_experiment_grid(
        EXPERIMENT_NAME, todo_native_large, _run_single_native, column_keys, logger, n_workers=large_graph_max_workers,
    )

    n_ok = n_ok1a + n_ok1b + n_ok2a + n_ok2b
    n_err = n_err1a + n_err1b + n_err2a + n_err2b
    logger.info(
        "Done. Newly successful: %d, newly failed: %d, skipped (already done): %d, skipped (skipped_max_n): %d.",
        n_ok, n_err, n_done, n_skipped_max_n,
    )
    write_done_file(EXPERIMENT_NAME, logger, extra_info={"mode": mode, "n_skipped_max_n": n_skipped_max_n})


if __name__ == "__main__":
    main()
