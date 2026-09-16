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
Order-preserving parallel execution via ProcessPoolExecutor.map (never
as_completed, so results are deterministically ordered the same as the inputs).

2026-09-10 (Task 2, see
documentation/2026-09-10_oprava_sparse_cg_a_paralelismus.md): each worker
process gets `_worker_initializer` at startup, which limits the number of
BLAS library threads (OpenBLAS/MKL) to `parallel.blas_threads_per_worker`
from config.yaml - without this, each of the `parallel.n_workers` processes
used more threads for numpy operations on its own (OpenBLAS inside numpy),
and the CPU was only partially utilized with a small number of concurrent
processes.
"""
from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import parent_process
from typing import Callable, Iterable, Iterator, TypeVar

from src.common.config import load_config

A = TypeVar("A")
R = TypeVar("R")

# Names of the env vars controlling the number of threads of the individual
# BLAS/compute libraries used by numpy/scipy/sklearn (order does not matter).
_BLAS_ENV_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def resolve_n_workers(n_workers: int | None = None) -> int:
    """Return the number of workers: explicit value, else from config, else cpu_count-1."""
    if n_workers is not None:
        return max(1, n_workers)
    cfg = load_config()
    configured = cfg.get("parallel", {}).get("n_workers")
    if configured is not None:
        return max(1, int(configured))
    cpu = os.cpu_count() or 2
    return max(1, cpu - 1)


def resolve_blas_threads_per_worker(blas_threads: int | None = None) -> int:
    """Return the number of BLAS threads per worker process: explicit value,
    else `parallel.blas_threads_per_worker` from config.yaml. A missing key
    in the config is an error (fail loud), not a silent fallback to 1."""
    if blas_threads is not None:
        return max(1, int(blas_threads))
    cfg = load_config()
    try:
        configured = cfg["parallel"]["blas_threads_per_worker"]
    except KeyError as exc:
        raise KeyError("Missing key 'parallel.blas_threads_per_worker' in config.yaml.") from exc
    return max(1, int(configured))


def resolve_orphan_poll_sec() -> float:
    """Period for checking the parent process's liveness in a worker (seconds),
    key `parallel.orphan_poll_sec` in config.yaml. A missing key is an error
    (fail loud), not a silent fallback."""
    cfg = load_config()
    try:
        configured = cfg["parallel"]["orphan_poll_sec"]
    except KeyError as exc:
        raise KeyError("Missing key 'parallel.orphan_poll_sec' in config.yaml.") from exc
    value = float(configured)
    if value <= 0:
        raise ValueError(f"parallel.orphan_poll_sec={value} must be > 0.")
    return value


def _parent_watchdog(poll_sec: float) -> None:
    """Watchdog running in the worker process: as soon as the parent process
    stops existing, the worker terminates itself (`os._exit`).

    Reason (2026-09-16): on a HARD kill of the main process (Stop-Process,
    Task Manager, crash), `ProcessPoolExecutor` workers on Windows do not
    receive EOF on the task queue - each worker also holds the write end of
    the pipe, so they wait forever. Verified: after killing E4, 12
    `python.exe` processes kept running for minutes afterward. Ctrl+C in the
    console does not have this problem (the signal reaches the whole process
    group), but it cannot be relied upon for unattended runs.

    `os._exit` (not `sys.exit`) is deliberate: the thread must not wait for
    the computation in the worker's main thread to finish - the point is to
    free the CPU immediately."""
    while True:
        time.sleep(poll_sec)
        parent = parent_process()
        if parent is None or not parent.is_alive():
            os._exit(1)


def _worker_initializer(blas_threads: int) -> None:
    """Initializer called once at the start of every `ProcessPoolExecutor`
    worker process (Task 2b, 2026-09-10): limits BLAS library threads to
    `blas_threads`, so that `n_workers` concurrent processes together do not
    use more CPU threads than the author allows (see config.yaml, `parallel`
    section).

    Sets env vars (effective for BLAS libraries loaded AFTER this call) and
    additionally calls `threadpoolctl.threadpool_limits` (a second, more
    reliable layer - can limit an already-loaded OpenBLAS/MKL library
    regardless of the numpy/scipy/sklearn import order in the worker)."""
    for var in _BLAS_ENV_VARS:
        os.environ[var] = str(blas_threads)
    import threadpoolctl

    threadpoolctl.threadpool_limits(limits=blas_threads)

    # Orphan watchdog (see `_parent_watchdog`). The period comes from the
    # config - no magic constant in the code.
    threading.Thread(target=_parent_watchdog, args=(resolve_orphan_poll_sec(),), daemon=True).start()


def run_parallel_map(
    func: Callable[[A], R],
    args: list[A],
    n_workers: int | None = None,
    chunksize: int = 1,
    blas_threads_per_worker: int | None = None,
) -> Iterator[R]:
    """Run `func` over all elements of `args` via `ProcessPoolExecutor.map`.

    Results are returned in the same order as the input `args` (map is
    order-preserving, unlike as_completed), which is required for
    result determinism.

    Each worker process is limited at startup to `blas_threads_per_worker`
    BLAS threads (see `_worker_initializer`), so that `n_workers` concurrent
    processes together do not exceed the thread budget allowed by the author.
    """
    workers = resolve_n_workers(n_workers)
    if workers == 1:
        # sequential run without a process pool - easier debugging, same order
        for a in args:
            yield func(a)
        return
    blas_threads = resolve_blas_threads_per_worker(blas_threads_per_worker)
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_initializer, initargs=(blas_threads,)) as executor:
        try:
            for result in executor.map(func, args, chunksize=chunksize):
                yield result
        except BaseException:
            # The consumer stopped pulling results (exception while writing
            # to CSV, GeneratorExit, KeyboardInterrupt): cancel all tasks
            # NOT YET STARTED. Without this, the `with` block would wait on
            # exit (shutdown(wait=True)) until the ENTIRE queue finishes -
            # i.e. hours of computation whose results nobody writes anymore
            # (case 2026-09-12, E1 resume with a stale CSV schema).
            executor.shutdown(wait=False, cancel_futures=True)
            raise
