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
Unified logging to both console and file results/logs/<experiment>_<timestamp>.log,
plus wall-clock timing of individual experiment steps.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Iterator

from src.common.config import get_mode_path, ensure_dir, get_path


def get_logger(experiment: str, level: int = logging.INFO, mode: str | None = None) -> logging.Logger:
    """Create/return a logger for the given experiment, writing to both console and file.

    Log file: results/logs/[<mode>/]<experiment>_<YYYYmmdd_HHMMSS>.log
    (`mode` quick/smoke -> mode subdirectory, None/full -> root; see
    src/common/config.py::get_mode_path). On repeated calls with the same
    name, the already-initialized logger is returned (mode is then ignored).
    """
    logger = logging.getLogger(f"sammon.{experiment}")
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        # logger already initialized (e.g. repeated call within the same process)
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    logs_dir = ensure_dir(get_mode_path("results_logs_dir", mode or "full"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"{experiment}_{timestamp}.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.info("Log file: %s", log_path)
    return logger


@contextmanager
def wall_clock(logger: logging.Logger, task_name: str) -> Iterator[None]:
    """Context manager measuring and logging the wall-clock time of a code block."""
    start = time.perf_counter()
    logger.info("Start: %s", task_name)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("End: %s (wall-clock %.3f s)", task_name, elapsed)
