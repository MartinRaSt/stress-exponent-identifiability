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
A thin wrapper around tqdm with a shared configuration (ETA, width, redraw
interval) loaded from config.yaml, so all scripts report progress the same way.
"""
from __future__ import annotations

from typing import Iterable, Iterator, TypeVar

from tqdm import tqdm

from src.common.config import load_config

T = TypeVar("T")


def progress_iter(iterable: Iterable[T], desc: str = "", total: int | None = None) -> Iterator[T]:
    """Wrap an iterable in a tqdm progress bar with ETA per the configuration.

    Parameters:
        iterable: the input iterable.
        desc: label shown before the progress bar.
        total: number of items, if it cannot be inferred from `len(iterable)`.
    """
    cfg = load_config()
    prog_cfg = cfg.get("progress", {})
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        mininterval=prog_cfg.get("mininterval_sec", 0.5),
        ncols=prog_cfg.get("ncols", 100),
    )
