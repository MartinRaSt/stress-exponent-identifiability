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
Unified seeding of all randomness sources used in the project, so that all
experiments are deterministic and reproducible.
"""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Set the seed for `random`, `numpy` and (if available) `torch`.

    Call at the start of every experiment run / worker process, so results
    are deterministic given the same seed from the configuration.
    """
    if seed is None:
        raise ValueError("Seed must not be None - everything must be seeded from the configuration.")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # deterministic algorithms where supported (without a hard crash)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def make_rng(seed: int) -> np.random.Generator:
    """Return a new independent `numpy.random.Generator` derived from the seed.

    Use wherever a local generator is needed (e.g. subsampling), without
    affecting the global `numpy.random` state.
    """
    if seed is None:
        raise ValueError("Seed must not be None - everything must be seeded from the configuration.")
    return np.random.default_rng(seed)
