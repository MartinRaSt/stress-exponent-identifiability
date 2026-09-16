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
Standalone loader for `src/experiments/config_experiments.yaml` (parameters
of experiments E1-E5), separate from `src/common/config.py` (which loads the
main `src/common/config.yaml` with a `sammon:` section maintained concurrently
by another agent - this file does NOT touch it).

Each experiment block in config_experiments.yaml has an optional `quick:`
subkey which, for the `--quick` mode, overrides same-named keys in the main
block (a smaller subset of datasets/methods/seeds/n for a quick test that
takes a few minutes).
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

from src.common.config import get_project_root

_CONFIG_FILENAME = "config_experiments.yaml"


@functools.lru_cache(maxsize=1)
def load_experiments_config(config_path: Path | None = None) -> dict[str, Any]:
    """Loads config_experiments.yaml and returns it as a dict (cached)."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent / _CONFIG_FILENAME
    if not config_path.exists():
        raise FileNotFoundError(
            f"Experiment configuration file not found: {config_path}. "
            "A missing configuration is an error, not a reason to use default values."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        raise ValueError(f"Experiment configuration file is empty or invalid: {config_path}")
    return cfg


_OVERRIDE_KEYS = ("quick", "smoke")


def _resolve_node(node: Any, target_key: str | None) -> Any:
    """Recursively removes the 'quick'/'smoke' keys at all levels and (if
    `target_key` is not None) overrides the keys at each level with values
    from its own `<target_key>:` subkey (if it exists) - this also works for
    nested blocks, e.g. exp2_solver_scaling.convergence.smoke, not just at
    the level of the whole experiment."""
    if not isinstance(node, dict):
        return node
    override = node.get(target_key, {}) if target_key is not None else {}
    merged = {k: _resolve_node(v, target_key) for k, v in node.items() if k not in _OVERRIDE_KEYS}
    if isinstance(override, dict):
        merged.update(override)
    return merged


def resolve_experiment_config(experiment_key: str, mode: str) -> dict[str, Any]:
    """Returns the configuration for a given experiment and mode ('quick'/'smoke'/'full').

    Mode 'full' returns the block unchanged (recursively, without any
    'quick'/'smoke' subkeys). Mode 'quick'/'smoke' recursively overlays
    same-named keys at EVERY level with values from its own
    'quick:'/'smoke:' subkey (e.g. a smaller list of datasets/methods/seeds,
    a smaller n_max, but also nested subparts like
    exp2_solver_scaling.convergence.smoke) - keys not listed in the
    respective block keep the value from 'full'.

    Convention for 'smoke' (see the spec): typically only COUNT-like keys
    are overridden (seeds->[0], n_max, max_iter/epochs, alpha_grid,
    lambda_grid reduced to small sets), WHILE the 'datasets'/'methods' lists
    are NOT listed in the 'smoke:' block (they stay full, so smoke tests
    EVERY combination)."""
    if mode not in ("quick", "smoke", "full"):
        raise ValueError(f"Unknown mode mode='{mode}' (expected 'quick', 'smoke', or 'full').")
    cfg = load_experiments_config()
    if experiment_key not in cfg:
        raise KeyError(f"Block for experiment '{experiment_key}' is missing in config_experiments.yaml.")
    block = cfg[experiment_key]
    target_key = None if mode == "full" else mode
    return _resolve_node(block, target_key)


def get_project_root_path() -> Path:
    """Re-export get_project_root for convenient import in experiments."""
    return get_project_root()
