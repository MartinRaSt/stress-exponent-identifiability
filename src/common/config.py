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
Loading of the project's central configuration (src/common/config.yaml) and
conversion of relative paths to absolute paths rooted at the project root.
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

_CONFIG_FILENAME = "config.yaml"


def get_project_root() -> Path:
    """Return the absolute path to the project root (the folder containing CLAUDE.md).

    Searches upward from this file's location so it works regardless of the
    current working directory the script is run from.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "CLAUDE.md").exists():
            return parent
    # fallback: src/common/config.py -> src/common -> src -> root
    return here.parents[2]


@functools.lru_cache(maxsize=1)
def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load config.yaml and return it as a dict. The result is cached.

    Parameters:
        config_path: optional explicit path to the config; otherwise
            src/common/config.yaml at the project root is used.
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parent / _CONFIG_FILENAME
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            "A missing config is an error, not a reason to fall back to defaults."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        raise ValueError(f"Config file is empty or invalid: {config_path}")
    return cfg


def get_path(key: str, config: dict[str, Any] | None = None) -> Path:
    """Return the absolute Path for a key from the `paths` section of config.yaml.

    Example: get_path("results_data_dir") -> D:/.../Sammon/results/data
    """
    cfg = config if config is not None else load_config()
    try:
        rel = cfg["paths"][key]
    except KeyError as exc:
        raise KeyError(f"Key '{key}' is not defined in the 'paths' section of config.yaml") from exc
    return get_project_root() / rel


def ensure_dir(path: Path) -> Path:
    """Create the directory (including parents) if it does not exist yet, and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


# allowed run modes (see src/experiments/exp_common.py::add_mode_args);
# 'full' = production outputs (results/* root, clanek/generated), 'quick' and
# 'smoke' = test outputs EXCLUSIVELY in the mode's subdirectory
MODES = ("full", "quick", "smoke")


def _check_mode(mode: str) -> str:
    """Fail-loud mode check (no silent fallback to 'full')."""
    if mode not in MODES:
        raise ValueError(f"Unknown mode '{mode}' - allowed values: {MODES}.")
    return mode


def get_mode_path(key: str, mode: str, config: dict[str, Any] | None = None) -> Path:
    """The single shared function for the MODE-specific output directory path
    (author's rule from 2026-09-13, "Separating smoke/quick from full outputs",
    ~/.claude/CLAUDE.md): for 'full' it returns the root `paths.<key>` (e.g.
    results/tables), for 'quick'/'smoke' a subdirectory `paths.<key>/<mode>`
    (results/tables/smoke). The directory is NOT created (see `ensure_dir`).

    Example: get_mode_path("results_tables_dir", "smoke") -> .../results/tables/smoke
    """
    base = get_path(key, config)
    return base if _check_mode(mode) == "full" else base / mode


def get_tables_dir(mode: str) -> Path:
    """Mode-specific tables directory (results/tables[/<mode>]), created."""
    return ensure_dir(get_mode_path("results_tables_dir", mode))


def get_generated_dir(mode: str) -> Path:
    """Directory for generated number macros (`numbers.tex`): 'full' ->
    `paths.article_generated_dir` (clanek/generated), 'quick'/'smoke' ->
    results/tables/<mode>/ (smoke/quick must NEVER write into the article)."""
    if _check_mode(mode) == "full":
        return ensure_dir(get_path("article_generated_dir"))
    return get_tables_dir(mode)
