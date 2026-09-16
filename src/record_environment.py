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
Records the computational environment for experiment reproducibility.

Writes `results/data/environment.json` with Python and key library versions,
the complete `pip freeze` output, GPU info (torch/CUDA), CPU and OS info, and
the record timestamp. Runs independently of the experiments
(`src/run_record_environment.bat`) so that every result in `results/` can be
linked to the exact state of the environment.
"""
from __future__ import annotations

import datetime as _dt
import importlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "results" / "data" / "environment.json"
KEY_PACKAGES = [
    "numpy", "scipy", "pandas", "sklearn", "networkx", "matplotlib",
    "umap", "pacmap", "trimap", "numba", "torch", "tqdm", "yaml",
]


def package_versions() -> dict:
    """Returns the versions of key packages (a missing package is recorded as None)."""
    versions = {}
    for name in KEY_PACKAGES:
        try:
            module = importlib.import_module(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[name] = None
    return versions


def pip_freeze() -> list[str]:
    """Returns the complete `pip freeze` output from the interpreter running this script."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def gpu_info() -> dict:
    """Returns torch/CUDA and GPU info; without torch returns an empty record with a reason."""
    try:
        import torch
    except ImportError:
        return {"available": False, "reason": "torch not installed"}
    info = {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["gpu_total_memory_bytes"] = int(props.total_memory)
        info["gpu_count"] = torch.cuda.device_count()
    return info


def cpu_info() -> dict:
    """Returns basic CPU info and the number of logical cores."""
    return {
        "processor": platform.processor(),
        "machine": platform.machine(),
        "logical_cores": os.cpu_count(),
    }


def collect_environment() -> dict:
    """Assembles the complete environment record."""
    return {
        "recorded_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
        },
        "cpu": cpu_info(),
        "gpu": gpu_info(),
        "key_packages": package_versions(),
        "pip_freeze": pip_freeze(),
    }


def main() -> None:
    """Writes the environment record to `results/data/environment.json` and prints a summary."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    env = collect_environment()
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(env, handle, indent=2, ensure_ascii=False)
    print(f"Environment recorded to {OUTPUT_PATH}")
    print(f"Python {sys.version.split()[0]}, packages: {env['key_packages']}")
    print(f"GPU: {env['gpu']}")


if __name__ == "__main__":
    main()
