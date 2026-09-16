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
Iterates over all registered datasets, downloads/loads whatever is missing,
and writes an overview table to results/tables/datasets.csv and datasets.tex
(booktabs). Datasets that fail to load (missing network source, missing
local file for DBLP, etc.) are recorded with status 'error' and a reason -
never replaced with fabricated values.

Run: venv\\python.exe -m src.datasets.list_datasets
or: src\\run_list_datasets.bat
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.common.config import get_tables_dir
from src.common.logging_utils import get_logger, wall_clock
from src.common.progress import progress_iter
from src.datasets.registry import list_registered_datasets, load_dataset
from src.experiments.exp_common import parse_mode_args

OUTPUT_CSV_NAME = "datasets.csv"
OUTPUT_TEX_NAME = "datasets.tex"


def _describe(name: str) -> dict[str, Any]:
    """Try to load the dataset and return a row of the description table (or status='error')."""
    try:
        ds = load_dataset(name)
    except Exception as exc:  # intentionally broad - every loader can fail differently (network, missing file, ...)
        return {
            "name": name, "n": None, "d": None, "n_classes": None, "kind": None,
            "source": None, "standardized": None, "status": "error", "error": f"{type(exc).__name__}: {exc}",
        }

    source = ds.meta.get("source") or ds.meta.get("citation") or ""
    return {
        "name": name,
        "n": ds.n_samples,
        "d": ds.n_features,
        "n_classes": ds.n_classes,
        "kind": ds.kind,
        "source": source,
        "standardized": bool(ds.meta.get("standardized", False)),
        "status": "ok",
        "error": "",
    }


def _write_tex_table(df_ok: pd.DataFrame, tex_path: Path) -> None:
    """Write a booktabs LaTeX table listing the successfully loaded datasets."""
    lines = [
        "% Automaticky generovano: src/datasets/list_datasets.py - needit rucne.",
        "\\begin{tabular}{lrrrlcl}",
        "\\toprule",
        "Dataset & $n$ & $d$ & Classes & Type & Std. & Source \\\\",
        "\\midrule",
    ]
    for _, row in df_ok.iterrows():
        d_str = "" if pd.isna(row["d"]) else str(int(row["d"]))
        c_str = "" if pd.isna(row["n_classes"]) else str(int(row["n_classes"]))
        name = str(row["name"]).replace("_", "\\_")
        source = str(row["source"]).replace("_", "\\_").replace("&", "\\&")
        std_str = "ano" if bool(row.get("standardized", False)) else "ne"
        lines.append(f"{name} & {int(row['n'])} & {d_str} & {c_str} & {row['kind']} & {std_str} & {source} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Iterate over the dataset registry, build and save the overview table to
    results/tables/[<mode>/]datasets.csv/.tex (--quick/--smoke only into the
    mode's subdirectory, see src/common/config.py::get_tables_dir)."""
    mode = parse_mode_args("Overview table of registered datasets (results/tables/[<mode>/]datasets.csv/.tex).")
    logger = get_logger("list_datasets", mode=mode)
    names = list_registered_datasets()
    logger.info("Registered %d datasets: %s", len(names), ", ".join(names))

    rows: list[dict[str, Any]] = []
    with wall_clock(logger, f"loading/downloading {len(names)} datasets"):
        for name in progress_iter(names, desc="datasets", total=len(names)):
            row = _describe(name)
            rows.append(row)
            if row["status"] == "error":
                logger.warning("Failed to load dataset '%s': %s", name, row["error"])
            else:
                logger.info("Dataset '%s': n=%s d=%s classes=%s kind=%s", name, row["n"], row["d"], row["n_classes"], row["kind"])

    df = pd.DataFrame(rows)
    tables_dir = get_tables_dir(mode)
    csv_path = tables_dir / OUTPUT_CSV_NAME
    df.to_csv(csv_path, index=False)
    logger.info("Saved: %s", csv_path)

    df_ok = df[df["status"] == "ok"].reset_index(drop=True)
    tex_path = tables_dir / OUTPUT_TEX_NAME
    _write_tex_table(df_ok, tex_path)
    logger.info("Saved: %s", tex_path)

    n_ok, n_err = (df["status"] == "ok").sum(), (df["status"] == "error").sum()
    logger.info("Done: %d succeeded, %d failed.", n_ok, n_err)
    if n_err:
        failed_names = df.loc[df["status"] == "error", "name"].tolist()
        logger.warning("Unavailable datasets (reported by name, not fabricated): %s", ", ".join(failed_names))


if __name__ == "__main__":
    main()
