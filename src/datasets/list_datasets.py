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

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.common.config import get_tables_dir
from src.common.display_labels import display_label
from src.common.logging_utils import get_logger, wall_clock
from src.common.progress import progress_iter
from src.datasets.registry import list_registered_datasets, load_dataset
from src.experiments.exp_common import add_mode_args, resolve_mode

OUTPUT_CSV_NAME = "datasets.csv"
OUTPUT_TEX_NAME = "datasets.tex"
# Brief variant (2026-09-18, supplement length reduction): same rows, drops
# the "Source" column - full provenance stays in OUTPUT_CSV_NAME/OUTPUT_TEX_NAME,
# this is only for the supplement's catalogue table (see clanek*/supplement/
# sections/s6_full_tables.tex), which does not need a repeated URL/module-path
# column for every one of the ~75 datasets to be useful as a reader overview.
OUTPUT_TEX_BRIEF_NAME = "datasets_brief.tex"


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
    """Write a `longtable` overview of the successfully loaded datasets (one
    row per dataset, so the table grows taller with the dataset count -
    2026-09-18 fix: a plain `table`+`tabular` overflowed the supplement page
    height, "Float too large", see
    `src/experiments/report_tables.py::longtable_head_foot` for why
    `longtable` is used here instead of the `write_booktabs_tex` DataFrame
    machinery this module otherwise mirrors by hand). `Source` is a raw
    registry URL/module path, potentially long and unbreakable at any
    character in a plain `l` column - a `p{}` column with `\\fp{}` (this
    project's `\\url{}` wrapper, see the supplement preamble) lets it wrap
    across lines instead of forcing the table width far past the page.
    `\\fp{}`/`\\url{}` reads its argument with every character's catcode
    forced to 'other', so `source` is passed RAW (unescaped) here, unlike
    the other text cells - a literal backslash inserted by `_`-escaping
    would render as a visible backslash inside `\\url{}`."""
    from src.experiments.report_tables import escape_latex_label, longtable_head_foot

    header_line = "Dataset & $n$ & $d$ & Classes & Type & Std. & Source \\\\"
    head, foot = longtable_head_foot([header_line], n_cols=7)
    # `foot` (\endfoot/\endlastfoot) MUST come before the body rows - see the
    # 2026-09-18 bugfix note in report_tables.write_booktabs_tex.
    lines = [
        "% AUTO-GENERATED: src/datasets/list_datasets.py - DO NOT EDIT BY HAND.",
        "\\begin{longtable}{lrrrlc>{\\raggedright\\arraybackslash}p{7cm}}",
        "\\caption{Overview of the datasets.}\\label{sup:tab:datasets_overview}\\\\",
        head,
        foot,
    ]
    for _, row in df_ok.iterrows():
        d_str = "" if pd.isna(row["d"]) else str(int(row["d"]))
        c_str = "" if pd.isna(row["n_classes"]) else str(int(row["n_classes"]))
        name = escape_latex_label(display_label(str(row["name"]), "dataset"))
        std_str = "yes" if bool(row.get("standardized", False)) else "no"
        lines.append(f"{name} & {int(row['n'])} & {d_str} & {c_str} & {row['kind']} & {std_str} & \\fp{{{row['source']}}} \\\\")
    lines.append("\\end{longtable}")
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_tex_table_brief(df_ok: pd.DataFrame, tex_path: Path) -> None:
    """Same rows as `_write_tex_table`, without the `Source` column (2026-09-18,
    supplement length reduction - see `OUTPUT_TEX_BRIEF_NAME`). Full provenance
    per dataset stays available in `OUTPUT_CSV_NAME`/`OUTPUT_TEX_NAME`; the
    supplement text points readers there."""
    from src.experiments.report_tables import escape_latex_label, longtable_head_foot

    header_line = "Dataset & $n$ & $d$ & Classes & Type & Std. \\\\"
    head, foot = longtable_head_foot([header_line], n_cols=6)
    lines = [
        "% AUTO-GENERATED: src/datasets/list_datasets.py (_write_tex_table_brief) - DO NOT EDIT BY HAND.",
        "% Same rows as datasets.tex, without the Source column; full provenance in datasets.csv/datasets.tex.",
        "\\begin{longtable}{lrrrlc}",
        "\\caption{Overview of the datasets (brief; full table with sources: \\texttt{results/tables/datasets.csv}).}\\label{sup:tab:datasets_overview}\\\\",
        head,
        foot,
    ]
    for _, row in df_ok.iterrows():
        d_str = "" if pd.isna(row["d"]) else str(int(row["d"]))
        c_str = "" if pd.isna(row["n_classes"]) else str(int(row["n_classes"]))
        name = escape_latex_label(display_label(str(row["name"]), "dataset"))
        std_str = "yes" if bool(row.get("standardized", False)) else "no"
        lines.append(f"{name} & {int(row['n'])} & {d_str} & {c_str} & {row['kind']} & {std_str} \\\\")
    lines.append("\\end{longtable}")
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Iterate over the dataset registry, build and save the overview tables to
    results/tables/[<mode>/]datasets.csv/.tex/_brief.tex (--quick/--smoke only
    into the mode's subdirectory, see src/common/config.py::get_tables_dir).

    `--tables-only` re-typesets datasets.tex/datasets_brief.tex from the
    already-saved datasets.csv without reloading/redownloading any dataset
    (fast, safe to run repeatedly; used e.g. after adding the brief-table
    formatting so a full re-run is not required)."""
    parser = argparse.ArgumentParser(description="Overview table of registered datasets (results/tables/[<mode>/]datasets.csv/.tex).")
    add_mode_args(parser)
    parser.add_argument("--tables-only", action="store_true",
                         help="only re-typeset datasets.tex/datasets_brief.tex from the existing datasets.csv; does not reload/redownload datasets")
    args = parser.parse_args()
    mode = resolve_mode(args)
    logger = get_logger("list_datasets", mode=mode)
    tables_dir = get_tables_dir(mode)

    if args.tables_only:
        csv_path = tables_dir / OUTPUT_CSV_NAME
        if not csv_path.exists():
            raise FileNotFoundError(f"{csv_path} does not exist - run this script without --tables-only first.")
        df = pd.read_csv(csv_path)
        df_ok = df[df["status"] == "ok"].reset_index(drop=True)
        _write_tex_table(df_ok, tables_dir / OUTPUT_TEX_NAME)
        _write_tex_table_brief(df_ok, tables_dir / OUTPUT_TEX_BRIEF_NAME)
        logger.info("--tables-only: re-typeset %s and %s from %s (datasets NOT reloaded).",
                    OUTPUT_TEX_NAME, OUTPUT_TEX_BRIEF_NAME, csv_path)
        return

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
    csv_path = tables_dir / OUTPUT_CSV_NAME
    df.to_csv(csv_path, index=False)
    logger.info("Saved: %s", csv_path)

    df_ok = df[df["status"] == "ok"].reset_index(drop=True)
    tex_path = tables_dir / OUTPUT_TEX_NAME
    _write_tex_table(df_ok, tex_path)
    logger.info("Saved: %s", tex_path)
    tex_brief_path = tables_dir / OUTPUT_TEX_BRIEF_NAME
    _write_tex_table_brief(df_ok, tex_brief_path)
    logger.info("Saved: %s", tex_brief_path)

    n_ok, n_err = (df["status"] == "ok").sum(), (df["status"] == "error").sum()
    logger.info("Done: %d succeeded, %d failed.", n_ok, n_err)
    if n_err:
        failed_names = df.loc[df["status"] == "error", "name"].tolist()
        logger.warning("Unavailable datasets (reported by name, not fabricated): %s", ", ".join(failed_names))


if __name__ == "__main__":
    main()
