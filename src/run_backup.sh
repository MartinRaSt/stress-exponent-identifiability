#!/usr/bin/env bash
set -euo pipefail
# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Created: 2026-09-18
# License: see the LICENSE file in the repository root
#
# Writes two dated archives NEXT TO the project directory (never inside it,
# so a backup never archives itself):
#   Sammon_sources_<date>.zip         sources: code, article text, SVG figure
#                                    sources, docs, git history (a few MB)
#   Sammon_data_<date>.zip  the measured experiment CSVs only
# Everything a script can rebuild (figures, tables, PDFs, embeddings, the
# dataset cache, the venv) is left out - see src/tools/make_backup.py.
#
# Usage: src/run_backup.sh [sources|data|both]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

PYTHON="venv/python.exe"
[ -x "$PYTHON" ] || PYTHON="venv/bin/python"

STAMP="$(date +%Y-%m-%d)"
OUTDIR=".."
PROFILE="${1:-both}"

backup_sources() {
    echo "=== backup: sources ==="
    "$PYTHON" -m src.tools.make_backup sources "$OUTDIR/Sammon_sources_$STAMP.zip"
}

backup_data() {
    echo "=== backup: measured data ==="
    "$PYTHON" -m src.tools.make_backup data "$OUTDIR/Sammon_data_$STAMP.zip"
}

case "$PROFILE" in
    sources) backup_sources ;;
    data)    backup_data ;;
    both)    backup_sources; backup_data ;;
    *)       echo "ERROR: unknown profile '$PROFILE' (use: sources | data | both)."; exit 2 ;;
esac
