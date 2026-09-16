#!/usr/bin/env bash
set -euo pipefail
# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-13
# License: see the LICENSE file in the repository root
#
# E4 recompute after changing the init of new nodes (K14, jitter in src/sammon/temporal.py)
# and refresh of the tables, macros and figures:
#   1. backup of the full E4 outputs (CSV, DONE, embeddings/pertrans cache) into
#      results/data/backup_e4_<date>/ (nothing is deleted),
#   2. src/run_exp4_temporal.sh full   (7 datasets, ~70 min, resume from scratch),
#   3. src/run_main.sh full --no-figures (statistics, tables, numbers.tex, ~1 min),
#   4. src/run_main.sh full --figures-only (48 figures, ~4 min).
# Step return codes are written to results/logs/run_e4_prepocet_a_main.log.
# Run ONLY by the author manually. Sleep is disabled by the called run_*.sh scripts.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STAMP="$(date '+%Y%m%d_%H%M')"
LOG="results/logs/run_e4_prepocet_a_main.log"
BK="results/data/backup_e4_$STAMP"
mkdir -p "results/logs"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] START run_e4_prepocet_a_main (backup $BK)" >> "$LOG"

mkdir -p "$BK"
for f in exp4_temporal_results.csv exp4_temporal_DONE.txt exp4_relative.csv \
         exp4_stats.csv exp4_trajectories.csv; do
    [ -e "results/data/$f" ] && mv -f "results/data/$f" "$BK/" 2>/dev/null || true
done
if [ -d "results/data/embeddings/exp4_temporal" ]; then
    mv -f "results/data/embeddings/exp4_temporal" "$BK/embeddings_exp4_temporal" 2>/dev/null || true
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] backup done" >> "$LOG"

RC1=""; RC2=""; RC3=""

if "$ROOT/src/run_exp4_temporal.sh" full; then RC1=0; else RC1=$?; fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_exp4_temporal full rc=$RC1" >> "$LOG"

if [ "$RC1" = "0" ]; then
    if "$ROOT/src/run_main.sh" full --no-figures; then RC2=0; else RC2=$?; fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_main full --no-figures rc=$RC2" >> "$LOG"
fi

if [ "$RC1" = "0" ] && [ "$RC2" = "0" ]; then
    if "$ROOT/src/run_main.sh" full --figures-only; then RC3=0; else RC3=$?; fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_main full --figures-only rc=$RC3" >> "$LOG"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] END rc=$RC1/$RC2/$RC3" >> "$LOG"
echo ""
echo "Done. Return codes: exp4=$RC1 main_tables=$RC2 main_figures=$RC3  (log: $LOG)"
echo "Backup of the original E4 outputs: $BK"
exit 0
