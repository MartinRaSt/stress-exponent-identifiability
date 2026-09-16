#!/usr/bin/env bash
set -euo pipefail
# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-15
# License: see the LICENSE file in the repository root
#
# ONE SCRIPT FOR THE WHOLE UNATTENDED RUN (K15/K16, 2026-09-15).
# Runs in this order:
#   0. (full only) backup and MOVE OUT the full E4 outputs into results/data/backup_e4_<stamp>/
#      - nothing is deleted, only moved; forces a full recompute of E4 with the new
#        neighborhood preservation metric columns (trust_k*, cont_k*, jacc_k*) for all 3 seeds
#        and with the corrected permutation p-value (1+count)/(1+B).
#   1. src/run_exp4_temporal.sh            - full E4 refit (~70 min, resumable)
#   2. src/run_exp4_neighbor_metrics.sh    - neighborhood metrics over trajectories (minutes)
#   3. src/run_main.sh --no-figures        - statistics, tables, clanek/generated/numbers.tex
#   4. src/run_main.sh --figures-only      - figures
#
# PARAMETER: full (default) | smoke | quick
#   smoke/quick does NOT move any full data and only writes to results/.../<mode>/.
#
# RESTART AFTER A CRASH: the script creates the marker results/data/exp4_refit_started.flag.
#   If it exists, step 0 is SKIPPED - a repeated run thus continues where the
#   run left off (resume from the checkpoint) instead of discarding rows already computed.
#   The marker is deleted only after the whole chain has completed successfully.
#
# COMPUTER SLEEP: disabled and restored by the called run_*.sh scripts (each step separately via
#   src/common/no_sleep.sh). This wrapper script deliberately does NOT call
#   sammon_no_sleep_reexec itself - a nested call would, on Windows, save the already-disabled
#   state as the "original" value; on systemd-inhibit this is harmless (see
#   no_sleep.sh, idempotent), but we keep the same structure as the .bat.
#
# Log: results/logs/run_vse_20260915.log (return codes of each step).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-full}"
case "$MODE" in
    full|smoke|quick) ;;
    *)
        echo "ERROR: unknown mode \"$MODE\". Use: full or smoke or quick" >&2
        exit 2
        ;;
esac

LOG="results/logs/run_vse_20260915.log"
FLAG="results/data/exp4_refit_started.flag"
mkdir -p "results/logs"

STAMP="$(date '+%Y%m%d_%H%M')"
BK="results/data/backup_e4_$STAMP"

RC1="-"; RC2="-"; RC3="-"; RC4="-"

echo "" >> "$LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] START run_vse_20260915 mode=$MODE" >> "$LOG"
echo "=== Sammon: full run, mode=$MODE - start $(date '+%Y-%m-%d %H:%M:%S') ==="

# ---- step 0: backup and move out the full E4 outputs (full only, only on the first run)
if [ "$MODE" = "full" ]; then
    if [ -f "$FLAG" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] step 0 SKIPPED - marker $FLAG exists, continuing the interrupted run" >> "$LOG"
        echo "[0/4] Backup skipped - resuming an interrupted run, continuing via resume."
    else
        echo "[0/4] Backing up and moving out the full E4 outputs to $BK ..."
        mkdir -p "$BK"
        for f in exp4_temporal_results.csv exp4_temporal_DONE.txt exp4_trajectories.csv \
                 exp4_stats.csv exp4_stats_partial.csv exp4_relative.csv \
                 exp4_neighbor_metrics_results.csv exp4_neighbor_metrics_DONE.txt \
                 exp4_neighbor_stats.csv; do
            [ -e "results/data/$f" ] && mv -f "results/data/$f" "$BK/" 2>/dev/null || true
        done
        if [ -d "results/data/embeddings/exp4_temporal" ]; then
            mv -f "results/data/embeddings/exp4_temporal" "$BK/embeddings_exp4_temporal" 2>/dev/null || true
        fi
        echo "moved out $(date '+%Y-%m-%d %H:%M:%S')" > "$FLAG"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] step 0 done, backup $BK" >> "$LOG"
    fi
fi

# ---- step 1: E4 refit
echo "[1/4] E4 temporal experiment, mode=$MODE - the longest step, roughly an hour for full ..."
if "$ROOT/src/run_exp4_temporal.sh" "$MODE"; then RC1=0; else RC1=$?; fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_exp4_temporal $MODE rc=$RC1" >> "$LOG"

if [ "$RC1" = "0" ]; then
    # ---- step 2: neighborhood metrics
    echo "[2/4] Neighborhood preservation metrics over trajectories, mode=$MODE ..."
    if "$ROOT/src/run_exp4_neighbor_metrics.sh" "$MODE"; then RC2=0; else RC2=$?; fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_exp4_neighbor_metrics $MODE rc=$RC2" >> "$LOG"
fi

if [ "$RC1" = "0" ] && [ "$RC2" = "0" ]; then
    # ---- step 3: statistics and tables
    echo "[3/4] Statistics, tables and number macros, mode=$MODE ..."
    if "$ROOT/src/run_main.sh" "$MODE" --no-figures; then RC3=0; else RC3=$?; fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_main $MODE --no-figures rc=$RC3" >> "$LOG"
fi

if [ "$RC1" = "0" ] && [ "$RC2" = "0" ] && [ "$RC3" = "0" ]; then
    # ---- step 4: figures
    echo "[4/4] Figures, mode=$MODE ..."
    if "$ROOT/src/run_main.sh" "$MODE" --figures-only; then RC4=0; else RC4=$?; fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_main $MODE --figures-only rc=$RC4" >> "$LOG"
fi

if [ "$MODE" = "full" ] && [ "$RC1" = "0" ] && [ "$RC2" = "0" ] && [ "$RC3" = "0" ] && [ "$RC4" = "0" ] && [ -f "$FLAG" ]; then
    rm -f "$FLAG" 2>/dev/null || true
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] END run_vse_20260915 rc=$RC1/$RC2/$RC3/$RC4" >> "$LOG"
echo ""
echo "=== End $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "Return codes: E4=$RC1  neighborhood=$RC2  tables=$RC3  figures=$RC4   -- 0 means OK"
echo "Log: $LOG"
if [ "$MODE" = "full" ]; then
    echo "Backup of the original E4 outputs: $BK"
fi
if [ -f "$FLAG" ]; then
    echo "WARNING: the run did NOT finish completely - after fixing, run the script AGAIN, it will resume the interrupted work via marker $FLAG"
fi
exit 0
