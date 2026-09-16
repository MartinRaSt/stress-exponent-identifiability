# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-16
# License: see the LICENSE file in the repository root
#
# Shared function that disables computer standby/hibernation for the duration
# of a long experiment on Linux/macOS - the Linux/macOS equivalent of the
# src/common/no_sleep_on.bat + no_sleep_off.bat pair (which on Windows changes
# the powercfg standby/hibernate timeouts and then restores them). Here, instead
# of an on/off pair, the rest of the script run is wrapped ONCE with the
# `systemd-inhibit --what=sleep` command (if available) - systemd-inhibit
# releases the sleep inhibition itself as soon as the wrapped process ends
# (even after an error), so no paired "off" step is needed.
#
# Usage (in every calling run_*.sh, right after computing ROOT and BEFORE
# the rest of the script's work):
#   source "$ROOT/src/common/no_sleep.sh"
#   sammon_no_sleep_reexec "$0" "$@"
#
# If systemd-inhibit is available and the script is not yet running under it,
# the function re-runs (exec) the script wrapped in systemd-inhibit, and the
# original process is replaced by it (the caller's return code therefore
# remains the return code of the actual run). It sets the environment variable
# SAMMON_NO_SLEEP_ACTIVE so that a nested call (e.g. from a chain script that
# calls further run_*.sh scripts) does not wrap in systemd-inhibit multiple
# times - nesting is thus safe, just like with the original Windows
# no_sleep_on/off.bat pair.
# If systemd-inhibit is not available (typically macOS), only a warning is
# printed and the run continues WITHOUT the sleep inhibition - on macOS run
# long jobs manually under `caffeinate -i <command>` (see CLAUDE.md, section
# "Long computations and computer sleep").
sammon_no_sleep_reexec() {
    local script="$1"
    shift
    if [ -n "${SAMMON_NO_SLEEP_ACTIVE:-}" ]; then
        # already running under systemd-inhibit (either directly, or we were
        # launched from another run_*.sh that already provided the wrapper)
        return 0
    fi
    if command -v systemd-inhibit >/dev/null 2>&1; then
        export SAMMON_NO_SLEEP_ACTIVE=1
        exec systemd-inhibit --what=sleep --who="sammon" --why="long scientific computation" bash "$script" "$@"
    fi
    echo "[no_sleep] WARNING: systemd-inhibit is not available - computer sleep/hibernation is NOT disabled for this run. On macOS use 'caffeinate -i <command>' manually." >&2
}
