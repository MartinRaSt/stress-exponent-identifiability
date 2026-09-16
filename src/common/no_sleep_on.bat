@echo off
REM Project: Sammon mapping revisited - adaptive weighting, a scalable solver
REM          and a temporal extension (alpha-Sammon)
REM Authors: Martin Radvansky <martin.radvansky@vsb.cz>
REM          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
REM Affiliation: Department of Computer Science, Faculty of Electrical Engineering
REM              and Computer Science, VSB - Technical University of Ostrava,
REM              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
REM Created: 2026-09-11
REM License: see the LICENSE file in the repository root
REM
REM Reads and saves the original AC standby/hibernate timeouts (in minutes, into
REM the env vars SAMMON_STANDBY_AC_ORIG / SAMMON_HIBERNATE_AC_ORIG) and temporarily
REM disables computer standby/hibernation (0 = never) for the duration of a long
REM experiment - author requirement, see ~/.claude/CLAUDE.md "Long computations
REM and computer sleep". Must be invoked via CALL WITHOUT its own setlocal,
REM so that SAMMON_*_ORIG stay visible after returning to the calling script.
REM Paired script: no_sleep_off.bat (restores the original values, call even after
REM an error). Windows-only; on other OSes this .bat would not run anyway
REM (only invoked from src/run_*.bat).
set "SAMMON_STANDBY_AC_ORIG="
set "SAMMON_HIBERNATE_AC_ORIG="
set "_SAMMON_STANDBY_HEX="
set "_SAMMON_HIBERNATE_HEX="

for /f "tokens=2 delims=:" %%A in ('powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE 2^>nul ^| findstr /C:"Current AC Power Setting Index"') do (
    for /f "tokens=1" %%B in ("%%A") do set "_SAMMON_STANDBY_HEX=%%B"
)
for /f "tokens=2 delims=:" %%A in ('powercfg /query SCHEME_CURRENT SUB_SLEEP HIBERNATEIDLE 2^>nul ^| findstr /C:"Current AC Power Setting Index"') do (
    for /f "tokens=1" %%B in ("%%A") do set "_SAMMON_HIBERNATE_HEX=%%B"
)

if defined _SAMMON_STANDBY_HEX (
    set /a _SAMMON_STANDBY_SEC=%_SAMMON_STANDBY_HEX%
    set /a SAMMON_STANDBY_AC_ORIG=_SAMMON_STANDBY_SEC/60
) else (
    echo [no_sleep_on] WARNING: could not determine the original standby-timeout-ac, will use the default 30 min on restore.
    set "SAMMON_STANDBY_AC_ORIG=30"
)
if defined _SAMMON_HIBERNATE_HEX (
    set /a _SAMMON_HIBERNATE_SEC=%_SAMMON_HIBERNATE_HEX%
    set /a SAMMON_HIBERNATE_AC_ORIG=_SAMMON_HIBERNATE_SEC/60
) else (
    echo [no_sleep_on] WARNING: could not determine the original hibernate-timeout-ac, will use the default 30 min on restore.
    set "SAMMON_HIBERNATE_AC_ORIG=30"
)

echo [no_sleep_on] Original standby-timeout-ac=%SAMMON_STANDBY_AC_ORIG% min, hibernate-timeout-ac=%SAMMON_HIBERNATE_AC_ORIG% min - temporarily disabling (0=never) for the run.
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0

set "_SAMMON_STANDBY_HEX="
set "_SAMMON_HIBERNATE_HEX="
set "_SAMMON_STANDBY_SEC="
set "_SAMMON_HIBERNATE_SEC="
