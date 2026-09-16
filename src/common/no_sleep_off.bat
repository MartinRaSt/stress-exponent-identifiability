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
REM Restores the original AC standby/hibernate timeouts saved in
REM SAMMON_STANDBY_AC_ORIG / SAMMON_HIBERNATE_AC_ORIG by no_sleep_on.bat.
REM Call even after the main command fails (see src/run_*.bat: `call ... & call
REM no_sleep_off.bat`, which runs regardless of the return code).
if not defined SAMMON_STANDBY_AC_ORIG set "SAMMON_STANDBY_AC_ORIG=30"
if not defined SAMMON_HIBERNATE_AC_ORIG set "SAMMON_HIBERNATE_AC_ORIG=30"
echo [no_sleep_off] Restoring standby-timeout-ac=%SAMMON_STANDBY_AC_ORIG% min, hibernate-timeout-ac=%SAMMON_HIBERNATE_AC_ORIG% min.
powercfg /change standby-timeout-ac %SAMMON_STANDBY_AC_ORIG%
powercfg /change hibernate-timeout-ac %SAMMON_HIBERNATE_AC_ORIG%
set "SAMMON_STANDBY_AC_ORIG="
set "SAMMON_HIBERNATE_AC_ORIG="
