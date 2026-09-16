# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-11
# License: see the LICENSE file in the repository root
"""Test for src/experiments/exp_common.py::keep_system_awake (disabling PC
sleep for the duration of long experiments, the author's requirement - see
~/.claude/CLAUDE.md "Long computations and preventing sleep")."""
from __future__ import annotations

import sys

import pytest

from src.experiments.exp_common import keep_system_awake

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


class _FakeKernel32:
    """Replaces ctypes.windll.kernel32 - records calls to SetThreadExecutionState
    without actually touching the power plan during the test."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def SetThreadExecutionState(self, flags: int) -> int:
        self.calls.append(flags)
        return 1  # a nonzero return code = success (the real API)


@pytest.mark.skipif(sys.platform != "win32", reason="SetThreadExecutionState is a Windows-only API.")
def test_keep_system_awake_sets_and_restores_execution_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """At the start of the block ES_CONTINUOUS|ES_SYSTEM_REQUIRED, at the end (success) ES_CONTINUOUS."""
    import ctypes

    fake = _FakeKernel32()
    monkeypatch.setattr(ctypes, "windll", type("_FakeWindll", (), {"kernel32": fake})())

    with keep_system_awake():
        assert fake.calls == [_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED]

    assert fake.calls == [_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED, _ES_CONTINUOUS]


@pytest.mark.skipif(sys.platform != "win32", reason="SetThreadExecutionState is a Windows-only API.")
def test_keep_system_awake_restores_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Even on an exception inside the block, ES_CONTINUOUS must be restored
    (finally) - otherwise a long crashing run could permanently prevent sleep."""
    import ctypes

    fake = _FakeKernel32()
    monkeypatch.setattr(ctypes, "windll", type("_FakeWindll", (), {"kernel32": fake})())

    with pytest.raises(ValueError):
        with keep_system_awake():
            raise ValueError("boom")

    assert fake.calls[-1] == _ES_CONTINUOUS


def test_keep_system_awake_noop_on_non_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """On OSes other than Windows, keep_system_awake() must not access
    ctypes.windll (it does not even exist on those OSes) - it must be a safe no-op."""
    monkeypatch.setattr(sys, "platform", "linux")
    with keep_system_awake():
        pass
