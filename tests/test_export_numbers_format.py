# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-17
# License: see the LICENSE file in the repository root
"""
Number formatting of the generated LaTeX macros (`src/experiments/export_numbers.py`).

Regression guard for the 2026-09-17 bug: a pooled Spearman p-value of 2.8e-190
was emitted in fixed decimal notation, i.e. with 190 decimal places, and
siunitx rejected the resulting macro with "Invalid number". Extreme magnitudes
must use scientific notation, ordinary ones must keep the fixed notation that
every number already typeset in the article uses.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.export_numbers import (  # noqa: E402
    SCIENTIFIC_MAX_MAGNITUDE,
    SCIENTIFIC_MIN_MAGNITUDE,
    _macro_line,
    fmt_sig,
)

# Same pattern as `_wrap_decimal`: what siunitx is handed must match it.
_DECIMAL_RE = re.compile(r"^[+-]?\d+\.\d+([eE][+-]?\d+)?$")


@pytest.mark.parametrize(
    "value,sig,expected",
    [
        (0.5, 2, "0.50"),
        (0.0001234, 2, "0.00012"),
        (1.0e-5, 2, "0.000010"),
        (123456.0, 3, "123456"),
        (0.0, 3, "0"),
    ],
)
def test_ordinary_magnitudes_stay_fixed(value: float, sig: int, expected: str) -> None:
    """Numbers inside the ordinary range keep their previous fixed formatting."""
    assert fmt_sig(value, sig) == expected


@pytest.mark.parametrize("value", [2.8e-190, 1.4e-9, 1e-7, 2.5e10, 7.0e12])
def test_extreme_magnitudes_use_scientific_notation(value: float) -> None:
    """Extreme magnitudes must not expand into hundreds of decimal places."""
    formatted = fmt_sig(value, 2)
    assert "e" in formatted
    assert len(formatted) < 12
    assert float(formatted) == pytest.approx(value, rel=0.05)


def test_scientific_output_is_wrapped_for_siunitx() -> None:
    """A scientific value must still be wrapped in \\num{...} by _macro_line."""
    line = _macro_line("testMacro", fmt_sig(2.8e-190, 2), "unit test")
    assert "\\num{2.8e-190}" in line


def test_threshold_boundaries_are_consistent() -> None:
    """Values just inside the thresholds stay fixed, just outside go scientific."""
    inside_low = 10.0 ** SCIENTIFIC_MIN_MAGNITUDE
    outside_low = 10.0 ** (SCIENTIFIC_MIN_MAGNITUDE - 1)
    inside_high = 10.0 ** SCIENTIFIC_MAX_MAGNITUDE
    outside_high = 10.0 ** (SCIENTIFIC_MAX_MAGNITUDE + 1)
    assert "e" not in fmt_sig(inside_low, 2)
    assert "e" in fmt_sig(outside_low, 2)
    assert "e" not in fmt_sig(inside_high, 2)
    assert "e" in fmt_sig(outside_high, 2)


def test_every_formatted_value_is_parseable_by_the_wrapper() -> None:
    """Whatever fmt_sig returns must be either an integer or a _DECIMAL_RE match."""
    for value in [0.5, 0.0001234, 2.8e-190, 2.5e10, 123456.0, -0.0031]:
        formatted = fmt_sig(value, 2)
        assert formatted.lstrip("+-").isdigit() or _DECIMAL_RE.match(formatted), formatted


def test_non_finite_is_fail_loud() -> None:
    """NaN/Inf must raise, never silently become a printed value."""
    for bad in [float("nan"), float("inf"), float("-inf")]:
        with pytest.raises(ValueError):
            fmt_sig(bad, 2)
