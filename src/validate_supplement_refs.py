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
Validate the main text's references into the Supplement.

The main text points at Supplement sections through `\\supref{N}` /
`\\suprefs{N}{M}` rather than a real `\\ref`. That is deliberate: at submission
the publisher compiles the main file ALONE, without `supplement.aux`, so an
`xr`-based cross reference would typeset as "??". The price of that choice is
that LaTeX cannot check the numbers - this script does it instead.

Checks performed:
  1. every referenced section number exists in the Supplement,
  2. no hand-written "Supplement, sekce SN" survives outside the macro,
  3. prints each reference with the TITLE of the section it points at, so a
     human can confirm it points at the right content (a number can be valid
     and still be wrong).

Exit code 1 on any failure, so it can gate a pre-submission check.

Run: venv\\python.exe -m src.validate_supplement_refs
or:  src\\run_validate_supplement_refs.bat
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN_SECTIONS = ROOT / "clanek" / "sections"
SUPPLEMENT = ROOT / "clanek" / "supplement" / "supplement.tex"
SUPPLEMENT_SECTIONS = ROOT / "clanek" / "supplement" / "sections"

_REF = re.compile(r"\\supref\{(\d+)\}")
_REFS = re.compile(r"\\suprefs\{(\d+)\}\{(\d+)\}")
_MANUAL = re.compile(r"Supplement, sekce~?S\d")
_INPUT = re.compile(r"\\input\{sections/([A-Za-z0-9_]+)\}")
_TITLE = re.compile(r"\\section\*?\{([^}]*)\}")


def supplement_titles() -> dict[int, str]:
    """{section number -> title}, numbered by order of \\input in supplement.tex.

    The Supplement renumbers its sections as S1, S2, ... via
    \\renewcommand{\\thesection}, so the number is simply the input order.
    """
    body = SUPPLEMENT.read_text(encoding="utf-8", errors="replace")
    titles: dict[int, str] = {}
    for index, stem in enumerate(_INPUT.findall(body), start=1):
        path = SUPPLEMENT_SECTIONS / f"{stem}.tex"
        if not path.exists():
            raise FileNotFoundError(f"supplement.tex inputs {stem}, but the file does not exist: {path}")
        match = _TITLE.search(path.read_text(encoding="utf-8", errors="replace"))
        titles[index] = match.group(1) if match else f"({stem}, untitled)"
    return titles


def main() -> int:
    titles = supplement_titles()
    print(f"Supplement has {len(titles)} sections:")
    for number, title in titles.items():
        print(f"  S{number}  {title}")
    print()

    problems: list[str] = []
    rows: list[tuple[str, int, str, str]] = []

    for path in sorted(MAIN_SECTIONS.glob("*.tex")):
        text = path.read_text(encoding="utf-8", errors="replace")

        for match in _MANUAL.finditer(text):
            line = text[: match.start()].count("\n") + 1
            problems.append(f"{path.name}:{line}: hand-written reference '{match.group(0)}' - use \\supref instead")

        targets: list[tuple[int, int]] = [
            (int(m.group(1)), m.start()) for m in _REF.finditer(text)
        ]
        for m in _REFS.finditer(text):
            targets.append((int(m.group(1)), m.start()))
            targets.append((int(m.group(2)), m.start()))

        for number, pos in targets:
            line = text[:pos].count("\n") + 1
            if number not in titles:
                problems.append(f"{path.name}:{line}: reference to S{number}, which does NOT EXIST in the supplement")
                continue
            rows.append((path.name, line, f"S{number}", titles[number]))

    rows.sort(key=lambda r: (r[0], r[1]))
    print(f"{'file':<24}{'line':>7}  {'target':<6} target section title")
    print("-" * 100)
    for name, line, target, title in rows:
        print(f"{name:<24}{line:>7}  {target:<6} {title}")
    print(f"\ntotal references: {len(rows)}")

    if problems:
        print("\nPROBLEMS:")
        for problem in problems:
            print("  " + problem)
        print(f"\nFAILED: {len(problems)} problem(s).")
        return 1

    print("\nOK: every reference points at an existing section.")
    print("NOTE: a valid number does not mean it is the RIGHT section - eyeball the table above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
