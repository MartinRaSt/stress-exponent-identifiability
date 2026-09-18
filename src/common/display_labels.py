# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-18
# License: see the LICENSE file in the repository root
"""
Human-readable labels for raw config/CSV identifiers (dataset/method/metric/
solver/regime/... names) - the ONE shared conversion point for both figures
(`src/figures/fig_common.py`, which re-exports `display_label` from here) and
LaTeX tables (`src/experiments/report_tables.py`). Moved out of
`fig_common.py` on 2026-09-18 (author feedback: table bodies still printed raw
identifiers like "tsne_auto"/"sammon_alpha_smacof" - the table generator must
NOT depend on matplotlib, so this module has no matplotlib import).

Author feedback 2026-09-17 (original motivation, still the reason this exists):
11 figure scripts printed raw identifiers straight into titles, axis labels,
ticks and legends - "vypada to blbe". `display_label` is the ONE shared
conversion point (fig_faithful_map used to keep a private method_labels/
dataset_labels config block just for itself - now merged into
`display_labels.<kind>` in config_experiments.yaml, used everywhere).
"""
from __future__ import annotations

import functools
import re

_KIND_WORD_SPLIT_RE = re.compile(r"_+")
_K_NUMBER_RE = re.compile(r"^[kK](\d+)$")


@functools.lru_cache(maxsize=1)
def _display_labels_config() -> dict:
    """The `display_labels` section of config_experiments.yaml: one dict of
    explicit name->label per `kind` (dataset/method/metric/scenario/solver/
    device/distance_metric/...) plus `word_overrides` (per-word fallback
    fragments, e.g. 'tsne' -> 't-SNE'). Missing section = empty (the
    fallback in `display_label` still guarantees no raw underscore is ever
    shown, see `_fallback_label`)."""
    from src.experiments.config_experiments import load_experiments_config

    return load_experiments_config().get("display_labels", {})


def _fallback_label(name: str, word_overrides: dict) -> str:
    """Fallback for a name with no explicit `display_labels.<kind>` entry:
    split on '_', map an individual word through `word_overrides` (case-
    insensitive) or the generic 'k7' -> 'k=7' neighbourhood-size pattern
    where it applies, otherwise just capitalize the word - NEVER leaves a
    raw underscore in the output (author feedback 2026-09-17)."""
    words = [w for w in _KIND_WORD_SPLIT_RE.split(str(name)) if w]
    if not words:
        return str(name)
    out_words = []
    for w in words:
        override = word_overrides.get(w.lower())
        if override is not None:
            out_words.append(override)
            continue
        k_match = _K_NUMBER_RE.match(w)
        if k_match is not None:
            out_words.append(f"k={k_match.group(1)}")
            continue
        out_words.append(w[:1].upper() + w[1:].lower() if w.isalpha() else w)
    return " ".join(out_words)


def kind_labels(kind: str) -> dict:
    """Raw `name -> label` mapping of one `display_labels.<kind>` config
    section (exact-match entries only, NO `_fallback_label` fill-in). Added
    2026-09-18 alongside `display_labels.column` (short LaTeX-table-header
    labels, see `src/experiments/report_tables.py::_column_label`), which
    needs to check whether a column NAME has an explicit curated entry
    before composing a `_median`/`_avg_rank`/... affix around it - a plain
    `display_label(name, kind)` call cannot answer that (it always returns
    SOME string, via the fallback). Returns `{}` for an unknown/empty
    section (missing config section = no explicit entries, not an error -
    `display_label` still falls back for any name)."""
    return dict(_display_labels_config().get(kind, {}))


def display_label(name: str, kind: str) -> str:
    """Human-readable label for a raw identifier (dataset/method/metric/...
    name) used in figure titles, axis labels, ticks, legends AND LaTeX table
    cells (see the module docstring). Looks up `display_labels.<kind>.<name>`
    in config_experiments.yaml first (exact, author-curated text); an
    unmapped name falls back to `_fallback_label` (word-by-word
    capitalization + `display_labels.word_overrides`), so a name is NEVER
    shown as a raw identifier with underscores, even before someone adds
    an explicit entry to the config."""
    cfg = _display_labels_config()
    mapping = cfg.get(kind, {})
    if name in mapping:
        return str(mapping[name])
    return _fallback_label(name, cfg.get("word_overrides", {}))
