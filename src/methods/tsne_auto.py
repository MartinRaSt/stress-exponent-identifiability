# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-11
# License: see the LICENSE file in the repository root
"""
Adapter: t-SNE with automatic perplexity selection (a "fair" baseline against
sammon_alpha_auto, see documentation/2026-09-11_kontrola_vysledku_plnych_behu.md
section 3.5 and documentation/2026-09-11_standardizace_a_ladene_baseline.md).

Grid-search over perplexity (methods.tsne_auto.perplexity_grid in config.yaml)
on the same subsample and with the same independent metric as
`sammon.alpha_selection` (src/methods/hyperparam_selection.py), then a
one-off fit on the full data with the best value.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from src.common.config import load_config
from src.methods.common import method_config
from src.methods.hyperparam_selection import select_hyperparam
from src.methods.registry import Method, register_method
from src.methods.tsne import fit_tsne


class TsneAutoMethod(Method):
    name = "tsne_auto"
    accepts = "both"

    def __init__(self) -> None:
        self.last_selected_hyperparam: str = ""

    def fit_transform(self, data: np.ndarray | nx.Graph, kind: str, seed: int, n_components: int = 2) -> np.ndarray:
        cfg = method_config("tsne_auto")
        sel_cfg = load_config()["sammon"]["alpha_selection"]

        def _fit_fn(data_sub: np.ndarray, kind_: str, perplexity: int, fit_seed: int, n_comp: int) -> np.ndarray:
            return fit_tsne(
                data_sub, kind_, fit_seed, n_comp, perplexity=perplexity,
                learning_rate=cfg["learning_rate"], max_iter=cfg["max_iter"], init=cfg["init"],
            )

        best_perplexity, _table = select_hyperparam(
            data, kind, grid=cfg["perplexity_grid"], fit_fn=_fit_fn,
            n_val=sel_cfg["n_val"], selection_seed=sel_cfg["seed"],
            metric=sel_cfg["metric"], n_components=n_components,
        )
        self.last_selected_hyperparam = f"perplexity={best_perplexity}"

        return fit_tsne(
            data, kind, seed, n_components, perplexity=int(best_perplexity),
            learning_rate=cfg["learning_rate"], max_iter=cfg["max_iter"], init=cfg["init"],
        )


register_method("tsne_auto")(TsneAutoMethod)
