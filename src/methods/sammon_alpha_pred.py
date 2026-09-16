# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-12
# License: see the LICENSE file in the repository root
"""
K7 (documentation/2026-09-12_plan_smeru_clanku.md) - adapter for the
`sammon_alpha_pred` method: alpha is NOT tuned via a grid (unlike
`sammon_alpha_auto`), but is predicted from the distance concentration
(nn_ratio_k1) via a rule derived by `src/experiments/fit_alpha_rule.py`
(leave-one-dataset-out, see `results/data/alpha_pred_rule.json`) - a SINGLE
SMACOF run, no grid search, hence as cheap as a fixed alpha=0/1/2.

Does not depend on the `_make_sammon_method` factory function in
`src/methods/sammon_alpha.py` (which assumes a fixed `alpha` in
`methods.<name>` in common/config.yaml) - here alpha is a dynamic function
of the data, hence a separate adapter.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from src.methods.common import method_config, to_distance_matrix
from src.methods.registry import Method, register_method
from src.sammon.alpha_predict import load_alpha_pred_rule, nn_ratio_k1_from_D, predict_alpha


class SammonAlphaPredMethod(Method):
    name = "sammon_alpha_pred"
    accepts = "both"

    def __init__(self) -> None:
        self.last_selected_hyperparam: str = ""

    def fit_transform(self, data: np.ndarray | nx.Graph, kind: str, seed: int, n_components: int = 2) -> np.ndarray:
        # local import - see src/methods/sammon_alpha.py for an explanation
        # of the circular import src.sammon.estimator <-> src.methods.common
        from src.sammon.estimator import SammonAlpha

        cfg = method_config("sammon_alpha_pred")
        D = to_distance_matrix(data, kind)
        nn_ratio_k1 = nn_ratio_k1_from_D(D)
        rule = load_alpha_pred_rule()
        alpha_pred = predict_alpha(nn_ratio_k1, rule)

        est = SammonAlpha(
            alpha=alpha_pred, solver=cfg["solver"], device=cfg["device"],
            n_components=n_components, seed=seed, verbose=False,
        )
        Y = est.fit_transform(data, kind=kind)
        self.last_selected_hyperparam = (
            f"alpha={alpha_pred} (nn_ratio_k1={nn_ratio_k1:.4g}, rule={rule['variant']})"
        )
        return Y


register_method("sammon_alpha_pred")(SammonAlphaPredMethod)
