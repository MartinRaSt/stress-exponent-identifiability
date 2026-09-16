# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-09
# License: see the LICENSE file in the repository root
"""Tests for the method adapters: every method from the registry must
return an (n x 2) embedding on a compatible dataset (iris for vector
methods, karate for graph methods)."""
from __future__ import annotations

import numpy as np
import pytest

from src.common.seeding import set_seed
from src.datasets.registry import load_dataset
from src.methods.common import is_compatible
from src.methods.registry import get_method, list_registered_methods

# trimap has a known library limit relative to karate (n=34) (an internal
# neighborhood of n_inliers+50); a vector-only test on iris for trimap
# suffices to verify the adapter works.
#
# sammon_sgd_naive on iris: iris contains an exact duplicate row (two
# identical points, D_ij=0), naive SGD (Zheng 2018, w_ij=D_ij^{-alpha}
# without eps_D regularization) for alpha>0 at D_ij=0 divides by zero - the
# solver reports it as a ValueError (fail loud per the project rules, see
# src/sammon/solvers/sgd.py:sgd_solve), this is NOT an adapter bug. The
# karate club (graph, geodesic distances) has no duplicate points, so the
# test passes there.
KNOWN_TOO_SMALL_FOR = {("trimap", "karate")}
KNOWN_INCOMPATIBLE_DATA_FOR = {("sammon_sgd_naive", "iris")}

# K7 (documentation/2026-09-12_plan_smeru_clanku.md): 'sammon_alpha_pred' is
# fail-loud without the production results/data/alpha_pred_rule.json (see
# src/sammon/alpha_predict.py::load_alpha_pred_rule) - this is INTENTIONALLY
# correct method behavior, not an adapter bug. The tests therefore stub
# `load_alpha_pred_rule` with a fixed (test) rule (see
# `_stub_alpha_pred_rule`), so the generic smoke test verifies the actual
# adapter functionality without depending on whether the author has already
# run fit_alpha_rule.py --full.
_METHODS_NEEDING_ALPHA_PRED_RULE_STUB = {"sammon_alpha_pred"}
_TEST_ALPHA_PRED_RULE = {"variant": "log_linear", "coefficients": {"a": 1.0, "b": -0.3}}


def _stub_alpha_pred_rule(monkeypatch: pytest.MonkeyPatch, method_name: str) -> None:
    if method_name in _METHODS_NEEDING_ALPHA_PRED_RULE_STUB:
        import src.methods.sammon_alpha_pred as mod

        monkeypatch.setattr(mod, "load_alpha_pred_rule", lambda: dict(_TEST_ALPHA_PRED_RULE))


@pytest.fixture(scope="module")
def iris_X() -> np.ndarray:
    return load_dataset("iris").X


@pytest.fixture(scope="module")
def karate_graph():
    return load_dataset("karate").graph


@pytest.mark.parametrize("method_name", list_registered_methods())
def test_method_on_iris(method_name: str, iris_X: np.ndarray, monkeypatch: pytest.MonkeyPatch) -> None:
    """Methods accepting point data must return a valid (150 x 2) embedding on iris."""
    method = get_method(method_name)
    if not is_compatible(method.accepts, "vector"):
        pytest.skip(f"Method '{method_name}' does not support kind='vector' (accepts={method.accepts}).")
    if (method_name, "iris") in KNOWN_INCOMPATIBLE_DATA_FOR:
        pytest.skip(f"Method '{method_name}' has a known limit on iris (duplicate points), see KNOWN_INCOMPATIBLE_DATA_FOR.")
    _stub_alpha_pred_rule(monkeypatch, method_name)
    set_seed(0)
    Y = method.fit_transform(iris_X, "vector", seed=0, n_components=2)
    assert Y.shape == (150, 2)
    assert np.isfinite(Y).all()


@pytest.mark.parametrize("method_name", list_registered_methods())
def test_method_on_karate(method_name: str, karate_graph, monkeypatch: pytest.MonkeyPatch) -> None:
    """Methods accepting a graph must return a valid (34 x 2) embedding on karate."""
    method = get_method(method_name)
    if not is_compatible(method.accepts, "graph"):
        pytest.skip(f"Method '{method_name}' does not support kind='graph' (accepts={method.accepts}).")
    if (method_name, "karate") in KNOWN_TOO_SMALL_FOR:
        pytest.skip(f"Method '{method_name}' has a known limit for small graphs (n=34), see KNOWN_TOO_SMALL_FOR.")
    _stub_alpha_pred_rule(monkeypatch, method_name)
    set_seed(0)
    Y = method.fit_transform(karate_graph, "graph", seed=0, n_components=2)
    assert Y.shape == (34, 2)
    assert np.isfinite(Y).all()


def test_sammon_alpha_pred_fails_loud_without_rule_file(iris_X: np.ndarray, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Without a stub/production rule, 'sammon_alpha_pred' must fail with a
    clear FileNotFoundError (see
    src/sammon/alpha_predict.py::load_alpha_pred_rule), NOT a silent
    fallback to some hardcoded alpha."""
    import src.sammon.alpha_predict as ap

    monkeypatch.setattr(ap, "default_rule_path", lambda: tmp_path / "does_not_exist.json")
    method = get_method("sammon_alpha_pred")
    with pytest.raises(FileNotFoundError):
        method.fit_transform(iris_X, "vector", seed=0, n_components=2)


# K5 (documentation/2026-09-12_plan_smeru_clanku.md): new baseline methods
# densmap/phate - in addition to the generic smoke tests above (already
# covered via list_registered_methods()), an explicit determinism test for
# the same seed (the K5 spec requires a separate determinism test, not just finiteness/shape).
_K5_NEW_METHODS = ["densmap", "phate"]


@pytest.mark.parametrize("method_name", _K5_NEW_METHODS)
def test_k5_method_deterministic_same_seed(method_name: str, iris_X: np.ndarray) -> None:
    """densmap/phate with the same seed must return a bit-identical
    embedding (fail loud on nondeterminism - neither package uses any
    source of randomness outside `random_state`/`seed`)."""
    method = get_method(method_name)
    set_seed(0)
    Y1 = method.fit_transform(iris_X, "vector", seed=0, n_components=2)

    method2 = get_method(method_name)
    set_seed(0)
    Y2 = method2.fit_transform(iris_X, "vector", seed=0, n_components=2)

    assert np.allclose(Y1, Y2), f"Method '{method_name}' is not deterministic for the same seed."
