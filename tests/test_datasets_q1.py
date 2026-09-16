# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-14
# License: see the LICENSE file in the repository root
"""Q1 step 2 (reserse/2026-09-14_specifikace_rozsireni_q1.md, A.9 item 13) -
tests for the new datasets (`src/datasets/uci_extra.py`,
`src/datasets/synthetic.py` extension): shape, determinism, dedup, no NaN,
d>=3, theta>0. Network loaders (OpenML/zip) are SKIPPED (skipif) if they
fail due to network unavailability - no hard-fail of the test on a clean
checkout without cache/connectivity; if they are already cached (typically
after the first run), the test runs normally."""
from __future__ import annotations

import numpy as np
import pytest

from src.datasets.registry import load_dataset
from src.datasets.subsample import subsample_dataset

SYNTHETIC_KEYS = [
    "swiss_roll_hole", "gaussian_clusters_separated", "mobius_strip",
    "klein_bottle_4d", "twin_peaks", "trefoil_knot",
    # Addendum 2026-09-14 (N>=26): 2 more synthetic datasets, different
    # curvature/topology (see the src/datasets/synthetic.py docstring).
    "hyperbolic_paraboloid", "anisotropic_ellipsoid",
]
NETWORK_KEYS = [
    "shuttle", "abalone", "page_blocks", "airfoil", "banknote",
    "mfeat_morphological", "wall_robot", "hill_valley", "phoneme",
    "dry_bean", "ccpp",
]
ALL_NEW_KEYS = SYNTHETIC_KEYS + NETWORK_KEYS

_EXPECTED_MIN_D = {
    "shuttle": 9, "abalone": 7, "page_blocks": 10, "airfoil": 5, "banknote": 4,
    "mfeat_morphological": 6, "wall_robot": 24, "hill_valley": 100, "phoneme": 5,
    "dry_bean": 16, "ccpp": 4,
    "swiss_roll_hole": 3, "gaussian_clusters_separated": 10, "mobius_strip": 3,
    "klein_bottle_4d": 4, "twin_peaks": 3, "trefoil_knot": 3,
    "hyperbolic_paraboloid": 5, "anisotropic_ellipsoid": 6,
}


def _load_or_skip(key: str):
    """Network/zip loaders: if they fail (connectivity unavailable, HTTP
    error), the test is SKIPPED (skipif behavior), not failed - see A.9 item
    13. Errors of another kind (e.g. a KeyError from misconfiguration) are
    NOT swallowed (they propagate as a test failure, since they are actual
    bugs in the code, not network unavailability)."""
    try:
        return load_dataset(key)
    except Exception as exc:  # noqa: BLE001 - network errors are heterogeneous (RuntimeError from uci_extra.py)
        if key in NETWORK_KEYS and isinstance(exc, RuntimeError):
            pytest.skip(f"'{key}': unavailable (network/cache) - skipping: {exc}")
        raise


@pytest.mark.parametrize("key", ALL_NEW_KEYS)
def test_shape_and_no_nan(key: str) -> None:
    ds = _load_or_skip(key)
    assert ds.kind == "vector"
    assert ds.X.ndim == 2
    assert ds.X.shape[0] > 0
    assert ds.X.shape[1] == _EXPECTED_MIN_D[key]
    assert np.all(np.isfinite(ds.X)), f"'{key}': X contains NaN/Inf."
    if ds.y is not None:
        assert np.all(np.isfinite(ds.y.astype(np.float64))), f"'{key}': y contains NaN/Inf."


@pytest.mark.parametrize("key", ALL_NEW_KEYS)
def test_d_at_least_3(key: str) -> None:
    """A.4 eligibility E1: ambient dimension d >= 3 for ALL new datasets."""
    ds = _load_or_skip(key)
    assert ds.X.shape[1] >= 3


@pytest.mark.parametrize("key", ALL_NEW_KEYS)
def test_dedup_metadata_present(key: str) -> None:
    """A.4 item 3: every new loader must write meta['dedup']=True and
    n_duplicates_removed (>=0, an integer) - even if the number removed is 0."""
    ds = _load_or_skip(key)
    assert ds.meta.get("dedup") is True
    n_removed = ds.meta.get("n_duplicates_removed")
    assert n_removed is not None and int(n_removed) >= 0
    n_before = ds.meta.get("n_before_dedup")
    assert n_before is not None and int(n_before) == ds.X.shape[0] + int(n_removed)


@pytest.mark.parametrize("key", SYNTHETIC_KEYS)
def test_theta_positive_synthetic(key: str) -> None:
    """A.4 eligibility E2 (theta>0): for synthetic datasets WITHOUT the
    network - the median distance to the nearest neighbor must be positive
    (no exact duplicate points after dedup)."""
    from scipy.spatial.distance import pdist, squareform

    from src.experiments.dataset_properties import nn_distances_from_D

    ds = _load_or_skip(key)
    ds_sub = subsample_dataset(ds, n_max=300, random_state=42)
    D = squareform(pdist(np.asarray(ds_sub.X, dtype=np.float64), metric="euclidean"))
    theta = float(np.median(nn_distances_from_D(D, 1)))
    assert theta > 0.0


@pytest.mark.parametrize("key", NETWORK_KEYS)
def test_theta_positive_network(key: str) -> None:
    """Same as `test_theta_positive_synthetic`, just for network datasets (a
    separate test so skipif does not depend on the synthetic ones)."""
    from scipy.spatial.distance import pdist, squareform

    from src.experiments.dataset_properties import nn_distances_from_D

    ds = _load_or_skip(key)
    ds_sub = subsample_dataset(ds, n_max=300, random_state=42)
    D = squareform(pdist(np.asarray(ds_sub.X, dtype=np.float64), metric="euclidean"))
    theta = float(np.median(nn_distances_from_D(D, 1)))
    assert theta > 0.0


@pytest.mark.parametrize("key", SYNTHETIC_KEYS)
def test_synthetic_determinism(key: str) -> None:
    """A.4 eligibility E5: two independent calls to `load_dataset` return
    bit-identical X (and y, if present) - a deterministic generator."""
    ds1 = load_dataset(key)
    ds2 = load_dataset(key)
    np.testing.assert_array_equal(ds1.X, ds2.X)
    if ds1.y is not None:
        np.testing.assert_array_equal(ds1.y, ds2.y)


@pytest.mark.parametrize("key", NETWORK_KEYS)
def test_network_determinism(key: str) -> None:
    ds1 = _load_or_skip(key)
    ds2 = _load_or_skip(key)
    np.testing.assert_array_equal(ds1.X, ds2.X)
    if ds1.y is not None:
        np.testing.assert_array_equal(ds1.y, ds2.y)


def test_frey_faces_not_registered() -> None:
    """R1 (frey_faces) is NOT implemented - the source repeatedly returned
    HTTP 403 on 2026-09-14 (verified before implementation, see
    documentation/2026-09-14_q1_krok2_datasety.md). This test documents the
    intent (not an oversight) - if frey_faces is registered in the future,
    this test will start failing and must be consciously updated/removed."""
    from src.datasets.registry import list_registered_datasets

    assert "frey_faces" not in list_registered_datasets()


def test_q1_holdout_config_lists_all_registered_new_keys() -> None:
    """config_experiments.yaml: q1_holdout_datasets must contain EXACTLY all
    the newly registered keys (none lost, none extra)."""
    from src.experiments.config_experiments import load_experiments_config

    cfg_keys = set(load_experiments_config()["q1_holdout_datasets"])
    assert cfg_keys == set(ALL_NEW_KEYS)


def test_all_new_datasets_registered() -> None:
    from src.datasets.registry import list_registered_datasets

    registered = set(list_registered_datasets())
    missing = set(ALL_NEW_KEYS) - registered
    assert not missing, f"New datasets are missing from the registry: {missing}"
