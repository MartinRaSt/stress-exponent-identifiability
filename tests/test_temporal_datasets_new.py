# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-13
# License: see the LICENSE file in the repository root
"""K11 (documentation/2026-09-13_e4_rozsireni_datasetu.md): tests for 4 new
temporal datasets for E4 (3 -> 7 total) - invs13, sfhh, ht09 (SocioPatterns,
format `t i j` with no class in the row) and email_eu_core (SNAP, a
non-SocioPatterns control, format `i j t`, classes = departments). These
tests actually DOWNLOAD data (small files, units to hundreds of kB) -
consistent with the existing pattern in `tests/test_sammon_temporal.py` and
`tests/test_datasets.py` (K8 large graphs).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.datasets.registry import load_dataset
from src.experiments.exp4_temporal import _load_snapshots

# (name, expected number of nodes in the static graph, expected number of
# node classes, or None when no public class metadata exists)
_NEW_TEMPORAL_SPECS = [
    ("invs13_temporal", 92, 5),
    ("sfhh_temporal", 403, None),
    ("ht09_temporal", 113, None),
    ("email_eu_core_temporal", 986, 42),
]

# (name, time_bin_sec, min_snapshot_nodes) matching exp4_temporal.dataset_params
# (config_experiments.yaml, block 'full') - see the comment there with the derivation.
_DATASET_PARAMS_FULL = {
    "invs13_temporal": (10800, 20),
    "sfhh_temporal": (3600, 20),
    "ht09_temporal": (3600, 20),
    "email_eu_core_temporal": (1814400, 20),
}


@pytest.mark.parametrize("name,expected_n,expected_n_classes", _NEW_TEMPORAL_SPECS)
def test_new_temporal_dataset_loads_with_expected_shape(name: str, expected_n: int, expected_n_classes) -> None:
    """Every new temporal dataset must load, have kind='temporal', the
    expected number of nodes in the static (aggregated) graph, and the
    expected number of node classes (None if the dataset has no publicly
    available classes - fail-loud, no fabricated substitute class)."""
    ds = load_dataset(name)
    assert ds.kind == "temporal"
    assert ds.graph.number_of_nodes() == expected_n
    assert ds.n_classes == expected_n_classes
    assert ds.meta["n_snapshots"] > 0


@pytest.mark.parametrize("name", [spec[0] for spec in _NEW_TEMPORAL_SPECS])
def test_new_temporal_dataset_snapshot_count_in_target_range(name: str) -> None:
    """After filtering out empty/small windows (min_snapshot_nodes=20, the
    same parameters as exp4_temporal.dataset_params full), 10-40 usable
    snapshots must remain (the K11 spec's goal - enough snapshots for the
    permutation test, not too many for a reasonable wall-clock of the full
    run)."""
    time_bin_sec, min_snapshot_nodes = _DATASET_PARAMS_FULL[name]
    list_of_D, node_ids = _load_snapshots(name, time_bin_sec, min_snapshot_nodes)
    assert 10 <= len(list_of_D) <= 40, f"{name}: {len(list_of_D)} usable snapshots outside the target range 10-40."
    assert len(list_of_D) == len(node_ids)


@pytest.mark.parametrize("name", [spec[0] for spec in _NEW_TEMPORAL_SPECS])
def test_new_temporal_dataset_distance_matrices_are_symmetric_and_finite(name: str) -> None:
    """Every per-snapshot distance matrix (shortest-path on the LCC) must be
    square, symmetric, finite, and have a zero diagonal (D[i,i]=0)."""
    time_bin_sec, min_snapshot_nodes = _DATASET_PARAMS_FULL[name]
    list_of_D, node_ids = _load_snapshots(name, time_bin_sec, min_snapshot_nodes)
    for D, ids in zip(list_of_D, node_ids):
        n = len(ids)
        assert D.shape == (n, n)
        assert np.all(np.isfinite(D))
        np.testing.assert_allclose(D, D.T, atol=1e-10)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-10)
        assert n >= min_snapshot_nodes


def test_email_eu_core_temporal_column_order_matches_static_dataset_node_ids() -> None:
    """Regression test against a column-order bug: email-Eu-core-temporal.txt.gz
    has the format `i j t` (NOT `t i j` like SocioPatterns) - the loader must
    correctly swap the columns, otherwise 't' would contain node ids (small
    integer values 0-999) instead of increasing time (0 to ~6.9e7 seconds,
    803 days)."""
    ds = load_dataset("email_eu_core_temporal")
    edges_df = ds.meta["edges_df"]
    assert edges_df["t"].max() > 1_000_000, "column 't' looks like a node id, not time - check the column order in the loader."
    assert edges_df["i"].max() < 1100
    assert edges_df["j"].max() < 1100


def test_invs13_temporal_metadata_covers_all_nodes() -> None:
    """InVS13 has classes (departments) in a separate metadata_url file - the
    loader must fail-loud raise an error if any node from the edge list were
    missing in the metadata (here just a check that this did not happen in
    the real data and y covers all nodes without NaN/missing values)."""
    ds = load_dataset("invs13_temporal")
    assert ds.y is not None
    assert len(ds.y) == ds.graph.number_of_nodes()
    assert np.all(ds.y >= 0)


@pytest.mark.parametrize("name", ["sfhh_temporal", "ht09_temporal"])
def test_datasets_without_public_metadata_have_no_fabricated_classes(name: str) -> None:
    """SFHH and HT09 have no publicly available node-class metadata - y must
    remain None (fail-loud rule: a missing input is an error/gap, not an
    opportunity to fabricate a substitute class)."""
    ds = load_dataset(name)
    assert ds.y is None
