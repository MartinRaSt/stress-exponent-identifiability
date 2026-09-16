# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-09
# License: see the LICENSE file in the repository root
"""
Loaders for temporal (dynamic) network datasets used in experiments with
the temporal Sammon projection. Every loader returns a Dataset with
kind="temporal", where meta contains the time series of network snapshots
(a list of networkx graphs) and the original edge list with timestamps.
"""
from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from src.common.config import ensure_dir, get_path, load_config
from src.datasets.download import download_file
from src.datasets.registry import Dataset, register_dataset


def _build_snapshots(edges_df: pd.DataFrame, time_bin_sec: float, nodes: list) -> list[nx.Graph]:
    """Split the edge list into time windows and build an aggregated graph for each."""
    t_min, t_max = edges_df["t"].min(), edges_df["t"].max()
    bin_edges = np.arange(t_min, t_max + time_bin_sec, time_bin_sec)
    snapshots: list[nx.Graph] = []
    bin_idx = np.digitize(edges_df["t"].to_numpy(), bin_edges) - 1
    for b in range(len(bin_edges) - 1):
        g = nx.Graph()
        g.add_nodes_from(nodes)
        mask = bin_idx == b
        if mask.any():
            sub = edges_df.loc[mask]
            for i, j in zip(sub["i"], sub["j"]):
                if g.has_edge(i, j):
                    g[i][j]["weight"] += 1
                else:
                    g.add_edge(i, j, weight=1)
        snapshots.append(g)
    return snapshots


def _load_sociopatterns_contact_dataset(
    dataset_name: str, spec_key: str, cache_filename: str, citation: str, **overrides: Any,
) -> Dataset:
    """Shared loader for SocioPatterns contact time series with the common
    file format `t i j <label_i> <label_j>` (time in seconds, two contact
    nodes, a categorical label for each - class/department/status). Used
    for primary_school, hospital and high_school (see config.yaml
    datasets.temporal.<spec_key>) - downloading and parsing is identical
    for all three, only the URL, cache filename and citation differ.
    """
    cfg = load_config()
    spec = cfg["datasets"]["temporal"][spec_key]
    time_bin_sec = overrides.get("time_bin_sec", spec["time_bin_sec"])

    cache_dir = ensure_dir(get_path("temporal_cache_dir"))
    gz_path = download_file(
        spec["data_url"], cache_dir / cache_filename,
        name=dataset_name, license_note="SocioPatterns, CC BY-NC-SA",
    )

    with gzip.open(gz_path, "rt") as f:
        edges_df = pd.read_csv(f, sep=r"\s+", header=None, names=["t", "i", "j", "Ci", "Cj"])

    nodes_classes: dict[int, str] = {}
    for i, ci in zip(edges_df["i"], edges_df["Ci"]):
        nodes_classes.setdefault(int(i), ci)
    for j, cj in zip(edges_df["j"], edges_df["Cj"]):
        nodes_classes.setdefault(int(j), cj)
    nodes = sorted(nodes_classes.keys())

    from sklearn.preprocessing import LabelEncoder

    y = LabelEncoder().fit_transform([nodes_classes[n] for n in nodes]).astype(np.int64)

    snapshots = _build_snapshots(edges_df, time_bin_sec=time_bin_sec, nodes=nodes)

    static_graph = nx.Graph()
    static_graph.add_nodes_from(nodes)
    for i, j in zip(edges_df["i"], edges_df["j"]):
        if static_graph.has_edge(i, j):
            static_graph[i][j]["weight"] += 1
        else:
            static_graph.add_edge(i, j, weight=1)

    return Dataset(
        name=dataset_name, kind="temporal", graph=static_graph, y=y,
        meta={
            "source": spec["data_url"],
            "citation": citation,
            "nodes": nodes,
            "edges_df": edges_df,
            "time_bin_sec": time_bin_sec,
            "snapshots": snapshots,
            "n_snapshots": len(snapshots),
        },
    )


@register_dataset("primary_school_temporal")
def load_primary_school_temporal(**overrides: Any) -> Dataset:
    """SocioPatterns primary school temporal contact network (32 school classes, 2 days of measurement).

    Source file format: `t i j Ci Cj` (time in seconds, two contact nodes,
    the class of each). Source: sociopatterns.org (see config.yaml).
    """
    return _load_sociopatterns_contact_dataset(
        "primary_school_temporal", "primary_school", "primaryschool.csv.gz",
        "Stehle et al. (2011), SocioPatterns primary school", **overrides,
    )


@register_dataset("hospital_temporal")
def load_hospital_temporal(**overrides: Any) -> Dataset:
    """SocioPatterns hospital ward dynamic contact network (4 days of measurement, 2010/2013).

    Source file format: `t i j Si Sj` (time in seconds, two contact nodes,
    the status of each - patient/nursing staff/administration).
    Source: sociopatterns.org (see config.yaml datasets.temporal.hospital).
    """
    return _load_sociopatterns_contact_dataset(
        "hospital_temporal", "hospital", "hospital_lyon_contacts.dat.gz",
        "Vanhems et al. (2013), PLoS ONE 8(9):e73970, SocioPatterns hospital ward", **overrides,
    )


@register_dataset("high_school_temporal")
def load_high_school_temporal(**overrides: Any) -> Dataset:
    """SocioPatterns high school 2013 proximity contact network (7 days of measurement).

    Source file format: `t i j Ci Cj` (time in seconds, two contact nodes,
    the class of each). Source: sociopatterns.org (see config.yaml
    datasets.temporal.high_school).
    """
    return _load_sociopatterns_contact_dataset(
        "high_school_temporal", "high_school", "highschool2013.csv.gz",
        "Mastrandrea, Fournet & Barrat (2015), PLoS ONE 10(9):e0136497, SocioPatterns high school 2013", **overrides,
    )


def _load_plain_tij_dataset(
    dataset_name: str, spec_key: str, cache_filename: str, citation: str,
    license_note: str, archive_kind: str, **overrides: Any,
) -> Dataset:
    """Shared loader for time series in the `t i j` format WITHOUT node
    classes directly in the row (unlike `_load_sociopatterns_contact_dataset`
    above) - used for invs13, sfhh and ht09 (see config.yaml datasets.temporal.
    <spec_key>). If the spec contains `metadata_url`, each node's class is
    looked up in a separate file (columns: node, class) - otherwise y
    remains None (SFHH and HT09 have no publicly available node metadata;
    fail-loud, fabricating a class is not permitted). `archive_kind`
    distinguishes the downloaded file's format: 'gz' (gzip) or 'zip' (a
    single .dat file inside the archive, InVS13).
    """
    cfg = load_config()
    spec = cfg["datasets"]["temporal"][spec_key]
    time_bin_sec = overrides.get("time_bin_sec", spec["time_bin_sec"])

    cache_dir = ensure_dir(get_path("temporal_cache_dir"))
    raw_path = download_file(
        spec["data_url"], cache_dir / cache_filename, name=dataset_name, license_note=license_note,
    )

    if archive_kind == "zip":
        import zipfile

        with zipfile.ZipFile(raw_path) as zf:
            inner_names = [n for n in zf.namelist() if not n.endswith("/")]
            if len(inner_names) != 1:
                raise ValueError(
                    f"Dataset '{dataset_name}': expected exactly 1 file in archive {raw_path}, "
                    f"found {inner_names}."
                )
            with zf.open(inner_names[0]) as f:
                edges_df = pd.read_csv(f, sep=r"\s+", header=None, names=["t", "i", "j"])
    elif archive_kind == "gz":
        with gzip.open(raw_path, "rt") as f:
            edges_df = pd.read_csv(f, sep=r"\s+", header=None, names=["t", "i", "j"])
    else:
        raise ValueError(
            f"Unknown archive_kind='{archive_kind}' for dataset '{dataset_name}' (expected 'gz' or 'zip')."
        )

    nodes = sorted(set(edges_df["i"]).union(edges_df["j"]))

    y: np.ndarray | None = None
    metadata_url = spec.get("metadata_url")
    if metadata_url:
        meta_cache_name = f"{Path(cache_filename).stem.split('.')[0]}_metadata.txt"
        meta_path = download_file(
            metadata_url, cache_dir / meta_cache_name, name=f"{dataset_name}_metadata", license_note=license_note,
        )
        labels_df = pd.read_csv(meta_path, sep=r"\s+", header=None, names=["node", "label"])
        label_map = dict(zip(labels_df["node"], labels_df["label"]))
        missing = [n for n in nodes if n not in label_map]
        if missing:
            raise ValueError(
                f"Dataset '{dataset_name}': {len(missing)} nodes from the edge list are missing from the metadata "
                f"({metadata_url}), e.g. {missing[:5]} - data is not fabricated, check the source."
            )
        from sklearn.preprocessing import LabelEncoder

        y = LabelEncoder().fit_transform([label_map[n] for n in nodes]).astype(np.int64)

    snapshots = _build_snapshots(edges_df, time_bin_sec=time_bin_sec, nodes=nodes)

    static_graph = nx.Graph()
    static_graph.add_nodes_from(nodes)
    for i, j in zip(edges_df["i"], edges_df["j"]):
        if static_graph.has_edge(i, j):
            static_graph[i][j]["weight"] += 1
        else:
            static_graph.add_edge(i, j, weight=1)

    return Dataset(
        name=dataset_name, kind="temporal", graph=static_graph, y=y,
        meta={
            "source": spec["data_url"],
            "citation": citation,
            "nodes": nodes,
            "edges_df": edges_df,
            "time_bin_sec": time_bin_sec,
            "snapshots": snapshots,
            "n_snapshots": len(snapshots),
        },
    )


@register_dataset("invs13_temporal")
def load_invs13_temporal(**overrides: Any) -> Dataset:
    """SocioPatterns InVS13 workplace dynamic contact network (11.4 days of measurement, 92 nodes).

    Source file format (inside a .zip archive): `t i j` (time in seconds,
    two contact nodes, WITHOUT classes directly in the row) - each node's
    department is in a separate metadata_url file. Source: sociopatterns.org
    (see config.yaml datasets.temporal.invs13).
    """
    return _load_plain_tij_dataset(
        "invs13_temporal", "invs13", "tij_InVS.dat.zip",
        "Genois & Barrat (2018), EPJ Data Science 7, 11 - SocioPatterns InVS13 workplace",
        license_note="SocioPatterns, CC0 (public domain)", archive_kind="zip", **overrides,
    )


@register_dataset("sfhh_temporal")
def load_sfhh_temporal(**overrides: Any) -> Dataset:
    """SocioPatterns SFHH conference dynamic contact network (1.3 days of measurement, 403 nodes).

    Source file format: `t i j` (time in seconds, two contact nodes) - no
    publicly available node-class metadata, so y=None (no substitute class
    is fabricated). Source: sociopatterns.org (see config.yaml
    datasets.temporal.sfhh).
    """
    return _load_plain_tij_dataset(
        "sfhh_temporal", "sfhh", "SFHH_tij.dat.gz",
        "Genois & Barrat (2018), EPJ Data Science 7, 11 - SocioPatterns SFHH conference",
        license_note="SocioPatterns, CC0 (public domain)", archive_kind="gz", **overrides,
    )


@register_dataset("ht09_temporal")
def load_ht09_temporal(**overrides: Any) -> Dataset:
    """SocioPatterns Hypertext 2009 conference dynamic contact network (2.5 days of measurement, 113 nodes).

    Source file format: `t i j` (time in seconds, two contact nodes) - no
    publicly available node-class metadata, so y=None (no substitute class
    is fabricated). Source: sociopatterns.org (see config.yaml
    datasets.temporal.ht09).
    """
    return _load_plain_tij_dataset(
        "ht09_temporal", "ht09", "ht2009_contact_list.dat.gz",
        "Isella et al. (2011), J. Theoretical Biology 271, 166 - SocioPatterns Hypertext 2009",
        license_note="SocioPatterns, CC BY-NC-SA 3.0", archive_kind="gz", **overrides,
    )


@register_dataset("email_eu_core_temporal")
def load_email_eu_core_temporal(**overrides: Any) -> Dataset:
    """SNAP email-Eu-core-temporal - a NON-SocioPatterns control dataset (803
    days of measurement, 986 active nodes out of 1005 with an assigned department).

    Source file format: `i j t` (NOTE - a different column order than the
    SocioPatterns contact networks above: sender, recipient, time in
    seconds). The directed graph (email sender -> recipient) is collapsed,
    for consistency with the other loaders, into an undirected contact graph
    (edge i-j with weight = number of emails in the given time window) - see
    the comment in config.yaml datasets.temporal.email_eu_core. The node
    class (42 departments) is in a separate metadata_url file. Source:
    snap.stanford.edu (Paranjape, Benson & Leskovec, WSDM 2017).
    """
    cfg = load_config()
    spec = cfg["datasets"]["temporal"]["email_eu_core"]
    time_bin_sec = overrides.get("time_bin_sec", spec["time_bin_sec"])
    dataset_name = "email_eu_core_temporal"
    license_note = "SNAP, free for research/teaching, cite Paranjape, Benson & Leskovec (WSDM 2017)"

    cache_dir = ensure_dir(get_path("temporal_cache_dir"))
    raw_path = download_file(
        spec["data_url"], cache_dir / "email-Eu-core-temporal.txt.gz", name=dataset_name, license_note=license_note,
    )
    with gzip.open(raw_path, "rt") as f:
        edges_df = pd.read_csv(f, sep=r"\s+", header=None, names=["i", "j", "t"])
    edges_df = edges_df[["t", "i", "j"]]

    meta_path = download_file(
        spec["metadata_url"], cache_dir / "email-Eu-core-department-labels.txt.gz",
        name=f"{dataset_name}_metadata", license_note=license_note,
    )
    with gzip.open(meta_path, "rt") as f:
        labels_df = pd.read_csv(f, sep=r"\s+", header=None, names=["node", "label"])

    nodes = sorted(set(edges_df["i"]).union(edges_df["j"]))
    label_map = dict(zip(labels_df["node"], labels_df["label"]))
    missing = [n for n in nodes if n not in label_map]
    if missing:
        raise ValueError(
            f"Dataset '{dataset_name}': {len(missing)} nodes from the edge list are missing from the department file "
            f"({spec['metadata_url']}), e.g. {missing[:5]} - data is not fabricated, check the source."
        )
    from sklearn.preprocessing import LabelEncoder

    y = LabelEncoder().fit_transform([label_map[n] for n in nodes]).astype(np.int64)

    snapshots = _build_snapshots(edges_df, time_bin_sec=time_bin_sec, nodes=nodes)

    static_graph = nx.Graph()
    static_graph.add_nodes_from(nodes)
    for i, j in zip(edges_df["i"], edges_df["j"]):
        if static_graph.has_edge(i, j):
            static_graph[i][j]["weight"] += 1
        else:
            static_graph.add_edge(i, j, weight=1)

    return Dataset(
        name=dataset_name, kind="temporal", graph=static_graph, y=y,
        meta={
            "source": spec["data_url"],
            "citation": "Paranjape, Benson & Leskovec (2017), WSDM - SNAP email-Eu-core-temporal",
            "nodes": nodes,
            "edges_df": edges_df,
            "time_bin_sec": time_bin_sec,
            "snapshots": snapshots,
            "n_snapshots": len(snapshots),
        },
    )


@register_dataset("dblp_yearly")
def load_dblp_yearly(**_overrides: Any) -> Dataset:
    """DBLP co-authorship, yearly slices - requires a locally prepared file.

    There is no single stable, directly downloadable URL with ready-made
    yearly slices of the DBLP co-authorship network. This loader therefore
    expects pre-prepared data in `src/data/temporal/dblp_yearly/` (one CSV
    per year, columns: author_i, author_j, year), which the author downloads
    and processes e.g. from the DBLP XML dump (https://dblp.org/xml/release/)
    or SNAP com-dblp (https://snap.stanford.edu/data/com-DBLP.html) via
    their own preprocessing.
    """
    cache_dir = get_path("temporal_cache_dir") / "dblp_yearly"
    if not cache_dir.exists() or not any(cache_dir.glob("*.csv")):
        raise FileNotFoundError(
            "DBLP yearly slices not found. Expected: CSV files "
            f"'{cache_dir}/<year>.csv' with columns author_i, author_j, year. "
            "No direct stable source with ready-made yearly slices exists - "
            "prepare them manually from the DBLP XML dump (https://dblp.org/xml/release/) "
            "or SNAP com-DBLP (https://snap.stanford.edu/data/com-DBLP.html). "
            "Data is not fabricated."
        )
    raise NotImplementedError(
        "Found local DBLP data, but parsing of yearly slices is not yet "
        "implemented (to be added together with the temporal experiment)."
    )
