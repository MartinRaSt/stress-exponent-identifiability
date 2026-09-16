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
Network (graph) datasets: bundled in networkx (karate, les_miserables,
florentine) and downloaded (dolphins/football/polbooks from M. Newman,
email-Eu-core from SNAP). Node classes (y) are taken from the original
metadata when available.

K8 (documentation/2026-09-12_plan_smeru_clanku.md): 5 larger graphs for E3
(2k-11k nodes, see reserse/2026-09-12_r1_r6_podklady_smeru_ab.md R5 for
sources/licenses/citations/n/m/LCC): `cora` (citation network, ground-truth
classes), `power_grid`, `facebook_combined`, `ca_grqc`, `pgp` (Louvain
communities, see `_louvain_labels`). Every dataset has in
`meta['community_source']` either 'ground_truth' (real classes from the
data) or 'louvain' (unsupervised detection with seed
`datasets.graphs.louvain_seed` from config.yaml) - propagated to the CSV
column `community_source` in `src/experiments/exp3_graph_layout.py`.
"""
from __future__ import annotations

import gzip
import io
import tarfile
import zipfile
from typing import Any

import networkx as nx
import numpy as np

from src.common.config import ensure_dir, get_path, load_config
from src.datasets.download import download_file
from src.datasets.registry import Dataset, register_dataset

# ---------------------------------------------------------------------------
# Bundled in networkx
# ---------------------------------------------------------------------------


@register_dataset("karate")
def load_karate(**_overrides: Any) -> Dataset:
    """Zachary's Karate Club (34 nodes), y = membership in one of the two factions."""
    g = nx.karate_club_graph()
    y = np.array([0 if g.nodes[n]["club"] == "Mr. Hi" else 1 for n in g.nodes()], dtype=np.int64)
    return Dataset(name="karate", kind="graph", graph=g, y=y,
                    meta={"source": "networkx.karate_club_graph", "citation": "Zachary (1977)"})


@register_dataset("les_miserables")
def load_les_miserables(**_overrides: Any) -> Dataset:
    """Co-occurrence network of characters in Les Miserables, no classes."""
    g = nx.les_miserables_graph()
    return Dataset(name="les_miserables", kind="graph", graph=g, y=None,
                    meta={"source": "networkx.les_miserables_graph", "citation": "Knuth (1993)"})


@register_dataset("florentine")
def load_florentine(**_overrides: Any) -> Dataset:
    """Florentine families marriage network (15th century), no classes."""
    g = nx.florentine_families_graph()
    return Dataset(name="florentine", kind="graph", graph=g, y=None,
                    meta={"source": "networkx.florentine_families_graph", "citation": "Padgett & Ansell (1993)"})


# ---------------------------------------------------------------------------
# M. Newman netdata (gml inside a zip archive)
# ---------------------------------------------------------------------------


def _load_netdata_graph(key: str, **_overrides: Any) -> Dataset:
    """Download and load a single graph from M. Newman's netdata collection (gml in a zip)."""
    cfg = load_config()
    base_url = cfg["datasets"]["graphs"]["netdata_base_url"]
    url = f"{base_url}{key}.zip"
    cache_dir = ensure_dir(get_path("graphs_cache_dir"))
    zip_path = cache_dir / f"{key}.zip"
    download_file(url, zip_path, name=f"netdata_{key}", license_note="M. Newman netdata, academic use")

    with zipfile.ZipFile(zip_path) as zf:
        gml_names = [n for n in zf.namelist() if n.endswith(".gml")]
        if not gml_names:
            raise RuntimeError(f"Archive {zip_path} does not contain a .gml file (contents: {zf.namelist()})")
        gml_text = zf.read(gml_names[0]).decode("utf-8", errors="ignore")

    gml_path = cache_dir / f"{key}.gml"
    if not gml_path.exists():
        gml_path.write_text(gml_text, encoding="utf-8")
    g = nx.read_gml(gml_path, label="id")

    y = None
    values = nx.get_node_attributes(g, "value")
    if values:
        raw = np.array([values[n] for n in g.nodes()])
        if raw.dtype.kind in ("U", "S", "O"):
            from sklearn.preprocessing import LabelEncoder

            y = LabelEncoder().fit_transform(raw).astype(np.int64)
        else:
            y = raw.astype(np.int64)

    return Dataset(name=key, kind="graph", graph=g, y=y,
                    meta={"source": url, "citation": "M. Newman netdata collection"})


@register_dataset("dolphins")
def load_dolphins(**overrides: Any) -> Dataset:
    """Dolphin social interaction network (62 nodes), no classes in the original data."""
    return _load_netdata_graph("dolphins", **overrides)


@register_dataset("football")
def load_football(**overrides: Any) -> Dataset:
    """American college football schedule network (115 nodes), y = conference (value in gml)."""
    return _load_netdata_graph("football", **overrides)


@register_dataset("polbooks")
def load_polbooks(**overrides: Any) -> Dataset:
    """Political books co-purchase network (105 nodes), y = political orientation."""
    return _load_netdata_graph("polbooks", **overrides)


# ---------------------------------------------------------------------------
# SNAP email-Eu-core
# ---------------------------------------------------------------------------


@register_dataset("email_eu_core")
def load_email_eu_core(**_overrides: Any) -> Dataset:
    """SNAP email-Eu-core (1005 nodes), y = department (department labels)."""
    cfg = load_config()
    spec = cfg["datasets"]["graphs"]["email_eu_core"]
    cache_dir = ensure_dir(get_path("graphs_cache_dir"))

    edges_gz = download_file(spec["edges_url"], cache_dir / "email-Eu-core.txt.gz",
                              name="email_eu_core_edges", license_note="SNAP, academic use")
    labels_gz = download_file(spec["labels_url"], cache_dir / "email-Eu-core-department-labels.txt.gz",
                               name="email_eu_core_labels", license_note="SNAP, academic use")

    with gzip.open(edges_gz, "rt") as f:
        edges = [tuple(int(x) for x in line.split()) for line in f if line.strip()]
    g = nx.Graph()
    g.add_edges_from(edges)

    with gzip.open(labels_gz, "rt") as f:
        label_map = dict(tuple(int(x) for x in line.split()) for line in f if line.strip())
    y = np.array([label_map[n] for n in g.nodes()], dtype=np.int64)

    return Dataset(name="email_eu_core", kind="graph", graph=g, y=y,
                    meta={"source": spec["edges_url"], "citation": "Leskovec & Krevl, SNAP Datasets, email-Eu-core"})


# ---------------------------------------------------------------------------
# K8: larger graphs for E3 (2k-11k nodes) - see
# reserse/2026-09-12_r1_r6_podklady_smeru_ab.md R5 for sources/licenses/citations
# ---------------------------------------------------------------------------


def _louvain_labels(g: nx.Graph, seed: int) -> tuple[np.ndarray, str]:
    """Louvain community detection (networkx.algorithms.community.louvain_communities)
    for graphs without ground-truth classes - deterministic thanks to `seed`
    from config.yaml (`datasets.graphs.louvain_seed`). Returns (y, 'louvain'),
    where y is in the same order as `list(g.nodes())` (same convention as
    the ground-truth labels elsewhere in this module)."""
    from networkx.algorithms.community import louvain_communities

    communities = louvain_communities(g, seed=seed)
    label_of: dict[Any, int] = {}
    for idx, community in enumerate(communities):
        for node in community:
            label_of[node] = idx
    y = np.array([label_of[node] for node in g.nodes()], dtype=np.int64)
    return y, "louvain"


@register_dataset("cora")
def load_cora(**_overrides: Any) -> Dataset:
    """Cora citation network (2708 machine-learning papers, 7 topic classes),
    LINQS distribution (the standard GNN-benchmark version of cora.content/cora.cites).

    Source: LINQS `https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz`
    (downloaded and verified directly on 2026-09-12, see reserse/
    2026-09-12_r1_r6_podklady_smeru_ab.md R5). Citation: Sen, Namata, Bilgic,
    Getoor, Galligher, Eliassi-Rad (2008), "Collective Classification in
    Network Data", AI Magazine 29(3):93-106. TODO (R5): the DOI of this
    citation was not verified with certainty - verify before use in the
    article (podklady/kontrola_referenci.md). License: LINQS - freely
    available for research purposes, no formal SPDX license stated on the
    source page.

    The original file has 2708 nodes / 5429 citation lines (5278 unique
    undirected edges after conversion to `nx.Graph` - see the K8
    documentation for the exact verification). The largest connected
    component (LCC) used further down in the E3 pipeline
    (`exp3_graph_layout._to_largest_component`) is SMALLER than the whole
    graph (2485 nodes, not 2708) - unlike the assumption in R5 ("the whole
    graph is the LCC"), which the K8 documentation explicitly corrects.

    y = ground-truth topic class of the paper (7 classes, last column of
    cora.content) -> `meta['community_source']='ground_truth'`.
    """
    cfg = load_config()
    spec = cfg["datasets"]["graphs"]["cora"]
    cache_dir = ensure_dir(get_path("graphs_cache_dir"))
    tgz_path = cache_dir / "cora.tgz"
    download_file(spec["archive_url"], tgz_path, name="cora_archive",
                   license_note="LINQS, academic use (see the load_cora docstring)")

    with tarfile.open(tgz_path, "r:gz") as tf:
        content_member = tf.extractfile("cora/cora.content")
        cites_member = tf.extractfile("cora/cora.cites")
        if content_member is None or cites_member is None:
            raise RuntimeError(f"Archive {tgz_path} does not contain the expected files cora/cora.content and cora/cora.cites.")
        content_bytes = content_member.read()
        cites_bytes = cites_member.read()

    node_ids: list[str] = []
    labels_raw: list[str] = []
    for line in content_bytes.decode("utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        node_ids.append(parts[0])
        labels_raw.append(parts[-1])
    if len(node_ids) != 2708:
        raise ValueError(f"cora.content: expected 2708 rows, found {len(node_ids)} - the downloaded archive does not match the expected version.")

    g = nx.Graph()
    g.add_nodes_from(node_ids)
    node_set = set(node_ids)
    for line in cites_bytes.decode("utf-8").splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        u, v = parts
        if u in node_set and v in node_set and u != v:
            g.add_edge(u, v)

    from sklearn.preprocessing import LabelEncoder

    y = LabelEncoder().fit_transform(np.array(labels_raw)).astype(np.int64)

    return Dataset(name="cora", kind="graph", graph=g, y=y,
                    meta={"source": spec["archive_url"],
                          "citation": "Sen, Namata, Bilgic, Getoor, Galligher & Eliassi-Rad (2008), AI Magazine 29(3):93-106",
                          "community_source": "ground_truth"})


def _load_konect_edge_graph(key: str, archive_url: str, inner_name: str, citation: str) -> Dataset:
    """Download and load a graph from KONECT (a tar.bz2 archive containing
    `<inner_name>/out.<inner_name>`: lines starting with '%' are comments,
    other lines are 1-indexed edges 'u v'). No ground-truth classes ->
    Louvain communities (`_louvain_labels`, seed `datasets.graphs.louvain_seed`)."""
    cfg = load_config()
    cache_dir = ensure_dir(get_path("graphs_cache_dir"))
    archive_path = cache_dir / f"{key}.tar.bz2"
    download_file(archive_url, archive_path, name=f"konect_{key}", license_note="KONECT, academic use (verify the license before use in the article)")

    with tarfile.open(archive_path, "r:bz2") as tf:
        member_name = f"{inner_name}/out.{inner_name}"
        member = tf.extractfile(member_name)
        if member is None:
            raise RuntimeError(f"Archive {archive_path} does not contain the expected file {member_name}.")
        data_bytes = member.read()

    g = nx.Graph()
    for line in data_bytes.decode("utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        u, v = int(parts[0]), int(parts[1])
        if u != v:
            g.add_edge(u, v)

    louvain_seed = int(cfg["datasets"]["graphs"]["louvain_seed"])
    y, community_source = _louvain_labels(g, louvain_seed)

    return Dataset(name=key, kind="graph", graph=g, y=y,
                    meta={"source": archive_url, "citation": citation, "community_source": community_source})


@register_dataset("power_grid")
def load_power_grid(**_overrides: Any) -> Dataset:
    """US power grid (Watts-Strogatz 1998 small-world benchmark, 4941
    nodes/6594 edges, already a whole connected graph), KONECT `opsahl-powergrid`.

    Source: `http://konect.cc/files/download.tsv.opsahl-powergrid.tar.bz2`
    (verified directly on 2026-09-12, see
    reserse/2026-09-12_r1_r6_podklady_smeru_ab.md R5 - a direct request to
    `konect.cc` from a previous run failed with ECONNREFUSED, this attempt
    succeeded). Citation: Watts, D. J., Strogatz, S. H. (1998), "Collective
    dynamics of 'small-world' networks", Nature 393:440-442, DOI
    `10.1038/30918`. License: KONECT - mostly CC terms, recommended to
    verify the specific license text before use in the article (R5).

    No ground-truth classes -> Louvain community detection,
    `meta['community_source']='louvain'`.
    """
    cfg = load_config()
    spec = cfg["datasets"]["graphs"]["power_grid"]
    return _load_konect_edge_graph(
        "power_grid", spec["archive_url"], "opsahl-powergrid",
        "Watts & Strogatz (1998), Nature 393:440-442, DOI 10.1038/30918",
    )


@register_dataset("pgp")
def load_pgp(**_overrides: Any) -> Dataset:
    """PGP web-of-trust network (arenas-pgp, 10680 nodes / 24316 edges,
    already distributed as a single connected component), KONECT
    `arenas-pgp` - the largest graph in the K8 selection, testing the upper
    scalability limit of E3 (resistance distance via an eigh-based
    pseudoinverse, see `src/datasets/graph_distance.py`).

    Source: `http://konect.cc/files/download.tsv.arenas-pgp.tar.bz2`
    (verified directly on 2026-09-12, see
    reserse/2026-09-12_r1_r6_podklady_smeru_ab.md R5). Citation: Boguna, M.,
    Pastor-Satorras, R., Diaz-Guilera, A., Arenas, A. (2004), "Models of
    social networks based on social distance attachment", Physical Review E
    70:056122. TODO (R5): the exact DOI format `10.1103/PhysRevE.70.056122`
    is standard for PRE, but was not directly verified in Crossref - confirm
    before use in the article (podklady/kontrola_referenci.md). License:
    KONECT/networkrepository - publicly available for research.

    No ground-truth classes -> Louvain community detection,
    `meta['community_source']='louvain'`.
    """
    cfg = load_config()
    spec = cfg["datasets"]["graphs"]["pgp"]
    return _load_konect_edge_graph(
        "pgp", spec["archive_url"], "arenas-pgp",
        "Boguna, Pastor-Satorras, Diaz-Guilera & Arenas (2004), Phys. Rev. E 70:056122",
    )


def _load_snap_edge_graph(key: str, edges_url: str, citation: str, skip_hash_comments: bool) -> Dataset:
    """Download and load a SNAP graph from a .txt.gz edge list ('u v',
    optionally with leading comment lines starting with '#'). No
    ground-truth classes -> Louvain communities (`_louvain_labels`)."""
    cfg = load_config()
    cache_dir = ensure_dir(get_path("graphs_cache_dir"))
    edges_gz = download_file(edges_url, cache_dir / f"{key}.txt.gz", name=f"{key}_edges", license_note="SNAP, academic use")

    g = nx.Graph()
    with gzip.open(edges_gz, "rt") as f:
        for line in f:
            line = line.strip()
            if not line or (skip_hash_comments and line.startswith("#")):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            if u != v:
                g.add_edge(u, v)

    louvain_seed = int(cfg["datasets"]["graphs"]["louvain_seed"])
    y, community_source = _louvain_labels(g, louvain_seed)

    return Dataset(name=key, kind="graph", graph=g, y=y,
                    meta={"source": edges_url, "citation": citation, "community_source": community_source})


@register_dataset("facebook_combined")
def load_facebook_combined(**_overrides: Any) -> Dataset:
    """SNAP ego-Facebook (`facebook_combined`, 4039 nodes / 88234 edges, a
    union of 10 ego networks, a dense social graph - a contrast to the
    sparse collaboration graphs in this collection).

    Source: `https://snap.stanford.edu/data/facebook_combined.txt.gz`
    (verified directly on 2026-09-12, see
    reserse/2026-09-12_r1_r6_podklady_smeru_ab.md R5). Citation: McAuley,
    J., Leskovec, J. (2012), "Learning to Discover Social Circles in Ego
    Networks", NIPS 2012. TODO (R5): the DOI of this citation was not
    verified with certainty - verify before use in the article. License:
    SNAP - publicly available for research, no formal license stated on the page.

    The original data has 10 overlapping ego circles ("circles", 193
    total) - this is NOT a clean node partition, so Louvain community
    detection is used here (`meta['community_source']='louvain'`), not the
    ego-circles directly."""
    cfg = load_config()
    spec = cfg["datasets"]["graphs"]["facebook_combined"]
    return _load_snap_edge_graph(
        "facebook_combined", spec["edges_url"], "McAuley & Leskovec (2012), NIPS", skip_hash_comments=False,
    )


@register_dataset("ca_grqc")
def load_ca_grqc(**_overrides: Any) -> Dataset:
    """SNAP `ca-GrQc` (a collaboration network of Arxiv General Relativity
    paper authors, 5241 nodes / 14484 edges after conversion to an
    undirected graph without self-loops, LCC 4158 nodes / 79.3% - a sparse
    graph, a contrast to `facebook_combined`).

    Source: `https://snap.stanford.edu/data/ca-GrQc.txt.gz` (verified
    directly on 2026-09-12, see reserse/2026-09-12_r1_r6_podklady_smeru_ab.md
    R5). Citation: Leskovec, J., Kleinberg, J., Faloutsos, C. (2007), "Graph
    Evolution: Densification and Shrinking Diameters", ACM Transactions on
    Knowledge Discovery from Data 1(1). TODO (R5): the exact DOI format is
    uncertain - verify before use in the article. License: SNAP - publicly
    available for research.

    No ground-truth classes -> Louvain community detection,
    `meta['community_source']='louvain'`.
    """
    cfg = load_config()
    spec = cfg["datasets"]["graphs"]["ca_grqc"]
    return _load_snap_edge_graph(
        "ca_grqc", spec["edges_url"], "Leskovec, Kleinberg & Faloutsos (2007), ACM TKDD 1(1)", skip_hash_comments=True,
    )
