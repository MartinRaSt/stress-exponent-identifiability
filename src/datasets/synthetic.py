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
Synthetic manifold/cluster datasets for the dimensionality-reduction
benchmark. All generators are seeded by the random_state parameter from
config.yaml (or an overriding kwarg), so repeated calls with the same seed
return bit-identical data.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_s_curve, make_swiss_roll

from src.common.config import load_config
from src.datasets.registry import Dataset, register_dataset


def _synth_params(name: str, overrides: dict[str, Any]) -> dict[str, Any]:
    """Load the generator's default parameters from config.yaml and override them with overrides."""
    cfg = load_config()
    try:
        defaults = dict(cfg["datasets"]["synthetic"][name])
    except KeyError as exc:
        raise KeyError(f"config.yaml is missing datasets.synthetic.{name}") from exc
    defaults.update({k: v for k, v in overrides.items() if v is not None})
    return defaults


@register_dataset("swiss_roll")
def load_swiss_roll(**overrides: Any) -> Dataset:
    """Swiss roll (3D manifold with an intrinsic 2D structure)."""
    p = _synth_params("swiss_roll", overrides)
    X, t = make_swiss_roll(n_samples=p["n_samples"], noise=p["noise"], random_state=p["random_state"])
    return Dataset(name="swiss_roll", kind="vector", X=X.astype(np.float32), y=t.astype(np.float32),
                    meta={"params": p, "description": "3D swiss roll manifold, y = position on the roll"})


@register_dataset("s_curve")
def load_s_curve(**overrides: Any) -> Dataset:
    """S-curve (3D manifold)."""
    p = _synth_params("s_curve", overrides)
    X, t = make_s_curve(n_samples=p["n_samples"], noise=p["noise"], random_state=p["random_state"])
    return Dataset(name="s_curve", kind="vector", X=X.astype(np.float32), y=t.astype(np.float32),
                    meta={"params": p, "description": "3D S-curve manifold, y = position on the curve"})


@register_dataset("sphere")
def load_sphere(**overrides: Any) -> Dataset:
    """Points uniformly distributed on the surface of a 3D unit sphere."""
    p = _synth_params("sphere", overrides)
    rng = np.random.default_rng(p["random_state"])
    n = p["n_samples"]
    # uniform sampling on the sphere surface (Marsaglia)
    vec = rng.normal(size=(n, 3))
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    if p["noise"] > 0:
        vec = vec + rng.normal(scale=p["noise"], size=vec.shape)
    colatitude = np.arccos(np.clip(vec[:, 2], -1.0, 1.0)).astype(np.float32)
    return Dataset(name="sphere", kind="vector", X=vec.astype(np.float32), y=colatitude,
                    meta={"params": p, "description": "Points on the surface of a 3D sphere, y = colatitude"})


@register_dataset("severed_sphere")
def load_severed_sphere(**overrides: Any) -> Dataset:
    """Sphere with a polar cap removed (a classic example for Isomap vs. MDS/Sammon)."""
    p = _synth_params("severed_sphere", overrides)
    rng = np.random.default_rng(p["random_state"])
    n_target = p["n_samples"]
    cap = p["cap_polar_angle"]
    # generate more points than needed, since the cap is cut off and discarded
    oversample = int(np.ceil(n_target / max(1e-6, (1.0 - cap / np.pi)))) + 100
    vec = rng.normal(size=(oversample, 3))
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    colatitude = np.arccos(np.clip(vec[:, 2], -1.0, 1.0))
    keep = colatitude > cap
    vec = vec[keep][:n_target]
    colatitude = colatitude[keep][:n_target]
    if vec.shape[0] < n_target:
        raise RuntimeError(
            f"severed_sphere: not enough points remain after removing the cap (cap_polar_angle={cap}) "
            f"({vec.shape[0]} < {n_target}). Reduce n_samples or cap_polar_angle in config.yaml."
        )
    if p["noise"] > 0:
        vec = vec + rng.normal(scale=p["noise"], size=vec.shape)
    return Dataset(name="severed_sphere", kind="vector", X=vec.astype(np.float32),
                    y=colatitude.astype(np.float32),
                    meta={"params": p, "description": "Sphere without a polar cap, y = colatitude"})


@register_dataset("helix")
def load_helix(**overrides: Any) -> Dataset:
    """3D helix with a given number of turns."""
    p = _synth_params("helix", overrides)
    rng = np.random.default_rng(p["random_state"])
    n = p["n_samples"]
    theta = np.linspace(0.0, 2.0 * np.pi * p["n_turns"], n)
    X = np.column_stack([np.cos(theta), np.sin(theta), theta / (2.0 * np.pi * p["n_turns"])])
    if p["noise"] > 0:
        X = X + rng.normal(scale=p["noise"], size=X.shape)
    return Dataset(name="helix", kind="vector", X=X.astype(np.float32), y=theta.astype(np.float32),
                    meta={"params": p, "description": "3D helix, y = angle theta"})


@register_dataset("torus")
def load_torus(**overrides: Any) -> Dataset:
    """Surface of a 3D torus parametrized by angles theta (main circle) and phi (tube)."""
    p = _synth_params("torus", overrides)
    rng = np.random.default_rng(p["random_state"])
    n = p["n_samples"]
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    R, r = p["r_center"], p["r_tube"]
    X = np.column_stack([
        (R + r * np.cos(phi)) * np.cos(theta),
        (R + r * np.cos(phi)) * np.sin(theta),
        r * np.sin(phi),
    ])
    if p["noise"] > 0:
        X = X + rng.normal(scale=p["noise"], size=X.shape)
    return Dataset(name="torus", kind="vector", X=X.astype(np.float32), y=theta.astype(np.float32),
                    meta={"params": p, "description": "Torus surface, y = angle theta"})


@register_dataset("two_moons")
def load_two_moons(**overrides: Any) -> Dataset:
    """Two interleaving half-moons (2D), a classic test of nonlinear methods."""
    p = _synth_params("two_moons", overrides)
    X, y = make_moons(n_samples=p["n_samples"], noise=p["noise"], random_state=p["random_state"])
    return Dataset(name="two_moons", kind="vector", X=X.astype(np.float32), y=y.astype(np.int64),
                    meta={"params": p, "description": "Two moons, y = class (0/1)"})


@register_dataset("gaussian_clusters")
def load_gaussian_clusters(**overrides: Any) -> Dataset:
    """Isotropic Gaussian clusters in high dimension."""
    p = _synth_params("gaussian_clusters", overrides)
    X, y = make_blobs(
        n_samples=p["n_samples"],
        n_features=p["n_features"],
        centers=p["n_clusters"],
        cluster_std=p["cluster_std"],
        random_state=p["random_state"],
    )
    return Dataset(name="gaussian_clusters", kind="vector", X=X.astype(np.float32), y=y.astype(np.int64),
                    meta={"params": p, "description": "Gaussian clusters, y = cluster"})


@register_dataset("hierarchical_clusters")
def load_hierarchical_clusters(**overrides: Any) -> Dataset:
    """Two-level hierarchical clusters (top-level + subclusters within each)."""
    p = _synth_params("hierarchical_clusters", overrides)
    rng = np.random.default_rng(p["random_state"])
    n_top, n_sub = p["n_top_clusters"], p["n_sub_clusters"]
    n_features = p["n_features"]
    n_total_clusters = n_top * n_sub
    n_samples = p["n_samples"]

    # top-level centers
    top_centers = rng.normal(scale=p["top_std"], size=(n_top, n_features))

    # distribute the sample count evenly across all fine-grained clusters
    base = n_samples // n_total_clusters
    remainder = n_samples - base * n_total_clusters
    counts = np.full(n_total_clusters, base, dtype=int)
    counts[:remainder] += 1  # deterministic remainder top-up

    X_parts, y_parts, y_top_parts = [], [], []
    cluster_idx = 0
    for top_i in range(n_top):
        # sub-centers around the top-level center
        sub_centers = top_centers[top_i] + rng.normal(scale=p["top_std"] / 3.0, size=(n_sub, n_features))
        for sub_i in range(n_sub):
            n_c = counts[cluster_idx]
            pts = sub_centers[sub_i] + rng.normal(scale=p["sub_std"], size=(n_c, n_features))
            X_parts.append(pts)
            y_parts.append(np.full(n_c, cluster_idx, dtype=np.int64))
            y_top_parts.append(np.full(n_c, top_i, dtype=np.int64))
            cluster_idx += 1

    X = np.vstack(X_parts).astype(np.float32)
    y = np.concatenate(y_parts)
    y_top = np.concatenate(y_top_parts)

    return Dataset(
        name="hierarchical_clusters", kind="vector", X=X, y=y,
        meta={"params": p, "y_top": y_top, "description": "Hierarchical clusters, y = fine cluster, meta['y_top'] = coarse cluster"},
    )


# ---------------------------------------------------------------------------
# Q1 step 2 (reserse/2026-09-14_specifikace_rozsireni_q1.md, A.3.3) - 6 new
# synthetic generators, deterministic (random_state from config.yaml),
# guaranteed to fall in regime L (low rho_NN) thanks to the low intrinsic
# manifold dimension (d_int~1-2) at n>=1000. y = index of one of 10
# equally-wide parameter bands (stratification/geometric metrics), NOT a
# class in the classification sense. Standardization disabled (see
# config.yaml datasets.standardize - NOT listed here).
# ---------------------------------------------------------------------------

def _band_labels(param: np.ndarray, n_bands: int) -> np.ndarray:
    """Split a 1D parameter (e.g. angle/manifold parametrization) into
    `n_bands` equally-wide bands [min,max] and return the band index
    (0..n_bands-1) for each point - deterministic, depends only on the
    values of `param` (no rng)."""
    lo, hi = float(np.min(param)), float(np.max(param))
    if hi <= lo:
        raise ValueError("_band_labels: the parameter has zero range, bands are not defined.")
    edges = np.linspace(lo, hi, n_bands + 1)
    idx = np.searchsorted(edges, param, side="right") - 1
    return np.clip(idx, 0, n_bands - 1).astype(np.int64)


def _dedup_new_dataset(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """A.4 point 3 (Q1 step 2 specification): removal of exact duplicate
    rows of X BEFORE standardization/subsampling - even here (continuous
    synthetic coordinates with Gaussian noise practically never collide,
    but the step is applied uniformly to ALL new datasets registered after
    2026-09-14, see src/datasets/subsample.py::deduplicate_rows and
    src/datasets/uci_extra.py)."""
    from src.datasets.subsample import deduplicate_rows

    X_dedup, y_dedup, n_removed = deduplicate_rows(X, y)
    meta_extra = {"dedup": True, "n_duplicates_removed": n_removed, "n_before_dedup": int(X.shape[0])}
    return X_dedup, y_dedup, meta_extra


@register_dataset("swiss_roll_hole")
def load_swiss_roll_hole(**overrides: Any) -> Dataset:
    """Swiss roll with a rectangular hole (Donoho & Grimes 2003,
    DOI 10.1073/pnas.1031596100) - rejection sampling of points in the hole t x h."""
    p = _synth_params("swiss_roll_hole", overrides)
    rng = np.random.default_rng(p["random_state"])
    n_target = int(p["n_samples"])
    t_lo, t_hi = float(p["t_min"]), float(p["t_max"])
    h_max = float(p["h_max"])
    hole_t_lo, hole_t_hi = float(p["hole_t_min"]), float(p["hole_t_max"])
    hole_h_lo, hole_h_hi = float(p["hole_h_min"]), float(p["hole_h_max"])
    noise = float(p["noise"])

    kept_t: list[np.ndarray] = []
    kept_h: list[np.ndarray] = []
    n_have = 0
    max_batches = 1000
    batch = max(2 * n_target, 1000)
    for _ in range(max_batches):
        t = rng.uniform(t_lo, t_hi, batch)
        h = rng.uniform(0.0, h_max, batch)
        in_hole = (t >= hole_t_lo) & (t <= hole_t_hi) & (h >= hole_h_lo) & (h <= hole_h_hi)
        keep = ~in_hole
        kept_t.append(t[keep])
        kept_h.append(h[keep])
        n_have += int(keep.sum())
        if n_have >= n_target:
            break
    else:
        raise RuntimeError(f"swiss_roll_hole: not enough points remain outside the hole after {max_batches} batches ({n_have} < {n_target}).")

    t_all = np.concatenate(kept_t)[:n_target]
    h_all = np.concatenate(kept_h)[:n_target]
    X = np.column_stack([t_all * np.cos(t_all), h_all, t_all * np.sin(t_all)])
    if noise > 0:
        X = X + rng.normal(scale=noise, size=X.shape)
    y = _band_labels(t_all, int(p["n_bands"]))
    X, y, dedup_meta = _dedup_new_dataset(X.astype(np.float32), y)
    return Dataset(name="swiss_roll_hole", kind="vector", X=X, y=y,
                    meta={"params": p, "description": "Swiss roll with a rectangular hole (Donoho & Grimes 2003), y = band of parameter t", **dedup_meta})


@register_dataset("gaussian_clusters_separated")
def load_gaussian_clusters_separated(**overrides: Any) -> Dataset:
    """K well-separated isotropic Gaussian clusters (min. center distance
    >= `min_center_dist`, rejection sampling of the whole center configuration)."""
    p = _synth_params("gaussian_clusters_separated", overrides)
    rng = np.random.default_rng(p["random_state"])
    K = int(p["n_clusters"])
    n_features = int(p["n_features"])
    center_std = float(p["center_std"])
    min_center_dist = float(p["min_center_dist"])
    cluster_std = float(p["cluster_std"])
    n_per_cluster = int(p["n_per_cluster"])
    max_attempts = int(p["max_attempts"])

    centers = None
    for _ in range(max_attempts):
        cand = rng.normal(scale=center_std, size=(K, n_features))
        d = np.sqrt(((cand[:, None, :] - cand[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(d, np.inf)
        if d.min() >= min_center_dist:
            centers = cand
            break
    if centers is None:
        raise RuntimeError(
            f"gaussian_clusters_separated: failed to find {K} centers with mutual distance "
            f">= {min_center_dist} in {max_attempts} attempts (increase center_std or decrease min_center_dist)."
        )

    X_parts, y_parts = [], []
    for k in range(K):
        pts = centers[k] + rng.normal(scale=cluster_std, size=(n_per_cluster, n_features))
        X_parts.append(pts)
        y_parts.append(np.full(n_per_cluster, k, dtype=np.int64))
    X = np.vstack(X_parts).astype(np.float32)
    y = np.concatenate(y_parts)
    X, y, dedup_meta = _dedup_new_dataset(X, y)
    return Dataset(name="gaussian_clusters_separated", kind="vector", X=X, y=y,
                    meta={"params": p, "centers": centers, "description": "K well-separated Gaussian clusters, y = cluster", **dedup_meta})


@register_dataset("mobius_strip")
def load_mobius_strip(**overrides: Any) -> Dataset:
    """Mobius strip in R^3 (a non-orientable surface with a boundary, d_int=2)."""
    p = _synth_params("mobius_strip", overrides)
    rng = np.random.default_rng(p["random_state"])
    n = int(p["n_samples"])
    u = rng.uniform(0.0, 2.0 * np.pi, n)
    v = rng.uniform(-1.0, 1.0, n)
    r = 1.0 + 0.5 * v * np.cos(u / 2.0)
    X = np.column_stack([r * np.cos(u), r * np.sin(u), 0.5 * v * np.sin(u / 2.0)])
    if p["noise"] > 0:
        X = X + rng.normal(scale=p["noise"], size=X.shape)
    y = _band_labels(u, int(p["n_bands"]))
    X, y, dedup_meta = _dedup_new_dataset(X.astype(np.float32), y)
    return Dataset(name="mobius_strip", kind="vector", X=X, y=y,
                    meta={"params": p, "description": "Mobius strip, y = band of parameter u", **dedup_meta})


@register_dataset("klein_bottle_4d")
def load_klein_bottle_4d(**overrides: Any) -> Dataset:
    """Klein bottle embedded in R^4 (without self-intersection, d_int=2, a=2, b=1)."""
    p = _synth_params("klein_bottle_4d", overrides)
    rng = np.random.default_rng(p["random_state"])
    n = int(p["n_samples"])
    a, b = float(p["a"]), float(p["b"])
    u = rng.uniform(0.0, 2.0 * np.pi, n)
    v = rng.uniform(0.0, 2.0 * np.pi, n)
    X = np.column_stack([
        (a + b * np.cos(v)) * np.cos(u),
        (a + b * np.cos(v)) * np.sin(u),
        b * np.sin(v) * np.cos(u / 2.0),
        b * np.sin(v) * np.sin(u / 2.0),
    ])
    if p["noise"] > 0:
        X = X + rng.normal(scale=p["noise"], size=X.shape)
    y = _band_labels(u, int(p["n_bands"]))
    X, y, dedup_meta = _dedup_new_dataset(X.astype(np.float32), y)
    return Dataset(name="klein_bottle_4d", kind="vector", X=X, y=y,
                    meta={"params": p, "description": "Klein bottle in R^4 (without self-intersection), y = band of parameter u", **dedup_meta})


@register_dataset("twin_peaks")
def load_twin_peaks(**overrides: Any) -> Dataset:
    """Twin peaks (Saul & Roweis 2003, JMLR 4:119-155) - a wavy surface."""
    p = _synth_params("twin_peaks", overrides)
    rng = np.random.default_rng(p["random_state"])
    n = int(p["n_samples"])
    x = rng.uniform(-1.0, 1.0, n)
    y_coord = rng.uniform(-1.0, 1.0, n)
    z = np.sin(np.pi * x) * np.tanh(3.0 * y_coord)
    X = np.column_stack([x, y_coord, z])
    if p["noise"] > 0:
        X = X + rng.normal(scale=p["noise"], size=X.shape)
    y = _band_labels(x, int(p["n_bands"]))
    X, y, dedup_meta = _dedup_new_dataset(X.astype(np.float32), y)
    return Dataset(name="twin_peaks", kind="vector", X=X, y=y,
                    meta={"params": p, "description": "Twin peaks (Saul & Roweis 2003), y = band of parameter x", **dedup_meta})


@register_dataset("trefoil_knot")
def load_trefoil_knot(**overrides: Any) -> Dataset:
    """Trefoil knot (a 1D curve in R^3, d_int=1) - the last synthetic reserve."""
    p = _synth_params("trefoil_knot", overrides)
    rng = np.random.default_rng(p["random_state"])
    n = int(p["n_samples"])
    t = rng.uniform(0.0, 2.0 * np.pi, n)
    X = np.column_stack([
        np.sin(t) + 2.0 * np.sin(2.0 * t),
        np.cos(t) - 2.0 * np.cos(2.0 * t),
        -np.sin(3.0 * t),
    ])
    if p["noise"] > 0:
        X = X + rng.normal(scale=p["noise"], size=X.shape)
    y = _band_labels(t, int(p["n_bands"]))
    X, y, dedup_meta = _dedup_new_dataset(X.astype(np.float32), y)
    return Dataset(name="trefoil_knot", kind="vector", X=X, y=y,
                    meta={"params": p, "description": "Trefoil knot (a 1D curve in R^3), y = band of parameter t", **dedup_meta})


# ---------------------------------------------------------------------------
# Addendum 2026-09-14 (reserse/2026-09-14_specifikace_rozsireni_q1.md,
# Addendum 2026-09-14: N>=26) - 2 more regime-L synthetic datasets,
# DELIBERATELY DIFFERENT from the existing 6 above (different
# curvature/topology, not just a parameter variant):
#   - hyperbolic_paraboloid: a saddle surface with a CONSTANT sign of
#     Gaussian curvature (K<0 everywhere) - unlike twin_peaks, where
#     sin(pi*x)*tanh(3*y) changes the sign of second derivatives across the
#     domain and saturates in y (K -> 0 toward the boundary). Embedded in
#     R^5 (2 smooth extra coordinates), to satisfy "embedding into a higher
#     dimension" beyond the existing R^3 surfaces.
#   - anisotropic_ellipsoid: a CLOSED orientable surface without boundary
#     (genus 0) with positive but strongly varying curvature due to an
#     extreme semi-axis ratio (1:5:25) - different from the Klein bottle
#     (non-orientable), the Mobius strip/twin_peaks/hyperbolic paraboloid
#     (surfaces with a boundary), and the already-existing core dataset
#     "sphere" (isotropic, R^3). Embedded in R^6 (3 extra angle coordinates).
# Expected rho_NN for both ~0.01-0.04 (d_int=2, n=1500, estimate from A.2:
# rho_NN ~ n^{-1/d_int} ~ 1500^{-1/2} = 0.026; anisotropic_ellipsoid is
# additionally lower due to point clustering near the poles of the short
# semi-axis - see the documentation).
# ---------------------------------------------------------------------------

@register_dataset("hyperbolic_paraboloid")
def load_hyperbolic_paraboloid(**overrides: Any) -> Dataset:
    """Hyperbolic paraboloid (a saddle surface, constant negative Gaussian
    curvature) embedded in R^5, d_int=2. Curvature differs from twin_peaks
    (see the module docstring above - Addendum 2026-09-14)."""
    p = _synth_params("hyperbolic_paraboloid", overrides)
    rng = np.random.default_rng(p["random_state"])
    n = int(p["n_samples"])
    curvature = float(p["curvature"])
    embed_amp = float(p["embed_amplitude"])
    x = rng.uniform(-1.0, 1.0, n)
    y_coord = rng.uniform(-1.0, 1.0, n)
    z = curvature * (x ** 2 - y_coord ** 2)
    e1 = embed_amp * np.sin(2.0 * np.pi * x)
    e2 = embed_amp * np.cos(2.0 * np.pi * y_coord)
    X = np.column_stack([x, y_coord, z, e1, e2])
    if p["noise"] > 0:
        X = X + rng.normal(scale=p["noise"], size=X.shape)
    y = _band_labels(x, int(p["n_bands"]))
    X, y, dedup_meta = _dedup_new_dataset(X.astype(np.float32), y)
    return Dataset(name="hyperbolic_paraboloid", kind="vector", X=X, y=y,
                    meta={"params": p, "description": "Hyperbolic paraboloid embedded in R^5 (constant negative Gaussian curvature), y = band of parameter x", **dedup_meta})


@register_dataset("anisotropic_ellipsoid")
def load_anisotropic_ellipsoid(**overrides: Any) -> Dataset:
    """Surface of a strongly anisotropic ellipsoid (semi-axis ratio a:b:c, a
    geometric progression with factor 5) embedded in R^6, d_int=2 - a
    closed orientable surface without boundary, with positive but strongly
    non-constant curvature (see the module docstring above - Addendum
    2026-09-14). The angular parametrization theta/phi is DELIBERATELY
    uniform in angles (not in area), which together with the anisotropy
    causes points to cluster near the short semi-axis (the mechanism behind
    the low rho_NN)."""
    p = _synth_params("anisotropic_ellipsoid", overrides)
    rng = np.random.default_rng(p["random_state"])
    n = int(p["n_samples"])
    a_ax, b_ax, c_ax = float(p["semi_axis_a"]), float(p["semi_axis_b"]), float(p["semi_axis_c"])
    embed_amp = float(p["embed_amplitude"])
    theta = rng.uniform(0.0, np.pi, n)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    x = a_ax * np.sin(theta) * np.cos(phi)
    y_coord = b_ax * np.sin(theta) * np.sin(phi)
    z = c_ax * np.cos(theta)
    e1 = embed_amp * np.sin(2.0 * phi)
    e2 = embed_amp * np.cos(3.0 * theta)
    e3 = embed_amp * np.sin(phi + theta)
    X = np.column_stack([x, y_coord, z, e1, e2, e3])
    if p["noise"] > 0:
        X = X + rng.normal(scale=p["noise"], size=X.shape)
    y = _band_labels(theta, int(p["n_bands"]))
    X, y, dedup_meta = _dedup_new_dataset(X.astype(np.float32), y)
    return Dataset(name="anisotropic_ellipsoid", kind="vector", X=X, y=y,
                    meta={"params": p, "description": "Strongly anisotropic ellipsoid (semi-axis ratio 1:5:25) embedded in R^6, y = band of colatitude theta", **dedup_meta})
