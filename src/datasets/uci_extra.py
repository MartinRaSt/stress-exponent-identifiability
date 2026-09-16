# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-14
# License: see the LICENSE file in the repository root
"""
Q1 step 2 (reserse/2026-09-14_specifikace_rozsireni_q1.md, part A) - hold-out
candidate datasets for regime L (low rho_NN, "separated neighbor"), used
only as `sammon_alpha_pred.holdout_datasets` / `exp1_dr_benchmark.datasets_holdout`
(NEVER in `exp1_dr_benchmark.datasets` - must not influence the rule fit, see A.5).

9 real OpenML datasets (fail-loud on network/spec unavailability - NO
substitute/simulated data) + 2 real reserves requiring a direct zip download
from the UCI archive (dry_bean, ccpp - xlsx/arff inside the archive, outside
fetch_openml). `frey_faces` is NOT implemented - the source
(cs.nyu.edu/~roweis) repeatedly returned HTTP 403 on 2026-09-14 (verified
before implementation, see documentation/2026-09-14_q1_krok2_datasety.md) -
fail-loud against a nonexistent network would just waste time, so the
candidate is recorded in the screening as unavailable without attempting a loader.

Every loader (A.4):
  1. loads the data (OpenML via fetch_openml(as_frame=True) - because of
     categorical columns such as 'Sex' for abalone/'Class' for the others -
     OR a direct zip);
  2. checks that X is finite (no NaN/Inf) - otherwise fail-loud;
  3. removes EXACT duplicate rows of X (`deduplicate_rows`) BEFORE
     standardization and subsampling, writing meta['n_duplicates_removed']
     and meta['dedup']=True;
  4. returns Dataset(kind='vector'). Standardization (config.yaml
     datasets.standardize allow-list) and subsampling (`subsample_dataset`,
     n_max=2000, random_state=42) happen centrally/externally the same way
     as for all other datasets - NOT done here.
"""
from __future__ import annotations

import io
import zipfile
from typing import Any

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder

from src.common.config import ensure_dir, get_path, load_config
from src.datasets.download import download_file
from src.datasets.registry import Dataset, register_dataset
from src.datasets.subsample import deduplicate_rows


def _openml_spec(key: str) -> dict[str, Any]:
    cfg = load_config()
    try:
        return dict(cfg["datasets"]["openml"]["sets"][key])
    except KeyError as exc:
        raise KeyError(f"config.yaml is missing datasets.openml.sets.{key}") from exc


def _fetch_openml_frame(key: str):
    """Download/load an OpenML dataset as a pandas DataFrame (as_frame=True -
    unlike `src/datasets/sklearn_sets.py::_load_openml`, which uses
    as_frame=False and assumes purely numeric features; several new
    candidates have a categorical column OUTSIDE the target - e.g. abalone
    'Sex' - which we need as y, not as a numeric feature)."""
    spec = _openml_spec(key)
    cache_dir = ensure_dir(get_path("openml_cache_dir"))
    try:
        bunch = fetch_openml(data_home=str(cache_dir), as_frame=True, parser="auto", **spec)
    except Exception as exc:  # network/HTTP/parsing errors from fetch_openml are heterogeneous
        raise RuntimeError(
            f"Failed to load/download OpenML dataset '{key}' (spec={spec}): {exc}. "
            "No substitute data is generated - check your connection or the spec in config.yaml."
        ) from exc
    return bunch


def _check_finite(X: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(X)):
        raise ValueError(f"Dataset '{name}': X contains NaN/Inf after loading - a loader/source bug, not fabricating a substitute.")


def _finalize(name: str, X: np.ndarray, y: np.ndarray | None, extra_meta: dict[str, Any]) -> Dataset:
    X = np.asarray(X, dtype=np.float64)
    _check_finite(X, name)
    X_dedup, y_dedup, n_removed = deduplicate_rows(X, y)
    meta = {
        "dedup": True, "n_duplicates_removed": n_removed, "n_before_dedup": int(X.shape[0]),
        **extra_meta,
    }
    return Dataset(name=name, kind="vector", X=X_dedup.astype(np.float32),
                    y=(y_dedup.astype(np.int64) if y_dedup is not None else None), meta=meta)


# ---------------------------------------------------------------------------
# A.3.1 - 9 real OpenML candidates (priority 1-9)
# ---------------------------------------------------------------------------

@register_dataset("shuttle")
def load_shuttle(**_overrides: Any) -> Dataset:
    """Statlog (Shuttle), OpenML data_id 40685, UCI DOI 10.24432/C5WS31.
    9 numeric sensor features (A1..A9), 7 classes (class)."""
    bunch = _fetch_openml_frame("shuttle")
    df = bunch.data
    X = df.to_numpy(dtype=np.float64)
    y = LabelEncoder().fit_transform(np.asarray(bunch.target).astype(str))
    return _finalize("shuttle", X, y, {
        "source": "OpenML data_id=40685 (shuttle)", "citation": "UCI Statlog (Shuttle), DOI 10.24432/C5WS31",
    })


@register_dataset("abalone")
def load_abalone(**_overrides: Any) -> Dataset:
    """Abalone, OpenML data_id 183, UCI DOI 10.24432/C55C7W. The 'Sex'
    column (M/F/I) is DROPPED from X and used as y (spec A.3.1 row 2) - the
    openml target 'Class_number_of_rings' (age) is NOT used HERE (X has 7
    numeric dimensional/weight features)."""
    bunch = _fetch_openml_frame("abalone")
    df = bunch.data
    if "Sex" not in df.columns:
        raise KeyError(f"abalone: expected column 'Sex' in the OpenML data, got columns {list(df.columns)}.")
    y = LabelEncoder().fit_transform(df["Sex"].astype(str).to_numpy())
    X = df.drop(columns=["Sex"]).to_numpy(dtype=np.float64)
    return _finalize("abalone", X, y, {
        "source": "OpenML data_id=183 (abalone)", "citation": "UCI Abalone, DOI 10.24432/C55C7W",
        "y_definition": "Sex (M/F/I), NOT Class_number_of_rings",
    })


@register_dataset("page_blocks")
def load_page_blocks(**_overrides: Any) -> Dataset:
    """Page Blocks Classification, OpenML data_id 30, UCI DOI 10.24432/C5J590.
    10 numeric page-block features, 5 classes (class)."""
    bunch = _fetch_openml_frame("page_blocks")
    df = bunch.data
    X = df.to_numpy(dtype=np.float64)
    y = LabelEncoder().fit_transform(np.asarray(bunch.target).astype(str))
    return _finalize("page_blocks", X, y, {
        "source": "OpenML data_id=30 (page-blocks)", "citation": "UCI Page Blocks Classification, DOI 10.24432/C5J590",
    })


@register_dataset("airfoil")
def load_airfoil(**_overrides: Any) -> Dataset:
    """Airfoil Self-Noise, OpenML data_id 43919, UCI DOI 10.24432/C5VW2C.
    X = all 5 input features (frequency, angle, length, velocity, thickness)
    - the target quantity 'pressure' (SPL) is NOT in X (OpenML target).
    y = index of the unique chord length 'length' (a grid experimental
    design, ~6 levels) for stratification/cluster geometry - spec A.3.1 row 4."""
    bunch = _fetch_openml_frame("airfoil")
    df = bunch.data
    if "length" not in df.columns:
        raise KeyError(f"airfoil: expected column 'length' in the OpenML data, got columns {list(df.columns)}.")
    X = df.to_numpy(dtype=np.float64)
    unique_lengths = np.sort(df["length"].unique())
    length_to_idx = {v: i for i, v in enumerate(unique_lengths)}
    y = df["length"].map(length_to_idx).to_numpy(dtype=np.int64)
    return _finalize("airfoil", X, y, {
        "source": "OpenML data_id=43919 (airfoil_self_noise)", "citation": "UCI Airfoil Self-Noise, DOI 10.24432/C5VW2C",
        "y_definition": f"index of the unique chord length 'length' ({len(unique_lengths)} levels)",
    })


@register_dataset("banknote")
def load_banknote(**_overrides: Any) -> Dataset:
    """Banknote Authentication, OpenML data_id 1462, UCI DOI 10.24432/C55P57.
    4 numeric features (V1..V4, wavelet transform of the image), 2 classes."""
    bunch = _fetch_openml_frame("banknote")
    df = bunch.data
    X = df.to_numpy(dtype=np.float64)
    y = LabelEncoder().fit_transform(np.asarray(bunch.target).astype(str))
    return _finalize("banknote", X, y, {
        "source": "OpenML data_id=1462 (banknote-authentication)", "citation": "UCI Banknote Authentication, DOI 10.24432/C55P57",
    })


@register_dataset("mfeat_morphological")
def load_mfeat_morphological(**_overrides: Any) -> Dataset:
    """Multiple Features (morphological features), OpenML data_id 18, UCI
    DOI 10.24432/C5HC70. 6 numeric features, 10 classes (digits 0-9)."""
    bunch = _fetch_openml_frame("mfeat_morphological")
    df = bunch.data
    X = df.to_numpy(dtype=np.float64)
    y = LabelEncoder().fit_transform(np.asarray(bunch.target).astype(str))
    return _finalize("mfeat_morphological", X, y, {
        "source": "OpenML data_id=18 (mfeat-morphological)", "citation": "UCI Multiple Features, DOI 10.24432/C5HC70",
    })


@register_dataset("wall_robot")
def load_wall_robot(**_overrides: Any) -> Dataset:
    """Wall-Following Robot Navigation (24 sensors), OpenML data_id 1497,
    UCI DOI 10.24432/C57C8W. 24 numeric features (V1..V24, ALL in meters -
    NO standardization, see config.yaml datasets.standardize), 4 classes."""
    bunch = _fetch_openml_frame("wall_robot")
    df = bunch.data
    X = df.to_numpy(dtype=np.float64)
    y = LabelEncoder().fit_transform(np.asarray(bunch.target).astype(str))
    return _finalize("wall_robot", X, y, {
        "source": "OpenML data_id=1497 (wall-robot-navigation, 24 sensors)",
        "citation": "UCI Wall-Following Robot Navigation, DOI 10.24432/C57C8W",
    })


@register_dataset("hill_valley")
def load_hill_valley(**_overrides: Any) -> Dataset:
    """Hill-Valley (without noise), OpenML data_id 1479, UCI DOI 10.24432/C5JC8P.
    100 numeric features (V1..V100, one smooth curve - NO standardization,
    see config.yaml datasets.standardize), 2 classes."""
    bunch = _fetch_openml_frame("hill_valley")
    df = bunch.data
    X = df.to_numpy(dtype=np.float64)
    y = LabelEncoder().fit_transform(np.asarray(bunch.target).astype(str))
    return _finalize("hill_valley", X, y, {
        "source": "OpenML data_id=1479 (hill-valley, without noise)", "citation": "UCI Hill-Valley, DOI 10.24432/C5JC8P",
    })


@register_dataset("phoneme")
def load_phoneme(**_overrides: Any) -> Dataset:
    """Phoneme (ELENA project), OpenML data_id 1489 (no DOI, cite the OpenML URL).
    Preregistered NEGATIVE CONTROL (A.3.1 row 9) - expected rho_NN outside
    regime L. 5 numeric features, 2 classes."""
    bunch = _fetch_openml_frame("phoneme")
    df = bunch.data
    X = df.to_numpy(dtype=np.float64)
    y = LabelEncoder().fit_transform(np.asarray(bunch.target).astype(str))
    return _finalize("phoneme", X, y, {
        "source": "OpenML data_id=1489 (phoneme, ELENA project)",
        "citation": "OpenML https://www.openml.org/d/1489 (no DOI)", "is_control": True,
    })


# ---------------------------------------------------------------------------
# A.3.2 - real reserves (direct zip download from the UCI archive, outside fetch_openml)
# ---------------------------------------------------------------------------

def _download_uci_zip_member(dataset_key: str) -> bytes:
    """Download (or read from local cache) the zip archive
    `datasets.uci_direct.<dataset_key>` and return the content of the
    requested file inside it (`member`) as bytes."""
    cfg = load_config()
    try:
        spec = dict(cfg["datasets"]["uci_direct"][dataset_key])
    except KeyError as exc:
        raise KeyError(f"config.yaml is missing datasets.uci_direct.{dataset_key}") from exc
    cache_dir = ensure_dir(get_path("uci_direct_cache_dir"))
    zip_path = cache_dir / f"{dataset_key}.zip"
    try:
        download_file(spec["archive_url"], zip_path, name=f"uci_direct_{dataset_key}",
                       license_note="UCI Machine Learning Repository, academic use")
    except Exception as exc:
        raise RuntimeError(
            f"'{dataset_key}': downloading archive {spec['archive_url']} failed: {exc}. "
            "No substitute data is generated."
        ) from exc
    with zipfile.ZipFile(zip_path) as zf:
        member = spec["member"]
        if member not in zf.namelist():
            raise RuntimeError(f"Archive {zip_path} does not contain the expected member '{member}' (contents: {zf.namelist()}).")
        return zf.read(member)


@register_dataset("dry_bean")
def load_dry_bean(**_overrides: Any) -> Dataset:
    """Dry Bean, UCI DOI 10.24432/C50S4B (13611 x 16, 7 bean-shape classes) -
    direct zip download (OpenML lookup failed with HTTP 412 on 2026-09-14),
    .arff member of the archive (scipy.io.arff - simpler/more reliable than
    xlsx, no openpyxl needed for this particular dataset)."""
    from scipy.io import arff

    raw = _download_uci_zip_member("dry_bean")
    data, meta = arff.loadarff(io.StringIO(raw.decode("utf-8")))
    field_names = list(meta.names())
    if "Class" not in field_names:
        raise KeyError(f"dry_bean: expected attribute 'Class' in the ARFF, got {field_names}.")
    feature_names = [f for f in field_names if f != "Class"]
    X = np.column_stack([np.asarray(data[f], dtype=np.float64) for f in feature_names])
    y_raw = np.array([v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in data["Class"]])
    y = LabelEncoder().fit_transform(y_raw)
    return _finalize("dry_bean", X, y, {
        "source": "UCI archive zip, DryBeanDataset/Dry_Bean_Dataset.arff",
        "citation": "UCI Dry Bean, DOI 10.24432/C50S4B (Koklu & Ozkan 2020, DOI 10.1016/j.compag.2020.105507)",
    })


@register_dataset("ccpp")
def load_ccpp(**_overrides: Any) -> Dataset:
    """Combined Cycle Power Plant, UCI DOI 10.24432/C5002N (9568 x 4, target
    PE dropped from X) - direct zip download, xlsx member of the archive
    (Sheet1 already contains all 9568 rows - the remaining 4 sheets are
    further random shuffles for 5x2-fold CV, not additional data - see
    UCI Readme.txt in the archive). No classes (y=None, a regression
    dataset with no natural category for stratification)."""
    import pandas as pd

    raw = _download_uci_zip_member("ccpp")
    cfg = load_config()
    sheet_name = cfg["datasets"]["uci_direct"]["ccpp"]["sheet_name"]
    df = pd.read_excel(io.BytesIO(raw), sheet_name=sheet_name, engine="openpyxl")
    expected_cols = {"AT", "V", "AP", "RH", "PE"}
    if not expected_cols.issubset(df.columns):
        raise KeyError(f"ccpp: expected columns {expected_cols}, got {list(df.columns)}.")
    X = df[["AT", "V", "AP", "RH"]].to_numpy(dtype=np.float64)
    return _finalize("ccpp", X, None, {
        "source": f"UCI archive zip, CCPP/Folds5x2_pp.xlsx ({sheet_name})",
        "citation": "UCI Combined Cycle Power Plant, DOI 10.24432/C5002N",
        "y_definition": "None (target PE dropped, no natural class)",
    })
