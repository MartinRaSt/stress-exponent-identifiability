# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-11
# License: see the LICENSE file in the repository root
"""Tests for `src/experiments/remove_rows.py` on a synthetic CSV in
tmp_path (never on real results/data/*.csv - see CLAUDE.md)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.experiments import remove_rows as rr

EXPERIMENT = "exp1_dr_benchmark"


def _make_csv(tmp_path: Path) -> Path:
    """Creates a small synthetic CSV with a schema matching checkpoint.py
    (experiment, dataset, method, seed, status, error, + one metric) in
    <tmp_path>/results/data/."""
    data_dir = tmp_path / "results" / "data"
    data_dir.mkdir(parents=True)
    csv_path = data_dir / f"{EXPERIMENT}_results.csv"
    df = pd.DataFrame([
        {"experiment": EXPERIMENT, "dataset": "wine", "method": "pca", "seed": 0, "status": "ok", "error": "", "auc_rnx": 0.9},
        {"experiment": EXPERIMENT, "dataset": "wine", "method": "mds", "seed": 0, "status": "ok", "error": "", "auc_rnx": 0.8},
        {"experiment": EXPERIMENT, "dataset": "iris", "method": "pca", "seed": 0, "status": "ok", "error": "", "auc_rnx": 0.7},
        {"experiment": EXPERIMENT, "dataset": "iris", "method": "mds", "seed": 1, "status": "ok", "error": "", "auc_rnx": 0.6},
    ])
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture()
def patched_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirects the `get_path` used in remove_rows.py to tmp_path, so the
    test never touches real results/data/*.csv or embeddings."""

    def _fake_get_path(key: str) -> Path:
        mapping = {
            "results_data_dir": tmp_path / "results" / "data",
            "embeddings_dir": tmp_path / "results" / "data" / "embeddings",
        }
        return mapping[key]

    monkeypatch.setattr(rr, "get_path", _fake_get_path)
    return tmp_path


def test_remove_rows_rejects_missing_filter(patched_paths: Path) -> None:
    """Without --dataset or --method, the CLI must exit with an error
    (fail-loud, no silent deletion of everything)."""
    csv_path = _make_csv(patched_paths)
    with pytest.raises(SystemExit):
        rr.main(["--csv", str(csv_path)])
    # the file remains unchanged
    assert len(pd.read_csv(csv_path)) == 4


def test_remove_rows_dry_run_does_not_modify(patched_paths: Path) -> None:
    """--dry-run only prints what would be deleted, the CSV remains unchanged."""
    csv_path = _make_csv(patched_paths)
    n = rr.remove_rows(csv_path, datasets=["wine"], methods=None, dry_run=True)
    assert n == 2
    assert len(pd.read_csv(csv_path)) == 4  # unchanged
    backups = list(csv_path.parent.glob(f"{csv_path.name}.bak_*"))
    assert backups == []


def test_remove_rows_filters_are_and(patched_paths: Path) -> None:
    """--dataset and --method are combined as AND (not OR)."""
    csv_path = _make_csv(patched_paths)
    n = rr.remove_rows(csv_path, datasets=["wine"], methods=["pca"], dry_run=True)
    assert n == 1  # only (wine, pca), not (wine, mds)


def test_remove_rows_deletes_rows_backup_and_embeddings(patched_paths: Path) -> None:
    """A real run (without --dry-run): the CSV is overwritten, a backup is
    created, and the corresponding .npy/.npz embedding/pertrans cache files
    are deleted."""
    csv_path = _make_csv(patched_paths)
    emb_dir = patched_paths / "results" / "data" / "embeddings" / EXPERIMENT
    emb_dir.mkdir(parents=True)
    (emb_dir / "wine__pca__0.npy").write_bytes(b"fake")
    (emb_dir / "wine__mds__0.npy").write_bytes(b"fake")
    (emb_dir / "iris__pca__0.npy").write_bytes(b"fake")
    (emb_dir / "pertrans").mkdir()
    (emb_dir / "pertrans" / "wine__pca__0.npz").write_bytes(b"fake")

    n = rr.remove_rows(csv_path, datasets=["wine"], methods=None, dry_run=False)
    assert n == 2

    remaining = pd.read_csv(csv_path)
    assert len(remaining) == 2
    assert set(remaining["dataset"]) == {"iris"}

    backups = list(csv_path.parent.glob(f"{csv_path.name}.bak_*"))
    assert len(backups) == 1
    backup_df = pd.read_csv(backups[0])
    assert len(backup_df) == 4  # the backup has the original row count

    assert not (emb_dir / "wine__pca__0.npy").exists()
    assert not (emb_dir / "wine__mds__0.npy").exists()
    assert not (emb_dir / "pertrans" / "wine__pca__0.npz").exists()
    assert (emb_dir / "iris__pca__0.npy").exists()  # untouched (a different dataset)


def test_remove_rows_no_match_is_noop(patched_paths: Path) -> None:
    """A filter that matches nothing changes nothing and returns 0."""
    csv_path = _make_csv(patched_paths)
    n = rr.remove_rows(csv_path, datasets=["nonexistent"], methods=None, dry_run=False)
    assert n == 0
    assert len(pd.read_csv(csv_path)) == 4
    assert list(csv_path.parent.glob(f"{csv_path.name}.bak_*")) == []
