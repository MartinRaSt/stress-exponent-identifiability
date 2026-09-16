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
Downloading external files with hash verification and writing to the
manifest src/data/manifest.json (name, url, sha256, download date,
license/source). Fail loud: a network/HTTP error is never replaced with
fabricated data.
"""
from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.common.config import ensure_dir, get_path, load_config


def _sha256(path: Path) -> str:
    """Compute the file's SHA-256 hash chunk by chunk (without loading the whole file into memory)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_manifest() -> dict[str, Any]:
    manifest_path = get_path("manifest_file")
    if not manifest_path.exists():
        return {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_manifest(manifest: dict[str, Any]) -> None:
    manifest_path = get_path("manifest_file")
    ensure_dir(manifest_path.parent)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, sort_keys=True)


def download_file(url: str, dest_path: Path, name: str, license_note: str = "") -> Path:
    """Download the file from `url` to `dest_path` if not already there, and write the manifest.

    If the file already exists locally, the download is skipped (idempotence
    on repeated pipeline runs). A download error propagates (fail loud).
    """
    dest_path = Path(dest_path)
    if dest_path.exists():
        return dest_path

    ensure_dir(dest_path.parent)
    cfg = load_config()
    dl_cfg = cfg.get("download", {})
    headers = {"User-Agent": dl_cfg.get("user_agent", "Mozilla/5.0")}
    timeout = dl_cfg.get("timeout_sec", 20)
    retries = dl_cfg.get("retries", 3)

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                data = response.read()
            tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
            with open(tmp_path, "wb") as f:
                f.write(data)
            tmp_path.replace(dest_path)
            last_exc = None
            break
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last_exc = exc
    if last_exc is not None:
        raise RuntimeError(
            f"Downloading file '{name}' from {url} failed after {retries} attempts: {last_exc}. "
            "Data is never fabricated - download the file manually and place it at the "
            f"expected path, or check your connection. Target path: {dest_path}"
        ) from last_exc

    manifest = _read_manifest()
    manifest[name] = {
        "url": url,
        "local_path": str(dest_path),
        "sha256": _sha256(dest_path),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "license_note": license_note,
    }
    _write_manifest(manifest)
    return dest_path
