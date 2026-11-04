"""Local feed store (data/external/feeds/, written by scripts/fetch_feeds.py).

The location can be changed with the FEEDS_DIR environment variable.
"""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

FEEDS = Path(os.environ.get("FEEDS_DIR", "data/external/feeds"))


def index(root: Path = FEEDS) -> dict:
    p = Path(root) / "index.json"
    return json.loads(p.read_text()) if p.exists() else {}


def load_live(pk, root: Path = FEEDS) -> dict:
    return json.loads(gzip.decompress((Path(root) / str(pk) / "live.json.gz").read_bytes()))


def timestamps(pk, root: Path = FEEDS) -> list[str]:
    p = Path(root) / str(pk) / "timestamps.json"
    return json.loads(p.read_text()) if p.exists() else []


def snapshot_codes(pk, root: Path = FEEDS) -> list[str]:
    d = Path(root) / str(pk) / "snapshots"
    return sorted(p.name.split(".")[0] for p in d.glob("*.json.gz")) if d.exists() else []


def load_snapshot(pk, code: str, root: Path = FEEDS) -> dict:
    return json.loads(gzip.decompress((Path(root) / str(pk) / "snapshots" / f"{code}.json.gz").read_bytes()))
