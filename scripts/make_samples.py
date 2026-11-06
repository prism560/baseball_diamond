#!/usr/bin/env python3
"""Sample timelines for the replay-screen mockup (roadmap step 2): samples/timeline_<game_id>.json.

Uncompressed JSON in the same shape as GET /games/{id}/timeline ({"meta", "timeline"}); every
timeline element is a message of docs/API.md section 2.  Three representative games:
  close_comeback   close game won by a team that trailed by >= 3 runs (final margin <= 2)
  extra_walkoff    extra-inning walk-off (World Series 2025 Game 3, 18 innings)
  blowout          one-sided game (final margin >= 9)
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

SAMPLES = {
    "close_comeback": "ATL202508092",
    "extra_walkoff": None,          # resolved from gamePk 813032 (2025 World Series Game 3)
    "blowout": "CIN202506070",
}


def main():
    idx = json.loads(Path("data/external/feeds/index.json").read_text())
    SAMPLES["extra_walkoff"] = idx["813032"]["retro_id"]
    out = Path("samples")
    out.mkdir(exist_ok=True)
    for kind, gid in SAMPLES.items():
        d = json.load(gzip.open(Path("timelines") / f"{gid}.json.gz", "rt", encoding="utf-8"))
        meta = {**d["meta"], "sample_kind": kind}
        p = out / f"timeline_{gid}.json"
        p.write_text(json.dumps({"meta": meta, "timeline": d["timeline"]}, ensure_ascii=False, separators=(",", ":")))
        size = p.stat().st_size
        print(f"{kind}: {p} {len(d['timeline'])} messages, {size / 1e6:.2f} MB")
        if size > 1_000_000:
            sys.exit(f"{p} is larger than 1 MB")


if __name__ == "__main__":
    main()
