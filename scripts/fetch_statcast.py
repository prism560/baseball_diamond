#!/usr/bin/env python3
"""Fetch Statcast pitch-level data on a local PC (Baseball Savant is not reachable
from the cloud build environment).

Usage (on your own computer, Python 3.10+, `pip install pandas pyarrow requests`):

    python scripts/fetch_statcast.py --start 2019-03-01 --end 2025-11-30 --out data/external/statcast

Then commit/upload the resulting ``data/external/statcast/statcast_<YYYY>.parquet``
files (or put them in that folder of the cloud session) and run
``make b4-statcast``.

Source: Baseball Savant "Statcast Search" CSV export
(https://baseballsavant.mlb.com/statcast_search, documented columns:
https://baseballsavant.mlb.com/csv-docs).  The export is limited to about
25,000 rows per request, so the date range is fetched in 3-day chunks per
game type "R" (regular season).  Only the columns listed in KEEP are stored.

Real-time availability (the model may only use fields that the live game feed
also provides; see reports/b4_report.md for the table):
  pitch_type, release_speed, pfx_x, pfx_z, plate_x, plate_z, zone, balls,
  strikes, pitch_number, at_bat_number, description, events.
"""
from __future__ import annotations

import argparse
import io
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

URL = "https://baseballsavant.mlb.com/statcast_search/csv"
KEEP = ["game_pk", "game_date", "game_type", "home_team", "away_team", "inning", "inning_topbot",
        "at_bat_number", "pitch_number", "batter", "pitcher", "stand", "p_throws", "balls", "strikes",
        "outs_when_up", "pitch_type", "release_speed", "release_spin_rate", "pfx_x", "pfx_z",
        "plate_x", "plate_z", "zone", "description", "events", "type"]


def fetch_chunk(d0: date, d1: date, session: requests.Session) -> pd.DataFrame:
    params = {"all": "true", "type": "details", "hfGT": "R|", "player_type": "pitcher",
              "game_date_gt": d0.isoformat(), "game_date_lt": d1.isoformat(), "min_pitches": "0",
              "min_results": "0", "group_by": "name", "sort_col": "pitches", "sort_order": "desc"}
    for attempt in range(5):
        try:
            r = session.get(URL, params=params, timeout=120)
            r.raise_for_status()
            if not r.text.strip():
                return pd.DataFrame(columns=KEEP)
            df = pd.read_csv(io.StringIO(r.text), low_memory=False)
            return df[[c for c in KEEP if c in df.columns]]
        except Exception as e:  # network hiccups: back off and retry
            print(f"  retry {attempt + 1} for {d0}..{d1}: {e}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"failed {d0}..{d1}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-03-01")
    ap.add_argument("--end", default="2025-11-30")
    ap.add_argument("--out", type=Path, default=Path("data/external/statcast"))
    ap.add_argument("--chunk-days", type=int, default=3)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers["User-Agent"] = "baseballdiamond-research"
    d = date.fromisoformat(a.start)
    end = date.fromisoformat(a.end)
    by_year: dict[int, list] = {}
    while d <= end:
        d1 = min(d + timedelta(days=a.chunk_days - 1), end)
        if 3 <= d.month <= 11:
            df = fetch_chunk(d, d1, s)
            print(f"{d}..{d1}: {len(df):,} pitches")
            if len(df) >= 24990:
                print("  WARNING: chunk may be truncated by the export limit; use --chunk-days 1")
            by_year.setdefault(d.year, []).append(df)
            time.sleep(1.0)
        d = d1 + timedelta(days=1)
    for y, parts in by_year.items():
        df = pd.concat(parts, ignore_index=True).drop_duplicates(["game_pk", "at_bat_number", "pitch_number"])
        p = a.out / f"statcast_{y}.parquet"
        df.to_parquet(p, index=False)
        print(f"wrote {p} ({len(df):,} pitches, {df['game_pk'].nunique():,} games)")


if __name__ == "__main__":
    main()
