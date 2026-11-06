#!/usr/bin/env python3
"""Collect MLB StatsAPI game feeds for the replay harness (standard library only).

Runs here (if statsapi.mlb.com is reachable) or on any PC with Python 3.9+:

    python scripts/fetch_feeds.py                       # default game list (scripts/feed_games_2025.json)
    python scripts/fetch_feeds.py --pk 813071 813064    # specific gamePks
    python scripts/fetch_feeds.py --postseason 2025     # every postseason game of a season
    python scripts/fetch_feeds.py --snapshots 813071    # also every timecode snapshot of these games

Output (data/external/feeds/ by default, resumable - existing files are skipped):
    schedule/<season>_<types>.json.gz      season schedule (gamePk <-> date/teams/gameNumber)
    <gamePk>/live.json.gz                  final live feed
    <gamePk>/timestamps.json               list of feed timecodes
    <gamePk>/snapshots/<timecode>.json.gz  feed as of a timecode (only with --snapshots)
    index.json                             gamePk -> {retro_id, date, away, home, gameType, ...}

Endpoints (confirmed against real responses on 2026-11-04, see docs/API.md "데이터 소스"):
    GET https://statsapi.mlb.com/api/v1/schedule?sportId=1&season=YYYY&gameType=R,F,D,L,W
    GET https://statsapi.mlb.com/api/v1.1/game/{gamePk}/feed/live
    GET https://statsapi.mlb.com/api/v1.1/game/{gamePk}/feed/live/timestamps    -> ["YYYYMMDD_HHMMSS", ...]
    GET https://statsapi.mlb.com/api/v1.1/game/{gamePk}/feed/live?timecode=YYYYMMDD_HHMMSS
    (api/v1/game/{pk}/feed/live/diffPatch returned 404 and is not used.)

Politeness: one request at a time, --delay seconds (default 1.0) between requests,
exponential backoff on errors.  Data (c) MLB Advanced Media - personal, non-commercial use.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://statsapi.mlb.com/api"
UA = "baseball_diamond-wp-replay/1.0 (personal, non-commercial)"
# Retrosheet team code -> MLB team id (2025; checked against /api/v1/teams?sportId=1&season=2025)
RETRO_TEAM = {"ANA": 108, "ARI": 109, "BAL": 110, "BOS": 111, "CHN": 112, "CIN": 113, "CLE": 114,
              "COL": 115, "DET": 116, "HOU": 117, "KCA": 118, "LAN": 119, "WAS": 120, "NYN": 121,
              "ATH": 133, "OAK": 133, "PIT": 134, "SDN": 135, "SEA": 136, "SFN": 137, "SLN": 138,
              "TBA": 139, "TEX": 140, "TOR": 141, "MIN": 142, "PHI": 143, "ATL": 144, "CHA": 145,
              "MIA": 146, "NYA": 147, "MIL": 158}
TEAM_RETRO = {v: k for k, v in RETRO_TEAM.items() if k != "OAK"}


class Client:
    def __init__(self, delay: float = 1.0, retries: int = 5):
        self.delay, self.retries, self._last = delay, retries, 0.0
        self.n_req = 0
        self.bytes = 0

    def get(self, url: str) -> bytes:
        wait = 2.0
        for attempt in range(self.retries + 1):
            dt = time.time() - self._last
            if dt < self.delay:
                time.sleep(self.delay - dt)
            self._last = time.time()
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "gzip"})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
                    if r.headers.get("Content-Encoding") == "gzip":
                        data = gzip.decompress(data)
                self.n_req += 1
                self.bytes += len(data)
                return data
            except urllib.error.HTTPError as e:
                if e.code in (400, 404):
                    raise
                err = e
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                err = e
            if attempt == self.retries:
                raise RuntimeError(f"giving up on {url}: {err}")
            print(f"  retry in {wait:.0f}s ({err})", file=sys.stderr)
            time.sleep(wait)
            wait *= 2
        raise AssertionError

    def json(self, url: str):
        return json.loads(self.get(url))


def write_gz(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wb", compresslevel=6) as f:
        f.write(data)
    tmp.replace(path)


def schedule(c: Client, out: Path, season: int, types: str = "R,F,D,L,W") -> list[dict]:
    p = out / "schedule" / f"{season}_{types.replace(',', '')}.json.gz"
    if p.exists():
        d = json.loads(gzip.decompress(p.read_bytes()))
    else:
        raw = c.get(f"{API}/v1/schedule?sportId=1&season={season}&gameType={types}")
        write_gz(p, raw)
        d = json.loads(raw)
    games = []
    for dt in d.get("dates", []):
        for g in dt["games"]:
            games.append(g)
    return games


def retro_id(g: dict) -> str | None:
    """Retrosheet-style id HHHYYYYMMDDN for a schedule entry (N = 0 single game, 1/2 doubleheader)."""
    home = TEAM_RETRO.get(g["teams"]["home"]["team"]["id"])
    if home is None:
        return None
    n = 0 if g.get("doubleHeader", "N") == "N" else int(g.get("gameNumber", 1))
    return f"{home}{g['officialDate'].replace('-', '')}{n}"


def entry(g: dict) -> dict:
    return {"gamePk": g["gamePk"], "retro_id": retro_id(g), "date": g["officialDate"], "gameType": g["gameType"],
            "away": g["teams"]["away"]["team"]["id"], "home": g["teams"]["home"]["team"]["id"],
            "status": g["status"]["detailedState"], "description": g.get("description", ""),
            "doubleHeader": g.get("doubleHeader"), "gameNumber": g.get("gameNumber")}


def fetch_game(c: Client, out: Path, pk: int, snapshots: bool = False) -> None:
    d = out / str(pk)
    live = d / "live.json.gz"
    if not live.exists():
        write_gz(live, c.get(f"{API}/v1.1/game/{pk}/feed/live"))
    ts = d / "timestamps.json"
    if not ts.exists():
        ts.write_bytes(c.get(f"{API}/v1.1/game/{pk}/feed/live/timestamps"))
    if snapshots:
        codes = json.loads(ts.read_text())
        todo = [t for t in codes if not (d / "snapshots" / f"{t}.json.gz").exists()]
        print(f"  {pk}: {len(codes)} timecodes, {len(todo)} to fetch", flush=True)
        for i, t in enumerate(todo):
            write_gz(d / "snapshots" / f"{t}.json.gz", c.get(f"{API}/v1.1/game/{pk}/feed/live?timecode={t}"))
            if (i + 1) % 50 == 0:
                print(f"    {i + 1}/{len(todo)}", flush=True)


def save_index(p: Path, index: dict) -> None:
    """Merge with the file on disk (another fetch may be running) and write atomically."""
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur.update(index)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(cur, indent=1))
    tmp.replace(p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/external/feeds"))
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--games", type=Path, default=Path(__file__).with_name("feed_games_2025.json"),
                    help="JSON list of {retro_id|gamePk, ...} (default list for replay validation)")
    ap.add_argument("--pk", type=int, nargs="*", default=[], help="extra gamePks")
    ap.add_argument("--retro-id", nargs="*", default=[], help="extra Retrosheet game ids (HHHYYYYMMDDN)")
    ap.add_argument("--postseason", action="store_true", help="also every postseason game of --season")
    ap.add_argument("--snapshots", type=int, nargs="*", default=None,
                    help="gamePks whose timecode snapshots are fetched too (no value: those flagged in --games)")
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--no-default", action="store_true", help="ignore --games")
    a = ap.parse_args()
    c = Client(a.delay)
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sched = schedule(c, out, a.season)
    by_retro = {retro_id(g): g for g in sched if g["gameType"] == "R"}
    by_pk = {g["gamePk"]: g for g in sched}
    want: dict[int, dict] = {}
    snap: set[int] = set(a.snapshots or [])
    listed = [] if a.no_default or not a.games.exists() else json.loads(a.games.read_text())
    for item in listed + [{"retro_id": r} for r in a.retro_id] + [{"gamePk": p} for p in a.pk]:
        g = by_pk.get(item.get("gamePk")) if item.get("gamePk") else by_retro.get(item.get("retro_id"))
        if g is None:
            print(f"not in schedule: {item}", file=sys.stderr)
            continue
        want[g["gamePk"]] = {**entry(g), "tags": item.get("tags", [])}
        if a.snapshots is not None and len(a.snapshots) == 0 and item.get("snapshots"):
            snap.add(g["gamePk"])
    if a.postseason:
        for g in sched:
            if g["gameType"] in ("F", "D", "L", "W") and g["status"]["abstractGameState"] == "Final":
                want.setdefault(g["gamePk"], {**entry(g), "tags": ["postseason"]})
    idx_p = out / "index.json"
    index = json.loads(idx_p.read_text()) if idx_p.exists() else {}
    print(f"{len(want)} games, snapshots for {sorted(snap)}", flush=True)
    for i, (pk, e) in enumerate(sorted(want.items())):
        try:
            fetch_game(c, out, pk, snapshots=pk in snap)
        except Exception as ex:          # keep going; rerun to resume
            print(f"{pk}: failed ({ex})", file=sys.stderr)
            continue
        index[str(pk)] = e
        if (i + 1) % 10 == 0:
            save_index(idx_p, index)
            print(f"{i + 1}/{len(want)} games, {c.n_req} requests, {c.bytes / 1e6:.0f} MB, "
                  f"{time.time() - t0:.0f}s", flush=True)
    save_index(idx_p, index)
    print(f"done: {c.n_req} requests, {c.bytes / 1e6:.0f} MB (uncompressed), {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
