"""Replay harness: past games as time-ordered feed snapshots through the real-time path.

Snapshot sources
  synthetic  built from the final feed: for every play event, the feed truncated to that
             event (completed plays + the current play's events so far, runner movements up
             to that event, result hidden until the play is complete).  The state is rebuilt
             only from liveData.plays, so a prefix is a faithful snapshot except for later
             corrections (which only real snapshots contain).
  real       the feed as of each timecode (data/external/feeds/<pk>/snapshots/, fetched with
             scripts/fetch_feeds.py --snapshots).
Faults (``Faults``): duplicate snapshots, swapped neighbours (out of order), dropped
snapshots (several pitches between polls), an injected correction (a pitch call changed
in one snapshot and restored in the next), network errors (snapshot fetch raises).
Pacing: ``speed`` 1 = real time from the event timestamps, 5 / 20 = faster, 0 = max.

The same harness is the engine of the website's replay (``wp.live.api``): a finished
replay's timeline is cached (``TimelineCache``) and served without recomputation.
"""
from __future__ import annotations

import copy
import gzip
import json
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from . import feeds
from .log import warn

TIMELINE_DIR = Path(os.environ.get("TIMELINE_CACHE_DIR", "timelines"))


def _tc(ts: str | None) -> str | None:
    """ISO time -> feed timecode YYYYMMDD_HHMMSS (sortable like metaData.timeStamp)."""
    if not ts:
        return None
    d = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return d.strftime("%Y%m%d_%H%M%S") + f"{d.microsecond // 1000:03d}"


def _secs(ts: str | None) -> float | None:
    if not ts:
        return None
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()


def synthetic_snapshots(feed: dict) -> list[dict]:
    """Prefix snapshots of a final feed (shallow copies sharing the event dicts)."""
    ld = feed["liveData"]
    plays = ld["plays"]["allPlays"]
    gd = dict(feed["gameData"])
    live_status = {"abstractGameState": "Live", "codedGameState": "I", "detailedState": "In Progress",
                   "statusCode": "I", "abstractGameCode": "L"}
    out = []

    def snap(done: list, cur: dict | None, ts: str | None, final: bool = False):
        ap = done + ([cur] if cur is not None else [])
        f = {"gamePk": feed["gamePk"], "metaData": {"timeStamp": _tc(ts), "wait": 10},
             "gameData": {**gd, "status": feed["gameData"]["status"] if final else live_status},
             "liveData": {"plays": {"allPlays": ap}, "boxscore": ld["boxscore"],
                          "linescore": ld["linescore"] if final else {}}}
        out.append(f)

    # pre-game snapshot: lineups known, no plays
    pre = dict(feed["gameData"])
    first_ts = plays[0]["about"].get("startTime") if plays else None
    out.append({"gamePk": feed["gamePk"], "metaData": {"timeStamp": _tc(first_ts)[:-3] + "000" if first_ts else None},
                "gameData": {**pre, "status": {"abstractGameState": "Preview", "codedGameState": "P",
                                               "detailedState": "Pre-Game", "statusCode": "P"}},
                "liveData": {"plays": {"allPlays": []}, "boxscore": ld["boxscore"], "linescore": {}}})
    for k, p in enumerate(plays):
        evs = p.get("playEvents", [])
        for j in range(len(evs)):
            idx = evs[j]["index"]
            cur = {**p, "playEvents": evs[:j + 1],
                   "runners": [r for r in p.get("runners", []) if (r["details"].get("playIndex") or 0) <= idx
                               and not _is_result_run(p, r, evs)],
                   "about": {**p["about"], "isComplete": False}, "result": {"type": "atBat"}}
            snap(plays[:k], cur, evs[j].get("startTime"))
        snap(plays[:k + 1], None, p["about"].get("endTime") or p["about"].get("startTime"),
             final=(k == len(plays) - 1))
    return out


def _is_result_run(p: dict, r: dict, evs: list) -> bool:
    """Result movements (on the last event) must not show before the play is complete."""
    if not evs:
        return False
    return (r["details"].get("playIndex") == evs[-1]["index"]) and p["about"].get("isComplete") and \
        len(evs) == len(p.get("playEvents", []))


def real_snapshots(pk) -> list[dict]:
    return [feeds.load_snapshot(pk, c) for c in feeds.snapshot_codes(pk)]


@dataclass
class Faults:
    duplicate: float = 0.0       # probability a snapshot is delivered twice
    reorder: float = 0.0         # probability two neighbours are swapped
    drop: float = 0.0            # probability a snapshot is skipped (missed poll)
    corrections: int = 0         # number of injected (then restored) call changes
    net_errors: float = 0.0      # probability a fetch raises
    seed: int = 0

    def any(self) -> bool:
        return any((self.duplicate, self.reorder, self.drop, self.corrections, self.net_errors))


class NetError(Exception):
    pass


def apply_faults(snaps: list[dict], f: Faults) -> list:
    """Snapshot delivery sequence with faults; items are feeds or NetError instances.
    The last snapshot (final state) is always delivered last."""
    rng = random.Random(f.seed)
    body, last = snaps[:-1], snaps[-1]
    seq: list = []
    inject = set(rng.sample(range(1, len(body)), min(f.corrections, max(len(body) - 1, 0)))) if f.corrections else set()
    for i, s in enumerate(body):
        if f.drop and rng.random() < f.drop:
            continue
        if i in inject:
            bad = _corrupt(s)
            if bad is not None:
                seq.append(bad)
        if f.net_errors and rng.random() < f.net_errors:
            seq.append(NetError(f"injected network error before snapshot {i}"))
        seq.append(s)
        if f.duplicate and rng.random() < f.duplicate:
            seq.append(s)
    if f.reorder:
        for i in range(len(seq) - 1):
            if rng.random() < f.reorder:
                seq[i], seq[i + 1] = seq[i + 1], seq[i]
    seq.append(last)
    return seq


def _corrupt(s: dict) -> dict | None:
    """Copy of ``s`` whose latest pitch call is changed (ball <-> called strike), with the same
    timecode - the next real snapshot restores it (a correction for the session layer)."""
    ap = s["liveData"]["plays"]["allPlays"]
    if not ap:
        return None
    p = ap[-1]
    evs = p.get("playEvents", [])
    for j in range(len(evs) - 1, -1, -1):
        e = evs[j]
        code = (e.get("details", {}).get("call") or {}).get("code")
        if e.get("isPitch") and code in ("B", "C"):
            e2 = copy.deepcopy(e)
            new = "C" if code == "B" else "B"
            e2["details"]["call"]["code"] = new
            e2["details"]["code"] = new
            p2 = {**p, "playEvents": evs[:j] + [e2] + evs[j + 1:]}
            return {**s, "liveData": {**s["liveData"], "plays": {"allPlays": ap[:-1] + [p2]}}}
    return None


@dataclass
class ReplayResult:
    server: object
    n_snapshots: int
    n_delivered: int
    n_net_errors: int
    latencies: list = field(default_factory=list)      # seconds per delivered snapshot
    seconds: float = 0.0


def run(server, snaps: list, faults: Faults | None = None, speed: float = 0.0, on_message=None,
        wait_solver: bool = True) -> ReplayResult:
    """Push snapshots through ``server.update`` (blocking).  speed 0 = as fast as possible."""
    seq = apply_faults(snaps, faults) if faults and faults.any() else list(snaps)
    t0 = time.time()
    n_err = 0
    prev_t = None
    lat = []
    for item in seq:
        if isinstance(item, NetError):
            n_err += 1
            warn("feed_fetch_error", game_id=server.game_id, err=str(item), injected=True)
            continue
        tc = (item.get("metaData") or {}).get("timeStamp")
        if speed and tc and prev_t:
            dt = (_tc_secs(tc) - prev_t) / speed
            if 0 < dt < 120:
                time.sleep(dt)
        if tc:
            prev_t = _tc_secs(tc)
        t = time.perf_counter()
        msgs = server.update(item)
        msgs += server.poll()
        lat.append(time.perf_counter() - t)
        if on_message:
            for m in msgs:
                on_message(m)
    if wait_solver and hasattr(server.solver, "wait"):
        while server.solver.futs:
            server.solver.wait(5)
            msgs = server.poll()
            if on_message:
                for m in msgs:
                    on_message(m)
    return ReplayResult(server, len(snaps), len(seq) - n_err, n_err, lat, time.time() - t0)


def _tc_secs(tc: str) -> float:
    base = datetime.strptime(tc[:15], "%Y%m%d_%H%M%S").timestamp()
    return base + (int(tc[15:]) / 1000 if len(tc) > 15 else 0)


class TimelineCache:
    """Finished timelines, one gzip JSON per game: <TIMELINE_CACHE_DIR>/<game_id>.json.gz"""

    def __init__(self, root: Path | str | None = None):
        self.root = Path(root or os.environ.get("TIMELINE_CACHE_DIR", "timelines"))

    def path(self, game_id: str) -> Path:
        return self.root / f"{game_id}.json.gz"

    def has(self, game_id: str) -> bool:
        return self.path(game_id).exists()

    def save(self, game_id: str, meta: dict, timeline: list, corrections: list | None = None) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        p = self.path(game_id)
        data = {"meta": meta, "timeline": timeline, "corrections": corrections or []}
        tmp = p.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            json.dump(data, f, separators=(",", ":"), default=str)
        tmp.replace(p)
        return p

    def load(self, game_id: str) -> dict:
        with gzip.open(self.path(game_id), "rt", encoding="utf-8") as f:
            return json.load(f)

    def list(self) -> list[dict]:
        out = []
        for p in sorted(self.root.glob("*.json.gz")):
            try:
                with gzip.open(p, "rt", encoding="utf-8") as f:
                    out.append(json.load(f)["meta"])
            except Exception:  # noqa: BLE001
                continue
        return out
