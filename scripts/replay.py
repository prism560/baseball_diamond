#!/usr/bin/env python3
"""Replay validation (and timeline cache build) for the games in data/external/feeds/index.json.

Per game (results: data/live/replay_runs/<game_id>.json, resumable):
  clean      synthetic snapshots -> GameServer (synchronous solves); timeline cached in
             TIMELINE_CACHE_DIR (default timelines/)
  offline    regular season: per-point values vs the offline evaluation (wp.live.offline_ref)
  final      last message is final with 100/0 matching the feed's final score
  faults     duplicates, out-of-order, dropped polls, injected corrections, network errors:
             final state, last WP and every point's final value must equal the clean replay
  async      (--async-games) worker-process solves: B2E shown while recomputing, recompute
             times, final timeline equal to the clean replay
  real       games with timecode snapshots: real-snapshot replay vs synthetic (values per point)

    OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/replay.py [--games G ...] [--part 0 --parts 3]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wp.live import feeds  # noqa: E402
from wp.live import log as LG  # noqa: E402
from wp.live.ids import IdMap  # noqa: E402
from wp.live.models import SeasonModels  # noqa: E402
from wp.live.offline_ref import offline_game  # noqa: E402
from wp.live.replay import Faults, TimelineCache, real_snapshots, run, synthetic_snapshots  # noqa: E402
from wp.live.serving import GameServer  # noqa: E402
from wp.live.solver import ProcessSolver, SyncSolver  # noqa: E402
from wp.live.validate_offline import compare, summarize  # noqa: E402

OUT = Path("data/live/replay_runs")
STATE_FIELDS = ("status", "inning", "half", "outs", "bases", "count", "away", "home")


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else None


def lat_stats(x):
    x = np.asarray(x) * 1000
    return {"n": int(len(x)), "median_ms": pct(x, 50), "p95_ms": pct(x, 95), "max_ms": float(x.max()) if len(x) else None}


def tl_compare(a: GameServer, b: GameServer) -> dict:
    """Point-by-point comparison of two servers' final timelines."""
    ta = {tuple(m["key"]): m for m in a.timeline()}
    tb = {tuple(m["key"]): m for m in b.timeline()}
    keys = set(ta) | set(tb)
    missing = [k for k in keys if k not in ta or k not in tb]
    dwp = [abs(ta[k]["wp_home"] - tb[k]["wp_home"]) for k in keys if k in ta and k in tb]
    st = sum(1 for k in keys if k in ta and k in tb and any(ta[k][f] != tb[k][f] for f in STATE_FIELDS))
    la, lb = a.state(), b.state()
    return {"points_a": len(ta), "points_b": len(tb), "missing": len(missing), "state_diff": st,
            "max_abs_wp_diff": float(max(dwp)) if dwp else None,
            "final_equal": bool(la and lb and all(la[f] == lb[f] for f in STATE_FIELDS)
                                and abs(la["wp_home"] - lb["wp_home"]) < 1e-12)}


def one_game(M, ids, solver, pk: str, e: dict, args) -> dict:
    gid = e["retro_id"]
    feed = feeds.load_live(pk)
    snaps = synthetic_snapshots(feed)
    rec = {"game_id": gid, "gamePk": int(pk), "gameType": e["gameType"], "tags": e.get("tags", []),
           "n_snapshots": len(snaps)}
    t = time.time()
    n_solve0 = len(solver.times)
    clean = GameServer(M, ids, solver, game_id=gid)
    r = run(clean, snaps)
    rec["clean_s"] = round(time.time() - t, 1)
    rec["latency_clean"] = lat_stats(r.latencies)
    rec["solve_times"] = [(k, round(dt, 3)) for k, dt, *_ in solver.times[n_solve0:]]
    rec["corrections_clean"] = len(clean.corrections)
    tl = clean.timeline()
    last = clean.state()
    fa, fh = clean.view.state.score
    rec["final"] = {"status": last["status"], "pct_home": last["pct_home"], "away": last["away"], "home": last["home"],
                    "feed_final": list(clean.view.game.final_score or []),
                    "ok": last["status"] == "final" and last["pct_home"] == (100 if fh > fa else 0)
                    and [fa, fh] == list(clean.view.game.final_score or [])}
    rec["n_points"] = len(tl)
    rec["model_counts"] = {m: sum(1 for x in tl if x["model"] == m) for m in ("AVG", "B2E", "B3", "B2")}
    TimelineCache().save(gid, {**clean.meta, "n_points": len(tl), "final": rec["final"],
                               "description": e.get("description", ""), "tags": e.get("tags", [])},
                         tl, clean.corrections)
    # offline match (regular season)
    if e["gameType"] == "R" and gid in M.games.index:
        t = time.time()
        off = offline_game(M, gid)
        df = compare(off, clean.points, clean.values, clean.view.rows)
        rec["offline"] = summarize(df)
        rec["offline_s"] = round(time.time() - t, 1)
        rec["offline_timing"] = {"pregame_s": off["pregame_s"], "n_resolve": off["n_resolve"],
                                 "resolve_s": off["resolve_s"]}
        lu_live = {"vis": [ids.retro(clean.view.game.lineups["away"][s]) for s in range(1, 10)],
                   "home": [ids.retro(clean.view.game.lineups["home"][s]) for s in range(1, 10)]}
        rec["lineup_first_appearance_differs"] = lu_live != off["lineups_first_appearance"]
        d_after = df[df["class"] == "after_runner_event"]
        rec["after_runner_event_diffs"] = d_after["d_AVG"].tolist()
    # faults
    f = Faults(duplicate=0.1, reorder=0.1, drop=0.3, corrections=3, net_errors=0.05, seed=int(pk) % 1000)
    faulted = GameServer(M, ids, solver, game_id=gid)
    rf = run(faulted, snaps, f)
    rec["faults"] = {**tl_compare(clean, faulted), "delivered": rf.n_delivered, "net_errors": rf.n_net_errors,
                     "corrections_logged": len(faulted.corrections), "stale_dropped": faulted.n_stale,
                     "duplicates": faulted.n_dup, "latency": lat_stats(rf.latencies)}
    # real snapshots
    if feeds.snapshot_codes(pk):
        rs = real_snapshots(pk)
        real = GameServer(M, ids, solver, game_id=gid)
        rr = run(real, rs)
        rec["real"] = {**tl_compare(clean, real), "n_snapshots": len(rs), "corrections": real.corrections,
                       "latency": lat_stats(rr.latencies)}
    # async (worker process)
    if gid in args.async_games or args.async_all:
        ps = ProcessSolver(M.eng)
        try:
            asrv = GameServer(M, ids, ps, game_id=gid)
            n_rec = []
            ra = run(asrv, snaps, on_message=lambda m: n_rec.append(m["recomputing"]))
            rec["async"] = {**tl_compare(clean, asrv), "messages": len(n_rec),
                            "recomputing_messages": int(sum(n_rec)),
                            "latency": lat_stats(ra.latencies),
                            "resolve_compute_s": [round(x[1], 3) for x in ps.times],
                            "resolve_wall_s": [round(x[2], 3) for x in ps.times],
                            "resolve_kind": [x[0] for x in ps.times]}
        finally:
            ps.close()
    rec["warnings"] = dict(LG.COUNTS)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="*", default=[], help="Retrosheet-style game ids (default: all in the index)")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--async-games", nargs="*", default=[])
    ap.add_argument("--async-all", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    idx = feeds.index()
    items = sorted(idx.items(), key=lambda kv: kv[1]["retro_id"])
    if args.games:
        items = [kv for kv in items if kv[1]["retro_id"] in args.games]
    items = [kv for i, kv in enumerate(items) if i % args.parts == args.part]
    todo = [kv for kv in items if args.force or not (OUT / f"{kv[1]['retro_id']}.json").exists()]
    print(f"{len(items)} games, {len(todo)} to run", flush=True)
    if not todo:
        return
    t0 = time.time()
    M = SeasonModels.load(2025)
    print(f"models loaded {time.time() - t0:.0f}s", flush=True)
    ids = IdMap()
    solver = SyncSolver(M.eng)
    for pk, e in todo:
        t = time.time()
        LG.COUNTS.clear()
        try:
            rec = one_game(M, ids, solver, pk, e, args)
        except Exception as ex:  # noqa: BLE001
            import traceback
            rec = {"game_id": e["retro_id"], "error": repr(ex), "trace": traceback.format_exc()}
        rec["seconds"] = round(time.time() - t, 1)
        tmp = OUT / f"{e['retro_id']}.json.tmp"
        tmp.write_text(json.dumps(rec, default=str))
        os.replace(tmp, OUT / f"{e['retro_id']}.json")
        print(f"{e['retro_id']}: {rec['seconds']}s {'ERROR ' + rec['error'] if 'error' in rec else ''}", flush=True)
        solver.results.clear()                # bound memory (each full solve ~116 MB)
        solver.cache.clear()


if __name__ == "__main__":
    main()
