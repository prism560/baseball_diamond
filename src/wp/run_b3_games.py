"""B3 per-game runs (resumable, parallel).

For each evaluation season and ablation, every selected game is solved and
evaluated PA by PA (wp.b3_game.evaluate_game); results are written to
data/processed/b3_runs/<ablation>/<season>/<game_id>.npz so an interrupted run
continues where it stopped.

Game selection: a deterministic hash of the game id (``--fraction``) gives a
reproducible random subset; the same subset is used by every ablation, so
comparisons are paired.

Run:  OMP_NUM_THREADS=1 PYTHONPATH=src python -m wp.run_b3_games --ablation full --fraction 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ABLATIONS = {
    "pregame": {},
    "full": {},
    "no_platoon": {"platoon": False},
    "no_park": {"park": False},
    "team_avg_bullpen": {"bullpen_tiers": False},
    "no_tto": {"tto": False},
}
EVAL = [2023, 2024, 2025]
_G = {}


def selected(game_id: str, fraction: float) -> bool:
    h = int(hashlib.sha256(game_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < fraction


def _init(processed: str, raw: str, params: dict):
    from . import b3_game as BG
    from . import dp_b3 as D3
    from . import ratings as RT
    pa = pd.read_parquet(Path(processed) / "pa.parquet")
    pa = pa[pa["season"] >= 2012]
    games = pd.read_parquet(Path(processed) / "games.parquet")
    hands = RT.load_hands(Path(raw), 2008, 2025)
    d_all = RT.pa_frame(pa[pa["season"] >= 2017], games, hands)
    p = RT.RatingParams(params["kappa_bat"], params["kappa_pit"], params["platoon_k"])
    store = RT.RatingStore(d_all, p)
    _G.update(pa=pa, games=games.set_index("game_id"), games_df=games, hands=hands, d_all=d_all,
              params=p, store=store, park=BG.park_table(d_all), lp=BG.platoon_table(d_all), ctx={},
              eng=D3.Engine(20, np.float32))


def _ctx(S):
    from . import b3_game as BG
    if S not in _G["ctx"]:
        _G["ctx"].clear()          # one season at a time per worker (memory)
        _G["ctx"][S] = BG.build_context(S, _G["pa"], _G["games_df"], _G["d_all"], _G["store"], _G["params"],
                                        _G["hands"], _G["park"], _G["lp"])
    return _G["ctx"][S]


def _work(task):
    try:
        return _work_inner(task)
    except Exception as e:                 # report and continue with the other games
        import traceback
        return task[2], f"error: {e!r} {traceback.format_exc(limit=3)}", 0.0


def _work_inner(task):
    from . import b3_game as BG
    S, ab_name, gid, out = task
    if Path(out).exists():
        return gid, "cached", 0.0
    ctx = _ctx(S)
    rows = ctx.ann[ctx.ann["game_id"] == gid]
    g = _G["games"].loc[gid].copy()
    g["home"] = gid[:3]
    t = time.time()
    if ab_name == "pregame":
        # pre-game solve only (diagnostics): WP at the first PA of the game
        r = BG.evaluate_game(ctx, _G["eng"], g, rows, BG.Ablation(), want_pitch=False, pregame_only=True)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, wp=r["wp"][:1], pregame_s=r["pregame_s"])
        return gid, "done", time.time() - t
    ab = BG.Ablation(**ABLATIONS[ab_name])
    r = BG.evaluate_game(ctx, _G["eng"], g, rows, ab, want_pitch=(ab_name == "full"))
    q = np.array([x if x is not None else np.full(11, np.nan) for x in r["q"]])
    G = np.array([x if x is not None else np.full(11, np.nan) for x in r["G"]])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(out) + ".tmp.npz"
    np.savez_compressed(tmp, wp=r["wp"], q=q, G=G, pregame_s=r["pregame_s"], n_resolve=r["n_resolve"],
                        resolve_s=r["resolve_s"], seconds=r["seconds"])
    os.replace(tmp, out)
    return gid, "done", time.time() - t


def main() -> None:
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--ablation", default="full", choices=list(ABLATIONS))
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--seasons", type=int, nargs="+", default=EVAL)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="max games per season (testing)")
    a = ap.parse_args()
    from .manifest import check_fresh
    for f in ("pa.parquet", "games.parquet"):
        check_fresh(Path(a.processed) / f)
    check_fresh(Path("reports/b3_check.json"))
    params = json.loads(Path("reports/b3_check.json").read_text())["params"]
    games = pd.read_parquet(Path(a.processed) / "games.parquet")
    tasks = []
    for S in a.seasons:
        gids = sorted(games.loc[games["included"] & (games["season"] == S), "game_id"])
        gids = [g for g in gids if selected(g, a.fraction)]
        if a.limit:
            gids = gids[:a.limit]
        out_dir = Path(a.processed) / "b3_runs" / a.ablation / str(S)
        tasks += [(S, a.ablation, g, str(out_dir / f"{g}.npz")) for g in gids]
    todo = [t for t in tasks if not Path(t[3]).exists()]
    print(f"{a.ablation}: {len(tasks)} games selected, {len(todo)} to run", flush=True)
    t0 = time.time()
    done = 0
    # tasks are ordered by season, so each worker builds one season context at a time
    with mp.get_context("fork").Pool(a.workers, initializer=_init, initargs=(a.processed, a.raw, params)) as pool:
        n_err = 0
        for gid, status, sec in pool.imap_unordered(_work, todo, chunksize=4):
            done += 1
            if status.startswith("error"):
                n_err += 1
                print(f"{gid}: {status}", flush=True)
                if n_err >= 10 and n_err > done // 2:
                    raise SystemExit("more than half of the games fail; stopping")
            if done % 50 == 0 or done == len(todo):
                el = time.time() - t0
                print(f"{done}/{len(todo)} games, {el/60:.1f} min, eta {el/done*(len(todo)-done)/60:.1f} min",
                      flush=True)


if __name__ == "__main__":
    main()
