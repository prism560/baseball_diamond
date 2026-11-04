"""B4 step 7: real-time cost of the serving candidates on real games.

For a sample of 2024 games: pre-game solve, a pitching-change re-solve (current
half only) and pinch-hitter re-solves from the 3rd, 5th and 7th inning, each
  (a) as in the B3 evaluation (all kernels rebuilt),
  (b) with the per-game kernel cache (only the pinch-hit slot's kernels rebuilt),
  (c) (b) plus a narrower diff range for the re-solve (R=15).
Single thread.  Output reports/b4_bench.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import b3_game as BG
from . import dp_b3 as D3
from . import ratings as RT


def main() -> None:
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    root = Path(".")
    pa = pd.read_parquet("data/processed/pa.parquet")
    pa = pa[pa["season"] >= 2012]
    games = pd.read_parquet("data/processed/games.parquet")
    hands = RT.load_hands(Path("data/raw"), 2008, 2025)
    d_all = RT.pa_frame(pa[pa["season"] >= 2017], games, hands)
    cp = json.loads(Path("reports/b3_check.json").read_text())["params"]
    p = RT.RatingParams(cp["kappa_bat"], cp["kappa_pit"], cp["platoon_k"])
    store = RT.RatingStore(d_all, p)
    ctx = BG.build_context(2024, pa, games, d_all, store, p, hands, BG.park_table(d_all), BG.platoon_table(d_all))
    gm = games.set_index("game_id")
    gids = ctx.ann["game_id"].unique()[::240][:10]
    e20, e15 = D3.Engine(20, np.float32), D3.Engine(15, np.float32)
    rec = []
    for gid in gids:
        rows = ctx.ann[ctx.ann["game_id"] == gid]
        g = gm.loc[gid].copy()
        g["home"] = gid[:3]
        lineups = {"home": [None] * 9, "vis": [None] * 9}
        for side, half in (("vis", 0), ("home", 1)):
            rr = rows[rows["half"] == half]
            for s in range(1, 10):
                x = rr.loc[rr["slot"] == s, "batter"]
                lineups[side][s - 1] = str(x.iloc[0]) if len(x) else "unknown"
        starters = {"home": str(rows.loc[rows["half"] == 0, "starter"].iloc[0]),
                    "vis": str(rows.loc[rows["half"] == 1, "starter"].iloc[0])}
        gi, L = BG.game_inputs(ctx, g, lineups, starters, BG.Ablation())
        kc = {}
        t = time.time(); res = e20.solve(gi, kernel_cache=kc); pre = time.time() - t
        relief = ctx.store.pitcher_rates(["x"], [2024], [g["date"]], L[None])[0]
        pdv = np.log(relief / L); pdv -= pdv[RT.REF]
        t = time.time(); e20.solve(gi, override={(6, 0): (pdv, 1)}, upto=(6, 0), cache=res, start=(6, 0),
                                   kernel_cache=kc); pc = time.time() - t
        r = {"game": gid, "pregame_s": pre, "pitching_change_s": pc}
        bench = [x for x in lineups["vis"] if x != "unknown"][0]
        for inn in (3, 5, 7):
            cur = {k: list(v) for k, v in lineups.items()}
            cur["vis"][4] = bench                       # synthetic pinch hitter in slot 5
            gi2, _ = BG.game_inputs(ctx, g, cur, starters, BG.Ablation())
            t = time.time(); e20.solve(gi2, start=(inn, 0)); a_ = time.time() - t
            kc2 = dict(kc)
            D3.KernelBuilder.invalidate(kc2, 0, 4)
            t = time.time(); e20.solve(gi2, start=(inn, 0), kernel_cache=kc2); b_ = time.time() - t
            kc3 = {}
            t = time.time(); e15.solve(gi2, start=(inn, 0), kernel_cache=kc3); c_ = time.time() - t
            r[f"ph_inn{inn}"] = {"full_s": a_, "kernel_cache_s": b_, "R15_s": c_}
        rec.append(r)
        print(gid, round(pre, 2), round(pc, 2), {k: {kk: round(vv, 2) for kk, vv in v.items()}
                                                 for k, v in r.items() if k.startswith("ph")}, flush=True)
    df = pd.json_normalize(rec)
    out = {"games": len(rec), "mean": df.drop(columns=["game"]).mean().round(3).to_dict(),
           "max": df.drop(columns=["game"]).max().round(3).to_dict(), "per_game": rec}
    Path("reports/b4_bench.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out["mean"], indent=1))


if __name__ == "__main__":
    main()
