"""Offline reference values for a regular-season game (target of the replay match test).

Same computation as the offline evaluation, on the Retrosheet PA rows:
  B3   wp.b3_game.evaluate_game (per PA start) and the count chain on its matchup q /
       after-outcome values G (per pitch, pitches before any in-PA runner event)   [run_b3_eval]
  B2   league DP value at the PA start / league pitch table at (outs1, bases1, d1, count)
  B2E  b2e.predict(B2, P_Elo, progress)                                          [run_b3_eval]
  AVG  sigma(0.55 logit B3 + 0.45 logit B2E)                                       [run_b4_combine]
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import b3_game as BG
from .. import count_chain as CC
from .. import states as ST
from .models import SeasonModels, avg


def offline_game(M: SeasonModels, gid: str) -> dict:
    ctx = M.ctx
    rows = ctx.ann[ctx.ann["game_id"] == gid]
    g = M.games.loc[gid].copy()
    g["home"] = gid[:3]
    r = BG.evaluate_game(ctx, M.eng, g, rows, BG.Ablation(), want_pitch=True)
    b2 = M.b2[True]
    p_pre = M.p_pregame(gid, gid[:3], g["visitor"], M.season)
    rr = rows.reset_index(drop=True)
    inn, half = rr["inning"].to_numpy(), rr["half"].to_numpy()
    B2 = np.array([b2.pa_value(a, h, o, b, d) for a, h, o, b, d in
                   zip(inn, half, rr["outs0"], rr["bases0"], rr["diff0"])])
    prog = ST.progress(inn, half, rr["outs0"].to_numpy())
    B2E = M.b2e.predict(B2, np.full(len(rr), p_pre), prog)
    pa = pd.DataFrame({"i": np.arange(len(rr)), "B3": r["wp"], "B2": B2, "B2E": B2E,
                       "AVG": avg(r["wp"], B2E, M.w)})
    recs = []
    for i, row in enumerate(rr.itertuples(index=False)):
        q, G = r["q"][i], r["G"][i]
        if q is None:
            continue
        P = CC.fit(M.base, q[None]).P[0]
        cnts = CC.counts_before_each_pitch(row.pitches)
        first_mid = min((int(x.split(":")[0]) for x in row.mid.split("|")), default=10 ** 6) if row.mid else 10 ** 6
        sgn = 1 if row.half == 1 else -1
        d1 = row.diff0 + sgn * row.rN
        pr = ST.progress(np.array([row.inning]), np.array([row.half]), np.array([row.outs1]))[0]
        for j, c in enumerate(cnts):
            b3 = float(P[c] @ G)
            v2 = b2.pitch_value(row.inning, row.half, row.outs1, row.bases1, d1, c)
            v2e = float(M.b2e.predict(np.array([v2]), np.array([p_pre]), np.array([pr]))[0])
            recs.append((i, j, c, j >= first_mid, b3, v2, v2e, float(avg(b3, v2e, M.w))))
    pt = pd.DataFrame(recs, columns=["i", "j", "count", "after_mid", "B3", "B2", "B2E", "AVG"])
    lineups = {"vis": [None] * 9, "home": [None] * 9}
    for side, h in (("vis", 0), ("home", 1)):
        x = rr[rr["half"] == h]
        for s in range(1, 10):
            b = x.loc[x["slot"] == s, "batter"]
            lineups[side][s - 1] = str(b.iloc[0]) if len(b) else "unknown"
    return {"pa": pa, "pitch": pt, "rows": rr, "lineups_first_appearance": lineups, "p_pregame": p_pre,
            "seconds": r["seconds"], "pregame_s": r["pregame_s"], "n_resolve": r["n_resolve"],
            "resolve_s": r["resolve_s"]}
