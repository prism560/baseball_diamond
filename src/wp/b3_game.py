"""Assemble B3 inputs for a game (as of the game date) and evaluate it PA by PA.

Season context (``SeasonContext``): ratings store, park / platoon tables,
league transitions of the fixed window (K=10 seasons before S), hazard, TTO
offsets, bullpen tiers and tier mixture - all fitted on data before S, or
(ratings, tiers, league environment) as of the day before each game.

Per game (``evaluate_game``):
  * pre-game solve with the starting lineups and starting pitchers;
  * walk through the PAs in order; whenever the fielding team's pitcher is a
    reliever, the current half is re-solved with that pitcher (his own rating
    until the end of the half, mixture afterwards); when a new batter appears
    in a lineup slot (pinch hitter), the lineup is updated and the remaining
    halves are re-solved;
  * B3 WP at every PA start, and the quantities needed for pitch-level WP
    (matchup q at the first pitch and the after-outcome values G).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from . import dp_b3 as D3
from . import matchup as MU
from . import pitching_policy as PP
from . import ratings as RT
from . import transitions as T
from .transitions import NB, NO, RMAX

HAND = {"L": 0, "R": 1}


@dataclass
class Ablation:
    platoon: bool = True
    park: bool = True
    bullpen_tiers: bool = True     # False -> one team-average bullpen tier
    tto: bool = True               # False -> no lap offsets (lap still drives the hazard)
    league_only: bool = False      # True -> every player league-average (reduction to B2)


@dataclass
class SeasonContext:
    season: int
    store: RT.RatingStore
    params: RT.RatingParams
    hands: pd.DataFrame
    park: dict                      # (site, home, season) -> [NR]
    lp: dict                        # (season, bside, phand) -> [NR] league platoon interaction
    tr: T.Transitions
    L_win: np.ndarray
    hazard: PP.Hazard
    tto: np.ndarray
    tiers: pd.DataFrame
    tier_w: np.ndarray
    tier_w_flat: np.ndarray
    ann: pd.DataFrame               # annotated PA table of season S (game order)
    timings: dict = field(default_factory=dict)
    cache: dict = field(default_factory=dict)  # (kind, id, date, ablation flags) -> deviations


def window(seasons, before, K=10):
    return [s for s in seasons if s < before and s != 2020][-K:]


def build_context(S: int, pa: pd.DataFrame, games: pd.DataFrame, d_all: pd.DataFrame,
                  store: RT.RatingStore, params: RT.RatingParams, hands: pd.DataFrame,
                  park_tab: dict, lp_tab: dict) -> SeasonContext:
    tm = {}
    t = time.time()
    seasons = sorted(int(s) for s in pa["season"].unique())
    w = window(seasons, S)
    inc = set(games.loc[games["included"], "game_id"])
    tr = T.estimate(pa[pa["game_id"].isin(inc)], w)
    dw = d_all[d_all["season"].isin(w) & (d_all["r"] >= 0)]
    cnt = np.bincount(dw["r"], minlength=RT.NR) + 1.0
    L_win = cnt / cnt.sum()
    pol_seasons = window(seasons, S, 3)
    ann = PP.annotate(pa[pa["season"].isin(pol_seasons + [S])], games)
    r = np.full(len(ann), -1)
    for o, i in RT.R_IDX.items():
        r[ann["o"].to_numpy() == T.O_IDX[o]] = i
    ann["r"] = r
    hz = PP.Hazard.fit(ann, pol_seasons)
    tto = PP.tto_offsets(ann, pol_seasons)
    tiers = PP.bullpen_tiers(ann, pol_seasons + [S])
    tier_w = PP.tier_mixture(ann, tiers, pol_seasons)
    tm["context_s"] = round(time.time() - t, 1)
    return SeasonContext(S, store, params, hands, park_tab, lp_tab, tr, L_win, hz, tto,
                         tiers[tiers["season"] == S], tier_w, np.ones((5, 5, 1)),
                         ann[ann["season"] == S].reset_index(drop=True), tm)


def park_table(d_all: pd.DataFrame) -> dict:
    pf = RT.park_factors(d_all)
    first = d_all[["site", "home", "season"]].reset_index(drop=True).drop_duplicates()
    return {(s, h, int(y)): pf[i] for i, (s, h, y) in zip(first.index, first.itertuples(index=False))}


def platoon_table(d_all: pd.DataFrame) -> dict:
    lp = RT.league_platoon(d_all)
    first = d_all[["season", "bside", "phand"]].reset_index(drop=True).drop_duplicates()
    return {(int(y), b, h): lp[i] for i, (y, b, h) in zip(first.index, first.itertuples(index=False))}


def _bside(bats: str, phand: str) -> str:
    if bats == "B":
        return "R" if phand == "L" else "L"
    return bats if bats in ("L", "R") else "R"


def batter_devs(ctx: SeasonContext, ids, date, L, ab: Ablation) -> np.ndarray:
    """[n, 2(pitcher hand L,R), NR] batter deviation incl. individual platoon (cached)."""
    keys = [("b", i, date, ab.platoon) for i in ids]
    miss = [i for i, k in zip(ids, keys) if k not in ctx.cache]
    if miss:
        res = _batter_devs(ctx, miss, date, L, ab)
        for i, v in zip(miss, res):
            ctx.cache[("b", i, date, ab.platoon)] = v
    return np.stack([ctx.cache[k] for k in keys])


def _batter_devs(ctx: SeasonContext, ids, date, L, ab: Ablation) -> np.ndarray:
    n = len(ids)
    S = ctx.season
    Ls = np.repeat(L[None], n, 0)
    rb, per_hand = ctx.store.batter_rates(ids, np.full(n, S), np.full(n, date), Ls)
    out = np.zeros((n, 2, RT.NR))
    bats = [ctx.hands["bats"].get(i, "R") for i in ids]
    for h, hi in HAND.items():
        dev = np.log(rb / L)
        if ab.platoon:
            Ch, Nh = per_hand[h]
            lp = np.array([ctx.lp.get((S, _bside(b, h), h), np.ones(RT.NR)) for b in bats])
            prior = rb * lp
            prior /= prior.sum(1, keepdims=True)
            rbh = RT.shrink(Ch, Nh, prior, ctx.params.platoon_k * ctx.params.kappa_bat)
            dev = dev + np.log(rbh / rb)
        out[:, hi] = dev
    return out - out[..., [RT.REF]]


def pitcher_devs(ctx: SeasonContext, ids, date, L) -> np.ndarray:
    keys = [("p", i, date) for i in ids]
    miss = [i for i, k in zip(ids, keys) if k not in ctx.cache]
    if miss:
        for i, v in zip(miss, _pitcher_devs(ctx, miss, date, L)):
            ctx.cache[("p", i, date)] = v
    return np.stack([ctx.cache[k] for k in keys])


def _pitcher_devs(ctx: SeasonContext, ids, date, L) -> np.ndarray:
    n = len(ids)
    rp = ctx.store.pitcher_rates(ids, np.full(n, ctx.season), np.full(n, date), np.repeat(L[None], n, 0))
    dev = np.log(rp / L)
    return dev - dev[:, [RT.REF]]


def team_bullpen(ctx: SeasonContext, team: str, date, L, ab: Ablation):
    key = ("bp", team, date, ab.bullpen_tiers)
    if key not in ctx.cache:
        ctx.cache[key] = _team_bullpen(ctx, team, date, L, ab)
    return ctx.cache[key]


def _team_bullpen(ctx: SeasonContext, team: str, date, L, ab: Ablation):
    tz = ctx.tiers[(ctx.tiers["fteam"] == team) & (ctx.tiers["date"] == date)]
    nt = 3 if ab.bullpen_tiers else 1
    dev = np.zeros((nt, 2, RT.NR))
    share = np.full((nt, 2), 0.5)
    if len(tz) == 0:
        return dev, share
    ids = tz["pitcher"].to_numpy()
    n = len(ids)
    rp = ctx.store.pitcher_rates(ids, np.full(n, ctx.season), np.full(n, date), np.repeat(L[None], n, 0))
    hand = np.array([HAND.get(ctx.hands["throws"].get(i, "R"), 1) for i in ids])
    tier = tz["tier"].to_numpy() if ab.bullpen_tiers else np.zeros(n, int)
    wgt = tz["weight"].to_numpy()
    for t in range(nt):
        mt = tier == t
        if not mt.any():
            continue
        allr = (rp[mt] * wgt[mt, None]).sum(0) / wgt[mt].sum()
        for h in (0, 1):
            m = mt & (hand == h)
            share[t, h] = wgt[m].sum() / wgt[mt].sum()
            r = (rp[m] * wgt[m, None]).sum(0) / wgt[m].sum() if m.any() else allr
            e = np.log(r / L)
            dev[t, h] = e - e[RT.REF]
    return dev, share


def game_inputs(ctx: SeasonContext, g: pd.Series, lineups: dict, starters: dict, ab: Ablation):
    """lineups: {"home": [9 ids], "vis": [9 ids]}; starters: {"home": id, "vis": id}."""
    S, date = ctx.season, g["date"]
    L = ctx.store.league([S], [date])[0]
    env = MU.env_offset(L, ctx.L_win)
    park = ctx.park.get((g["site"], g["home"], S), np.zeros(RT.NR)) if ab.park else np.zeros(RT.NR)
    common = env + park - (park[RT.REF] if ab.park else 0)
    teams = {}
    for side, code in (("home", g["home"]), ("vis", g["visitor"])):
        bd = batter_devs(ctx, lineups[side], date, L, ab)
        sp = starters[side]
        sd = pitcher_devs(ctx, [sp], date, L)[0]
        tdev, tshare = team_bullpen(ctx, code, date, L, ab)
        teams[side] = D3.TeamInputs(bd, sd, HAND.get(ctx.hands["throws"].get(sp, "R"), 1), tdev, tshare)
    if ab.league_only:
        for side in teams:
            tdi = teams[side]
            teams[side] = D3.TeamInputs(np.zeros_like(tdi.bat_dev), np.zeros(RT.NR), tdi.starter_hand,
                                        np.zeros_like(tdi.tier_dev), tdi.tier_hand_share)
        common = np.zeros(RT.NR)
    tier_w = ctx.tier_w if ab.bullpen_tiers else ctx.tier_w_flat
    tto = ctx.tto if (ab.tto and not ab.league_only) else np.zeros_like(ctx.tto)
    gi = D3.GameInputs(teams["home"], teams["vis"], ctx.tr.q, ctx.tr.N, ctx.tr.A, common, ctx.hazard,
                       tier_w, tto, xrule=True)
    return gi, L


def evaluate_game(ctx: SeasonContext, eng: D3.Engine, g: pd.Series, rows: pd.DataFrame,
                  ab: Ablation = Ablation(), want_pitch: bool = True) -> dict:
    """rows: annotated PA rows of this game in order.  Returns per-PA B3 WP and
    (optionally) matchup q / after-outcome values for pitch-level evaluation."""
    t0 = time.time()
    lineups = {"home": [None] * 9, "vis": [None] * 9}
    for side, half in (("vis", 0), ("home", 1)):
        rr = rows[rows["half"] == half]
        for s in range(1, 10):
            x = rr.loc[rr["slot"] == s, "batter"]
            lineups[side][s - 1] = str(x.iloc[0]) if len(x) else "unknown"
    starters = {"home": str(rows.loc[rows["half"] == 0, "starter"].iloc[0]),
                "vis": str(rows.loc[rows["half"] == 1, "starter"].iloc[0])}
    gi, L = game_inputs(ctx, g, lineups, starters, ab)
    res = eng.solve(gi)
    pregame_s = res.seconds
    n_resolve = 0
    t_resolve = 0.0
    cur = {"home": list(lineups["home"]), "vis": list(lineups["vis"])}
    out_wp = np.empty(len(rows))
    qs, Gs = [], []
    # per team progress: next slot, PAs faced vs opposing starter, opposing starter removed
    nslot = {0: 0, 1: 0}
    n_vs_starter = {0: 0, 1: 0}          # PAs of each batting side against the opposing starter
    removed = {0: False, 1: False}       # batting side -> opposing starter already removed
    half_res = {}
    for i, r in enumerate(rows.itertuples(index=False)):
        half = int(r.half)
        side = "vis" if half == 0 else "home"
        inn = min(int(r.inning), 10)
        slot = int(r.slot) - 1 if r.slot > 0 else nslot[half]
        # pinch hitter / new batter in the slot -> update lineup, re-solve remaining halves
        if ab.league_only is False and str(r.batter) != cur[side][slot] and r.o >= 0:
            cur[side][slot] = str(r.batter)
            gi, L = game_inputs(ctx, g, cur, starters, ab)
            t = time.time()
            res = eng.solve(gi, start=(inn, half))
            t_resolve += time.time() - t
            n_resolve += 1
            half_res = {}
        is_st = bool(r.is_starter)
        if not is_st:
            removed[half] = True
        stage = 3 if removed[half] or inn >= 10 else int(r.lap) - 1
        key = (inn, half)
        rres = res
        if removed[half] and not is_st and ab.league_only is False:
            pid = str(r.pitcher)
            hk = (key, pid)
            if hk not in half_res:
                pdv = pitcher_devs(ctx, [pid], g["date"], L)[0]
                hand = HAND.get(ctx.hands["throws"].get(pid, "R"), 1)
                t = time.time()
                half_res[hk] = eng.solve(gi, override={key: (pdv, hand)}, upto=key, cache=res, start=key)
                t_resolve += time.time() - t
                n_resolve += 1
            rres = half_res[hk]
        other = 1 - half
        o_stage = 3 if removed[other] or inn >= 10 else min(n_vs_starter[other] // 9 + 1, 3) - 1
        c_bat = D3.composite(slot, stage)
        c_oth = D3.composite(nslot[other], o_stage)
        out_wp[i] = rres.wp(inn, half, int(r.outs0), int(r.bases0), int(r.diff0), c_bat, c_oth)
        if want_pitch and r.o >= 0 and r.outs1 < 3:
            q, G = first_pitch_parts(gi, rres, r, inn, half, slot, stage, c_oth, removed[half] and not is_st,
                                     ctx, g, L)
            qs.append(q)
            Gs.append(G)
        else:
            qs.append(None)
            Gs.append(None)
        if is_st:
            n_vs_starter[half] += 1
        if r.o >= 0 and r.o != T.O_IDX["INT"]:
            nslot[half] = (slot + 1) % 9
    return {"wp": out_wp, "q": qs, "G": Gs, "seconds": time.time() - t0, "pregame_s": pregame_s,
            "n_resolve": n_resolve, "resolve_s": t_resolve,
            "pregame_wp": out_wp[0] if len(out_wp) else np.nan}


def nslot_pa(rows: pd.DataFrame, i: int, team_half: int) -> int:
    """PAs the team batting in ``team_half`` has had against the opposing starter before row i."""
    prev = rows.iloc[:i]
    m = (prev["half"] == team_half) & prev["is_starter"]
    return int(m.sum())


def first_pitch_parts(gi, res, r, inn, half, slot, stage, c_oth, override, ctx, g, L):
    """Matchup q at the first-pitch state and after-outcome values G[O] (same
    composites/values as the DP), for the count-chain pitch-level WP."""
    bat = gi.vis if half == 0 else gi.home
    fld = gi.home if half == 0 else gi.vis
    if override:
        pid = str(r.pitcher)
        pdv = pitcher_devs(ctx, [pid], g["date"], L)[0]
        hand = HAND.get(ctx.hands["throws"].get(pid, "R"), 1)
    elif stage < 3:
        pdv, hand = fld.starter_dev + gi.tto[stage], fld.starter_hand
    else:
        pdv, hand = None, None
    o1, b1 = int(r.outs1), int(r.bases1)
    if pdv is not None:
        q = D3.q_for(gi, half, bat.bat_dev[slot, hand], pdv)[o1, b1]
    else:   # bullpen mixture at this inning and lead
        k = PP.fd_bucket(int(r.diff0) if half == 0 else -int(r.diff0))
        w = gi.tier_w[PP.inn_bucket(inn), k]
        q = np.zeros(NO)
        for t in range(len(w)):
            for h in (0, 1):
                s = w[t] * fld.tier_hand_share[t, h]
                if s > 0:
                    q += s * D3.q_for(gi, half, bat.bat_dev[slot, h], fld.tier_dev[t, h])[o1, b1]
    d1 = int(r.diff0) + (1 if half == 1 else -1) * int(r.rN)
    G = after_outcome_G(gi, res, inn, half, o1, b1, d1, slot, stage, c_oth)
    return q, G


def after_outcome_G(gi, res, inn, half, o1, b1, d1, slot, stage, c_oth) -> np.ndarray:
    """Expected home WP right after each outcome O from first-pitch state
    (o1, b1, diff d1), batting composite (slot, stage), other composite c_oth -
    using the solved DP values (same transitions as the engine)."""
    sgn = 1 if half == 1 else -1
    V = res.V[(inn, half)]
    E = res.E[(inn, half)]
    ci = (lambda c: c // D3.NST) if inn >= 10 else (lambda c: c)
    nxt_stage = stage if stage == 3 or inn >= 10 else (min(stage + 1, 2) if slot == 8 else stage)
    c_next = {0: D3.composite(slot, stage), 1: D3.composite((slot + 1) % 9, nxt_stage)}
    lim = res.R + res.pad
    # values of every (o2, b2, r) landing state for both slot advances
    U = np.zeros((2, 4, NB, RMAX + 1))
    for a in (0, 1):
        cb, co = ci(c_next[a]), ci(c_oth)
        for rr in range(RMAX + 1):
            dd = int(np.clip(d1 + sgn * rr, -lim, lim))
            di = dd + lim
            if half == 1 and inn >= 9 and dd > 0:
                U[a, :, :, rr] = 1.0
                continue
            U[a, :3, :, rr] = V[:, :, di, cb, co]
            U[a, 3, :, rr] = E[di, cb, co]
    adv = np.ones(NO, int)
    adv[T.O_IDX["INT"]] = 0
    return np.einsum("Ozwr,Ozwr->O", gi.A[:, o1, b1], U[adv])
