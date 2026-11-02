"""Leak-free Elo pre-game win probability from Retrosheet game logs.

* Ratings are updated strictly in chronological order (date, then game number);
  the pre-game probability of a game is recorded *before* its result is used.
  Games on the same date are processed sequentially by game number, so a
  doubleheader's game 2 sees game 1 (which really was played first).
* Home-field advantage ``hfa`` (Elo points) and between-season regression to
  the mean ``revert`` (fraction pulled back to 1500) are included.
* Ties and forfeits do not update ratings (they still get a pre-game prob.).
* Franchise continuity across relocations / code changes: ``FRANCHISE`` maps
  Retrosheet team codes to a franchise id.  Year ranges were checked against
  Retrosheet's reference/teams.csv (FIRST/LAST columns) in the Chadwick mirror.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

FRANCHISE = {
    "MLA": "BAL", "SLA": "BAL",          # Milwaukee Brewers 1901 -> St. Louis Browns 1902-53 -> Orioles 1954
    "BLA": "NYA",                        # Baltimore Orioles 1901-02 -> New York 1903
    "BSN": "ATL", "MLN": "ATL",          # Boston 1876-1952 -> Milwaukee 1953-65 -> Atlanta 1966
    "PHA": "ATH", "KC1": "ATH", "OAK": "ATH",  # Phila. -> Kansas City 1955 -> Oakland 1968 -> Sacramento 2025
    "BRO": "LAN", "NY1": "SFN",          # 1958 moves
    "WS1": "MIN", "WS2": "TEX",          # Senators -> Twins 1961, expansion Senators -> Rangers 1972
    "LAA": "ANA", "CAL": "ANA",          # Angels code changes 1965, 1997
    "SE1": "MIL",                        # Pilots 1969 -> Brewers 1970
    "MON": "WAS",                        # Expos -> Nationals 2005
    "FLO": "MIA",                        # Marlins 2012
}


def franchise(code: str) -> str:
    return FRANCHISE.get(code, code)


@dataclass(frozen=True)
class EloParams:
    k: float = 4.0
    hfa: float = 24.0
    revert: float = 0.33
    base: float = 1500.0


def win_prob(diff_points):
    return 1.0 / (1.0 + 10.0 ** (-np.asarray(diff_points) / 400.0))


def run_elo(gl: pd.DataFrame, p: EloParams = EloParams()) -> pd.DataFrame:
    """Return ``gl`` (sorted) with columns elo_home, elo_vis, p_pregame.

    ``gl`` needs: game_id, season, date, number, home, visitor, home_score,
    vis_score, forfeit.
    """
    g = gl.sort_values(["date", "number", "game_id"], kind="mergesort").reset_index(drop=True)
    home = g["home"].map(franchise).to_numpy()
    vis = g["visitor"].map(franchise).to_numpy()
    season = g["season"].to_numpy()
    hs, vs = g["home_score"].to_numpy(), g["vis_score"].to_numpy()
    forfeit = (g["forfeit"].fillna("") != "").to_numpy()
    ratings: dict[str, float] = {}
    eh = np.empty(len(g)); ev = np.empty(len(g)); pp = np.empty(len(g))
    cur = None
    for i in range(len(g)):
        if season[i] != cur:
            cur = season[i]
            for t in ratings:
                ratings[t] = p.base + (1 - p.revert) * (ratings[t] - p.base)
        rh = ratings.get(home[i], p.base)
        rv = ratings.get(vis[i], p.base)
        e = 1.0 / (1.0 + 10.0 ** (-(rh + p.hfa - rv) / 400.0))
        eh[i], ev[i], pp[i] = rh, rv, e
        if forfeit[i] or hs[i] == vs[i] or math.isnan(hs[i]) or math.isnan(vs[i]):
            continue
        y = 1.0 if hs[i] > vs[i] else 0.0
        delta = p.k * (y - e)
        ratings[home[i]] = rh + delta
        ratings[vis[i]] = rv - delta
    g["elo_home"], g["elo_vis"], g["p_pregame"] = eh, ev, pp
    return g


def elo_logloss(g: pd.DataFrame, seasons) -> float:
    m = g["season"].isin(seasons) & (g["home_score"] != g["vis_score"]) & (g["forfeit"].fillna("") == "")
    y = (g.loc[m, "home_score"] > g.loc[m, "vis_score"]).to_numpy(float)
    q = np.clip(g.loc[m, "p_pregame"].to_numpy(), 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))


def tune(gl: pd.DataFrame, tune_seasons, ks=(2, 3, 4, 6, 8), hfas=(16, 24, 32),
         reverts=(0.2, 0.33, 0.5)) -> tuple[EloParams, dict]:
    """Grid search on log-loss over ``tune_seasons``.  Because Elo is computed
    sequentially, scoring seasons <= T only uses information up to each game,
    and choosing hyper-parameters on seasons < evaluation season S never sees S."""
    res = {}
    for k in ks:
        for h in hfas:
            for r in reverts:
                pr = EloParams(k, h, r)
                res[(k, h, r)] = elo_logloss(run_elo(gl, pr), tune_seasons)
    best = min(res, key=res.get)
    return EloParams(*best), res
