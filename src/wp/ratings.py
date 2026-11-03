"""Leak-free daily player ratings, park factors and platoon effects (B3).

Every quantity attached to a plate appearance (PA) of a game played on date D
uses only PAs from dates strictly before D (a doubleheader's second game does
not see the first).  Outcome categories are the 10 matchup categories
``RCATS`` (INT stays league-level, see matchup.py).

Player rate for a category o (batter or pitcher, "as of" D in season S):

    C_o = 1.0*c_o(S, <D) + 0.8*c_o(S-1) + 0.6*c_o(S-2) + 0.4*c_o(S-3)
    N   = same weights on PA counts
    r_o = (C_o + k_o * L_o) / (N + k_o),  then renormalised over o

  L = league rates of season S up to D (blended with season S-1, see
  ``league_asof``); k_o = kappa_role * BASE_K[o] (category-specific
  regression constants; kappa tuned on seasons before the evaluation seasons).

Platoon (batters): the batter's rate against this pitcher hand, shrunk
strongly (k = PLATOON_K x kappa x BASE_K) toward prior = his overall rate x
league platoon interaction (``league_platoon``); the log ratio to his overall
rate is added to the matchup log-odds.

Park factor: classic home/road ratio per park (Retrosheet ``info,site``):
rate of PAs in games at the park vs the same home team's road games, over
the three previous seasons (walk-forward by season), log ratio shrunk to 0.

Hands: Retrosheet roster files (``id,last,first,bats,throws,team,pos``).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .transitions import O_IDX, OUTCOMES

RCATS = [o for o in OUTCOMES if o != "INT"]               # 10 matchup categories
R_IDX = {o: i for i, o in enumerate(RCATS)}
NR = len(RCATS)
REF = R_IDX["OUT"]
# regression constants (PA) per category, a la stabilisation points; scaled by kappa
BASE_K = np.array([60, 120, 300, 240, 290, 1600, 1600, 170, 300, 2000], float)
SEASON_W = {0: 1.0, 1: 0.8, 2: 0.6, 3: 0.4}
PLATOON_K = 6.0


def load_hands(raw: Path, first: int = 1995, last: int = 2025) -> pd.DataFrame:
    rows = []
    for y in range(first, last + 1):
        for f in sorted((raw / str(y)).glob("*.ROS")):
            for line in f.read_text(encoding="latin1").splitlines():
                p = line.split(",")
                if len(p) >= 5:
                    rows.append((p[0], p[3], p[4], y))
    h = pd.DataFrame(rows, columns=["player", "bats", "throws", "season"])
    h = h.sort_values("season").drop_duplicates("player", keep="last")
    return h.set_index("player")[["bats", "throws"]]


def pa_frame(pa: pd.DataFrame, games: pd.DataFrame, hands: pd.DataFrame) -> pd.DataFrame:
    """PAs with a matchup outcome (o in RCATS or INT) + date, site, hands."""
    d = pa[pa["o"] >= 0].copy()
    g = games.set_index("game_id")
    d["date"] = d["game_id"].map(g["date"])
    d["site"] = d["game_id"].map(g["site"]).fillna("")
    d["home"] = d["game_id"].str[:3]
    d["vis"] = d["game_id"].map(g["visitor"]).fillna("")
    d["included"] = d["game_id"].map(g["included"]).fillna(False).astype(bool)
    d["season"] = d["season"].astype(int)
    b = d["batter"].astype(str).map(hands["bats"]).fillna("R")
    t = d["pitcher"].astype(str).map(hands["throws"]).fillna("R")
    d["phand"] = t.where(t.isin(["L", "R"]), "R")
    # switch hitters bat from the side opposite the pitcher's throwing hand
    d["bside"] = np.where(b == "B", np.where(d["phand"] == "L", "R", "L"), b.where(b.isin(["L", "R"]), "R"))
    oi = d["o"].to_numpy()
    ri = np.full(len(d), -1)
    for o, i in R_IDX.items():
        ri[oi == O_IDX[o]] = i
    d["r"] = ri                                           # -1 for INT
    d = d.sort_values(["date", "game_id"], kind="mergesort").reset_index(drop=True)
    return d


def _onehot(r: np.ndarray) -> np.ndarray:
    m = np.zeros((len(r), NR), np.float32)
    ok = r >= 0
    m[np.flatnonzero(ok), r[ok]] = 1
    return m


def asof_counts(d: pd.DataFrame, keys: list[str], weights: np.ndarray | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
    """Season-weighted counts for the entity ``keys`` (e.g. ["batter"]) using only
    PAs on dates strictly before each row's date.  Returns (C [n, NR], N [n]).

    ``weights`` [n, NR] multiplies each PA's one-hot outcome (park-neutral counts:
    1 / park factor of the PA's park); C is then rescaled so that sum_o C_o = N
    (ratio estimator - the PA count is unchanged)."""
    oh = _onehot(d["r"].to_numpy())
    if weights is not None:
        oh = (oh * weights).astype(np.float32)
    cols = [f"c{i}" for i in range(NR)]
    base = d[keys + ["season", "date"]].copy()
    base[cols] = oh
    base["n"] = (d["r"].to_numpy() >= 0).astype(np.float32)
    daily = base.groupby(keys + ["season", "date"], observed=True)[cols + ["n"]].sum()
    cum = daily.groupby(level=list(range(len(keys) + 1)), observed=True).cumsum() - daily
    cur = cum.reindex(pd.MultiIndex.from_frame(base[keys + ["season", "date"]])).to_numpy()
    tot = base.groupby(keys + ["season"], observed=True)[cols + ["n"]].sum()
    out = cur.astype(np.float64) * SEASON_W[0]
    for lag in (1, 2, 3):
        idx = base[keys + ["season"]].copy()
        idx["season"] = idx["season"] - lag
        prev = tot.reindex(pd.MultiIndex.from_frame(idx)).to_numpy()
        out += SEASON_W[lag] * np.nan_to_num(prev)
    C, N = out[:, :NR], out[:, NR]
    if weights is not None:
        tot_c = C.sum(1)
        C = np.where(tot_c[:, None] > 0, C * (N / np.where(tot_c > 0, tot_c, 1))[:, None], C)
    return C, N


def league_asof(d: pd.DataFrame, m: float = 20000.0) -> np.ndarray:
    """League category rates for each row: season-to-date (before the date)
    blended with the previous full season using m pseudo-PAs."""
    oh = _onehot(d["r"].to_numpy()).astype(np.float64)
    df = pd.DataFrame(oh, columns=range(NR))
    df["season"], df["date"] = d["season"].to_numpy(), d["date"].to_numpy()
    daily = df.groupby(["season", "date"]).sum()
    cum = daily.groupby(level=0).cumsum() - daily
    seas = df.groupby("season")[list(range(NR))].sum()
    prev = seas.shift(1).reindex(seas.index)
    prev_rate = prev.div(prev.sum(1), axis=0)
    cur = cum.reindex(pd.MultiIndex.from_arrays([df["season"], df["date"]])).to_numpy()
    pr = prev_rate.reindex(df["season"]).to_numpy()
    pr = np.where(np.isnan(pr), (seas.sum() / seas.sum().sum()).to_numpy()[None, :], pr)
    L = (cur + m * pr) / (cur.sum(1, keepdims=True) + m)
    return L


def shrink(C: np.ndarray, N: np.ndarray, L: np.ndarray, kappa: float) -> np.ndarray:
    k = kappa * BASE_K[None, :]
    r = (C + k * L) / (N[:, None] + k)
    return r / r.sum(1, keepdims=True)


def park_factors(d: pd.DataFrame, k_scale: float = 4.0) -> np.ndarray:
    """log park factor per row and category (seasons S-3..S-1 only)."""
    oh = _onehot(d["r"].to_numpy()).astype(np.float64)
    df = pd.DataFrame(oh, columns=range(NR))
    df["season"] = d["season"].to_numpy()
    df["site"] = d["site"].to_numpy()
    df["home"] = d["home"].to_numpy()
    df["vis"] = d["vis"].to_numpy()
    home = df.groupby(["site", "home", "season"])[list(range(NR))].sum()
    road_h = df.groupby(["vis", "season"])[list(range(NR))].sum()      # road PAs of team = as visitor
    out = np.zeros((len(d), NR))
    keys = d[["site", "home", "season"]].drop_duplicates()
    rows = {}
    for site, hm, s in keys.itertuples(index=False):
        hc = sum((home.loc[(site, hm, s - l)].to_numpy() for l in (1, 2, 3)
                  if (site, hm, s - l) in home.index), np.zeros(NR))
        rc = sum((road_h.loc[(hm, s - l)].to_numpy() for l in (1, 2, 3)
                  if (hm, s - l) in road_h.index), np.zeros(NR))
        nh, nr = hc.sum(), rc.sum()
        if nh == 0 or nr == 0:
            rows[(site, hm, s)] = np.zeros(NR)
            continue
        lr = (rc + 1) / (nr + NR)
        hr = (hc + 1) / (nh + NR)
        k = k_scale * BASE_K
        w = nh / (nh + k)                                   # shrink toward 0 by home sample size
        rows[(site, hm, s)] = w * np.log(hr / lr)
    idx = pd.MultiIndex.from_frame(d[["site", "home", "season"]])
    tab = pd.DataFrame.from_dict(rows, orient="index")
    tab.index = pd.MultiIndex.from_tuples(tab.index)
    out = tab.reindex(idx).to_numpy()
    return np.nan_to_num(out)


def league_platoon(d: pd.DataFrame) -> np.ndarray:
    """League platoon *interaction* per row: L(o|bside,phand) L(o) / (L(o|bside) L(o|phand)),
    from seasons S-3..S-1 (main effects of batter side and pitcher hand are already in
    the individual ratings)."""
    oh = _onehot(d["r"].to_numpy()).astype(np.float64)
    df = pd.DataFrame(oh, columns=range(NR))
    df["season"], df["bs"], df["ph"] = d["season"].to_numpy(), d["bside"].to_numpy(), d["phand"].to_numpy()
    out = np.ones((len(d), NR))
    for s in np.unique(df["season"]):
        w = df[df["season"].between(s - 3, s - 1)]
        if len(w) == 0:
            continue
        def rate(g):
            x = g[list(range(NR))].sum().to_numpy() + 1.0
            return x / x.sum()
        Lall = rate(w)
        Lb = {b: rate(w[w["bs"] == b]) for b in ("L", "R")}
        Lp = {h: rate(w[w["ph"] == h]) for h in ("L", "R")}
        m_s = (df["season"] == s).to_numpy()
        for b in ("L", "R"):
            for h in ("L", "R"):
                inter = rate(w[(w["bs"] == b) & (w["ph"] == h)]) * Lall / (Lb[b] * Lp[h])
                m = m_s & (df["bs"] == b).to_numpy() & (df["ph"] == h).to_numpy()
                out[m] = inter
    return out


@dataclass
class RatingParams:
    kappa_bat: float = 1.0
    kappa_pit: float = 1.0
    platoon_k: float = PLATOON_K
    park: bool = True
    platoon: bool = True


def build_features(d: pd.DataFrame, park_neutral: bool = False) -> dict:
    """All as-of count arrays needed for ratings (computed once, reused for tuning).

    park_neutral: player counts are deflated by the park factor of the park each
    PA was played in (B4 1.1), so that the park is applied only once, in the matchup."""
    L = league_asof(d)
    pf = park_factors(d)
    w = np.exp(-pf) if park_neutral else None
    Cb, Nb = asof_counts(d, ["batter"], w)
    Cp, Np = asof_counts(d, ["pitcher"], w)
    Cbh, Nbh = asof_counts(d, ["batter", "phand"], w)       # batter vs this pitcher hand
    lp = league_platoon(d)
    return {"L": L, "Cb": Cb, "Nb": Nb, "Cp": Cp, "Np": Np, "Cbh": Cbh, "Nbh": Nbh, "park": pf, "LP": lp}


def deviations(F: dict, p: RatingParams) -> np.ndarray:
    """Per-row log-odds deviation (vs OUT) from league for batter + pitcher +
    park + individual platoon.  Shape [n, NR]."""
    L = F["L"]
    rb = shrink(F["Cb"], F["Nb"], L, p.kappa_bat)
    rp = shrink(F["Cp"], F["Np"], L, p.kappa_pit)
    dev = np.log(rb / L) + np.log(rp / L)
    if p.platoon:
        # individual batter split vs this hand, shrunk strongly toward the batter's overall
        # rate times the league platoon interaction (the prior)
        prior = rb * F["LP"]
        prior /= prior.sum(1, keepdims=True)
        rbh = shrink(F["Cbh"], F["Nbh"], prior, p.platoon_k * p.kappa_bat)
        dev += np.log(rbh / rb)
    if p.park:
        dev += F["park"]
    return dev - dev[:, [REF]]


class RatingStore:
    """As-of lookups for arbitrary (player, date) pairs, consistent with
    ``build_features``/``deviations`` (same season weights and shrinkage).

    ``batter_rates(ids, dates, L)`` -> r_b [n, NR] and per-hand rates {"L","R"}.
    ``pitcher_rates(ids, dates, L)`` -> r_p [n, NR].
    ``league(dates)`` -> league rates L [n, NR] (season-to-date blend).
    """

    def __init__(self, d: pd.DataFrame, params: RatingParams):
        self.p = params
        oh = _onehot(d["r"].to_numpy())
        self.cols = [f"c{i}" for i in range(NR)]
        base = pd.DataFrame(oh, columns=self.cols)
        base["n"] = (d["r"].to_numpy() >= 0).astype(np.float32)
        for c in ("batter", "pitcher", "phand", "season", "date"):
            base[c] = d[c].astype(str).to_numpy() if c in ("batter", "pitcher", "phand") else d[c].to_numpy()
        self.tables = {}
        for role, keys in (("bat", ["batter"]), ("pit", ["pitcher"]), ("bath", ["batter", "phand"])):
            daily = base.groupby(keys + ["season", "date"])[self.cols + ["n"]].sum()
            cum = daily.groupby(level=list(range(len(keys) + 1))).cumsum().reset_index()
            tot = base.groupby(keys + ["season"])[self.cols + ["n"]].sum()
            self.tables[role] = (keys, cum.sort_values("date"), tot)
        # league daily cumulative
        lg = base.groupby(["season", "date"])[self.cols].sum()
        self.lg_cum = lg.groupby(level=0).cumsum().reset_index().sort_values("date")
        seas = base.groupby("season")[self.cols].sum()
        self.lg_season = seas
        self.lg_all = (seas.sum() / seas.sum().sum()).to_numpy()

    def _asof(self, role, key_vals: dict, seasons, dates) -> tuple[np.ndarray, np.ndarray]:
        keys, cum, tot = self.tables[role]
        q = pd.DataFrame({**key_vals, "season": seasons, "date": pd.to_datetime(dates)})
        q["_i"] = np.arange(len(q))
        q = q.sort_values("date")
        m = pd.merge_asof(q, cum, on="date", by=keys + ["season"], direction="backward",
                          allow_exact_matches=False).sort_values("_i")
        cur = np.nan_to_num(m[self.cols + ["n"]].to_numpy(np.float64)) * SEASON_W[0]
        for lag in (1, 2, 3):
            idx = pd.MultiIndex.from_frame(pd.DataFrame({**{k: key_vals[k] for k in keys},
                                                         "season": np.asarray(seasons) - lag}))
            cur += SEASON_W[lag] * np.nan_to_num(tot.reindex(idx).to_numpy())
        return cur[:, :NR], cur[:, NR]

    def league(self, seasons, dates, m: float = 20000.0) -> np.ndarray:
        q = pd.DataFrame({"season": seasons, "date": pd.to_datetime(dates)})
        q["_i"] = np.arange(len(q))
        mm = pd.merge_asof(q.sort_values("date"), self.lg_cum, on="date", by="season",
                           direction="backward", allow_exact_matches=False).sort_values("_i")
        cur = np.nan_to_num(mm[self.cols].to_numpy(np.float64))
        prev = self.lg_season.reindex(np.asarray(seasons) - 1).to_numpy()
        prev = prev / prev.sum(1, keepdims=True)
        prev = np.where(np.isnan(prev), self.lg_all[None], prev)
        return (cur + m * prev) / (cur.sum(1, keepdims=True) + m)

    def batter_rates(self, ids, seasons, dates, L):
        C, N = self._asof("bat", {"batter": np.asarray(ids, str)}, seasons, dates)
        rb = shrink(C, N, L, self.p.kappa_bat)
        per_hand = {}
        for h in ("L", "R"):
            Ch, Nh = self._asof("bath", {"batter": np.asarray(ids, str), "phand": np.full(len(ids), h)},
                                seasons, dates)
            per_hand[h] = (Ch, Nh)
        return rb, per_hand

    def pitcher_rates(self, ids, seasons, dates, L):
        C, N = self._asof("pit", {"pitcher": np.asarray(ids, str)}, seasons, dates)
        return shrink(C, N, L, self.p.kappa_pit)
