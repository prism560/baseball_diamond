"""Season model bundle for serving: B3 context + engine, B2 / B2E tables, count chain, Elo.

Everything is built with the same functions and windows as the offline
evaluation (wp.run_b3_games, wp.run_b3_eval, wp.run_b4_combine), so a replayed
game reproduces the offline values:

  B3    wp.b3_game.build_context(S) (ratings as of the day before each game), Engine(R=20, float32)
  B2    league DP on the fixed K=10 window before S = ctx.tr (dp.Engine(30, 1)), pitch table via
        the league count chain fitted to tr.q
  B2E   logit = a + b logit(B2) + c logit(P_Elo) + d logit(P_Elo) progress, coefficients of
        season S from reports/b3_results.json (offline fit), or models/live/b2e_<S>.json
  AVG   sigma(w logit(B3) + (1 - w) logit(B2E)), w = 0.55 (B4 serving decision)
  Elo   wp.elo.run_elo on the game logs (+ feed-derived results, wp.live.ratings_update)

Postseason games have no automatic runner: the B2 engine and B3 inputs use xrule=False.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .. import b3_game as BG
from .. import count_chain as CC
from .. import dp
from .. import dp_b3 as D3
from .. import elo as E
from .. import ratings as RT
from ..b1_model import B1Model, logit
from .log import info

AVG_W = 0.55
PROCESSED = Path("data/processed")
RAW = Path("data/raw")
LIVE_MODELS = Path("models/live")


def avg(b3, b2e, w: float = AVG_W):
    z = w * logit(b3) + (1 - w) * logit(b2e)
    return 1 / (1 + np.exp(-z))


@dataclass
class B2Tables:
    res: object          # dp.DPResult
    G: dict              # (inning, half) -> [outs, bases, d, O] after-outcome values
    pitch: dict          # (inning, half) -> [outs, bases, d, count]
    xrule: bool

    def d_index(self, diff):
        return int(np.clip(diff, -self.res.R - self.res.pad, self.res.R + self.res.pad)) + self.res.R + self.res.pad

    def pa_value(self, inning, half, outs, bases, diff) -> float:
        i = min(int(inning), 10)
        return float(self.res.V[(i, half)][outs, bases, self.d_index(diff), 0, 0])

    def pitch_value(self, inning, half, outs, bases, diff, count_idx) -> float:
        return float(self.pitch[(min(int(inning), 10), half)][outs, bases, self.d_index(diff), count_idx])

    def after(self, inning, half, outs, bases, diff) -> np.ndarray:
        return self.G[(min(int(inning), 10), half)][outs, bases, self.d_index(diff)]


@dataclass
class SeasonModels:
    season: int
    ctx: BG.SeasonContext
    eng: D3.Engine
    games: pd.DataFrame          # games table (Retrosheet) indexed by game_id
    base: CC.BaseRates
    b2: dict                     # xrule -> B2Tables
    b2e: B1Model
    w: float
    elo_p: dict                  # game_id -> pre-game P(home)
    elo_state: dict              # franchise -> rating after the last game in the logs
    elo_params: E.EloParams
    site_of: dict                # home team -> modal park code in season S
    timings: dict = field(default_factory=dict)

    @classmethod
    def load(cls, S: int, processed: Path = PROCESSED, raw: Path = RAW) -> "SeasonModels":
        t0 = time.time()
        tm = {}
        params = json.loads(Path("reports/b3_check.json").read_text())["params"]
        pa_all = pd.read_parquet(processed / "pa.parquet")
        live_pa = processed / "pa_live.parquet"
        if live_pa.exists():                  # feed-derived rows for seasons not in Retrosheet yet
            pa_all = pd.concat([pa_all, pd.read_parquet(live_pa)], ignore_index=True)
        games = pd.read_parquet(processed / "games.parquet")
        live_g = processed / "games_live.parquet"
        if live_g.exists():
            games = pd.concat([games, pd.read_parquet(live_g)], ignore_index=True)
        pa = pa_all[pa_all["season"] >= 2012]
        hands = RT.load_hands(raw, 2008, 2025)
        d_all = RT.pa_frame(pa[pa["season"] >= 2017], games, hands)
        p = RT.RatingParams(params["kappa_bat"], params["kappa_pit"], params["platoon_k"])
        store = RT.RatingStore(d_all, p)
        tm["ratings_s"] = round(time.time() - t0, 1)
        ctx = BG.build_context(S, pa, games, d_all, store, p, hands, BG.park_table(d_all), BG.platoon_table(d_all))
        tm["context_s"] = round(time.time() - t0, 1)
        inc = set(games.loc[games["included"], "game_id"])
        seasons_avail = sorted(int(s) for s in pa_all["season"].unique() if s != 2020)
        base = CC.estimate_base(pa_all[pa_all["game_id"].isin(inc)], BG.window(seasons_avail, S))
        b2 = {x: _b2_tables(ctx.tr, base, x) for x in (True, False)}
        tm["b2_s"] = round(time.time() - t0, 1)
        b2e = load_b2e(S)
        gl = pd.read_parquet(processed / "gamelogs.parquet")
        live_gl = processed / "gamelogs_live.parquet"
        if live_gl.exists():
            gl = pd.concat([gl, pd.read_parquet(live_gl)], ignore_index=True)
        ep = E.EloParams(**json.loads(Path("models/elo_params.json").read_text()))
        el = E.run_elo(gl, ep)
        elo_p = dict(zip(el["game_id"], el["p_pregame"]))
        state = _elo_final_state(el, ep)
        gS = games[games["season"] == S]
        site_of = gS.groupby("home")["site"].agg(lambda s: s.value_counts().index[0]).to_dict() if len(gS) else {}
        tm["total_s"] = round(time.time() - t0, 1)
        info("season_models_loaded", season=S, **tm)
        return cls(S, ctx, D3.Engine(20, np.float32), games.set_index("game_id"), base, b2, b2e, AVG_W, elo_p,
                   state, ep, site_of, tm)

    # ---- Elo --------------------------------------------------------------
    def p_pregame(self, game_id: str, home: str, visitor: str, season: int) -> float:
        if game_id in self.elo_p:
            return float(self.elo_p[game_id])
        # not in the logs (postseason / current season before the logs are updated): current ratings
        p = self.elo_params
        rh = self.elo_state.get(E.franchise(home), p.base)
        rv = self.elo_state.get(E.franchise(visitor), p.base)
        return float(1.0 / (1.0 + 10.0 ** (-(rh + p.hfa - rv) / 400.0)))

    def b2e_value(self, b2, p_pre, progress):
        return float(self.b2e.predict(np.atleast_1d(b2), np.atleast_1d(p_pre), np.atleast_1d(progress))[0])


def _b2_tables(tr, base, xrule: bool) -> B2Tables:
    res = dp.Engine(30, 1, xrule=xrule).solve(dp.kernels_from(tr))
    G = dp.after_outcome_values(res, tr, xrule=xrule)
    ch = CC.fit(base, tr.q)
    return B2Tables(res, G, CC.pitch_wp_table(ch, G), xrule)


def load_b2e(S: int) -> B1Model:
    f = LIVE_MODELS / f"b2e_{S}.json"
    if f.exists():
        return B1Model(**json.loads(f.read_text()))
    r = json.loads(Path("reports/b3_results.json").read_text())
    return B1Model(**r["per_season"][str(S)]["b2e_coef"])


def _elo_final_state(el: pd.DataFrame, p: E.EloParams) -> dict:
    """Ratings after the last game of ``el`` (run_elo output, chronological)."""
    st: dict = {}
    for row in el.itertuples(index=False):
        h, v = E.franchise(row.home), E.franchise(row.visitor)
        rh, rv = row.elo_home, row.elo_vis
        st[h], st[v] = rh, rv
        if row.home_score == row.vis_score or pd.isna(row.home_score) or (getattr(row, "forfeit", "") or ""):
            continue
        y = 1.0 if row.home_score > row.vis_score else 0.0
        e = 1.0 / (1.0 + 10.0 ** (-(rh + p.hfa - rv) / 400.0))
        st[h], st[v] = rh + p.k * (y - e), rv - p.k * (y - e)
    return st
