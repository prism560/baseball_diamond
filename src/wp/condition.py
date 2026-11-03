"""B4 step 4: in-game condition of the current pitcher from Retrosheet pitch sequences.

For a PA, only pitches thrown *before* the PA by the same pitcher in the same game
are used:
  n_prev        pitches thrown so far
  strikes_prev  strikes among them (called/swinging/foul/foul tip/bunt/in play; the
                strike classes of Chadwick game.c, see retro_parse.check_pitches)
  base          the pitcher's strike rate as of the day before the game (season to
                date + previous season at half weight), shrunk to the league rate of
                all pitches on earlier dates
  cond_dev      Bayesian deviation (strikes_prev - n_prev*base) / (n_prev + k)
                (k pseudo-pitches: strong prior, tuned before the evaluation season)
  pc            n_prev / 100 (pitch count)
Velocity relative to the season baseline needs Statcast (step 3) and is not used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .retro_parse import BALLS, FOULS, INPLAY, STRIKES_ALWAYS

STRIKE_CODES = set(STRIKES_ALWAYS) | set(FOULS) | set(INPLAY)
PITCH_CODES = STRIKE_CODES | set(BALLS) | {"H"}


def count_pitches(seq: str) -> tuple[int, int]:
    n = s = 0
    for c in seq:
        if c in PITCH_CODES:
            n += 1
            s += c in STRIKE_CODES
    return n, s


def condition_features(d: pd.DataFrame, league_rate: float | None = None, m_prior: float = 500.0) -> pd.DataFrame:
    """d: PA rows in game order (columns game_id, pitcher, pitches, season, date).
    Returns n_prev, strikes_prev, base for every row (no information from the row itself
    or from later rows)."""
    ns = np.array([count_pitches(str(p)) for p in d["pitches"].to_numpy()])
    n, s = ns[:, 0].astype(float), ns[:, 1].astype(float)
    key = [d["game_id"].to_numpy(), d["pitcher"].astype(str).to_numpy()]
    df = pd.DataFrame({"g": key[0], "p": key[1], "n": n, "s": s})
    grp = df.groupby(["g", "p"], sort=False)
    n_prev = (grp["n"].cumsum() - df["n"]).to_numpy()
    s_prev = (grp["s"].cumsum() - df["s"]).to_numpy()
    # pitcher baseline strike rate before the game date
    bd = pd.DataFrame({"p": key[1], "season": d["season"].to_numpy(), "date": d["date"].to_numpy(), "n": n, "s": s})
    daily = bd.groupby(["p", "season", "date"])[["n", "s"]].sum()
    cum = daily.groupby(level=[0, 1]).cumsum() - daily
    cur = cum.reindex(pd.MultiIndex.from_frame(bd[["p", "season", "date"]])).to_numpy()
    tot = bd.groupby(["p", "season"])[["n", "s"]].sum()
    prev = tot.reindex(pd.MultiIndex.from_arrays([bd["p"], bd["season"] - 1])).to_numpy()
    acc = cur + 0.5 * np.nan_to_num(prev)
    if league_rate is None:
        # league strike rate from every pitch on earlier dates only (no future data)
        lg = bd.groupby("date")[["n", "s"]].sum().sort_index()
        lc = lg.cumsum() - lg
        ln = lc["n"].reindex(bd["date"]).to_numpy()
        ls = lc["s"].reindex(bd["date"]).to_numpy()
        league_rate = np.where(ln > 0, ls / np.where(ln > 0, ln, 1), 0.62)   # 0.62: placeholder before any data
    base = (acc[:, 1] + m_prior * league_rate) / (acc[:, 0] + m_prior)
    return pd.DataFrame({"n_prev": n_prev, "strikes_prev": s_prev, "base": base}, index=d.index)


def cond_dev(F: pd.DataFrame, k: float) -> np.ndarray:
    return ((F["strikes_prev"] - F["n_prev"] * F["base"]) / (F["n_prev"] + k)).to_numpy()
