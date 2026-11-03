"""B4 step 3: Statcast pitch data -> features for the PA matchup model.

Player ids: Chadwick Baseball Bureau register (people-*.csv, columns key_mlbam,
key_retro), the same mapping a real-time pipeline (MLB ids in the live feed)
must use.  Downloaded to data/raw/register/ by ``fetch_register``.

Linking Statcast to the Retrosheet PA table
  game: (game date, home starting pitcher) - Statcast's first pitcher of the top of
        the 1st is the home starter; Retrosheet's ``starter`` of the top-half PAs.
        Team codes differ between the sources, so they are not used.
  PA:   within a game, Statcast plate appearances in at_bat_number order are
        matched to Retrosheet PAs with a batting result in order, requiring the
        same (batter, pitcher) pair; unmatched PAs are counted, not guessed.

Live-feed availability of candidate fields (only fields the real-time feed also
provides may be used): see LIVE_FEED below.  The cloud environment cannot reach
statsapi.mlb.com, so the table records the field names of the public MLB Stats API
game feed ("GUMBO" liveData.plays.allPlays[].playEvents[].pitchData / details) as
expected availability; it must be re-checked against a real feed before serving.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REGISTER_URL = "https://raw.githubusercontent.com/chadwickbureau/register/master/data/people-{}.csv"
LIVE_FEED = [
    # statcast column, live-feed field (MLB Stats API game feed), used by the model
    ("pitch_type", "playEvents[].details.type.code", True),
    ("release_speed", "playEvents[].pitchData.startSpeed", True),
    ("pfx_x / pfx_z", "playEvents[].pitchData.breaks (breakHorizontal, breakVerticalInduced)", True),
    ("plate_x / plate_z", "playEvents[].pitchData.coordinates (pX, pZ)", False),
    ("zone", "playEvents[].pitchData.zone", False),
    ("release_spin_rate", "playEvents[].pitchData.breaks.spinRate", False),
    ("launch_speed / launch_angle", "hitData (after contact only)", False),
    ("estimated_woba_using_speedangle", "not in the live feed (Savant model output)", False),
]


def fetch_register(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    for h in "0123456789abcdef":
        p = dest / f"people-{h}.csv"
        if not p.exists():
            urllib.request.urlretrieve(REGISTER_URL.format(h), p)
    return dest


def mlbam_to_retro(register_dir: Path) -> dict[int, str]:
    parts = []
    for p in sorted(register_dir.glob("people-*.csv")):
        df = pd.read_csv(p, usecols=["key_mlbam", "key_retro"], dtype=str, low_memory=False)
        parts.append(df.dropna())
    m = pd.concat(parts)
    m = m[(m["key_mlbam"] != "") & (m["key_retro"] != "")]
    return dict(zip(m["key_mlbam"].astype(float).astype(int), m["key_retro"]))


def link(sc: pd.DataFrame, pa: pd.DataFrame, games: pd.DataFrame, idmap: dict[int, str]) -> tuple[pd.DataFrame, dict]:
    """Return (Statcast pitches with retro game_id and PA row index ``pa_idx``, link report)."""
    sc = sc.copy()
    sc["bat_r"] = sc["batter"].map(idmap)
    sc["pit_r"] = sc["pitcher"].map(idmap)
    rep = {"pitches": int(len(sc)), "batter_id_unmapped": float(sc["bat_r"].isna().mean()),
           "pitcher_id_unmapped": float(sc["pit_r"].isna().mean())}
    first = sc[(sc["inning"] == 1) & (sc["inning_topbot"] == "Top")].sort_values(["game_pk", "at_bat_number"])
    hs = first.groupby("game_pk").agg(date=("game_date", "first"), home_sp=("pit_r", "first")).reset_index()
    hs["date"] = pd.to_datetime(hs["date"])
    g = games.set_index("game_id")
    top = pa[pa["half"] == 0].groupby("game_id")["starter"].first().astype(str).rename("home_sp").reset_index()
    top["date"] = top["game_id"].map(g["date"])
    gm = hs.merge(top, on=["date", "home_sp"], how="left")
    rep["games"] = int(len(hs))
    rep["games_linked"] = int(gm["game_id"].notna().sum())
    gmap = dict(zip(gm["game_pk"], gm["game_id"]))
    sc["game_id"] = sc["game_pk"].map(gmap)
    # PA alignment
    pa_b = pa[pa["o"] >= 0].reset_index().rename(columns={"index": "pa_idx"})
    sc_pa = sc.dropna(subset=["game_id"]).groupby(["game_id", "at_bat_number"]).agg(
        bat_r=("bat_r", "first"), pit_r=("pit_r", "first")).reset_index()
    out_links = []
    n_ok = n_tot = 0
    by_g = dict(tuple(pa_b.groupby("game_id")))
    for gid, s in sc_pa.groupby("game_id"):
        r = by_g.get(gid)
        if r is None:
            continue
        rb, rp, ri = r["batter"].astype(str).to_numpy(), r["pitcher"].astype(str).to_numpy(), r["pa_idx"].to_numpy()
        j = 0
        for ab, b, p in zip(s["at_bat_number"], s["bat_r"], s["pit_r"]):
            n_tot += 1
            k = j
            while k < len(rb) and k < j + 3 and not (rb[k] == b and rp[k] == p):
                k += 1                                    # allow skipping up to 2 Retrosheet-only rows
            if k < len(rb) and rb[k] == b and rp[k] == p:
                out_links.append((gid, ab, ri[k]))
                n_ok += 1
                j = k + 1
    lk = pd.DataFrame(out_links, columns=["game_id", "at_bat_number", "pa_idx"])
    rep["pa_statcast"] = int(n_tot)
    rep["pa_linked"] = int(n_ok)
    rep["pa_link_failure_rate"] = float(1 - n_ok / max(n_tot, 1))
    sc = sc.merge(lk, on=["game_id", "at_bat_number"], how="left")
    return sc, rep


FASTBALLS = {"FF", "SI", "FC", "FT", "FA"}


def pitcher_features(sc: pd.DataFrame, k_pitches: float = 300.0) -> pd.DataFrame:
    """As-of (strictly before the game date) pitcher fastball velocity and fastball share,
    season-to-date shrunk toward the league (k_pitches pseudo-pitches)."""
    s = sc.dropna(subset=["pit_r"]).copy()
    s["date"] = pd.to_datetime(s["game_date"])
    s["season"] = s["date"].dt.year
    s["fb"] = s["pitch_type"].isin(FASTBALLS).astype(float)
    s["fbv"] = np.where(s["fb"] == 1, s["release_speed"], np.nan)
    daily = s.groupby(["pit_r", "season", "date"]).agg(n=("fb", "size"), nfb=("fb", "sum"),
                                                        vsum=("fbv", lambda x: np.nansum(x)),
                                                        vn=("fbv", lambda x: np.isfinite(x).sum()))
    cum = daily.groupby(level=[0, 1]).cumsum() - daily
    lg = s.groupby("season").agg(fbshare=("fb", "mean"), fbv=("fbv", "mean"))
    cum = cum.reset_index().merge(lg, left_on="season", right_index=True)
    cum["fb_share"] = (cum["nfb"] + k_pitches * cum["fbshare"]) / (cum["n"] + k_pitches)
    cum["fb_velo"] = (cum["vsum"] + k_pitches * cum["fbv"]) / (cum["vn"] + k_pitches)
    cum["fb_velo_dev"] = cum["fb_velo"] - cum["fbv"]
    return cum[["pit_r", "date", "fb_share", "fb_velo", "fb_velo_dev"]]
