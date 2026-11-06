#!/usr/bin/env python3
"""Daily update during the season: completed game feeds -> PA / games / game-log rows.

Retrosheet publishes a season only after it ends, so during the season the ratings,
bullpen tiers and Elo of wp.live.models are fed from the live feeds:

    data/processed/pa_live.parquet        PA table rows (wp.transitions format, Retrosheet ids)
    data/processed/games_live.parquet     games-table rows (date, teams, park, included, score)
    data/processed/gamelogs_live.parquet  game-log rows for Elo

wp.live.models.SeasonModels.load() appends these to the Retrosheet tables, so restarting the
server (or the next day's model load) uses ratings and Elo through yesterday.

    python scripts/update_ratings_daily.py                     # yesterday's final games (StatsAPI)
    python scripts/update_ratings_daily.py --date 2026-04-01   # one date
    python scripts/update_ratings_daily.py --from-local        # every final game in data/external/feeds
    python scripts/update_ratings_daily.py --season-setup 2026 # B2E coefficients for the season

Games already in the Retrosheet tables are skipped (the season's Retrosheet files replace the
feed rows once they are published; rows of the same game_id are then dropped here).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

P = Path("data/processed")


def convert(feed: dict, ids, gid: str | None = None):
    from wp.live.normalize import normalize
    from wp.live.pa_from_feed import game_row, pa_rows
    from wp.live.reconstruct import reconstruct
    view = reconstruct(normalize(feed))
    g = view.game
    if g.status != "final":
        return None
    gid = gid or None
    gr = game_row(view, gid)
    gid = gr["game_id"]
    pa = pa_rows(view, ids, gid)
    a, h = view.state.score
    gl = {"date": pd.Timestamp(g.date), "number": int(gid[-1]), "visitor": gr["visitor"], "vis_league": "",
          "home": gr["home"], "home_league": "", "vis_score": a, "home_score": h, "outs": 0, "completion": "",
          "forfeit": "", "season": int(g.date[:4]), "game_id": gid}
    return pa, gr, gl


def upsert(path: Path, df: pd.DataFrame, retro_ids: set) -> int:
    old = pd.read_parquet(path) if path.exists() else df.iloc[:0]
    keep = old[~old["game_id"].isin(set(df["game_id"]) | retro_ids)]
    out = pd.concat([keep, df], ignore_index=True)
    for c in ("batter", "pitcher", "starter"):
        if c in out:
            out[c] = out[c].astype(str)
    out.to_parquet(path, index=False)
    return len(out)


def site_of(home: str, season: int) -> str:
    g = pd.read_parquet(P / "games.parquet", columns=["home", "site", "season"])
    g = g[g["home"] == home]
    g = g[g["season"] == g["season"].max()]
    return g["site"].value_counts().index[0] if len(g) else home


def season_setup(S: int) -> None:
    """B2E coefficients for season S, fitted exactly as wp.run_b3_eval does (five seasons before S,
    out-of-sample B2 values, Elo pre-game probability) -> models/live/b2e_<S>.json."""
    from wp import b0_table as B0
    from wp import dp
    from wp import elo as E
    from wp import transitions as T
    from wp.b1_model import B1Model
    from wp.run_b3_eval import b2_values, window
    from wp.run_pipeline import load as load_states
    games, gl, st = load_states(P)
    pa = pd.read_parquet(P / "pa.parquet")
    inc = set(games.loc[games["included"], "game_id"])
    seasons_avail = sorted(int(s) for s in pa["season"].unique() if s != 2020)
    ep = E.EloParams(**json.loads(Path("models/elo_params.json").read_text()))
    el = E.run_elo(gl, ep)
    st = st.merge(el[["game_id", "p_pregame"]], on="game_id", how="left")
    st = st[st["p_pregame"].notna()]
    st_seasons = sorted(st["season"].unique())
    parts = []
    for Tt in [s for s in st_seasons if s < S][-5:]:
        tr = T.estimate(pa[pa["game_id"].isin(inc)], window(seasons_avail, Tt))
        res = dp.Engine(30, 1).solve(dp.kernels_from(tr))
        d_ = st.loc[st["season"] == Tt, ["p_pregame", "progress", "home_win", "inning", "half", "outs", "bases",
                                          "diff"]].copy()
        d_["wp_b2"] = b2_values(res, d_["inning"], d_["half"], d_["outs"], d_["bases"], d_["diff"])
        parts.append(d_)
    trb = pd.concat(parts)
    m = B1Model.fit(trb["wp_b2"], trb["p_pregame"], trb["progress"], trb["home_win"])
    out = Path("models/live") / f"b2e_{S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(m.as_dict(), indent=1))
    from wp.manifest import code_files, write_meta
    write_meta([out], [P / "pa.parquet", P / "states.parquet", P / "gamelogs.parquet"], "update_ratings_daily",
               code_files("run_b3_eval", "b1_model", "elo", "dp", "transitions"))
    print(f"{out}: {m.as_dict()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None)
    ap.add_argument("--from-local", action="store_true")
    ap.add_argument("--season-setup", type=int, default=None)
    a = ap.parse_args()
    if a.season_setup:
        season_setup(a.season_setup)
        return
    from wp.live import feeds
    from wp.live.ids import IdMap
    ids = IdMap()
    retro = set(pd.read_parquet(P / "games.parquet", columns=["game_id"])["game_id"])
    items = []
    if a.from_local:
        for pk, e in feeds.index().items():
            items.append((feeds.load_live(pk), e.get("retro_id")))
    else:
        import urllib.request
        d = a.date or (date.today() - timedelta(days=1)).isoformat()
        ua = {"User-Agent": "baseball_diamond-wp/1.0 (personal, non-commercial)"}
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={d}&gameType=R,F,D,L,W"
        sched = json.load(urllib.request.urlopen(urllib.request.Request(url, headers=ua), timeout=60))
        for dt in sched.get("dates", []):
            for g in dt["games"]:
                if g["status"]["abstractGameState"] != "Final":
                    continue
                f = json.load(urllib.request.urlopen(urllib.request.Request(
                    f"https://statsapi.mlb.com/api/v1.1/game/{g['gamePk']}/feed/live", headers=ua), timeout=60))
                from wp.live.ids import team_code
                n = 0 if g.get("doubleHeader", "N") == "N" else int(g.get("gameNumber", 1))
                items.append((f, f"{team_code(g['teams']['home']['team']['id'])}{d.replace('-', '')}{n}"))
    pas, grs, gls = [], [], []
    for feed, gid in items:
        if gid in retro:
            continue                      # already in Retrosheet
        r = convert(feed, ids, gid)
        if r is None:
            continue
        pa, gr, gl = r
        gr["site"] = site_of(gr["home"], gr["season"])
        pas.append(pa)
        grs.append(gr)
        gls.append(gl)
    if not pas:
        print("no new final games")
        return
    pa = pd.concat(pas, ignore_index=True)
    gm = pd.DataFrame(grs)
    gl = pd.DataFrame(gls)
    n1 = upsert(P / "pa_live.parquet", pa, retro)
    n2 = upsert(P / "games_live.parquet", gm, retro)
    n3 = upsert(P / "gamelogs_live.parquet", gl, retro)
    print(f"{len(gm)} games added; pa_live {n1} rows, games_live {n2}, gamelogs_live {n3}; "
          f"unmapped players {len(ids.unmapped)}")


if __name__ == "__main__":
    main()
