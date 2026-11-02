"""Build the processed datasets from raw Retrosheet files.

Outputs (data/processed/):
  games.parquet     one row per regular-season game found in the event files,
                    with game-log result, exclusion flag and reason
  states.parquet    one row per plate-appearance start state (all games,
                    filter with games.included)
  gamelogs.parquet  every regular-season game-log row 1901-..., used by Elo

Game-log field positions (1-based) follow the Retrosheet game-log guide,
https://www.retrosheet.org/gamelogs/glfields.txt :
  1 date, 2 game number, 4 visitor, 5 visitor league, 7 home, 8 home league,
  10 visitor score, 11 home score, 12 length in outs, 14 completion info,
  15 forfeit info.

Exclusion rules (task spec + Retrosheet field meanings):
  * season 2020 (60-game season)
  * scheduled 7-inning games (event ``info,innings,7``: 2020-2021 doubleheaders)
  * non-regular ``info,gametype`` (e.g. tie-breaker playoff games)
  * home team batted first (``info,htbf,true`` / play order) - semantics of
    "home" ambiguous
  * forfeits (game-log field 15 non-empty)
  * suspended / completed on a later date (game-log field 14 non-empty)
  * ties and games called before the result was decided (rain-shortened)
  * parser errors, or parsed final score != game-log final score
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .retro_parse import parse_file

GL_COLS = {0: "date", 1: "number", 3: "visitor", 4: "vis_league", 6: "home",
           7: "home_league", 9: "vis_score", 10: "home_score", 11: "outs",
           13: "completion", 14: "forfeit"}


def read_gamelog(path: Path, season: int) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, usecols=list(GL_COLS), dtype=str,
                     encoding="latin1", keep_default_na=False)
    df = df.rename(columns=GL_COLS)
    for c in ("vis_score", "home_score", "outs"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["season"] = season
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df["number"] = df["number"].replace({"A": "1", "B": "2"})
    df["game_id"] = df["home"] + df["date"].dt.strftime("%Y%m%d") + df["number"]
    return df


def normal_end(inn: int, half: int, outs: int, vis: int, home: int) -> bool:
    """True if the game ended in a decided, complete manner (>= 9 innings)."""
    if vis == home or inn < 9:
        return False
    if half == 1:
        # bottom half: walk-off (home ahead, any outs) or 3 outs with visitor ahead
        return home > vis or outs == 3
    # last play in top half: home team did not need to bat
    return outs == 3 and home > vis


def parse_season(season: int, raw: Path):
    games, states = [], []
    for f in sorted((raw / str(season)).glob("*.EV?")):
        for g in parse_file(f):
            info = g.info
            games.append({
                "game_id": g.game_id, "season": season,
                "innings_sched": int(info.get("innings", "9") or 9),
                "gametype": info.get("gametype", "regular") or "regular",
                "htbf": info.get("htbf", "false") == "true",
                "site": info.get("site", ""),
                "tiebreaker_base": int(info.get("tiebreaker", "0") or 0),
                "parse_error": g.error,
                "p_vis": g.final[0], "p_home": g.final[1],
                "last_inning": g.last_inning, "last_half": g.last_half,
                "last_outs": g.last_outs, "n_states": len(g.states),
            })
            if g.states:
                a = np.asarray(g.states, dtype=np.int16)
                states.append(pd.DataFrame({
                    "game_id": g.game_id, "seq": np.arange(len(a), dtype=np.int16),
                    "inning": a[:, 0], "half": a[:, 1].astype(np.int8),
                    "outs": a[:, 2].astype(np.int8), "bases": a[:, 3].astype(np.int8),
                    "vis_score": a[:, 4], "home_score": a[:, 5]}))
    return season, pd.DataFrame(games), (pd.concat(states, ignore_index=True) if states else None)


def build(raw: Path, out: Path, first: int, last: int, workers: int = 4) -> dict:
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    def gl_path(y):  # mirror uses GL<YYYY>.TXT; retrosheet.org zips contain gl<yyyy>.txt
        for name in (f"GL{y}.TXT", f"gl{y}.txt"):
            if (raw / str(y) / name).exists():
                return raw / str(y) / name
        return None

    gl = pd.concat([read_gamelog(gl_path(y), y) for y in range(first, last + 1) if gl_path(y)],
                   ignore_index=True)
    gl.to_parquet(out / "gamelogs.parquet", index=False)

    years = [y for y in range(first, last + 1) if any((raw / str(y)).glob("*.EV?"))]
    gparts, sparts = [], []
    with cf.ProcessPoolExecutor(workers) as ex:
        for _, g, s in ex.map(parse_season, years, [raw] * len(years)):
            gparts.append(g)
            if s is not None:
                sparts.append(s)
    games = pd.concat(gparts, ignore_index=True)
    states = pd.concat(sparts, ignore_index=True)

    games = games.merge(gl[["game_id", "date", "visitor", "home", "vis_score", "home_score",
                            "outs", "completion", "forfeit"]], on="game_id", how="left")
    games["date"] = games["date"].fillna(pd.to_datetime(games["game_id"].str[3:11], format="%Y%m%d"))
    games["home"] = games["home"].fillna(games["game_id"].str[:3])

    reason = pd.Series("", index=games.index, dtype=object)

    def mark(mask, why):
        m = mask & (reason == "")
        reason[m] = why

    mark(games["season"] == 2020, "season_2020")
    mark(games["innings_sched"] != 9, "scheduled_7_innings")
    mark(games["gametype"] != "regular", "non_regular_gametype")
    mark(games["vis_score"].isna(), "not_in_gamelog")
    mark(games["forfeit"].fillna("") != "", "forfeit")
    mark(games["completion"].fillna("") != "", "suspended_or_completed_later")
    mark(games["htbf"] | games["parse_error"].fillna("").str.contains("backwards"), "home_bats_first")
    mark(games["parse_error"].notna(), "parse_error")
    mark((games["p_vis"] != games["vis_score"]) | (games["p_home"] != games["home_score"]),
         "score_mismatch_vs_gamelog")
    ne = [normal_end(*r) for r in games[["last_inning", "last_half", "last_outs", "p_vis", "p_home"]]
          .itertuples(index=False, name=None)]
    mark(~pd.Series(ne, index=games.index), "tie_or_called_early")
    games["exclude_reason"] = reason
    games["included"] = reason == ""
    games["home_win"] = (games["p_home"] > games["p_vis"]).astype(np.int8)
    games.to_parquet(out / "games_base.parquet", index=False)   # wp.validate adds checks -> games.parquet
    states.to_parquet(out / "states.parquet", index=False)
    summary = {
        "seasons": f"{min(years)}-{max(years)}",
        "games_in_event_files": int(len(games)),
        "games_included": int(games["included"].sum()),
        "excluded_by_reason": games.loc[~games["included"], "exclude_reason"].value_counts().to_dict(),
        "pa_states_total": int(len(states)),
        "gamelog_rows": int(len(gl)),
        "seconds": round(time.time() - t0, 1),
    }
    return summary


def main() -> None:
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--first", type=int, default=1901)
    ap.add_argument("--last", type=int, default=2025)
    a = ap.parse_args()
    s = build(a.raw, a.out, a.first, a.last)
    (a.out / "build_summary.json").write_text(json.dumps(s, indent=1))
    from .manifest import check_fresh, code_files, write_meta
    man = a.raw / "download_manifest.json"
    check_fresh(man)
    write_meta([a.out / f for f in ("games_base.parquet", "states.parquet", "gamelogs.parquet",
                                    "build_summary.json")],
               [man], "games", code_files("games", "retro_parse"))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
