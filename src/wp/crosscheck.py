"""Row-level cross-check of our parser against Retrosheet's official parsed play-by-play.

Input: ``<YYYY>plays.zip`` (one ``<YYYY>plays.csv`` each) downloaded from Retrosheet's
"Parsed play-by-play data" page and committed at the repository root by the user.

Column correspondence follows Retrosheet's crosswalk between bevent / cwevent
(Chadwick) / the parsed "plays" files (table supplied by the user, Retrosheet
download page):

  ours (wp.retro_parse)          plays column      cwevent header
  ---------------------------    --------------    -----------------
  game_id                        gid               GAME_ID
  inning                         inning            INN_CT
  half (0 top / 1 bottom)        top_bot           BAT_HOME_ID
  outs before the play           outs_pre          OUTS_CT
  outs after the play            outs_post         (EVENT_OUTS_CT = outs_post - outs_pre)
  runner ids 1B/2B/3B before     br1_pre..br3_pre  BASE1_RUN_ID..BASE3_RUN_ID
  runner ids 1B/2B/3B after      br1_post..br3_post  (RUN*_DEST_ID / END_BASES_CD)
  visitor / home score before    score_v / score_h AWAY_SCORE_CT / HOME_SCORE_CT
  runs on the play               runs              EVENT_RUNS_CT
  batter (play line)             batter            BAT_ID
  pitcher (fielding team)        pitcher           PIT_ID
  lineup slot of the batter      lp                BAT_LINEUP_ID

Row alignment: within a game, the plays file has one row per ``play`` record;
rows whose event is exactly ``NP`` (no play, substitutions) are dropped, which
matches our per-play table (wp.retro_parse skips NP lines without advances).
Games whose row counts differ after that are reported separately.

Run: PYTHONPATH=src python -m wp.crosscheck
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .retro_parse import parse_file

PLAYS_COLS = ["gid", "event", "inning", "top_bot", "outs_pre", "outs_post", "br1_pre", "br2_pre",
              "br3_pre", "br1_post", "br2_post", "br3_post", "score_v", "score_h", "runs", "batter",
              "pitcher", "lp", "pn", "gametype"]
FIELDS = ["inning", "half", "outs_pre", "outs_post", "bases_pre", "bases_post", "runner_ids_pre",
          "runner_ids_post", "score", "runs", "batter", "pitcher", "lineup_slot"]


def ours_season(year: int, raw: Path) -> pd.DataFrame:
    rows = []
    for f in sorted((raw / str(year)).glob("*.EV?")):
        for g in parse_file(f):
            if g.error:
                continue
            for k, (e, (pre, post)) in enumerate(zip(g.events, g.occ)):
                (pa, inn, half, outs, bases, vis, home, cat, bat, oa, ba, runs, cnt, pit,
                 batter, pitcher, slot, starter) = e
                rows.append((g.game_id, k, inn, half, outs, oa, vis, home, runs, batter, pitcher or "",
                             slot, *[x or "" for x in pre], *[x or "" for x in post]))
    return pd.DataFrame(rows, columns=["gid", "k", "inning", "half", "outs_pre", "outs_post", "score_v",
                                       "score_h", "runs", "batter", "pitcher", "lp", "br1_pre", "br2_pre",
                                       "br3_pre", "br1_post", "br2_post", "br3_post"])


def theirs_season(zpath: Path) -> pd.DataFrame:
    t = pd.read_csv(zpath, usecols=lambda c: c in PLAYS_COLS, dtype=str, keep_default_na=False)
    t = t[t["event"].str.strip() != "NP"].copy()
    t["k"] = t.groupby("gid").cumcount()
    for c in ("inning", "top_bot", "outs_pre", "outs_post", "score_v", "score_h", "runs", "lp"):
        t[c] = pd.to_numeric(t[c], errors="coerce").fillna(-1).astype(int)
    return t


def bits(a, b, c):
    return (a != "").astype(int) + 2 * (b != "").astype(int) + 4 * (c != "").astype(int)


def compare(o: pd.DataFrame, t: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    no, nt = o.groupby("gid").size(), t.groupby("gid").size()
    both = no.index.intersection(nt.index)
    eq = both[(no[both] == nt[both]).to_numpy()]
    summ = {"games_ours": int(len(no)), "games_plays_file": int(len(nt)),
            "games_in_both": int(len(both)), "games_row_count_equal": int(len(eq)),
            "games_row_count_differ": sorted(both.difference(eq).tolist())[:50],
            "plays_file_gametypes": t["gametype"].value_counts().to_dict() if "gametype" in t else {}}
    m = o[o["gid"].isin(eq)].merge(t[t["gid"].isin(eq)], on=["gid", "k"], suffixes=("_o", "_t"))
    after3 = m["outs_post_t"] >= 3
    mis = pd.DataFrame({"gid": m["gid"], "k": m["k"], "event": m["event"]})
    mis["inning"] = m["inning_o"] != m["inning_t"]
    mis["half"] = m["half"] != m["top_bot"]
    mis["outs_pre"] = m["outs_pre_o"] != m["outs_pre_t"]
    mis["outs_post"] = m["outs_post_o"] != m["outs_post_t"]
    mis["bases_pre"] = bits(m["br1_pre_o"], m["br2_pre_o"], m["br3_pre_o"]) != \
        bits(m["br1_pre_t"], m["br2_pre_t"], m["br3_pre_t"])
    mis["bases_post"] = (bits(m["br1_post_o"], m["br2_post_o"], m["br3_post_o"]) !=
                         bits(m["br1_post_t"], m["br2_post_t"], m["br3_post_t"])) & ~after3
    mis["runner_ids_pre"] = (m["br1_pre_o"] != m["br1_pre_t"]) | (m["br2_pre_o"] != m["br2_pre_t"]) | \
        (m["br3_pre_o"] != m["br3_pre_t"])
    mis["runner_ids_post"] = ((m["br1_post_o"] != m["br1_post_t"]) | (m["br2_post_o"] != m["br2_post_t"]) |
                              (m["br3_post_o"] != m["br3_post_t"])) & ~after3
    mis["score"] = (m["score_v_o"] != m["score_v_t"]) | (m["score_h_o"] != m["score_h_t"])
    mis["runs"] = m["runs_o"] != m["runs_t"]
    mis["batter"] = m["batter_o"] != m["batter_t"]
    mis["pitcher"] = m["pitcher_o"] != m["pitcher_t"]
    mis["lineup_slot"] = m["lp_o"] != m["lp_t"]
    for c in ("bases_post", "runner_ids_post"):
        mis[c + "_detail"] = np.where(mis[c], m["br1_post_o"] + "/" + m["br2_post_o"] + "/" + m["br3_post_o"]
                                      + " vs " + m["br1_post_t"] + "/" + m["br2_post_t"] + "/" + m["br3_post_t"], "")
    summ["rows_compared"] = int(len(m))
    summ["mismatch_rate"] = {f: float(mis[f].mean()) for f in FIELDS}
    summ["mismatch_rows"] = {f: int(mis[f].sum()) for f in FIELDS}
    summ["games_with_any_state_mismatch"] = int(mis.loc[mis[["outs_pre", "outs_post", "bases_pre", "bases_post",
                                                            "score", "runs"]].any(axis=1), "gid"].nunique())
    return summ, mis


def pattern(ev: str) -> str:
    """Event text with fielders/bases abstracted, for grouping mismatches."""
    return re.sub(r"[0-9]", "#", ev)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--zips", type=Path, default=Path("."))
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--seasons", type=int, nargs="+", default=[2023, 2024, 2025])
    a = ap.parse_args()
    from .manifest import check_fresh, code_files, write_meta
    games_p = a.processed / "games.parquet"
    check_fresh(games_p)
    games = pd.read_parquet(games_p)[["game_id", "included", "exclude_reason"]]
    zips = [a.zips / f"{y}plays.zip" for y in a.seasons]
    out = {"source": "Retrosheet parsed play-by-play (plays.csv), crosswalk to cwevent per Retrosheet table",
           "seasons": {}}
    allmis = []
    with cf.ProcessPoolExecutor(3) as ex:
        ours = dict(zip(a.seasons, ex.map(ours_season, a.seasons, [a.raw] * len(a.seasons))))
    for y, z in zip(a.seasons, zips):
        t = theirs_season(z)
        s, mis = compare(ours[y], t)
        mis["season"] = y
        out["seasons"][str(y)] = s
        allmis.append(mis)
    mis = pd.concat(allmis, ignore_index=True)
    mis = mis.merge(games, left_on="gid", right_on="game_id", how="left")
    state = ["outs_pre", "outs_post", "bases_pre", "bases_post", "score", "runs"]
    bad = mis[mis[state + ["runner_ids_pre", "runner_ids_post"]].any(axis=1)]
    out["any_state_mismatch_rows"] = int(mis[state].any(axis=1).sum())
    out["any_state_mismatch_rows_in_included_games"] = int((mis[state].any(axis=1) & mis["included"].fillna(False)).sum())
    out["rows_total"] = int(len(mis))
    pats = {}
    for f in ["bases_post", "runner_ids_post", "runner_ids_pre", "outs_post", "lineup_slot", "pitcher", "batter"]:
        x = mis[mis[f]]
        pats[f] = x["event"].map(pattern).value_counts().head(10).to_dict()
    out["top_event_patterns"] = pats
    out["examples"] = bad.head(30)[["gid", "k", "event"] + [c for c in mis.columns if c.endswith("_detail")]] \
        .to_dict(orient="records")
    o = a.processed / "crosscheck_retrosheet.json"
    o.write_text(json.dumps(out, indent=1, default=str))
    mis.to_parquet(a.processed / "crosscheck_mismatches.parquet", index=False)
    write_meta([o, a.processed / "crosscheck_mismatches.parquet"], zips + [games_p], "crosscheck",
               code_files("crosscheck", "retro_parse"))
    print(json.dumps({k: out[k] for k in ("rows_total", "any_state_mismatch_rows",
                                          "any_state_mismatch_rows_in_included_games")}, indent=1))
    for y, s in out["seasons"].items():
        print(y, s["games_row_count_equal"], "/", s["games_in_both"], {k: round(v, 5) for k, v in s["mismatch_rate"].items()})
    print(json.dumps(pats, indent=1)[:3000])


if __name__ == "__main__":
    main()
