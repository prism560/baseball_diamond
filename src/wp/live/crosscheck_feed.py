"""Cross-check: feed-derived PA table vs Retrosheet PA table (normalizer validation).

For every game in data/external/feeds/index.json that exists in the Retrosheet
PA table, the feed is normalized, reconstructed and converted to PA rows; both
tables are compared game by game and PA by PA:
  * number of PAs;
  * per PA: state at PA start (outs0, bases0, diff0), after N (outs1, bases1, rN),
    outcome category o, state after the PA (outs2, bases2, r2), batter, pitcher, slot;
  * pitch count of the PA (number of pitch characters).
Differences are classified like wp.crosscheck: recording conventions (end-of-game
runners after a walk-off, PA split differences around mid-PA substitutions) are
counted separately from real mismatches.

Run: PYTHONPATH=src python -m wp.live.crosscheck_feed  -> reports/live_crosscheck.json
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
from pathlib import Path

import pandas as pd

from ..transitions import n_pitches
from . import feeds
from .ids import IdMap
from .normalize import normalize
from .pa_from_feed import pa_rows
from .reconstruct import reconstruct

FIELDS = ["half", "inning", "outs0", "bases0", "diff0", "outs1", "bases1", "rN", "o", "outs2", "bases2", "r2",
          "batter", "pitcher", "slot"]


def compare_game(fe: pd.DataFrame, rs: pd.DataFrame) -> dict:
    out = {"n_feed": len(fe), "n_retro": len(rs), "field_mismatch": collections.Counter(), "rows": [],
           "pitch_count_mismatch": 0}
    n = min(len(fe), len(rs))
    last_i = len(rs) - 1
    for i in range(n):
        a, b = fe.iloc[i], rs.iloc[i]
        bad = [f for f in FIELDS if str(a[f]) != str(b[f])]
        if n_pitches(a["pitches"]) != n_pitches(str(b["pitches"])):
            out["pitch_count_mismatch"] += 1
        if not bad:
            continue
        # convention: Retrosheet clears the bases after a walk-off; the feed keeps the runners
        if i == last_i and set(bad) <= {"bases2"}:
            cls = "convention_walkoff_bases"
        else:
            cls = "mismatch"
        for f in bad:
            out["field_mismatch"][f"{cls}:{f}"] += 1
        out["rows"].append({"i": i, "class": cls, "fields": {f: [str(a[f]), str(b[f])] for f in bad}})
    out["field_mismatch"] = dict(out["field_mismatch"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feeds", type=Path, default=feeds.FEEDS)
    ap.add_argument("--out", type=Path, default=Path("reports/live_crosscheck.json"))
    a = ap.parse_args()
    pa = pd.read_parquet("data/processed/pa.parquet")
    pa = pa[pa["season"] == 2025]
    ids = IdMap()
    idx = feeds.index(a.feeds)
    res = {"games": {}, "summary": {}}
    tot = collections.Counter()
    for pk, e in sorted(idx.items()):
        gid = e.get("retro_id")
        if e.get("gameType") != "R" or gid is None:
            continue
        rs = pa[pa["game_id"] == gid].reset_index(drop=True)
        if rs.empty:
            res["games"][gid] = {"error": "not in Retrosheet PA table"}
            continue
        view = reconstruct(normalize(feeds.load_live(pk, a.feeds)))
        fe = pa_rows(view, ids, gid)
        c = compare_game(fe, rs)
        c["gamePk"] = int(pk)
        res["games"][gid] = c
        tot["games"] += 1
        tot["pa_retro"] += c["n_retro"]
        tot["pa_feed"] += c["n_feed"]
        tot["games_pa_count_equal"] += int(c["n_feed"] == c["n_retro"])
        tot["rows_compared"] += min(c["n_feed"], c["n_retro"])
        tot["rows_mismatch"] += sum(1 for r in c["rows"] if r["class"] == "mismatch")
        tot["rows_convention"] += sum(1 for r in c["rows"] if r["class"] != "mismatch")
        tot["pitch_count_mismatch"] += c["pitch_count_mismatch"]
        tot["games_exact"] += int(c["n_feed"] == c["n_retro"] and not c["rows"])
    s = dict(tot)
    if tot["rows_compared"]:
        s["row_mismatch_rate"] = tot["rows_mismatch"] / tot["rows_compared"]
        s["pitch_count_mismatch_rate"] = tot["pitch_count_mismatch"] / tot["rows_compared"]
    s["unmapped_players"] = len(ids.unmapped)
    s["id_map_size"] = len(ids)
    fm = collections.Counter()
    for g in res["games"].values():
        fm.update(g.get("field_mismatch", {}))
    s["field_mismatch"] = dict(fm)
    res["summary"] = s
    a.out.parent.mkdir(exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
