"""Section 0: parser validation beyond final scores.

Independent checks of the parsed mid-game state, per game:
  1. line score   - runs per half-inning vs game-log line scores (fields 20, 21)
  2. three outs   - every half-inning but the last ends with exactly 3 outs;
                    the last one may end early only on a walk-off
  3. LOB          - runners left on base at the end of each half-inning, summed
                    per team, vs game-log team LOB (fields 38, 66).  This checks
                    the base state at every third out against an independent
                    source.  A walk-off half's remaining runners are excluded
                    (variant counted both ways and reported).
  4. continuity   - parser-independent identity checks: the batter of every new
                    PA must be the next lineup slot (batting-order continuity),
                    a pinch runner must replace a player who is on base, and the
                    counters of parser repairs (pushes of unmentioned runners,
                    implied moves naming empty bases) must be zero.
  5. box counts   - team H, HR, BB, IBB, K, HBP, SB, CS vs game log (informational)
  6. pitch seq.   - see retro_parse.check_pitches (seasons with pitch data)
  7. Chadwick     - row-level cross-check if data/external/chadwick/ exists

Game-log field positions (1-based; Retrosheet glfields.txt), confirmed on the
data by the AB/H/.../LOB block layout: visitor offense 22-38 (H 23, HR 26,
HBP 30, BB 31, IBB 32, K 33, SB 34, CS 35, LOB 38), home offense 50-66
(H 51, HR 54, HBP 58, BB 59, IBB 60, K 61, SB 62, CS 63, LOB 66).

Outputs: data/processed/validation.parquet (per game), events.parquet
(per play record, seasons >= --events-from), pitch_viol.parquet,
validation_summary.json.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .retro_parse import EVENT_COLS, parse_file

GL_FIELDS = {0: "date", 1: "number", 6: "home", 19: "line_vis", 20: "line_home"}
BOX = {"H": (22, 50), "HR": (25, 53), "HBP": (29, 57), "BB": (30, 58), "IBB": (31, 59),
       "K": (32, 60), "SB": (33, 61), "CS": (34, 62), "LOB": (37, 65)}
HIT_CATS = ("1B", "2B", "3B", "HR")


def read_gl(path: Path) -> pd.DataFrame:
    cols = dict(GL_FIELDS)
    for k, (v, h) in BOX.items():
        cols[v] = f"v_{k}"
        cols[h] = f"h_{k}"
    df = pd.read_csv(path, header=None, usecols=list(cols), dtype=str, encoding="latin1",
                     keep_default_na=False).rename(columns=cols)
    df["number"] = df["number"].replace({"A": "1", "B": "2"})
    df["game_id"] = df["home"] + df["date"] + df["number"]
    for k in BOX:
        for p in "vh":
            df[f"{p}_{k}"] = pd.to_numeric(df[f"{p}_{k}"], errors="coerce")
    return df


_LS = re.compile(r"\((\d+)\)|(\d)|([xX])")


def parse_linescore(s: str) -> list | None:
    if not s:
        return None
    out = []
    for m in _LS.finditer(s):
        if m.group(1):
            out.append(int(m.group(1)))
        elif m.group(2):
            out.append(int(m.group(2)))
        else:
            out.append(None)
    return out


def season_validate(year: int, raw: Path, events_from: int):
    gl_path = next((raw / str(year) / n for n in (f"GL{year}.TXT", f"gl{year}.txt")
                    if (raw / str(year) / n).exists()), None)
    gl = read_gl(gl_path).set_index("game_id") if gl_path else pd.DataFrame()
    rows, ev_parts, pv_rows = [], [], []
    patterns = Counter()
    for f in sorted((raw / str(year)).glob("*.EV?")):
        for g in parse_file(f):
            r = {"game_id": g.game_id, "season": year, "parse_error": g.error is not None}
            r.update({f"d_{k}": v for k, v in g.diag.items()})
            r["n_pitch_viol"] = len(g.pitch_viol)
            r["n_pa"] = len(g.states)
            for _, tag in g.pitch_viol:
                r[f"pv_{tag}"] = r.get(f"pv_{tag}", 0) + 1
            last = max(g.half_outs) if g.half_outs else None
            # 2. three outs
            bad3 = 0
            for key, o in g.half_outs.items():
                if key == last:
                    inn, half = key
                    walkoff = half == 1 and g.final[1] > g.final[0]
                    if o != 3 and not walkoff:
                        bad3 += 1
                elif o != 3:
                    bad3 += 1
            r["three_out_violations"] = bad3
            if g.game_id in gl.index:
                row = gl.loc[g.game_id]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                # 1. line score
                mism = 0
                avail = True
                for team, col in ((0, "line_vis"), (1, "line_home")):
                    ls = parse_linescore(row[col])
                    if ls is None:
                        avail = False
                        continue
                    parsed = {inn: v for (inn, h), v in g.half_runs.items() if h == team}
                    for i, v in enumerate(ls, start=1):
                        pv = parsed.get(i)
                        if v is None:
                            if pv not in (None, 0):
                                mism += 1
                        elif pv is None or pv != v:
                            mism += 1
                    if max(parsed, default=0) > len(ls):
                        mism += 1
                r["linescore_available"] = avail
                r["linescore_mismatch_halves"] = mism if avail else np.nan
                # 3. LOB
                for team, p in ((0, "v"), (1, "h")):
                    lob_all = sum(v for (inn, h), v in g.half_lob.items() if h == team)
                    lob_ex = sum(v for (inn, h), v in g.half_lob.items()
                                 if h == team and not ((inn, h) == last and g.half_outs[(inn, h)] != 3))
                    gl_lob = row[f"{p}_LOB"]
                    r[f"{p}_lob_gl"] = gl_lob
                    r[f"{p}_lob_parsed"] = lob_ex
                    r[f"{p}_lob_parsed_incl_last"] = lob_all
                    # 5. box counts
                    for k in BOX:
                        if k == "LOB":
                            continue
                        if k == "H":
                            val = sum(g.counts.get((team, c), 0) for c in HIT_CATS)
                        elif k == "BB":   # game-log BB includes intentional walks
                            val = g.counts.get((team, "BB"), 0) + g.counts.get((team, "IBB"), 0)
                        elif k == "CS":
                            val = g.counts.get((team, "CS"), 0) + g.counts.get((team, "POCS"), 0)
                        else:
                            val = g.counts.get((team, k), 0)
                        r[f"{p}_{k}_diff"] = val - row[f"{p}_{k}"] if not pd.isna(row[f"{p}_{k}"]) else np.nan
            rows.append(r)
            if year >= events_from and g.events:
                e = pd.DataFrame(g.events, columns=EVENT_COLS)
                e.insert(0, "game_id", g.game_id)
                ev_parts.append(e)
            for pa, tag in g.pitch_viol:
                pv_rows.append((g.game_id, year, pa, tag))
    ev = pd.concat(ev_parts, ignore_index=True) if ev_parts else None
    return pd.DataFrame(rows), ev, pv_rows


def chadwick_crosscheck(ext: Path, events: pd.DataFrame | None) -> dict:
    files = sorted(ext.glob("*.csv")) if ext.exists() else []
    if not files:
        return {"status": "미실시 (data/external/chadwick/ 에 cwevent 출력 없음)"}
    # cwevent default fields: GAME_ID, ..., INN_CT(2), BAT_HOME_ID(3), OUTS_CT(4), ..., base runners
    # are checked by START_BASES_CD (field 29 in -f 0-96 output) when present.
    res = {"status": "실시", "files": [f.name for f in files]}
    try:
        cw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        need = {"GAME_ID", "INN_CT", "BAT_HOME_ID", "OUTS_CT", "START_BASES_CD"}
        if not need.issubset(cw.columns):
            res["status"] = f"형식 불일치: 필요한 열 {sorted(need)} (cwevent -n 헤더 포함 출력 필요)"
            return res
        if events is None:
            res["status"] = "이벤트 표 없음"
            return res
        e = events.copy()
        e["k"] = e.groupby("game_id").cumcount()
        cw["k"] = cw.groupby("GAME_ID").cumcount()
        m = e.merge(cw, left_on=["game_id", "k"], right_on=["GAME_ID", "k"], how="inner")
        res["rows_compared"] = int(len(m))
        res["outs_mismatch"] = int((m["outs"] != m["OUTS_CT"]).sum())
        res["bases_mismatch"] = int((m["bases"] != m["START_BASES_CD"]).sum())
    except Exception as ex:  # report rather than crash
        res["status"] = f"오류: {ex}"
    return res


VALIDATION_REASONS = [
    ("linescore_mismatch", lambda v: v["linescore_mismatch_halves"].fillna(0) > 0),
    ("three_out_violation", lambda v: v["three_out_violations"] > 0),
    ("lob_mismatch", lambda v: v["lob_mismatch_incl_last"]),
    ("parser_repair", lambda v: (v["d_repair_push"] > 0) | (v["d_stale_implied"] > 0)),
    ("batting_order_break", lambda v: v["d_batting_order_break"] > 0),
    ("pinch_runner_not_on_base", lambda v: v["d_pinch_runner_not_on_base"] > 0),
    ("pitch_sequence_violation",
     lambda v: (v["n_pitch_viol"] - v.get("pv_no_or_unknown_pitches", 0).fillna(0)) > 0),
]
ERAS = [(1908, 1949), (1950, 1987), (1988, 1999), (2000, 2019), (2021, 2025)]


def apply_exclusions(processed: Path, val: pd.DataFrame) -> dict:
    """Add validation failures to games.parquet as exclusion reasons (only for
    games not already excluded) and return summary tables."""
    games = pd.read_parquet(processed / "games.parquet")
    for c in ("exclude_reason_pre_validation",):
        if c not in games:
            games[c] = games["exclude_reason"]
    games["exclude_reason"] = games["exclude_reason_pre_validation"]
    v = val.set_index("game_id").reindex(games["game_id"]).reset_index()
    out = {"validation_flags_among_previously_included": {}}
    base_inc = games["exclude_reason"] == ""
    for name, fn in VALIDATION_REASONS:
        flag = fn(v).fillna(False).to_numpy(bool)
        out["validation_flags_among_previously_included"][name] = int((flag & base_inc).sum())
        m = flag & (games["exclude_reason"] == "")
        games.loc[m, "exclude_reason"] = f"validation:{name}"
    games["included"] = games["exclude_reason"] == ""
    games.to_parquet(processed / "games.parquet", index=False)
    any_flag = np.zeros(len(games), bool)
    for _, fn in VALIDATION_REASONS:
        any_flag |= fn(v).fillna(False).to_numpy(bool)
    era_rows = []
    for lo, hi in ERAS:
        m = base_inc & games["season"].between(lo, hi)
        era_rows.append({"era": f"{lo}-{hi}", "games": int(m.sum()), "flagged": int((any_flag & m).sum()),
                         "rate": float((any_flag & m).sum() / max(m.sum(), 1))})
    out["by_era"] = era_rows
    out["excluded_by_reason_after_validation"] = games.loc[~games["included"], "exclude_reason"].value_counts().to_dict()
    out["games_included_after_validation"] = int(games["included"].sum())
    box = {}
    vi = v[base_inc.to_numpy() & (v["season"] >= 2000).to_numpy()]
    for k in BOX:
        if k == "LOB":
            continue
        d = vi[[f"v_{k}_diff", f"h_{k}_diff"]]
        box[k] = float(((d != 0) & d.notna()).any(axis=1).mean())
    out["box_count_mismatch_rate_2000plus"] = box
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--external", type=Path, default=Path("data/external/chadwick"))
    ap.add_argument("--first", type=int, default=1908)
    ap.add_argument("--last", type=int, default=2025)
    ap.add_argument("--events-from", type=int, default=1990)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    years = [y for y in range(a.first, a.last + 1) if any((a.raw / str(y)).glob("*.EV?"))]
    vparts, eparts, pv = [], [], []
    with cf.ProcessPoolExecutor(a.workers) as ex:
        for v, e, p in ex.map(season_validate, years, [a.raw] * len(years), [a.events_from] * len(years)):
            vparts.append(v)
            if e is not None:
                eparts.append(e)
            pv += p
    val = pd.concat(vparts, ignore_index=True).fillna({c: 0 for c in ("d_repair_push", "d_stale_implied",
                                                                       "d_implicit_force",
                                                                       "d_batting_order_break",
                                                                       "d_pinch_runner_not_on_base",
                                                                       "d_batter_not_in_lineup")})
    for c in ("d_repair_push", "d_stale_implied", "d_implicit_force", "d_batting_order_break",
              "d_pinch_runner_not_on_base", "d_batter_not_in_lineup"):
        if c not in val:
            val[c] = 0
    val["lob_mismatch"] = ((val["v_lob_parsed"] != val["v_lob_gl"]) | (val["h_lob_parsed"] != val["h_lob_gl"])) \
        & val["v_lob_gl"].notna() & val["h_lob_gl"].notna()
    val["lob_mismatch_incl_last"] = ((val["v_lob_parsed_incl_last"] != val["v_lob_gl"])
                                     | (val["h_lob_parsed_incl_last"] != val["h_lob_gl"])) \
        & val["v_lob_gl"].notna() & val["h_lob_gl"].notna()
    events = pd.concat(eparts, ignore_index=True) if eparts else None
    a.processed.mkdir(parents=True, exist_ok=True)
    val.to_parquet(a.processed / "validation.parquet", index=False)
    if events is not None:
        for c in ("inning", "outs", "bases", "outs_after", "bases_after", "runs", "half"):
            events[c] = events[c].astype(np.int8 if c != "inning" else np.int16)
        events["cat"] = events["cat"].astype("category")
        events.to_parquet(a.processed / "events.parquet", index=False)
    pd.DataFrame(pv, columns=["game_id", "season", "pa", "tag"]).to_parquet(
        a.processed / "pitch_viol.parquet", index=False)
    cc = chadwick_crosscheck(a.external, events)
    summ = {"seconds": round(time.time() - t0, 1), "games": int(len(val)), "chadwick": cc}
    summ.update(apply_exclusions(a.processed, val))
    (a.processed / "validation_summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False))
    print(json.dumps(summ, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
