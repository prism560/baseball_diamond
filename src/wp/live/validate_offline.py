"""Offline-match check: replayed per-point values vs the offline evaluation (tolerance 1e-6).

Matching points to offline rows (row i = i-th PA of the game, j = pitch index in the PA):
  PA start    offline PA row i  <-> the last start/sub point of PA i before any pitch or runner event
  pitch j>=1  offline (i, j)    <-> the 'pitch' point after the j-th pitch (count before pitch j+1)
  pitch j=0   offline (i, 0)    <-> the last 'pre' point of PA i (runner events before the first pitch)
Classes excluded from the strict check (reported separately, intended differences):
  after_runner_event     pitches after an in-PA runner event: live re-evaluates at the actual
                         state, offline keeps the first-pitch state
  mid_pa_pitching_change live uses the pitcher on the mound, offline the pitcher who ended the PA
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MODELS = ("B3", "B2E", "AVG")


def _val(v, m):
    x = v[m] if isinstance(v, dict) else getattr(v, m)
    return np.nan if x is None else float(x)


def compare(off: dict, points: list, values: list, rows: list) -> pd.DataFrame:
    """One record per matched point: kind, class, offline and live values per model."""
    pa_pt: dict = {}
    recs = []
    for p, v in zip(points, values):
        i = p["pa_i"]
        final_pitcher = rows[i]["pitcher"]
        cls = "mid_pa_pitching_change" if p["pitcher"] != final_pitcher else "strict"
        if p["phase"] in ("start", "sub") and p["n_pitch"] == 0 and not any(
                q["phase"] == "pre" and q["pa_i"] == i for q in points[:0]):
            if (p["outs"], p["bases"], p["home"] - p["away"]) == (rows[i]["outs0"], rows[i]["bases0"], rows[i]["diff0"]):
                pa_pt[i] = (p, v, cls)
        elif p["phase"] == "pitch":
            c = "after_runner_event" if p["mid"] else cls
            recs.append(("pitch", i, p["n_pitch"], c, p, v))
        elif p["phase"] == "pre":
            recs.append(("pre", i, 0, cls, p, v))
    out = []
    opa = off["pa"].set_index("i")
    for i, (p, v, cls) in pa_pt.items():
        if i not in opa.index:
            continue
        o = opa.loc[i]
        out.append({"kind": "pa", "i": i, "j": -1, "class": cls,
                    **{f"off_{m}": float(o[m]) for m in MODELS}, **{f"live_{m}": _val(v, m) for m in MODELS}})
    opt = off["pitch"].set_index(["i", "j"])
    # keep only the last 'pre' point of each PA (state at the first pitch)
    last_pre = {}
    for r in recs:
        if r[0] == "pre":
            last_pre[r[1]] = r
    for r in [x for x in recs if x[0] == "pitch"] + list(last_pre.values()):
        kind, i, j, cls, p, v = r
        if (i, j) not in opt.index:
            continue
        o = opt.loc[(i, j)]
        out.append({"kind": "pitch", "i": i, "j": j, "class": cls,
                    **{f"off_{m}": float(o[m]) for m in MODELS}, **{f"live_{m}": _val(v, m) for m in MODELS}})
    df = pd.DataFrame(out)
    for m in MODELS:
        df[f"d_{m}"] = df[f"live_{m}"] - df[f"off_{m}"]
    return df


def summarize(df: pd.DataFrame, tol: float = 1e-6) -> dict:
    s = {}
    for (kind, cls), d in df.groupby(["kind", "class"]):
        rec = {"n": int(len(d))}
        for m in MODELS:
            a = d[f"d_{m}"].abs()
            rec[m] = {"max_abs": float(a.max()), "mean_abs": float(a.mean()), "n_over_tol": int((a > tol).sum()),
                      "share_gt_2.5pp": float((a > 0.025).mean()), "n_nan": int(d[f"d_{m}"].isna().sum())}
        s[f"{kind}:{cls}"] = rec
    return s
