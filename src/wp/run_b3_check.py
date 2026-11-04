"""B3 section 4 checkpoint: does the matchup q beat the league q at predicting PA outcomes?

1. Rating hyper-parameters (kappa_bat, kappa_pit, platoon_k) tuned by multiclass
   log loss on 2019, 2021, 2022 (all before the evaluation seasons).
2. On 2023-2025 (included games): multiclass log loss of the league state q
   (B2, window K=10) vs the matchup q, with ablations, game-block bootstrap CI.
3. Calibration of matchup probabilities per category (deciles), to look for
   log5 bias in extreme matchups.
Outputs reports/b3_check.json (+ meta) and data/processed/b3_features.npz
(cached as-of rating arrays for later stages).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import matchup as MU
from . import ratings as RT
from . import transitions as T
from .manifest import check_fresh, code_files, write_meta

K_FIXED = 10
TUNE = [2019, 2021, 2022]
EVAL = [2023, 2024, 2025]


def window(seasons, before, K=K_FIXED):
    return [s for s in seasons if s < before and s != 2020][-K:]


def state_q(d: pd.DataFrame, pa: pd.DataFrame, seasons_avail, years) -> tuple[np.ndarray, np.ndarray]:
    """League q at the PA's first-pitch state and the window's 10-category rates, per row."""
    qs = np.full((len(d), T.NO), np.nan)
    Lw = np.full((len(d), RT.NR), np.nan)
    for S in years:
        m = (d["season"] == S).to_numpy()
        if not m.any():
            continue
        w = window(seasons_avail, S)
        tr = T.estimate(pa, w)
        qs[m] = tr.q[d.loc[m, "half"], d.loc[m, "outs1"], d.loc[m, "bases1"]]
        dw = d[d["season"].isin(w) & (d["r"] >= 0)]
        cnt = np.bincount(dw["r"], minlength=RT.NR) + 1.0
        Lw[m] = cnt / cnt.sum()
    return qs, Lw


def game_boot(loss_diff: np.ndarray, games: np.ndarray, n_boot=1000, seed=0):
    codes, uniq = pd.factorize(games)
    G = len(uniq)
    s = np.bincount(codes, weights=loss_diff, minlength=G)
    n = np.bincount(codes, minlength=G).astype(float)
    W = np.random.default_rng(seed).multinomial(G, np.full(G, 1 / G), size=n_boot).astype(np.float32)
    bs = (W @ s) / (W @ n)
    return float(s.sum() / n.sum()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--reports", type=Path, default=Path("reports"))
    a = ap.parse_args()
    ins = [a.processed / "pa.parquet", a.processed / "games.parquet", a.raw / "download_manifest.json"]
    for f in ins:
        check_fresh(f)
    t0 = time.time()
    tm = {}
    pa = pd.read_parquet(ins[0])
    games = pd.read_parquet(ins[1])
    hands = RT.load_hands(a.raw)
    d = RT.pa_frame(pa, games, hands)
    tm["load_s"] = round(time.time() - t0, 1)
    t = time.time()
    F = RT.build_features(d)
    tm["features_s"] = round(time.time() - t, 1)
    seasons_avail = sorted(int(s) for s in d["season"].unique())
    years = TUNE + EVAL
    t = time.time()
    qs, Lw = state_q(d, pa, seasons_avail, years)
    tm["state_q_s"] = round(time.time() - t, 1)
    env = MU.env_offset(F["L"], np.where(np.isnan(Lw), F["L"], Lw))
    o = d["o"].to_numpy()

    def loss(mask, dev):
        q = MU.combine(qs[mask], dev[mask])
        return MU.multiclass_logloss(q, o[mask])

    # ---- tuning on TUNE seasons ----
    t = time.time()
    tune_m = d["season"].isin(TUNE).to_numpy() & d["included"].to_numpy()
    grid = {}
    for kb in (0.5, 1, 2, 4):
        for kp in (0.5, 1, 2, 4):
            p = RT.RatingParams(kb, kp)
            grid[(kb, kp)] = float(loss(tune_m, RT.deviations(F, p) + env).mean())
    kb, kp = min(grid, key=grid.get)
    pgrid = {}
    for pk in (2, 6, 18):
        p = RT.RatingParams(kb, kp, pk)
        pgrid[pk] = float(loss(tune_m, RT.deviations(F, p) + env).mean())
    pk = min(pgrid, key=pgrid.get)
    best = RT.RatingParams(kb, kp, pk)
    tm["tuning_s"] = round(time.time() - t, 1)

    # ---- evaluation ----
    ev_m = d["season"].isin(EVAL).to_numpy() & d["included"].to_numpy()
    gid = d.loc[ev_m, "game_id"].to_numpy()
    zero = np.zeros((len(d), RT.NR))
    base_loss = loss(ev_m, zero)                         # league q (B2)
    variants = {
        "matchup": RT.deviations(F, best) + env,
        "no_platoon": RT.deviations(F, RT.RatingParams(kb, kp, pk, platoon=False)) + env,
        "no_park": RT.deviations(F, RT.RatingParams(kb, kp, pk, park=False)) + env,
        "no_env": RT.deviations(F, best),
        "env_only": env,
    }
    res = {"league_q_logloss": float(base_loss.mean()), "n_pa": int(ev_m.sum()), "variants": {}}
    for name, dev in variants.items():
        l = loss(ev_m, dev)
        est, lo, hi = game_boot(l - base_loss, gid, seed=1)
        res["variants"][name] = {"logloss": float(l.mean()), "minus_league": est, "ci": [lo, hi]}
    per_season = {}
    for S in EVAL:
        mS = ev_m & (d["season"] == S).to_numpy()
        l0 = loss(mS, zero)
        l1 = loss(mS, variants["matchup"])
        est, lo, hi = game_boot(l1 - l0, d.loc[mS, "game_id"].to_numpy(), seed=S)
        per_season[S] = {"league": float(l0.mean()), "matchup": float(l1.mean()), "diff": est, "ci": [lo, hi]}
    res["per_season"] = per_season

    # ---- calibration (log5 bias in extreme matchups) ----
    q = MU.combine(qs[ev_m], variants["matchup"][ev_m])
    cal = {}
    for cat in ("K", "BB", "HR", "1B", "OUT"):
        j = T.O_IDX[cat]
        pr = q[:, j]
        y = (o[ev_m] == j).astype(float)
        bins = np.quantile(pr, np.linspace(0, 1, 11))
        idx = np.clip(np.searchsorted(bins, pr, side="right") - 1, 0, 9)
        rows = []
        for b in range(10):
            mm = idx == b
            rows.append({"bin": b, "n": int(mm.sum()), "mean_pred": float(pr[mm].mean()),
                         "obs": float(y[mm].mean()),
                         "se": float(np.sqrt(max(y[mm].mean() * (1 - y[mm].mean()), 1e-9) / max(mm.sum(), 1)))})
        cal[cat] = rows
    res["calibration"] = cal
    res["params"] = {"kappa_bat": kb, "kappa_pit": kp, "platoon_k": pk}
    res["tune_grid"] = {f"{k[0]},{k[1]}": v for k, v in grid.items()}
    res["platoon_grid"] = {str(k): v for k, v in pgrid.items()}
    res["timings"] = tm
    res["hands_coverage"] = float(d["batter"].astype(str).isin(hands.index).mean())
    out = a.reports / "b3_check.json"
    out.write_text(json.dumps(res, indent=1, default=float))
    write_meta([out], ins, "run_b3_check", code_files("run_b3_check", "ratings", "matchup", "transitions"))
    print(json.dumps({k: res[k] for k in ("league_q_logloss", "variants", "per_season", "params", "timings")},
                     indent=1, default=float))


if __name__ == "__main__":
    main()
