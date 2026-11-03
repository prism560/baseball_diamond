"""B4 step 1: B3 matchup fixes evaluated at the plate-appearance level.

Variants (11-category multiclass log loss per PA, 2023-2025 included games):
  A  B3 q: log5, ratings with home-park effects baked in          (baseline)
  B  log5 + park-neutral ratings                                   (1.1)
  C  multinomial logistic (league-q offset, L2 toward log5)        (1.2)
  D  C on park-neutral ratings                                     (q' = 1.1 + 1.2)

For evaluation season S the logistic coefficients are fitted on the three
seasons before S (2020 excluded); the L2 strength is chosen by fitting on the
first two of them and scoring the third.  Rating hyper-parameters are those of
the B3 checkpoint (tuned on 2019/2021/2022).
Checks: HR top-decile calibration, home/away calibration, Coors Field (DEN02).
Output: reports/b4_q.json (+ meta).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import matchup as MU
from . import matchup_v2 as M2
from . import ratings as RT
from . import transitions as T
from .manifest import check_fresh, code_files, write_meta
from .run_b3_check import game_boot, state_q, window

EVAL = [2023, 2024, 2025]
LAMS = [1e-4, 1e-3, 1e-2, 1e-1]


def features(F: dict, p: RT.RatingParams, env: np.ndarray) -> np.ndarray:
    """[n, NF, NR] per-category log-odds deviations vs OUT."""
    L = F["L"]
    rb = RT.shrink(F["Cb"], F["Nb"], L, p.kappa_bat)
    rp = RT.shrink(F["Cp"], F["Np"], L, p.kappa_pit)
    prior = rb * F["LP"]
    prior /= prior.sum(1, keepdims=True)
    rbh = RT.shrink(F["Cbh"], F["Nbh"], prior, p.platoon_k * p.kappa_bat)
    feats = [np.log(rb / L), np.log(rp / L), F["park"], np.log(rbh / rb), env]
    X = np.stack([f - f[:, [RT.REF]] for f in feats], axis=1)
    return X


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--reports", type=Path, default=Path("reports"))
    ap.add_argument("--max-train", type=int, default=450000)
    a = ap.parse_args()
    ins = [a.processed / "pa.parquet", a.processed / "games.parquet", a.raw / "download_manifest.json",
           Path("reports/b3_check.json")]
    for f in ins:
        check_fresh(f)
    t0 = time.time()
    tm = {}
    pa = pd.read_parquet(ins[0])
    pa = pa[pa["season"] >= 2012]
    games = pd.read_parquet(ins[1])
    hands = RT.load_hands(a.raw, 2008, 2025)
    d = RT.pa_frame(pa, games, hands)
    cp = json.loads(Path("reports/b3_check.json").read_text())["params"]
    p = RT.RatingParams(cp["kappa_bat"], cp["kappa_pit"], cp["platoon_k"])
    t = time.time()
    F = {"std": RT.build_features(d), "neu": RT.build_features(d, park_neutral=True)}
    tm["features_s"] = round(time.time() - t, 1)
    seasons_avail = sorted(int(s) for s in d["season"].unique())
    years = sorted({y for S in EVAL for y in window(seasons_avail, S, 3)} | set(EVAL))
    qs, Lw = state_q(d, pa, seasons_avail, years)
    env = MU.env_offset(F["std"]["L"], np.where(np.isnan(Lw), F["std"]["L"], Lw))
    X = {k: features(v, p, env) for k, v in F.items()}
    o = d["o"].to_numpy()
    r = d["r"].to_numpy()
    inc = d["included"].to_numpy()
    seas = d["season"].to_numpy()
    rng = np.random.default_rng(0)
    pred = {k: np.full((len(d), T.NO), np.nan) for k in "ABCD"}
    coefs = {}
    for S in EVAL:
        mS = (seas == S) & inc
        log5 = M2.LogitMatchup.log5()
        pred["A"][mS] = log5.predict(qs[mS], X["std"][mS])
        pred["B"][mS] = log5.predict(qs[mS], X["neu"][mS])
        trn = window(seasons_avail, S, 3)
        for key, xk in (("C", "std"), ("D", "neu")):
            def rows(sea):
                m = np.isin(seas, sea) & inc & (r >= 0)
                idx = np.flatnonzero(m)
                if len(idx) > a.max_train:
                    idx = np.sort(rng.choice(idx, a.max_train, replace=False))
                return idx
            fit_idx, val_idx = rows(trn[:-1]), rows(trn[-1:])
            val = {}
            for lam in LAMS:
                mdl = M2.fit(qs[fit_idx], X[xk][fit_idx], r[fit_idx], lam)
                pv = mdl.predict(qs[val_idx], X[xk][val_idx])
                val[lam] = float(M2.multiclass_logloss(pv, o[val_idx]).mean())
            lam = min(val, key=val.get)
            all_idx = rows(trn)
            mdl = M2.fit(qs[all_idx], X[xk][all_idx], r[all_idx], lam)
            pred[key][mS] = mdl.predict(qs[mS], X[xk][mS])
            coefs[f"{key}_{S}"] = {"lambda": lam, "val": {str(k): v for k, v in val.items()},
                                   "beta": {RT.RCATS[i]: dict(zip(M2.FEATURES, mdl.beta[i].round(3).tolist()))
                                            for i in M2.NONREF},
                                   "gamma": dict(zip([RT.RCATS[i] for i in M2.NONREF],
                                                     mdl.gamma[M2.NONREF].round(3).tolist()))}
        print(S, {k: coefs[f"{k}_{S}"]["lambda"] for k in "CD"}, flush=True)

    ev = np.isin(seas, EVAL) & inc
    gid = d["game_id"].to_numpy()[ev]
    loss = {k: M2.multiclass_logloss(pred[k][ev], o[ev]) for k in "ABCD"}
    res = {"n_pa": int(ev.sum()), "logloss": {k: float(v.mean()) for k, v in loss.items()}, "vs_A": {},
           "per_season": {}}
    for k in "BCD":
        est, lo, hi = game_boot(loss[k] - loss["A"], gid, seed=7)
        res["vs_A"][k] = {"diff": est, "ci": [lo, hi]}
    est, lo, hi = game_boot(loss["D"] - loss["C"], gid, seed=8)
    res["D_vs_C"] = {"diff": est, "ci": [lo, hi]}
    for S in EVAL:
        m = seas[ev] == S
        res["per_season"][str(S)] = {k: float(loss[k][m].mean()) for k in "ABCD"}

    # calibration checks
    def calib(pk, mask, cat, bins=10):
        j = T.O_IDX[cat]
        pr = pred[pk][ev][mask][:, j]
        y = (o[ev][mask] == j).astype(float)
        q = np.quantile(pr, np.linspace(0, 1, bins + 1))
        idx = np.clip(np.searchsorted(q, pr, side="right") - 1, 0, bins - 1)
        return [{"bin": b, "n": int((idx == b).sum()), "pred": float(pr[idx == b].mean()),
                 "obs": float(y[idx == b].mean())} for b in range(bins)]

    allm = np.ones(int(ev.sum()), bool)
    half = d["half"].to_numpy()[ev]
    site = d["site"].to_numpy()[ev]
    res["calibration"] = {
        "HR_deciles": {k: calib(k, allm, "HR") for k in "ABCD"},
        "K_deciles": {k: calib(k, allm, "K") for k in "ABCD"},
    }

    def cat_bias(pk, mask):
        """observed - predicted rate per category (home/away, park subsets)."""
        pr = pred[pk][ev][mask]
        out = {}
        for cat in ("K", "BB", "1B", "2B", "HR", "OUT"):
            j = T.O_IDX[cat]
            out[cat] = float((o[ev][mask] == j).mean() - pr[:, j].mean())
        return out

    res["bias_by_split"] = {name: {k: cat_bias(k, m) for k in "ABCD"} for name, m in
                            (("home batting", half == 1), ("away batting", half == 0),
                             ("Coors Field (DEN02)", site == "DEN02"),
                             ("not Coors", site != "DEN02"))}
    res["coefs"] = coefs
    res["timings"] = tm | {"total_s": round(time.time() - t0, 1)}
    out = a.reports / "b4_q.json"
    out.write_text(json.dumps(res, indent=1, default=float))
    write_meta([out], ins, "run_b4_q", code_files("run_b4_q", "matchup_v2", "matchup", "ratings", "transitions",
                                                  "run_b3_check"))
    print(json.dumps({k: res[k] for k in ("logloss", "vs_A", "D_vs_C", "per_season")}, indent=1))


if __name__ == "__main__":
    main()
