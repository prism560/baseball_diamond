"""B4 step 4 gate: in-game condition features on top of q' at the PA level.

Models (multinomial logistic with league-q offset, wp.matchup_v2 extended):
  Q      q' features (batter, pitcher, park, platoon, env) on park-neutral ratings
  Q+C    + condition features (cond_dev, pitch count), per-category coefficients
  Q+T    + times-through-the-order feature (league TTO offset of the starter's lap)
  Q+T+C  both - comparing the condition coefficients of Q+C and Q+T+C shows how much
         of the condition effect overlaps with TTO (task: no double counting; the
         hazard is never changed by condition).
Prior strength k of cond_dev is chosen on the last training season.
Gate: Q+C - Q <= -0.002 (overall or on later PAs of the same pitcher, n_prev >= 30)
with a CI below 0.  Output reports/b4_condition.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import condition as CD
from . import matchup as MU
from . import matchup_v2 as M2
from . import ratings as RT
from .manifest import check_fresh, code_files, write_meta
from .run_b3_check import game_boot, state_q, window
from .run_b4_q import EVAL, LAMS, features

KS = [50.0, 100.0, 200.0, 400.0]
GATE = 0.002


def extend(X: np.ndarray, extra: list[np.ndarray]) -> np.ndarray:
    """Append per-row scalar features broadcast to every category (REF column 0)."""
    cols = []
    for e in extra:
        f = np.repeat(np.asarray(e, float)[:, None], RT.NR, 1)
        f[:, RT.REF] = 0.0
        cols.append(f[:, None, :])
    return np.concatenate([X] + cols, axis=1)


def fit_ext(qs, X, r, lam):
    """q' coefficients shrunk toward log5 (1), extra feature coefficients toward 0."""
    return M2.fit(qs, X, r, lam, n_base=M2.NF)


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
    pa = pd.read_parquet(ins[0])
    pa = pa[pa["season"] >= 2012]
    games = pd.read_parquet(ins[1])
    hands = RT.load_hands(a.raw, 2008, 2025)
    d = RT.pa_frame(pa, games, hands)
    cp = json.loads(Path("reports/b3_check.json").read_text())["params"]
    p = RT.RatingParams(cp["kappa_bat"], cp["kappa_pit"], cp["platoon_k"])
    F = RT.build_features(d, park_neutral=True)
    seasons_avail = sorted(int(s) for s in d["season"].unique())
    years = sorted({y for S in EVAL for y in window(seasons_avail, S, 3)} | set(EVAL))
    qs, Lw = state_q(d, pa, seasons_avail, years)
    env = MU.env_offset(F["L"], np.where(np.isnan(Lw), F["L"], Lw))
    Xq = features(F, p, env)
    # condition features (in game order: pa_frame keeps game order within a game)
    CF = CD.condition_features(d)
    # times through the order of the starter: per-category coefficient on the lap (0/1/2)
    o, r = d["o"].to_numpy(), d["r"].to_numpy()
    inc, seas = d["included"].to_numpy(), d["season"].to_numpy()
    starter_in = (d["pitcher"].astype(str) == d["starter"].astype(str)).to_numpy()
    bf = d.groupby(["game_id", "half"], sort=False).cumcount().to_numpy()   # PAs of this side so far
    lapf = np.where(starter_in, np.minimum(bf // 9, 2), 0).astype(float)    # 0/1/2 = lap 1/2/3+
    rng = np.random.default_rng(0)
    pred = {k: np.full((len(d), 11), np.nan) for k in ("Q", "QC", "QT", "QTC")}
    coefs = {}

    def rows(sea):
        m = np.isin(seas, sea) & inc & (r >= 0)
        idx = np.flatnonzero(m)
        if len(idx) > a.max_train:
            idx = np.sort(rng.choice(idx, a.max_train, replace=False))
        return idx

    for S in EVAL:
        trn = window(seasons_avail, S, 3)
        fit_idx, val_idx, all_idx = rows(trn[:-1]), rows(trn[-1:]), rows(trn)
        mS = (seas == S) & inc
        # k for cond_dev and lambda chosen on the last training season
        best = None
        for k in KS:
            cdv = CD.cond_dev(CF, k)
            Xc = extend(Xq, [cdv, CF["n_prev"].to_numpy() / 100])
            for lam in LAMS[:2]:
                mdl = fit_ext(qs[fit_idx], Xc[fit_idx], r[fit_idx], lam)
                v = float(M2.multiclass_logloss(mdl.predict(qs[val_idx], Xc[val_idx]), o[val_idx]).mean())
                if best is None or v < best[0]:
                    best = (v, k, lam)
        _, k, lam = best
        cdv = CD.cond_dev(CF, k)
        pcf = CF["n_prev"].to_numpy() / 100
        variants = {"Q": Xq, "QC": extend(Xq, [cdv, pcf]), "QT": extend(Xq, [lapf]),
                    "QTC": extend(Xq, [lapf, cdv, pcf])}
        for name, Xv in variants.items():
            mdl = fit_ext(qs[all_idx], Xv[all_idx], r[all_idx], lam)
            pred[name][mS] = mdl.predict(qs[mS], Xv[mS])
            if name in ("QC", "QTC"):
                ex = Xv.shape[1] - M2.NF
                coefs[f"{name}_{S}"] = {c: mdl.beta[RT.R_IDX[c], M2.NF:].round(4).tolist()
                                        for c in ("K", "BB", "1B", "HR")} | {"extras": ex}
        coefs[f"k_lambda_{S}"] = {"k": k, "lambda": lam}
        print(S, coefs[f"k_lambda_{S}"], flush=True)

    ev = np.isin(seas, EVAL) & inc
    gid = d["game_id"].to_numpy()[ev]
    loss = {k: M2.multiclass_logloss(pred[k][ev], o[ev]) for k in pred}
    later = CF["n_prev"].to_numpy()[ev] >= 30
    res = {"n_pa": int(ev.sum()), "n_pa_later_same_pitcher": int(later.sum()),
           "logloss": {k: float(v.mean()) for k, v in loss.items()}, "diff": {}}
    for name, (x, y) in {"QC-Q": ("QC", "Q"), "QT-Q": ("QT", "Q"), "QTC-QT": ("QTC", "QT")}.items():
        e1 = game_boot(loss[x] - loss[y], gid, seed=3)
        e2 = game_boot((loss[x] - loss[y])[later], gid[later], seed=4)
        res["diff"][name] = {"all": {"diff": e1[0], "ci": [e1[1], e1[2]]},
                             "later_PAs_same_pitcher": {"diff": e2[0], "ci": [e2[1], e2[2]]}}
    g = res["diff"]["QC-Q"]
    res["gate_pass"] = bool((g["all"]["diff"] <= -GATE and g["all"]["ci"][1] < 0) or
                            (g["later_PAs_same_pitcher"]["diff"] <= -GATE and g["later_PAs_same_pitcher"]["ci"][1] < 0))
    res["gate_threshold"] = GATE
    res["coefs"] = coefs
    res["seconds"] = round(time.time() - t0, 1)
    out = a.reports / "b4_condition.json"
    out.write_text(json.dumps(res, indent=1, default=float))
    write_meta([out], ins, "run_b4_condition", code_files("run_b4_condition", "condition", "matchup_v2",
                                                          "ratings", "run_b4_q", "run_b3_check"))
    print(json.dumps({k: res[k] for k in ("logloss", "diff", "gate_pass")}, indent=1))


if __name__ == "__main__":
    main()
