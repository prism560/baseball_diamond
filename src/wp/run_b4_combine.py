"""B4 step 2: combining B3 with Elo / B2E, using the per-game B3 results already computed.

  B3E   logit = a + b logit(B3) + c logit(Elo) + d logit(Elo) * progress
  AVG   logit = w logit(B3) + (1 - w) logit(B2E)

Walk-forward: for evaluation season S the coefficients (B3E) and the weight w
(AVG, grid 0..1 by 0.05) are fitted on the seasons before S that have B3 values:
S=2023 -> 2022 (one-third game subset, run for this purpose), S=2024 -> 2022-2023,
S=2025 -> 2022-2024.  Same at the pitch level (progress at the pitch's state).

Disagreement analysis: only the PA starts / pitches where |B3 - B2E| > 2.5 pp.
Output: reports/b4_combine.json (+ meta).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluate as EV
from . import states as ST
from .b1_model import B1Model, logit
from .manifest import check_fresh, code_files, write_meta
from .run_b3_eval import strata

EVAL = [2023, 2024, 2025]
LED = 0.025
WGRID = np.round(np.arange(0, 1.0001, 0.05), 2)


def avg(b3, b2e, w):
    z = w * logit(b3) + (1 - w) * logit(b2e)
    return 1 / (1 + np.exp(-z))


def train_seasons(S, available):
    return [s for s in available if s < S]


def fit_combos(train: pd.DataFrame) -> tuple[B1Model, float, dict]:
    b3e = B1Model.fit(train["B3"], train["Elo"], train["progress"], train["home_win"])
    losses = {float(w): float(EV.logloss_vec(avg(train["B3"], train["B2E"], w), train["home_win"]).mean())
              for w in WGRID}
    w = min(losses, key=losses.get)
    return b3e, w, losses


def apply(df: pd.DataFrame, available, level: str) -> tuple[pd.DataFrame, dict]:
    out, info = [], {}
    for S in EVAL:
        tr = df[df["season"].isin(train_seasons(S, available))]
        te = df[df["season"] == S].copy()
        b3e, w, losses = fit_combos(tr)
        te["B3E"] = b3e.predict(te["B3"], te["Elo"], te["progress"])
        te["AVG"] = avg(te["B3"], te["B2E"], w)
        info[str(S)] = {"train_seasons": sorted(int(x) for x in tr["season"].unique()), "train_rows": int(len(tr)),
                        "b3e_coef": b3e.as_dict(), "avg_weight_on_B3": w}
        out.append(te)
    return pd.concat(out, ignore_index=True), info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--reports", type=Path, default=Path("reports"))
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    ins = [a.processed / "b3_eval_pa.parquet", a.processed / "b3_eval_pitch.parquet"]
    for f in ins:
        check_fresh(f)
    pa = pd.read_parquet(ins[0])
    pt = pd.read_parquet(ins[1])
    pa["Elo"] = pa["p_pregame"]
    pt["Elo"] = pt["p_pregame"]
    pt["progress"] = ST.progress(pt["inning"].to_numpy(), pt["half"].to_numpy(), pt["outs1"].to_numpy())
    avail = sorted(int(s) for s in pa["season"].unique())
    res = {"seasons_available": avail,
           "games_per_season": pa.groupby("season")["game_id"].nunique().to_dict()}
    pa_ev, res["fit_pa"] = apply(pa, avail, "pa")
    pt_ev, res["fit_pitch"] = apply(pt, avail, "pitch")
    nb = a.n_boot
    SD = strata(pa_ev)
    models = ["B2E", "B3", "B3E", "AVG", "B1"]
    res["pa_vs_B2E"] = EV.block_bootstrap(pa_ev, models, n_boot=nb, seed=11, strata=SD).to_dict("records")
    res["pa_vs_B3"] = EV.block_bootstrap(pa_ev, ["B3", "B3E", "AVG"], n_boot=nb, seed=12, strata=SD).to_dict("records")
    res["pa_vs_B1"] = EV.block_bootstrap(pa_ev, ["B1", "B3E", "AVG"], n_boot=nb, seed=13, strata=SD).to_dict("records")
    res["pitch_vs_B2E"] = EV.block_bootstrap(pt_ev, models, n_boot=nb, seed=14, strata=SD).to_dict("records")
    res["pitch_vs_B3"] = EV.block_bootstrap(pt_ev, ["B3", "B3E", "AVG"], n_boot=nb, seed=15, strata=SD).to_dict("records")
    # disagreement analysis
    dis = {}
    for name, df in (("pa_start", pa_ev), ("pitch", pt_ev)):
        m = (df["B3"] - df["B2E"]).abs() > LED
        sub = df[m].copy()
        sub["B3_higher"] = sub["B3"] > sub["B2E"]
        st = {"all disagreements": lambda d: np.ones(len(d), bool),
              "B3 above B2E": lambda d: d["B3_higher"].to_numpy(),
              "B3 below B2E": lambda d: ~d["B3_higher"].to_numpy(),
              "starter pitching": lambda d: d["starter_in"].to_numpy(),
              "bullpen pitching": lambda d: ~d["starter_in"].to_numpy(),
              "inning>=7": lambda d: d["inning"].to_numpy() >= 7}
        dis[name] = {"share": float(m.mean()), "n": int(m.sum()),
                     "boot": EV.block_bootstrap(sub, ["B2E", "B3", "AVG", "B3E"], n_boot=nb, seed=21, strata=st)
                     .to_dict("records"),
                     "calibration": {k: EV.calibration(sub[k], sub["home_win"]).reset_index()
                                     .rename(columns={"index": "bin"}).to_dict("records") for k in ("B2E", "B3")},
                     "mean_pred": {k: float(sub[k].mean()) for k in ("B2E", "B3")},
                     "obs_rate": float(sub["home_win"].mean())}
    res["disagreement"] = dis
    res["n_games_eval"] = int(pa_ev["game_id"].nunique())
    o = a.reports / "b4_combine.json"
    o.write_text(json.dumps(res, indent=1, default=float))
    write_meta([o], ins, "run_b4_combine", code_files("run_b4_combine", "b1_model", "evaluate", "run_b3_eval"))
    print(json.dumps({"fit_pa": res["fit_pa"]}, indent=1))


if __name__ == "__main__":
    main()
