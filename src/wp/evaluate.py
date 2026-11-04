"""Reusable evaluation utilities for win-probability models.

* log loss, Brier score
* 10-bin calibration (equal-width bins on predicted P(home win))
* game-level block bootstrap: games are resampled with replacement and every
  PA state of a drawn game comes along, because states within one game are
  strongly dependent.
* standard strata: all / inning >= 7 / |diff| <= 2 / both
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-6


def logloss_vec(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def brier_vec(p, y):
    return (np.asarray(p, float) - np.asarray(y, float)) ** 2


def calibration(p, y, bins: int = 10) -> pd.DataFrame:
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    idx = np.clip((p * bins).astype(int), 0, bins - 1)
    df = pd.DataFrame({"bin": idx, "p": p, "y": y})
    out = df.groupby("bin").agg(n=("y", "size"), mean_pred=("p", "mean"), obs_rate=("y", "mean"))
    out = out.reindex(range(bins))
    out.index = [f"{b / bins:.1f}-{(b + 1) / bins:.1f}" for b in range(bins)]
    return out


STRATA = {
    "all": lambda d: np.ones(len(d), bool),
    "inning>=7": lambda d: d["inning"].to_numpy() >= 7,
    "close(|diff|<=2)": lambda d: np.abs(d["diff"].to_numpy()) <= 2,
    "inning>=7 & close": lambda d: (d["inning"].to_numpy() >= 7) & (np.abs(d["diff"].to_numpy()) <= 2),
}


def block_bootstrap(df: pd.DataFrame, models: list[str], y_col: str = "home_win",
                    game_col: str = "game_id", n_boot: int = 1000, seed: int = 0,
                    strata=STRATA) -> pd.DataFrame:
    """Point estimates and 95% percentile CIs for each model, stratum and
    metric, plus paired differences (model - models[0]) on the same draws."""
    rng = np.random.default_rng(seed)
    codes, games = pd.factorize(df[game_col])
    G = len(games)
    W = rng.multinomial(G, np.full(G, 1.0 / G), size=n_boot).astype(np.float32)  # (B, G)
    y = df[y_col].to_numpy(float)
    rows = []
    for sname, fn in strata.items():
        m = fn(df)
        cnt = np.bincount(codes[m], minlength=G).astype(np.float64)
        boot_n = W @ cnt
        per = {}
        for mod in models:
            p = df[mod].to_numpy(float)
            for met, f in (("logloss", logloss_vec), ("brier", brier_vec)):
                s = np.bincount(codes[m], weights=f(p[m], y[m]), minlength=G)
                per[(mod, met)] = (s.sum() / cnt.sum(), (W @ s) / boot_n)
        for (mod, met), (pt, bs) in per.items():
            rows.append({"stratum": sname, "model": mod, "metric": met, "n_states": int(cnt.sum()),
                         "n_games": int((cnt > 0).sum()), "estimate": pt,
                         "ci_low": np.percentile(bs, 2.5), "ci_high": np.percentile(bs, 97.5)})
            base = per[(models[0], met)]
            if mod != models[0]:
                d = bs - base[1]
                rows.append({"stratum": sname, "model": f"{mod}-{models[0]}", "metric": met,
                             "n_states": int(cnt.sum()), "n_games": int((cnt > 0).sum()),
                             "estimate": pt - base[0], "ci_low": np.percentile(d, 2.5),
                             "ci_high": np.percentile(d, 97.5),
                             "p_better": float(np.mean(d < 0))})
    return pd.DataFrame(rows)
