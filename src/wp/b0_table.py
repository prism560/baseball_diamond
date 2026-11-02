"""B0: state-table win probability with hierarchical Bayesian shrinkage.

Smoothing ("backoff by shrinkage"):  for every cell of a level L with parent
level P,

    wp_L = (wins_L + k * wp_P) / (n_L + k)

applied top-down along the hierarchy (coarsest first):

  root : (half)                                   prior = overall home win rate
  L0   : (inning, half, xrule, diff_band)          diff in coarse bands
  L1   : (inning, half, xrule, diff)
  L2   : (inning, half, xrule, outs, n_runners, diff)   base state simplified
  L3   : (inning, half, xrule, outs, bases, diff)       full state

Cells with many observations keep their empirical rate; sparse cells are
pulled toward the coarser estimate.  One pseudo-count ``k`` is chosen by
walk-forward validation (fit on seasons < S-1, score on S-1).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import states as S

DIFF_BANDS = np.array([-10, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 11])  # bin edges (left-closed)


def diff_band(diff):
    return np.digitize(np.asarray(diff), DIFF_BANDS[1:-1])


def all_keys_frame() -> pd.DataFrame:
    keys = np.arange(S.N_STATES, dtype=np.int32)
    i, h, o, b, d, x = S.decode(keys)
    df = pd.DataFrame({"key": keys, "inning": i.astype(np.int8), "half": h.astype(np.int8),
                       "outs": o.astype(np.int8), "bases": b.astype(np.int8),
                       "diff": d.astype(np.int8), "xrule": x.astype(np.int8)})
    # xrule only exists for the 10+ bucket; drop duplicate (non-canonical) keys
    df = df[(df["inning"] == S.N_INN) | (df["xrule"] == 0)].reset_index(drop=True)
    df["possible"] = S.is_possible(df["inning"], df["half"], df["outs"], df["bases"], df["diff"])
    return df


LEVELS = {
    "L0": ["inning", "half", "xrule", "dband"],
    "L1": ["inning", "half", "xrule", "diff"],
    "L2": ["inning", "half", "xrule", "outs", "nrun", "diff"],
    "L3": ["inning", "half", "xrule", "outs", "bases", "diff"],
}


def season_counts(states: pd.DataFrame) -> pd.DataFrame:
    """Aggregate PA states to (season, key) -> n, wins.  ``states`` must have
    season, key, home_win."""
    g = states.groupby(["season", "key"], observed=True)["home_win"].agg(["size", "sum"])
    g.columns = ["n", "wins"]
    return g.reset_index()


def fit_table(counts: pd.DataFrame, k: float) -> pd.DataFrame:
    """Build the smoothed table from aggregated (key, n, wins) counts."""
    c = counts.groupby("key", as_index=False)[["n", "wins"]].sum()
    t = all_keys_frame().merge(c, on="key", how="left")
    t[["n", "wins"]] = t[["n", "wins"]].fillna(0)
    t["nrun"] = S.n_runners(t["bases"])
    t["dband"] = diff_band(t["diff"])

    root = t.groupby("half")[["n", "wins"]].sum()
    root_p = ((root["wins"] + 1) / (root["n"] + 2)).rename("p_root")
    t = t.merge(root_p, left_on="half", right_index=True)
    parent = t["p_root"].to_numpy()
    for name, cols in LEVELS.items():
        agg = t.groupby(cols)[["n", "wins"]].transform("sum")
        n_l, w_l = agg["n"].to_numpy(), agg["wins"].to_numpy()
        if name == "L3":
            n_l, w_l = t["n"].to_numpy(), t["wins"].to_numpy()
        # the parent estimate itself must be constant within the group: use group mean
        p = (w_l + k * parent) / (n_l + k)
        if name != "L3":
            p = pd.Series(p).groupby([t[c] for c in cols]).transform("mean").to_numpy()
        t[f"n_{name}"] = n_l.astype(np.int64)
        t[f"wp_{name}"] = p
        parent = p
    t["p_raw"] = np.where(t["n"] > 0, t["wins"] / t["n"].clip(lower=1), np.nan)
    t["wp"] = t["wp_L3"]
    t["k"] = k
    t["n"] = t["n"].astype(np.int64)
    t["wins"] = t["wins"].astype(np.int64)
    keep = ["key", "inning", "half", "outs", "bases", "diff", "xrule", "possible", "n", "wins",
            "p_raw", "n_L2", "n_L1", "n_L0", "wp_L0", "wp_L1", "wp_L2", "wp", "k"]
    return t[keep].sort_values("key").reset_index(drop=True)


@dataclass
class WPTable:
    """Lookup wrapper: ``WPTable(table).wp(inning, half, outs, bases, diff, xrule)``."""
    table: pd.DataFrame

    def __post_init__(self):
        self._dense = np.full(S.N_STATES, np.nan)
        self._dense[self.table["key"].to_numpy()] = self.table["wp"].to_numpy()
        self._n = np.zeros(S.N_STATES, dtype=np.int64)
        self._n[self.table["key"].to_numpy()] = self.table["n"].to_numpy()

    @classmethod
    def load(cls, path: str | Path) -> "WPTable":
        return cls(pd.read_parquet(path))

    def wp_key(self, key):
        return self._dense[np.asarray(key)]

    def wp(self, inning, half, outs, bases, diff, xrule=0):
        """Home-team win probability for a PA-start state (scalars or arrays)."""
        r = self.wp_key(S.encode(inning, half, outs, bases, diff, xrule))
        return float(r) if np.ndim(r) == 0 else r

    def n(self, inning, half, outs, bases, diff, xrule=0):
        return self._n[np.asarray(S.encode(inning, half, outs, bases, diff, xrule))]


def lookup_wp(table_path: str | Path, inning: int, half: int, outs: int, bases: int,
              diff: int, xrule: int = 0) -> float:
    """Convenience: load ``wp_table_b0.parquet`` and return WP for one state."""
    return WPTable.load(table_path).wp(inning, half, outs, bases, diff, xrule)


def logloss(p, y, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def tune_k(counts: pd.DataFrame, val_states: pd.DataFrame,
           grid=(1, 3, 10, 30, 100, 300, 1000)) -> tuple[float, dict]:
    """Pick k minimising log-loss on ``val_states`` (key, home_win)."""
    res = {}
    for k in grid:
        t = WPTable(fit_table(counts, k))
        res[k] = logloss(t.wp_key(val_states["key"].to_numpy()), val_states["home_win"].to_numpy())
    best = min(res, key=res.get)
    return best, res


def monotonic_violations(table: pd.DataFrame, tol: float = 0.0) -> pd.DataFrame:
    """Cells where WP decreases as the home lead increases (other keys fixed).
    Only possible states are compared."""
    t = table[table["possible"]].sort_values(["xrule", "inning", "half", "outs", "bases", "diff"])
    g = t.groupby(["xrule", "inning", "half", "outs", "bases"])
    t = t.assign(prev_wp=g["wp"].shift(1), prev_diff=g["diff"].shift(1))
    return t[(t["prev_wp"] - t["wp"]) > tol]


def _pav(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted pool-adjacent-violators: non-decreasing fit."""
    vals, wts, cnt = [], [], []
    for yi, wi in zip(y, w):
        vals.append(yi); wts.append(wi); cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, c2 = vals.pop(), wts.pop(), cnt.pop()
            v1, w1, c1 = vals.pop(), wts.pop(), cnt.pop()
            ws = w1 + w2
            vals.append((v1 * w1 + v2 * w2) / ws); wts.append(ws); cnt.append(c1 + c2)
    return np.repeat(vals, cnt)


def monotone_project(table: pd.DataFrame) -> pd.DataFrame:
    """Return a copy whose ``wp`` is non-decreasing in home lead within each
    (xrule, inning, half, outs, bases) among possible cells (weights n + k).
    Kept as an optional variant; the raw smoothed ``wp`` is in ``wp_unproj``."""
    t = table.sort_values(["xrule", "inning", "half", "outs", "bases", "diff"]).copy()
    t["wp_unproj"] = t["wp"]
    w = (t["n"] + t["k"]).to_numpy(float)
    wp = t["wp"].to_numpy().copy()
    poss = t["possible"].to_numpy()
    grp = t.groupby(["xrule", "inning", "half", "outs", "bases"], sort=False).indices
    for idx in grp.values():
        idx = idx[poss[idx]]
        if len(idx) > 1:
            wp[idx] = _pav(wp[idx], w[idx])
    t["wp"] = wp
    return t.sort_values("key").reset_index(drop=True)
