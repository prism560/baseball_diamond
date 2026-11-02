"""Pitching-change policy for B3: starter removal hazard, bullpen leverage tiers,
tier mixture weights and the league times-through-the-order (TTO) effect.

Definitions (all from the PA table, fielding-team perspective):
  lap     times through the order of the *starter*: batters faced // 9 + 1, capped 3
  fdiff   fielding team's lead (home - vis for the home team fielding in the top)
  fresh   outs == 0 and bases empty (proxy for "start of the half"; the same
          definition is used in training and in the DP, so it is consistent)

Hazard: P(starter is replaced before this PA | still in at the previous PA),
logistic in one-hot(inning x fresh), lap, fdiff bucket, outs, trained on the
three seasons before the evaluation season.  Extra innings (10+) are bullpen
only in the DP (approved simplification), so every starter eventually leaves.

Bullpen tiers: for each team and game date, relievers seen in relief for that
team earlier in the season (plus the previous season at half weight) are
ranked by the share of appearances entered "close and late" (inning >= 7,
|fdiff| <= 2), smoothed; cumulative appearance thirds give high/mid/low tiers.
Tier quality = appearance-weighted average of the relievers' as-of rates;
tier hand mix = their share of left-handers.

Tier mixture weights P(tier | inning bucket, fdiff bucket) are league-wide,
from relief PAs of the three previous seasons.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .ratings import NR, REF, R_IDX

TIERS = ["high", "mid", "low"]
FD_EDGES = np.array([-2.5, -0.5, 0.5, 2.5])            # buckets: <=-3, -2..-1, 0, 1..2, >=3
N_FD = len(FD_EDGES) + 1
INN_B = [5, 6, 7, 8, 9]                                  # mixture inning buckets: <=5,6,7,8,9+


def fd_bucket(fdiff):
    return np.digitize(np.asarray(fdiff), FD_EDGES)


def inn_bucket(inning):
    i = np.asarray(inning)
    return np.clip(i, 5, 9) - 5


def annotate(pa: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Add fielding-perspective columns, starter flags and lap to the PA table (game order)."""
    d = pa.copy()
    g = games.set_index("game_id")
    d["date"] = d["game_id"].map(g["date"])
    home = d["game_id"].str[:3].to_numpy()
    vis = d["game_id"].map(g["visitor"]).astype(str).to_numpy()
    top = d["half"].to_numpy() == 0
    d["fteam"] = np.where(top, home, vis)
    d["fdiff"] = np.where(top, d["diff0"], -d["diff0"]).astype(int)
    d["fresh"] = ((d["outs0"] == 0) & (d["bases0"] == 0)).astype(int)
    d["is_starter"] = (d["pitcher"].astype(str) == d["starter"].astype(str)).to_numpy()
    grp = [d["game_id"], d["half"]]
    st = d["is_starter"].astype(int)
    d["bf_starter"] = st.groupby(grp).cumsum() - st
    d["lap"] = np.minimum(d["bf_starter"] // 9 + 1, 3).astype(int)
    rem = (~d["is_starter"]).astype(int)
    d["removed_before"] = (rem.groupby(grp).cumsum() - rem) > 0
    return d


def _hazard_X(inning, fresh, lap, fdiff, outs) -> np.ndarray:
    inn = np.clip(np.asarray(inning), 1, 9) - 1
    cols = [np.eye(18)[inn * 2 + np.asarray(fresh)], np.eye(3)[np.asarray(lap) - 1],
            np.eye(N_FD)[fd_bucket(fdiff)], np.eye(3)[np.asarray(outs)],
            np.eye(9)[(np.asarray(lap) - 1) * 3 + np.minimum(inn // 3, 2)]]
    return np.hstack(cols)


@dataclass
class Hazard:
    model: LogisticRegression

    @classmethod
    def fit(cls, d: pd.DataFrame, seasons) -> "Hazard":
        m = d["season"].isin(list(seasons)) & ~d["removed_before"] & (d["inning"] <= 9)
        x = d[m]
        y = (~x["is_starter"]).astype(int).to_numpy()
        X = _hazard_X(x["inning"], x["fresh"], x["lap"], x["fdiff"], x["outs0"])
        lr = LogisticRegression(C=10.0, max_iter=2000).fit(X, y)
        return cls(lr)

    def prob(self, inning, fresh, lap, fdiff, outs, pitch_count_adj: float = 0.0) -> np.ndarray:
        """Removal probability before a PA.  ``pitch_count_adj`` is a log-odds
        shift reserved for real-time use (pitch count is not a DP state)."""
        X = _hazard_X(inning, fresh, lap, fdiff, outs)
        z = self.model.decision_function(X) + pitch_count_adj
        return 1.0 / (1.0 + np.exp(-z))


def tto_offsets(d: pd.DataFrame, seasons) -> np.ndarray:
    """League log-odds offsets (vs OUT) for starter PAs on lap 2 and 3+ relative to lap 1.
    Returns [3, NR] (row 0 = zeros)."""
    m = d["season"].isin(list(seasons)) & d["is_starter"] & (d["o"] >= 0)
    x = d[m]
    out = np.zeros((3, NR))
    rates = []
    for lap in (1, 2, 3):
        r = x.loc[x["lap"] == lap, "r"]
        c = np.bincount(r[r >= 0], minlength=NR) + 1.0
        rates.append(c / c.sum())
    for k in (1, 2):
        e = np.log(rates[k]) - np.log(rates[0])
        out[k] = e - e[REF]
    # intentional walks rise late in games for situational reasons (score, base state, inning),
    # not because the starter tires: no TTO offset for IBB
    out[:, R_IDX["IBB"]] = 0.0
    return out


def bullpen_tiers(d: pd.DataFrame, seasons) -> pd.DataFrame:
    """For each (team, date) in ``seasons``: reliever -> tier, using relief
    appearances strictly before the date (season-to-date + previous season x 0.5).

    Returns long frame: fteam, date, season, pitcher, tier, weight (appearances)."""
    rel = d[~d["is_starter"]]
    first = rel.drop_duplicates(["game_id", "pitcher"])          # entry PA of each relief appearance
    first = pd.DataFrame({"fteam": first["fteam"].to_numpy(), "date": first["date"].to_numpy(),
                          "season": first["season"].to_numpy().astype(int),
                          "pitcher": first["pitcher"].astype(str).to_numpy(),
                          "cl": ((first["inning"] >= 7) & (first["fdiff"].abs() <= 2)).to_numpy(float)})
    game_dates = d[d["season"].isin(list(seasons))].groupby("fteam")["date"].unique()
    out = []
    for team, dates in game_dates.items():
        apps = first[first["fteam"] == team].sort_values("date")
        a_dates, a_seas = apps["date"].to_numpy(), apps["season"].to_numpy()
        a_pid, a_cl = apps["pitcher"].to_numpy(), apps["cl"].to_numpy()
        for s in sorted({pd.Timestamp(x).year for x in dates}):
            w: dict[str, float] = {}
            c: dict[str, float] = {}
            for pid, cl in zip(a_pid[a_seas == s - 1], a_cl[a_seas == s - 1]):
                w[pid] = w.get(pid, 0.0) + 0.5
                c[pid] = c.get(pid, 0.0) + 0.5 * cl
            cur = np.flatnonzero(a_seas == s)
            j = 0
            for date in np.sort([x for x in dates if pd.Timestamp(x).year == s]):
                while j < len(cur) and a_dates[cur[j]] < date:
                    k = cur[j]
                    w[a_pid[k]] = w.get(a_pid[k], 0.0) + 1.0
                    c[a_pid[k]] = c.get(a_pid[k], 0.0) + a_cl[k]
                    j += 1
                if not w:
                    continue
                pids = np.array(list(w))
                ww = np.array([w[x] for x in pids])
                score = (np.array([c[x] for x in pids]) + 1) / (ww + 3)
                o = np.argsort(-score, kind="mergesort")
                cs = np.cumsum(ww[o]) / ww.sum()
                tier = np.where(cs <= 1 / 3 + 1e-9, 0, np.where(cs <= 2 / 3 + 1e-9, 1, 2))
                tier[0] = 0
                out.append(pd.DataFrame({"fteam": team, "date": date, "season": s, "pitcher": pids[o],
                                         "tier": tier, "weight": ww[o]}))
    return pd.concat(out, ignore_index=True)


def tier_mixture(d: pd.DataFrame, tiers: pd.DataFrame, seasons, prior: float = 20.0) -> np.ndarray:
    """P(tier | inning bucket, fdiff bucket) [5, N_FD, 3] from relief PAs in ``seasons``."""
    rel = d[~d["is_starter"] & d["season"].isin(list(seasons))]
    t = tiers.rename(columns={"pitcher": "pid"})
    m = rel.assign(pid=rel["pitcher"].astype(str)).merge(t[["fteam", "date", "pid", "tier"]],
                                                       on=["fteam", "date", "pid"], how="inner")
    cnt = np.zeros((len(INN_B), N_FD, 3))
    np.add.at(cnt, (inn_bucket(m["inning"]), fd_bucket(m["fdiff"]), m["tier"].to_numpy()), 1)
    tot = cnt.sum(axis=(0, 1))
    base = tot / tot.sum()
    w = (cnt + prior * base) / (cnt.sum(-1, keepdims=True) + prior)
    return w
