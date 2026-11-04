"""League-average plate-appearance transitions for the B2 DP engine.

A plate appearance (PA) is decomposed as

    s0 --N--> s1 --q--> outcome o --A--> s2 (+ runs)

  s0  state when the PA begins (B0's PA-start state)
  N   "between-PA" transition: runner events that happen before the first
      pitch of the PA (pickoffs, steals, balks, wild pitches on a pitchless
      line).  Identified with the pitch string: a non-batting play line whose
      pitch sequence so far contains no pitch.  N may end the half-inning
      (pickoff / caught stealing for the third out); then the batter leads off
      the next inning.
  s1  state at the first pitch
  q   outcome distribution q(o | batting side, outs, bases) at s1
  A   advancement P(outs', bases', runs | o, outs, bases).  Runner events
      *during* the PA (after the first pitch) are absorbed here, conditional
      on the PA outcome (documented simplification for the count chain).

Outcome categories (``OUTCOMES``): K, BB, IBB, HBP, 1B, 2B, 3B, HR, OUT (fielded
out incl. fielder's choice), ROE (reached on error; catcher interference is
merged here), INT (the PA never finished: a runner was put out for the third
out, or a walk-off run scored, after the first pitch).

The direct control ``D`` maps PA-start state to next PA-start state (+ runs)
without decomposition.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .retro_parse import PITCH_CODES

OUTCOMES = ["K", "BB", "IBB", "HBP", "1B", "2B", "3B", "HR", "OUT", "ROE", "INT"]
O_IDX = {o: i for i, o in enumerate(OUTCOMES)}
CAT_TO_O = {**{o: O_IDX[o] for o in OUTCOMES if o != "INT"}, "CI": O_IDX["ROE"]}
NO = len(OUTCOMES)
RMAX = 4          # at most 4 runs can score from one PA-start state (3 runners + batter)
NB, NOUT = 8, 3


def n_pitches(s: str) -> int:
    return sum(1 for c in s if c in PITCH_CODES and c != "U")


def build_pa_table(events: pd.DataFrame) -> pd.DataFrame:
    """One row per PA from the per-play ``events`` table (wp.validate).

    Columns: game_id, half, inning, outs0, bases0, diff0, outs1, bases1, rN,
    o (outcome index, -1 if the half ended in N), outs2, bases2, r2,
    pitches (sequence of the PA), mid (runner events after the first pitch,
    "npitch:outs:bases:diff|..."), n_next (index of next PA in same half or -1).
    """
    gid = events["game_id"].to_numpy()
    pa = events["pa"].to_numpy()
    inn = events["inning"].to_numpy()
    half = events["half"].to_numpy()
    outs = events["outs"].to_numpy()
    bases = events["bases"].to_numpy()
    diff = (events["home"].to_numpy().astype(int) - events["vis"].to_numpy().astype(int))
    cat = events["cat"].astype(str).to_numpy()
    bat = events["batting"].to_numpy()
    oa = events["outs_after"].to_numpy()
    ba = events["bases_after"].to_numpy()
    runs = events["runs"].to_numpy()
    pit = events["pitches"].astype(str).to_numpy()
    has_ids = "batter" in events
    bt = events["batter"].astype(str).to_numpy() if has_ids else np.full(len(events), "")
    pt = events["pitcher"].astype(str).to_numpy() if has_ids else np.full(len(events), "")
    sl = events["slot"].to_numpy() if has_ids else np.zeros(len(events), int)
    sp = events["starter"].astype(str).to_numpy() if has_ids else np.full(len(events), "")

    key = np.empty(len(events), dtype=object)
    starts = np.flatnonzero(np.r_[True, (gid[1:] != gid[:-1]) | (pa[1:] != pa[:-1])])
    ends = np.r_[starts[1:], len(events)]
    rows = []
    for s, e in zip(starts, ends):
        h = half[s]
        sgn = 1 if h == 1 else -1
        o0, b0, d0 = outs[s], bases[s], diff[s]
        i = s
        rN = 0
        o1, b1 = o0, b0
        # N: leading non-batting lines with no pitch thrown yet
        while i < e and not bat[i] and n_pitches(pit[i]) == 0:
            rN += runs[i]
            o1, b1 = oa[i], ba[i]
            i += 1
        mid = []
        if o1 >= 3:
            rows.append((gid[s], h, inn[s], o0, b0, d0, 3, 0, rN, -1, 3, 0, 0, "", "",
                         bt[s], pt[s], sl[s], sp[s]))
            continue
        if i >= e:
            # PA consisted only of N lines without ending the half (e.g. data where the
            # batter changes mid-PA); treat as INT with no pitches
            rows.append((gid[s], h, inn[s], o0, b0, d0, o1, b1, rN, O_IDX["INT"], o1, b1, 0, "", "",
                         bt[s], pt[s], sl[s], sp[s]))
            continue
        r2 = 0
        o2, b2 = o1, b1
        last_cat = None
        for j in range(i, e):
            r2 += runs[j]
            o2, b2 = oa[j], ba[j]
            if not bat[j]:
                mid.append(f"{n_pitches(pit[j])}:{oa[j]}:{ba[j]}:{d0 + sgn * (rN + r2)}")
            else:
                last_cat = cat[j]
        o = CAT_TO_O.get(last_cat, O_IDX["INT"]) if last_cat is not None else O_IDX["INT"]
        if last_cat is not None and o == O_IDX["INT"]:
            o = O_IDX["OUT"]  # unknown batting category (none expected)
        if o2 >= 3:
            b2 = 0
        # batter/pitcher of the PA = those on the line that ended it (a pinch hitter or reliever
        # entering mid-PA is the one charged); slot = lineup slot at PA start
        rows.append((gid[s], h, inn[s], o0, b0, d0, o1, b1, rN, o, o2, b2, r2, pit[e - 1], "|".join(mid),
                     bt[e - 1], pt[e - 1], sl[s], sp[s]))
    df = pd.DataFrame(rows, columns=["game_id", "half", "inning", "outs0", "bases0", "diff0", "outs1",
                                     "bases1", "rN", "o", "outs2", "bases2", "r2", "pitches", "mid",
                                     "batter", "pitcher", "slot", "starter"])
    for c in ("batter", "pitcher", "starter"):
        df[c] = df[c].astype("category")
    df["slot"] = df["slot"].astype(np.int8)
    for c in ("half", "outs0", "bases0", "outs1", "bases1", "rN", "outs2", "bases2", "r2"):
        df[c] = df[c].astype(np.int8)
    df["o"] = df["o"].astype(np.int8)
    df["inning"] = df["inning"].astype(np.int16)
    df["diff0"] = df["diff0"].astype(np.int16)
    df["season"] = df["game_id"].str[3:7].astype(np.int16)
    # next PA start within the same half (for the direct-transition control)
    same = (df["game_id"].to_numpy()[1:] == df["game_id"].to_numpy()[:-1]) & \
           (df["half"].to_numpy()[1:] == df["half"].to_numpy()[:-1]) & \
           (df["inning"].to_numpy()[1:] == df["inning"].to_numpy()[:-1])
    nxt = np.where(np.r_[same, False], np.arange(1, len(df) + 1), -1)
    df["n_next"] = nxt
    return df


def _norm(counts: np.ndarray, prior: np.ndarray, k: float, axes: tuple) -> np.ndarray:
    """(counts + k*prior) normalised over ``axes``; rows with no data -> prior."""
    x = counts + k * prior
    s = x.sum(axis=axes, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(s > 0, x / np.where(s > 0, s, 1), 0.0)
    return p


@dataclass
class Transitions:
    """League-average transition arrays (all probabilities).

    N[side, outs, bases, outs', bases', r]       outs' in 0..3, r in 0..RMAX
    q[side, outs, bases, o]
    A[o, outs, bases, outs', bases', r]
    D[side, outs, bases, outs', bases', r]       direct control (PA start -> next PA start)
    """
    N: np.ndarray
    q: np.ndarray
    A: np.ndarray
    D: np.ndarray
    seasons: list = field(default_factory=list)
    n_pa: int = 0

    def without_N(self) -> "Transitions":
        N = np.zeros_like(self.N)
        for o in range(NOUT):
            for b in range(NB):
                N[:, o, b, o, b, 0] = 1.0
        return Transitions(N, self.q, self.A, self.D, self.seasons, self.n_pa)


def _rel_prior(counts: np.ndarray) -> np.ndarray:
    """Prior for a [..., outs, bases, outs', bases', r] count array obtained by
    pooling over outs in relative terms (delta outs, capped at 3)."""
    lead = counts.shape[:-5]
    rel = np.zeros(lead + (NB, 4, NB, RMAX + 1))  # [..., bases, dOuts, bases', r]
    for o in range(NOUT):
        for o2 in range(4):
            if o2 >= o:
                rel[..., :, o2 - o, :, :] += counts[..., o, :, o2, :, :]
    prior = np.zeros_like(counts, dtype=float)
    for o in range(NOUT):
        for dlt in range(4):
            o2 = min(o + dlt, 3)
            prior[..., o, :, o2, :, :] += rel[..., :, dlt, :, :]
    s = prior.sum(axis=(-3, -2, -1), keepdims=True)
    return np.where(s > 0, prior / np.where(s > 0, s, 1), 0)


def estimate(pa: pd.DataFrame, seasons, k: float = 5.0) -> Transitions:
    d = pa[pa["season"].isin(list(seasons))]
    side = d["half"].to_numpy()
    o0, b0 = d["outs0"].to_numpy(), d["bases0"].to_numpy()
    o1, b1 = d["outs1"].to_numpy(), d["bases1"].to_numpy()
    o2, b2 = d["outs2"].to_numpy(), d["bases2"].to_numpy()
    rN = np.minimum(d["rN"].to_numpy(), RMAX)
    r2 = np.minimum(d["r2"].to_numpy(), RMAX)
    oc = d["o"].to_numpy()

    Nc = np.zeros((2, NOUT, NB, 4, NB, RMAX + 1))
    np.add.at(Nc, (side, o0, b0, o1, np.where(o1 >= 3, 0, b1), rN), 1)
    m = oc >= 0
    qc = np.zeros((2, NOUT, NB, NO))
    np.add.at(qc, (side[m], o1[m], b1[m], oc[m]), 1)
    Ac = np.zeros((NO, NOUT, NB, 4, NB, RMAX + 1))
    np.add.at(Ac, (oc[m], o1[m], b1[m], o2[m], b2[m], r2[m]), 1)

    # direct: PA start -> next PA start in the same half (computed on ``d`` itself, so any
    # row filtering by the caller is safe), else the half's end (3 outs or a walk-off)
    g_, h_, i_ = d["game_id"].to_numpy(), d["half"].to_numpy(), d["inning"].to_numpy()
    same = np.r_[(g_[1:] == g_[:-1]) & (h_[1:] == h_[:-1]) & (i_[1:] == i_[:-1]), False]
    nxt = np.where(same, np.arange(1, len(d) + 1), 0)
    on = np.where(same, o0[np.minimum(nxt, len(d) - 1)], o2)
    bn = np.where(same, b0[np.minimum(nxt, len(d) - 1)], np.where(o2 >= 3, 0, b2))
    rD = np.minimum(rN + r2, RMAX)
    Dc = np.zeros((2, NOUT, NB, 4, NB, RMAX + 1))
    np.add.at(Dc, (side, o0, b0, on, bn, rD), 1)

    N = _norm(Nc, _rel_prior(Nc.sum(axis=0, keepdims=True)).repeat(2, 0), k, (-3, -2, -1))
    pooled = qc.sum(axis=2, keepdims=True)                       # pool bases within (side, outs)
    q = _norm(qc, (pooled / np.maximum(pooled.sum(axis=-1, keepdims=True), 1)).repeat(NB, 2), k, (-1,))
    A = _norm(Ac, _rel_prior(Ac), k, (-3, -2, -1))
    D = _norm(Dc, _rel_prior(Dc.sum(axis=0, keepdims=True)).repeat(2, 0), k, (-3, -2, -1))
    # once the half is over the base state is meaningless: fold outs'=3 mass onto bases'=0
    # (the outs-pooled prior can otherwise put tiny mass on (3 outs, runners on))
    for arr in (N, A, D):
        arr[..., 3, 0, :] = arr[..., 3, :, :].sum(axis=-2)
        arr[..., 3, 1:, :] = 0.0
    # rows that remain empty (impossible start states): absorbing identity / uniform
    for arr in (N, D):
        for s_ in range(2):
            for o in range(NOUT):
                for b in range(NB):
                    if arr[s_, o, b].sum() == 0:
                        arr[s_, o, b, o, b, 0] = 1.0
    for o_ in range(NO):
        for o in range(NOUT):
            for b in range(NB):
                if A[o_, o, b].sum() == 0:
                    A[o_, o, b, 3, 0, 0] = 1.0
    for s_ in range(2):
        for o in range(NOUT):
            for b in range(NB):
                if q[s_, o, b].sum() == 0:
                    q[s_, o, b, O_IDX["OUT"]] = 1.0
    return Transitions(N, q, A, D, sorted(int(x) for x in set(seasons)), int(len(d)))


def main() -> None:
    """Build data/processed/pa.parquet (seasons >= --from) from the validated events."""
    import argparse
    from pathlib import Path

    from .manifest import check_fresh, code_files, write_meta
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--from", dest="first", type=int, default=2000)
    a = ap.parse_args()
    evp, gp = a.processed / "events.parquet", a.processed / "games.parquet"
    check_fresh(evp)
    check_fresh(gp)
    ev = pd.read_parquet(evp)
    ev = ev[ev["game_id"].str[3:7].astype(int) >= a.first]
    pa = build_pa_table(ev)
    out = a.processed / "pa.parquet"
    pa.to_parquet(out, index=False)
    write_meta([out], [evp, gp], "transitions", code_files("transitions", "retro_parse"))
    print(f"pa rows {len(pa):,}")


if __name__ == "__main__":
    main()
