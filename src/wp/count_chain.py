"""Ball-strike count chain for the plate appearance in progress (B2).

Counts: 12 states (balls 0-3 x strikes 0-2).  Pitch events at a count:

  ball, called strike, swinging strike, foul, foul bunt, hit by pitch,
  in play (split into 1B, 2B, 3B, HR, OUT, ROE), plus two rare PA-ending
  events treated as terminal at the current count: intentional walk (IBB:
  decided at the count where the first ``I`` pitch is thrown, or at 0-0 for an
  automatic walk) and INT (the PA ended by a runner event after the first pitch).

Pitch-code classes follow Chadwick src/cwlib/game.c (see retro_parse):
  ball  B P V      called strike C A K      swinging S M Q T (foul tip = strike
  in all counts)   foul F R      foul bunt L O      HBP H      in play X Y

Transitions: ball 4 -> BB; a called/swinging strike or foul bunt with two
strikes -> K; a foul with two strikes leaves the count unchanged.

Count-conditional outcome model:  at count c the terminal-generating events
are weighted by a stratum factor f_o(s):

    w(e | c, s) = base(e | c) * f_{o(e,c)}(s)        (non-terminal events: 1)

normalised per count (Z).  base() is the league estimate, i.e.
p(o | c, s) = q-dependent factor x count multiplier m(o, c) / Z, the form
required in the task.  The factors f are fitted by iterative proportional
fitting so that the outcome distribution of the chain started at 0-0 equals the
DP's q(o | side, outs, bases) for every stratum (max abs error <= 1e-6, tested).

Simplification (documented in the report): runner events during the PA are
not modelled pitch by pitch; they are absorbed into the advancement table A.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .transitions import NB, NO, NOUT, O_IDX, OUTCOMES

COUNTS = [(b, k) for b in range(4) for k in range(3)]
C_IDX = {c: i for i, c in enumerate(COUNTS)}
NC = len(COUNTS)
EV = ["ball", "cstr", "swstr", "foul", "bfoul", "hbp", "inplay", "ibb", "int"]
E_IDX = {e: i for i, e in enumerate(EV)}
NE = len(EV)
INPLAY_O = [O_IDX[o] for o in ("1B", "2B", "3B", "HR", "OUT", "ROE")]
CHAR_EV = {**{c: "ball" for c in "BPV"}, **{c: "cstr" for c in "CAK"},
           **{c: "swstr" for c in "SMQT"}, **{c: "foul" for c in "FR"},
           **{c: "bfoul" for c in "LO"}, "H": "hbp", "X": "inplay", "Y": "inplay", "I": "ibb"}


def pitch_events(pitches: str, o: int):
    """Return list of (count_idx, event_idx) for a PA and whether the
    sequence is consistent with outcome ``o``; the last pair is terminal."""
    b = k = 0
    out = []
    for ch in pitches:
        e = CHAR_EV.get(ch)
        if e is None:
            continue
        out.append((C_IDX[(b, k)], E_IDX[e]))
        if e == "ibb":
            return out, o == O_IDX["IBB"]
        if e == "ball":
            b += 1
            if b == 4:
                return out, o == O_IDX["BB"]
        elif e in ("cstr", "swstr", "bfoul"):
            k += 1
            if k == 3:
                return out, o == O_IDX["K"]
        elif e == "foul":
            k = min(k + 1, 2)
        elif e == "hbp":
            return out, o == O_IDX["HBP"]
        elif e == "inplay":
            return out, o in INPLAY_O
    if o == O_IDX["IBB"]:
        out.append((C_IDX[(b, k)], E_IDX["ibb"]))
        return out, True
    if o == O_IDX["INT"]:
        out.append((C_IDX[(b, k)], E_IDX["int"]))
        return out, True
    return out, False


def counts_before_each_pitch(pitches: str):
    """Count index before every pitch (for pitch-level evaluation)."""
    b = k = 0
    res = []
    for ch in pitches:
        e = CHAR_EV.get(ch)
        if e is None:
            continue
        res.append(C_IDX[(b, k)])
        if e in ("ibb", "hbp", "inplay"):
            break
        if e == "ball":
            b += 1
            if b == 4:
                break
        elif e in ("cstr", "swstr", "bfoul"):
            k += 1
            if k == 3:
                break
        elif e == "foul":
            k = min(k + 1, 2)
    return res


@dataclass
class BaseRates:
    ev: np.ndarray       # [count, event] probabilities
    split: np.ndarray    # [count, len(INPLAY_O)] in-play outcome split
    n_pa: int
    n_skipped: int


def estimate_base(pa: pd.DataFrame, seasons, k: float = 1.0) -> BaseRates:
    d = pa[pa["season"].isin(list(seasons)) & (pa["o"] >= 0)]
    evc = np.zeros((NC, NE))
    spl = np.zeros((NC, len(INPLAY_O)))
    pos = {o: i for i, o in enumerate(INPLAY_O)}
    skipped = 0
    for pitches, o in zip(d["pitches"].to_numpy(), d["o"].to_numpy()):
        seq, ok = pitch_events(pitches, int(o))
        if not ok:
            skipped += 1
            continue
        for c, e in seq:
            evc[c, e] += 1
        c, e = seq[-1]
        if e == E_IDX["inplay"]:
            spl[c, pos[int(o)]] += 1
    ev = (evc + k / NE) / (evc + k / NE).sum(1, keepdims=True)
    split = (spl + k / len(INPLAY_O)) / (spl + k / len(INPLAY_O)).sum(1, keepdims=True)
    return BaseRates(ev, split, int(len(d) - skipped), skipped)


def _weights(base: BaseRates, f: np.ndarray):
    """Per-count weights.  f[..., NO] -> (term[..., count, NO], trans[..., count, NE]) normalised.

    term[c, o]: prob. the PA ends at this pitch with outcome o.
    trans[c, e]: prob. of non-terminal event e (ball/strike/foul moving the count).
    """
    lead = f.shape[:-1]
    term = np.zeros(lead + (NC, NO))
    trans = np.zeros(lead + (NC, NE))
    for ci, (b, k) in enumerate(COUNTS):
        ev = base.ev[ci]
        # ball
        if b == 3:
            term[..., ci, O_IDX["BB"]] += ev[E_IDX["ball"]] * f[..., O_IDX["BB"]]
        else:
            trans[..., ci, E_IDX["ball"]] = ev[E_IDX["ball"]]
        for e in ("cstr", "swstr", "bfoul"):
            if k == 2:
                term[..., ci, O_IDX["K"]] += ev[E_IDX[e]] * f[..., O_IDX["K"]]
            else:
                trans[..., ci, E_IDX[e]] = ev[E_IDX[e]]
        trans[..., ci, E_IDX["foul"]] = ev[E_IDX["foul"]]
        term[..., ci, O_IDX["HBP"]] += ev[E_IDX["hbp"]] * f[..., O_IDX["HBP"]]
        term[..., ci, O_IDX["IBB"]] += ev[E_IDX["ibb"]] * f[..., O_IDX["IBB"]]
        term[..., ci, O_IDX["INT"]] += ev[E_IDX["int"]] * f[..., O_IDX["INT"]]
        for j, o in enumerate(INPLAY_O):
            term[..., ci, o] += ev[E_IDX["inplay"]] * base.split[ci, j] * f[..., o]
    Z = term.sum(-1) + trans.sum(-1)
    return term / Z[..., None], trans / Z[..., None]


def outcome_dist(base: BaseRates, f: np.ndarray) -> np.ndarray:
    """P[..., count, o]: PA outcome distribution starting from each count."""
    term, trans = _weights(base, f)
    lead = f.shape[:-1]
    P = np.zeros(lead + (NC, NO))
    for b in (3, 2, 1, 0):
        for k in (2, 1, 0):
            ci = C_IDX[(b, k)]
            acc = term[..., ci, :].copy()
            if b < 3:
                acc += trans[..., ci, E_IDX["ball"], None] * P[..., C_IDX[(b + 1, k)], :]
            if k < 2:
                s = (trans[..., ci, E_IDX["cstr"]] + trans[..., ci, E_IDX["swstr"]]
                     + trans[..., ci, E_IDX["bfoul"]] + trans[..., ci, E_IDX["foul"]])
                acc += s[..., None] * P[..., C_IDX[(b, k + 1)], :]
                P[..., ci, :] = acc
            else:   # foul with two strikes: self loop
                P[..., ci, :] = acc / (1.0 - trans[..., ci, E_IDX["foul"]])[..., None]
    return P


@dataclass
class CountChain:
    base: BaseRates
    f: np.ndarray        # [side, outs, bases, NO]
    P: np.ndarray        # [side, outs, bases, count, NO]
    max_err: float
    iterations: int


def fit(base: BaseRates, q: np.ndarray, tol: float = 1e-12, max_iter: int = 20000) -> CountChain:
    """IPF: find f so that P(o | 0-0) == q for every (side, outs, bases)."""
    f = np.ones(q.shape)
    for it in range(1, max_iter + 1):
        P0 = outcome_dist(base, f)[..., 0, :]
        err = np.abs(P0 - q).max()
        if err < tol:
            break
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(P0 > 0, q / P0, 0.0)
        f = f * ratio
    P = outcome_dist(base, f)
    return CountChain(base, f, P, float(np.abs(P[..., 0, :] - q).max()), it)


def pitch_wp_table(chain: CountChain, G: dict) -> dict:
    """WP before a pitch: sum_o P(o | count, side, outs, bases) * G(o).
    Returns {(inning, half): array [outs, bases, d, count]}."""
    out = {}
    for (inn, half), g in G.items():                      # g[o1, b1, d, O]
        out[(inn, half)] = np.einsum("xycO,xydO->xydc", chain.P[half], g)
    return out


__all__ = ["COUNTS", "OUTCOMES", "estimate_base", "fit", "outcome_dist", "pitch_wp_table",
           "counts_before_each_pitch", "pitch_events", "NB", "NOUT"]
