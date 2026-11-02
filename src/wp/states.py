"""Reusable game-state encoding for win-probability models.

A *state* is the situation at the start of a plate appearance:

  inning   1..9, extra innings pooled into bucket 10 ("10+")
  half     0 = top (visitor bats), 1 = bottom (home bats)
  outs     0..2
  bases    3-bit occupancy: bit0 = 1B, bit1 = 2B, bit2 = 3B  (0..7)
  diff     home score - visitor score, clipped to [-10, 10]
  xrule    extra-inning rule era: 1 if the "runner on second" tie-breaker
           applies (Retrosheet ``info,tiebreaker,2``; MLB regular seasons 2020+),
           else 0.  Only meaningful for inning bucket 10; forced to 0 otherwise
           so that regulation innings pool across eras.

``encode``/``decode`` map states to a dense integer key (mixed radix) and back.
The same encoding is intended for later stages (B2 exact DP etc.).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_INN = 10          # 1..9, 10 = 10+
N_HALF = 2
N_OUTS = 3
N_BASES = 8
DIFF_CAP = 10
N_DIFF = 2 * DIFF_CAP + 1
N_XRULE = 2
N_STATES = N_INN * N_HALF * N_OUTS * N_BASES * N_DIFF * N_XRULE

# radices, innermost last
_RADIX = (N_XRULE, N_INN, N_HALF, N_OUTS, N_BASES, N_DIFF)


@dataclass(frozen=True)
class State:
    inning: int
    half: int
    outs: int
    bases: int
    diff: int
    xrule: int = 0


def inning_bucket(inning):
    return np.minimum(np.asarray(inning), N_INN)


def clip_diff(diff):
    return np.clip(np.asarray(diff), -DIFF_CAP, DIFF_CAP)


def canonical_xrule(inning, xrule):
    return np.where(inning_bucket(inning) >= N_INN, np.asarray(xrule), 0)


def encode(inning, half, outs, bases, diff, xrule=0):
    """Vectorised state -> integer key. Inputs may be scalars or arrays."""
    inn = inning_bucket(inning)
    x = canonical_xrule(inning, xrule)
    d = clip_diff(diff) + DIFF_CAP
    key = x
    for v, r in ((inn - 1, N_INN), (np.asarray(half), N_HALF), (np.asarray(outs), N_OUTS),
                 (np.asarray(bases), N_BASES), (d, N_DIFF)):
        key = key * r + v
    return key.astype(np.int32) if isinstance(key, np.ndarray) else int(key)


def decode(key):
    """Integer key -> tuple of arrays (inning_bucket, half, outs, bases, diff, xrule)."""
    k = np.asarray(key).astype(np.int64)
    d = k % N_DIFF; k //= N_DIFF
    b = k % N_BASES; k //= N_BASES
    o = k % N_OUTS; k //= N_OUTS
    h = k % N_HALF; k //= N_HALF
    i = k % N_INN + 1; k //= N_INN
    x = k
    return i, h, o, b, d - DIFF_CAP, x


def encode_state(s: State) -> int:
    return encode(s.inning, s.half, s.outs, s.bases, s.diff, s.xrule)


def decode_state(key: int) -> State:
    i, h, o, b, d, x = (int(v) for v in decode(key))
    return State(i, h, o, b, d, x)


def n_runners(bases):
    b = np.asarray(bases)
    return (b & 1) + ((b >> 1) & 1) + ((b >> 2) & 1)


def is_possible(inning, half, outs, bases, diff):
    """Logical feasibility of a PA-start state (vectorised).

    Impossible: bottom of the 9th or later with the home team ahead (the game
    would already be over). (Top-half states are all reachable.)
    """
    inn = np.asarray(inning)
    return ~((np.asarray(half) == 1) & (inn >= 9) & (np.asarray(diff) > 0))


def outs_remaining(inning, half, outs):
    """Outs remaining in regulation for the *batting order of the game*
    (54 total outs: 27 per team).  Extra innings -> 0 floor; used for the
    progress variable in B1:  progress = 1 - outs_remaining/54."""
    inn = np.minimum(np.asarray(inning), 9)
    half = np.asarray(half)
    outs = np.asarray(outs)
    # outs already made by both teams before this PA
    made = (inn - 1) * 6 + half * 3 + outs
    return np.clip(54 - made, 0, 54)


def progress(inning, half, outs):
    return 1.0 - outs_remaining(inning, half, outs) / 54.0


def add_state_columns(df, xrule_col="xrule"):
    """Add ``diff``, ``key`` (and helpers) to a states DataFrame with columns
    inning, half, outs, bases, vis_score, home_score[, xrule]."""
    df = df.copy()
    df["diff"] = (df["home_score"].astype(np.int16) - df["vis_score"].astype(np.int16))
    x = df[xrule_col].to_numpy() if xrule_col in df else 0
    df["key"] = encode(df["inning"].to_numpy(), df["half"].to_numpy(), df["outs"].to_numpy(),
                       df["bases"].to_numpy(), df["diff"].to_numpy(), x)
    df["progress"] = progress(df["inning"].to_numpy(), df["half"].to_numpy(), df["outs"].to_numpy())
    return df
