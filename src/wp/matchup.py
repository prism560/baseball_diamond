"""Matchup outcome distribution (generalised multinomial log5) for B3.

For a PA in state (side, outs, bases):

  log q(o)/q(OUT) = log q_league(o|state)/q_league(OUT|state)     state shape (B2, window K=10)
                  + env_o                                          season-to-date environment
                  + dev_bat_o + dev_pit_o + park_o + platoon_o     player/park deviations
                                                                   (ratings.deviations)

over the 10 categories ``ratings.RCATS``; the result is scaled by 1 - q_INT and
INT keeps its league state-dependent probability (task decision B3 0.1).
With all deviations and env at zero this is exactly the league q (reduction).
"""
from __future__ import annotations

import numpy as np

from .ratings import NR, RCATS, REF
from .transitions import O_IDX, NO

RC = np.array([O_IDX[o] for o in RCATS])
INT = O_IDX["INT"]


def combine(q_state: np.ndarray, dev: np.ndarray) -> np.ndarray:
    """q_state [..., NO] league probabilities; dev [..., NR] log-odds deviations
    (vs OUT) broadcastable.  Returns matchup q [..., NO]."""
    qi = q_state[..., INT]
    q10 = q_state[..., RC]
    with np.errstate(divide="ignore"):
        z = np.log(q10) - np.log(q10[..., [REF]]) + dev
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    p10 = e / e.sum(axis=-1, keepdims=True)
    out = np.zeros(np.broadcast_shapes(q_state.shape, dev.shape[:-1] + (NO,)))
    out[..., RC] = p10 * (1.0 - qi)[..., None]
    out[..., INT] = qi
    return out


def env_offset(L_cur: np.ndarray, L_win: np.ndarray) -> np.ndarray:
    """Season-to-date environment relative to the transition window (log-odds vs OUT)."""
    e = np.log(L_cur) - np.log(L_win)
    return e - e[..., [REF]]


def window_rates(q_counts_10: np.ndarray) -> np.ndarray:
    x = np.asarray(q_counts_10, float)
    return x / x.sum()


def multiclass_logloss(q: np.ndarray, o: np.ndarray) -> np.ndarray:
    p = q[np.arange(len(o)), o]
    return -np.log(np.clip(p, 1e-12, 1))


__all__ = ["combine", "env_offset", "multiclass_logloss", "NR", "RCATS"]
