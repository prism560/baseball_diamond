"""Exact dynamic-programming win-probability engine (B2).

State at the start of a plate appearance:
  inning (1..9, 10 = any extra inning), half (0 top / 1 bottom), outs (0..2),
  bases (bit0=1B, bit1=2B, bit2=3B), diff = home - visitor in [-R, R],
  batting-team lineup slot ``sb`` and the other team's next slot ``so``
  (0..S-1; S = 9 for a lineup axis, S = 1 collapses it).

Values are P(home wins).  Backward induction is exact:

* Within a half-inning the PA kernel K moves (outs, bases, sb) to
  (outs', bases', sb + a) with r runs; ``a`` = 1 when the batter's PA completed,
  0 when the half ended (or the game ended) before the PA finished.
  Outs never decrease and runs move ``diff`` monotonically toward the batting
  team, so states are solved level by level: outs 2 -> 0, and inside an outs
  level diff in the order runs push it.  The only cycles left are
  (bases, sb) transitions with zero runs and no out, solved exactly with the
  matrix inverse (I - T0)^-1.
* Termination (official baseball rules, e.g. MLB Official Baseball Rules 7.01):
  the bottom of the 9th is not played when the home team leads; in the bottom
  of the 9th or later the game ends the moment the home team leads (walk-off);
  after a completed extra inning a lead ends the game.
* Extra innings repeat the same structure.  The tie value at the start of an
  extra inning X(s_v, s_h) satisfies X = C + P X (P = probability that the
  inning ends tied with given next slots), solved exactly as a linear system -
  no truncation at an arbitrary inning.
* ``xrule``: extra half-innings start with a runner on second (Retrosheet
  ``info,tiebreaker,2``; MLB regular seasons 2020-).
* diff beyond +-R is treated as decided (home win if > R, loss if < -R); the
  report checks that widening R changes no value by more than 1e-6.

Kernels are built per half-inning by a callable, so B3 can plug in
batter-specific (per slot) and state-dependent pitcher distributions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .transitions import NB, NO, NOUT, O_IDX, RMAX, Transitions

INT = O_IDX["INT"]


def compose_kernel(N: np.ndarray, q: np.ndarray, A: np.ndarray) -> np.ndarray:
    """K[o, b, sb, o', b', r, a] for one batting side.

    N[o, b, o1, b1, rN]; q[sb, o1, b1, O]; A[O, o1, b1, o2, b2, r2].
    """
    S = q.shape[0]
    K = np.zeros((NOUT, NB, S, 4, NB, RMAX + 1, 2))
    # half ended (or nothing more) inside N: a = 0, state (3, 0)
    for rN in range(RMAX + 1):
        K[:, :, :, 3, 0, rN, 0] += N[:, :, 3, 0, rN][:, :, None]
    adv = np.ones(NO, int)
    adv[INT] = 0
    for a in (0, 1):
        Oset = np.flatnonzero(adv == a)
        # post-N kernel restricted to outcomes with this slot advance
        # W[o1,b1,sb,o2,b2,r2] = sum_O q[sb,o1,b1,O] A[O,o1,b1,o2,b2,r2]
        W = np.einsum("sxyO,Oxyzwr->xyszwr", q[..., Oset], A[Oset])
        for rN in range(RMAX + 1):
            Nr = N[:, :, :3, :, rN]                       # [o,b,o1,b1]
            if not Nr.any():
                continue
            part = np.einsum("obxy,xyszwr->obszwr", Nr, W)
            for r2 in range(RMAX + 1):
                r = min(rN + r2, RMAX)
                K[:, :, :, :, :, r, a] += part[..., r2]
    return K


def post_n_kernel(q: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Kernel starting at the first pitch (no N): K[o1, b1, sb, o2, b2, r, a]."""
    eye = np.zeros((NOUT, NB, 4, NB, RMAX + 1))
    for o in range(NOUT):
        for b in range(NB):
            eye[o, b, o, b, 0] = 1
    return compose_kernel(eye, q, A)


def direct_kernel(D: np.ndarray) -> np.ndarray:
    """Kernel from the direct (undecomposed) transition D[o,b,o',b',r]; a = 1
    (except a transition straight to 3 outs, which we cannot attribute)."""
    K = np.zeros((NOUT, NB, 1, 4, NB, RMAX + 1, 2))
    K[:, :, 0, :, :, :, 1] = D
    return K


@dataclass
class DPResult:
    V: dict          # (inning, half) -> array [outs, bases, Dp, S, S]  (value at PA start)
    R: int
    pad: int
    S: int
    X: np.ndarray    # extra-inning tie value at top start [S, S]
    seconds: float = 0.0
    peak_bytes: int = 0

    def d_index(self, diff):
        return np.asarray(diff) + self.R + self.pad

    def wp(self, inning, half, outs, bases, diff, sb=0, so=0):
        inn = min(int(inning), 10)
        return float(self.V[(inn, half)][outs, bases, self.d_index(diff), sb, so])


def _pad_const(Dp: int, R: int, pad: int):
    """Constant (decided) values at padded diff indices."""
    d = np.arange(Dp) - R - pad
    return (d > R).astype(float), (np.abs(d) > R)


class Engine:
    def __init__(self, R: int = 30, S: int = 1, xrule: bool = True):
        self.R, self.S, self.xrule = R, S, xrule
        self.pad = RMAX
        self.Dp = 2 * R + 1 + 2 * self.pad
        self.dvals = np.arange(self.Dp) - R - self.pad
        self.valid = np.abs(self.dvals) <= R
        self.decided_val, self.decided = _pad_const(self.Dp, R, self.pad)

    # ---- one half-inning -------------------------------------------------
    def solve_half(self, K: np.ndarray, E: np.ndarray, half: int, walkoff: bool) -> np.ndarray:
        """Solve V[o, b, d, sb, so, B] for a half given end-of-half values
        E[d, sb_next, so, B] (B = basis dim; 1 for numeric values)."""
        S, Dp = self.S, self.Dp
        Bd = E.shape[-1]
        V = np.zeros((4, NB, Dp, S, S, Bd))
        V[3] = E[None]                                   # outs == 3 -> half over
        const = np.zeros((Dp, Bd))
        const[:, 0] = self.decided_val
        V[:3, :, self.decided] = const[self.decided][None, None, :, None, None, :]
        fixed_one = np.zeros(Dp, bool)
        if walkoff:                                      # home ahead in bottom 9+ : game over
            fixed_one = self.dvals > 0
            one = np.zeros(Bd); one[0] = 1
            V[:, :, fixed_one] = one
        sgn = 1 if half == 1 else -1
        order = np.flatnonzero(self.valid & ~fixed_one)
        order = order[::-1] if half == 1 else order
        # K2[o][b,sb, o',b',r, sb_next]
        sb_next = (np.arange(S)[:, None] + np.arange(2)[None, :]) % S       # [sb, a]
        K2 = np.zeros((NOUT, NB, S, 4, NB, RMAX + 1, S))
        for a in range(2):
            for s in range(S):
                K2[:, :, s, :, :, :, sb_next[s, a]] += K[:, :, s, :, :, :, a]
        for o in (2, 1, 0):
            T0 = K2[o, :, :, o, :, 0, :].reshape(NB * S, NB * S)
            Minv = np.linalg.inv(np.eye(NB * S) - T0)
            Kr = K2[o].copy()
            Kr[:, :, o, :, 0, :] = 0.0                   # same-level zero-run part is in Minv
            Kr = Kr[:, :, o:].reshape(NB * S, -1)        # rows (b,sb); cols (o'>=o, b', r, sb')
            for di in order:
                idx = di + sgn * np.arange(RMAX + 1)     # d' for r = 0..RMAX
                U = V[o:4][:, :, idx]                    # [o', b', r, sb', so, B]
                rhs = Kr @ U.reshape(-1, S * Bd)         # (b,sb) x (so,B)
                V[o, :, di] = (Minv @ rhs).reshape(NB, S, S, Bd)
        return V[:3]

    # ---- whole game ------------------------------------------------------
    def solve(self, kernel_for: Callable[[int, int], np.ndarray]) -> DPResult:
        """``kernel_for(inning, half)`` returns K[o,b,sb,o',b',r,a] (inning 10 = extras)."""
        import time
        import tracemalloc
        tracemalloc.start()
        t0 = time.time()
        S, Dp = self.S, self.Dp
        bstart_x = 0b010 if self.xrule else 0
        d = self.dvals
        # ---- extra inning: affine in X (basis 1 + S*S) ----
        Bd = 1 + S * S
        Eb = np.zeros((Dp, S, S, Bd))                    # [d, sb_next(home), so(vis), B]
        Eb[d > 0, :, :, 0] = 1.0
        for sv in range(S):
            for sh in range(S):
                Eb[d == 0, sh, sv, 1 + sv * S + sh] = 1.0
        Vb = self.solve_half(kernel_for(10, 1), Eb, 1, walkoff=True)
        Et = np.zeros((Dp, S, S, Bd))                    # [d, sb_next(vis), so(home), B]
        Et[d > 0, :, :, 0] = 1.0
        swap = np.swapaxes(Vb[0, bstart_x], 1, 2)        # [d, so(vis)->sb_next, sb(home)->so]
        Et[d <= 0] = swap[d <= 0]
        Vt = self.solve_half(kernel_for(10, 0), Et, 0, walkoff=False)
        top0 = Vt[0, bstart_x, d == 0][0]                # [sv, sh, B]
        C = top0[..., 0].reshape(S * S)
        P = top0[..., 1:].reshape(S * S, S * S)
        X = np.linalg.solve(np.eye(S * S) - P, C).reshape(S, S)
        coef = np.concatenate([[1.0], X.reshape(-1)])
        V = {(10, 1): Vb @ coef, (10, 0): Vt @ coef}
        # ---- regulation, inning 9 down to 1 ----
        for inn in range(9, 0, -1):
            # bottom half: end -> next inning top (or decided after 9)
            if inn == 9:
                Eb = np.zeros((Dp, S, S))
                Eb[d > 0] = 1.0
                Eb[d == 0] = X.T[None]                   # [sb_next(home), so(vis)] -> X[sv, sh]
            else:
                Eb = np.swapaxes(V[(inn + 1, 0)][0, 0], 1, 2)
            Vb = self.solve_half(kernel_for(inn, 1), Eb[..., None], 1, walkoff=(inn >= 9))[..., 0]
            V[(inn, 1)] = Vb
            Et = np.swapaxes(Vb[0, 0], 1, 2).copy()
            if inn == 9:
                Et[d > 0] = 1.0                          # home leads after top 9: game over
            Vt = self.solve_half(kernel_for(inn, 0), Et[..., None], 0, walkoff=False)[..., 0]
            V[(inn, 0)] = Vt
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return DPResult(V, self.R, self.pad, S, X, time.time() - t0, peak)


def kernels_from(tr: Transitions, use_N: bool = True, S: int = 1, direct: bool = False):
    """Return kernel_for(inning, half) for league-average transitions."""
    if direct:
        Ks = {h: direct_kernel(tr.D[h]) for h in (0, 1)}
    else:
        trn = tr if use_N else tr.without_N()
        Ks = {h: compose_kernel(trn.N[h], np.repeat(trn.q[h][None], S, 0), trn.A) for h in (0, 1)}
    return lambda inning, half: Ks[half]


def after_outcome_values(res: DPResult, tr: Transitions, xrule: bool = True) -> dict:
    """G[(inning, half)][o1, b1, d, O]: expected home WP right after outcome O
    from first-pitch state (o1, b1, diff d), using the collapsed (S=1) table.
    W = sum_O q * G is the value at the first pitch (after N)."""
    assert res.S == 1
    out = {}
    Dp = 2 * res.R + 1 + 2 * res.pad
    dvals = np.arange(Dp) - res.R - res.pad
    for (inn, half), V in res.V.items():
        # values after the PA: same half (outs < 3) or end-of-half value
        U = np.zeros((4, NB, Dp))
        U[:3] = V[:, :, :, 0, 0]
        U[3] = _end_of_half(res, inn, half, dvals)[None]
        sgn = 1 if half == 1 else -1
        G = np.zeros((NOUT, NB, Dp, NO))
        for r in range(RMAX + 1):
            idx = np.clip(np.arange(Dp) + sgn * r, 0, Dp - 1)
            Ur = U[:, :, idx]                              # [o2, b2, d]
            if half == 1 and inn >= 9:
                Ur = np.where((dvals[idx] > 0)[None, None], 1.0, Ur)
            G += np.einsum("Oxyzw,zwd->xydO", tr.A[..., r], Ur)
        out[(inn, half)] = G
    return out


def _end_of_half(res: DPResult, inn: int, half: int, dvals) -> np.ndarray:
    xb = 0b010
    if half == 0:
        v = res.V[(inn, 1)][0, 0 if inn < 10 else xb, :, 0, 0].copy()
        if inn >= 9:
            v[dvals > 0] = 1.0
        return v
    if inn >= 9:
        v = np.where(dvals > 0, 1.0, 0.0)
        v[dvals == 0] = res.X[0, 0]
        return v
    return res.V[(inn + 1, 0)][0, 0, :, 0, 0]
