"""B3 DP engine: players, starter-removal hazard and bullpen tiers inside the exact DP.

Per team (as the batting side) a composite state c = slot * 4 + stage:
  slot  0..8   next batter in the lineup
  stage 0, 1, 2 = facing the opposing *starter* on lap 1, 2, 3+ (lap = times
                  through this team's order, i.e. the starter's TTO)
        3       = facing the opposing bullpen
=> 36 composites.  State of a half: (outs, bases, diff, c_bat, c_other); the other
team's composite is passive during the half.  Extra innings: bullpen for both
teams (approved simplification), composites = slot only (9); the tie value of an
extra inning is solved exactly as a linear system (as in B2).

Per PA (batting team composite c, fielding team F):
  1. if F's starter is in (stage < 3): replaced with probability h(inning,
     fresh, lap, F's lead bucket, outs) -> stage 3
  2. league pre-pitch transition N, then the matchup outcome q (batter of the
     slot vs F's starter [+TTO offset] or F's bullpen tier mixture
     P(tier | inning, lead bucket)), league advancement A
  3. slot advances when the PA completes; lap advances when the slot wraps.
Hazard and tier mixture depend on the fielding lead, so kernels exist per lead
bucket (5) and the inverse of the zero-run block is computed per bucket.

Real-time hooks (see docs/REALTIME.md): ``override`` replaces the fielding
team's pitcher for one half by a known pitcher; lineup changes (pinch
hitters) are applied by re-solving with the new lineup.
"""
from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass, field

import numpy as np

from . import dp
from . import matchup as MU
from . import pitching_policy as PP
from .ratings import NR, REF
from .transitions import NB, NO, NOUT, RMAX

NST = 4
NC = 9 * NST
FD_REP = np.array([-3, -1, 0, 1, 3])          # representative fielding leads of the 5 buckets


@dataclass
class TeamInputs:
    """Log-odds deviations (vs OUT, length NR) for one team's players on a date."""
    bat_dev: np.ndarray        # [9, 2(phand L,R), NR]  batter + individual platoon vs that hand
    starter_dev: np.ndarray    # [NR]  pitcher deviation of the starter
    starter_hand: int          # 0 = L, 1 = R
    tier_dev: np.ndarray       # [T, 2, NR] bullpen tier x hand
    tier_hand_share: np.ndarray  # [T, 2]  share of PAs by L/R pitchers in the tier


@dataclass
class GameInputs:
    home: TeamInputs
    vis: TeamInputs
    q_state: np.ndarray        # [2(side), 3, 8, NO] league q (side = batting half)
    N: np.ndarray              # [2, 3, 8, 4, 8, R+1]
    A: np.ndarray              # [NO, 3, 8, 4, 8, R+1]
    common_dev: np.ndarray     # [NR] park + environment (applies to both teams)
    hazard: object             # PP.Hazard or None (no starter-removal, i.e. starter forever is
                               # replaced by bullpen at extras only)
    tier_w: np.ndarray         # [5 inning buckets, 5 lead buckets, T]
    tto: np.ndarray            # [3, NR]
    xrule: bool = True


def q_for(gi: GameInputs, side: int, bat_dev_h: np.ndarray, pit_dev: np.ndarray) -> np.ndarray:
    """Matchup q over all states [3, 8, NO] for one batter (dev vs the pitcher's hand) and pitcher."""
    return MU.combine(gi.q_state[side], bat_dev_h + pit_dev + gi.common_dev)


def _kernel(N, q, A):
    """[o, b, o', b', r, a] for one q [3, 8, NO]."""
    return dp.compose_kernel(N, q[None], A)[:, :, 0]


class KernelBuilder:
    def __init__(self, gi: GameInputs, override: dict | None = None, cache: dict | None = None):
        """``cache`` (optional, shared across solves of one game) holds the per-slot PA
        kernels; after a pinch hitter only that team's slot must be dropped
        (``invalidate``) - every other kernel is reused."""
        self.gi = gi
        self.override = override or {}      # (inning, half) -> (pit_dev [NR], hand)
        self._cache = {} if cache is None else cache

    @staticmethod
    def invalidate(cache: dict, half: int, slot: int) -> int:
        """Drop cached kernels of the team batting in ``half`` for lineup ``slot``."""
        keys = [k for k in cache if k[1] == half and k[2] == slot]
        for k in keys:
            del cache[k]
        return len(keys)

    def team(self, half):
        """(batting team, fielding team) for a half."""
        return (self.gi.vis, self.gi.home) if half == 0 else (self.gi.home, self.gi.vis)

    def _starter_K(self, half, slot, lap):
        key = ("s", half, slot, lap)
        if key not in self._cache:
            bat, fld = self.team(half)
            dev = fld.starter_dev + self.gi.tto[lap]
            q = q_for(self.gi, half, bat.bat_dev[slot, fld.starter_hand], dev)
            self._cache[key] = _kernel(self.gi.N[half], q, self.gi.A)
        return self._cache[key]

    def _pitcher_K(self, half, slot, dev, hand):
        bat, _ = self.team(half)
        q = q_for(self.gi, half, bat.bat_dev[slot, hand], dev)
        return _kernel(self.gi.N[half], q, self.gi.A)

    def _bullpen_K(self, half, slot, ib, k):
        key = ("b", half, slot, ib, k)
        if key not in self._cache:
            bat, fld = self.team(half)
            w = self.gi.tier_w[ib, k]
            q = np.zeros((NOUT, NB, NO))
            for t in range(len(w)):
                for h in (0, 1):
                    s = w[t] * fld.tier_hand_share[t, h]
                    if s > 0:
                        q += s * q_for(self.gi, half, bat.bat_dev[slot, h], fld.tier_dev[t, h])
            self._cache[key] = _kernel(self.gi.N[half], q, self.gi.A)
        return self._cache[key]

    def regulation(self, inning: int, half: int):
        """Sparse kernel for a regulation half: (W, tgt).

        W   [nch, 5 lead buckets, o, b, C, o', b', r]  transition weights
        tgt [nch, C]                                   next composite of each channel
        Channels: 0 keep pitcher & PA incomplete, 1 keep & PA complete,
                  2 replaced by bullpen & incomplete, 3 replaced & complete.
        """
        W = np.zeros((4, 5, NOUT, NB, NC, 4, NB, RMAX + 1))
        tgt = np.zeros((4, NC), int)
        ib = PP.inn_bucket(inning)
        ov = self.override.get((inning, half))
        o_, b_ = np.meshgrid(range(NOUT), range(NB), indexing="ij")
        fresh = ((o_ == 0) & (b_ == 0)).astype(int).ravel()
        for slot in range(9):
            nxt = (slot + 1) % 9
            for st in range(NST):
                c = slot * NST + st
                if st == 3 or ov is not None:
                    tgt[0, c], tgt[1, c] = slot * NST + 3, nxt * NST + 3
                else:
                    st_wrap = min(st + 1, 2) if slot == 8 else st
                    tgt[0, c], tgt[1, c] = slot * NST + st, nxt * NST + st_wrap
                tgt[2, c], tgt[3, c] = slot * NST + 3, nxt * NST + 3
        ov_K = {slot: self._pitcher_K(half, slot, *ov) for slot in range(9)} if ov is not None else None
        for k in range(5):
            h_lap = {}
            if self.gi.hazard is not None and ov is None:
                for st in range(3):
                    h_lap[st] = self.gi.hazard.prob(np.full(o_.size, min(inning, 9)), fresh,
                                                    np.full(o_.size, st + 1), np.full(o_.size, FD_REP[k]),
                                                    o_.ravel()).reshape(NOUT, NB)
            for slot in range(9):
                Kbp = self._bullpen_K(half, slot, ib, k) if ov is None else ov_K[slot]
                for st in range(NST):
                    c = slot * NST + st
                    if st == 3 or ov is not None:
                        W[0, k, :, :, c] = Kbp[..., 0]
                        W[1, k, :, :, c] = Kbp[..., 1]
                        continue
                    Ks = self._starter_K(half, slot, st)
                    h = h_lap.get(st, np.zeros((NOUT, NB)))[:, :, None, None, None]
                    W[0, k, :, :, c] = (1 - h) * Ks[..., 0]
                    W[1, k, :, :, c] = (1 - h) * Ks[..., 1]
                    W[2, k, :, :, c] = h * Kbp[..., 0]
                    W[3, k, :, :, c] = h * Kbp[..., 1]
        return W, tgt

    def extra(self, half: int):
        """Sparse kernel with slot-only composites (bullpen): (W [2, 5, o, b, 9, o', b', r], tgt [2, 9])."""
        W = np.zeros((2, 5, NOUT, NB, 9, 4, NB, RMAX + 1))
        tgt = np.stack([np.arange(9), (np.arange(9) + 1) % 9])
        ov = self.override.get((10, half))
        ov_K = {slot: self._pitcher_K(half, slot, *ov) for slot in range(9)} if ov is not None else None
        for k in range(5):
            for slot in range(9):
                Kb = self._bullpen_K(half, slot, PP.inn_bucket(9), k) if ov is None else ov_K[slot]
                W[0, k, :, :, slot] = Kb[..., 0]
                W[1, k, :, :, slot] = Kb[..., 1]
        return W, tgt


@dataclass
class B3Result:
    V: dict                 # (inning, half) -> [3, 8, Dp, Cbat, Cother]
    E: dict                 # (inning, half) -> end-of-half values [Dp, Cbat_next, Cother]
    X: np.ndarray           # [9, 9] extra-inning tie value (vis slot, home slot)
    R: int
    pad: int
    seconds: float = 0.0
    peak_bytes: int = 0
    timings: dict = field(default_factory=dict)

    def wp(self, inning, half, outs, bases, diff, c_bat, c_other):
        i = min(int(inning), 10)
        d = int(np.clip(diff, -self.R - self.pad, self.R + self.pad)) + self.R + self.pad
        if i == 10:
            c_bat, c_other = c_bat // NST, c_other // NST
        return float(self.V[(i, half)][outs, bases, d, c_bat, c_other])


class Engine:
    def __init__(self, R: int = 20, dtype=np.float64):
        self.dtype = dtype
        self.R = R
        self.pad = RMAX
        self.Dp = 2 * R + 1 + 2 * self.pad
        self.d = np.arange(self.Dp) - R - self.pad
        self.valid = np.abs(self.d) <= R

    def lead_bucket(self, half):
        """Lead bucket of the *fielding* team for every diff index."""
        fd = self.d if half == 0 else -self.d          # top: home fields, lead = diff
        return PP.fd_bucket(fd)

    def solve_half(self, K, E: np.ndarray, half: int, walkoff: bool) -> np.ndarray:
        """K = (W [nch, k, o, b, C, o', b', r], tgt [nch, C]); E [Dp, C, P, B] -> V [3, 8, Dp, C, P, B]."""
        from scipy import sparse
        W, tgt = K
        W = W.astype(self.dtype, copy=False)
        E = E.astype(self.dtype, copy=False)
        nch, nk, _, _, C = W.shape[:5]
        Dp = self.Dp
        P, Bd = E.shape[2], E.shape[3]
        PB = P * Bd
        V = np.zeros((4, NB, Dp, C, P, Bd), dtype=self.dtype)
        V[3] = E[None]
        one = np.zeros(Bd, dtype=self.dtype)
        one[0] = 1.0
        dec = ~self.valid
        V[:3, :, dec & (self.d > 0)] = one
        V[:3, :, dec & (self.d < 0)] = 0.0
        fixed = np.zeros(Dp, bool)
        if walkoff:
            fixed = self.d > 0
            V[:, :, fixed] = one
        sgn = 1 if half == 1 else -1
        order = np.flatnonzero(self.valid & ~fixed)
        order = order[::-1] if half == 1 else order
        buck = self.lead_bucket(half)
        cidx = np.arange(C)
        # lead buckets are monotone along the solve order (lead of the fielding team moves
        # with diff), so processing buckets in ascending order respects the dependencies
        assert np.all(np.diff(buck[order]) >= 0)
        for o in (2, 1, 0):
            for k in np.unique(buck[order]):
                dk = order[buck[order] == k]               # diff indices of this lead bucket, solve order
                # zero-run same-outs block -> (I - T0)^-1
                T0 = np.zeros((NB, C, NB, C))
                Ms = np.zeros((NB, C, NB, RMAX, C))        # same outs, r >= 1: columns (b', r, c')
                for j in range(nch):
                    # advanced indices (cidx, tgt[j]) move to the front of the target view
                    T0[:, cidx, :, tgt[j]] += W[j, k, o, :, :, o, :, 0].transpose(1, 0, 2)
                    Ms[:, cidx, :, :, tgt[j]] += W[j, k, o, :, :, o, :, 1:].transpose(1, 0, 2, 3)
                Minv = np.linalg.inv(np.eye(NB * C) - T0.reshape(NB * C, NB * C)).astype(self.dtype)
                Msp = sparse.csr_matrix(Ms.reshape(NB * C, NB * RMAX * C))
                # contributions from higher outs levels (already solved) for all d at once
                nd = len(dk)
                hi = np.zeros((C, NB, nd * PB), dtype=self.dtype)
                if o < 3:
                    for j in range(nch):
                        for r in range(RMAX + 1):
                            Aw = W[j, k, o, :, :, o + 1:, :, r].reshape(NB, C, -1).transpose(1, 0, 2)  # [C, b, x']
                            if not Aw.any():
                                continue
                            U = V[o + 1:4][:, :, dk + sgn * r][:, :, :, tgt[j]]   # [o', b', nd, C, P, B]
                            U = U.transpose(3, 0, 1, 2, 4, 5).reshape(C, Aw.shape[2], nd * PB)
                            hi += np.matmul(Aw, U)
                hi = hi.reshape(C, NB, nd, PB).transpose(2, 1, 0, 3).reshape(nd, NB * C, PB)
                Yhi = np.matmul(Minv[None], hi)                # [nd, (b,c), PB]
                for i, di in enumerate(dk):
                    Vs = V[o][:, di + sgn * np.arange(1, RMAX + 1)]   # [b', r, C', P, B]
                    t = Msp @ Vs.reshape(NB * RMAX * C, PB)
                    V[o, :, di] = (Yhi[i] + Minv @ t).reshape(NB, C, P, Bd)
        return V[:3]

    def solve(self, gi: GameInputs, override: dict | None = None, upto: tuple | None = None,
              cache: B3Result | None = None, start: tuple | None = None,
              track_memory: bool = False, kernel_cache: dict | None = None) -> B3Result:
        """Backward induction.

        cache + upto=(inning, half): halves strictly after ``upto`` are reused from
          ``cache`` (a pitching change only alters the current half).
        start=(inning, half): halves before it are not computed (they are in the past).
        """
        if track_memory:
            tracemalloc.start()
        t0 = time.time()
        kb = KernelBuilder(gi, override, kernel_cache)
        d = self.d
        V, Es = {}, {}
        tm = {"kernel_s": 0.0, "solve_s": 0.0}
        bx = 0b010 if gi.xrule else 0

        def later(key):
            if cache is None or upto is None:
                return False
            return (key[0], key[1]) > (upto[0], upto[1]) and key in cache.V

        # ---- extra innings (bullpen, slot-only composites, affine in X) ----
        if later((10, 0)) and later((10, 1)):
            V[(10, 1)], V[(10, 0)], X = cache.V[(10, 1)], cache.V[(10, 0)], cache.X
            Es[(10, 1)], Es[(10, 0)] = cache.E[(10, 1)], cache.E[(10, 0)]
        else:
            Bd = 1 + 81
            Eb = np.zeros((self.Dp, 9, 9, Bd))
            Eb[d > 0, :, :, 0] = 1.0
            for sv in range(9):
                for sh in range(9):
                    Eb[d == 0, sh, sv, 1 + sv * 9 + sh] = 1.0
            t = time.time(); K = kb.extra(1); tm["kernel_s"] += time.time() - t
            t = time.time(); Vb = self.solve_half(K, Eb, 1, True); tm["solve_s"] += time.time() - t
            Et = np.zeros((self.Dp, 9, 9, Bd))
            Et[d > 0, :, :, 0] = 1.0
            Et[d <= 0] = np.swapaxes(Vb[0, bx], 1, 2)[d <= 0]
            t = time.time(); K = kb.extra(0); tm["kernel_s"] += time.time() - t
            t = time.time(); Vt = self.solve_half(K, Et, 0, False); tm["solve_s"] += time.time() - t
            top0 = Vt[0, bx, d == 0][0]
            X = np.linalg.solve(np.eye(81) - top0[..., 1:].reshape(81, 81), top0[..., 0].reshape(81)).reshape(9, 9)
            coef = np.concatenate([[1.0], X.reshape(-1)])
            V[(10, 1)], V[(10, 0)] = Vb @ coef, Vt @ coef
            Es[(10, 1)], Es[(10, 0)] = Eb @ coef, Et @ coef
        # ---- regulation ----
        slot_of = np.arange(NC) // NST
        for inn in range(9, 0, -1):
            for half in (1, 0):
                key = (inn, half)
                if start is not None and key < tuple(start):
                    continue
                if later(key):
                    V[key], Es[key] = cache.V[key], cache.E[key]
                    continue
                if half == 1:
                    if inn == 9:
                        E = np.zeros((self.Dp, NC, NC))
                        E[d > 0] = 1.0
                        # tie after 9: extra-inning value X[vis slot, home slot]
                        E[d == 0] = X[slot_of[None, :], slot_of[:, None]][None]
                    else:
                        E = np.swapaxes(V[(inn + 1, 0)][0, 0], 1, 2)
                else:
                    E = np.swapaxes(V[(inn, 1)][0, 0], 1, 2).copy()
                    if inn == 9:
                        E[d > 0] = 1.0
                t = time.time(); K = kb.regulation(inn, half); tm["kernel_s"] += time.time() - t
                t = time.time()
                V[key] = self.solve_half(K, E[..., None], half, walkoff=(half == 1 and inn >= 9))[..., 0]
                Es[key] = E
                tm["solve_s"] += time.time() - t
        peak = 0
        if track_memory:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        return B3Result(V, Es, X, self.R, self.pad, time.time() - t0, peak,
                        {k: round(v, 3) for k, v in tm.items()})


def composite(slot: int, stage: int) -> int:
    return slot * NST + stage


__all__ = ["Engine", "GameInputs", "TeamInputs", "KernelBuilder", "B3Result", "composite", "NC", "NST", "REF"]
