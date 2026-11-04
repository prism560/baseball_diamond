"""Incremental B3 game tracker (the real-time counterpart of wp.b3_game.evaluate_game).

The tracker follows evaluate_game step by step so that a replayed game reproduces
the offline values exactly:
  * pre-game solve with the starting lineups and starting pitchers;
  * a new batter in a lineup slot (pinch hitter / substitute batting for the first time)
    updates the lineup and re-solves the remaining halves (start = current half);
  * a reliever on the mound re-solves the current half with his own rating (override);
  * stage of the batting composite = lap - 1 against the starter, 3 for the bullpen / extras.
Differences from the offline walk (by design, see docs/REALTIME.md):
  * the pitcher used for a point is the one on the mound at that point (offline: the
    pitcher charged with the whole PA);
  * a runner event during the PA re-evaluates the count chain at the actual state.

Solves are requested from a ``Solver`` (wp.live.solver); while one is pending the B3 value
is ``None`` and the server shows B2E (recomputing).
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field

import numpy as np

from .. import b3_game as BG
from .. import count_chain as CC
from .. import dp_b3 as D3
from .. import transitions as T

INT = T.O_IDX["INT"]


@dataclass
class _R:                     # adapter with the attributes b3_game.first_pitch_parts reads
    pitcher: str
    outs1: int
    bases1: int
    diff0: int
    rN: int


@dataclass
class Committed:
    cur: dict
    nslot: dict
    n_vs: dict
    removed: dict
    lineup_ver: str
    rows_done: int = 0
    full_start: tuple | None = None    # start half of the latest lineup re-solve (None: pre-game)


@dataclass
class PAContext:
    ok: bool                   # B3 values available (no pending solve)
    inn: int
    half: int
    slot: int
    stage: int
    c_bat: int
    c_oth: int
    rres: object | None
    override: bool
    pitcher: str


def _ver(cur: dict) -> str:
    return hashlib.sha1(repr((cur["vis"], cur["home"])).encode()).hexdigest()[:12]


class LiveB3:
    def __init__(self, M, game: dict, lineups: dict, starters: dict, solver, xrule: bool = True):
        """game: {game_id, date (Timestamp), site, home, visitor (Retrosheet codes)};
        lineups {"vis": [9 ids], "home": [9 ids]}; starters {"vis": id, "home": id}."""
        self.M, self.ctx, self.solver, self.xrule = M, M.ctx, solver, xrule
        self.g = dict(game)
        self.starters = dict(starters)
        self.ab = BG.Ablation()
        cur = {"home": list(lineups["home"]), "vis": list(lineups["vis"])}
        self.state = Committed(cur, {0: 0, 1: 0}, {0: 0, 1: 0}, {0: False, 1: False}, _ver(cur))
        self.checkpoints = [copy.deepcopy(self.state)]
        self._gi = {}
        self._chain = {}
        self.pregame_key = self._request_full(cur, None)

    # ---- inputs / solves ---------------------------------------------------
    def gi(self, cur: dict):
        v = _ver(cur)
        if v not in self._gi:
            gi, L = BG.game_inputs(self.ctx, self.g, cur, self.starters, self.ab)
            gi.xrule = self.xrule
            self._gi[v] = (gi, L)
        return self._gi[v]

    def _request_full(self, cur: dict, start):
        gi, _ = self.gi(cur)
        key = ("full", _ver(cur), start)
        self.solver.request(key, "full", gi=gi, start=start)
        return key

    def full_res(self, ver_start):
        return self.solver.get(("full",) + ver_start)

    # ---- per point ------------------------------------------------------------
    def context(self, row: dict, pitcher_now: str) -> PAContext:
        """PA context for a point of ``row`` (in-progress or complete), committing lineup changes."""
        st = self.state
        half = int(row["half"])
        side = "vis" if half == 0 else "home"
        inn = min(int(row["inning"]), 10)
        slot = int(row["slot"]) - 1 if row["slot"] > 0 else st.nslot[half]
        if str(row["batter"]) != st.cur[side][slot] and row.get("o") != -1:
            st.cur[side][slot] = str(row["batter"])
            st.lineup_ver = _ver(st.cur)
            st.full_start = (inn, half)
            self._request_full(st.cur, (inn, half))
        full_key = st.full_start
        res = self.solver.get(("full", st.lineup_ver, full_key)) if full_key is not None else \
            self.solver.get(self.pregame_key)
        is_st = str(pitcher_now) == str(row["starter"])
        if not is_st:
            st.removed[half] = True
        stage = 3 if st.removed[half] or inn >= 10 else min(st.n_vs[half] // 9 + 1, 3) - 1
        rres, override = res, False
        if st.removed[half] and not is_st:
            override = True
            if res is None:
                rres = None
            else:
                gi, L = self.gi(st.cur)
                pid = str(pitcher_now)
                hk = ("ovr", st.lineup_ver, full_key, (inn, half), pid)
                if self.solver.get(hk) is None and not self.solver.pending(hk):
                    pdv = BG.pitcher_devs(self.ctx, [pid], self.g["date"], L)[0]
                    hand = BG.HAND.get(self.ctx.hands["throws"].get(pid, "R"), 1)
                    self.solver.request(hk, "override", gi=gi, override={(inn, half): (pdv, hand)},
                                        upto=(inn, half), base=("full", st.lineup_ver, full_key), start=(inn, half))
                rres = self.solver.get(hk)
        other = 1 - half
        o_stage = 3 if st.removed[other] or inn >= 10 else min(st.n_vs[other] // 9 + 1, 3) - 1
        return PAContext(rres is not None, inn, half, slot, stage, D3.composite(slot, stage),
                         D3.composite(st.nslot[other], o_stage), rres, override, str(pitcher_now))

    def commit(self, row: dict, pitcher: str) -> None:
        st = self.state
        half = int(row["half"])
        slot = int(row["slot"]) - 1 if row["slot"] > 0 else st.nslot[half]
        if str(pitcher) == str(row["starter"]):
            st.n_vs[half] += 1
        o = row["o"]
        if o is not None and o >= 0 and o != INT:
            st.nslot[half] = (slot + 1) % 9
        st.rows_done += 1
        if len(self.checkpoints) <= st.rows_done:
            self.checkpoints.append(None)
        self.checkpoints[st.rows_done] = copy.deepcopy(st)

    def restore(self, rows_done: int) -> None:
        """Roll back to the state after ``rows_done`` committed rows (feed correction)."""
        self.state = copy.deepcopy(self.checkpoints[rows_done])
        del self.checkpoints[rows_done + 1:]

    # ---- values ---------------------------------------------------------------
    def wp_state(self, pc: PAContext, outs, bases, diff) -> float:
        return pc.rres.wp(pc.inn, pc.half, outs, bases, diff, pc.c_bat, pc.c_oth)

    def pitch_parts(self, pc: PAContext, row: dict, outs, bases, diff):
        """(P[count, O], G[O]) at state (outs, bases, diff) of the PA in progress."""
        gi, L = self.gi(self.state.cur)
        sgn = 1 if pc.half == 1 else -1
        r = _R(pc.pitcher, int(outs), int(bases), int(row["diff0"]), sgn * (int(diff) - int(row["diff0"])))
        q, G = BG.first_pitch_parts(gi, pc.rres, r, pc.inn, pc.half, pc.slot, pc.stage, pc.c_oth, pc.override,
                                    self.ctx, self.g, L)
        k = q.tobytes()
        if k not in self._chain:
            self._chain[k] = CC.fit(self.M.base, q[None]).P[0]
        return self._chain[k], G

    def after_values(self, pc: PAContext, outs2, bases2, diff2, half_over: bool) -> float:
        """B3 value right after a PA (state of the next PA start, before any substitution)."""
        st = self.state
        inn, half = pc.inn, pc.half
        res = pc.rres
        lim = res.R + res.pad
        di = int(np.clip(diff2, -lim, lim)) + lim
        if half == 1 and inn >= 9 and diff2 > 0:
            return 1.0
        ci = (lambda c: c // D3.NST) if inn >= 10 else (lambda c: c)
        b_stage = 3 if st.removed[half] or inn >= 10 else min(st.n_vs[half] // 9 + 1, 3) - 1
        c_next = D3.composite(st.nslot[half], b_stage)
        if half_over:
            return float(res.E[(inn, half)][di, ci(c_next), ci(pc.c_oth)])
        return float(res.V[(inn, half)][outs2, bases2, di, ci(c_next), ci(pc.c_oth)])
