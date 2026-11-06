"""Normalized feed -> game state, PA rows and output points (rebuilt from scratch every time).

Principle: the state is never patched incrementally.  Every snapshot is replayed
from the first play, so corrections in the feed (overturned calls, scoring
changes) are reflected automatically; the session layer only compares event
hashes to *log* what changed.

PA rows follow the Retrosheet PA table (wp.transitions.build_pa_table):
  half, inning, outs0, bases0, diff0   state at the start of the PA
  outs1, bases1, rN                    after the runner events before the first pitch (N)
  o                                    outcome index (-1: the half ended in N; None: in progress)
  outs2, bases2, r2                    after the PA
  pitches                              Retrosheet pitch characters (pickoff throws as 1/2/3)
  mid                                  "npitch:outs:bases:diff|..." runner events after the first pitch
  batter, pitcher, slot, starter       MLBAM ids (mapped to Retrosheet ids by wp.live.ids)
Extra-inning automatic runners (runner_placed) are part of the PA-start state, as in Retrosheet.

Output points (one per state-changing event, in feed order):
  phase start  PA start (s0)                 pre   runner event before the first pitch
        pitch  pitch that did not end the PA  mid   runner event after the first pitch
        sub    substitution                  end   PA ended (state after the PA)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..transitions import O_IDX
from . import codes
from .normalize import Event, Move, NormGame, Play, _h

BASE_BIT = {"1B": 1, "2B": 2, "3B": 4}
BASE_NUM = {"1B": 1, "2B": 2, "3B": 3}


@dataclass
class State:
    inning: int = 1
    half: int = 0
    outs: int = 0
    occ: dict = field(default_factory=dict)      # base number -> runner id
    score: list = field(default_factory=lambda: [0, 0])   # away, home
    balls: int = 0
    strikes: int = 0

    @property
    def bases(self) -> int:
        return sum(1 << (b - 1) for b in self.occ)

    @property
    def diff(self) -> int:
        return self.score[1] - self.score[0]


def apply_moves(st: State, moves: list[Move]) -> tuple[int, int]:
    """Apply one event's runner movements (grouped per runner).  Returns (outs, runs) added."""
    if not moves:
        return 0, 0
    order, per = [], {}
    for m in moves:
        if m.runner not in per:
            order.append(m.runner)
            per[m.runner] = []
        per[m.runner].append(m)
    final = []
    for rid in order:
        ms = per[rid]
        origin = ms[0].start
        for b, r in list(st.occ.items()):      # the runner leaves wherever he is
            if r == rid:
                st.occ.pop(b)
        if origin in BASE_NUM:                 # trust the feed's origin base
            st.occ.pop(BASE_NUM[origin], None)
        final.append((rid, origin, any(m.is_out for m in ms), ms[-1].end))
    d_out = d_run = 0
    for rid, origin, out, end in final:
        if out:
            d_out += 1
        elif end == "score":
            d_run += 1
        elif end in BASE_NUM:
            st.occ[BASE_NUM[end]] = rid
        elif origin in BASE_NUM:    # no end, not out: stays
            st.occ[BASE_NUM[origin]] = rid
    st.outs += d_out
    side = st.half
    st.score[side] += d_run
    return d_out, d_run


@dataclass
class GameView:
    game: NormGame
    rows: list            # PA rows (dict), one per play
    points: list          # output points (dict)
    state: State          # current state (after the last event)
    lineups: dict         # current lineups side -> {slot: id}
    pitcher: dict         # current pitcher of each side (fielding)
    over: bool            # game over according to the plays (walk-off / 3rd out of the last half)
    corrections_hint: list = field(default_factory=list)


def _side(half: int, batting: bool) -> str:
    return ("away" if half == 0 else "home") if batting else ("home" if half == 0 else "away")


def reconstruct(g: NormGame) -> GameView:
    st = State()
    lineups = {s: dict(g.lineups.get(s, {})) for s in ("away", "home")}
    pitcher = {s: g.starters.get(s) for s in ("away", "home")}
    starter = dict(pitcher)
    rows, points = [], []
    over = False
    xrule = g.game_type not in ("F", "D", "L", "W")

    def point(phase, key, pl: Play, ev: Event | None, last: str, pa_i: int, npitch: int, mid: bool, extra=None):
        p = {"key": key, "phase": phase, "ab": pl.ab, "pa_i": pa_i, "inning": st.inning, "half": st.half,
             "outs": st.outs, "bases": st.bases, "away": st.score[0], "home": st.score[1],
             "balls": st.balls, "strikes": st.strikes, "batter": pl.batter,
             "pitcher": pitcher[_side(st.half, False)], "n_pitch": npitch, "mid": mid,
             "ts": ev.ts if ev is not None else pl.ts, "last_event": last,
             "overturned": bool(ev.overturned) if ev is not None else False}
        if extra:
            p.update(extra)
        # content hash: the event itself + the state after it (duplicates vs corrections)
        # (pitch vs pitch_end is not content: a PA can end on an ordinary-looking pitch,
        #  e.g. catcher's interference on a foul, which is known only with the result)
        p["ehash"] = ev.hash if ev is not None else None
        p["hash"] = _h(["pitch" if phase == "pitch_end" else phase, ev.hash if ev is not None else None, p["outs"], p["bases"], p["away"], p["home"],
                        p["balls"], p["strikes"], p["batter"], p["pitcher"], p.get("o")])
        points.append(p)

    for pl in g.plays:
        if over:
            break
        if (pl.inning, pl.half) != (st.inning, st.half):
            st.inning, st.half, st.outs, st.occ = pl.inning, pl.half, 0, {}
            if xrule and pl.inning > g.scheduled_innings:
                # automatic runner (regular season): known before the feed publishes the
                # runner_placed event; a placeholder id, replaced when the event arrives
                st.occ[2] = -1
        st.balls = st.strikes = 0
        bat_side, fld_side = _side(pl.half, True), _side(pl.half, False)
        pa_i = len(rows)
        # extra-inning automatic runner belongs to the PA-start state
        for ev in pl.events:
            if ev.kind == "runner_placed":
                apply_moves(st, ev.moves)
        # pinch hitter announced inside this play: lineup slot from the substitution
        o0, b0, d0 = st.outs, st.bases, st.diff
        slot = next((s for s, pid in lineups[bat_side].items() if pid == pl.batter), 0)
        if not slot:      # pinch hitter whose substitution is an event of this play
            for ev in pl.events:
                if ev.kind == "sub:offense" and ev.player == pl.batter and ev.batting_order:
                    slot = int(str(ev.batting_order)[0])
        res_keys = _result_events(pl)
        row = {"ab": pl.ab, "half": pl.half, "inning": pl.inning, "outs0": o0, "bases0": b0, "diff0": d0,
               "outs1": o0, "bases1": b0, "rN": 0, "o": None, "outs2": o0, "bases2": b0, "r2": 0,
               "pitches": "", "mid": "", "batter": pl.batter, "pitcher": pl.pitcher, "slot": slot,
               "starter": starter[fld_side], "complete": pl.complete, "bat_side": pl.bat_side,
               "pitch_hand": pl.pitch_hand}
        rows.append(row)
        point("start", (pl.ab, "s"), pl, None, "", pa_i, 0, False)
        npitch = 0
        chars = []
        mid_list = []
        rN = r2 = 0
        mid_seen = False
        for ev in pl.events:
            if ev.kind == "runner_placed":
                continue
            ctx = {"game_pk": g.pk, "ab": pl.ab, "index": ev.key[1]}
            if ev.kind.startswith("sub:"):
                k = ev.kind[4:]
                side = bat_side if k == "offense" else fld_side
                if k == "pitcher":
                    pitcher[fld_side] = ev.player
                if ev.batting_order and ev.player:
                    s = int(str(ev.batting_order)[0])
                    if 1 <= s <= 9:
                        if k == "offense" or (k in ("defense", "switch", "pitcher") and side == fld_side):
                            lineups[side][s] = ev.player
                if k == "offense" and ev.base in (1, 2, 3) and ev.player:
                    st.occ[int(ev.base)] = ev.player          # pinch runner takes the base
                point("sub", ev.key, pl, ev, ev.description or ev.code or "", pa_i, npitch, mid_seen,
                      {"sub": k, "player": ev.player, "replaced": ev.replaced})
                if ev.moves:
                    apply_moves(st, ev.moves)
                continue
            if ev.kind == "pitch":
                npitch += 1
                chars.append(ev.rchar)
                if not _ends_pa(ev):
                    st.balls, st.strikes = min(ev.balls, 3), min(ev.strikes, 2)
            elif ev.kind == "pickoff":
                chars.append(ev.rchar)
            terminal = ev.kind == "pitch" and (ev.key in res_keys or _ends_pa(ev))
            if terminal:
                # the pitch that ends the PA: shown before its result is applied (the result
                # movements, possibly published later, belong to the 'end' point)
                point("pitch_end", ev.key, pl, ev, ev.description or "", pa_i, npitch, mid_seen)
                continue
            if ev.key in res_keys:
                continue                                      # applied with the PA result
            if ev.moves and ev.kind != "pitch":
                d_out, d_run = apply_moves(st, ev.moves)
                if npitch == 0:
                    rN += d_run
                    row["outs1"], row["bases1"], row["rN"] = min(st.outs, 3), (st.bases if st.outs < 3 else 0), rN
                else:
                    r2 += d_run
                    mid_seen = True
                    mid_list.append(f"{npitch}:{min(st.outs, 3)}:{st.bases if st.outs < 3 else 0}:{st.diff}")
                ph = "pre" if npitch == 0 else "mid"
                point(ph, ev.key, pl, ev, ev.description or ev.code or "", pa_i, npitch, mid_seen)
                if st.outs >= 3:
                    break
                continue
            if ev.kind == "pitch":
                is_last = ev.key in res_keys
                if ev.moves:     # the result's own movements (batter + runners), applied at the end
                    if not is_last:
                        d_out, d_run = apply_moves(st, ev.moves)
                        r2 += d_run
                        mid_seen = True
                        mid_list.append(f"{npitch}:{min(st.outs, 3)}:{st.bases if st.outs < 3 else 0}:{st.diff}")
                if not is_last:
                    point("pitch", ev.key, pl, ev, ev.description or "", pa_i, npitch, mid_seen)
        row["pitches"] = "".join(chars)
        row["mid"] = "|".join(mid_list)
        if not pl.complete:
            row["o"] = None
            continue
        # PA result: movements of the last pitch + any without a matching event
        lp = _last_pitch(pl)
        res_moves = [m for ev in pl.events if ev.key in res_keys for m in ev.moves] + pl.result_moves
        bmv = [m for m in res_moves if m.runner == pl.batter and m.start is None]
        b_end = bmv[-1].end if bmv else None
        b_out = any(m.is_out for m in bmv)
        o = codes.result_outcome(pl.result_type, b_end, b_out, {"game_pk": g.pk, "ab": pl.ab})
        if st.outs < 3:
            d_out, d_run = apply_moves(st, res_moves)
            r2 += d_run
        if o is None:
            o = -1 if npitch == 0 and st.outs >= 3 else O_IDX["INT"]
        row["o"] = o
        row["pitches"] = "".join(chars)
        row["mid"] = "|".join(mid_list)
        row["r2"] = r2
        row["outs2"] = min(st.outs, 3)
        row["bases2"] = st.bases if st.outs < 3 else 0
        if npitch == 0 and o != -1 and row["outs1"] == o0 and row["bases1"] == b0:
            pass
        if npitch == 0 and o == -1:
            row["outs1"], row["bases1"] = 3, 0
            row["outs2"], row["bases2"], row["r2"] = 3, 0, 0
        st.balls = st.strikes = 0
        if st.half == 1 and st.inning >= g.scheduled_innings and st.diff > 0:
            over = True                       # walk-off
        if st.outs >= 3:
            st.occ = {}
            if st.inning >= g.scheduled_innings and ((st.half == 0 and st.diff > 0) or
                                                     (st.half == 1 and st.diff != 0)):
                over = True
        point("end", (pl.ab, "r"), pl, lp, pl.result_desc or (pl.result_type or ""), pa_i, npitch, mid_seen,
              {"o": o})
    if g.status == "final":
        over = True
    return GameView(g, rows, points, st, lineups, pitcher, over)


def _result_events(pl: Play) -> set:
    """Keys of the events whose movements are the PA result: the last pitch, and later
    events (e.g. the no_pitch events of an automatic intentional walk) carrying
    movements of the result's own event type.  Plays ended by a runner event have none."""
    if not pl.complete or pl.result_type in codes.NON_PA_RESULT:
        return set()
    lp = _last_pitch(pl)
    keys = {lp.key} if lp is not None else set()
    start = lp.key[1] if lp is not None else -1
    for ev in pl.events:
        if ev.key[1] > start and ev.moves and ev.kind != "runner_placed" and \
                all(m.event_type == pl.result_type for m in ev.moves):
            keys.add(ev.key)
    return keys


def _ends_pa(ev: Event) -> bool:
    """Pitch that ends the PA by itself (feed count after it shows ball four / strike three)."""
    return ev.pclass in ("in_play", "hbp") or ev.balls >= 4 or ev.strikes >= 3


def _last_pitch(pl: Play) -> Event | None:
    for ev in reversed(pl.events):
        if ev.kind == "pitch":
            return ev
    return None
