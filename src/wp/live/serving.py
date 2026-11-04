"""Per-game server: feed snapshot -> rebuilt state -> model values -> output messages.

``GameServer.update(feed)`` is the whole real-time path:
  1. normalize + reconstruct from the full feed (wp.live.normalize / reconstruct);
  2. drop stale snapshots (feed timecode older than the newest one seen);
  3. compare point keys / content hashes with the previous snapshot:
       same key, same hash  -> duplicate, nothing to do
       same key, new hash   -> correction (logged with the WP change); values are recomputed
                               from the first changed PA (tracker checkpoint)
       new keys             -> new points (several pitches between polls are caught up at once)
  4. model values per point: B3 (tracker), B2 / B2E (tables), AVG; while a B3 solve is pending,
     or after a solve failure, the B2E value is served with ``recomputing`` / ``model: "B2E"``;
  5. one message per new or changed point, ``seq`` increasing within the game.

``GameServer.poll()`` collects finished worker solves and re-emits the affected points.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import count_chain as CC
from .. import states as ST
from .ids import IdMap, team_code
from .log import info, warn
from .models import SeasonModels, avg
from .normalize import normalize
from .output import LedHysteresis, split_pct
from .reconstruct import GameView, reconstruct
from .tracker import LiveB3

MODELS = ("AVG", "B3", "B2E", "B2")
POSTSEASON = {"F", "D", "L", "W"}


def _half_next(inn, half):
    return (inn, 1) if half == 0 else (inn + 1, 0)


@dataclass
class PointValue:
    B3: float | None
    B2: float
    B2E: float
    AVG: float | None
    if_hr: dict = field(default_factory=dict)     # model -> value
    if_k: dict = field(default_factory=dict)


class GameServer:
    def __init__(self, M: SeasonModels, ids: IdMap, solver, model: str = "AVG", n_leds: int = 40,
                 hysteresis: float = 0.3, game_id: str | None = None):
        assert model in MODELS
        self.M, self.ids, self.solver, self.model = M, ids, solver, model
        self.leds = LedHysteresis(n_leds, hysteresis)
        self.game_id = game_id
        self.seq = 0
        self.last_tc: str | None = None
        self.view: GameView | None = None
        self.points: list = []          # current points (dict)
        self.values: list = []          # PointValue per point
        self.msgs: dict = {}            # point key -> latest message
        self.order: list = []           # point keys in game order
        self.corrections: list = []
        self.log: list = []             # every message emitted (seq order)
        self.tracker: LiveB3 | None = None
        self.info_level = "none"
        self.lat: list = []             # per-update processing seconds
        self.n_dup = self.n_stale = 0
        self.meta: dict = {}

    # ---- setup -------------------------------------------------------------
    def _setup(self, view: GameView) -> None:
        g = view.game
        home, vis = team_code(g.home_id), team_code(g.away_id)
        self.game_id = self.game_id or f"{home}{g.date.replace('-', '')}0"
        S = int(g.date[:4])
        site = self.M.site_of.get(home, home)
        if self.game_id in self.M.games.index:
            site = self.M.games.loc[self.game_id, "site"]
        self.xrule = g.game_type not in POSTSEASON
        self.b2 = self.M.b2[self.xrule]
        self.p_pre = self.M.p_pregame(self.game_id, home, vis, S)
        self.meta = {"game_id": self.game_id, "game_pk": g.pk, "date": g.date, "home": home, "visitor": vis,
                     "site": site, "game_type": g.game_type, "p_pregame": self.p_pre, "xrule": self.xrule}
        self._gdict = {"game_id": self.game_id, "date": pd.Timestamp(g.date), "site": site, "home": home,
                       "visitor": vis}
        for pid, (b, t) in g.hands.items():       # feed handedness for players unknown to Retrosheet rosters
            r = self.ids.retro(pid) if pid in self.ids.m else None
            if r is None:
                rid = f"mlb{pid}"
                if rid not in self.M.ctx.hands.index:
                    self.M.ctx.hands.loc[rid] = [b, t]

    def _init_tracker(self, view: GameView) -> None:
        g = view.game
        lu = view.game.lineups
        if len(lu.get("away", {})) == 9 and len(lu.get("home", {})) == 9 and all(g.starters.values()):
            lineups = {"vis": [self.ids.retro(lu["away"][s]) for s in range(1, 10)],
                       "home": [self.ids.retro(lu["home"][s]) for s in range(1, 10)]}
            starters = {"vis": self.ids.retro(g.starters["away"]), "home": self.ids.retro(g.starters["home"])}
            self.tracker = LiveB3(self.M, self._gdict, lineups, starters, self.solver, self.xrule)
            self.info_level = "lineup"
            info("pregame_solve_requested", game_id=self.game_id)

    # ---- main entry ------------------------------------------------------------
    def update(self, feed: dict, ts_recv: float | None = None) -> list[dict]:
        t0 = time.perf_counter()
        ts_recv = ts_recv or time.time()
        tc = (feed.get("metaData") or {}).get("timeStamp")
        if self.last_tc is not None and tc is not None and tc < self.last_tc:
            self.n_stale += 1
            return []
        g = normalize(feed)
        view = reconstruct(g)
        if self.view is None:
            self._setup(view)
        if tc:
            self.last_tc = tc
        self.view = view
        if self.tracker is None:
            self._init_tracker(view)
        new_points = view.points
        k = 0
        n = min(len(self.points), len(new_points))
        while k < n and self.points[k]["key"] == new_points[k]["key"] and \
                self.points[k]["hash"] == new_points[k]["hash"]:
            k += 1
        out = []
        if k == len(new_points) and k == len(self.points):
            self.n_dup += 1
            out += self._status_message(view, tc, ts_recv)
        else:
            if k < len(self.points):      # correction / removal of earlier points
                self._log_corrections(k, new_points)
            out += self._recompute(k, new_points, tc, ts_recv)
            out += self._status_message(view, tc, ts_recv)
        self.lat.append(time.perf_counter() - t0)
        return out

    def poll(self) -> list[dict]:
        """Finished solves -> recompute from the first provisional point."""
        done = self.solver.poll()
        if not done or self.view is None:
            return []
        k = next((i for i, v in enumerate(self.values) if v.AVG is None), None)
        if k is None:
            return []
        return self._recompute(k, self.points, self.last_tc, time.time())

    # ---- internals -------------------------------------------------------------
    def _log_corrections(self, k: int, new_points: list) -> None:
        old = {p["key"]: (p, self.msgs.get(p["key"])) for p in self.points[k:]}
        newk = {p["key"]: p for p in new_points[k:]}
        for key, (p, m) in old.items():
            q = newk.get(key)
            if q is None or q["hash"] != p["hash"]:
                rec = {"key": list(map(str, key)), "kind": "removed" if q is None else "changed",
                       "inning": p["inning"], "half": p["half"], "last_event_old": p["last_event"],
                       "last_event_new": q["last_event"] if q else None,
                       "wp_before": m["wp_home"] if m else None, "seq_at": self.seq, "overturned":
                       bool(q and q.get("overturned"))}
                self.corrections.append(rec)
                warn("feed_correction", game_id=self.game_id, **{k2: v for k2, v in rec.items() if k2 != "key"},
                     point=str(key))

    def _recompute(self, k: int, new_points: list, tc, ts_recv) -> list[dict]:
        view = self.view
        # roll the tracker back to the start of the PA containing point k
        pa_k = new_points[k]["pa_i"] if k < len(new_points) else len(view.rows)
        k0 = k
        while k0 > 0 and new_points[k0 - 1]["pa_i"] == pa_k:
            k0 -= 1
        if self.tracker is not None:
            self.tracker.restore(min(pa_k, self.tracker.state.rows_done))
        self.points = list(new_points)
        self.values = self.values[:k0]
        out = []
        pc = None
        for i in range(k0, len(new_points)):
            p = new_points[i]
            v, pc = self._value(p, pc)
            self.values.append(v)
            msg = self._message(p, v, tc, ts_recv, i)
            prev = self.msgs.get(p["key"])
            if prev is None or _changed(prev, msg):
                self.seq += 1
                msg["seq"] = self.seq
                self.msgs[p["key"]] = msg
                out.append(msg)
                self.log.append(msg)
        # corrections: WP after
        for c in self.corrections:
            if c.get("wp_after") is None and c["seq_at"] < self.seq:
                key = tuple(int(x) if x.isdigit() else x for x in c["key"])
                m = self.msgs.get(key)
                if m:
                    c["wp_after"] = m["wp_home"]
                    if c["wp_before"] is not None:
                        c["wp_delta"] = m["wp_home"] - c["wp_before"]
        self.order = [p["key"] for p in self.points]
        return out

    def _row(self, i: int) -> dict:
        r = dict(self.view.rows[i])
        for c in ("batter", "pitcher", "starter"):
            r[c] = self.ids.retro(r[c])
        return r

    def _value(self, p: dict, pc):
        view, M, b2 = self.view, self.M, self.b2
        row = self._row(p["pa_i"])
        inn, half = p["inning"], p["half"]
        diff = p["home"] - p["away"]
        phase = p["phase"]
        sgn = 1 if half == 1 else -1
        cidx = CC.C_IDX[(min(p["balls"], 3), min(p["strikes"], 2))]
        tr = self.tracker
        over_now = phase == "end" and (p is view.points[-1]) and view.over
        if phase == "end":
            # state after the PA
            if pc is None and tr is not None:
                pc = tr.context(row, self.ids.retro(p["pitcher"]))
            outs2 = p["outs"]
            half_over = outs2 >= 3
            final = self._final_wp(p, view) if over_now or (view.over and p is view.points[-1]) else None
            if tr is not None:
                tr.commit(row, row["pitcher"])
            if final is not None:
                return PointValue(final, final, final, final), None
            if half_over:
                ni, nh = _half_next(inn, half)
                bst = 0b010 if (self.xrule and ni >= 10) else 0
                v2 = b2.pa_value(ni, nh, 0, bst, diff)
                prog = ST.progress(np.array([ni]), np.array([nh]), np.array([0]))[0]
            else:
                v2 = b2.pa_value(inn, half, outs2, p["bases"], diff)
                prog = ST.progress(np.array([inn]), np.array([half]), np.array([outs2]))[0]
            v2e = M.b2e_value(v2, self.p_pre, prog)
            v3 = None
            if pc is not None and pc.ok:
                v3 = tr.after_values(pc, min(outs2, 2), p["bases"], diff, half_over)
            return PointValue(v3, v2, v2e, None if v3 is None else float(avg(v3, v2e, M.w))), None
        # in-PA points
        if phase in ("start", "sub") and p["n_pitch"] == 0 and (p["outs"], p["bases"]) == (row["outs0"], row["bases0"]) \
                and diff == row["diff0"]:
            st_o, st_b, st_d, at_start = row["outs0"], row["bases0"], row["diff0"], True
        elif p["mid"] or phase in ("pre", "start", "sub"):
            st_o, st_b, st_d, at_start = p["outs"], p["bases"], diff, False
        else:                                   # pitch before any in-PA runner event: first-pitch state
            st_o, st_b = row["outs1"], row["bases1"]
            st_d, at_start = row["diff0"] + sgn * row["rN"], False
        if st_o >= 3:
            st_o = 2
        prog = ST.progress(np.array([inn]), np.array([half]), np.array([st_o]))[0]
        if at_start:
            v2 = b2.pa_value(inn, half, st_o, st_b, st_d)
        else:
            v2 = b2.pitch_value(inn, half, st_o, st_b, st_d, cidx)
        v2e = M.b2e_value(v2, self.p_pre, prog)
        G2 = b2.after(inn, half, st_o, st_b, st_d)
        v3 = None
        G3 = None
        if tr is not None:
            pc = tr.context(row, self.ids.retro(p["pitcher"]))
            if pc.ok:
                try:
                    P, G3 = tr.pitch_parts(pc, row, st_o, st_b, st_d)
                    v3 = tr.wp_state(pc, st_o, st_b, st_d) if at_start else float(P[cidx] @ G3)
                except Exception as e:   # noqa: BLE001 - fall back to B2E
                    warn("b3_value_failed", game_id=self.game_id, err=repr(e))
                    v3 = None
        a = None if v3 is None else float(avg(v3, v2e, M.w))
        pv = PointValue(v3, v2, v2e, a)
        from ..transitions import O_IDX
        for o, dct, douts in ((O_IDX["HR"], pv.if_hr, 0), (O_IDX["K"], pv.if_k, 1)):
            prog_o = ST.progress(np.array([inn]), np.array([half]), np.array([min(st_o + douts, 2)]))[0]
            e2 = M.b2e_value(float(G2[o]), self.p_pre, prog_o)
            dct.update({"B2": float(G2[o]), "B2E": e2})
            if G3 is not None:
                dct.update({"B3": float(G3[o]), "AVG": float(avg(float(G3[o]), e2, M.w))})
        return pv, pc

    def _final_wp(self, p, view) -> float:
        a, h = view.state.score
        return 1.0 if h > a else 0.0 if a > h else 0.5

    def _display(self, v: PointValue) -> tuple[float, str, bool]:
        m = self.model
        if m == "B2":
            return v.B2, "B2", False
        if m == "B2E":
            return v.B2E, "B2E", False
        val = v.AVG if m == "AVG" else v.B3
        if val is None:
            return v.B2E, "B2E", True
        return val, m, False

    def _message(self, p: dict, v: PointValue, tc, ts_recv, i: int) -> dict:
        wp, model, recomputing = self._display(v)
        prev = self.values[i - 1] if i > 0 else None
        wp_prev = self._display(prev)[0] if prev is not None else None
        over = self.view.over and i == len(self.view.points) - 1 and p["phase"] == "end"
        pa, ph = split_pct(wp)
        if over:
            pa, ph = (0, 100) if wp >= 0.5 else (100, 0)
        key = model if model in v.if_hr else "B2E"
        hr = v.if_hr.get(key)
        kk = v.if_k.get(key)
        return {"game_id": self.game_id, "ts_feed": p.get("ts") or tc, "ts_out": round(time.time(), 3),
                "seq": None, "status": self._point_status(over, i),
                "inning": p["inning"], "half": "top" if p["half"] == 0 else "bottom", "outs": min(p["outs"], 3),
                "bases": [bool(p["bases"] & 1), bool(p["bases"] & 2), bool(p["bases"] & 4)],
                "count": [p["balls"], p["strikes"]], "away": p["away"], "home": p["home"],
                "batter": self.view.game.names.get(p["batter"], str(p["batter"])),
                "pitcher": self.view.game.names.get(p["pitcher"], str(p["pitcher"])),
                "last_event": p["last_event"], "wp_home": float(wp), "pct_away": pa, "pct_home": ph,
                "leds_home": self.leds(1.0 if over and ph == 100 else 0.0 if over else wp),
                "wp_if_hr": hr, "wp_if_k": kk,
                "wpa_last": None if wp_prev is None else float(wp - wp_prev),
                "model": model, "info_level": self.info_level, "recomputing": recomputing,
                "key": [str(x) for x in p["key"]], "phase": p["phase"]}

    def _point_status(self, over: bool, i: int) -> str:
        if over:
            return "final"
        st = self.view.game.status
        if i == len(self.view.points) - 1 and st in ("review", "delayed", "suspended"):
            return st
        return "live"

    def _status_message(self, view: GameView, tc, ts_recv) -> list[dict]:
        """Non-play messages: pre-game / delayed / suspended / postponed (no points yet or status change)."""
        st = view.game.status
        if view.points and st in ("live", "review", "final"):
            return []
        last = self.msgs.get(("status",))
        if last is not None and last["status"] == st and (st != "pregame" or last["info_level"] == self.info_level):
            return []
        v = self._pregame_value()
        wp, model, rec = self._display(v)
        pa, ph = split_pct(wp)
        self.seq += 1
        msg = {"game_id": self.game_id, "ts_feed": tc, "ts_out": round(time.time(), 3), "seq": self.seq,
               "status": st, "inning": 1, "half": "top", "outs": 0, "bases": [False] * 3, "count": [0, 0],
               "away": view.state.score[0], "home": view.state.score[1], "batter": None, "pitcher": None,
               "last_event": "", "wp_home": float(wp), "pct_away": pa, "pct_home": ph, "leds_home": self.leds(wp),
               "wp_if_hr": None, "wp_if_k": None, "wpa_last": None, "model": model, "info_level": self.info_level,
               "recomputing": rec, "key": ["status"], "phase": "status"}
        self.msgs[("status",)] = msg
        self.log.append(msg)
        return [msg]

    def _pregame_value(self) -> PointValue:
        v2 = self.b2.pa_value(1, 0, 0, 0, 0)
        v2e = self.M.b2e_value(v2, self.p_pre, 0.0)
        v3 = None
        if self.tracker is not None:
            res = self.solver.get(self.tracker.pregame_key)
            if res is not None:
                from .. import dp_b3 as D3
                v3 = res.wp(1, 0, 0, 0, 0, D3.composite(0, 0), D3.composite(0, 0))
        return PointValue(v3, v2, v2e, None if v3 is None else float(avg(v3, v2e, self.M.w)))

    # ---- views -------------------------------------------------------------
    def timeline(self) -> list[dict]:
        return [self.msgs[k] for k in self.order if k in self.msgs]

    def state(self) -> dict | None:
        tl = self.timeline()
        return tl[-1] if tl else self.msgs.get(("status",))


def _changed(a: dict, b: dict) -> bool:
    for f in ("wp_home", "model", "recomputing", "status", "outs", "bases", "count", "away", "home", "last_event"):
        x, y = a.get(f), b.get(f)
        if isinstance(x, float) and isinstance(y, float):
            if not (abs(x - y) <= 1e-12):
                return True
        elif x != y:
            return True
    return False
