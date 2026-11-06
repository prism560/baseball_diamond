"""Feed (StatsAPI GUMBO live feed) -> normalized game events.

Field meanings were confirmed on real 2025 feeds (data/external/feeds/, fetched
2026-11-04):

  gameData.game.pk / .type (R regular, F wild card, D division, L LCS, W WS)
  gameData.datetime.officialDate            game date (local)
  gameData.status                           -> codes.game_status
  gameData.teams.{away,home}.id             MLB team id
  gameData.probablePitchers.{away,home}.id  probable starters (before lineups)
  gameData.players["ID<id>"].batSide/pitchHand.code
  liveData.boxscore.teams.{side}.players["ID<id>"].battingOrder   "700" = slot 7 starter,
                                            "701" = first substitute in slot 7
  liveData.boxscore.teams.{side}.pitchers[0]                       starting pitcher
  liveData.plays.allPlays[i]                one plate appearance (or runner-event play)
      .atBatIndex, .about.{inning, isTopInning, isComplete, startTime}
      .matchup.{batter, pitcher}.id, .matchup.batSide/pitchHand.code
      .result.eventType                     -> codes.result_outcome (only when complete)
      .playEvents[j]                        j = .index; type pitch | action | pickoff | no_pitch | stepoff
          .details.call.code / .details.code   pitch call (codes.PITCH)
          .details.eventType                action type (codes.action_kind)
          .count.{balls, strikes, outs}     count *after* the event
          .player.id, .replacedPlayer.id, .battingOrder, .position.code   substitutions
          .reviewDetails                    challenge / review (isOverturned)
      .runners[k].movement.{start, end, isOut}, .details.{runner.id, playIndex, eventType}
          playIndex = index of the playEvent that caused the movement

Event key = (atBatIndex, playEvent index); the PA result has key (atBatIndex, "result").
Each event carries a content hash over the fields that drive the state, so the session
layer can tell duplicates (same key, same hash) from corrections (same key, new hash).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from . import codes


@dataclass
class Move:
    runner: int
    start: str | None          # None (batter), "1B", "2B", "3B"
    end: str | None            # None (out or stays), "1B", "2B", "3B", "score"
    is_out: bool
    event_type: str | None


@dataclass
class Event:
    key: tuple
    kind: str                  # pitch | pickoff | runner | runner_placed | sub:<k> | ignore
    code: str | None           # pitch call code or action eventType
    pclass: str | None = None  # pitch class (codes.PITCH)
    rchar: str = ""            # Retrosheet pitch-string character(s)
    balls: int = 0             # count after the event (feed)
    strikes: int = 0
    outs: int = 0
    moves: list = field(default_factory=list)
    player: int | None = None
    replaced: int | None = None
    batting_order: str | None = None
    position: str | None = None
    overturned: bool = False
    base: int | None = None    # substitution of a runner / automatic runner: base number
    ts: str | None = None
    description: str = ""
    hash: str = ""


@dataclass
class Play:
    ab: int
    inning: int
    half: int                  # 0 top, 1 bottom
    batter: int
    pitcher: int
    bat_side: str
    pitch_hand: str
    complete: bool
    result_type: str | None
    result_desc: str
    events: list
    result_moves: list         # movements whose playIndex is missing / past the last event
    ts: str | None
    hash: str = ""


@dataclass
class NormGame:
    pk: int
    game_type: str
    date: str
    status: str
    status_raw: dict
    away_id: int
    home_id: int
    lineups: dict              # side -> {slot(1..9): mlbam id} starting lineup (may be empty)
    starters: dict             # side -> mlbam id (boxscore pitchers[0] or probable)
    probable: dict
    hands: dict                # mlbam id -> (bats, throws)
    plays: list
    timecode: str | None
    final_score: tuple | None  # (away, home) when final
    names: dict = field(default_factory=dict)
    scheduled_innings: int = 9


def _h(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _moves(play: dict) -> dict:
    out: dict = {}
    for r in play.get("runners", []):
        m, d = r.get("movement", {}), r.get("details", {})
        mv = Move(int(d.get("runner", {}).get("id", 0)), m.get("start"), m.get("end"), bool(m.get("isOut")),
                  d.get("eventType"))
        out.setdefault(d.get("playIndex"), []).append(mv)
    return out


def normalize(feed: dict) -> NormGame:
    gd, ld = feed["gameData"], feed["liveData"]
    pk = int(feed.get("gamePk") or gd["game"]["pk"])
    ctx = {"game_pk": pk}
    status = codes.game_status(gd.get("status", {}), ctx)
    box = ld.get("boxscore", {}).get("teams", {})
    lineups, starters = {}, {}
    for side in ("away", "home"):
        lu = {}
        for p in box.get(side, {}).get("players", {}).values():
            bo = p.get("battingOrder")
            if bo and str(bo).endswith("00"):
                lu[int(str(bo)[0])] = int(p["person"]["id"])
        lineups[side] = lu if len(lu) == 9 else {}
        pit = box.get(side, {}).get("pitchers", [])
        prob = gd.get("probablePitchers", {}).get(side, {})
        starters[side] = int(pit[0]) if pit else (int(prob["id"]) if prob.get("id") else None)
    probable = {s: (gd.get("probablePitchers", {}).get(s, {}) or {}).get("id") for s in ("away", "home")}
    hands, names = {}, {}
    for p in gd.get("players", {}).values():
        hands[int(p["id"])] = ((p.get("batSide") or {}).get("code", "R"), (p.get("pitchHand") or {}).get("code", "R"))
        names[int(p["id"])] = p.get("boxscoreName") or p.get("fullName", "")
    plays = []
    for ap in ld.get("plays", {}).get("allPlays", []):
        ab = int(ap["atBatIndex"])
        about, mu, res = ap["about"], ap["matchup"], ap.get("result", {})
        mv = _moves(ap)
        evs = []
        idxs = set()
        for e in ap.get("playEvents", []):
            j = int(e["index"])
            idxs.add(j)
            det = e.get("details", {})
            cnt = e.get("count", {})
            ectx = {**ctx, "ab": ab, "index": j}
            ev = Event((ab, j), "ignore", None, balls=int(cnt.get("balls", 0)), strikes=int(cnt.get("strikes", 0)),
                       outs=int(cnt.get("outs", 0)), moves=mv.get(j, []), ts=e.get("startTime"),
                       description=det.get("description", ""),
                       overturned=bool((e.get("reviewDetails") or {}).get("isOverturned")))
            typ = e.get("type")
            nocode = (det.get("call") or {}).get("code") or det.get("code")
            # automatic balls/strikes (pitch timer, automatic intentional walk) are no_pitch
            # events with call codes VB/VP/VC/AC/AB; Retrosheet writes them as V/A pitches
            auto = typ == "no_pitch" and codes.PITCH.get(nocode, ("none", ""))[0] != "none"
            if e.get("isPitch") or typ == "pitch" or auto:
                code = (det.get("call") or {}).get("code") or det.get("code")
                ev.kind, ev.code = "pitch", code
                ev.pclass, ev.rchar = codes.pitch_class(code, ectx)
                if ev.pclass == "none":
                    ev.kind = "ignore"
            elif typ == "pickoff":
                ev.kind, ev.code = "pickoff", det.get("code")
                ev.rchar = codes.PITCH.get(det.get("code"), ("none", ""))[1]
            elif typ in ("no_pitch", "stepoff"):
                ev.kind, ev.code = "ignore", typ
            else:
                et = det.get("eventType")
                ev.code = et
                ev.kind = codes.action_kind(et, ectx)
                if ev.kind.startswith("sub:"):
                    ev.player = (e.get("player") or {}).get("id")
                    ev.replaced = (e.get("replacedPlayer") or {}).get("id")
                    ev.batting_order = e.get("battingOrder")
                    ev.position = (e.get("position") or {}).get("code")
                    ev.base = e.get("base")
                elif ev.kind == "runner_placed":
                    # automatic runner: no runner movement in the feed; the runner is .player, base .base
                    pid = (e.get("player") or {}).get("id")
                    b = {1: "1B", 2: "2B", 3: "3B"}.get(e.get("base", 2), "2B")
                    ev.player, ev.base = pid, e.get("base", 2)
                    if pid and not ev.moves:
                        ev.moves = [Move(int(pid), None, b, False, "runner_placed")]
            # runner movements are not part of the event hash: their effect is in the state
            # carried by the point hash (wp.live.reconstruct), and result movements may be
            # published after the pitch itself
            ev.hash = _h([ev.kind, ev.code, ev.balls, ev.strikes, ev.outs,
                          ev.player, ev.replaced, ev.batting_order, ev.position, ev.base])
            evs.append(ev)
        extra = [m for k, ms in mv.items() if k not in idxs for m in ms]
        complete = bool(about.get("isComplete"))
        rt = res.get("eventType") if complete else None
        pl = Play(ab, int(about["inning"]), 0 if about["isTopInning"] else 1, int(mu["batter"]["id"]),
                  int(mu["pitcher"]["id"]), (mu.get("batSide") or {}).get("code", "R"),
                  (mu.get("pitchHand") or {}).get("code", "R"), complete, rt, res.get("description", ""),
                  evs, extra, about.get("startTime"))
        pl.hash = _h([complete, rt, pl.batter, pl.pitcher, [(m.runner, m.start, m.end, m.is_out) for m in extra]])
        plays.append(pl)
    final = None
    if status == "final":
        ls = ld.get("linescore", {}).get("teams", {})
        final = (int(ls.get("away", {}).get("runs", 0)), int(ls.get("home", {}).get("runs", 0)))
    return NormGame(pk, gd["game"].get("type", "R"), gd.get("datetime", {}).get("officialDate", ""), status,
                    gd.get("status", {}), int(gd["teams"]["away"]["id"]), int(gd["teams"]["home"]["id"]),
                    lineups, starters, probable, hands, plays, feed.get("metaData", {}).get("timeStamp"), final,
                    names, int(ld.get("linescore", {}).get("scheduledInnings", 9) or 9))
