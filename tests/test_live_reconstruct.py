"""State reconstruction on small hand-built feeds (structure as in real StatsAPI feeds)."""
import copy

from wp.live.normalize import normalize
from wp.live.reconstruct import reconstruct
from wp.live.replay import Faults, apply_faults, synthetic_snapshots
from wp.transitions import O_IDX


def pitch(i, code, b, s, outs=0):
    return {"index": i, "type": "pitch", "isPitch": True, "details": {"call": {"code": code}, "code": code},
            "count": {"balls": b, "strikes": s, "outs": outs}, "startTime": f"2025-05-01T20:00:{i:02d}Z"}


def action(i, et, **kw):
    return {"index": i, "type": "action", "isPitch": False, "details": {"eventType": et, "description": et},
            "count": {"balls": 0, "strikes": 0, "outs": 0}, "startTime": f"2025-05-01T20:00:{i:02d}Z", **kw}


def run(rid, start, end, idx, et, out=False):
    return {"movement": {"start": start, "end": end, "isOut": out},
            "details": {"runner": {"id": rid}, "playIndex": idx, "eventType": et}}


def play(ab, inn, top, batter, events, runners, result, complete=True):
    return {"atBatIndex": ab, "about": {"inning": inn, "isTopInning": top, "isComplete": complete,
                                        "startTime": "2025-05-01T20:00:00Z", "endTime": "2025-05-01T20:01:00Z"},
            "matchup": {"batter": {"id": batter}, "pitcher": {"id": 900 if top else 800},
                        "batSide": {"code": "R"}, "pitchHand": {"code": "R"}},
            "result": {"eventType": result, "description": result}, "playEvents": events, "runners": runners}


def feed(plays, gtype="R", final=False):
    lineup = lambda base: {f"ID{base + s}": {"person": {"id": base + s}, "battingOrder": f"{s}00"} for s in range(1, 10)}
    return {"gamePk": 1, "metaData": {"timeStamp": "20250501_200000"},
            "gameData": {"game": {"pk": 1, "type": gtype}, "datetime": {"officialDate": "2025-05-01"},
                         "status": {"codedGameState": "F" if final else "I", "statusCode": "F" if final else "I",
                                    "detailedState": "Final" if final else "In Progress"},
                         "teams": {"away": {"id": 147}, "home": {"id": 111}}, "players": {}, "probablePitchers": {}},
            "liveData": {"plays": {"allPlays": plays},
                         "boxscore": {"teams": {"away": {"players": lineup(100), "pitchers": [800]},
                                                "home": {"players": lineup(200), "pitchers": [900]}}},
                         "linescore": {"teams": {"away": {"runs": 0}, "home": {"runs": 0}}}}}


def test_single_then_stolen_base_mid_pa_and_strikeout():
    p0 = play(0, 1, True, 101, [pitch(0, "X", 0, 0)], [run(101, None, "1B", 0, "single")], "single")
    p1 = play(1, 1, True, 102, [pitch(0, "B", 1, 0), action(1, "stolen_base_2b"), pitch(2, "S", 1, 1),
                                pitch(3, "C", 1, 2, 0), pitch(4, "S", 1, 3, 1)],
              [run(101, "1B", "2B", 1, "stolen_base_2b"), run(102, None, None, 4, "strikeout", True)], "strikeout")
    v = reconstruct(normalize(feed([p0, p1])))
    r = v.rows[1]
    assert (r["outs0"], r["bases0"], r["outs1"], r["bases1"]) == (0, 1, 0, 1)
    assert r["o"] == O_IDX["K"] and (r["outs2"], r["bases2"]) == (1, 2)
    assert r["mid"] == "1:0:2:0" and r["pitches"] == "BSCS"
    phases = [p["phase"] for p in v.points if p["pa_i"] == 1]
    assert phases == ["start", "pitch", "mid", "pitch", "pitch", "pitch_end", "end"]
    assert v.points[-1]["outs"] == 1 and v.points[-1]["bases"] == 2


def test_automatic_ibb_no_pitch_events():
    ev = [{"index": i, "type": "no_pitch", "details": {"call": {"code": "VB"}, "code": "VB"},
           "count": {"balls": i + 1, "strikes": 0, "outs": 0}} for i in range(4)]
    p0 = play(0, 1, True, 101, ev, [run(101, None, "1B", 3, "intent_walk")], "intent_walk")
    v = reconstruct(normalize(feed([p0])))
    r = v.rows[0]
    assert r["o"] == O_IDX["IBB"] and r["pitches"] == "VVVV" and r["bases1"] == 0 and r["bases2"] == 1


def test_automatic_runner_regular_season_only():
    plays = []
    for inn in range(1, 10):
        for top in (True, False):
            b = 101 if top else 201
            plays.append(play(len(plays), inn, top, b, [pitch(0, "X", 0, 0, 0)],
                              [run(b, None, None, 0, "field_out", True)], "field_out"))
    p_ex = play(len(plays), 10, True, 102, [action(0, "runner_placed", player={"id": 109}, base=2),
                                            pitch(1, "B", 1, 0)], [], None, complete=False)
    v = reconstruct(normalize(feed(plays + [p_ex])))
    assert v.rows[-1]["bases0"] == 2
    # the play appears before its runner_placed event: the runner is already there
    p_early = copy.deepcopy(p_ex)
    p_early["playEvents"] = []
    v2 = reconstruct(normalize(feed(plays + [p_early])))
    assert v2.points[-1]["bases"] == 2 and v2.points[-1]["hash"] == v.points[[p["phase"] for p in v.points].index("start", len(v2.points) - 1)]["hash"]
    # postseason: no automatic runner
    v3 = reconstruct(normalize(feed(plays + [p_early], gtype="D")))
    assert v3.points[-1]["bases"] == 0


def test_pinch_runner_takes_the_base():
    p0 = play(0, 1, True, 101, [pitch(0, "X", 0, 0)], [run(101, None, "1B", 0, "single")], "single")
    p1 = play(1, 1, True, 102, [action(0, "offensive_substitution", player={"id": 150}, replacedPlayer={"id": 101},
                                       battingOrder="101", base=1, position={"code": "12"}),
                                pitch(1, "X", 0, 0)],
              [run(150, "1B", "3B", 1, "single"), run(102, None, "1B", 1, "single")], "single")
    v = reconstruct(normalize(feed([p0, p1])))
    assert v.rows[1]["bases2"] == 0b101 and v.lineups["away"][1] == 150


def test_snapshots_dedupe_and_faults_keep_final():
    p0 = play(0, 1, True, 101, [pitch(0, "B", 1, 0), pitch(1, "C", 1, 1), pitch(2, "X", 1, 1)],
              [run(101, None, "1B", 2, "single")], "single")
    p1 = play(1, 1, True, 102, [pitch(0, "B", 1, 0), pitch(1, "X", 1, 0)],
              [run(101, "1B", "3B", 1, "double"), run(102, None, "2B", 1, "double")], "double")
    full = feed([p0, p1], final=True)
    snaps = synthetic_snapshots(full)
    final_pts = reconstruct(normalize(full)).points
    keys = {p["key"]: p["hash"] for p in final_pts}
    # every point of every prefix snapshot is identical to the final one (no false corrections)
    for s in snaps:
        for p in reconstruct(normalize(s)).points:
            assert keys[p["key"]] == p["hash"], p
    seq = apply_faults(snaps, Faults(duplicate=0.5, reorder=0.3, drop=0.3, corrections=1, net_errors=0.2, seed=3))
    assert seq[-1] is snaps[-1]
