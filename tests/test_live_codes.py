"""Mapping tables of the live normalizer: every entry, unknown-code fallbacks, output rules."""
import json
from pathlib import Path

import pytest

from wp.live import codes
from wp.live import log as LG
from wp.live.output import LedHysteresis, split_pct
from wp.transitions import O_IDX, OUTCOMES

META = Path("data/external/meta")

# ---- pitch codes -----------------------------------------------------------------------
PITCH_EXPECT = {
    "B": "ball", "*B": "ball", "P": "ball", "I": "intentional_ball",
    "V": "auto_ball", "VP": "auto_ball", "VC": "auto_ball", "VS": "auto_ball", "VB": "auto_ball",
    "C": "called_strike", "K": "called_strike", "A": "auto_strike", "AC": "auto_strike", "AB": "auto_strike",
    "S": "swinging_strike", "W": "swinging_strike", "M": "swinging_strike", "Q": "swinging_strike",
    "T": "swinging_strike", "O": "foul_bunt", "F": "foul", "R": "foul", "L": "foul_bunt", "H": "hbp",
    "X": "in_play", "D": "in_play", "E": "in_play", "Y": "in_play", "J": "in_play", "Z": "in_play",
    "N": "none", ".": "none", "PSO": "none", "1": "none", "2": "none", "3": "none",
    "+1": "none", "+2": "none", "+3": "none",
}


@pytest.mark.parametrize("code,cls", sorted(PITCH_EXPECT.items()))
def test_pitch_code(code, cls):
    assert codes.pitch_class(code)[0] == cls


@pytest.mark.parametrize("code", [c for c, v in codes.PITCH.items() if v[0] != "none"])
def test_pitch_retro_char_counts_like_retrosheet(code):
    """Retrosheet character of every real pitch is a pitch code of the count chain."""
    from wp.count_chain import CHAR_EV
    ch = codes.PITCH[code][1]
    assert ch in CHAR_EV or ch in "BV"


def test_unknown_pitch_code_is_ball_and_warns():
    n = LG.COUNTS["unknown_pitch_code"]
    assert codes.pitch_class("??") == codes.PITCH_DEFAULT
    assert LG.COUNTS["unknown_pitch_code"] == n + 1


@pytest.mark.skipif(not (META / "pitchCodes.json").exists(), reason="StatsAPI meta not fetched")
def test_every_official_pitch_code_mapped():
    official = {r["code"] for r in json.loads((META / "pitchCodes.json").read_text())}
    assert official <= set(codes.PITCH), official - set(codes.PITCH)


# ---- results ---------------------------------------------------------------------------
@pytest.mark.parametrize("et,o", sorted(codes.RESULT.items()))
def test_result_type(et, o):
    assert codes.result_outcome(et, None, False) == O_IDX[o]


@pytest.mark.parametrize("et", sorted(codes.NON_PA_RESULT))
def test_runner_event_play_is_not_a_pa_outcome(et):
    assert codes.result_outcome(et, None, False) is None


def test_result_categories_are_model_outcomes():
    assert set(codes.RESULT.values()) <= set(OUTCOMES) - {"INT"}


@pytest.mark.parametrize("end,out,o", [("1B", False, "1B"), ("2B", False, "2B"), ("3B", False, "3B"),
                                       ("score", False, "HR"), (None, True, "OUT"), (None, False, "OUT")])
def test_unknown_result_follows_batter(end, out, o):
    assert codes.result_outcome("brand_new_type", end, out) == O_IDX[o]


@pytest.mark.skipif(not (META / "eventTypes.json").exists(), reason="StatsAPI meta not fetched")
def test_every_official_event_type_classified():
    official = {r["code"] for r in json.loads((META / "eventTypes.json").read_text())}
    known = set(codes.RESULT) | codes.NON_PA_RESULT | codes.BY_BATTER_END | codes.RUNNER_EVENTS | \
        set(codes.SUBSTITUTIONS) | codes.IGNORED_ACTIONS | {codes.RUNNER_PLACED}
    assert official <= known, official - known


# ---- actions ---------------------------------------------------------------------------
@pytest.mark.parametrize("et", sorted(codes.RUNNER_EVENTS))
def test_runner_actions(et):
    assert codes.action_kind(et) == "runner"


@pytest.mark.parametrize("et,k", sorted(codes.SUBSTITUTIONS.items()))
def test_substitutions(et, k):
    assert codes.action_kind(et) == "sub:" + k


def test_runner_placed_and_unknown_action():
    assert codes.action_kind("runner_placed") == "runner_placed"
    n = LG.COUNTS["unknown_action_type"]
    assert codes.action_kind("something_new") == "ignore"
    assert LG.COUNTS["unknown_action_type"] == n + 1


# ---- game status -----------------------------------------------------------------------
@pytest.mark.parametrize("st,exp", [
    ({"codedGameState": "S", "statusCode": "S", "detailedState": "Scheduled"}, "scheduled"),
    ({"codedGameState": "P", "statusCode": "P", "detailedState": "Pre-Game"}, "pregame"),
    ({"codedGameState": "P", "statusCode": "PW", "detailedState": "Warmup"}, "pregame"),
    ({"codedGameState": "P", "statusCode": "PR", "detailedState": "Delayed Start: Rain"}, "delayed"),
    ({"codedGameState": "I", "statusCode": "I", "detailedState": "In Progress"}, "live"),
    ({"codedGameState": "I", "statusCode": "IR", "detailedState": "Delayed: Rain"}, "delayed"),
    ({"codedGameState": "I", "statusCode": "IH", "detailedState": "Instant Replay"}, "review"),
    ({"codedGameState": "M", "statusCode": "MF", "detailedState": "Manager challenge"}, "review"),
    ({"codedGameState": "N", "statusCode": "NJ", "detailedState": "Umpire Challenge"}, "review"),
    ({"codedGameState": "T", "statusCode": "TR", "detailedState": "Suspended: Rain"}, "suspended"),
    ({"codedGameState": "U", "statusCode": "U", "detailedState": "Suspended"}, "suspended"),
    ({"codedGameState": "F", "statusCode": "F", "detailedState": "Final"}, "final"),
    ({"codedGameState": "O", "statusCode": "O", "detailedState": "Game Over"}, "final"),
    ({"codedGameState": "D", "statusCode": "DR", "detailedState": "Postponed: Rain"}, "postponed"),
    ({"codedGameState": "C", "statusCode": "CR", "detailedState": "Cancelled: Rain"}, "cancelled"),
])
def test_game_status(st, exp):
    assert codes.game_status(st) == exp


@pytest.mark.skipif(not (META / "gameStatus.json").exists(), reason="StatsAPI meta not fetched")
def test_every_official_status_mapped():
    for r in json.loads((META / "gameStatus.json").read_text()):
        if r["codedGameState"] in ("X", "W"):
            continue
        assert codes.game_status(r) != "unknown", r


# ---- output rules ----------------------------------------------------------------------
@pytest.mark.parametrize("wp", [0.0, 0.004999, 0.005, 0.12345, 0.5, 0.995, 0.999, 1.0])
def test_split_sums_to_100(wp):
    a, h = split_pct(wp)
    assert a + h == 100 and h == int(100 * wp + 0.5)


def test_led_hysteresis_no_flicker():
    led = LedHysteresis(40, 0.3)
    assert led(0.5) == 20
    # oscillating around the 20/21 boundary (x = 20.5) does not flicker
    for wp in (0.5130, 0.5120, 0.5135, 0.5120):
        assert led(wp) == 20
    assert led(0.525) == 21          # clearly past the boundary
    assert led(0.512) == 21          # back near the boundary: stays
    assert led(0.50) == 20
    assert led(1.0) == 40 and led(0.0) == 0
