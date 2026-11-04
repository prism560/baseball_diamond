"""Mapping tables: StatsAPI feed codes -> model categories.

Sources (fetched 2026-11-04, copies in data/external/meta/):
  GET /api/v1/pitchCodes   pitch call codes (playEvents[].details.call.code)
  GET /api/v1/eventTypes   result / action event types (result.eventType, details.eventType)
  GET /api/v1/gameStatus   game status codes (gameData.status.codedGameState / statusCode)
Codes observed in the 2025 feeds are a subset of these lists (see tests/test_live_codes.py).

Unknown codes never stop a game: the lookup returns a safe default and emits a
structured warning (``wp.live.log.warn``) that is later forwarded to Sentry.
"""
from __future__ import annotations

from ..transitions import O_IDX
from .log import warn

# ---- pitch calls -------------------------------------------------------------
# feed code -> (pitch class, Retrosheet pitch character)
# classes: ball, called_strike, swinging_strike, foul, foul_bunt, hbp, in_play,
#          intentional_ball, auto_ball, auto_strike, none (not a pitch: pickoff throws, step-offs)
PITCH = {
    "B": ("ball", "B"), "*B": ("ball", "B"), "P": ("ball", "P"),
    "I": ("intentional_ball", "I"),
    "V": ("auto_ball", "V"), "VP": ("auto_ball", "V"), "VC": ("auto_ball", "V"),
    "VS": ("auto_ball", "V"), "VB": ("auto_ball", "V"),   # VB: automatic ball of an intentional walk
    "C": ("called_strike", "C"), "K": ("called_strike", "K"),
    "A": ("auto_strike", "A"), "AC": ("auto_strike", "A"), "AB": ("auto_strike", "A"),
    "S": ("swinging_strike", "S"), "W": ("swinging_strike", "S"), "M": ("swinging_strike", "M"),
    "Q": ("swinging_strike", "Q"), "T": ("swinging_strike", "T"), "O": ("foul_bunt", "O"),
    "F": ("foul", "F"), "R": ("foul", "R"),
    "L": ("foul_bunt", "L"),
    "H": ("hbp", "H"),
    "X": ("in_play", "X"), "D": ("in_play", "X"), "E": ("in_play", "X"),
    "Y": ("in_play", "Y"), "J": ("in_play", "Y"), "Z": ("in_play", "Y"),
    # not pitches (count unchanged)
    "N": ("none", ""), ".": ("none", ""), "PSO": ("none", ""),
    "1": ("none", "1"), "2": ("none", "2"), "3": ("none", "3"),
    "+1": ("none", "1"), "+2": ("none", "2"), "+3": ("none", "3"),
}
PITCH_DEFAULT = ("ball", "B")       # unknown pitch code: treat as a ball (count +1 ball)


def pitch_class(code: str | None, ctx: dict | None = None) -> tuple[str, str]:
    if code in PITCH:
        return PITCH[code]
    warn("unknown_pitch_code", code=code, **(ctx or {}))
    return PITCH_DEFAULT


# ---- plate-appearance results -----------------------------------------------
# result.eventType -> model outcome (transitions.OUTCOMES); None = not a PA outcome
# (the play ended with a runner event: INT if pitches were thrown, else the half ended in N)
RESULT = {
    "strikeout": "K", "strike_out": "K", "strikeout_double_play": "K", "strikeout_triple_play": "K",
    "walk": "BB", "intent_walk": "IBB", "hit_by_pitch": "HBP",
    "single": "1B", "double": "2B", "triple": "3B", "home_run": "HR",
    "field_out": "OUT", "force_out": "OUT", "grounded_into_double_play": "OUT",
    "grounded_into_triple_play": "OUT", "double_play": "OUT", "triple_play": "OUT",
    "fielders_choice": "OUT", "fielders_choice_out": "OUT", "sac_fly": "OUT",
    "sac_fly_double_play": "OUT", "sac_bunt": "OUT", "sac_bunt_double_play": "OUT",
    "batter_interference": "OUT",
    "field_error": "ROE", "catcher_interf": "ROE", "fielder_interference": "ROE",
}
NON_PA_RESULT = {
    "caught_stealing", "caught_stealing_2b", "caught_stealing_3b", "caught_stealing_home",
    "pickoff_1b", "pickoff_2b", "pickoff_3b", "pickoff_error_1b", "pickoff_error_2b", "pickoff_error_3b",
    "pickoff_caught_stealing_2b", "pickoff_caught_stealing_3b", "pickoff_caught_stealing_home",
    "stolen_base", "stolen_base_2b", "stolen_base_3b", "stolen_base_home",
    "wild_pitch", "passed_ball", "balk", "forced_balk", "other_out", "other_advance", "error",
    "runner_double_play", "cs_double_play", "defensive_indiff", "runner_interference",
    "game_advisory", "injury", "ejection",
}
# fan interference etc.: outcome follows the batter's own movement
BY_BATTER_END = {"fan_interference", "batter_turn"}
END_TO_O = {"1B": "1B", "2B": "2B", "3B": "3B", "score": "HR"}


def result_outcome(event_type: str | None, batter_end: str | None, batter_out: bool,
                   ctx: dict | None = None) -> int | None:
    """Outcome index (transitions.O_IDX) or None for a runner-event play."""
    if event_type in RESULT:
        return O_IDX[RESULT[event_type]]
    if event_type in NON_PA_RESULT:
        return None
    if event_type not in BY_BATTER_END:
        warn("unknown_result_type", event_type=event_type, **(ctx or {}))
    if batter_out:
        return O_IDX["OUT"]
    if batter_end in END_TO_O:
        return O_IDX[END_TO_O[batter_end]]
    return O_IDX["OUT"]


# ---- action events (playEvents[].details.eventType of non-pitch events) -----
RUNNER_EVENTS = {
    "stolen_base", "stolen_base_2b", "stolen_base_3b", "stolen_base_home",
    "caught_stealing", "caught_stealing_2b", "caught_stealing_3b", "caught_stealing_home",
    "pickoff_1b", "pickoff_2b", "pickoff_3b", "pickoff_error_1b", "pickoff_error_2b", "pickoff_error_3b",
    "pickoff_caught_stealing_2b", "pickoff_caught_stealing_3b", "pickoff_caught_stealing_home",
    "wild_pitch", "passed_ball", "balk", "forced_balk", "defensive_indiff", "other_advance", "other_out",
    "error", "runner_double_play", "cs_double_play", "runner_interference",
}
RUNNER_PLACED = "runner_placed"            # extra-inning automatic runner
SUBSTITUTIONS = {"pitching_substitution": "pitcher", "offensive_substitution": "offense",
                 "defensive_substitution": "defense", "defensive_switch": "switch",
                 "pitcher_switch": "pitcher_hand"}
IGNORED_ACTIONS = {"batter_timeout", "mound_visit", "game_advisory", "injury", "ejection", "at_bat_start",
                   "os_ruling_pending_prior", "os_ruling_pending_primary", "umpire_substitution",
                   "pitcher_step_off", "no_pitch", "fan_interference"}


def action_kind(event_type: str | None, ctx: dict | None = None) -> str:
    """'runner', 'runner_placed', 'sub:<kind>', 'ignore' (unknown -> 'ignore' + warning;
    runner movements attached to the event are applied regardless of its kind)."""
    if event_type in RUNNER_EVENTS:
        return "runner"
    if event_type == RUNNER_PLACED:
        return "runner_placed"
    if event_type in SUBSTITUTIONS:
        return "sub:" + SUBSTITUTIONS[event_type]
    if event_type in IGNORED_ACTIONS or event_type is None:
        return "ignore"
    warn("unknown_action_type", event_type=event_type, **(ctx or {}))
    return "ignore"


# ---- game status ------------------------------------------------------------
# codedGameState -> status of the output message
STATUS = {"S": "scheduled", "P": "pregame", "I": "live", "M": "live", "N": "live",
          "D": "postponed", "C": "cancelled", "O": "final", "F": "final",
          "T": "suspended", "U": "suspended", "Q": "final", "R": "final"}


def game_status(status: dict, ctx: dict | None = None) -> str:
    """scheduled | pregame | live | delayed | review | suspended | final | postponed | cancelled | unknown."""
    coded = status.get("codedGameState")
    code = status.get("statusCode", "") or ""
    det = status.get("detailedState", "") or ""
    if coded not in STATUS:
        warn("unknown_game_status", coded=coded, code=code, detailed=det, **(ctx or {}))
        return "unknown"
    s = STATUS[coded]
    if coded == "P" and (code.startswith("P") and code not in ("P", "PW")):
        return "delayed"                    # "Delayed Start: ..."
    if coded == "I" and code not in ("I", "IH") and det.startswith("Delayed"):
        return "delayed"
    if coded in ("M", "N") or code == "IH":
        return "review"
    return s
