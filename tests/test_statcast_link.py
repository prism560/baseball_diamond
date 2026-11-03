"""Statcast <-> Retrosheet linking on a synthetic game."""
import pandas as pd

from wp import statcast as SC


def test_link_game_and_pas():
    games = pd.DataFrame({"game_id": ["NYA202404010"], "date": [pd.Timestamp("2024-04-01")]})
    pa = pd.DataFrame({"game_id": ["NYA202404010"] * 4, "half": [0, 0, 1, 1], "o": [0, 8, 4, -1],
                       "batter": ["b1", "b2", "h1", "h2"], "pitcher": ["hp", "hp", "vp", "vp"],
                       "starter": ["hp", "hp", "vp", "vp"]})
    sc = pd.DataFrame({"game_pk": [1] * 5, "game_date": ["2024-04-01"] * 5, "inning": [1] * 5,
                       "inning_topbot": ["Top", "Top", "Top", "Bot", "Bot"], "at_bat_number": [1, 1, 2, 3, 4],
                       "batter": [11, 11, 12, 21, 99], "pitcher": [101, 101, 101, 201, 201],
                       "pitch_type": ["FF"] * 5, "release_speed": [95.0] * 5})
    idmap = {11: "b1", 12: "b2", 21: "h1", 101: "hp", 201: "vp"}
    out, rep = SC.link(sc, pa, games, idmap)
    assert rep["games_linked"] == 1
    assert rep["pa_linked"] == 3 and rep["pa_statcast"] == 4      # unmapped batter 99 is not guessed
    assert out.loc[out["at_bat_number"] == 3, "pa_idx"].iloc[0] == 2
