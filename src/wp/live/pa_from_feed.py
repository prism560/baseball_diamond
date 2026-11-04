"""Completed game feed -> rows of the PA table (wp.transitions.build_pa_table format).

Used for (1) the daily rating / Elo update during the season (Retrosheet is
published only after the season) and (2) the cross-check against the
Retrosheet-based PA table (wp.live.crosscheck_feed).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ids import IdMap, team_code
from .normalize import NormGame
from .reconstruct import GameView

PA_COLS = ["game_id", "half", "inning", "outs0", "bases0", "diff0", "outs1", "bases1", "rN", "o", "outs2",
           "bases2", "r2", "pitches", "mid", "batter", "pitcher", "slot", "starter"]


def game_id(g: NormGame) -> str:
    """Retrosheet-style id HHHYYYYMMDDN (N from the schedule's doubleheader game number)."""
    return f"{team_code(g.home_id)}{g.date.replace('-', '')}{getattr(g, 'dh_number', 0)}"


def pa_rows(view: GameView, ids: IdMap, gid: str | None = None, complete_only: bool = True) -> pd.DataFrame:
    gid = gid or game_id(view.game)
    recs = []
    for r in view.rows:
        if complete_only and r["o"] is None:
            continue
        recs.append((gid, r["half"], r["inning"], r["outs0"], r["bases0"], r["diff0"], r["outs1"], r["bases1"],
                     r["rN"], -2 if r["o"] is None else r["o"], r["outs2"], r["bases2"], r["r2"], r["pitches"],
                     r["mid"], ids.retro(r["batter"]), ids.retro(r["pitcher"]), r["slot"],
                     ids.retro(r["starter"])))
    df = pd.DataFrame(recs, columns=PA_COLS)
    for c in ("half", "outs0", "bases0", "outs1", "bases1", "rN", "outs2", "bases2", "r2", "slot", "o"):
        df[c] = df[c].astype(np.int8)
    df["inning"] = df["inning"].astype(np.int16)
    df["diff0"] = df["diff0"].astype(np.int16)
    df["season"] = np.int16(int(view.game.date[:4])) if view.game.date else np.int16(0)
    return df


def game_row(view: GameView, gid: str | None = None) -> dict:
    """One row in the games-table format used by Elo / ratings (wp.games)."""
    g = view.game
    a, h = view.state.score
    return {"game_id": gid or game_id(g), "season": int(g.date[:4]), "date": pd.Timestamp(g.date),
            "visitor": team_code(g.away_id), "home": team_code(g.home_id), "site": team_code(g.home_id),
            "vis_score": a, "home_score": h, "home_win": int(h > a), "gametype": g.game_type,
            "game_pk": g.pk, "included": g.status == "final", "forfeit": ""}
