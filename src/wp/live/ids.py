"""MLBAM (live feed) -> Retrosheet player ids, via the Chadwick register (B4 mapping).

Unmapped players get the pseudo id ``"mlb<id>"``: the rating store has no rows
for it, so its rates are the league prior (full shrinkage); handedness then
comes from the feed.  Every unmapped id is logged once per process.
"""
from __future__ import annotations

from pathlib import Path

from .. import statcast
from .log import warn

REGISTER = Path("data/raw/register")
# Retrosheet team code -> MLB team id (2025; /api/v1/teams?sportId=1&season=2025)
RETRO_TEAM = {"ANA": 108, "ARI": 109, "BAL": 110, "BOS": 111, "CHN": 112, "CIN": 113, "CLE": 114,
              "COL": 115, "DET": 116, "HOU": 117, "KCA": 118, "LAN": 119, "WAS": 120, "NYN": 121,
              "ATH": 133, "PIT": 134, "SDN": 135, "SEA": 136, "SFN": 137, "SLN": 138, "TBA": 139,
              "TEX": 140, "TOR": 141, "MIN": 142, "PHI": 143, "ATL": 144, "CHA": 145, "MIA": 146,
              "NYA": 147, "MIL": 158}
TEAM_RETRO = {v: k for k, v in RETRO_TEAM.items()}


class IdMap:
    def __init__(self, register_dir: Path = REGISTER):
        if not any(Path(register_dir).glob("people-*.csv")):
            statcast.fetch_register(Path(register_dir))
        self.m = statcast.mlbam_to_retro(Path(register_dir))
        self.unmapped: set[int] = set()

    def __len__(self) -> int:
        return len(self.m)

    def retro(self, mlbam: int | None) -> str:
        if mlbam is None:
            return "unknown"
        r = self.m.get(int(mlbam))
        if r is None:
            if int(mlbam) not in self.unmapped:
                self.unmapped.add(int(mlbam))
                warn("unmapped_player", mlbam=int(mlbam))
            return f"mlb{int(mlbam)}"
        return r


def team_code(mlb_id: int) -> str:
    return TEAM_RETRO.get(int(mlb_id), f"T{mlb_id}")
