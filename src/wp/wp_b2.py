"""Lookup API for the B2 tables.

    from wp.wp_b2 import PitchWP, PAWP
    pw = PitchWP.load("models/wp_b2_pitch.parquet")
    pw.wp(inning=9, half=1, outs=2, bases=0b111, diff=0, balls=3, strikes=2)   # P(home wins)
    pw.split(...)  -> (visitor %, home %) rounded for display

Innings >= 10 use the extra-inning table (runner-on-second era); diff outside
the stored range is clipped (values there are within 1e-6 of decided).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class PitchWP:
    arr: np.ndarray      # [inning 1..10, half, outs, bases, diff, balls, strikes]
    R: int

    @classmethod
    def load(cls, path: str | Path) -> "PitchWP":
        t = pd.read_parquet(path)
        R = int(t["diff"].max())
        arr = np.full((10, 2, 3, 8, 2 * R + 1, 4, 3), np.nan)
        arr[t["inning"] - 1, t["half"], t["outs"], t["bases"], t["diff"] + R, t["balls"], t["strikes"]] = t["wp"]
        return cls(arr, R)

    def wp(self, inning, half, outs, bases, diff, balls=0, strikes=0) -> float:
        i = min(int(inning), 10) - 1
        d = min(max(int(diff), -self.R), self.R) + self.R
        return float(self.arr[i, half, outs, bases, d, balls, strikes])

    def split(self, *args, **kw) -> tuple[int, int]:
        h = round(100 * self.wp(*args, **kw))
        return 100 - h, h


@dataclass
class PAWP:
    arr: np.ndarray      # [inning 1..10, half, outs, bases, diff]
    R: int

    @classmethod
    def load(cls, path: str | Path, column: str = "wp") -> "PAWP":
        t = pd.read_parquet(path)
        R = int(t["diff"].max())
        arr = np.full((10, 2, 3, 8, 2 * R + 1), np.nan)
        arr[t["inning"] - 1, t["half"], t["outs"], t["bases"], t["diff"] + R] = t[column]
        return cls(arr, R)

    def wp(self, inning, half, outs, bases, diff) -> float:
        i = min(int(inning), 10) - 1
        d = min(max(int(diff), -self.R), self.R) + self.R
        return float(self.arr[i, half, outs, bases, d])
