"""Output rules (7.1): percentages, LED cells with hysteresis, message schema.

* pct_home = round(100 * wp_home), pct_away = 100 - pct_home (sum is always 100).
* final: 100 / 0 by the final score.
* LEDs: N cells (default 40) for the home side.  The displayed count only moves when
  wp crosses a cell boundary by more than ``hysteresis`` cells (default 0.3), so a value
  sitting on a boundary does not flicker.  The WP value itself is never smoothed.
"""
from __future__ import annotations

import math

FIELDS = ["game_id", "ts_feed", "ts_out", "seq", "status", "inning", "half", "outs", "bases", "count",
          "away", "home", "batter", "pitcher", "last_event", "wp_home", "pct_away", "pct_home", "leds_home",
          "wp_if_hr", "wp_if_k", "wpa_last", "model", "info_level", "recomputing"]


def split_pct(wp_home: float) -> tuple[int, int]:
    h = int(math.floor(100 * wp_home + 0.5))
    h = min(max(h, 0), 100)
    return 100 - h, h


class LedHysteresis:
    def __init__(self, n: int = 40, hysteresis: float = 0.3):
        self.n, self.h = n, hysteresis
        self.cur: int | None = None

    def __call__(self, wp_home: float) -> int:
        x = wp_home * self.n
        target = int(math.floor(x + 0.5))
        if self.cur is None or wp_home in (0.0, 1.0):
            self.cur = target
        elif target != self.cur:
            # move only when x is clearly past the boundary between cur and its neighbour
            if target > self.cur and x >= self.cur + 0.5 + self.h:
                self.cur = target
            elif target < self.cur and x <= self.cur - 0.5 - self.h:
                self.cur = target
        self.cur = min(max(self.cur, 0), self.n)
        return self.cur

    def reset(self):
        self.cur = None
