"""StatsAPI client for live polling (endpoints confirmed 2026-11-04, see scripts/fetch_feeds.py).

Polling etiquette (7.2 of the task):
  * live game: every ``base`` seconds (default 7, within 5-10);
  * unchanged response (same metaData.timeStamp): interval grows x1.5 up to ``max_unchanged`` (30 s);
  * errors: exponential backoff 2, 4, 8 ... up to 120 s;
  * pre-game / delayed / suspended: ``slow`` (60 s); final: polling stops.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import httpx

from .log import info, warn

API = "https://statsapi.mlb.com/api"
UA = "baseball_diamond-wp/1.0 (personal, non-commercial)"


@dataclass
class PollPolicy:
    base: float = 7.0
    max_unchanged: float = 30.0
    slow: float = 60.0
    max_backoff: float = 120.0


class FeedClient:
    def __init__(self, policy: PollPolicy | None = None, timeout: float = 20.0):
        self.policy = policy or PollPolicy()
        self.http = httpx.AsyncClient(timeout=timeout, headers={"User-Agent": UA}, http2=False)

    async def close(self):
        await self.http.aclose()

    async def get_json(self, url: str):
        r = await self.http.get(url)
        r.raise_for_status()
        return r.json()

    async def schedule(self, date: str, team_id: int | None = None) -> list[dict]:
        url = f"{API}/v1/schedule?sportId=1&date={date}" + (f"&teamId={team_id}" if team_id else "")
        d = await self.get_json(url)
        return [g for dt in d.get("dates", []) for g in dt["games"]]

    async def live(self, pk: int) -> dict:
        return await self.get_json(f"{API}/v1.1/game/{pk}/feed/live")

    async def follow(self, pk: int, on_feed, stop: asyncio.Event | None = None):
        """Poll one game until final (or ``stop``); ``on_feed(feed)`` is awaited for each response."""
        p = self.policy
        interval, backoff, last_tc = p.base, 2.0, None
        while not (stop and stop.is_set()):
            try:
                feed = await self.live(pk)
                backoff = 2.0
            except Exception as e:  # noqa: BLE001
                warn("feed_fetch_error", game_pk=pk, err=repr(e), retry_s=backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, p.max_backoff)
                continue
            tc = (feed.get("metaData") or {}).get("timeStamp")
            state = (feed.get("gameData", {}).get("status") or {}).get("abstractGameState")
            coded = (feed.get("gameData", {}).get("status") or {}).get("codedGameState")
            await on_feed(feed, time.time())
            if state == "Final":
                info("game_final_polling_stopped", game_pk=pk)
                return
            if state == "Preview" or coded in ("T", "U") or "Delayed" in \
                    (feed.get("gameData", {}).get("status") or {}).get("detailedState", ""):
                interval = p.slow
            elif tc == last_tc:
                interval = min(interval * 1.5, p.max_unchanged)
            else:
                interval = p.base
            last_tc = tc
            await asyncio.sleep(interval)
