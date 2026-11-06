"""HTTP / WebSocket API for the website (and the LED device).

Run:  PYTHONPATH=src uvicorn wp.live.api:app --port $PORT      (or: python -m wp.live.api)

Configuration (environment variables, see .env.example):
  PORT                 HTTP port (python -m entry point; default 8000)
  CORS_ORIGINS         comma-separated allowed origins ("*" for any; default none)
  TIMELINE_CACHE_DIR   finished-game timelines (default timelines/)
  FEED_SOURCE          statsapi (poll live games) | off (replay only; default off)
  MODEL                AVG | B3 | B2E | B2 (default AVG)
  TEAM_ID              MLB team id followed live (default: none = every game of the day, max MAX_GAMES)
  SEASON               season of the models for live serving (default: current year)
  N_LEDS               LED cells (default 40)
  MQTT_URL             optional, e.g. mqtt://host:1883/topic-prefix (needs paho-mqtt)
  SENTRY_DSN           optional error tracking (needs sentry-sdk)

Endpoints
  GET  /health                       liveness + what is loaded
  GET  /games/today                  today's schedule with the latest message of followed games
  GET  /games/{id}/state             latest message of a game (live or cached)
  GET  /games/{id}/timeline          every point's latest message, in game order
  GET  /games/{id}/corrections       feed corrections (key, events, WP before/after)
  WS   /ws/games/{id}                live messages (first the current state)
  GET  /replay/games                 replayable past games (timeline cache)
  WS   /ws/replay/{id}?speed=1|5|20|max&from=seq
                                     cached timeline streamed like a live game; client commands
                                     {"cmd":"pause"} {"cmd":"resume"} {"cmd":"seek","seq":n} {"cmd":"speed","value":"5"}
  GET  /debug                        development page (score, state, WP graph, corrections)
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from datetime import date, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from . import log as LG
from .replay import TimelineCache

CFG = {
    "cors": [o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()],
    "feed_source": os.environ.get("FEED_SOURCE", "off"),
    "model": os.environ.get("MODEL", "AVG"),
    "team_id": int(os.environ["TEAM_ID"]) if os.environ.get("TEAM_ID") else None,
    "season": int(os.environ.get("SEASON", date.today().year)),
    "n_leds": int(os.environ.get("N_LEDS", "40")),
    "max_games": int(os.environ.get("MAX_GAMES", "1")),
    "mqtt": os.environ.get("MQTT_URL"),
}
SPEEDS = {"1": 1.0, "5": 5.0, "20": 20.0, "max": 0.0}

app = FastAPI(title="baseball_diamond WP API", version="1.0",
              description="MLB win probability, live and replay. Data (c) MLB Advanced Media, "
                          "personal non-commercial use.")
if CFG["cors"]:
    app.add_middleware(CORSMiddleware, allow_origins=CFG["cors"], allow_methods=["GET"], allow_headers=["*"])


class Hub:
    """Live games: GameServer per game + WebSocket subscribers.  Season models load in a
    background thread so /health and the replay endpoints answer immediately."""

    def __init__(self):
        self.servers: dict = {}
        self.subs: dict = {}
        self.models = None
        self.ids = None
        self.solver = None
        self.status = "not_loaded"
        self.schedule: list = []
        self.mqtt = None

    def load_models(self):
        self.status = "loading"
        try:
            from .ids import IdMap
            from .models import SeasonModels
            from .solver import ProcessSolver
            self.models = SeasonModels.load(CFG["season"])
            self.ids = IdMap()
            self.solver = ProcessSolver(self.models.eng)    # worker in the same container
            self.status = "ready"
        except Exception as e:  # noqa: BLE001
            self.status = f"error: {e!r}"
            LG.error("model_load_failed", err=repr(e))

    def server(self, pk: int, gid: str):
        from .serving import GameServer
        if pk not in self.servers:
            self.servers[pk] = GameServer(self.models, self.ids, self.solver, model=CFG["model"],
                                          n_leds=CFG["n_leds"], game_id=gid)
        return self.servers[pk]

    async def publish(self, gid: str, msgs: list):
        for m in msgs:
            for q in list(self.subs.get(gid, [])):
                q.put_nowait(m)
            if self.mqtt is not None:
                try:
                    self.mqtt.publish(f"{self.mqtt_prefix}/{gid}", json.dumps(m, default=str))
                except Exception as e:  # noqa: BLE001
                    LG.warn("mqtt_publish_failed", err=repr(e))


HUB = Hub()
CACHE = TimelineCache()


def _find_live(gid: str):
    for s in HUB.servers.values():
        if s.game_id == gid:
            return s
    return None


@app.on_event("startup")
async def _startup():
    LG.init_sentry()
    if CFG["mqtt"]:
        try:
            import urllib.parse

            import paho.mqtt.client as mqtt
            u = urllib.parse.urlparse(CFG["mqtt"])
            c = mqtt.Client()
            c.connect(u.hostname, u.port or 1883)
            c.loop_start()
            HUB.mqtt, HUB.mqtt_prefix = c, (u.path.strip("/") or "wp")
        except Exception as e:  # noqa: BLE001
            LG.warn("mqtt_unavailable", err=repr(e))
    if CFG["feed_source"] == "statsapi":
        threading.Thread(target=HUB.load_models, daemon=True).start()
        asyncio.create_task(_live_loop())


async def _live_loop():
    from .feed_client import FeedClient
    from .ids import team_code
    fc = FeedClient()
    following: dict = {}
    while True:
        try:
            today = date.today().isoformat()
            HUB.schedule = await fc.schedule(today, CFG["team_id"])
            if HUB.status == "ready":
                for g in HUB.schedule[:CFG["max_games"]]:
                    pk = g["gamePk"]
                    if pk in following or g["status"]["abstractGameState"] == "Final" and pk not in HUB.servers:
                        continue
                    n = 0 if g.get("doubleHeader", "N") == "N" else int(g.get("gameNumber", 1))
                    gid = f"{team_code(g['teams']['home']['team']['id'])}{today.replace('-', '')}{n}"
                    following[pk] = asyncio.create_task(_follow(fc, pk, gid))
        except Exception as e:  # noqa: BLE001
            LG.warn("schedule_poll_failed", err=repr(e))
        await asyncio.sleep(300)


async def _follow(fc, pk: int, gid: str):
    srv = HUB.server(pk, gid)
    loop = asyncio.get_running_loop()

    async def on_feed(feed, ts):
        msgs = await loop.run_in_executor(None, srv.update, feed, ts)
        await HUB.publish(gid, msgs)

    async def poll_solver():
        while True:
            msgs = await loop.run_in_executor(None, srv.poll)
            await HUB.publish(gid, msgs)
            await asyncio.sleep(0.5)

    t = asyncio.create_task(poll_solver())
    try:
        await fc.follow(pk, on_feed)
        if srv.view is not None and srv.view.over:
            CACHE.save(gid, {**srv.meta, "n_points": len(srv.timeline())}, srv.timeline(), srv.corrections)
    finally:
        await asyncio.sleep(30)
        t.cancel()


# ---- REST ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "models": HUB.status, "feed_source": CFG["feed_source"], "model": CFG["model"],
            "live_games": [s.game_id for s in HUB.servers.values()], "cached_timelines": len(list(
                CACHE.root.glob("*.json.gz"))) if CACHE.root.exists() else 0, "time": time.time()}


@app.get("/games/today")
def games_today():
    out = []
    for g in HUB.schedule:
        s = HUB.servers.get(g["gamePk"])
        out.append({"gamePk": g["gamePk"], "away": g["teams"]["away"]["team"]["name"],
                    "home": g["teams"]["home"]["team"]["name"], "status": g["status"]["detailedState"],
                    "game_id": s.game_id if s else None, "state": s.state() if s else None})
    return {"date": date.today().isoformat(), "games": out}


def _cached(gid: str) -> dict:
    if not CACHE.has(gid):
        raise HTTPException(404, f"unknown game {gid}")
    return CACHE.load(gid)


@app.get("/games/{gid}/state")
def game_state(gid: str):
    s = _find_live(gid)
    if s is not None:
        return s.state()
    tl = _cached(gid)["timeline"]
    return tl[-1] if tl else None


@app.get("/games/{gid}/timeline")
def game_timeline(gid: str):
    s = _find_live(gid)
    if s is not None:
        return {"meta": s.meta, "timeline": s.timeline()}
    d = _cached(gid)
    return {"meta": d["meta"], "timeline": d["timeline"]}


@app.get("/games/{gid}/corrections")
def game_corrections(gid: str):
    s = _find_live(gid)
    if s is not None:
        return s.corrections
    return _cached(gid).get("corrections", [])


@app.get("/replay/games")
def replay_games():
    return CACHE.list()


# ---- WebSocket ------------------------------------------------------------------------
@app.websocket("/ws/games/{gid}")
async def ws_game(ws: WebSocket, gid: str):
    await ws.accept()
    q: asyncio.Queue = asyncio.Queue()
    HUB.subs.setdefault(gid, set()).add(q)
    try:
        s = _find_live(gid)
        cur = s.state() if s else (CACHE.load(gid)["timeline"][-1] if CACHE.has(gid) else None)
        if cur:
            await ws.send_json(cur)
        while True:
            await ws.send_json(await q.get())
    except WebSocketDisconnect:
        pass
    finally:
        HUB.subs[gid].discard(q)


def _ts(m) -> float | None:
    t = m.get("ts_feed")
    if not t:
        return None
    try:
        return datetime.fromisoformat(str(t).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


@app.websocket("/ws/replay/{gid}")
async def ws_replay(ws: WebSocket, gid: str, speed: str = Query("1"), from_: int = Query(0, alias="from")):
    await ws.accept()
    if not CACHE.has(gid):
        await ws.send_json({"error": f"unknown game {gid}"})
        await ws.close()
        return
    tl = CACHE.load(gid)["timeline"]
    ctl = {"paused": False, "speed": SPEEDS.get(speed, 1.0), "i": 0}
    ctl["i"] = next((k for k, m in enumerate(tl) if (m.get("seq") or 0) >= from_), 0) if from_ else 0

    async def reader():
        while True:
            c = await ws.receive_json()
            cmd = c.get("cmd")
            if cmd == "pause":
                ctl["paused"] = True
            elif cmd == "resume":
                ctl["paused"] = False
            elif cmd == "speed":
                ctl["speed"] = SPEEDS.get(str(c.get("value")), ctl["speed"])
            elif cmd == "seek":
                if "index" in c:
                    ctl["i"] = max(0, min(int(c["index"]), len(tl) - 1))
                else:
                    ctl["i"] = next((k for k, m in enumerate(tl) if (m.get("seq") or 0) >= int(c.get("seq", 0))),
                                    len(tl) - 1)
                ctl["seeked"] = True

    rt = asyncio.create_task(reader())
    try:
        await ws.send_json({"type": "replay_start", "game_id": gid, "n": len(tl), "speed": speed})
        ended = False
        while True:
            if ctl["paused"] or ctl["i"] >= len(tl):
                if ctl["i"] >= len(tl) and not ended:
                    await ws.send_json({"type": "replay_end", "game_id": gid})
                    ended = True
                await asyncio.sleep(0.1)       # stay open: seek / resume after the end
                continue
            ended = False
            i = ctl["i"]
            m = tl[i]
            await ws.send_json({**m, "replay_index": i})
            ctl["i"] = i + 1
            ctl.pop("seeked", None)
            if ctl["i"] < len(tl) and ctl["speed"] > 0:
                a, b = _ts(m), _ts(tl[ctl["i"]])
                dt = (b - a) / ctl["speed"] if a and b else 0.5
                t_end = time.time() + min(max(dt, 0.0), 60.0)
                while time.time() < t_end and not ctl.get("seeked") and not ctl["paused"]:
                    await asyncio.sleep(min(0.1, max(t_end - time.time(), 0)))
            else:
                await asyncio.sleep(0)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        rt.cancel()


@app.get("/debug", response_class=HTMLResponse)
def debug_page():
    return (Path(__file__).with_name("debug.html")).read_text(encoding="utf-8")


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    main()
