"""B3 solves: synchronous (validation, tests) or in a separate worker process (serving).

Jobs are identified by a hashable key.  ``request`` queues a job (idempotent),
``get`` returns the result or None while it is pending, ``pending`` tells whether
a job is in flight.  The worker process is forked after the season models are
loaded, so it shares the (read-only) rating store and context with the server.

Job kinds
  full      Engine.solve(gi, start=...)                          (pre-game, lineup change)
  override  Engine.solve(gi, override, upto, cache=<full>, start)  (reliever, current half)
A failed job is recorded (``failed``) and logged; the tracker then keeps serving B2E.
"""
from __future__ import annotations

import concurrent.futures as cf
import multiprocessing as mp
import time

from .log import error, info

_W: dict = {}            # worker-process globals: engine, cached full results


def _worker_init(eng):
    _W["eng"] = eng
    _W["cache"] = {}


def _run(eng, cache: dict, key, kind: str, kw: dict):
    t = time.time()
    if kind == "full":
        res = eng.solve(kw["gi"], start=kw.get("start"))
        cache[key] = res
        if len(cache) > 24:                       # bound worker memory (~116 MB per result)
            for k in list(cache)[:-24]:
                cache.pop(k)
    elif kind == "override":
        base = cache.get(kw["base"])
        if base is None:
            raise KeyError(f"base solve {kw['base']} not cached")
        res = eng.solve(kw["gi"], override=kw["override"], upto=kw["upto"], cache=base, start=kw["start"])
    else:
        raise ValueError(kind)
    return res, time.time() - t


def _worker_job(key, kind, kw):
    return _run(_W["eng"], _W["cache"], key, kind, kw)


class SyncSolver:
    """Solves immediately in the calling thread."""

    def __init__(self, eng):
        self.eng = eng
        self.cache: dict = {}
        self.results: dict = {}
        self.failed: dict = {}
        self.times: list = []          # (kind, seconds)

    def request(self, key, kind, **kw):
        if key in self.results or key in self.failed:
            return
        try:
            res, dt = _run(self.eng, self.cache, key, kind, kw)
        except Exception as e:          # noqa: BLE001 - keep serving B2E
            self.failed[key] = repr(e)
            error("resolve_failed", key=str(key), err=repr(e))
            return
        self.results[key] = res
        self.times.append((kind, dt))

    def get(self, key):
        return self.results.get(key)

    def pending(self, key) -> bool:
        return False

    def poll(self) -> list:
        return []

    def close(self):
        pass


class ProcessSolver(SyncSolver):
    """One worker process (fork); results come back through futures."""

    def __init__(self, eng, workers: int = 1):
        super().__init__(eng)
        self.pool = cf.ProcessPoolExecutor(workers, mp_context=mp.get_context("fork"),
                                           initializer=_worker_init, initargs=(eng,))
        self.futs: dict = {}
        self.t_submit: dict = {}

    def request(self, key, kind, **kw):
        if key in self.results or key in self.failed or key in self.futs:
            return
        self.t_submit[key] = time.time()
        self.futs[key] = self.pool.submit(_worker_job, key, kind, kw)
        info("resolve_submitted", key=str(key), kind=kind)

    def pending(self, key) -> bool:
        return key in self.futs

    def poll(self) -> list:
        """Collect finished jobs; returns their keys."""
        done = []
        for key, f in list(self.futs.items()):
            if not f.done():
                continue
            self.futs.pop(key)
            try:
                res, dt = f.result()
                self.results[key] = res
                wall = time.time() - self.t_submit.pop(key)
                self.times.append((key[0], dt, wall))
                if len(self.results) > 48:
                    for k in list(self.results)[:-48]:
                        self.results.pop(k)
            except Exception as e:      # noqa: BLE001
                self.failed[key] = repr(e)
                error("resolve_failed", key=str(key), err=repr(e))
            done.append(key)
        return done

    def get(self, key):
        return self.results.get(key)

    def wait(self, timeout: float | None = None) -> list:
        if self.futs:
            cf.wait(list(self.futs.values()), timeout=timeout, return_when=cf.FIRST_COMPLETED)
        return self.poll()

    def close(self):
        self.pool.shutdown(cancel_futures=True)
