"""Structured (JSON-lines) logging for the live pipeline.

Every warning is one JSON object with an ``event`` name; they go to the
``wp.live`` logger, to an in-memory ring buffer (debug page, tests) and - only
when ``SENTRY_DSN`` is set and ``sentry_sdk`` is installed - to Sentry as a
breadcrumb/message.  Unknown feed codes, re-solve failures and B2E fallbacks
are logged through here so they can be forwarded unchanged later.
"""
from __future__ import annotations

import collections
import json
import logging
import os
import time

LOG = logging.getLogger("wp.live")
RECENT: collections.deque = collections.deque(maxlen=2000)
COUNTS: collections.Counter = collections.Counter()
_sentry = None


def init_sentry() -> bool:
    """Enable Sentry if SENTRY_DSN is set and the optional package is installed."""
    global _sentry
    dsn = os.environ.get("SENTRY_DSN")
    if not dsn:
        return False
    try:
        import sentry_sdk
    except ImportError:
        LOG.warning(json.dumps({"event": "sentry_unavailable", "reason": "sentry_sdk not installed"}))
        return False
    sentry_sdk.init(dsn=dsn, traces_sample_rate=0.0)
    _sentry = sentry_sdk
    return True


def _emit(level: int, event: str, fields: dict) -> dict:
    rec = {"ts": round(time.time(), 3), "level": logging.getLevelName(level).lower(), "event": event, **fields}
    RECENT.append(rec)
    COUNTS[event] += 1
    LOG.log(level, json.dumps(rec, default=str, ensure_ascii=False))
    if _sentry is not None and level >= logging.WARNING:
        _sentry.capture_message(f"{event}: {json.dumps(fields, default=str)}",
                                level="warning" if level == logging.WARNING else "error")
    return rec


def info(event: str, **fields) -> dict:
    return _emit(logging.INFO, event, fields)


def warn(event: str, **fields) -> dict:
    return _emit(logging.WARNING, event, fields)


def error(event: str, **fields) -> dict:
    return _emit(logging.ERROR, event, fields)
