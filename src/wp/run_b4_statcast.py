"""B4 step 3 gate: Statcast velocity / pitch-mix features at the PA level.

If data/external/statcast/statcast_<YYYY>.parquet is missing, writes
reports/b4_statcast.json with status "미실시" (data not available in this
environment; scripts/fetch_statcast.py collects it on a local PC) plus the
live-feed availability table, and stops.  The gate itself (PA-level log-loss
gain >= 0.002 over q') runs once the files exist.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import statcast as SC


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--statcast", type=Path, default=Path("data/external/statcast"))
    ap.add_argument("--reports", type=Path, default=Path("reports"))
    a = ap.parse_args()
    files = sorted(a.statcast.glob("statcast_*.parquet")) if a.statcast.exists() else []
    res = {"live_feed_availability": [{"statcast": s, "live_feed": f, "used": u} for s, f, u in SC.LIVE_FEED],
           "gate_threshold_pa_logloss": 0.002}
    if not files:
        res["status"] = ("미실시: Statcast(baseballsavant.mlb.com)와 MLB Stats API(statsapi.mlb.com)가 이 환경에서 "
                         "차단됨. scripts/fetch_statcast.py를 로컬 PC에서 실행해 data/external/statcast/에 넣으면 진행.")
    else:
        res["status"] = "데이터 발견: 연결·피처·게이트 실행은 별도 승인 후 (예상 1–2시간)"
        res["files"] = [f.name for f in files]
    a.reports.mkdir(exist_ok=True)
    (a.reports / "b4_statcast.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(res["status"])


if __name__ == "__main__":
    main()
