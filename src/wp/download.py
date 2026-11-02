"""Download Retrosheet event files, team files and game logs.

Source: GitHub mirror maintained by Chadwick Baseball Bureau
  https://github.com/chadwickbureau/retrosheet  (branch ``master``)
which republishes Retrosheet data (see README for the required notice).
retrosheet.org itself was not reachable from the build environment.

Layout in the mirror (verified by probing, 2026-11):
  seasons/<YYYY>/TEAM<YYYY>            team list  (id,league,city,nickname)
  seasons/<YYYY>/<YYYY><TEAM>.EVA|EVN  play-by-play event files (AL / NL home teams)
  seasons/<YYYY>/GL<YYYY>.TXT          game log (2024 is lower-case gl2024.txt)

Usage:  python -m wp.download --first 1901 --last 2025 --out data/raw
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

MIRROR = "https://raw.githubusercontent.com/chadwickbureau/retrosheet/master/seasons"
MLB_LEAGUES = {"A": "A", "N": "N", "AL": "A", "NL": "N"}


def fetch(url: str, retries: int = 4) -> bytes | None:
    for i in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
        except Exception:
            pass
        time.sleep(2 ** (i + 1))
    raise RuntimeError(f"failed to fetch {url}")


def save(url: str, dest: Path) -> int:
    if dest.exists() and dest.stat().st_size > 0:
        return dest.stat().st_size
    data = fetch(url)
    if data is None:
        return -1
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return len(data)


def season(year: int, out: Path) -> dict:
    d = out / str(year)
    team_bytes = save(f"{MIRROR}/{year}/TEAM{year}", d / f"TEAM{year}")
    if team_bytes < 0:
        return {"year": year, "status": "no TEAM file"}
    teams = []
    for line in (d / f"TEAM{year}").read_text(encoding="latin1").splitlines():
        p = line.split(",")
        if len(p) >= 2 and p[1] in MLB_LEAGUES and p[0] not in ("ALS", "NLS"):
            teams.append((p[0], MLB_LEAGUES[p[1]]))
    info = {"year": year, "teams": len(teams), "event_files": 0, "bytes": team_bytes}
    for tm, lg in teams:
        name = f"{year}{tm}.EV{lg}"
        n = save(f"{MIRROR}/{year}/{name}", d / name)
        if n > 0:
            info["event_files"] += 1
            info["bytes"] += n
    gl = save(f"{MIRROR}/{year}/GL{year}.TXT", d / f"GL{year}.TXT")
    if gl < 0:
        gl = save(f"{MIRROR}/{year}/gl{year}.txt", d / f"GL{year}.TXT")
    info["gamelog_bytes"] = gl
    info["bytes"] += max(gl, 0)
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=1901)
    ap.add_argument("--last", type=int, default=2025)
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    t0 = time.time()
    with cf.ThreadPoolExecutor(a.workers) as ex:
        res = sorted(ex.map(lambda y: season(y, a.out), range(a.first, a.last + 1)),
                     key=lambda r: r["year"])
    summary = {"seasons": res, "total_bytes": sum(r.get("bytes", 0) for r in res),
               "seconds": round(time.time() - t0, 1)}
    (a.out / "download_manifest.json").write_text(json.dumps(summary, indent=1))
    print(f"downloaded {summary['total_bytes']/1e6:.1f} MB in {summary['seconds']} s")


if __name__ == "__main__":
    main()
