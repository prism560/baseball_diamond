"""Realtime report: aggregates data/live/replay_runs/*.json (scripts/replay.py) and
reports/live_crosscheck.json (wp.live.crosscheck_feed) into reports/realtime_report.md and
reports/realtime_results.json.

    PYTHONPATH=src python -m wp.live.report_realtime
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

RUNS = Path("data/live/replay_runs")
OUT_MD = Path("reports/realtime_report.md")
OUT_JSON = Path("reports/realtime_results.json")


def q(x, p):
    return float(np.percentile(x, p)) if len(x) else float("nan")


def dist(x, scale=1.0):
    x = np.asarray(x, float) * scale
    return {"n": int(len(x)), "median": q(x, 50), "p95": q(x, 95), "max": float(x.max()) if len(x) else float("nan")}


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    if isinstance(v, float):
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e5):
            return f"{v:.1e}"
        return f"{v:.{nd}f}"
    return str(v)


def main() -> None:
    runs = [json.loads(p.read_text()) for p in sorted(RUNS.glob("*.json"))]
    errs = [r for r in runs if "error" in r]
    ok = [r for r in runs if "error" not in r]
    reg = [r for r in ok if r["gameType"] == "R"]
    post = [r for r in ok if r["gameType"] != "R"]
    cc = json.loads(Path("reports/live_crosscheck.json").read_text())["summary"]
    res: dict = {"n_games": len(runs), "n_errors": len(errs), "n_regular": len(reg), "n_postseason": len(post),
                 "crosscheck": cc}
    # ---- offline match ----
    agg: dict = {}
    for r in reg:
        for k, v in r.get("offline", {}).items():
            a = agg.setdefault(k, {"n": 0, **{m: {"max_abs": 0.0, "n_over_tol": 0} for m in ("B3", "B2E", "AVG")}})
            a["n"] += v["n"]
            for m in ("B3", "B2E", "AVG"):
                if not np.isnan(v[m]["max_abs"]):
                    a[m]["max_abs"] = max(a[m]["max_abs"], v[m]["max_abs"])
                a[m]["n_over_tol"] += v[m]["n_over_tol"]
    res["offline"] = agg
    after = np.abs(np.concatenate([np.asarray(r.get("after_runner_event_diffs", []), float) for r in reg]
                                  or [np.zeros(0)]))
    res["after_runner_event"] = {"n": int(len(after)), "mean_abs": float(after.mean()) if len(after) else None,
                                 "max_abs": float(after.max()) if len(after) else None,
                                 "share_gt_2.5pp": float((after > 0.025).mean()) if len(after) else None}
    res["lineup_first_appearance_differs"] = [r["game_id"] for r in reg if r.get("lineup_first_appearance_differs")]
    games_over = [r["game_id"] for r in reg if any(v[m]["n_over_tol"] for k, v in r.get("offline", {}).items()
                                                   if k.endswith(":strict") for m in ("B3", "B2E", "AVG"))]
    res["games_with_strict_mismatch"] = games_over
    # ---- final consistency ----
    res["final_ok"] = sum(r["final"]["ok"] for r in ok)
    res["final_fail"] = [r["game_id"] for r in ok if not r["final"]["ok"]]
    res["corrections_clean_total"] = sum(r["corrections_clean"] for r in ok)
    # ---- faults ----
    fl = [r["faults"] for r in ok]
    res["faults"] = {"final_equal": sum(f["final_equal"] for f in fl), "games": len(fl),
                     "missing_points": sum(f["missing"] for f in fl), "state_diff": sum(f["state_diff"] for f in fl),
                     "max_abs_wp_diff": max((f["max_abs_wp_diff"] or 0) for f in fl) if fl else None,
                     "snapshots_delivered": sum(f["delivered"] for f in fl),
                     "net_errors": sum(f["net_errors"] for f in fl),
                     "stale_dropped": sum(f["stale_dropped"] for f in fl),
                     "duplicates": sum(f["duplicates"] for f in fl),
                     "corrections_logged": sum(f["corrections_logged"] for f in fl)}
    # ---- real snapshots ----
    res["real"] = []
    for r in ok:
        if "real" in r:
            x = r["real"]
            ovt = [c for c in x["corrections"] if c.get("overturned")]
            res["real"].append({"game_id": r["game_id"], "gamePk": r["gamePk"], "n_snapshots": x["n_snapshots"],
                                "points_synthetic": x["points_a"], "points_real": x["points_b"],
                                "missing": x["missing"], "state_diff": x["state_diff"],
                                "max_abs_wp_diff": x["max_abs_wp_diff"], "final_equal": x["final_equal"],
                                "corrections": len(x["corrections"]), "overturned": len(ovt),
                                "corrections_detail": [{k: c.get(k) for k in ("key", "kind", "last_event_old",
                                                                               "last_event_new", "wp_before",
                                                                               "wp_after", "wp_delta")}
                                                       for c in x["corrections"]][:20],
                                "latency": x["latency"]})
    # ---- latency / solve times ----
    res["latency_ms"] = {
        "clean_sync_median_of_games": dist([r["latency_clean"]["median_ms"] for r in ok]),
        "clean_sync_p95_of_games": dist([r["latency_clean"]["p95_ms"] for r in ok]),
        "faults_sync_max_of_games": dist([r["faults"]["latency"]["max_ms"] for r in ok]),
    }
    asy = [r for r in ok if "async" in r]
    if asy:
        res["latency_ms"]["async_median_of_games"] = dist([r["async"]["latency"]["median_ms"] for r in asy])
        res["latency_ms"]["async_p95_of_games"] = dist([r["async"]["latency"]["p95_ms"] for r in asy])
        res["latency_ms"]["async_max_of_games"] = dist([r["async"]["latency"]["max_ms"] for r in asy])
        res["async"] = {"games": [r["game_id"] for r in asy],
                        "final_equal": sum(r["async"]["final_equal"] for r in asy),
                        "max_abs_wp_diff_vs_sync": max(r["async"]["max_abs_wp_diff"] or 0 for r in asy),
                        "recomputing_share": float(np.mean([r["async"]["recomputing_messages"] /
                                                            max(r["async"]["messages"], 1) for r in asy])),
                        "resolve_compute_s": dist([x for r in asy for x in r["async"].get("resolve_compute_s", [])]),
                        "resolve_wall_s": dist([x for r in asy for x in r["async"]["resolve_wall_s"]])}
    st = [(k, t) for r in ok for k, t in r["solve_times"]]
    pre = [r["solve_times"][0][1] for r in ok if r["solve_times"]]
    full_later = [t for r in ok for i, (k, t) in enumerate(r["solve_times"]) if k == "full" and i > 0]
    ovr = [t for k, t in st if k == "override"]
    res["solve_s"] = {"pregame": dist(pre), "lineup_change": dist(full_later), "pitching_change": dist(ovr),
                      "per_game_total": dist([sum(t for _, t in r["solve_times"]) for r in ok])}
    off_t = [r["offline_timing"] for r in reg if "offline_timing" in r]
    res["offline_eval"] = {"pregame_s": dist([t["pregame_s"] for t in off_t]),
                           "resolves_per_game": float(np.mean([t["n_resolve"] for t in off_t])) if off_t else None}
    w = Counter()
    for r in ok:
        w.update(r.get("warnings", {}))
    res["warnings"] = dict(w)
    res["errors"] = [{"game_id": r["game_id"], "error": r["error"]} for r in errs]
    tl = Path("timelines")
    res["timeline_cache_bytes"] = sum(p.stat().st_size for p in tl.glob("*.json.gz")) if tl.exists() else 0
    res["timeline_cache_files"] = len(list(tl.glob("*.json.gz"))) if tl.exists() else 0
    OUT_JSON.write_text(json.dumps(res, indent=1, default=float))
    OUT_MD.write_text(render(res, ok))
    from ..manifest import ROOT, write_meta
    live = ROOT / "src" / "wp" / "live"
    write_meta([OUT_JSON, OUT_MD], [Path("reports/live_crosscheck.json")] + sorted(RUNS.glob("*.json")),
               "live_report", [live / f for f in ("report_realtime.py", "serving.py", "tracker.py", "replay.py",
                                                   "models.py", "offline_ref.py", "validate_offline.py",
                                                   "reconstruct.py", "normalize.py", "codes.py")]
               + [ROOT / "scripts" / "replay.py"])
    print(OUT_MD, OUT_JSON)


def render(r: dict, runs: list) -> str:
    cc = r["crosscheck"]
    L = []
    a = L.append
    a("# 실시간 파이프라인 리포트")
    a("")
    a("`scripts/replay.py`(경기별 결과는 `data/live/replay_runs/`)와 `wp.live.crosscheck_feed`의 결과를 "
      "`wp.live.report_realtime`이 모은 것이다. 수치 원본은 `reports/realtime_results.json`이다.")
    a("")
    a(f"- 리플레이 경기: {r['n_games']}경기(정규시즌 {r['n_regular']}, 포스트시즌 {r['n_postseason']}). "
      f"실행 오류: {r['n_errors']}경기.")
    a("- 모든 경기는 합성 스냅샷으로 재생했다. 합성 스냅샷은 최종 피드를 이벤트마다 잘라 만든다.")
    a("- 실제 시점 스냅샷이 있는 3경기는 실제 스냅샷으로도 재생했다.")
    a("- 기준 설정: 모델 AVG(B3 0.55, B2E 0.45), 시즌 2025 모델. 능력치는 각 경기 전날까지의 정보만 쓴다.")
    a("")
    a("## 1. 정규화 교차검증 (피드 PA ↔ Retrosheet PA)")
    a("")
    a(f"- 대상: 정규시즌 {cc['games']}경기, {cc['pa_retro']:,}타석.")
    a(f"- 타석 수가 같은 경기: {cc['games_pa_count_equal']}/{cc['games']}.")
    a(f"- 타석 단위 불일치: {cc['rows_mismatch']}건(불일치율 {cc['row_mismatch_rate']:.4f}).")
    a("  - 비교 항목: 이닝, 초말, 시작·N 이후·종료 시점의 아웃·주자·점수차, 결과 범주(INT 포함), 타자·투수 ID, 타순 슬롯.")
    a(f"- 투구 수 불일치: {cc['pitch_count_mismatch']}건.")
    a("- 기록 관행 차이로 따로 분류한 건수: 0건.")
    a("  - 끝내기 후 주자 정리 같은 관행은 분류 규칙을 두었지만 이번 대상에서는 나오지 않았다.")
    a("- 처음 비교에서 나온 불일치는 모두 정규화기 오류였고, 고쳤다.")
    a("  - 대타 슬롯을 같은 플레이 안의 교체 이벤트에서 읽지 못함.")
    a("  - 연장 자동 주자 이벤트에 이동 항목이 없음.")
    a("  - 자동 고의4구가 `no_pitch` 이벤트로 옴.")
    a("  - 대주자의 루 교체를 반영하지 못함.")
    a("  - 피치클락 자동 볼·스트라이크(`VP`/`VC`/`AC`)를 투구로 세지 않음.")
    a("")
    a("## 2. 오프라인 일치 (허용오차 1e-6)")
    a("")
    a("- **기준값**: 같은 경기를 오프라인 평가 코드로 계산한 값이다.")
    a("  - `evaluate_game`과 카운트 체인(`run_b3_eval`), B2E 계수, AVG 결합 방식은 모두 평가 때와 같다.")
    a("  - 입력은 Retrosheet PA 행이다. 실시간 쪽 입력은 피드다.")
    a("  - 따라서 이 비교는 정규화와 상태 재구성의 정확성을 검증한다.")
    a("- **엄격 비교 대상**: 타석 시작 시점, 그리고 타석 중 주자 이벤트 전의 투구 시점.")
    a("")
    a("| 비교 | 시점 수 | B3 최대 절대차 | B2E 최대 절대차 | AVG 최대 절대차 | 1e-6 초과 |")
    a("|---|---:|---:|---:|---:|---:|")
    names = {"pa:strict": "타석 시작", "pitch:strict": "투구(주자 이벤트 전)",
             "pa:mid_pa_pitching_change": "타석 시작, 타석 중 투수 교체(의도된 차이)",
             "pitch:mid_pa_pitching_change": "투구, 타석 중 투수 교체(의도된 차이)",
             "pitch:after_runner_event": "투구, 주자 이벤트 뒤(의도된 차이)"}
    for k in ("pa:strict", "pitch:strict", "pa:mid_pa_pitching_change", "pitch:mid_pa_pitching_change",
              "pitch:after_runner_event"):
        v = r["offline"].get(k)
        if not v:
            continue
        a(f"| {names[k]} | {v['n']:,} | {fmt(v['B3']['max_abs'])} | {fmt(v['B2E']['max_abs'])} | "
          f"{fmt(v['AVG']['max_abs'])} | "
          f"{v['B3']['n_over_tol'] + v['B2E']['n_over_tol'] + v['AVG']['n_over_tol'] if k.endswith(':strict') else '해당 없음'} |")
    a("")
    ar = r["after_runner_event"]
    a(f"- **엄격 비교에서 1e-6을 넘은 경기**: {len(r['games_with_strict_mismatch'])}경기 "
      f"{', '.join(r['games_with_strict_mismatch']) or ''}")
    a("- **주자 이벤트 뒤 투구의 의도된 차이**(AVG 기준, 실시간 − 오프라인)")
    a("  - 실시간은 도루·폭투 등이 일어난 실제 상태로 다시 조회하고, 오프라인은 첫 투구 시점 상태를 유지한다.")
    a(f"  - 시점 {ar['n']}개, 평균 |차| {fmt(ar['mean_abs'])}, 최대 {fmt(ar['max_abs'])}, "
      f"2.5%p 초과 비율 {fmt(ar['share_gt_2.5pp'])}.")
    a("- **타석 중 투수 교체**: 실시간은 그 시점에 던지는 투수를, 오프라인은 타석을 끝낸 투수를 쓴다(의도된 차이).")
    a(f"- **오프라인 라인업 구성과 실제 선발 라인업이 다른 경기**: {len(r['lineup_first_appearance_differs'])}경기 "
      f"{', '.join(r['lineup_first_appearance_differs'])}")
    a("  - 오프라인은 타순별 첫 등장 타자를 선발로 간주한다. 서빙은 피드의 실제 선발 라인업을 쓴다.")
    a("- **포스트시즌**은 Retrosheet에 없어 오프라인 기준값이 없다. 종료 일관성, 장애 내성, 지연만 검증했다.")
    a("")
    a("## 3. 종료 일관성")
    a("")
    a(f"- 최종 메시지가 `final`이고 최종 점수에 맞는 100/0으로 끝난 경기: {r['final_ok']}/{len(runs)}. "
      f"실패: {', '.join(r['final_fail']) or '없음'}.")
    a(f"- 장애 없는 합성 재생에서 기록된 정정: {r['corrections_clean_total']}건.")
    a("  - 합성 스냅샷에는 정정이 없으므로 0이어야 한다.")
    a("  - 개발 중 연장 자동 주자가 이벤트보다 늦게 게시되어 생긴 거짓 정정을 발견했고, 하프 시작 시점에 자동 주자를 미리 두도록 고쳤다.")
    a("")
    a("## 4. 장애 주입")
    a("")
    f = r["faults"]
    a("- 주입한 장애(경기마다 시드 고정)")
    a("  - 스냅샷 중복 10%, 이웃 순서 뒤바뀜 10%, 폴링 누락 30%(여러 투구 건너뜀)")
    a("  - 판정 정정 3회: 볼↔루킹 스트라이크를 바꾼 스냅샷 뒤에 원래 스냅샷을 보냄")
    a("  - 네트워크 오류 5%")
    a(f"- 최종 상태와 마지막 WP가 장애 없는 재생과 같은 경기: {f['final_equal']}/{f['games']}.")
    a(f"- 모든 시점의 최종 값 비교: 빠진 시점 {f['missing_points']}개, 상태가 다른 시점 {f['state_diff']}개, "
      f"WP 최대 차이 {fmt(f['max_abs_wp_diff'])}.")
    a("- 처리 내역")
    a(f"  - 전달된 스냅샷 {f['snapshots_delivered']:,}개, 네트워크 오류 {f['net_errors']}회")
    a(f"  - 역행으로 버린 스냅샷 {f['stale_dropped']}개, 중복으로 무시한 스냅샷 {f['duplicates']}개")
    a(f"  - 정정으로 기록된 시점 {f['corrections_logged']}개(주입한 정정과, 그 정정이 바꾼 뒤따르는 시점)")
    a("")
    a("## 5. 실제 시점 스냅샷과 합성 스냅샷 대조")
    a("")
    a("| 경기 | 스냅샷 | 시점(합성/실제) | 빠진 시점 | 상태 차이 | WP 최대 차이 | 최종 일치 | 정정(번복) |")
    a("|---|---:|---:|---:|---:|---:|---|---:|")
    for x in r["real"]:
        a(f"| {x['game_id']} ({x['gamePk']}) | {x['n_snapshots']} | {x['points_synthetic']}/{x['points_real']} | "
          f"{x['missing']} | {x['state_diff']} | {fmt(x['max_abs_wp_diff'])} | {'예' if x['final_equal'] else '아니오'} | "
          f"{x['corrections']} ({x['overturned']}) |")
    a("")
    a("- **합성 대조**: 세 경기 모두 시점 집합이 같고, 최종 값과 WP가 합성 재생과 일치한다(WP 최대 차이 0).")
    a("  - CLE202510010의 상태 차이 1건은 정정으로 바뀐 중간 시점이다. 최종 값은 같다.")
    a("- **챌린지 번복**: 실제 스냅샷에서 번복이 정정으로 나타나지 않았다.")
    a("  - 813071 4회말 태그 플레이 번복을 타임코드별로 보면, 리뷰 동안 그 플레이는 미완료 상태다.")
    a("  - 리뷰가 끝난 뒤 번복된 결과로 처음 완료된다(`reviewDetails.isOverturned: true`). 원래 판정은 피드에 게시되지 않았다.")
    a("  - 따라서 정정 처리(같은 키, 새 해시 → 다시 계산하고 정정 기록)는 4절의 정정 주입으로 검증했다.")
    a("- **실제 피드에서 기록된 정정**: 대타가 타석 시작보다 늦게 게시된 경우다. 타석 시작 시점의 타자가 바뀌었다.")
    a("")
    a("**실제 스냅샷에서 기록된 정정**(일부)")
    a("")
    for x in r["real"]:
        for c in x["corrections_detail"][:8]:
            a(f"- {x['game_id']} {'/'.join(c['key'])} {c['kind']}: \"{c['last_event_old']}\" → "
              f"\"{c['last_event_new']}\", WP {fmt(c['wp_before'])} → {fmt(c.get('wp_after'))}")
    a("")
    a("## 6. 지연과 재계산 시간")
    a("")
    lat = r["latency_ms"]
    a("- **지연 정의**: 스냅샷 하나를 받아 출력 메시지가 나올 때까지의 시간(ms)이다.")
    a("  - 정규화, 재구성, 중복·정정 판정, 모델 조회, 메시지 생성을 모두 포함한다. 네트워크 시간은 뺀다.")
    a("  - 목표는 100ms 이하다.")
    a("")
    a("| 구분 | 경기 수 | 중앙값 | 95% | 최대 |")
    a("|---|---:|---:|---:|---:|")
    rows = [("동기 재생, 경기별 중앙값", "clean_sync_median_of_games"),
            ("동기 재생, 경기별 95%", "clean_sync_p95_of_games"),
            ("비동기(워커) 재생, 경기별 중앙값", "async_median_of_games"),
            ("비동기(워커) 재생, 경기별 95%", "async_p95_of_games"),
            ("비동기(워커) 재생, 경기별 최대", "async_max_of_games")]
    for lab, k in rows:
        d = lat.get(k)
        if d:
            a(f"| {lab} | {d['n']} | {fmt(d['median'], 1)} | {fmt(d['p95'], 1)} | {fmt(d['max'], 1)} |")
    a("")
    a("- 실시간 서빙 경로는 비동기다. 재계산을 워커가 하므로 스냅샷 처리에는 조회만 남는다.")
    a("- 동기 재생의 최대값에는 재계산 시간이 그대로 들어간다(검증용 경로).")
    a("")
    s = r["solve_s"]
    a("| B3 계산(초, 단일 스레드) | 횟수 | 중앙값 | 95% | 최대 |")
    a("|---|---:|---:|---:|---:|")
    for lab, k in (("경기 전 계산", "pregame"), ("라인업 변경(대타 등) 재계산", "lineup_change"),
                   ("투수 교체 재계산(현재 하프)", "pitching_change"), ("경기당 합계", "per_game_total")):
        d = s[k]
        a(f"| {lab} | {d['n']} | {fmt(d['median'], 2)} | {fmt(d['p95'], 2)} | {fmt(d['max'], 2)} |")
    if r.get("async"):
        x = r["async"]
        a("")
        a(f"**비동기 재생**({len(x['games'])}경기, 최대 속도)")
        a(f"- 재계산 중 B2E로 나간 메시지 비율: {x['recomputing_share']:.2f}.")
        a("  - 최대 속도 재생이라 재계산이 쌓여 높게 나온다. 실제 경기 속도에서는 교체 직후 몇 투구 동안만 B2E다.")
        a(f"- 워커 순수 계산 시간: 중앙값 {fmt(x['resolve_compute_s']['median'], 2)}초, "
          f"95% {fmt(x['resolve_compute_s']['p95'], 2)}초.")
        a(f"- 대기 포함 시간: 중앙값 {fmt(x['resolve_wall_s']['median'], 2)}초.")
        a(f"- 재계산이 모두 끝난 뒤 최종 타임라인이 동기 재생과 같은 경기: {x['final_equal']}/{len(x['games'])}. "
          f"WP 최대 차이 {fmt(x['max_abs_wp_diff_vs_sync'])}.")
    a("")
    a("## 7. ID 매핑")
    a("")
    a(f"- Chadwick 등록부 MLBAM↔Retrosheet 매핑: {cc['id_map_size']:,}명(B4와 같은 매핑).")
    a(f"- 교차검증 32경기 매핑 실패: {cc['unmapped_players']}명.")
    a(f"- 리플레이 전체 매핑 실패 경고: {r['warnings'].get('unmapped_player', 0)}건.")
    a("- 매핑이 없는 선수는 의사 ID `mlb<id>`를 받는다.")
    a("  - 능력치 표에 행이 없으므로 리그 사전값이 그대로 쓰인다(완전 축소).")
    a("  - 투타 방향은 피드에서 가져온다.")
    a("")
    a("## 8. 경고 집계 (리플레이 전체)")
    a("")
    for k, v in sorted(r["warnings"].items()):
        a(f"- `{k}`: {v}")
    a("")
    a("- 주입한 네트워크 오류(`feed_fetch_error`)와 정정(`feed_correction`)은 장애 주입에서 나온 것이다.")
    a("- 모르는 피드 코드 경고(`unknown_*`)가 없으면, 수집한 79경기의 모든 코드가 매핑 표에 있다는 뜻이다.")
    a("")
    a(f"타임라인 캐시: {r['timeline_cache_files']}경기, 합계 {r['timeline_cache_bytes'] / 1e6:.1f}MB(gzip, `timelines/`).")
    a("")
    a("## 9. 알려진 한계")
    a("")
    a("- **연장**: B3는 연장 이닝에서 양 팀 모두 불펜이 던진다고 고정한다(B3 단순화, 유지하기로 결정). "
      "2루 자동 주자는 상태(주자)로 반영된다. 포스트시즌에는 자동 주자가 없어 xrule=False로 계산한다.")
    a("- **라인업 변경 반영 시점**: 오프라인 평가와 같다. 새 선수가 타석에 설 때 재계산한다.")
    a("  - 대주자나 수비 교체로 들어온 선수의 타격 능력치는 그 선수가 첫 타석에 설 때 반영된다.")
    a("- **타석 중 투수 교체**: 교체 시점부터 새 투수로 다시 조회한다. 오프라인 평가(타석을 끝낸 투수 기준)와 다르다.")
    a("- **포스트시즌 Elo**: 정규시즌 종료 시점 레이팅으로 계산한다. 포스트시즌 경기 결과로 갱신하지 않는다.")
    a("- **포스트시즌 불펜 등급**: 정규시즌 경기 날짜에만 정의되어 있다. 포스트시즌 경기에서는 팀 불펜이 리그 평균으로 처리된다.")
    a("- **`wp_if_hr`/`wp_if_k`의 B2E 부분**: 결과 이후 B2 기대값에 Elo 결합을 적용한 근사다. B3 부분은 정확하다.")
    a("- **동시 경기**: 서버는 경기마다 GameServer를 하나씩 둔다. 재계산 워커는 하나다(`MAX_GAMES` 기본 1). "
      "여러 경기를 볼 때는 워커 수를 늘려야 한다.")
    a("- **대칭성 0.501, 점수차 ±20 절단**: B3에서 이어진 구조적 한계다(`docs/HANDOFF.md`).")
    a("")
    a("## 10. 확인 필요 사항 (이용 조건)")
    a("")
    a("- MLB StatsAPI 데이터의 저작권은 MLB Advanced Media에 있다. 이번 작업은 개인·비상업 용도다.")
    a("- 웹사이트를 공개 배포할 경우 다음을 확인해야 한다.")
    a("  1. 공개 웹에서 StatsAPI 파생 데이터를 표시해도 되는지(MLBAM 이용 약관, 저작권 고지 방식).")
    a("  2. 서버의 자동 폴링 빈도와 요청 출처(User-Agent)에 대한 요구 사항.")
    a("  3. 선수 이름·팀 이름 표시와 팀 로고 사용 가능 여부(로고는 쓰지 않는 것을 기본으로 한다).")
    a("  4. 리플레이 캐시(`timelines/`)를 공개 저장소에 두어도 되는지. 피드 원본은 커밋하지 않았다.")
    a("  5. Retrosheet 고지 문구 유지(README에 있음).")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
