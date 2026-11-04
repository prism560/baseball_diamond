# 인수인계 (B0–B4 완료 시점)

이 파일 하나로 다음 세션이 이어서 일할 수 있게 정리했다. 상세는 `reports/b0_b1_report.md`, `reports/b2_report.md`, `reports/b3_report.md`, `reports/b4_report.md`에 있다.
- 작업 브랜치: B0–B4는 `claude/elegant-babbage-gngg3x`(`main`에 병합됨), 실시간 단계는 `claude/handoff-docs-review-v2lwvp`
- 저장소: `prism560/baseballdiamond`

## 현재 상태

평가는 2023–2025 워크포워드이고, 수치는 경기 WP 로그 손실(타석 시작 시점)이다. 신뢰구간은 경기 단위 블록 부트스트랩 1,000회로 구했다.

| 단계 | 결과 |
|---|---|
| 데이터 | Retrosheet 1908–2025 이벤트 파일을 자체 파이썬 파서로 처리. 2023–2025 공식 파싱 파일과 행 단위 대조에서 상태 오류 0건(568,664행) |
| B0 | 상태 테이블 + 계층 축소. 0.4705 |
| B1 | B0 + Elo. B0보다 −0.0043 |
| B2 | 리그 평균 정확 DP + 카운트 체인(IPF 정합 1e-12). B0와 같은 수준(+0.0001, CI가 0 포함) |
| B3 | 선수 매치업 + 투수 정책을 DP 안에 넣음(Elo 없음). B1·B2E와 구분 안 됨(B3−B1 +0.0000 [−0.0015, +0.0014]) |
| B4 | 결합 모델 AVG가 B3보다 −0.0010, B1보다 −0.0009로 유의하게 나음. B2E와는 −0.0006 [−0.0016, +0.0002]로 구분 안 됨 |

- **B2E**는 B2 + Elo로, B1과 같은 형태로 만든 기준선이다.
- **경기 수준 최소 탐지 효과는 약 0.0021**이다. 이보다 작은 차이는 3시즌 전체로도 "탐지할 수 없음"이다.

## 서빙 결정

- **기본 모델은 AVG**다: `WP = σ(0.55·logit(B3) + 0.45·logit(B2E))`. 가중치는 평가 시즌 이전 데이터로 정했다.
- **장애 시 대체 모델은 B2E**다. 테이블 조회 + 로지스틱이라 계산이 즉시 끝난다.
- **대타 재계산 등 오래 걸리는 재계산**은 비동기로 돌리고, 그동안 B2E 값을 표시한다.
- **B3 엔진의 매치업은 기존 q를 쓴다.** q'은 아래 이유로 서빙에 넣지 않았다.
- 호출 순서는 `docs/REALTIME.md`에 있다.

## 게이트에서 빠진 것과 이유

- **당일 컨디션** (스트라이크 비율, 투구 수)
  - 타석 수준 개선이 −0.0006으로, 게이트 기준 0.002에 못 미친다.
  - TTO와 약 80% 겹친다. TTO 위에 더하면 추가 이득이 −0.0001이다.
- **구속·구종** (Statcast): 데이터가 없어 미실시다. 수집·연결 코드와 테스트는 준비돼 있다(`src/wp/statcast.py`).
- **B4 5단계** (q'로 경기 WP 재평가): 실시하지 않기로 했다. q'의 타석 이득 0.0007이 최소 탐지 효과의 약 1/3이라 검정력이 부족하다.
  - q' = 구장 중립 능력치 + 다항 로지스틱, 코드는 `src/wp/matchup_v2.py`

## 알려진 한계

- **쿠어스 필드 단타 편향**: q'에서 실제−예측 +0.010이다. 기존 q에서는 +0.001이었다.
- **대타 재계산 시간**: 5회 이전 대타는 6.3초(단일 스레드)로 5초 기준을 넘는다. 커널 캐시로 줄어드는 시간은 5–10%뿐이다.
- **대칭성 0.501 (구조적)**: 투수 정책이 점수차에 의존해, 말 공격 쪽이 정보 이점을 가진다. 정책이 점수에 의존하지 않으면 정확히 0.5다(테스트로 확인).
- **점수차 ±20 절단**: B3 DP의 WP 오차는 최대 3e-5, |점수차| ≤ 10에서 1e-7이다.
- **B3 단순화**:
  - 연장은 양 팀 모두 불펜으로 고정한다.
  - 대타 투입 확률은 모델링하지 않는다.
  - 타석 중 주자 이벤트는 진루 테이블에 흡수한다.

## 환경 제약

- **차단된 사이트**: retrosheet.org, statsapi.mlb.com, baseballsavant.mlb.com.
  - Retrosheet 원본은 GitHub 미러 `chadwickbureau/retrosheet`에서 받는다.
  - 선수 ID 매핑은 `chadwickbureau/register`를 쓴다.
- **외부 C 코드 빌드 불가**: 정책상 Chadwick을 빌드할 수 없어 자체 파서를 쓴다.
- **컨테이너 재시작**: 약 30분마다 재시작된다. 장시간 실행은 경기별 저장과 이어하기(`wp.run_b3_games`)로 처리하고, 주기적으로 커밋·푸시한다. `data/`는 git에서 제외된다.
- **메모리**: 약 15GB다. 타석 피처 배열(행 수 × 피처 × 10범주)을 여러 개 동시에 들면 메모리 부족으로 종료된다. 이 경우 2019년 이후 행만 남겨서 처리한다.
- **출처 기록**: 모든 산출물에 `*.meta.json`(입력·코드 해시)이 있다. 코드를 바꾸면 하위 산출물이 오래된 것으로 판정되어 실패하므로 `make`로 다시 만든다.

## 실행 방법

| `make` 목표 | 내용 | 시간 |
|---|---|---|
| `data/raw/download_manifest.json` | 미러 다운로드 1901–2025 | 약 1분 |
| `data/processed/games.parquet` | 파싱, 검증(0절) | 약 4분 |
| `data/processed/pa.parquet` | 타석 표 | 약 1분 |
| `b0b1` | B0·B1 평가와 리포트 | 약 1분 |
| `b2` | B2 평가와 리포트 | 약 2분 |
| `reports/b3_check.json` | 매치업 중간 점검 | 약 5분 |
| `b3-runs` | 경기별 B3 실행. 본 비교 전체 + ablation 1/9, 이어하기 가능 | 약 15시간 |
| `b3` | B3 평가와 리포트 | 약 10분 |
| `data/processed/crosscheck_retrosheet.json` | 공식 파싱 파일 교차검증 | 약 2분 |
| `b4` | B4 게이트, 결합, 리포트 | 약 30분 |
| `test` | 테스트 62개 | 약 2분 |

- `b3-runs`의 4코어 기준 내역: 본 비교 약 6.5시간, ablation 약 5시간, 2022 시즌 부분집합 약 1시간.
- `b4`의 30분에는 `wp.run_b4_q`, `wp.run_b4_condition`, `wp.run_b4_combine`이 포함된다.

## 다음 작업

- **실시간 파이프라인**: `src/wp/live/`는 아직 없다. 지시서는 별도로 받는다.
- **재사용할 함수**:
  - `wp.b3_game.build_context`, `game_inputs`, `evaluate_game`
  - `wp.dp_b3.Engine.solve`: `override`, `start`, `kernel_cache` 인자 사용
  - `wp.count_chain.fit`
  - `wp.wp_b2.PitchWP`: B2 조회
- **ID 매핑**: 라이브 피드의 MLBAM ID는 `wp.statcast.mlbam_to_retro`로 Retrosheet ID에 대응시킨다.

## 사용자가 로컬에서 할 일

- **(선택) Statcast 수집**: `python scripts/fetch_statcast.py --start 2019-03-01 --end 2025-11-30`을 실행해 `data/external/statcast/`에 넣으면 B4 3단계 게이트를 진행할 수 있다.
- **(예정) 실시간 피드 수집**: 실시간 단계에서 피드 수집 스크립트를 준비한다. statsapi.mlb.com이 이 환경에서 막혀 있어 로컬 실행이 필요하다.

## 실시간 단계 진행 상황 (2026-11-04, 약 30% 지점에서 중단)

지시서: `claude_code_prompt_realtime.md`와 승인 답신(수정 사항 포함). 사용자 요청으로 중간에 멈췄다.

### 끝난 것
- **네트워크**: `statsapi.mlb.com`, `baseballsavant.mlb.com`이 이제 열려 있다(사용자가 Allowed domains에 추가).
- **엔드포인트 확정**(실제 응답 기준, `scripts/fetch_feeds.py` 주석):
  - `v1/schedule`, `v1.1/game/{pk}/feed/live`, `.../feed/live/timestamps`, `.../feed/live?timecode=`
  - `diffPatch`는 404라 쓰지 않는다.
- **피드 수집**: `data/external/feeds/`에 있다(git 제외, 135MB). 다시 받으려면 `python3 scripts/fetch_feeds.py --postseason`(약 2분)을 실행한다.
  - 2025 정규시즌 32경기: `scripts/feed_games_2025.json`. Retrosheet로 연장, 끝내기, 투수 교체 다수, 5회 이전 대타, 역전, 일방적 경기, 무작위를 골랐다.
  - 2025 포스트시즌 47경기 전체
  - 실제 시점 스냅샷 3경기: 813071·813073(챌린지 번복), 813042(연장 끝내기), 1,545개
  - StatsAPI 코드 목록: `data/external/meta/`
- **정규화기**: `src/wp/live/codes.py`, `normalize.py`, `reconstruct.py`, `ids.py`, `pa_from_feed.py`
- **교차검증**(`python -m wp.live.crosscheck_feed` → `reports/live_crosscheck.json`)
  - 32경기 2,587타석에서 상태·결과·ID·투구 수 불일치 **0건**
  - ID 매핑 24,672명, 매핑 실패 0명
  - 확인된 피드 관행:
    - 자동 고의4구와 피치클락 판정은 `no_pitch` 이벤트(VB/VP/VC/AC)로 온다. Retrosheet의 V/A에 해당한다.
    - 연장 자동 주자(`runner_placed`)에는 주자 이동 정보가 없다. 주자는 이벤트의 `player`와 `base`로 처리한다.
    - 대주자는 교체 이벤트의 `base`로 그 루를 차지한다.
    - **포스트시즌에는 자동 주자가 없다**(xrule=False).
- **서빙 골격**: `models.py`(시즌 묶음, 적재 약 74초), `tracker.py`(evaluate_game의 증분판), `solver.py`(동기 / fork 워커), `serving.py`(GameServer: 중복·정정·역행 처리, B2E 대체, 메시지 7.2), `output.py`(퍼센트, LED 히스테리시스), `log.py`(구조화 로그, `SENTRY_DSN` 선택 연동), `offline_ref.py`(오프라인 기준값)
- **오프라인 일치 (1경기만 확인)**: MIA202508230의 98타석 시작 시점에서 B3·B2E·AVG 차이가 최대 1.1e-16이다.

### 측정값 (참고)
- B3 시즌 문맥 구축 약 60초
- 경기 전 계산 6.2초
- 경기당 재계산 21회, 합계 14.7초
- 전체 해 1개는 116MB다. 워커에서 받는 비용은 unpickle 0.02초다.

### 남은 일 (지시서 순서)
1. 투구 단위 오프라인 일치 검증, 그리고 주자 이벤트 뒤 투구의 의도된 차이 분포(평균, 최대, 2.5%p 초과 비율)
2. `replay.py` 하네스
   - 합성 스냅샷과 실제 스냅샷 재생, 둘의 대조
   - 속도 조절, 장애 주입
   - 타임라인 캐시: `timelines/`(git 추적), 경로는 `TIMELINE_CACHE_DIR`
3. 32+47경기 전체 검증(종료 일관성, 장애 내성, 지연 분포)
4. API 서버(FastAPI: REST, WebSocket, 리플레이 제어, CORS, OpenAPI, 디버그 페이지), 선택 옵션 MQTT
5. `scripts/replay.py`, `scripts/update_ratings_daily.py`
   - 피드 PA로 `pa_live.parquet`, `games_live.parquet`, `gamelogs_live.parquet`를 만든다. `models.py`가 이미 이 파일을 읽는다.
   - 2026 시즌 B2E 계수 적합
6. `Dockerfile`, `.env.example`(`PORT`, `CORS_ORIGINS`, `TIMELINE_CACHE_DIR`, `FEED_SOURCE`, `MODEL`), `/health` 로컬 확인
7. 문서와 리포트
   - `docs/API.md`(데이터 소스 절 포함), `docs/RUNBOOK.md`, `docs/ROADMAP.md`
   - `reports/realtime_report.md`
   - `samples/timeline_<id>.json` 3개
   - 테스트(매핑 표 항목별 포함)

### 알아둘 점
- 포스트시즌 경기는 Retrosheet에 없다. 따라서 오프라인 일치 검증은 정규시즌 경기만 할 수 있다. 포스트시즌은 종료 일관성, 장애 내성, 지연만 검증한다.
- 포스트시즌 Elo는 정규시즌 종료 시점 레이팅으로 계산한다(포스트시즌 경기로 갱신하지 않음). 리포트에 적어야 한다.
- 라인업 변경은 오프라인과 같이 **새 선수가 타석에 설 때** 반영한다. 대주자나 수비 교체는 그 선수가 처음 타석에 설 때 재계산된다.
- 타석 도중 투수가 바뀌면, 실시간은 그 시점의 투수를 쓰고 오프라인은 타석을 끝낸 투수를 쓴다. 이 차이는 의도된 것으로 분류한다.
- 컨테이너가 새로 뜨면 `pip install -r requirements.txt fastapi uvicorn httpx`를 실행한 뒤 `make data/processed/pa.parquet`(약 6분)을 돌린다.
