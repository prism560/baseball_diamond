# API (실시간 WP 서버)

서버 코드는 `src/wp/live/api.py`이고 FastAPI로 만들었다. 실행 방법은 다음과 같다.

```bash
PYTHONPATH=src uvicorn wp.live.api:app --port 8000
```

- 전체 스키마는 `docs/openapi.json`에 있다(`python -m wp.live.export_openapi`로 갱신). 서버가 떠 있으면 `/openapi.json`에서도 받을 수 있다.
- 설정은 모두 환경 변수로 받는다. 목록은 `.env.example`과 아래 표를 본다.

## 1. 엔드포인트

| 메서드 | 경로 | 내용 |
|---|---|---|
| GET | `/health` | 서버 상태: 모델 적재 상태, 피드 소스, 추적 중인 경기, 캐시된 타임라인 수 |
| GET | `/games/today` | 오늘 일정과, 추적 중인 경기의 최신 메시지 |
| GET | `/games/{id}/state` | 경기의 최신 메시지(실시간 경기 또는 캐시) |
| GET | `/games/{id}/timeline` | `{meta, timeline}`. 투구 시점마다 최신 메시지 하나씩, 경기 순서대로 |
| GET | `/games/{id}/corrections` | 정정 기록: 시점 키, 이전·이후 이벤트, 정정 전후 WP |
| WS | `/ws/games/{id}` | 실시간 메시지. 연결하면 현재 상태부터 보낸다 |
| GET | `/replay/games` | 재생 가능한 과거 경기 목록(타임라인 캐시의 `meta`) |
| WS | `/ws/replay/{id}?speed=1\|5\|20\|max&from=seq` | 캐시된 타임라인을 실시간 경기처럼 흘려보낸다 |
| GET | `/debug` | 개발용 디버그 페이지(점수, 상태, WP 그래프, 정정 로그) |

**`{id}` 형식**은 Retrosheet식 경기 ID `HHHYYYYMMDDN`이다.
- `HHH`는 홈팀 코드, `N`은 단일 경기 0, 더블헤더 1·2다.
- 예: `MIA202508230`

**리플레이 WebSocket**
- 클라이언트가 보내는 명령:
  - `{"cmd":"pause"}`, `{"cmd":"resume"}`
  - `{"cmd":"seek","seq":n}` 또는 `{"cmd":"seek","index":i}`
  - `{"cmd":"speed","value":"5"}`
- 서버는 시작할 때 `{"type":"replay_start","n":...}`를 보낸다. 이어서 메시지마다 `replay_index`(타임라인 위치)를 붙여 보낸다.
- 끝나면 `{"type":"replay_end"}`를 보낸다. 끝난 뒤에도 연결을 유지하므로 `seek`로 다시 재생할 수 있다.
- 재생 간격은 메시지 `ts_feed`의 차이를 `speed`로 나눈 값이다. `max`는 기다리지 않는다.
- 리플레이는 캐시에서만 읽는다. B3를 다시 계산하지 않는다.

**CORS**: `CORS_ORIGINS`에 넣은 도메인만 허용한다(쉼표로 구분). 비워 두면 CORS 헤더를 보내지 않는다.

## 2. 메시지 형식 (지시서 7.2)

```json
{"game_id":"MIA202508230","ts_feed":"2025-08-23T23:10:05.123Z","ts_out":1793946882.2,"seq":57,
 "status":"live","inning":3,"half":"bottom","outs":1,"bases":[true,false,false],"count":[1,2],
 "away":2,"home":1,"batter":"Stowers","pitcher":"Gausman","last_event":"Foul",
 "wp_home":0.4123,"pct_away":59,"pct_home":41,"leds_home":16,
 "wp_if_hr":0.5871,"wp_if_k":0.3902,"wpa_last":-0.0061,
 "model":"AVG","info_level":"lineup","recomputing":false,
 "key":["24","3"],"phase":"pitch"}
```

| 필드 | 뜻 |
|---|---|
| `game_id` | 경기 ID(위 형식) |
| `ts_feed` | 이 시점을 만든 피드 이벤트 시각(ISO). 상태 메시지에서는 피드 타임코드 |
| `ts_out` | 서버가 메시지를 만든 시각(유닉스 초) |
| `seq` | 경기 안에서 증가하는 번호. 정정되거나 재계산이 끝나 같은 시점의 값이 바뀌면, 같은 `key`로 더 큰 `seq`의 메시지가 다시 나간다 |
| `status` | `scheduled`, `pregame`, `live`, `review`(챌린지·리뷰 중), `delayed`, `suspended`, `final`, `postponed`, `cancelled` |
| `inning`, `half` | 이닝, `top`/`bottom` |
| `outs` | 아웃 수(0–3). 3은 하프이닝이 끝난 직후다 |
| `bases` | `[1루, 2루, 3루]` 주자 유무 |
| `count` | `[볼, 스트라이크]`. 이 시점 이후의 카운트 |
| `away`, `home` | 점수 |
| `batter`, `pitcher` | 박스스코어 표시 이름 |
| `last_event` | 이 시점을 만든 이벤트 설명(피드 원문) |
| `wp_home` | 홈 승리확률. 반올림하지 않은 실수이고 평활하지 않는다. 그래프와 로그 오즈 눈금에 쓴다 |
| `pct_home`, `pct_away` | `pct_home = round(100·wp_home)`, `pct_away = 100 − pct_home`. 합은 항상 100이다. 경기가 끝나면 100/0 |
| `leds_home` | 홈 쪽 LED 칸 수(N칸 중, 기본 40). 경계에서 깜빡이지 않도록 0.3칸 히스테리시스를 둔다 |
| `wp_if_hr`, `wp_if_k` | 현재 타석이 홈런·삼진으로 끝나면 바뀔 홈 WP. 타석 사이 시점에서는 `null`이다 |
| `wpa_last` | 직전 시점 대비 WP 변화량 |
| `model` | 표시 값을 낸 모델: `AVG`, `B3`, `B2E`, `B2` |
| `info_level` | `lineup`(실제 선발 라인업으로 계산), `none`(라인업 전, B2E만) |
| `recomputing` | B3 재계산이 진행 중이어서 B2E를 대신 표시하고 있으면 `true` |
| `key`, `phase` | 시점 키 `[타석 인덱스, 이벤트 인덱스]`, 그리고 시점 종류. 키의 이벤트 자리에는 숫자 외에 `s`(타석 시작), `r`(타석 결과)도 온다 |

**`phase` 종류**

| 값 | 시점 |
|---|---|
| `start` | 타석 시작 |
| `sub` | 교체 |
| `pre` | 첫 투구 전 주자 이벤트(도루, 견제 등) |
| `pitch` | 타석이 이어지는 투구 |
| `mid` | 첫 투구 뒤 주자 이벤트 |
| `pitch_end` | 타석을 끝낸 투구(결과 반영 전) |
| `end` | 타석 결과 반영 후 |
| `status` | 경기 전·지연 등 상태 메시지 |

**모델 표시 규칙**
- 기본은 `MODEL=AVG`다.
- B3 재계산이 진행 중이거나 B3 계산이 실패하면, 해당 시점에 B2E 값을 보내고 `model:"B2E"`로 표시한다. 재계산 중이면 `recomputing:true`도 붙인다.
- 재계산이 끝나면 영향을 받은 시점들을 AVG 값으로 다시 보낸다(더 큰 `seq`).

**`wp_if_hr`/`wp_if_k` 계산**
- B3 값은 결과별 이후 값 `G`다. DP 조회로 정확히 구한다.
- B2E 값은 결과 이후의 B2 기대값에 Elo 결합을 적용한 근사다. AVG 값은 그 둘을 결합한 것이다.

## 3. 타임라인 캐시

- 위치는 `TIMELINE_CACHE_DIR`(기본 `timelines/`)이고, 경기당 gzip JSON 파일 하나다.
- 형식: `{"meta":{game_id, game_pk, date, home, visitor, game_type, p_pregame, xrule, n_points, final, tags}, "timeline":[메시지…], "corrections":[…]}`
- 웹 요청은 이 파일만 읽는다. 경기당 B3 계산(약 13–20초)을 반복하지 않는다.
- 배포할 때는 이 경로를 볼륨으로 바꾼다. 나중에 DB로 옮기려면 `TimelineCache`의 `save`/`load`/`list`만 바꾸면 된다.

## 4. 데이터 소스 (MLB StatsAPI, 2026-11-04 실제 응답으로 확인)

| 용도 | 엔드포인트 |
|---|---|
| 일정 | `GET https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=YYYY-MM-DD[&teamId=..]` (시즌 전체는 `&season=YYYY&gameType=R,F,D,L,W`) |
| 경기 라이브 피드 | `GET https://statsapi.mlb.com/api/v1.1/game/{gamePk}/feed/live` |
| 피드 타임코드 목록 | `GET .../api/v1.1/game/{gamePk}/feed/live/timestamps` → `["YYYYMMDD_HHMMSS", …]` |
| 특정 시점 스냅샷 | `GET .../api/v1.1/game/{gamePk}/feed/live?timecode=YYYYMMDD_HHMMSS` |
| 코드 목록 | `GET .../api/v1/pitchCodes`, `/eventTypes`, `/gameStatus`, `/reviewReasons` (사본은 `data/external/meta/`) |
| 쓰지 않음 | `api/v1/game/{pk}/feed/live/diffPatch`는 404를 반환했다 |

**확인한 피드 관행** (정규화기 주석과 테스트에 근거가 있다)
- 경기 상태 `gameData.status.codedGameState`
  - `S`/`P` 경기 전, `I` 진행 중, `M`/`N` 챌린지·리뷰 중
  - `T`/`U` 서스펜디드, `F`/`O` 종료, `D` 연기, `C` 취소
- 투구 판정 `playEvents[].details.call.code`
  - 자동 고의4구와 피치클락 판정은 `type:"no_pitch"` 이벤트에 `VB`/`VP`/`VC`/`AC`/`AB` 코드로 온다. Retrosheet는 이를 V/A 투구로 기록한다.
- 이벤트 뒤 카운트 `count`
  - 타석을 끝낸 투구 뒤에는 볼 4 또는 스트라이크 3으로 표시된다.
  - 인플레이 투구는 투구 전 카운트 그대로다.
- 주자 이동 `runners[].details.playIndex`: 원인이 된 `playEvents` 인덱스를 가리킨다.
- 연장 자동 주자 `runner_placed`: 주자 이동 항목이 없다. 주자는 이벤트의 `player`, 위치는 `base`로 정한다.
  - 포스트시즌에는 자동 주자가 없다.
- 대주자 `offensive_substitution`: `base` 필드로 그 루의 주자를 교체한다.
- 라인업 `boxscore.teams.{side}.players[].battingOrder`: `"700"`은 7번 선발, `"701"`은 7번 자리의 첫 교체 선수다.

**폴링 예의** (`src/wp/live/feed_client.py`)

| 상황 | 간격 |
|---|---|
| 진행 중 | 7초 |
| 응답이 바뀌지 않음 | 1.5배씩 늘려 최대 30초 |
| 오류 | 2초부터 두 배씩, 최대 120초 |
| 경기 전·지연·서스펜디드 | 60초 |
| 종료 | 폴링 중지 |

**이용 조건**
- MLB 데이터의 저작권은 MLB Advanced Media에 있다. 개인·비상업 용도로만 쓴다.
- 공개 배포할 때의 조건은 `reports/realtime_report.md`의 "확인 필요 사항"을 본다.
