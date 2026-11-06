# 운영 절차 (RUNBOOK)

시즌 중 매일 서버를 돌리는 절차다. 명령은 저장소 루트에서 실행한다. 경로 설정은 `PYTHONPATH=src`다.

## 0. 처음 한 번 (새 컨테이너·서버)

1. 패키지를 설치한다.
   ```bash
   pip install -r requirements.txt -r requirements-server.txt
   ```
2. Retrosheet 데이터를 다운로드하고 파싱해 PA 표를 만든다. 약 6분 걸린다.
   ```bash
   make data/processed/pa.parquet
   ```
3. 선수 ID 매핑(Chadwick register)은 처음 쓸 때 `data/raw/register/`에 자동으로 받는다.
4. 시즌 B2E 계수를 만든다. 시즌마다 한 번, 약 10분 걸린다.
   ```bash
   python scripts/update_ratings_daily.py --season-setup 2026
   ```
   결과는 `models/live/b2e_2026.json`이다.

## 1. 매일 (경기 다음 날 아침)

1. 어제 종료된 경기를 PA 표, 경기 표, 게임로그 행으로 변환한다.
   ```bash
   python scripts/update_ratings_daily.py
   ```
   - 결과 파일: `data/processed/pa_live.parquet`, `games_live.parquet`, `gamelogs_live.parquet`
   - 이미 Retrosheet에 있는 경기는 건너뛴다.
   - 출력에 `unmapped players`가 있으면 Chadwick 등록부에 아직 없는 선수다.
     - 그 선수는 리그 평균 사전값과 완전 축소로 처리된다.
     - 등록부가 갱신되면 `data/raw/register/`를 지우고 다시 받는다.
2. 서버를 재시작한다. 모델 묶음은 시작할 때 다시 적재한다(능력치, 불펜 등급, Elo를 어제까지 반영, 약 75초).
   - 재시작하는 동안에도 `/health`와 리플레이는 바로 응답한다.
3. 서버 로그에서 다음 경고를 확인한다.

   | 경고 | 뜻 | 조치 |
   |---|---|---|
   | `unknown_pitch_code`, `unknown_result_type`, `unknown_action_type`, `unknown_game_status` | 매핑 표에 없는 새 피드 코드. 경기는 멈추지 않고 안전한 기본값으로 처리된다 | `src/wp/live/codes.py`에 추가하고 `tests/test_live_codes.py`에 항목을 넣는다 |
   | `resolve_failed` | B3 재계산 실패. 그 경기는 B2E로 표시된다 | 원인을 확인한다 |
   | `feed_correction` | 피드 정정(챌린지 번복 등). 정상 동작이다 | 정정 전후 WP는 `/games/{id}/corrections`에서 본다 |

## 2. 경기 중

- 서버 설정: `FEED_SOURCE=statsapi`, `TEAM_ID=<팀 id>`
- 서버가 하는 일:
  - 일정은 5분마다 확인한다.
  - 추적하는 경기의 피드는 7초 간격으로 폴링한다(규칙은 `docs/API.md` 4절).
  - B3 재계산(투수 교체, 대타)은 같은 컨테이너의 워커 프로세스에서 돈다. 그동안 B2E 값을 `recomputing:true`로 내보낸다.
- 경기가 끝나면 타임라인이 `TIMELINE_CACHE_DIR`에 저장된다. 이후 리플레이 목록(`/replay/games`)에 나온다.
- **재시작 시 복구**: 상태는 매번 최신 피드 전체에서 다시 만든다. 경기 중에 재시작해도 첫 폴링에서 현재 상태가 복구된다.
  - 다만 재계산이 진행 중이던 B3 값은 처음부터 다시 계산한다. 경기 전 계산 약 8초, 그 뒤 교체 재계산이 따로 든다.

## 3. 리플레이 검증과 캐시 갱신

1. 과거 경기 피드를 받는다. 약 2분 걸린다.
   ```bash
   python scripts/fetch_feeds.py --postseason
   ```
2. 검증을 돌리고 타임라인 캐시를 만든다. 2개 프로세스로 약 40분 걸리고, 이어하기가 된다.
   ```bash
   OMP_NUM_THREADS=1 python scripts/replay.py --part 0 --parts 2 &
   ```
   ```bash
   OMP_NUM_THREADS=1 python scripts/replay.py --part 1 --parts 2
   ```
3. 리포트를 만든다.
   ```bash
   PYTHONPATH=src python -m wp.live.report_realtime
   ```
   결과는 `reports/realtime_report.md`다.
4. 정규화기 교차검증(피드 PA와 Retrosheet PA 대조)을 돌린다.
   ```bash
   PYTHONPATH=src python -m wp.live.crosscheck_feed
   ```

## 4. 컨테이너

1. 이미지를 빌드한다.
   ```bash
   docker build -t wp-api .
   ```
2. 리플레이 전용으로 실행한다. 이미지에 들어 있는 `timelines/`를 쓴다.
   ```bash
   docker run -p 8000:8000 -e CORS_ORIGINS=http://localhost:3000 wp-api
   ```
3. 실시간 모드로 실행한다. `data/`는 호스트에서 만들어 볼륨으로 붙인다.
   ```bash
   docker run -p 8000:8000 -v $PWD/data:/app/data -e FEED_SOURCE=statsapi -e TEAM_ID=147 wp-api
   ```

**빌드 옵션** (TLS를 가로채는 프록시나 Docker Hub 제한이 있는 환경용)

- `--build-arg BASE_IMAGE=mirror.gcr.io/library/python:3.12-slim`: Docker Hub 대신 미러를 쓴다(429 응답 회피).
- `--secret id=pipca,src=<CA 번들>`: pip가 신뢰할 CA를 넘긴다. 빌드 중에만 쓰이고 이미지에는 남지 않는다.

이 개발 환경에서 확인한 빌드 명령은 다음과 같다(2026-11-06).

```bash
docker build --network host --secret id=pipca,src=/root/.ccr/ca-bundle.crt --build-arg HTTPS_PROXY=$HTTPS_PROXY --build-arg BASE_IMAGE=mirror.gcr.io/library/python:3.12-slim -t wp-api .
```

- 이미지 크기는 1.1GB다.
- 리플레이 전용으로 실행하면 `/health`가 응답하고 헬스체크가 `healthy`가 된다.

## 5. 이 개발 환경의 제약

- 클라우드 개발 컨테이너는 재시작되면 `data/`가 사라질 수 있다. 이때는 0절을 다시 한다.
- 오래 걸리는 작업은 경기별로 저장되고 이어하기가 된다(`data/live/replay_runs/`).
- 타임라인 캐시 `timelines/`는 git에 커밋한다.
