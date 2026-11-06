# 로드맵

새 세션은 이 파일과 `docs/HANDOFF.md`를 함께 읽는다.

| 단계 | 내용 | 시작 조건 | 산출물 | 상태 |
|---|---|---|---|---|
| ① | 실시간 파이프라인 + API | B0–B4 완료 | `src/wp/live/`, `docs/API.md`, `docs/RUNBOOK.md`, `reports/realtime_report.md`, `timelines/`, `samples/` | 완료(이 브랜치) |
| ② | 리플레이 화면 시안 | ①의 `samples/timeline_<id>.json` 3개와 `docs/API.md` 2절(메시지 필드) | 별도 Claude 대화의 아티팩트(화면 시안) | 대기 |
| ③ | 웹사이트 본 개발 | ②의 시안 확정 | 웹 앱 저장소 또는 폴더. 실시간 화면(`/ws/games/{id}`), 리플레이 화면(`/replay/games`, `/ws/replay/{id}`), WP 그래프(`/games/{id}/timeline`) | 대기 |
| ④ | 배포 | ③의 웹 앱이 로컬에서 API와 연동됨 | 서버는 Railway(이 저장소의 `Dockerfile`), 웹은 Vercel. 환경 변수는 `.env.example`, 타임라인 캐시는 볼륨 | 대기 |
| ⑤ | 오류 추적 | 시즌 개막 전 | Sentry 프로젝트와 `SENTRY_DSN` 설정. 연동 지점은 `src/wp/live/log.py`에 있고, 경고 이벤트 이름은 `docs/RUNBOOK.md` 1절에 있다 | 대기 |

## 단계별 메모

- **②**
  - 샘플은 7.2 메시지 형식 그대로다.
  - 접전 역전승, 연장 끝내기, 일방적 경기 3종이다.
  - 필드 설명은 `docs/API.md`에 있다.
- **③**
  - CORS 허용 도메인은 서버의 `CORS_ORIGINS`로 정한다.
  - 리플레이 WebSocket은 `pause`/`resume`/`seek`/`speed` 명령을 받는다.
  - `wp_home`은 반올림 전 실수다. 그래프와 로그 오즈 눈금에 쓴다.
- **④**
  - 서버는 한 프로세스로 뜨고, 재계산 워커는 같은 컨테이너에서 fork된다.
  - 실시간 모드는 메모리 약 3GB가 필요하다(시즌 모델 약 2GB + 워커).
  - 리플레이 전용 모드(`FEED_SOURCE=off`)는 모델을 적재하지 않아 가볍다.
  - 공개 배포 전에 MLB 데이터 이용 조건을 확인해야 한다(`reports/realtime_report.md`의 "확인 필요 사항").
- **⑤**
  - 지금은 구조화된 JSON 경고 로그로만 남는다.
  - `SENTRY_DSN`을 설정하고 `sentry-sdk`를 설치하면 경고 이상이 Sentry로 간다.
