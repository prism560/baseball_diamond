# syntax=docker/dockerfile:1
# WP API server (one process; the B3 re-solve worker is forked inside the same container).
# Replay-only mode (FEED_SOURCE=off) needs only the timeline cache (timelines/).
# Live mode (FEED_SOURCE=statsapi) also needs the processed data and models: mount data/ as a volume
# (built with `make data/processed/pa.parquet`, see docs/RUNBOOK.md).
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}
WORKDIR /app
ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 \
    PORT=8000 FEED_SOURCE=off MODEL=AVG TIMELINE_CACHE_DIR=/app/timelines CORS_ORIGINS=
COPY requirements.txt requirements-server.txt ./
# optional build secret "pipca": extra CA bundle for pip behind a TLS-intercepting proxy (not stored in the image)
RUN --mount=type=secret,id=pipca sh -c 'if [ -f /run/secrets/pipca ]; then export PIP_CERT=/run/secrets/pipca; fi; \
    pip install --no-cache-dir -r requirements.txt -r requirements-server.txt'
COPY src ./src
COPY models ./models
COPY reports/b3_check.json reports/b3_results.json ./reports/
COPY timelines ./timelines
COPY scripts ./scripts
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import os,urllib.request;urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health',timeout=4)"
CMD ["sh", "-c", "uvicorn wp.live.api:app --host 0.0.0.0 --port ${PORT}"]
