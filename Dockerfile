FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

COPY pyproject.toml uv.lock LICENSE README.md ./
COPY rag/ ./rag/
RUN uv sync --frozen --no-dev

# Pre-download the default embedder so the first search doesn't pay the
# ~440 MB download on cold start. Reranker stays opt-in and downloads on
# first use (see RAG_RERANKER_ENABLED).
RUN HF_HOME=/models /app/.venv/bin/python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-base-en-v1.5')"

# Unprivileged user for runtime. /data and /models are the writable paths.
# The image path /app stays read-only to the runtime user.
RUN useradd --create-home --uid 1000 rag \
 && mkdir -p /data /models \
 && chown -R rag:rag /data /models

USER rag

EXPOSE 8765

ENV TRANSPORT=sse \
    PORT=8765 \
    LOG_LEVEL=INFO \
    RAG_LOG_FORMAT=json \
    RAG_DATA_DIR=/data \
    HF_HOME=/models \
    RAG_WATCHER=poll \
    RAG_HEALTH_PORT=8766

EXPOSE 8766

VOLUME ["/data", "/models", "/sources"]

# Liveness probe hits the dedicated /healthz endpoint on the sidecar
# health server. Uses stdlib only (no curl in the image). The start
# period covers worst-case first-run model loading.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD /app/.venv/bin/python -c "\
import urllib.request, sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8766/healthz', timeout=3).status == 200 else 1)" \
    || exit 1

CMD ["/app/.venv/bin/rag-server"]
