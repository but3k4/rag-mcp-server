FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

# Tesseract OCR + language data files for English, Portuguese, Spanish.
# tesserocr is a Cython wrapper that needs libtesseract / libleptonica
# headers at install time, so the -dev packages are installed alongside
# the runtime libraries.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libleptonica-dev \
        libtesseract-dev \
        pkg-config \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-osd \
        tesseract-ocr-por \
        tesseract-ocr-spa \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock LICENSE README.md ./
COPY rag/ ./rag/
RUN uv sync --frozen --no-dev

# Pre-download the default embedder and reranker so the container runs
# fully offline at runtime. The image sets HF_HUB_OFFLINE=1, so any model
# that is not cached here would fail to load.
RUN HF_HOME=/models /app/.venv/bin/python -c "\
from sentence_transformers import CrossEncoder, SentenceTransformer; \
from rag.config import DEFAULT_EMBEDDER_MODEL, DEFAULT_RERANKER_MODEL; \
SentenceTransformer(DEFAULT_EMBEDDER_MODEL); \
CrossEncoder(DEFAULT_RERANKER_MODEL)"

# Pre-download Docling layout + TableFormer into the HF cache so PDF
# parsing works under HF_HUB_OFFLINE=1.
# DOCX/PPTX/XLSX use rule-based backends and do not need extra weights.
RUN HF_HOME=/models /app/.venv/bin/python -c "\
from docling.datamodel.base_models import InputFormat; \
from docling.document_converter import DocumentConverter; \
DocumentConverter().initialize_pipeline(InputFormat.PDF)"

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
    HF_HUB_OFFLINE=1 \
    RAG_WATCHER=poll \
    RAG_HEALTH_PORT=8766 \
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata/

EXPOSE 8766

VOLUME ["/data", "/models", "/sources"]

# Liveness probe hits the dedicated /healthz endpoint on the sidecar
# health server.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD /app/.venv/bin/python -c "\
import urllib.request, sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8766/healthz', timeout=3).status == 200 else 1)" \
    || exit 1

CMD ["/app/.venv/bin/rag-server"]
