# RAG Server

A local-first MCP server that indexes documents from configured directories and exposes semantic search to Claude Code, Claude Desktop, Cursor, and VSCode. Runs fully offline, with no external API calls.

## Background

I've been running this locally for some time and found it useful in day-to-day engineering workflows, so I decided to share it.

## What it does

Once the server is running, six MCP tools are available:

- **search_docs(query)**: Hybrid search combining dense semantic similarity and BM25 keyword matching via Reciprocal Rank Fusion. Returns the top 5 most relevant section excerpts, including source filename, section title, and score.
- **reindex()**: Incremental re-index on demand. Only processes files whose SHA-256 hash has changed, and prunes entries for files that were renamed or deleted. Returns a summary of scanned, updated, failed, and pruned files.
- **list_indexed_files()**: Lists all currently indexed file paths.
- **add_directory(path)**: Registers a new directory at runtime and indexes its contents. Persisted across restarts.
- **remove_directory(path)**: Unregisters a directory and removes its indexed data.
- **get_status()**: Shows configured directories, last indexed timestamps, and total file count.

Supported file types: `.txt`, `.md`, `.pdf`, `.docx`, `.pptx`, `.csv`, `.xlsx`, `.xml`. The structured formats (PDF, DOCX, PPTX, XLSX) are parsed by [Docling](https://docling-project.github.io/docling/), which produces ATX-headed markdown with preserved tables. The same chunker handles markdown, Docling output, and `.md` files uniformly.

## Why I built this

I built this project to explore how AI can be integrated into real engineering workflows beyond simple prompt usage, using a local-first approach.

## Architecture

Documents are parsed into section-aware chunks and indexed using a parent-child layout. Small overlapping chunks (children) are embedded and stored in ChromaDB. Parent sections (full section text) and all other non-vector state are stored in a sidecar SQLite database.

The Chroma index acts as a rebuildable cache over the canonical data stored in SQLite. A file watcher keeps both in sync with the filesystem. SHA-256 hashes drive incremental reindexing.

```
                   +------------------+
   source_dirs --- |     Indexer      | --- SHA-256 hash check (skip unchanged)
                   +------------------+
                             | upsert
                 +-----------+-----------+
                 |                       |
  +------------------------+  +----------------------------+
  | ChromaDB               |  | MetadataDB (SQLite, WAL)   |
  |                        |  |                            |
  |   docs   child chunks  |  |   kv          schema_ver,  |
  |          + embeddings  |  |               model name,  |
  |                        |  |               parser ver   |
  |                        |  |   file_hashes path -> sha  |
  |   (cosine)             |  |   parents     id, source,  |
  |                        |  |               section,     |
  |   BM25 index (memory)  |  |               text         |
  +------------------------+  +----------------------------+
                              |
                              | search
                              |
    query --> dense semantic -+
                              |-- Reciprocal Rank Fusion --+
     query --> BM25 keyword --+                            |
                                                           |
                      (optional) cross-encoder reranker <--+
                                                           |
                                              parent lookup (SQLite)
                                                           |
                                                         top-k
```

SQLite is the source of truth. The vector index can be rebuilt from SQLite without re-parsing source files, enabling embedding model swaps and future schema migrations.

## Install

```sh
# Clone or copy this project and enter the directory
cd rag-server

# Install dependencies from pyproject.toml (creates .venv).
uv sync
```

On first startup, the server downloads three sets of models from HuggingFace and caches them locally:

- `sentence-transformers/all-mpnet-base-v2` embedder (~420 MB)
- Docling layout + TableFormer weights (~360 MB), used when parsing PDF, DOCX, PPTX, or XLSX files
- (optional) `cross-encoder/ms-marco-MiniLM-L-12-v2` reranker (~140 MB), only if `reranker_enabled = true`

After that, everything runs offline. The Dockerfile pre-bakes all three sets, so the container image starts without any network call.

## Configure

Two configuration sources, in order of precedence:

1. **Environment variables** (override everything)
2. **`config.toml`** at the project root

Copy the template and edit paths:

```sh
cp config.toml.example config.toml
```

The template documents every knob. A minimal working file:

```toml
[indexing]
source_dirs = [
    "~/Documents",
]
chunk_size = 1000
chunk_overlap = 100
```

Full structure (see `config.toml.example` for comments on each knob):

```toml
[indexing]
source_dirs = [ ... ]
chunk_size = 1000
chunk_overlap = 100
max_file_size_bytes = 52428800
watcher_debounce_seconds = 0.5

[retrieval]
top_k = 5
fetch_n_multiplier = 4
rrf_k = 60
embedder_model = "sentence-transformers/all-mpnet-base-v2"
query_prefix = ""
reranker_enabled = false
reranker_model = "cross-encoder/ms-marco-MiniLM-L-12-v2"
reranker_pool_size = 20
max_query_length = 1000

[security]
# allowed_base_dirs = [ "~/Documents", "/data/shared" ]
```

Invalid values are rejected at startup with a clear `ConfigError`. The server exits instead of running with invalid configuration.

### Environment variables

**Transport and paths**

| Variable         | Default                 | Description                                                       |
|------------------|-------------------------|-------------------------------------------------------------------|
| `TRANSPORT`      | `stdio`                 | `stdio`, `sse`, or `streamable-http`. See note below.             |
| `PORT`           | `8765`                  | TCP port for HTTP transports                                      |
| `RAG_DATA_DIR`   | `<project>/.rag-data`   | Where Chroma and `metadata.db` (SQLite) persist                   |
| `RAG_CONFIG`     | `<project>/config.toml` | Alternate config file path                                        |
| `RAG_SOURCE_DIRS`| *(unset)*               | Colon-separated paths. Overrides `[indexing].source_dirs`         |

**Transport note**: `streamable-http` is the most capable transport and is recommended when you have a reverse proxy (nginx, Caddy, Traefik) terminating SSL. Claude Code and Claude Desktop require HTTPS for remote `streamable-http` endpoints, so plain HTTP only works on localhost. `sse` is simpler to get running without SSL. The Docker image defaults to it for this reason. `stdio` is the right choice for local installs launched directly by a client.

**Indexing**

| Variable                       | Default     | Description                                                             |
|--------------------------------|-------------|-------------------------------------------------------------------------|
| `RAG_CHUNK_SIZE`               | `1400`      | Chunk size in characters                                                |
| `RAG_CHUNK_OVERLAP`            | `150`       | Overlap between chunks                                                  |
| `RAG_MAX_FILE_SIZE_BYTES`      | `52428800`  | 50 MiB. Files larger than this are skipped with a warning               |
| `RAG_WATCHER`                  | `native`    | `native` or `poll`. Use `poll` for bind mounts on Docker Desktop        |
| `RAG_WATCHER_DEBOUNCE_SECONDS` | `0.5`       | Coalesce rapid save events on the same path. Set `0` to disable         |

**Retrieval**

| Variable                  | Default                                                          | Description                                                          |
|---------------------------|------------------------------------------------------------------|----------------------------------------------------------------------|
| `RAG_TOP_K`               | `5`                                                              | Number of results returned by `search_docs`                          |
| `RAG_FETCH_N_MULTIPLIER`  | `4`                                                              | Candidate pool multiplier for RRF (`top_k * multiplier` per side)    |
| `RAG_RRF_K`               | `60`                                                             | RRF K parameter. Higher flattens the ranking curve                   |
| `RAG_EMBEDDER_MODEL`      | `sentence-transformers/all-mpnet-base-v2`                        | SentenceTransformer identifier. Changing this triggers a full reindex |
| `RAG_QUERY_PREFIX`        | `""`                                                             | Prefix prepended to queries before embedding. Must match the embedder's convention |
| `RAG_RERANKER_ENABLED`    | `false`                                                          | Enable cross-encoder reranking after RRF                             |
| `RAG_RERANKER_MODEL`      | `cross-encoder/ms-marco-MiniLM-L-12-v2`                         | CrossEncoder identifier when reranking is on                         |
| `RAG_RERANKER_POOL_SIZE`  | `20`                                                             | How many deduped parents to rerank. Must be `>= top_k`               |
| `RAG_MAX_QUERY_LENGTH`    | `1000`                                                           | Reject `search_docs` queries longer than this                        |

**Security**

| Variable                | Default     | Description                                                                            |
|-------------------------|-------------|----------------------------------------------------------------------------------------|
| `RAG_ALLOWED_BASE_DIRS` | *(unset)*   | Colon-separated allow-list for `add_directory`. When unset, any path is accepted        |

**Logging**

| Variable         | Default  | Description                                                                      |
|------------------|----------|----------------------------------------------------------------------------------|
| `LOG_LEVEL`      | `INFO`   | Standard `logging` level name                                                    |
| `RAG_LOG_FORMAT` | `json`   | `json` (one object per line, for aggregators) or `text` (human-readable console) |

**Observability**

| Variable          | Default | Description                                                                     |
|-------------------|---------|---------------------------------------------------------------------------------|
| `RAG_HEALTH_PORT` | `0`     | TCP port for the `/healthz` + `/ready` sidecar. `0` disables it entirely        |

### Optional: cross-encoder reranker

Enable a cross-encoder reranker to re-score the top RRF candidates before returning them:

```sh
RAG_RERANKER_ENABLED=true uv run rag-server
```

The first search after enabling downloads `ms-marco-MiniLM-L-12-v2` (~140 MB) and loads it in the embedder's thread pool. Subsequent queries add ~50–200 ms of cross-encoder inference each but noticeably improve precision on ambiguous or short queries. The `score` field in results becomes the cross-encoder logit (0–1 range on typical inputs) instead of the RRF fusion score. Ordering semantics are the same, magnitudes are not comparable across modes.

### Offline mode

`sentence-transformers` validates the cached model against the HuggingFace Hub on every load, which means the server still makes a network call on startup even though the model files are local. Set `HF_HUB_OFFLINE=1` to skip that call and load straight from the cache:

```sh
HF_HUB_OFFLINE=1 uv run rag-server
```

The Docker image sets `HF_HUB_OFFLINE=1` by default. Both the embedder and the reranker are pre-baked into the image, so no override is needed.

### Choosing an embedding model

The defaults use models that are not affiliated with any government-linked research institution. If your organization has a policy against models originating from Chinese institutions, the defaults are the right choice and no extra configuration is needed.

If you prefer the BAAI/BGE models (from the Beijing Academy of Artificial Intelligence), you can switch by setting all three variables together. The query prefix is part of the BGE embedding convention and must be changed along with the model name. Mixing them produces poor results:

```sh
RAG_EMBEDDER_MODEL="BAAI/bge-base-en-v1.5" \
RAG_QUERY_PREFIX="Represent this sentence for searching relevant passages: " \
RAG_RERANKER_MODEL="BAAI/bge-reranker-base" \
uv run rag-server
```

Or in `config.toml`:

```toml
[retrieval]
embedder_model = "BAAI/bge-base-en-v1.5"
query_prefix = "Represent this sentence for searching relevant passages: "
reranker_model = "BAAI/bge-reranker-base"
```

Changing `embedder_model` triggers a full reindex on the next startup. So does upgrading to a parser version that produces different chunk text. The metadata DB tracks both the embedder name and a `parser_version` key, and a mismatch on either drops the docs collection and re-embeds every source file. The BGE embedder downloads ~440 MB on first use. The BGE reranker adds ~280 MB if enabled.

### Dependency origin

The default dependency closure contains no packages or models from Chinese institutions. This matters because some companies have policies against shipping code or weights from PRC-affiliated entities, and Docling's `[standard]` extra would otherwise pull in [`rapidocr`](https://github.com/RapidAI/RapidOCR), a Chinese-origin OCR engine whose PP-OCRv4 weights are hosted on [modelscope.cn](https://www.modelscope.cn) (Alibaba's model hub).

`rapidocr` is excluded via `[tool.uv]` `override-dependencies` in `pyproject.toml`:

```toml
[tool.uv]
override-dependencies = [
    "rapidocr ; sys_platform == 'never'",
]
```

The marker `sys_platform == 'never'` is unsatisfiable, so uv resolves `rapidocr` as not-installed on any platform while still letting `docling-slim[standard]` install everything else. Because nothing else in the closure pulled in OpenCV transitively after dropping `rapidocr` (and `docling-ibm-models` imports `cv2` without declaring it), `opencv-python-headless` is added explicitly.

**OCR for scanned PDFs.** Tesseract (HP/Google, Apache-2) is installed by default with English, Portuguese, and Spanish language data. Docling auto-detects `tesserocr` at pipeline init and uses it on pages with no extractable text. Digital PDFs are unaffected. To add another language, append the corresponding `tesseract-ocr-<lang>` apt package to the `Dockerfile` (e.g. `tesseract-ocr-fra`, `tesseract-ocr-deu`) and rebuild. Tesseract reads `.traineddata` files from the system `tessdata` directory, so there is no model download at runtime.

Origin map for what does ship:

| Origin | Packages and models |
|---|---|
| Anthropic (US) | `mcp[cli]` |
| Chroma (US) | `chromadb` |
| IBM Research (US) | `docling-slim`, `docling-core`, `docling-ibm-models`, `docling-parse`, `docling-project/docling-layout-heron`, `docling-project/docling-models` (TableFormer) |
| Meta (US) | `torch`, `torchvision` |
| Microsoft (US) | `sentence-transformers/all-mpnet-base-v2`, `cross-encoder/ms-marco-MiniLM-L-12-v2`, `onnxruntime` |
| Google (US) | `pypdfium2` (wraps PDFium) |
| HuggingFace (FR/US) | `transformers`, `huggingface-hub` |
| UKP Lab Darmstadt (DE) | `sentence-transformers` |
| OpenCV (open-source, EU/global) | `opencv-python-headless` |
| HP / Google (Apache-2) | `tesseract-ocr` (system package), `tesserocr` (Python bindings), language data files for English / Portuguese / Spanish |
| Open-source / NumFOCUS / community | `numpy`, `scipy`, `scikit-learn`, `pydantic`, `lxml`, `structlog`, `watchdog`, `marko`, `pylatexenc`, `rank-bm25`, others |

PS: If you opt back into the BAAI/BGE models above, you reintroduce Chinese-origin model weights at the *model* layer. The package closure is unaffected.

### Logging and shutdown

Output is structured JSON by default: one object per line with `timestamp`, `level`, `logger`, `event`, and per-request fields like `trace_id` and `tool`. Third-party libraries (Chroma, watchdog, sentence-transformers) route through the same formatter, so everything lands in one stream. For local development, `RAG_LOG_FORMAT=text` switches to a coloured console renderer.

The server handles `SIGTERM` the same way it handles `SIGINT` (Ctrl-C): stop the file watcher, close the SQLite connection cleanly, exit 0. Container runtimes (`docker stop`, Kubernetes) get a graceful shutdown instead of abrupt termination.

### Health and readiness endpoints

Set `RAG_HEALTH_PORT` to expose a sidecar HTTP server for probes:

```sh
RAG_HEALTH_PORT=8766 uv run rag-server
```

The sidecar serves two endpoints:

- **`GET /healthz`**: liveness. Always returns `200 {"status":"ok"}` while the server process is up and the HTTP thread can respond. Meant for crash/deadlock detection.
- **`GET /ready`**: readiness. Returns `200 {"status":"ready", "store_reachable": true, "watcher_alive": true}` when every predicate passes, `503 {"status":"not ready", ...}` with failure details otherwise. Checks that the metadata DB responds to a basic query and the watcher thread is alive.

The sidecar uses stdlib `http.server` in a daemon thread, independent of the MCP transport. It works with `stdio` too. The Docker image sets `RAG_HEALTH_PORT=8766` by default and ships a `HEALTHCHECK` that hits `/healthz`.

### Security

For deployments where the MCP server is reachable by clients you don't fully trust (team/shared SSE endpoint), set `RAG_ALLOWED_BASE_DIRS` to an allow-list:

```sh
RAG_ALLOWED_BASE_DIRS=/data/shared:/home/team/docs uv run rag-server
```

The `add_directory` tool rejects any path that is not at or under one of those bases. When unset, any resolvable path is accepted (single-user default). `search_docs` also caps query length at `RAG_MAX_QUERY_LENGTH` characters and the indexer skips files over `RAG_MAX_FILE_SIZE_BYTES`.

## Run locally

```sh
uv run rag-server

# or

TRANSPORT=sse uv run rag-server

# or

TRANSPORT=streamable-http uv run rag-server
```

The server performs an incremental index on startup (only changed files) and then runs as an MCP server process. Default transport is stdio. Data is persisted under `.rag-data/`.

## Run with Docker or Podman

Build the image once:

```sh
docker build -t rag-server .

# or

podman build -t rag-server .
```

The image:
- Runs as a non-root user (`uid=1000`).
- Pre-bakes the embedder, the reranker, and the Docling parser models so the container runs fully offline. No HuggingFace fetch is needed at any point during normal operation.
- Ships a `HEALTHCHECK` that hits `:8766/healthz` on the sidecar health server (`RAG_HEALTH_PORT=8766`).
- Handles `SIGTERM` for clean shutdowns when an orchestrator stops it.

Three volumes matter:

- **`/data`**: Chroma DB, `metadata.db` (SQLite), and `runtime_state.json`. Persist this or lose the index on every restart.
- **`/models`**: HuggingFace cache (`HF_HOME`). The base image already contains the embedder, the reranker, and the Docling parser models. The default compose setup leaves this as an anonymous volume populated from the image so a `--build` always picks up the freshly baked weights. Bind-mounting a host path here is supported but masks the baked cache, so the host directory must contain compatible model files.
- **`/sources/...`**: bind mounts for the documents you want indexed. Mount paths must match whatever you pass in `RAG_SOURCE_DIRS`.

### Persistent SSE server (recommended for containers)

```sh
docker run -d --name rag-server \
  -p 8765:8765 \
  -e TRANSPORT=sse \
  -e RAG_SOURCE_DIRS=/sources/notes:/sources/work \
  -v rag-data:/data \
  -v rag-models:/models \
  -v "$HOME/Documents/notes:/sources/notes:ro" \
  -v "$HOME/Documents/work:/sources/work:ro" \
  rag-server
```

Then register it with Claude Code:

```sh
claude mcp add --transport sse rag http://localhost:8765/sse
```

### Stdio container (launched per Claude Code session)

```sh
claude mcp add rag -s user -- docker run --rm -i \
  -e TRANSPORT=stdio \
  -e RAG_SOURCE_DIRS=/sources/notes \
  -v rag-data:/data \
  -v rag-models:/models \
  -v "$HOME/Documents/notes:/sources/notes:ro" \
  rag-server
```

**Podman note**: on rootless Podman, add `--network=host` if you also run with SSE and need the host to reach port 8765 without port mapping.

**File watching note**: `RAG_WATCHER=poll` is set by default in the image because Docker Desktop on macOS and Windows does not forward inotify/FSEvents through its VM. On native Linux you can override with `-e RAG_WATCHER=native` for lower overhead.

## Register with Claude Desktop

### Locally (stdio via uv)

Add to `claude_desktop_config.json` (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "rag-server": {
      "command": "uv",
      "args": [
        "run",
        "--project", "/absolute/path/to/rag-server",
        "rag-server"
      ]
    }
  }
}
```

### Docker (stdio)

```json
{
  "mcpServers": {
    "rag-server": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i",
        "-e", "TRANSPORT=stdio",
        "-e", "RAG_SOURCE_DIRS=/sources/notes",
        "-v", "rag-data:/data",
        "-v", "rag-models:/models",
        "-v", "/Users/you/Documents/notes:/sources/notes:ro",
        "rag-server"
      ]
    }
  }
}
```

## Register with Claude Code

### Locally

```sh
claude mcp add rag -s user -- uv --directory /absolute/path/to/rag-server run rag-server
```

### Remote SSE

```sh
claude mcp add --transport sse rag http://<host>:8765/sse
```

## Test

```sh
uv run pytest -v tests --cov
```

## Evaluate retrieval quality

The `rag-eval` CLI runs a labeled query set against your indexed corpus and reports recall@k and mean reciprocal rank. Use it to compare config changes (chunk size, embedder swap, reranker on/off) on a representative dataset rather than eyeballing search output.

The dataset is a JSON array of `{query, relevant}` pairs:

```json
[
  {"query": "how do I add an item to a python list", "relevant": ["corpus/python_basics.md"]},
  {"query": "how to merge two git branches", "relevant": ["corpus/git_basics.md"]}
]
```

Relative `relevant` paths resolve against the dataset file's directory.

A small sample lives under `eval/`. To smoke-test the harness against it:

```sh
RAG_SOURCE_DIRS=$PWD/eval/corpus uv run rag-server &
sleep 5  # let it index, then stop with Ctrl-C
uv run rag-eval --dataset eval/dataset.json --per-query
```

For an A/B comparison, point each run at its own indexed data dir:

```sh
RAG_DATA_DIR=/tmp/rag-a RAG_RERANKER_ENABLED=false uv run rag-eval --dataset eval/dataset.json > a.json
RAG_DATA_DIR=/tmp/rag-b RAG_RERANKER_ENABLED=true  uv run rag-eval --dataset eval/dataset.json > b.json
diff <(jq . a.json) <(jq . b.json)
```

Each data dir must already be indexed against its respective config.

## Lint and type-check

```sh
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run bandit -r rag
```

## License

This project is licensed under the [MIT License](LICENSE).
