# RAG Server

A local MCP server that indexes documents from configured directories and exposes semantic search to Claude Code, Claude Desktop, Cursor and VSCode. Works fully offline, no external API calls.

## What it does

Six MCP tools are available once the server is running:

- **search_docs(query)**: hybrid search combining BGE semantic similarity and BM25 keyword matching via Reciprocal Rank Fusion. Returns the top 5 most relevant section excerpts, each with source filename, section title, and score.
- **reindex()**: incremental re-index on demand. Only processes files whose SHA-256 hash has changed, and prunes index entries for files that were renamed or deleted since the last run. Returns a summary of scanned, updated, failed, and pruned files.
- **list_indexed_files()**: lists all currently indexed file paths.
- **add_directory(path)**: register a new directory at runtime and index its contents. Persisted across restarts.
- **remove_directory(path)**: unregister a directory and delete its index entries.
- **get_status()**: show configured directories, last-indexed timestamps, and total file count.

Supported file types: `.txt`, `.md`, `.pdf`, `.docx`, `.pptx`, `.csv`, `.xlsx`, `.xml`

## Architecture

Documents are parsed into section-aware chunks and indexed with a parent-child layout. Small overlapping chunks (children) are embedded with BGE and stored in ChromaDB. Their parent sections (full section text) plus all other non-vector state live in a sidecar SQLite database. The Chroma index is a rebuildable cache over the canonical data in SQLite. A file watcher keeps both in sync with the filesystem. SHA-256 hashes drive incremental reindexing.

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
  |          + BGE embeds  |  |               model name   |
  |                        |  |   file_hashes path -> sha  |
  |   (cosine)             |  |   parents     id, source,  |
  |                        |  |               section,     |
  |   BM25 index (memory)  |  |               text         |
  +------------------------+  +----------------------------+
                              |
                              | search
                              |
     query --> BGE semantic --+
                              |-- Reciprocal Rank Fusion --+
     query --> BM25 keyword --+                            |
                                                           |
                      (optional) cross-encoder reranker <--+
                                                           |
                                              parent lookup (SQLite)
                                                           |
                                                         top-k
```

SQLite is the source of truth. The vector index can be rebuilt from SQLite without re-parsing the source files (via stored parent text), which unblocks embedding-model swaps and future schema migrations.

## Install

```sh
# 1. Clone or copy this project, then enter the directory.
cd rag-server

# 2. Install dependencies from pyproject.toml (creates .venv).
uv sync
```

The first time the server starts it will download the `BAAI/bge-base-en-v1.5` embedding model (~440 MB) from HuggingFace and cache it locally. If you enable the optional reranker, the first query also downloads `BAAI/bge-reranker-base` (~280 MB). After that, everything runs offline.

## Configure

Two ways to configure, in order of precedence:

1. **Environment variables** (override everything below)
2. **`config.toml`** at the project root

Copy the template and edit paths for your machine:

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
embedder_model = "BAAI/bge-base-en-v1.5"
query_prefix = "Represent this sentence for searching relevant passages: "
reranker_enabled = false
reranker_model = "BAAI/bge-reranker-base"
reranker_pool_size = 20
max_query_length = 1000

[security]
# allowed_base_dirs = [ "~/Documents", "/data/shared" ]
```

Invalid values are rejected at startup with a clear `ConfigError` message. The server exits 1 rather than limping along with bad config.

### Environment variables

**Transport and paths**

| Variable         | Default                 | Description                                                       |
|------------------|-------------------------|-------------------------------------------------------------------|
| `TRANSPORT`      | `stdio`                 | `stdio`, `sse`, or `streamable-http`                              |
| `PORT`           | `8765`                  | TCP port for HTTP transports                                      |
| `RAG_DATA_DIR`   | `<project>/.rag-data`   | Where Chroma and `metadata.db` (SQLite) persist                   |
| `RAG_CONFIG`     | `<project>/config.toml` | Alternate config file path                                        |
| `RAG_SOURCE_DIRS`| *(unset)*               | Colon-separated paths. Overrides `[indexing].source_dirs`         |

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
| `RAG_EMBEDDER_MODEL`      | `BAAI/bge-base-en-v1.5`                                          | SentenceTransformer identifier. Changing this triggers a full reindex |
| `RAG_QUERY_PREFIX`        | `"Represent this sentence for searching relevant passages: "`   | Prefix prepended to queries before embedding. Must match the embedder's convention |
| `RAG_RERANKER_ENABLED`    | `false`                                                          | Enable cross-encoder reranking after RRF                             |
| `RAG_RERANKER_MODEL`      | `BAAI/bge-reranker-base`                                         | CrossEncoder identifier when reranking is on                         |
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

The first search after enabling downloads `bge-reranker-base` (~280 MB) and loads it in the embedder's thread pool. Subsequent queries add ~50–200 ms of cross-encoder inference each but noticeably improve precision on ambiguous or short queries. The `score` field in results becomes the cross-encoder logit (0–1 range on typical inputs) instead of the RRF fusion score. Ordering semantics are the same, magnitudes are not comparable across modes.

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
```

The server indexes on startup (incremental. Only changed files) and then stays running as an MCP server process. Default transport is stdio. The index and vector store persist in `.rag-data/` inside the project root.

## Run with Docker or Podman

Build the image once:

```sh
docker build -t rag-server .
# or
podman build -t rag-server .
```

The image:
- Runs as a non-root user (`uid=1000`).
- Pre-bakes the BGE embedder, so the first search doesn't pay the ~440 MB download on cold start. The reranker stays opt-in and downloads on first use when `RAG_RERANKER_ENABLED=true`.
- Ships a `HEALTHCHECK` that hits `/sse`. Meaningful when `TRANSPORT=sse` (the default in the image). Disable or override for `TRANSPORT=stdio`.
- Handles `SIGTERM` for clean shutdowns when an orchestrator stops it.

Three volumes matter:

- **`/data`**: Chroma DB, `metadata.db` (SQLite), and `runtime_state.json`. Persist this or lose the index on every restart.
- **`/models`**: HuggingFace cache (`HF_HOME`). The base image already contains the BGE model. Persisting this volume avoids re-downloading the reranker (if enabled) on every restart.
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

## Lint and type-check

```sh
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run bandit -r rag
```

## License

This project is licensed under the [MIT License](LICENSE).
