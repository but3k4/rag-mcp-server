"""
RAG MCP server entry point.

Exposes six MCP tools to Claude Code and Claude Desktop:
    - search_docs: hybrid semantic + BM25 search over indexed documents
    - reindex: trigger an incremental re-index on demand
    - list_indexed_files: list all currently indexed file paths
    - add_directory: add a new directory at runtime without restarting
    - remove_directory: remove a directory and its index entries at runtime
    - get_status: show configured directories, timestamps, and file count

Configuration precedence: environment variables override config.toml, which
seeds first-run state. After first run, directory state is persisted in
.rag-data/runtime_state.json so runtime add/remove survive restarts.

Environment Variables:
    TRANSPORT:        stdio (default), sse, or streamable-http.
    PORT:             TCP port when TRANSPORT is sse/streamable-http. Default
                      8765.
    RAG_DATA_DIR:     Override the data directory (default:
                      <project>/.rag-data).
    RAG_CONFIG:       Override the config.toml path (default:
                      <project>/config.toml).
    RAG_SOURCE_DIRS:  Colon-separated source directories. Overrides config.toml
                      on first run.
    RAG_CHUNK_SIZE:   Override chunk_size from config.toml.
    RAG_CHUNK_OVERLAP: Override chunk_overlap from config.toml.
    RAG_WATCHER:      native (default) or poll. Use poll for bind-mounted
                      directories on Docker Desktop (macOS/Windows).
    LOG_LEVEL:        Logging level name. Default INFO.
    RAG_LOG_FORMAT:   json (default) or text. JSON is structured and one
                      object per line. Intended for log aggregators. Text is
                      human-readable for local development.
    RAG_HEALTH_PORT:  TCP port for the /healthz and /ready sidecar. 0
                      (default) disables it.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import signal
import sys
from typing import Any
import uuid

import mcp.server.fastmcp as _fastmcp

from rag.config import ConfigError, RagConfig, load_config
from rag.health import HealthServer
from rag.indexer import Indexer
from rag.logging_config import bound_context, configure_logging
from rag.store import VectorStore
from rag.watcher import DirectoryWatcher

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_path(value: str) -> Path:
    """Expand ~ and resolve symlinks to return an absolute Path."""

    return Path(value).expanduser().resolve()


def _new_trace_id() -> str:
    """Return a short trace id suitable for per-request log context."""

    return uuid.uuid4().hex[:8]


def _install_shutdown_handlers() -> None:
    """
    Install SIGTERM handler so SIGTERM and SIGINT share the shutdown path.

    SIGINT already raises KeyboardInterrupt by Python default. Mirroring
    that for SIGTERM lets the existing try/finally in main() run cleanup
    uniformly whether the operator pressed Ctrl-C or a container runtime
    sent SIGTERM.
    """

    def _raise_keyboard_interrupt(
        signum: int, _frame: object
    ) -> None:  # pragma: no cover
        signame = signal.Signals(signum).name
        logger.info("Received %s, initiating shutdown", signame)
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)


def _is_path_allowed(path: Path, allowed: tuple[Path, ...]) -> bool:
    """
    Return True if path is within one of the allowed base dirs.

    An empty allowed tuple means no restriction. Any path passes. A
    populated tuple enforces containment: the path must equal or be a
    descendant of at least one base. Both path and bases are expected
    to be already resolved.
    """

    if not allowed:
        return True
    return any(path == base or path.is_relative_to(base) for base in allowed)


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""

    return datetime.now(tz=timezone.utc).isoformat()  # noqa: UP017


def _load_runtime_state(
    runtime_state_path: Path, seed_dirs: tuple[Path, ...]
) -> list[dict[str, Any]]:
    """Load directory state, seeding from the configured source dirs on first run."""

    if runtime_state_path.is_file():
        try:
            data = json.loads(runtime_state_path.read_text(encoding="utf-8"))
            return list(data.get("directories", []))
        except (json.JSONDecodeError, OSError):
            logger.warning(
                "Could not read %s. Seeding from config/env", runtime_state_path
            )
    return [{"path": str(p), "last_indexed": None} for p in seed_dirs]


def _save_runtime_state(
    runtime_state_path: Path, directories: list[dict[str, Any]]
) -> None:
    """Persist directory state to runtime_state.json."""

    runtime_state_path.write_text(
        json.dumps({"directories": directories}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    """
    Run the RAG MCP server.

    Reads TRANSPORT from the environment to select stdio (default), sse, or
    streamable-http. Binds to 0.0.0.0 on PORT (default 8765) for HTTP
    transports so the server is reachable from container hosts.
    """

    try:
        cfg = load_config(PROJECT_ROOT)
    except ConfigError as exc:
        sys.stderr.write(f"Configuration error: {exc}\n")
        sys.exit(1)

    configure_logging(level=cfg.log_level, log_format=cfg.log_format)
    _install_shutdown_handlers()

    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    runtime_state_path = cfg.data_dir / "runtime_state.json"

    with VectorStore(
        cfg.data_dir,
        model_name=cfg.embedder_model,
        query_prefix=cfg.query_prefix,
        fetch_n_multiplier=cfg.fetch_n_multiplier,
        rrf_k=cfg.rrf_k,
        reranker_enabled=cfg.reranker_enabled,
        reranker_model=cfg.reranker_model,
        reranker_pool_size=cfg.reranker_pool_size,
    ) as store:
        _run_mcp_server(cfg, store, runtime_state_path)


def _run_mcp_server(
    cfg: RagConfig,
    store: VectorStore,
    runtime_state_path: Path,
) -> None:
    """
    Wire up the indexer, watcher, and MCP tools. Run until interrupted.

    Extracted from main() so the VectorStore context manager can wrap the
    whole lifecycle. The store closes on exit whether or not this body
    returns normally.
    """

    dir_state: list[dict[str, Any]] = _load_runtime_state(
        runtime_state_path, cfg.source_dirs
    )
    source_dirs = [_resolve_path(d["path"]) for d in dir_state]

    indexer = Indexer(
        store=store,
        source_dirs=source_dirs,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        max_file_size_bytes=cfg.max_file_size_bytes,
    )

    logger.info("Running startup index pass...")
    startup_summary = indexer.run()
    logger.info(
        "Startup index complete: scanned=%d updated=%d failed=%d pruned=%d",
        startup_summary.files_scanned,
        startup_summary.files_updated,
        startup_summary.files_failed,
        startup_summary.files_pruned,
    )

    startup_ts = _now_iso()
    for d in dir_state:
        d["last_indexed"] = startup_ts
    _save_runtime_state(runtime_state_path, dir_state)

    watcher = DirectoryWatcher(
        indexer,
        use_polling=cfg.use_polling,
        debounce_seconds=cfg.watcher_debounce_seconds,
    )

    mcp = _fastmcp.FastMCP("rag-server", host="0.0.0.0", port=cfg.port)  # nosec B104

    @mcp.tool()
    async def search_docs(query: str) -> list[dict[str, Any]] | dict[str, str]:
        """
        Perform hybrid semantic and keyword search over all indexed documents.

        Uses BGE embeddings for semantic similarity combined with BM25 keyword
        matching, fused via Reciprocal Rank Fusion. Returns full section text
        rather than individual chunk excerpts.

        Args:
            query: The search query in natural language. Length is capped by
                   max_query_length. Longer queries are rejected with an error
                   dict.

        Returns:
            A list of result dicts on success (keys: filename, excerpt,
            section, score), or a dict with an 'error' key on failure.
        """

        with bound_context(trace_id=_new_trace_id(), tool="search_docs"):
            if len(query) > cfg.max_query_length:
                return {
                    "error": (
                        f"Query length {len(query)} exceeds "
                        f"max_query_length {cfg.max_query_length}"
                    )
                }

            results = await asyncio.to_thread(store.search, query, top_k=cfg.top_k)

            return [
                {
                    "filename": r.filename,
                    "excerpt": r.excerpt,
                    "section": r.section,
                    "score": r.score,
                }
                for r in results
            ]

    @mcp.tool()
    async def reindex() -> dict[str, Any]:
        """
        Trigger an incremental re-index of all configured source directories.

        Only files whose SHA-256 hash has changed since the last index run are
        re-processed.

        Returns:
            A summary dict with 'files_scanned', 'files_updated',
            'files_failed', 'files_pruned', and 'failed_paths' keys.
        """

        with bound_context(trace_id=_new_trace_id(), tool="reindex"):
            summary = await asyncio.to_thread(indexer.run)
            ts = _now_iso()

            for entry in dir_state:
                entry["last_indexed"] = ts

            _save_runtime_state(runtime_state_path, dir_state)

            return {
                "files_scanned": summary.files_scanned,
                "files_updated": summary.files_updated,
                "files_failed": summary.files_failed,
                "files_pruned": summary.files_pruned,
                "failed_paths": summary.failed_paths,
            }

    @mcp.tool()
    async def list_indexed_files() -> list[str]:
        """
        List all currently indexed file paths.

        Returns:
            Sorted list of absolute file path strings.
        """

        with bound_context(trace_id=_new_trace_id(), tool="list_indexed_files"):
            indexed = await asyncio.to_thread(store.get_indexed_sources)
            return sorted(indexed.keys())

    @mcp.tool()
    async def add_directory(path: str) -> dict[str, Any]:
        """
        Add a new directory to the configuration and index its contents.

        The directory is persisted to runtime_state.json and watched for future
        file changes. Only new or changed files are processed (incremental).

        Args:
            path: Absolute or home-relative path to the directory.

        Returns:
            A dict with 'status', 'path', 'files_indexed', and 'errors' keys,
            or an 'error' key if the path is invalid.
        """

        with bound_context(trace_id=_new_trace_id(), tool="add_directory"):
            resolved = _resolve_path(path)
            if not _is_path_allowed(resolved, cfg.allowed_base_dirs):
                return {
                    "error": (f"Path is not under any allowed base directory: {path}")
                }
            if not resolved.exists():
                return {"error": f"Path does not exist: {path}"}

            if not resolved.is_dir():
                return {"error": f"Not a directory: {path}"}

            existing = {_resolve_path(d["path"]) for d in dir_state}
            if resolved in existing:
                return {"status": "already configured", "path": path}

            entry: dict[str, Any] = {"path": path, "last_indexed": None}
            dir_state.append(entry)
            indexer.add_source_dir(resolved)
            watcher.watch([resolved])
            _save_runtime_state(runtime_state_path, dir_state)

            summary = await asyncio.to_thread(indexer.run_for_dir, resolved)
            entry["last_indexed"] = _now_iso()
            _save_runtime_state(runtime_state_path, dir_state)

            return {
                "status": "added",
                "path": path,
                "files_indexed": summary.files_updated,
                "errors": summary.failed_paths,
            }

    @mcp.tool()
    async def remove_directory(path: str) -> dict[str, Any]:
        """
        Remove a directory from the configuration and delete its index entries.

        Args:
            path: Directory path that was previously configured.

        Returns:
            A dict with 'status' and 'path' keys, or an 'error' key if not found.
        """

        with bound_context(trace_id=_new_trace_id(), tool="remove_directory"):
            resolved = _resolve_path(path)
            new_state = [d for d in dir_state if _resolve_path(d["path"]) != resolved]

            if len(new_state) == len(dir_state):
                return {"error": f"Directory not configured: {path}"}

            dir_state.clear()
            dir_state.extend(new_state)
            indexer.remove_source_dir(resolved)
            watcher.unwatch(resolved)
            _save_runtime_state(runtime_state_path, dir_state)

            return {"status": "removed", "path": path}

    @mcp.tool()
    async def get_status() -> dict[str, Any]:
        """
        Return configured directories, their last-indexed timestamps, and total file count.

        Returns:
            A dict with 'directories' (list of path + last_indexed) and
            'total_indexed_files' (int).
        """

        with bound_context(trace_id=_new_trace_id(), tool="get_status"):
            indexed = await asyncio.to_thread(store.get_indexed_sources)
            return {
                "directories": [
                    {"path": d["path"], "last_indexed": d["last_indexed"]}
                    for d in dir_state
                ],
                "total_indexed_files": len(indexed),
            }

    watcher.watch(source_dirs)
    watcher.start()

    health: HealthServer | None = None
    if cfg.health_port > 0:
        health = HealthServer(cfg.health_port, store, watcher)
        health.start()

    try:
        mcp.run(transport=cfg.transport)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop in reverse dependency order: health probe first (so it stops
        # answering "ready"), then the watcher, then the enclosing with
        # VectorStore closes the store. Order matters because a
        # watcher-dispatched indexer call could otherwise race the close.
        logger.info("Shutting down RAG server")
        if health is not None:
            health.stop()

        watcher.stop()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
