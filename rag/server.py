"""
RAG MCP server entry point.

Exposes six MCP tools to Claude Code and Claude Desktop:
    - search_docs: hybrid semantic + BM25 search over indexed documents
    - reindex: trigger an incremental re-index on demand
    - list_indexed_files: list all currently indexed file paths
    - add_directory: add a new directory at runtime without restarting
    - remove_directory: remove a directory and its index entries at runtime
    - get_status: show configured directories, timestamps, and file count

Tool definitions live in rag.tools. This module owns the lifecycle:
config load, structured logging, signal handling, store / indexer / watcher
construction, the optional health sidecar, and shutdown ordering.

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

import logging
from pathlib import Path
import signal
import sys

import mcp.server.fastmcp as _fastmcp

from rag.config import ConfigError, RagConfig, load_config
from rag.health import HealthServer
from rag.indexer import Indexer
from rag.logging_config import configure_logging
from rag.runtime_state import load_directories, now_iso, save_directories
from rag.store import VectorStore
from rag.tools import ToolDeps, register_tools
from rag.watcher import DirectoryWatcher

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


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

    dir_state = load_directories(runtime_state_path, cfg.source_dirs)
    source_dirs = [Path(d["path"]).expanduser().resolve() for d in dir_state]

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

    startup_ts = now_iso()
    for d in dir_state:
        d["last_indexed"] = startup_ts
    save_directories(runtime_state_path, dir_state)

    watcher = DirectoryWatcher(
        indexer,
        use_polling=cfg.use_polling,
        debounce_seconds=cfg.watcher_debounce_seconds,
    )

    mcp = _fastmcp.FastMCP("rag-server", host="0.0.0.0", port=cfg.port)  # nosec B104
    register_tools(
        mcp,
        ToolDeps(
            cfg=cfg,
            store=store,
            indexer=indexer,
            watcher=watcher,
            dir_state=dir_state,
            runtime_state_path=runtime_state_path,
        ),
    )

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
