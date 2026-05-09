"""
MCP tool definitions for the RAG server.

Defines the six tools the server exposes (search_docs, reindex,
list_indexed_files, add_directory, remove_directory, get_status) and
registers them against a FastMCP instance via register_tools.

Extracted from server.py so the tools have an explicit dependency
surface (ToolDeps) and can be unit-tested without standing up the full
server lifecycle. Tests pass a MagicMock for the FastMCP instance and
invoke the returned callables directly.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any
import uuid

from rag.logging_config import bound_context
from rag.runtime_state import now_iso, save_directories

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    import mcp.server.fastmcp as _fastmcp

    from rag.config import RagConfig
    from rag.indexer import Indexer
    from rag.store import VectorStore
    from rag.watcher import DirectoryWatcher

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _resolve_path(value: str) -> Path:
    """Expand ~ and resolve symlinks to return an absolute Path."""

    return Path(value).expanduser().resolve()


def _is_path_allowed(path: Path, allowed: tuple[Path, ...]) -> bool:
    """
    Return True if path equals or is a descendant of any allowed base.

    An empty allowed tuple means no restriction. Any path passes. A
    populated tuple enforces containment. Both path and bases are
    expected to be already resolved.
    """

    if not allowed:
        return True
    return any(path == base or path.is_relative_to(base) for base in allowed)


def _new_trace_id() -> str:
    """Return a short trace id suitable for per-request log context."""

    return uuid.uuid4().hex[:8]


@dataclass
class ToolDeps:
    """
    Shared dependencies threaded through every MCP tool body.

    Attributes:
        cfg: Loaded RagConfig. Tools read max_query_length, top_k, and
             allowed_base_dirs.
        store: VectorStore for search and indexed-source queries.
        indexer: Indexer for run / run_for_dir / add_source_dir /
                 remove_source_dir.
        watcher: DirectoryWatcher for runtime watch / unwatch.
        dir_state: Mutable list of {'path', 'last_indexed'} dicts. Mutated
                   in place by add_directory / remove_directory / reindex.
        runtime_state_path: Path to runtime_state.json for persistence.
    """

    cfg: RagConfig
    store: VectorStore
    indexer: Indexer
    watcher: DirectoryWatcher
    dir_state: list[dict[str, Any]]
    runtime_state_path: Path


def register_tools(
    mcp: _fastmcp.FastMCP,
    deps: ToolDeps
) -> dict[str, Callable[..., Awaitable[Any]]]:
    """
    Register every MCP tool against mcp and return a name to callable map.

    The returned dict lets tests invoke tool bodies directly with a
    MagicMock for mcp, bypassing FastMCP routing.

    Args:
        mcp: The FastMCP instance the tools register against.
        deps: Shared dependencies.

    Returns:
        Mapping of tool name to the bound async callable.
    """

    cfg = deps.cfg
    store = deps.store
    indexer = deps.indexer
    watcher = deps.watcher
    dir_state = deps.dir_state
    runtime_state_path = deps.runtime_state_path

    @mcp.tool()
    async def search_docs(query: str) -> list[dict[str, Any]] | dict[str, str]:
        """
        Perform hybrid semantic and keyword search over all indexed documents.

        Uses dense embeddings for semantic similarity combined with BM25
        keyword matching, fused via Reciprocal Rank Fusion. Returns full
        section text rather than individual chunk excerpts.

        Args:
            query: The search query in natural language. Length is capped
                   by max_query_length. Longer queries are rejected with
                   an error dict.

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

        Only files whose SHA-256 hash has changed since the last index
        run are re-processed.

        Returns:
            A summary dict with 'files_scanned', 'files_updated',
            'files_failed', 'files_pruned', and 'failed_paths' keys.
        """

        with bound_context(trace_id=_new_trace_id(), tool="reindex"):
            summary = await asyncio.to_thread(indexer.run)
            ts = now_iso()
            for entry in dir_state:
                entry["last_indexed"] = ts
            save_directories(runtime_state_path, dir_state)
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

        The directory is persisted to runtime_state.json and watched for
        future file changes. Only new or changed files are processed
        (incremental).

        Args:
            path: Absolute or home-relative path to the directory.

        Returns:
            A dict with 'status', 'path', 'files_indexed', and 'errors'
            keys, or an 'error' key if the path is invalid.
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
            save_directories(runtime_state_path, dir_state)

            summary = await asyncio.to_thread(indexer.run_for_dir, resolved)
            entry["last_indexed"] = now_iso()
            save_directories(runtime_state_path, dir_state)

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
            A dict with 'status' and 'path' keys, or an 'error' key if
            not found.
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
            save_directories(runtime_state_path, dir_state)

            return {"status": "removed", "path": path}

    @mcp.tool()
    async def get_status() -> dict[str, Any]:
        """
        Return configured directories, last-indexed timestamps, and total file count.

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

    return {
        "search_docs": search_docs,
        "reindex": reindex,
        "list_indexed_files": list_indexed_files,
        "add_directory": add_directory,
        "remove_directory": remove_directory,
        "get_status": get_status,
    }
