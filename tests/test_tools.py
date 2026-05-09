"""
Tests for MCP tool bodies and helpers in rag.tools.

The tools are unit-tested by passing a MagicMock for FastMCP (with a
transparent .tool() decorator) and per-component MagicMock fixtures for
cfg, store, indexer, and watcher. register_tools returns the bound
callables so each tool can be invoked directly without going through
FastMCP routing.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

from rag.indexer import IndexSummary
from rag.store import SearchResult
from rag.tools import ToolDeps, _is_path_allowed, _resolve_path, register_tools

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path


@pytest.fixture
def mcp_mock() -> MagicMock:
    """A FastMCP-shaped mock whose .tool() returns a transparent decorator."""

    m = MagicMock()
    m.tool.return_value = lambda fn: fn
    return m


@pytest.fixture
def cfg() -> MagicMock:
    """Mock RagConfig with sensible defaults."""

    m = MagicMock()
    m.max_query_length = 1000
    m.top_k = 5
    m.allowed_base_dirs = ()
    return m


@pytest.fixture
def store() -> MagicMock:
    """Mock VectorStore with no indexed sources."""

    m = MagicMock()
    m.get_indexed_sources.return_value = {}
    return m


@pytest.fixture
def indexer() -> MagicMock:
    """Mock Indexer with empty IndexSummary returns."""

    m = MagicMock()
    m.run.return_value = IndexSummary()
    m.run_for_dir.return_value = IndexSummary()
    return m


@pytest.fixture
def watcher() -> MagicMock:
    """Mock DirectoryWatcher."""

    return MagicMock()


@pytest.fixture
def dir_state() -> list[dict[str, Any]]:
    """Empty initial directory state."""

    return []


@pytest.fixture
def runtime_state_path(tmp_path: Path) -> Path:
    """Path for runtime_state.json under the test temp directory."""

    return tmp_path / "runtime_state.json"


@pytest.fixture
def tools(
    mcp_mock: MagicMock,
    cfg: MagicMock,
    store: MagicMock,
    indexer: MagicMock,
    watcher: MagicMock,
    dir_state: list[dict[str, Any]],
    runtime_state_path: Path,
) -> dict[str, Callable[..., Awaitable[Any]]]:
    """Register the tools against a mocked FastMCP and return the callable map."""

    deps = ToolDeps(
        cfg=cfg,
        store=store,
        indexer=indexer,
        watcher=watcher,
        dir_state=dir_state,
        runtime_state_path=runtime_state_path,
    )
    return register_tools(mcp_mock, deps)


class TestIsPathAllowed:
    """Tests for the allow-list check used by add_directory."""

    def test_empty_allow_list_permits_everything(self, tmp_path: Path) -> None:
        """An empty tuple means no restriction. Any path is allowed."""

        assert _is_path_allowed(tmp_path / "anywhere", ()) is True

    def test_path_under_base_is_allowed(self, tmp_path: Path) -> None:
        """A descendant of an allowed base is permitted."""

        base = tmp_path / "allowed"
        base.mkdir()
        sub = base / "nested" / "deep"
        sub.mkdir(parents=True)
        assert _is_path_allowed(sub.resolve(), (base.resolve(),)) is True

    def test_base_itself_is_allowed(self, tmp_path: Path) -> None:
        """The base directory itself satisfies the check."""

        base = tmp_path / "allowed"
        base.mkdir()
        assert _is_path_allowed(base.resolve(), (base.resolve(),)) is True

    def test_sibling_is_rejected(self, tmp_path: Path) -> None:
        """A sibling directory outside every base is rejected."""

        allowed = tmp_path / "allowed"
        allowed.mkdir()
        sibling = tmp_path / "other"
        sibling.mkdir()
        assert _is_path_allowed(sibling.resolve(), (allowed.resolve(),)) is False

    def test_prefix_trap_is_rejected(self, tmp_path: Path) -> None:
        """A path with a matching name prefix but different directory is rejected."""

        base = tmp_path / "foo"
        base.mkdir()
        trap = tmp_path / "foobar"
        trap.mkdir()
        assert _is_path_allowed(trap.resolve(), (base.resolve(),)) is False

    def test_multiple_bases_any_match_allows(self, tmp_path: Path) -> None:
        """Membership in any one of several allowed bases permits the path."""

        base_a = tmp_path / "a"
        base_b = tmp_path / "b"
        base_a.mkdir()
        base_b.mkdir()
        under_b = base_b / "nested"
        under_b.mkdir()
        bases = (base_a.resolve(), base_b.resolve())
        assert _is_path_allowed(under_b.resolve(), bases) is True


class TestResolvePath:
    """Tests for the path normalisation helper."""

    def test_expands_user_home(self) -> None:
        """A path starting with ~ expands to the user's home directory."""

        from pathlib import Path  # noqa: PLC0415

        resolved = _resolve_path("~")
        assert resolved == Path.home().resolve()


class TestRegisterTools:
    """Tests for register_tools."""

    def test_registers_six_tools(
        self,
        mcp_mock: MagicMock,
        cfg: MagicMock,
        store: MagicMock,
        indexer: MagicMock,
        watcher: MagicMock,
        dir_state: list[dict[str, Any]],
        runtime_state_path: Path,
    ) -> None:
        """Six tools are registered against the FastMCP instance."""

        deps = ToolDeps(
            cfg=cfg,
            store=store,
            indexer=indexer,
            watcher=watcher,
            dir_state=dir_state,
            runtime_state_path=runtime_state_path,
        )
        result = register_tools(mcp_mock, deps)
        expected = {
            "search_docs",
            "reindex",
            "list_indexed_files",
            "add_directory",
            "remove_directory",
            "get_status",
        }
        assert set(result) == expected
        # mcp.tool() is invoked once per registered tool.
        assert mcp_mock.tool.call_count == len(expected)


class TestSearchDocs:
    """Tests for the search_docs tool."""

    async def test_long_query_returns_error(
        self,
        cfg: MagicMock,
        tools: dict[str, Callable[..., Awaitable[Any]]],
    ) -> None:
        """A query above max_query_length is rejected with an error dict."""

        cfg.max_query_length = 10
        result = await tools["search_docs"]("x" * 100)
        assert isinstance(result, dict)
        assert "error" in result

    async def test_returns_result_dicts(
        self,
        store: MagicMock,
        tools: dict[str, Callable[..., Awaitable[Any]]],
    ) -> None:
        """Each SearchResult is mapped to the documented dict shape."""

        store.search.return_value = [
            SearchResult(filename="/a.txt", excerpt="hello", section="intro", score=0.5)
        ]
        result = await tools["search_docs"]("hello")
        assert isinstance(result, list)
        assert result == [
            {
                "filename": "/a.txt",
                "excerpt": "hello",
                "section": "intro",
                "score": 0.5,
            }
        ]

    async def test_passes_top_k_from_cfg(
        self,
        cfg: MagicMock,
        store: MagicMock,
        tools: dict[str, Callable[..., Awaitable[Any]]],
    ) -> None:
        """search_docs forwards cfg.top_k to store.search."""

        cfg.top_k = 7
        store.search.return_value = []
        await tools["search_docs"]("anything")
        store.search.assert_called_once_with("anything", top_k=7)


class TestReindex:
    """Tests for the reindex tool."""

    async def test_returns_summary_fields(
        self,
        indexer: MagicMock,
        tools: dict[str, Callable[..., Awaitable[Any]]],
    ) -> None:
        """The summary returned by indexer.run is mapped into the response dict."""

        indexer.run.return_value = IndexSummary(
            files_scanned=5,
            files_updated=2,
            files_failed=1,
            files_pruned=0,
            failed_paths=["/bad.txt"],
        )
        result = await tools["reindex"]()
        assert result == {
            "files_scanned": 5,
            "files_updated": 2,
            "files_failed": 1,
            "files_pruned": 0,
            "failed_paths": ["/bad.txt"],
        }

    async def test_stamps_timestamp_on_every_directory(
        self,
        dir_state: list[dict[str, Any]],
        tools: dict[str, Callable[..., Awaitable[Any]]],
    ) -> None:
        """Every entry in dir_state has its last_indexed updated to a non-null value."""

        dir_state.extend(
            [
                {"path": "/a", "last_indexed": None},
                {"path": "/b", "last_indexed": "2025-01-01T00:00:00+00:00"},
            ]
        )
        await tools["reindex"]()
        for entry in dir_state:
            assert entry["last_indexed"] is not None

    async def test_persists_state(
        self,
        dir_state: list[dict[str, Any]],
        runtime_state_path: Path,
        tools: dict[str, Callable[..., Awaitable[Any]]],
    ) -> None:
        """The mutated dir_state is written to runtime_state_path."""

        dir_state.append({"path": "/a", "last_indexed": None})
        await tools["reindex"]()
        payload = json.loads(runtime_state_path.read_text(encoding="utf-8"))  # noqa: ASYNC240
        assert payload["directories"][0]["path"] == "/a"
        assert payload["directories"][0]["last_indexed"] is not None


class TestListIndexedFiles:
    """Tests for the list_indexed_files tool."""

    async def test_returns_sorted_paths(
        self,
        store: MagicMock,
        tools: dict[str, Callable[..., Awaitable[Any]]],
    ) -> None:
        """Returned paths are sorted in ascending order."""

        store.get_indexed_sources.return_value = {
            "/c.txt": "h3",
            "/a.txt": "h1",
            "/b.txt": "h2",
        }
        result = await tools["list_indexed_files"]()
        assert result == ["/a.txt", "/b.txt", "/c.txt"]


class TestAddDirectory:
    """Tests for the add_directory tool."""

    async def test_rejects_path_outside_allowlist(
        self,
        cfg: MagicMock,
        tools: dict[str, Callable[..., Awaitable[Any]]],
        tmp_path: Path,
    ) -> None:
        """A path outside cfg.allowed_base_dirs is rejected."""

        allowed = tmp_path / "allowed"
        allowed.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        cfg.allowed_base_dirs = (allowed.resolve(),)
        result = await tools["add_directory"](str(outside))
        assert "error" in result

    async def test_rejects_nonexistent_path(
        self,
        tools: dict[str, Callable[..., Awaitable[Any]]],
        tmp_path: Path,
    ) -> None:
        """A path that does not exist on disk is rejected."""

        result = await tools["add_directory"](str(tmp_path / "missing"))
        assert "error" in result
        assert "does not exist" in result["error"]

    async def test_rejects_file_path(
        self,
        tools: dict[str, Callable[..., Awaitable[Any]]],
        tmp_path: Path,
    ) -> None:
        """A path that points to a file rather than a directory is rejected."""

        f = tmp_path / "file.txt"
        f.write_text("content")
        result = await tools["add_directory"](str(f))
        assert "error" in result
        assert "Not a directory" in result["error"]

    async def test_returns_already_configured_for_duplicate(
        self,
        dir_state: list[dict[str, Any]],
        tools: dict[str, Callable[..., Awaitable[Any]]],
        tmp_path: Path,
    ) -> None:
        """Adding the same directory twice returns the 'already configured' status."""

        d = tmp_path / "data"
        d.mkdir()
        dir_state.append({"path": str(d), "last_indexed": None})
        result = await tools["add_directory"](str(d))
        assert result["status"] == "already configured"

    async def test_adds_new_directory_and_persists(
        self,
        indexer: MagicMock,
        watcher: MagicMock,
        dir_state: list[dict[str, Any]],
        runtime_state_path: Path,
        tools: dict[str, Callable[..., Awaitable[Any]]],
        tmp_path: Path,
    ) -> None:
        """A new directory is appended, registered with indexer + watcher, and persisted."""

        d = tmp_path / "data"
        d.mkdir()
        indexer.run_for_dir.return_value = IndexSummary(
            files_scanned=1, files_updated=1
        )
        result = await tools["add_directory"](str(d))
        assert result["status"] == "added"
        assert result["files_indexed"] == 1
        assert dir_state[-1]["path"] == str(d)
        assert dir_state[-1]["last_indexed"] is not None
        indexer.add_source_dir.assert_called_once()
        watcher.watch.assert_called_once()
        indexer.run_for_dir.assert_called_once()
        # Persisted to disk.
        payload = json.loads(runtime_state_path.read_text(encoding="utf-8"))  # noqa: ASYNC240
        assert payload["directories"][-1]["path"] == str(d)


class TestRemoveDirectory:
    """Tests for the remove_directory tool."""

    async def test_rejects_unconfigured_path(
        self,
        tools: dict[str, Callable[..., Awaitable[Any]]],
        tmp_path: Path,
    ) -> None:
        """Removing a directory that was never added returns an error."""

        result = await tools["remove_directory"](str(tmp_path / "missing"))
        assert "error" in result

    async def test_removes_configured_directory(
        self,
        indexer: MagicMock,
        watcher: MagicMock,
        dir_state: list[dict[str, Any]],
        runtime_state_path: Path,
        tools: dict[str, Callable[..., Awaitable[Any]]],
        tmp_path: Path,
    ) -> None:
        """A configured directory is removed and indexer / watcher are notified."""

        d = tmp_path / "data"
        d.mkdir()
        dir_state.append({"path": str(d), "last_indexed": None})
        result = await tools["remove_directory"](str(d))
        assert result == {"status": "removed", "path": str(d)}
        assert dir_state == []
        indexer.remove_source_dir.assert_called_once()
        watcher.unwatch.assert_called_once()
        # Persisted to disk.
        payload = json.loads(runtime_state_path.read_text(encoding="utf-8"))  # noqa: ASYNC240
        assert payload["directories"] == []


class TestGetStatus:
    """Tests for the get_status tool."""

    async def test_returns_directories_and_total(
        self,
        store: MagicMock,
        dir_state: list[dict[str, Any]],
        tools: dict[str, Callable[..., Awaitable[Any]]],
    ) -> None:
        """get_status returns dir_state entries and the indexed file count."""

        dir_state.extend(
            [
                {"path": "/a", "last_indexed": None},
                {"path": "/b", "last_indexed": "2026-05-06T12:00:00+00:00"},
            ]
        )
        store.get_indexed_sources.return_value = {"/x.txt": "h1", "/y.txt": "h2"}
        result = await tools["get_status"]()
        assert result == {
            "directories": [
                {"path": "/a", "last_indexed": None},
                {"path": "/b", "last_indexed": "2026-05-06T12:00:00+00:00"},
            ],
            "total_indexed_files": 2,
        }
