"""
Unit tests for the lifecycle helpers in rag.server.

Tool bodies live in rag.tools and are tested in tests/test_tools.py.
This module covers the shutdown signal handler plus the lifecycle
wiring in _run_mcp_server and main.
"""

from __future__ import annotations

import signal
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

from rag.config import ConfigError, RagConfig
from rag.server import _install_shutdown_handlers, _run_mcp_server, main

if TYPE_CHECKING:
    from pathlib import Path


def _make_cfg(tmp_path: Path, **overrides: Any) -> RagConfig:  # noqa: ANN401
    """Build a RagConfig with safe defaults for lifecycle tests."""

    defaults: dict[str, Any] = {
        "transport": "stdio",
        "port": 8765,
        "data_dir": tmp_path / "data",
        "config_path": tmp_path / "config.toml",
        "source_dirs": (),
        "chunk_size": 1000,
        "chunk_overlap": 100,
        "use_polling": False,
        "log_level": "INFO",
        "top_k": 5,
        "fetch_n_multiplier": 4,
        "rrf_k": 60,
        "embedder_model": "test-model",
        "query_prefix": "",
        "reranker_enabled": False,
        "reranker_model": "test-reranker",
        "reranker_pool_size": 20,
        "allowed_base_dirs": (),
        "max_file_size_bytes": 50 * 1024 * 1024,
        "max_query_length": 1000,
        "watcher_debounce_seconds": 0.5,
        "log_format": "json",
        "health_port": 0,
    }
    defaults.update(overrides)
    return RagConfig(**defaults)


class TestShutdown:
    """Tests for signal handling used by the graceful-shutdown path."""

    def test_sigterm_handler_raises_keyboard_interrupt(self) -> None:
        """SIGTERM is rewired to raise KeyboardInterrupt, sharing the SIGINT path."""

        try:
            _install_shutdown_handlers()
            handler = signal.getsignal(signal.SIGTERM)
            assert callable(handler)
            with pytest.raises(KeyboardInterrupt):
                handler(signal.SIGTERM, None)
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)


@patch("rag.server.HealthServer")
@patch("rag.server.DirectoryWatcher")
@patch("rag.server._fastmcp.FastMCP")
@patch("rag.server.register_tools")
class TestRunMcpServer:
    """
    Tests for the _run_mcp_server lifecycle wiring.

    Mocks every external boundary (FastMCP, DirectoryWatcher, HealthServer,
    register_tools) so the test exercises the wiring without binding ports
    or spawning watcher threads. The Indexer runs against an empty
    source_dirs tuple talking to a MagicMock store, so the startup index
    pass is a no-op.
    """

    def test_runs_full_lifecycle(
        self,
        mock_register_tools: MagicMock,
        mock_mcp_class: MagicMock,
        mock_watcher_class: MagicMock,
        mock_health_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Happy path: tools register, watcher starts and stops, mcp.run is invoked."""

        cfg = _make_cfg(tmp_path)
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        store = MagicMock()
        store.get_indexed_sources.return_value = {}

        _run_mcp_server(cfg, store, cfg.data_dir / "runtime_state.json")

        mock_register_tools.assert_called_once()
        deps = mock_register_tools.call_args.args[1]
        assert deps.cfg is cfg
        assert deps.store is store

        mock_watcher = mock_watcher_class.return_value
        mock_watcher.watch.assert_called()
        mock_watcher.start.assert_called_once()
        mock_watcher.stop.assert_called_once()

        mock_mcp_class.return_value.run.assert_called_once_with(transport=cfg.transport)
        mock_health_class.assert_not_called()

    def test_starts_health_server_when_port_set(
        self,
        mock_register_tools: MagicMock,
        mock_mcp_class: MagicMock,
        mock_watcher_class: MagicMock,
        mock_health_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        """A non-zero health_port spins up the health sidecar and shuts it down."""

        cfg = _make_cfg(tmp_path, health_port=8766)
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        store = MagicMock()
        store.get_indexed_sources.return_value = {}

        _run_mcp_server(cfg, store, cfg.data_dir / "runtime_state.json")

        mock_health_class.assert_called_once()
        mock_health = mock_health_class.return_value
        mock_health.start.assert_called_once()
        mock_health.stop.assert_called_once()

    def test_health_stops_before_watcher(
        self,
        mock_register_tools: MagicMock,
        mock_mcp_class: MagicMock,
        mock_watcher_class: MagicMock,
        mock_health_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        """
        Shutdown order: health server stops before the file watcher.

        a watcher-dispatched indexer call could otherwise race a /ready probe
        that's still answering 200.
        """

        cfg = _make_cfg(tmp_path, health_port=8766)
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        store = MagicMock()
        store.get_indexed_sources.return_value = {}

        order: list[str] = []
        mock_health_class.return_value.stop.side_effect = lambda: order.append("health")
        mock_watcher_class.return_value.stop.side_effect = lambda: order.append(
            "watcher"
        )

        _run_mcp_server(cfg, store, cfg.data_dir / "runtime_state.json")

        assert order == ["health", "watcher"]

    def test_keyboard_interrupt_still_runs_shutdown(
        self,
        mock_register_tools: MagicMock,
        mock_mcp_class: MagicMock,
        mock_watcher_class: MagicMock,
        mock_health_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        """A KeyboardInterrupt from mcp.run is swallowed and the finally still fires."""

        cfg = _make_cfg(tmp_path)
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        store = MagicMock()
        store.get_indexed_sources.return_value = {}

        mock_mcp_class.return_value.run.side_effect = KeyboardInterrupt

        _run_mcp_server(cfg, store, cfg.data_dir / "runtime_state.json")

        mock_watcher_class.return_value.stop.assert_called_once()


class TestMain:
    """Tests for the main() entry point."""

    @patch("rag.server.load_config")
    def test_exits_on_config_error(self, mock_load_config: MagicMock) -> None:
        """A ConfigError from load_config exits with status 1."""

        mock_load_config.side_effect = ConfigError("bad config")

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1

    @patch("rag.server._run_mcp_server")
    @patch("rag.server.VectorStore")
    @patch("rag.server._install_shutdown_handlers")
    @patch("rag.server.configure_logging")
    @patch("rag.server.load_config")
    def test_invokes_run_mcp_server_under_vector_store_context(
        self,
        mock_load_config: MagicMock,
        mock_configure_logging: MagicMock,
        mock_install_shutdown: MagicMock,
        mock_vector_store_class: MagicMock,
        mock_run_mcp_server: MagicMock,
        tmp_path: Path,
    ) -> None:
        """main() loads config, opens the store, and delegates to _run_mcp_server."""

        cfg = _make_cfg(tmp_path)
        mock_load_config.return_value = cfg

        main()

        mock_configure_logging.assert_called_once_with(
            level=cfg.log_level, log_format=cfg.log_format
        )
        mock_install_shutdown.assert_called_once()
        mock_vector_store_class.assert_called_once()
        mock_run_mcp_server.assert_called_once()
        args = mock_run_mcp_server.call_args.args
        assert args[0] is cfg
        assert args[2] == cfg.data_dir / "runtime_state.json"
