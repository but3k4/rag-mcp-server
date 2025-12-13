"""Tests for the /healthz and /ready sidecar HTTP server."""

from __future__ import annotations

import json
import socket
import time
from typing import TYPE_CHECKING
from unittest.mock import MagicMock
import urllib.error
import urllib.request

import pytest

from rag.health import HealthServer
from rag.store import VectorStore

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

HTTP_OK = 200
HTTP_SERVICE_UNAVAILABLE = 503
_PROBE_TIMEOUT_SECONDS = 2.0


def _free_port() -> int:
    """Pick a free TCP port by letting the OS assign one."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _get(path: str, port: int) -> tuple[int, dict[str, object]]:
    """Fetch path from the local health server and decode the JSON body."""

    url = f"http://127.0.0.1:{port}{path}"
    try:
        with urllib.request.urlopen(url, timeout=_PROBE_TIMEOUT_SECONDS) as resp:
            body: dict[str, object] = json.loads(resp.read().decode("utf-8"))
            return resp.status, body
    except urllib.error.HTTPError as exc:
        body = json.loads(exc.read().decode("utf-8"))
        return exc.code, body


@pytest.fixture
def running_watcher() -> MagicMock:
    """Watcher mock that reports itself as running."""

    watcher = MagicMock()
    watcher.is_running.return_value = True
    return watcher


@pytest.fixture
def stopped_watcher() -> MagicMock:
    """Watcher mock that reports itself as stopped."""

    watcher = MagicMock()
    watcher.is_running.return_value = False
    return watcher


@pytest.fixture
def healthy_store(tmp_path: Path) -> Iterator[VectorStore]:
    """A real, working VectorStore usable from the readiness check."""

    store = VectorStore(tmp_path / "health_store")
    try:
        yield store
    finally:
        store.close()


@pytest.fixture
def broken_store() -> MagicMock:
    """Store mock whose get_indexed_sources raises, simulating a corrupt DB."""

    store = MagicMock()
    store.get_indexed_sources.side_effect = RuntimeError("sqlite boom")
    return store


def _with_server(store: object, watcher: object) -> tuple[HealthServer, int]:
    """Start a HealthServer on a free port. Caller is responsible for stop()."""

    port = _free_port()
    server = HealthServer(port, store, watcher)  # type: ignore[arg-type]
    server.start()
    time.sleep(0.05)  # let the thread actually bind
    return server, port


class TestHealthz:
    """Tests for /healthz liveness."""

    def test_returns_ok(
        self, healthy_store: VectorStore, running_watcher: MagicMock
    ) -> None:
        """Healthz always returns 200 when the HTTP thread is responsive."""

        server, port = _with_server(healthy_store, running_watcher)
        try:
            status, body = _get("/healthz", port)
        finally:
            server.stop()

        assert status == HTTP_OK
        assert body["status"] == "ok"


class TestReady:
    """Tests for /ready readiness."""

    def test_ready_when_all_predicates_pass(
        self, healthy_store: VectorStore, running_watcher: MagicMock
    ) -> None:
        """Returns 200 and status=ready when store + watcher are healthy."""

        server, port = _with_server(healthy_store, running_watcher)
        try:
            status, body = _get("/ready", port)
        finally:
            server.stop()

        assert status == HTTP_OK
        assert body["status"] == "ready"
        assert body["store_reachable"] is True
        assert body["watcher_alive"] is True

    def test_not_ready_when_watcher_stopped(
        self, healthy_store: VectorStore, stopped_watcher: MagicMock
    ) -> None:
        """Returns 503 when the watcher thread is not alive."""

        server, port = _with_server(healthy_store, stopped_watcher)
        try:
            status, body = _get("/ready", port)
        finally:
            server.stop()

        assert status == HTTP_SERVICE_UNAVAILABLE
        assert body["watcher_alive"] is False
        assert body["status"] == "not ready"

    def test_not_ready_when_store_broken(
        self, broken_store: MagicMock, running_watcher: MagicMock
    ) -> None:
        """Returns 503 and surfaces the store error when the DB call raises."""

        server, port = _with_server(broken_store, running_watcher)
        try:
            status, body = _get("/ready", port)
        finally:
            server.stop()

        assert status == HTTP_SERVICE_UNAVAILABLE
        assert body["store_reachable"] is False
        assert "sqlite boom" in str(body["store_error"])


class TestUnknownPath:
    """Unknown paths return 404."""

    def test_unknown_path_returns_404(
        self, healthy_store: VectorStore, running_watcher: MagicMock
    ) -> None:
        """An unrouted path under the health port returns 404 JSON."""

        server, port = _with_server(healthy_store, running_watcher)
        try:
            status, _body = _get("/nope", port)
        finally:
            server.stop()

        assert status == 404  # noqa: PLR2004


class TestLifecycle:
    """Tests for start/stop cleanliness."""

    def test_stop_is_idempotent(
        self, healthy_store: VectorStore, running_watcher: MagicMock
    ) -> None:
        """Calling stop twice does not raise."""

        server, _port = _with_server(healthy_store, running_watcher)
        server.stop()
        server.stop()  # must not raise
