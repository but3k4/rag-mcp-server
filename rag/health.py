"""
Sidecar HTTP server exposing /healthz (liveness) and /ready (readiness).

Runs in a daemon thread separate from FastMCP so it works with any MCP
transport (stdio, sse, streamable-http). Uses stdlib http.server so no
extra dependency is needed.

/healthz always returns 200 as long as the Python process is up and the
server thread responds. Intended for liveness probing (crash detection,
OOM kills, deadlocks). /ready exercises real predicates: the metadata DB
is reachable and the file watcher thread is alive. Returns 503 with a JSON
body describing which predicate failed when the server is not ready to
accept MCP traffic.
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rag.store import VectorStore
    from rag.watcher import DirectoryWatcher

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class HealthServer:
    """
    HTTP server for health and readiness probing.

    Start once via :meth:start and the server runs in a daemon thread
    bound to the configured port. Call :meth:stop to shut it down
    cleanly. Part of the graceful-shutdown path in server.main().
    """

    def __init__(
        self,
        port: int,
        store: VectorStore,
        watcher: DirectoryWatcher,
    ) -> None:
        """
        Initialise the health server.

        Args:
            port: TCP port to listen on. Must be >= 1.
            store: VectorStore instance whose health is exposed on /ready.
            watcher: DirectoryWatcher whose thread liveness is exposed on /ready.
        """

        self._port = port
        self._store = store
        self._watcher = watcher
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Bind the server and start the serving thread."""

        handler_cls = _make_handler(self._store, self._watcher)
        self._server = ThreadingHTTPServer(("0.0.0.0", self._port), handler_cls)  # nosec B104
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="health-server",
        )
        self._thread.start()
        logger.info("Health server listening on :%d", self._port)

    def stop(self) -> None:
        """Shut down the server and join the serving thread."""

        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._server = None
        self._thread = None


def _make_handler(
    store: VectorStore, watcher: DirectoryWatcher
) -> type[BaseHTTPRequestHandler]:
    """Build a request handler class closed over the store and watcher."""

    class _HealthHandler(BaseHTTPRequestHandler):
        """Route handler for /healthz and /ready."""

        def do_GET(self) -> None:
            """Dispatch GET to the healthz or ready handler."""

            if self.path == "/healthz":
                self._respond(200, {"status": "ok"})
            elif self.path == "/ready":
                ready, body = _check_readiness(store, watcher)
                self._respond(200 if ready else 503, body)
            else:
                self._respond(404, {"error": "not found"})

        def _respond(self, status: int, body: dict[str, Any]) -> None:
            """Write a JSON response with the given status and body."""

            payload = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            """Suppress per-request stderr logs; probes fire every ~30s."""

    return _HealthHandler


def _check_readiness(
    store: VectorStore, watcher: DirectoryWatcher
) -> tuple[bool, dict[str, Any]]:
    """
    Check readiness predicates and return (ready, detail_body).

    Predicates:
      - Metadata DB reachable (SQLite + basic query).
      - Watcher thread alive.
    """

    details: dict[str, Any] = {}
    ok = True

    try:
        store.get_indexed_sources()
    except Exception as exc:  # noqa: BLE001 (any failure means not-ready)
        details["store_reachable"] = False
        details["store_error"] = str(exc)
        ok = False
    else:
        details["store_reachable"] = True

    watcher_alive = watcher.is_running()
    details["watcher_alive"] = watcher_alive
    if not watcher_alive:
        ok = False

    details["status"] = "ready" if ok else "not ready"
    return ok, details
