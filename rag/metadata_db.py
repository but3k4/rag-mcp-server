"""
SQLite-backed source of truth for non-vector RAG state.

Stores the embedding model name, per-file SHA-256 hashes, and parent
section text. Kept separate from the Chroma vector index so the index
is a rebuildable cache of embeddings rather than a data store.
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

CURRENT_SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS kv (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS file_hashes (
    path TEXT PRIMARY KEY,
    hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS parents (
    id      TEXT PRIMARY KEY,
    source  TEXT NOT NULL,
    section TEXT NOT NULL,
    text    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_parents_source ON parents(source);

CREATE INDEX IF NOT EXISTS idx_file_hashes_hash ON file_hashes(hash);
"""

# Maps target version -> callable that migrates the connection from (target-1)
# to target. Each migration runs inside a transaction along with the
# schema_version bump, so a failure rolls both back.
_MIGRATIONS: dict[int, Callable[[sqlite3.Connection], None]] = {}


class SchemaVersionError(RuntimeError):
    """Raised when the stored schema version is incompatible with this code."""


class MetadataDB:
    """SQLite wrapper for RAG metadata (model name, file hashes, parent text)."""

    def __init__(self, db_path: Path) -> None:
        """
        Open (or create) the database at db_path.

        Args:
            db_path: Path to the SQLite file. Parent directory must already
                     exist. VectorStore creates it before constructing this.
        """

        # check_same_thread=False lets the MCP server's thread-pool workers
        # (used by asyncio.to_thread) share this connection with the main
        # thread and the watchdog observer thread.
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._lock = threading.RLock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._initialize_schema()

    def close(self) -> None:
        """
        Close the underlying connection.

        Idempotent: safe to call multiple times. A second call (including
        from __del__) becomes a no-op.
        """

        if self._conn is None:
            return
        with self._lock:
            if self._conn is None:
                return
            self._conn.close()
            self._conn = None  # type: ignore[assignment]

    def __del__(self) -> None:
        """Best-effort cleanup if close() was not called explicitly."""

        conn = getattr(self, "_conn", None)
        if conn is None:
            return

        with contextlib.suppress(sqlite3.Error):
            conn.close()

    def _initialize_schema(self) -> None:
        """
        Ensure the schema exists and apply any pending migrations.

        A fresh DB is stamped with the current version. A DB from before
        versioning was introduced (tables present but no schema_version row)
        is treated as version 1 and stamped accordingly. Its layout already
        matches v1. A DB at a future version is rejected so we never run
        against a layout this code does not understand.
        """

        self._conn.executescript(_SCHEMA)

        stored = self._read_schema_version()
        if stored is None:
            self._write_schema_version(CURRENT_SCHEMA_VERSION)
            return

        if stored > CURRENT_SCHEMA_VERSION:
            raise SchemaVersionError(
                f"Database schema version {stored} is newer than code "
                f"version {CURRENT_SCHEMA_VERSION}. Upgrade the server."
            )

        if stored < CURRENT_SCHEMA_VERSION:
            self._apply_migrations(from_version=stored)

    def _apply_migrations(self, from_version: int) -> None:
        """Run each migration from from_version+1 up to CURRENT_SCHEMA_VERSION."""

        for target in range(from_version + 1, CURRENT_SCHEMA_VERSION + 1):
            migration = _MIGRATIONS.get(target)
            if migration is None:
                raise SchemaVersionError(f"No migration defined for schema v{target}")
            logger.info("Applying schema migration v%d -> v%d", target - 1, target)
            with self._lock, self._conn:
                migration(self._conn)
                self._conn.execute(
                    "INSERT INTO kv(key, value) VALUES('schema_version', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (str(target),),
                )

    def _read_schema_version(self) -> int | None:
        """Return the stored schema_version as an int, or None if unset."""

        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM kv WHERE key = 'schema_version'"
            ).fetchone()

        if row is None:
            return None

        try:
            return int(row[0])
        except ValueError as exc:
            raise SchemaVersionError(
                f"Stored schema_version is not an integer: {row[0]!r}"
            ) from exc

    def _write_schema_version(self, version: int) -> None:
        """Persist the schema_version row."""

        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO kv(key, value) VALUES('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (str(version),),
            )

    def get_model_name(self) -> str | None:
        """Return the stored embedding model name, or None if never set."""

        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM kv WHERE key = 'model'"
            ).fetchone()
        return row[0] if row else None

    def set_model_name(self, name: str) -> None:
        """Store (or overwrite) the embedding model name."""

        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO kv(key, value) VALUES('model', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (name,),
            )

    def get_parser_version(self) -> str | None:
        """Return the stored parser pipeline version, or None if never set."""

        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM kv WHERE key = 'parser_version'"
            ).fetchone()
        return row[0] if row else None

    def set_parser_version(self, version: str) -> None:
        """Store (or overwrite) the parser pipeline version."""

        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO kv(key, value) VALUES('parser_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (version,),
            )

    def set_file_hash(self, path: str, file_hash: str) -> None:
        """Upsert the SHA-256 hash for a source file path."""

        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO file_hashes(path, hash) VALUES(?, ?) "
                "ON CONFLICT(path) DO UPDATE SET hash = excluded.hash",
                (path, file_hash),
            )

    def delete_file_hash(self, path: str) -> None:
        """Remove the hash entry for a source file path."""

        with self._lock, self._conn:
            self._conn.execute("DELETE FROM file_hashes WHERE path = ?", (path,))

    def all_file_hashes(self) -> dict[str, str]:
        """Return a mapping of every indexed source path to its stored hash."""

        with self._lock:
            rows = self._conn.execute("SELECT path, hash FROM file_hashes").fetchall()
        return dict(rows)

    def find_path_by_hash(self, file_hash: str, exclude_path: str) -> str | None:
        """
        Return another indexed path that has this hash, or None.

        Used by the indexer to detect content duplicates so the same file
        sitting under multiple source directories gets embedded only once.
        """

        with self._lock:
            row = self._conn.execute(
                "SELECT path FROM file_hashes WHERE hash = ? AND path != ? LIMIT 1",
                (file_hash, exclude_path),
            ).fetchone()
        return row[0] if row else None

    def upsert_parents(self, rows: Iterable[tuple[str, str, str, str]]) -> None:
        """
        Upsert parent section rows.

        Args:
            rows: Iterable of (id, source, section, text) tuples.
        """

        payload = list(rows)
        if not payload:
            return
        with self._lock, self._conn:
            self._conn.executemany(
                "INSERT INTO parents(id, source, section, text) "
                "VALUES(?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET "
                "source = excluded.source, "
                "section = excluded.section, "
                "text = excluded.text",
                payload,
            )

    def get_parent_texts(self, ids: list[str]) -> dict[str, str]:
        """Return {id: text} for the given parent IDs. Missing IDs are absent."""

        if not ids:
            return {}

        placeholders = ",".join("?" * len(ids))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT id, text FROM parents WHERE id IN ({placeholders})",  # nosec B608
                ids,
            ).fetchall()

        return dict(rows)

    def delete_parents_by_source(self, source: str) -> None:
        """Delete all parent rows for a given source file path."""

        with self._lock, self._conn:
            self._conn.execute("DELETE FROM parents WHERE source = ?", (source,))

    def clear_indexed_state(self) -> None:
        """Wipe file_hashes and parents. Used on embedding-model change."""

        with self._lock, self._conn:
            self._conn.execute("DELETE FROM file_hashes")
            self._conn.execute("DELETE FROM parents")
