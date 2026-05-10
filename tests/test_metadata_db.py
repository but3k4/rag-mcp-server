"""Tests for rag.metadata_db.MetadataDB schema handling and migrations."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

import pytest

from rag.metadata_db import (
    CURRENT_SCHEMA_VERSION,
    MetadataDB,
    SchemaVersionError,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestSchemaVersioning:
    """Tests for schema version detection and migration."""

    def test_fresh_db_stamps_current_version(self, tmp_path: Path) -> None:
        """A newly created DB is stamped with CURRENT_SCHEMA_VERSION."""

        db = MetadataDB(tmp_path / "fresh.db")
        version = db._read_schema_version()
        db.close()
        assert version == CURRENT_SCHEMA_VERSION

    def test_pre_versioning_db_stamped_to_current(self, tmp_path: Path) -> None:
        """
        A DB with tables but no schema_version row is stamped on open.

        Simulates an existing deployment from before versioning was
        introduced: the tables already match v1 so the fix is just to
        write the version marker.
        """

        db_path = tmp_path / "pre_versioning.db"
        MetadataDB(db_path).close()
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("DELETE FROM kv WHERE key = 'schema_version'")

        db = MetadataDB(db_path)
        version = db._read_schema_version()
        db.close()
        assert version == CURRENT_SCHEMA_VERSION

    def test_future_version_raises(self, tmp_path: Path) -> None:
        """A DB stamped with a version newer than the code refuses to open."""

        db_path = tmp_path / "future.db"
        MetadataDB(db_path).close()
        future = CURRENT_SCHEMA_VERSION + 99
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                "UPDATE kv SET value = ? WHERE key = 'schema_version'",
                (str(future),),
            )

        with pytest.raises(SchemaVersionError, match="newer than code"):
            MetadataDB(db_path)

    def test_non_integer_version_raises(self, tmp_path: Path) -> None:
        """A corrupted schema_version value is surfaced as SchemaVersionError."""

        db_path = tmp_path / "corrupt.db"
        MetadataDB(db_path).close()
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                "UPDATE kv SET value = 'not-a-number' WHERE key = 'schema_version'"
            )

        with pytest.raises(SchemaVersionError, match="not an integer"):
            MetadataDB(db_path)

    def test_reopen_at_current_version_is_noop(self, tmp_path: Path) -> None:
        """Reopening a DB already at the current version preserves data."""

        db_path = tmp_path / "reopen.db"
        db = MetadataDB(db_path)
        db.set_model_name("some-model")
        db.set_file_hash("/tmp/x.txt", "abc123")
        db.close()

        db2 = MetadataDB(db_path)
        assert db2.get_model_name() == "some-model"
        assert db2.all_file_hashes() == {"/tmp/x.txt": "abc123"}
        assert db2._read_schema_version() == CURRENT_SCHEMA_VERSION
        db2.close()


class TestFindPathByHash:
    """Tests for the duplicate-detection lookup."""

    def test_returns_other_path_with_same_hash(self, tmp_path: Path) -> None:
        """A second path with the same hash is returned to the caller."""

        db = MetadataDB(tmp_path / "dup.db")
        db.set_file_hash("/a.txt", "h1")
        db.set_file_hash("/b.txt", "h1")
        try:
            assert db.find_path_by_hash("h1", "/b.txt") == "/a.txt"
        finally:
            db.close()

    def test_returns_none_when_only_excluded_path_matches(self, tmp_path: Path) -> None:
        """The excluded path is filtered out even if its hash matches."""

        db = MetadataDB(tmp_path / "dup.db")
        db.set_file_hash("/a.txt", "h1")
        try:
            assert db.find_path_by_hash("h1", "/a.txt") is None
        finally:
            db.close()

    def test_returns_none_when_hash_unseen(self, tmp_path: Path) -> None:
        """An unknown hash returns None."""

        db = MetadataDB(tmp_path / "dup.db")
        try:
            assert db.find_path_by_hash("missing", "/a.txt") is None
        finally:
            db.close()


class TestCloseIdempotency:
    """close() must be safe to call multiple times."""

    def test_double_close_is_noop(self, tmp_path: Path) -> None:
        """Closing a second time does not raise."""

        db = MetadataDB(tmp_path / "double.db")
        db.close()
        db.close()  # must not raise


class TestCrossThreadAccess:
    """
    The connection is created on one thread but used from many.

    FastMCP tool bodies are dispatched to asyncio's thread pool via
    asyncio.to_thread. file-watcher events fire on the watchdog observer
    thread. Both ultimately reach MetadataDB, so the connection must
    tolerate cross-thread use.
    """

    def test_read_from_other_thread(self, tmp_path: Path) -> None:
        """A second thread can read data written on the main thread."""

        import threading  # noqa: PLC0415

        db = MetadataDB(tmp_path / "threaded.db")
        db.set_file_hash("/tmp/x.txt", "h1")

        result: list[dict[str, str]] = []

        def worker() -> None:
            result.append(db.all_file_hashes())

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        db.close()

        assert result == [{"/tmp/x.txt": "h1"}]

    def test_concurrent_writes_do_not_raise(self, tmp_path: Path) -> None:
        """Multiple threads writing different keys serialise without errors."""

        import threading  # noqa: PLC0415

        db = MetadataDB(tmp_path / "concurrent.db")

        def writer(prefix: str) -> None:
            for i in range(20):
                db.set_file_hash(f"{prefix}/{i}", f"hash-{i}")

        threads = [threading.Thread(target=writer, args=(f"/t{n}",)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total = len(db.all_file_hashes())
        db.close()

        assert total == 80  # noqa: PLR2004
