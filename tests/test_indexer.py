"""Tests for the incremental indexer logic."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from rag.errors import StoreError
from rag.indexer import Indexer, IndexSummary

if TYPE_CHECKING:
    import pytest


class TestIndexerRun:
    """Tests for the Indexer.run() method."""

    def _make_indexer(self, store: MagicMock, source_dirs: list[Path]) -> Indexer:
        """Build an Indexer with the given store and source dirs for use in tests."""

        return Indexer(
            store=store, source_dirs=source_dirs, chunk_size=100, chunk_overlap=10
        )

    def test_returns_index_summary(self, tmp_path: Path) -> None:
        """run() always returns an IndexSummary instance."""

        store = MagicMock()
        store.get_indexed_sources.return_value = {}
        indexer = self._make_indexer(store, [tmp_path])
        result = indexer.run()
        assert isinstance(result, IndexSummary)

    def test_empty_directory_scans_zero_files(self, tmp_path: Path) -> None:
        """An empty source directory produces zero scanned files."""

        store = MagicMock()
        store.get_indexed_sources.return_value = {}
        indexer = self._make_indexer(store, [tmp_path])
        summary = indexer.run()
        assert summary.files_scanned == 0

    def test_new_file_is_indexed(self, tmp_path: Path) -> None:
        """A new .txt file is parsed and upserted into the store."""

        (tmp_path / "doc.txt").write_text("hello world content")
        store = MagicMock()
        store.get_indexed_sources.return_value = {}
        indexer = self._make_indexer(store, [tmp_path])
        summary = indexer.run()
        assert summary.files_scanned == 1
        assert summary.files_updated == 1
        assert summary.files_failed == 0
        store.upsert_file.assert_called_once()

    def test_unchanged_file_is_skipped(self, tmp_path: Path) -> None:
        """A file whose hash matches the stored value is not re-indexed."""

        doc = tmp_path / "doc.txt"
        doc.write_text("content")
        current_hash = hashlib.sha256(doc.read_bytes()).hexdigest()
        store = MagicMock()
        store.get_indexed_sources.return_value = {str(doc): current_hash}
        indexer = self._make_indexer(store, [tmp_path])
        summary = indexer.run()
        assert summary.files_updated == 0
        store.upsert_file.assert_not_called()

    def test_parse_failure_is_counted_not_raised(self, tmp_path: Path) -> None:
        """A file that cannot be parsed increments files_failed and continues."""

        doc = tmp_path / "broken.pdf"
        doc.write_bytes(b"not a real pdf")
        store = MagicMock()
        store.get_indexed_sources.return_value = {}
        indexer = self._make_indexer(store, [tmp_path])
        summary = indexer.run()
        assert summary.files_failed == 1
        assert str(doc) in summary.failed_paths
        store.upsert_file.assert_not_called()

    def test_nonexistent_source_dir_is_skipped(self, tmp_path: Path) -> None:
        """A source directory that does not exist is skipped without error."""

        missing = tmp_path / "missing"
        store = MagicMock()
        store.get_indexed_sources.return_value = {}
        indexer = self._make_indexer(store, [missing])
        summary = indexer.run()
        assert summary.files_scanned == 0
        assert summary.files_failed == 0

    def test_unsupported_extension_not_scanned(self, tmp_path: Path) -> None:
        """Files with unsupported extensions are not collected."""

        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        store = MagicMock()
        store.get_indexed_sources.return_value = {}
        indexer = self._make_indexer(store, [tmp_path])
        summary = indexer.run()
        assert summary.files_scanned == 0

    def test_subdirectory_files_are_scanned(self, tmp_path: Path) -> None:
        """Files in subdirectories within a source dir are included."""

        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "doc.txt").write_text("content")
        store = MagicMock()
        store.get_indexed_sources.return_value = {}
        indexer = self._make_indexer(store, [tmp_path])
        summary = indexer.run()
        assert summary.files_scanned == 1

    def test_symlink_escaping_source_dir_is_skipped(self, tmp_path: Path) -> None:
        """A symlink pointing outside the source directory is skipped."""

        src = tmp_path / "src"
        src.mkdir()
        outside = tmp_path / "outside.txt"
        outside.write_text("secret content")
        link = src / "link.txt"
        link.symlink_to(outside)
        store = MagicMock()
        store.get_indexed_sources.return_value = {}
        indexer = self._make_indexer(store, [src])
        summary = indexer.run()
        assert summary.files_scanned == 0

    def test_bm25_rebuilt_after_updates(self, tmp_path: Path) -> None:
        """build_bm25 is called once when files were updated."""

        (tmp_path / "doc.txt").write_text("hello world content")
        store = MagicMock()
        store.get_indexed_sources.return_value = {}
        indexer = self._make_indexer(store, [tmp_path])
        indexer.run()
        store.build_bm25.assert_called_once()

    def test_bm25_not_rebuilt_when_nothing_changed(self, tmp_path: Path) -> None:
        """build_bm25 is not called when all files are already up-to-date."""

        doc = tmp_path / "doc.txt"
        doc.write_text("content")
        store = MagicMock()
        store.get_indexed_sources.return_value = {
            str(doc): hashlib.sha256(doc.read_bytes()).hexdigest()
        }
        indexer = self._make_indexer(store, [tmp_path])
        indexer.run()
        store.build_bm25.assert_not_called()

    def test_read_failure_increments_failed_count(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the file cannot be read for hashing, it is counted as failed."""

        doc = tmp_path / "doc.txt"
        doc.write_text("content")
        store = MagicMock()
        store.get_indexed_sources.return_value = {}

        original_read_bytes = Path.read_bytes

        def _bad_read(self: Path) -> bytes:
            """Monkeypatch stand-in for Path.read_bytes that fails on the target path."""

            if self == doc:
                raise OSError("permission denied")
            return original_read_bytes(self)

        monkeypatch.setattr(Path, "read_bytes", _bad_read)
        indexer = self._make_indexer(store, [tmp_path])
        summary = indexer.run()
        assert summary.files_failed == 1
        assert str(doc) in summary.failed_paths


class TestIndexerNewMethods:
    """Tests for run_for_dir, index_file, remove_file, add_source_dir, remove_source_dir."""

    def _make_indexer(self, store: MagicMock, source_dirs: list[Path]) -> Indexer:
        """Build an Indexer with the given store and source dirs for use in tests."""

        return Indexer(
            store=store, source_dirs=source_dirs, chunk_size=100, chunk_overlap=10
        )

    def test_run_for_dir_indexes_only_given_directory(self, tmp_path: Path) -> None:
        """run_for_dir scans only the specified directory."""

        other = tmp_path / "other"
        other.mkdir()
        target = tmp_path / "target"
        target.mkdir()
        (target / "doc.txt").write_text("content")
        (other / "doc.txt").write_text("other content")
        store = MagicMock()
        store.get_indexed_sources.return_value = {}
        indexer = self._make_indexer(store, [tmp_path])
        summary = indexer.run_for_dir(target)
        assert summary.files_scanned == 1
        assert summary.files_updated == 1

    def test_index_file_calls_upsert_for_changed_file(self, tmp_path: Path) -> None:
        """index_file upserts a file whose hash differs from the stored value."""

        doc = tmp_path / "doc.txt"
        doc.write_text("hello")
        store = MagicMock()
        store.get_indexed_sources.return_value = {}
        indexer = self._make_indexer(store, [tmp_path])
        indexer.index_file(doc)
        store.upsert_file.assert_called_once()

    def test_index_file_skips_unchanged_file(self, tmp_path: Path) -> None:
        """index_file skips a file whose hash is already stored."""

        doc = tmp_path / "doc.txt"
        doc.write_text("hello")
        current_hash = hashlib.sha256(doc.read_bytes()).hexdigest()
        store = MagicMock()
        store.get_indexed_sources.return_value = {str(doc): current_hash}
        indexer = self._make_indexer(store, [tmp_path])
        indexer.index_file(doc)
        store.upsert_file.assert_not_called()

    def test_index_file_ignores_unsupported_extension(self, tmp_path: Path) -> None:
        """index_file does nothing for unsupported file types."""

        img = tmp_path / "photo.png"
        img.write_bytes(b"\x89PNG")
        store = MagicMock()
        store.get_indexed_sources.return_value = {}
        indexer = self._make_indexer(store, [tmp_path])
        indexer.index_file(img)
        store.upsert_file.assert_not_called()

    def test_remove_file_calls_store_delete(self, tmp_path: Path) -> None:
        """remove_file delegates to store.delete_file."""

        store = MagicMock()
        indexer = self._make_indexer(store, [tmp_path])
        indexer.remove_file(tmp_path / "gone.txt")
        store.delete_file.assert_called_once_with(str(tmp_path / "gone.txt"))

    def test_add_source_dir_appends_path(self, tmp_path: Path) -> None:
        """add_source_dir adds the path to the internal source dirs list."""

        store = MagicMock()
        indexer = self._make_indexer(store, [])
        new_dir = tmp_path / "new"
        indexer.add_source_dir(new_dir)
        assert new_dir in indexer._source_dirs

    def test_add_source_dir_is_idempotent(self, tmp_path: Path) -> None:
        """add_source_dir does not add duplicates."""

        store = MagicMock()
        indexer = self._make_indexer(store, [tmp_path])
        indexer.add_source_dir(tmp_path)
        assert indexer._source_dirs.count(tmp_path) == 1

    def test_remove_source_dir_removes_path(self, tmp_path: Path) -> None:
        """remove_source_dir removes the path and deletes its index entries."""

        store = MagicMock()
        indexer = self._make_indexer(store, [tmp_path])
        indexer.remove_source_dir(tmp_path)
        assert tmp_path not in indexer._source_dirs
        store.delete_directory.assert_called_once()

    def test_remove_source_dir_nonexistent_is_noop(self, tmp_path: Path) -> None:
        """remove_source_dir for an unconfigured directory does not raise."""

        store = MagicMock()
        indexer = self._make_indexer(store, [])
        indexer.remove_source_dir(tmp_path / "missing")  # must not raise

    def test_run_for_dir_read_failure_counted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run_for_dir counts files that cannot be read as failed."""

        doc = tmp_path / "doc.txt"
        doc.write_text("content")
        store = MagicMock()
        store.get_indexed_sources.return_value = {}
        original_read_bytes = Path.read_bytes

        def _bad_read(self: Path) -> bytes:
            """Monkeypatch stand-in for Path.read_bytes that fails on the target path."""

            if self == doc:
                raise OSError("denied")
            return original_read_bytes(self)

        monkeypatch.setattr(Path, "read_bytes", _bad_read)
        indexer = self._make_indexer(store, [tmp_path])
        summary = indexer.run_for_dir(tmp_path)
        assert summary.files_failed == 1
        assert str(doc) in summary.failed_paths

    def test_run_for_dir_parse_failure_counted(self, tmp_path: Path) -> None:
        """run_for_dir counts files that fail to parse as failed."""

        doc = tmp_path / "broken.pdf"
        doc.write_bytes(b"not a pdf")
        store = MagicMock()
        store.get_indexed_sources.return_value = {}
        indexer = self._make_indexer(store, [tmp_path])
        summary = indexer.run_for_dir(tmp_path)
        assert summary.files_failed == 1

    def test_index_file_read_failure_is_silent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """index_file does not raise when the file cannot be read."""

        doc = tmp_path / "doc.txt"
        doc.write_text("content")
        store = MagicMock()
        store.get_indexed_sources.return_value = {}
        original_read_bytes = Path.read_bytes

        def _bad_read(self: Path) -> bytes:
            """Monkeypatch stand-in for Path.read_bytes that fails on the target path."""

            if self == doc:
                raise OSError("denied")
            return original_read_bytes(self)

        monkeypatch.setattr(Path, "read_bytes", _bad_read)
        indexer = self._make_indexer(store, [tmp_path])
        indexer.index_file(doc)  # must not raise
        store.upsert_file.assert_not_called()

    def test_index_file_parse_failure_is_silent(self, tmp_path: Path) -> None:
        """index_file does not raise when the file cannot be parsed."""

        doc = tmp_path / "broken.pdf"
        doc.write_bytes(b"not a pdf")
        store = MagicMock()
        store.get_indexed_sources.return_value = {}
        indexer = self._make_indexer(store, [tmp_path])
        indexer.index_file(doc)  # must not raise
        store.upsert_file.assert_not_called()


class TestMaxFileSize:
    """Tests for max_file_size_bytes enforcement."""

    def test_oversized_file_is_skipped_in_run(self, tmp_path: Path) -> None:
        """run() counts oversized files as failed and does not upsert them."""

        doc = tmp_path / "big.txt"
        doc.write_text("x" * 1024)
        store = MagicMock()
        store.get_indexed_sources.return_value = {}
        indexer = Indexer(
            store=store,
            source_dirs=[tmp_path],
            chunk_size=100,
            chunk_overlap=10,
            max_file_size_bytes=100,
        )
        summary = indexer.run()
        store.upsert_file.assert_not_called()
        assert summary.files_failed == 1
        assert str(doc) in summary.failed_paths

    def test_oversized_file_is_skipped_in_index_file(self, tmp_path: Path) -> None:
        """index_file() returns silently when the file is oversized."""

        doc = tmp_path / "big.txt"
        doc.write_text("x" * 1024)
        store = MagicMock()
        store.get_indexed_sources.return_value = {}
        indexer = Indexer(
            store=store,
            source_dirs=[tmp_path],
            chunk_size=100,
            chunk_overlap=10,
            max_file_size_bytes=100,
        )
        indexer.index_file(doc)
        store.upsert_file.assert_not_called()

    def test_small_file_indexed_when_under_cap(self, tmp_path: Path) -> None:
        """A file smaller than the cap is indexed normally."""

        doc = tmp_path / "small.txt"
        doc.write_text("tiny")
        store = MagicMock()
        store.get_indexed_sources.return_value = {}
        indexer = Indexer(
            store=store,
            source_dirs=[tmp_path],
            chunk_size=100,
            chunk_overlap=10,
            max_file_size_bytes=1024,
        )
        summary = indexer.run()
        store.upsert_file.assert_called_once()
        assert summary.files_updated == 1


class TestStoreErrorHandling:
    """Batch and watcher paths survive per-file StoreError failures."""

    def test_run_counts_store_error_as_failed_and_continues(
        self, tmp_path: Path
    ) -> None:
        """If one file's upsert raises StoreError, others still index."""

        good = tmp_path / "good.txt"
        bad = tmp_path / "bad.txt"
        good.write_text("good content")
        bad.write_text("bad content")

        store = MagicMock()
        store.get_indexed_sources.return_value = {}

        def upsert_side_effect(path: str, *_args: object, **_kw: object) -> None:
            if path == str(bad):
                raise StoreError("chroma down")

        store.upsert_file.side_effect = upsert_side_effect
        indexer = Indexer(
            store=store, source_dirs=[tmp_path], chunk_size=100, chunk_overlap=10
        )
        summary = indexer.run()
        assert summary.files_updated == 1
        assert summary.files_failed == 1
        assert str(bad) in summary.failed_paths

    def test_index_file_swallows_store_error(self, tmp_path: Path) -> None:
        """The watcher path logs StoreError instead of propagating."""

        doc = tmp_path / "doc.txt"
        doc.write_text("content")
        store = MagicMock()
        store.get_indexed_sources.return_value = {}
        store.upsert_file.side_effect = StoreError("chroma down")
        indexer = Indexer(
            store=store, source_dirs=[tmp_path], chunk_size=100, chunk_overlap=10
        )
        indexer.index_file(doc)  # must not raise

    def test_remove_file_swallows_store_error(self, tmp_path: Path) -> None:
        """remove_file logs StoreError instead of propagating."""

        store = MagicMock()
        store.delete_file.side_effect = StoreError("chroma down")
        indexer = Indexer(
            store=store, source_dirs=[tmp_path], chunk_size=100, chunk_overlap=10
        )
        indexer.remove_file(tmp_path / "gone.txt")  # must not raise

    def test_prune_continues_past_store_error(self, tmp_path: Path) -> None:
        """_prune_missing skips entries that fail to delete and counts the rest."""

        store = MagicMock()
        # Two stale entries, one of them fails to delete.
        store.get_indexed_sources.return_value = {
            str(tmp_path / "a.txt"): "h1",
            str(tmp_path / "b.txt"): "h2",
        }

        def delete_side_effect(path: str) -> None:
            if path == str(tmp_path / "a.txt"):
                raise StoreError("chroma down")

        store.delete_file.side_effect = delete_side_effect
        indexer = Indexer(
            store=store, source_dirs=[tmp_path], chunk_size=100, chunk_overlap=10
        )
        summary = indexer.run()
        assert summary.files_pruned == 1


class TestPruneMissing:
    """Tests for the prune-on-reindex behaviour."""

    def _make_indexer(self, store: MagicMock, source_dirs: list[Path]) -> Indexer:
        """Build an Indexer with the given store and source dirs for use in tests."""

        return Indexer(
            store=store, source_dirs=source_dirs, chunk_size=100, chunk_overlap=10
        )

    def test_run_prunes_entry_for_missing_file(self, tmp_path: Path) -> None:
        """run() deletes a stored entry whose file is no longer on disk."""

        gone = tmp_path / "gone.txt"
        store = MagicMock()
        store.get_indexed_sources.return_value = {str(gone): "old-hash"}
        indexer = self._make_indexer(store, [tmp_path])
        summary = indexer.run()
        store.delete_file.assert_called_once_with(str(gone))
        assert summary.files_pruned == 1

    def test_run_does_not_prune_present_file(self, tmp_path: Path) -> None:
        """run() leaves entries alone when the file still exists on disk."""

        doc = tmp_path / "doc.txt"
        doc.write_text("content")
        current_hash = hashlib.sha256(doc.read_bytes()).hexdigest()
        store = MagicMock()
        store.get_indexed_sources.return_value = {str(doc): current_hash}
        indexer = self._make_indexer(store, [tmp_path])
        summary = indexer.run()
        store.delete_file.assert_not_called()
        assert summary.files_pruned == 0

    def test_prune_respects_scope_in_run_for_dir(self, tmp_path: Path) -> None:
        """run_for_dir() only prunes entries under the given directory."""

        target = tmp_path / "target"
        other = tmp_path / "other"
        target.mkdir()
        other.mkdir()
        stale_in_scope = target / "gone.txt"
        stale_out_of_scope = other / "gone.txt"
        store = MagicMock()
        store.get_indexed_sources.return_value = {
            str(stale_in_scope): "h1",
            str(stale_out_of_scope): "h2",
        }
        indexer = self._make_indexer(store, [tmp_path])
        summary = indexer.run_for_dir(target)
        store.delete_file.assert_called_once_with(str(stale_in_scope))
        assert summary.files_pruned == 1

    def test_rename_prunes_old_and_indexes_new(self, tmp_path: Path) -> None:
        """After a rename on disk, run() prunes the old path and indexes the new."""

        old = tmp_path / "old.txt"
        new = tmp_path / "new.txt"
        new.write_text("renamed content")
        store = MagicMock()
        store.get_indexed_sources.return_value = {str(old): "old-hash"}
        indexer = self._make_indexer(store, [tmp_path])
        summary = indexer.run()
        store.delete_file.assert_called_once_with(str(old))
        store.upsert_file.assert_called_once()
        assert store.upsert_file.call_args.args[0] == str(new)
        assert summary.files_pruned == 1
        assert summary.files_updated == 1

    def test_vanished_file_during_walk_not_counted_as_failed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """FileNotFoundError during hashing is skipped silently, not counted."""

        doc = tmp_path / "doc.txt"
        doc.write_text("content")
        original_read_bytes = Path.read_bytes

        def _vanish(self: Path) -> bytes:
            """Raise FileNotFoundError for the target path to simulate a race."""

            if self == doc:
                raise FileNotFoundError(str(doc))
            return original_read_bytes(self)

        monkeypatch.setattr(Path, "read_bytes", _vanish)
        store = MagicMock()
        store.get_indexed_sources.return_value = {}
        indexer = self._make_indexer(store, [tmp_path])
        summary = indexer.run()
        assert summary.files_failed == 0
        assert summary.failed_paths == []

    def test_index_file_vanished_file_is_silent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """index_file returns silently when the file vanishes during hashing."""

        doc = tmp_path / "doc.txt"
        doc.write_text("content")
        original_read_bytes = Path.read_bytes

        def _vanish(self: Path) -> bytes:
            """Raise FileNotFoundError for the target path to simulate a race."""

            if self == doc:
                raise FileNotFoundError(str(doc))
            return original_read_bytes(self)

        monkeypatch.setattr(Path, "read_bytes", _vanish)
        store = MagicMock()
        store.get_indexed_sources.return_value = {}
        indexer = self._make_indexer(store, [tmp_path])
        indexer.index_file(doc)  # must not raise
        store.upsert_file.assert_not_called()
