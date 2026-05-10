"""
Incremental document indexer.

Scans configured source directories, detects files whose SHA-256 hash has
changed since the last index run, parses them into section-aware chunks, and
updates the vector store. Files that fail to parse are logged and skipped.
The BM25 index is rebuilt once at the end of each run rather than per file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from rag.errors import StoreError
from rag.parsers import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    SUPPORTED_EXTENSIONS,
    ParseError,
    PasswordProtectedError,
    chunk_file,
)

if TYPE_CHECKING:
    from rag.store import VectorStore

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class IndexSummary:
    """Result returned by a single index run."""

    files_scanned: int = 0
    files_updated: int = 0
    files_failed: int = 0
    files_pruned: int = 0
    failed_paths: list[str] = field(default_factory=list)


def _file_hash(path: Path) -> str:
    """
    Compute the SHA-256 hash of a file.

    Args:
        path: Path to the file.

    Returns:
        Hex-encoded SHA-256 digest string.

    Raises:
        OSError: If the file cannot be read.
    """

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _collect_files(source_dirs: list[Path]) -> list[Path]:
    """
    Walk each source directory and return all files with supported extensions.

    Traversal is bounded to within each source directory. Symlinks that escape
    the configured root are skipped with a warning.

    Args:
        source_dirs: List of expanded, absolute directory paths.

    Returns:
        Sorted list of matching file paths.
    """

    found: list[Path] = []

    for src_dir in source_dirs:
        if not src_dir.is_dir():
            logger.warning("Source directory does not exist, skipping: %s", src_dir)
            continue

        for path in src_dir.rglob("*"):
            if not path.is_file():
                continue

            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            try:
                path.resolve().relative_to(src_dir.resolve())
            except ValueError:
                logger.warning("Path escapes source directory, skipping: %s", path)

                continue

            found.append(path)
    return sorted(found)


class Indexer:
    """Coordinates incremental indexing across configured source directories."""

    def __init__(
        self,
        store: VectorStore,
        source_dirs: list[Path],
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        max_file_size_bytes: int | None = None,
    ) -> None:
        """
        Initialise the indexer.

        Args:
            store: The vector store instance to read from and write to.
            source_dirs: Expanded, absolute source directory paths.
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Characters of overlap between consecutive chunks.
            max_file_size_bytes: Files larger than this are skipped with a
                                 warning rather than parsed. None disables the
                                 check.
        """

        self._store = store
        self._source_dirs = source_dirs
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._max_file_size_bytes = max_file_size_bytes

    def run(self) -> IndexSummary:
        """
        Perform one incremental index pass across all configured directories.

        For each file found in the source directories, checks whether its
        SHA-256 hash has changed since it was last indexed. Only changed or
        new files are re-parsed and re-embedded. Files that fail to parse are
        logged and counted in the summary without aborting the run.

        The BM25 index is rebuilt once at the end rather than after each
        individual file upsert.

        Returns:
            IndexSummary with counts of scanned, updated, failed, and pruned
            files.
        """

        files = _collect_files(self._source_dirs)
        summary = self._index_files(files)
        summary.files_pruned = self._prune_missing(files, self._source_dirs)
        return summary

    def run_for_dir(self, directory: Path) -> IndexSummary:
        """
        Perform an incremental index pass limited to a single directory.

        Args:
            directory: Expanded, absolute path to the directory to index.

        Returns:
            IndexSummary with counts for that directory only, including
            pruned entries whose files no longer exist.
        """

        files = _collect_files([directory])
        summary = self._index_files(files)
        summary.files_pruned = self._prune_missing(files, [directory])
        return summary

    def _is_oversized(self, path: Path) -> bool:
        """Return True when the file exceeds max_file_size_bytes (if set)."""

        if self._max_file_size_bytes is None:
            return False

        try:
            size = path.stat().st_size
        except OSError:
            return False
        return size > self._max_file_size_bytes

    def _index_files(self, files: list[Path]) -> IndexSummary:
        """Hash, parse, and upsert a list of files, skipping unchanged ones.

        Defers the BM25 rebuild until after the batch to avoid rebuilding
        once per file.
        """

        summary = IndexSummary(files_scanned=len(files))
        indexed = self._store.get_indexed_sources()

        for file_path in files:
            file_str = str(file_path)

            if self._is_oversized(file_path):
                logger.warning(
                    "Skipping %s: exceeds max_file_size_bytes (%d)",
                    file_path,
                    self._max_file_size_bytes,
                )
                summary.files_failed += 1
                summary.failed_paths.append(file_str)
                continue

            try:
                current_hash = _file_hash(file_path)
            except FileNotFoundError:
                logger.debug("Skipping vanished file: %s", file_path)
                continue
            except OSError:
                logger.exception("Cannot read %s", file_path)
                summary.files_failed += 1
                summary.failed_paths.append(file_str)
                continue

            if indexed.get(file_str) == current_hash:
                continue

            duplicate_of = self._store.find_path_by_hash(current_hash, file_str)
            if duplicate_of is not None:
                logger.info("Already indexed: %s (matches %s)", file_path, duplicate_of)
                continue

            logger.info("Indexing %s (%d bytes)", file_path, file_path.stat().st_size)

            try:
                chunks = chunk_file(file_path, self._chunk_size, self._chunk_overlap)
            except PasswordProtectedError as exc:
                logger.warning("Skipping %s: %s", file_path, exc)
                summary.files_failed += 1
                summary.failed_paths.append(file_str)
                continue
            except ParseError:
                logger.exception("Failed to parse %s", file_path)
                summary.files_failed += 1
                summary.failed_paths.append(file_str)
                continue

            try:
                self._store.upsert_file(
                    file_str, chunks, current_hash, rebuild_bm25=False
                )
            except StoreError:
                logger.exception("Failed to upsert %s", file_path)
                summary.files_failed += 1
                summary.failed_paths.append(file_str)
                continue
            summary.files_updated += 1

        if summary.files_updated > 0:
            self._store.build_bm25()

        return summary

    def index_file(self, path: Path) -> None:
        """
        Index or re-index a single file if its content has changed.

        Skips unsupported file types and files whose hash matches the stored
        value. Called by the file watcher on create/modify events.

        Args:
            path: Path to the file to index.
        """

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return

        if self._is_oversized(path):
            logger.warning(
                "Skipping %s: exceeds max_file_size_bytes (%d)",
                path,
                self._max_file_size_bytes,
            )
            return

        file_str = str(path)

        try:
            current_hash = _file_hash(path)
        except FileNotFoundError:
            logger.debug("Skipping vanished file: %s", path)
            return
        except OSError:
            logger.exception("Cannot read %s", path)
            return

        if self._store.get_indexed_sources().get(file_str) == current_hash:
            return

        duplicate_of = self._store.find_path_by_hash(current_hash, file_str)
        if duplicate_of is not None:
            logger.info("Already indexed: %s (matches %s)", path, duplicate_of)
            return

        logger.info("Indexing %s (%d bytes)", path, path.stat().st_size)

        try:
            chunks = chunk_file(path, self._chunk_size, self._chunk_overlap)
        except PasswordProtectedError as exc:
            logger.warning("Skipping %s: %s", path, exc)
            return
        except ParseError:
            logger.exception("Failed to parse %s", path)
            return

        try:
            self._store.upsert_file(file_str, chunks, current_hash)
        except StoreError:
            logger.exception("Failed to upsert %s", path)

    def _prune_missing(self, present: list[Path], scope: list[Path]) -> int:
        """
        Delete index entries under any scope directory whose files are gone.

        Args:
            present: Paths that were found by the most recent directory walk.
            scope: Directories whose entries are eligible for pruning. Entries
                   outside these directories are left alone.

        Returns:
            Number of entries removed from the store.
        """

        present_set = {str(p) for p in present}
        resolved_scope = [s.resolve() for s in scope]
        pruned = 0

        for indexed_path in list(self._store.get_indexed_sources()):
            if indexed_path in present_set:
                continue

            path = Path(indexed_path)
            if not any(path.is_relative_to(s) for s in resolved_scope):
                continue

            try:
                self._store.delete_file(indexed_path)
            except StoreError:
                logger.exception("Failed to prune stale entry %s", indexed_path)
                continue
            logger.info("Pruned stale index entry: %s", indexed_path)
            pruned += 1

        return pruned

    def remove_file(self, path: Path) -> None:
        """
        Remove a file from the index.

        Called by the file watcher on delete events. Store failures are
        logged rather than raised so a failing delete cannot crash the
        watcher thread.

        Args:
            path: Path of the deleted file.
        """

        try:
            self._store.delete_file(str(path))
        except StoreError:
            logger.exception("Failed to remove %s from index", path)
            return
        logger.info("Removed from index: %s", path)

    def add_source_dir(self, path: Path) -> None:
        """
        Add a directory to the configured source directories.

        Args:
            path: Resolved absolute path to the new directory.
        """

        if path not in self._source_dirs:
            self._source_dirs.append(path)

    def remove_source_dir(self, path: Path) -> None:
        """
        Remove a directory from the configured source directories and delete all its indexed entries.

        Args:
            path: Resolved absolute path of the directory to remove.
        """

        resolved = path.resolve()
        self._source_dirs = [d for d in self._source_dirs if d.resolve() != resolved]
        self._store.delete_directory(path)
