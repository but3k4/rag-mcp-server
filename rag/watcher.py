"""Watchdog-based file system watcher for automatic incremental re-indexing."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import threading
from typing import TYPE_CHECKING, Any

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver

from rag.parsers import SUPPORTED_EXTENSIONS

if TYPE_CHECKING:
    from collections.abc import Callable

    from rag.indexer import Indexer

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _Debouncer:
    """
    Per-path timer coalescer.

    Rapid events on the same path collapse into a single deferred action.
    When an event arrives, any pending timer for that path is cancelled
    and a new one is scheduled with the configured delay. The most recently
    submitted action wins. So a modify-then-delete sequence fires only
    remove_file, not index_file.
    """

    def __init__(self, delay_seconds: float) -> None:
        """
        Initialise the debouncer.

        Args:
            delay_seconds: How long to wait after the last event for a
                given path before firing its action. Must be >= 0.
        """

        self._delay = delay_seconds
        self._pending: dict[Path, _PendingEntry] = {}
        self._lock = threading.Lock()

    def submit(self, path: Path, action: Callable[[], None]) -> None:
        """
        Schedule action for path, replacing any previous pending action.

        Args:
            path: The file path keying the debounce slot.
            action: Zero-arg callable to invoke once the delay elapses
                with no further submissions for this path.
        """

        timer = threading.Timer(self._delay, self._fire, args=(path,))
        timer.daemon = True

        with self._lock:
            existing = self._pending.pop(path, None)
            self._pending[path] = _PendingEntry(timer=timer, action=action)

        if existing is not None:
            existing.timer.cancel()
        timer.start()

    def _fire(self, path: Path) -> None:
        """Invoke the pending action for path. Runs on the Timer thread."""

        with self._lock:
            entry = self._pending.pop(path, None)

        if entry is None:
            return

        try:
            entry.action()
        except Exception:
            logger.exception("Debounced action failed for %s", path)

    def flush(self) -> None:
        """Cancel all pending timers. Safe to call from any thread."""

        with self._lock:
            timers = [e.timer for e in self._pending.values()]
            self._pending.clear()

        for t in timers:
            t.cancel()


class _PendingEntry:
    """Holds the timer and action for one pending debounced submission."""

    __slots__ = ("action", "timer")

    def __init__(self, timer: threading.Timer, action: Callable[[], None]) -> None:
        """Store the timer handle and the deferred callable."""

        self.timer = timer
        self.action = action


class _Handler(FileSystemEventHandler):
    """Handles file system events for a single watched directory."""

    def __init__(
        self,
        indexer: Indexer,
        base: Path,
        debouncer: _Debouncer | None,
    ) -> None:
        """
        Initialise the event handler for the given base directory.

        Args:
            indexer: The shared Indexer instance to call on file events.
            base: Resolved absolute path of the watched root directory.
            debouncer: Coalescer for rapid events. When None, actions fire
                       synchronously. Matching the pre-debounce behaviour when
                       watcher_debounce_seconds is 0.
        """

        super().__init__()
        self._indexer = indexer
        self._base = base.resolve()
        self._debouncer = debouncer

    def _is_valid(self, raw_path: str | bytes) -> bool:
        """Return True if the path has a supported extension and lies under base."""

        path = Path(os.fsdecode(raw_path))
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return False

        try:
            path.resolve().relative_to(self._base)
        except ValueError:
            return False

        return True

    def _dispatch(self, path: Path, action: Callable[[], None]) -> None:
        """Run action now or schedule it via the debouncer if configured."""

        if self._debouncer is None:
            action()
        else:
            self._debouncer.submit(path, action)

    def on_created(self, event: FileSystemEvent) -> None:
        """Index a newly created file."""

        if event.is_directory:
            return

        if self._is_valid(event.src_path):
            path = Path(os.fsdecode(event.src_path))
            self._dispatch(path, lambda: self._indexer.index_file(path))

    def on_modified(self, event: FileSystemEvent) -> None:
        """Re-index a modified file."""

        if event.is_directory:
            return

        if self._is_valid(event.src_path):
            path = Path(os.fsdecode(event.src_path))
            self._dispatch(path, lambda: self._indexer.index_file(path))

    def on_deleted(self, event: FileSystemEvent) -> None:
        """Remove a deleted file from the index."""

        if event.is_directory:
            return

        path = Path(os.fsdecode(event.src_path))
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            self._dispatch(path, lambda: self._indexer.remove_file(path))


class DirectoryWatcher:
    """Watches configured directories and triggers re-indexing on file changes."""

    def __init__(
        self,
        indexer: Indexer,
        use_polling: bool = False,
        debounce_seconds: float = 0.0,
    ) -> None:
        """
        Initialise the watcher with the shared Indexer instance.

        Args:
            indexer: The Indexer instance to call when files change.
            use_polling: Use a polling observer instead of the platform-native
                         backend. Required for bind-mounted directories on
                         Docker Desktop (macOS/Windows), where
                         inotify/FSEvents do not propagate through the virtual
                         machine.
            debounce_seconds: Coalesce rapid events on the same path by this
                              interval. Editor saves often fire multiple create
                              / modify events within milliseconds. Debouncing
                              prevents each from triggering its own re-index.
                              Zero disables.
        """

        self._indexer = indexer
        self._observer: Any = PollingObserver() if use_polling else Observer()
        self._handles: dict[Path, Any] = {}
        self._debouncer = _Debouncer(debounce_seconds) if debounce_seconds > 0 else None

    def watch(self, directories: list[Path]) -> None:
        """
        Schedule watching for each directory not already being watched.

        Directories that do not exist are skipped with a warning. Safe to call
        before start(). Schedules are queued and activated when the observer
        thread starts.

        Args:
            directories: List of resolved absolute directory paths to watch.
        """

        for d in directories:
            resolved = d.resolve()
            if resolved in self._handles:
                continue

            if not resolved.is_dir():
                logger.warning("Cannot watch missing directory: %s", d)
                continue

            handler = _Handler(self._indexer, resolved, self._debouncer)
            handle = self._observer.schedule(handler, str(resolved), recursive=True)
            self._handles[resolved] = handle
            logger.info("Watching: %s", resolved)

    def unwatch(self, directory: Path) -> None:
        """
        Stop watching a directory.

        Args:
            directory: Resolved absolute path to stop watching.
        """

        resolved = directory.resolve()
        handle = self._handles.pop(resolved, None)
        if handle is not None:
            self._observer.unschedule(handle)
            logger.info("Stopped watching: %s", resolved)

    def start(self) -> None:
        """Start the observer thread."""

        self._observer.start()
        logger.info("File watcher started")

    def stop(self) -> None:
        """Stop the observer thread, cancel pending debounces, and join."""

        self._observer.stop()
        if self._debouncer is not None:
            self._debouncer.flush()
        self._observer.join()
        logger.info("File watcher stopped")

    def is_running(self) -> bool:
        """Return True if the observer thread is alive and accepting events."""

        is_alive = self._observer.is_alive()
        return bool(is_alive)
