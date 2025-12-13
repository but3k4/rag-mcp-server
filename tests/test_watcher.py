"""Tests for the file system watcher."""

from __future__ import annotations

from pathlib import Path
import time
from unittest.mock import MagicMock

from watchdog.events import FileCreatedEvent, FileDeletedEvent, FileModifiedEvent

from rag.watcher import DirectoryWatcher, _Debouncer, _Handler

_DEBOUNCE_DELAY = 0.05
_DEBOUNCE_WAIT = 0.15  # long enough for _DEBOUNCE_DELAY to fire


class TestHandler:
    """Tests for the _Handler event handler."""

    def _make_handler(self, indexer: MagicMock, base: Path) -> _Handler:
        """Build a _Handler bound to the given indexer and base directory."""

        return _Handler(indexer, base, debouncer=None)

    def test_on_created_indexes_valid_file(self, tmp_path: Path) -> None:
        """on_created calls index_file for a supported file inside the base dir."""

        indexer = MagicMock()
        handler = self._make_handler(indexer, tmp_path)
        (tmp_path / "doc.txt").touch()
        event = FileCreatedEvent(str(tmp_path / "doc.txt"))
        handler.on_created(event)
        indexer.index_file.assert_called_once()

    def test_on_created_ignores_unsupported_extension(self, tmp_path: Path) -> None:
        """on_created does nothing for files with unsupported extensions."""

        indexer = MagicMock()
        handler = self._make_handler(indexer, tmp_path)
        event = FileCreatedEvent(str(tmp_path / "image.png"))
        handler.on_created(event)
        indexer.index_file.assert_not_called()

    def test_on_created_ignores_directory_events(self, tmp_path: Path) -> None:
        """on_created is a no-op when the event is for a directory."""

        indexer = MagicMock()
        handler = self._make_handler(indexer, tmp_path)
        event = FileCreatedEvent(str(tmp_path / "subdir"))
        event.is_directory = True
        handler.on_created(event)
        indexer.index_file.assert_not_called()

    def test_on_modified_indexes_valid_file(self, tmp_path: Path) -> None:
        """on_modified calls index_file for a supported file inside the base dir."""

        indexer = MagicMock()
        handler = self._make_handler(indexer, tmp_path)
        (tmp_path / "note.md").write_text("updated")
        event = FileModifiedEvent(str(tmp_path / "note.md"))
        handler.on_modified(event)
        indexer.index_file.assert_called_once()

    def test_on_modified_ignores_directory_events(self, tmp_path: Path) -> None:
        """on_modified is a no-op when the event is for a directory."""

        indexer = MagicMock()
        handler = self._make_handler(indexer, tmp_path)
        event = FileModifiedEvent(str(tmp_path / "subdir"))
        event.is_directory = True
        handler.on_modified(event)
        indexer.index_file.assert_not_called()

    def test_on_deleted_ignores_directory_events(self, tmp_path: Path) -> None:
        """on_deleted is a no-op when the event is for a directory."""

        indexer = MagicMock()
        handler = self._make_handler(indexer, tmp_path)
        event = FileDeletedEvent(str(tmp_path / "subdir"))
        event.is_directory = True
        handler.on_deleted(event)
        indexer.remove_file.assert_not_called()

    def test_on_deleted_removes_valid_extension(self, tmp_path: Path) -> None:
        """on_deleted calls remove_file for a supported file extension."""

        indexer = MagicMock()
        handler = self._make_handler(indexer, tmp_path)
        event = FileDeletedEvent(str(tmp_path / "old.docx"))
        handler.on_deleted(event)
        indexer.remove_file.assert_called_once()

    def test_on_deleted_ignores_unsupported_extension(self, tmp_path: Path) -> None:
        """on_deleted does nothing for unsupported extensions."""

        indexer = MagicMock()
        handler = self._make_handler(indexer, tmp_path)
        event = FileDeletedEvent(str(tmp_path / "image.png"))
        handler.on_deleted(event)
        indexer.remove_file.assert_not_called()

    def test_path_outside_base_is_rejected(self, tmp_path: Path) -> None:
        """Files whose resolved path falls outside the base directory are not indexed."""

        base = tmp_path / "base"
        base.mkdir()
        outside = tmp_path / "outside.txt"
        outside.write_text("content")
        indexer = MagicMock()
        handler = self._make_handler(indexer, base)
        event = FileCreatedEvent(str(outside))
        handler.on_created(event)
        indexer.index_file.assert_not_called()


class TestDirectoryWatcher:
    """Tests for DirectoryWatcher scheduling and lifecycle."""

    def test_watch_schedules_existing_directory(self, tmp_path: Path) -> None:
        """watch() schedules an observer for a directory that exists."""

        indexer = MagicMock()
        watcher = DirectoryWatcher(indexer)
        mock_observer = MagicMock()
        mock_observer.schedule.return_value = MagicMock()
        watcher._observer = mock_observer
        watcher.watch([tmp_path])
        mock_observer.schedule.assert_called_once()

    def test_watch_skips_missing_directory(self, tmp_path: Path) -> None:
        """watch() silently skips directories that do not exist."""

        indexer = MagicMock()
        watcher = DirectoryWatcher(indexer)
        mock_observer = MagicMock()
        watcher._observer = mock_observer
        watcher.watch([tmp_path / "nonexistent"])
        mock_observer.schedule.assert_not_called()

    def test_watch_is_idempotent(self, tmp_path: Path) -> None:
        """Calling watch() twice for the same directory only schedules once."""

        indexer = MagicMock()
        watcher = DirectoryWatcher(indexer)
        mock_observer = MagicMock()
        mock_observer.schedule.return_value = MagicMock()
        watcher._observer = mock_observer
        watcher.watch([tmp_path])
        watcher.watch([tmp_path])
        assert mock_observer.schedule.call_count == 1

    def test_unwatch_calls_unschedule(self, tmp_path: Path) -> None:
        """unwatch() unschedules the observer handle for the directory."""

        indexer = MagicMock()
        watcher = DirectoryWatcher(indexer)
        handle = MagicMock()
        mock_observer = MagicMock()
        mock_observer.schedule.return_value = handle
        watcher._observer = mock_observer
        watcher.watch([tmp_path])
        watcher.unwatch(tmp_path)
        mock_observer.unschedule.assert_called_once_with(handle)

    def test_unwatch_removes_from_handles(self, tmp_path: Path) -> None:
        """After unwatch(), the directory is no longer in the handles dict."""

        indexer = MagicMock()
        watcher = DirectoryWatcher(indexer)
        mock_observer = MagicMock()
        mock_observer.schedule.return_value = MagicMock()
        watcher._observer = mock_observer
        watcher.watch([tmp_path])
        watcher.unwatch(tmp_path)
        assert tmp_path.resolve() not in watcher._handles

    def test_unwatch_unknown_directory_is_noop(self, tmp_path: Path) -> None:
        """unwatch() for a directory that was never watched does not raise."""

        indexer = MagicMock()
        watcher = DirectoryWatcher(indexer)
        mock_observer = MagicMock()
        watcher._observer = mock_observer
        watcher.unwatch(tmp_path / "never_watched")  # must not raise
        mock_observer.unschedule.assert_not_called()

    def test_start_delegates_to_observer(self) -> None:
        """start() calls start() on the underlying observer."""

        indexer = MagicMock()
        watcher = DirectoryWatcher(indexer)
        mock_observer = MagicMock()
        watcher._observer = mock_observer
        watcher.start()
        mock_observer.start.assert_called_once()

    def test_stop_delegates_to_observer(self) -> None:
        """stop() calls stop() and join() on the underlying observer."""

        indexer = MagicMock()
        watcher = DirectoryWatcher(indexer)
        mock_observer = MagicMock()
        watcher._observer = mock_observer
        watcher.stop()
        mock_observer.stop.assert_called_once()
        mock_observer.join.assert_called_once()


class TestDebouncer:
    """Unit tests for the per-path debouncer."""

    def test_single_submit_fires_after_delay(self) -> None:
        """A lone submission runs its action once the delay elapses."""

        debouncer = _Debouncer(_DEBOUNCE_DELAY)
        action = MagicMock()
        debouncer.submit(Path("/tmp/a.txt"), action)
        time.sleep(_DEBOUNCE_WAIT)
        action.assert_called_once()

    def test_rapid_submits_collapse_to_latest_action(self) -> None:
        """Only the most recently submitted action for a path runs."""

        debouncer = _Debouncer(_DEBOUNCE_DELAY)
        first = MagicMock(name="first")
        second = MagicMock(name="second")
        third = MagicMock(name="third")
        path = Path("/tmp/a.txt")
        debouncer.submit(path, first)
        debouncer.submit(path, second)
        debouncer.submit(path, third)
        time.sleep(_DEBOUNCE_WAIT)
        first.assert_not_called()
        second.assert_not_called()
        third.assert_called_once()

    def test_different_paths_do_not_interfere(self) -> None:
        """Submissions keyed by different paths run independently."""

        debouncer = _Debouncer(_DEBOUNCE_DELAY)
        a_action = MagicMock()
        b_action = MagicMock()
        debouncer.submit(Path("/tmp/a.txt"), a_action)
        debouncer.submit(Path("/tmp/b.txt"), b_action)
        time.sleep(_DEBOUNCE_WAIT)
        a_action.assert_called_once()
        b_action.assert_called_once()

    def test_flush_cancels_pending(self) -> None:
        """flush() prevents the pending action from running."""

        debouncer = _Debouncer(_DEBOUNCE_DELAY)
        action = MagicMock()
        debouncer.submit(Path("/tmp/a.txt"), action)
        debouncer.flush()
        time.sleep(_DEBOUNCE_WAIT)
        action.assert_not_called()

    def test_exception_in_action_does_not_crash_thread(self) -> None:
        """A raising action is logged, not propagated."""

        debouncer = _Debouncer(_DEBOUNCE_DELAY)

        def _bad() -> None:
            raise RuntimeError("boom")

        followup = MagicMock()
        debouncer.submit(Path("/tmp/bad.txt"), _bad)
        time.sleep(_DEBOUNCE_WAIT)
        # Debouncer still works after the failed action.
        debouncer.submit(Path("/tmp/ok.txt"), followup)
        time.sleep(_DEBOUNCE_WAIT)
        followup.assert_called_once()


class TestWatcherDebounceIntegration:
    """Tests that DirectoryWatcher wires the debouncer correctly."""

    def test_handler_uses_debouncer_when_configured(self, tmp_path: Path) -> None:
        """With debouncing enabled, rapid modify events collapse to one call."""

        indexer = MagicMock()
        watcher = DirectoryWatcher(indexer, debounce_seconds=_DEBOUNCE_DELAY)
        (tmp_path / "doc.txt").touch()
        # Grab a real _Handler via the watcher's machinery.
        handler = _Handler(indexer, tmp_path, debouncer=watcher._debouncer)
        event = FileModifiedEvent(str(tmp_path / "doc.txt"))
        for _ in range(5):
            handler.on_modified(event)
        time.sleep(_DEBOUNCE_WAIT)
        assert indexer.index_file.call_count == 1

    def test_modify_then_delete_fires_only_remove(self, tmp_path: Path) -> None:
        """A delete event replaces a pending modify."""

        indexer = MagicMock()
        watcher = DirectoryWatcher(indexer, debounce_seconds=_DEBOUNCE_DELAY)
        (tmp_path / "doc.txt").touch()
        handler = _Handler(indexer, tmp_path, debouncer=watcher._debouncer)
        handler.on_modified(FileModifiedEvent(str(tmp_path / "doc.txt")))
        handler.on_deleted(FileDeletedEvent(str(tmp_path / "doc.txt")))
        time.sleep(_DEBOUNCE_WAIT)
        indexer.index_file.assert_not_called()
        indexer.remove_file.assert_called_once()

    def test_debouncer_disabled_fires_synchronously(self, tmp_path: Path) -> None:
        """debounce_seconds=0 keeps the old synchronous behaviour."""

        indexer = MagicMock()
        watcher = DirectoryWatcher(indexer, debounce_seconds=0.0)
        assert watcher._debouncer is None
        (tmp_path / "doc.txt").touch()
        handler = _Handler(indexer, tmp_path, debouncer=None)
        handler.on_modified(FileModifiedEvent(str(tmp_path / "doc.txt")))
        indexer.index_file.assert_called_once()

    def test_stop_flushes_debouncer(self) -> None:
        """stop() cancels pending debounced actions before joining observer."""

        indexer = MagicMock()
        watcher = DirectoryWatcher(indexer, debounce_seconds=_DEBOUNCE_DELAY)
        mock_observer = MagicMock()
        watcher._observer = mock_observer
        action = MagicMock()
        assert watcher._debouncer is not None
        watcher._debouncer.submit(Path("/tmp/a.txt"), action)
        watcher.stop()
        time.sleep(_DEBOUNCE_WAIT)
        action.assert_not_called()
