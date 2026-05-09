"""Tests for the directory-state persistence helpers in rag.runtime_state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag.runtime_state import load_directories, now_iso, save_directories


class TestLoadDirectories:
    """Tests for load_directories."""

    def test_missing_file_seeds_from_config(self, tmp_path: Path) -> None:
        """When the file does not exist, seed_dirs becomes the directory list."""

        path = tmp_path / "runtime_state.json"
        seed = (Path("/a"), Path("/b"))
        result = load_directories(path, seed)
        assert result == [
            {"path": "/a", "last_indexed": None},
            {"path": "/b", "last_indexed": None},
        ]

    def test_missing_file_with_no_seed_returns_empty(self, tmp_path: Path) -> None:
        """A missing file and empty seed produce an empty list."""

        path = tmp_path / "runtime_state.json"
        assert load_directories(path, ()) == []

    def test_existing_file_returns_persisted_directories(self, tmp_path: Path) -> None:
        """A valid runtime_state.json is parsed and returned as-is."""

        path = tmp_path / "runtime_state.json"
        payload = {
            "directories": [
                {"path": "/x", "last_indexed": "2026-01-01T00:00:00+00:00"},
            ]
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        result = load_directories(path, (Path("/seed"),))
        assert result == payload["directories"]

    def test_corrupt_file_falls_back_to_seed(self, tmp_path: Path) -> None:
        """An unparseable file falls back to seed_dirs without raising."""

        path = tmp_path / "runtime_state.json"
        path.write_text("not valid json", encoding="utf-8")
        result = load_directories(path, (Path("/seed"),))
        assert result == [{"path": "/seed", "last_indexed": None}]

    def test_missing_directories_key_returns_empty(self, tmp_path: Path) -> None:
        """A valid JSON file without the 'directories' key yields an empty list."""

        path = tmp_path / "runtime_state.json"
        path.write_text(json.dumps({"other": []}), encoding="utf-8")
        assert load_directories(path, ()) == []


class TestSaveDirectories:
    """Tests for save_directories."""

    def test_writes_round_trippable_json(self, tmp_path: Path) -> None:
        """Save followed by load returns the same directory list."""

        path = tmp_path / "runtime_state.json"
        directories: list[dict[str, Any]] = [
            {"path": "/a", "last_indexed": None},
            {"path": "/b", "last_indexed": "2026-05-06T12:00:00+00:00"},
        ]
        save_directories(path, directories)
        assert load_directories(path, ()) == directories

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        """A second save replaces the previous contents in place."""

        path = tmp_path / "runtime_state.json"
        old: list[dict[str, Any]] = [{"path": "/old", "last_indexed": None}]
        new: list[dict[str, Any]] = [{"path": "/new", "last_indexed": None}]
        save_directories(path, old)
        save_directories(path, new)
        assert load_directories(path, ()) == new


class TestNowIso:
    """Tests for now_iso."""

    def test_returns_string_with_offset(self) -> None:
        """now_iso returns an ISO-8601 string with a UTC offset suffix."""

        value = now_iso()
        assert isinstance(value, str)
        # Either +00:00 or Z. datetime.isoformat uses +00:00 for tz=utc.
        assert value.endswith("+00:00")
