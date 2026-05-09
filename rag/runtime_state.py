"""
Persistence for the runtime directory list.

The MCP server seeds the directory list from configured source_dirs on
first run, then persists subsequent mutations made by add_directory /
remove_directory to runtime_state.json under data_dir so clients can
add or remove directories at runtime without losing them on restart.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""

    return datetime.now(tz=timezone.utc).isoformat()  # noqa: UP017


def load_directories(path: Path, seed_dirs: tuple[Path, ...]) -> list[dict[str, Any]]:
    """
    Load the persisted directory list, seeding from config on first run.

    Args:
        path: Location of runtime_state.json.
        seed_dirs: Source directories from configuration. Used only when
                   the file does not exist or cannot be parsed.

    Returns:
        List of dicts with 'path' (str) and 'last_indexed' (str | None) keys.
    """

    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return list(data.get("directories", []))
        except (json.JSONDecodeError, OSError):
            logger.warning("Could not read %s. Seeding from config/env", path)
    return [{"path": str(p), "last_indexed": None} for p in seed_dirs]


def save_directories(path: Path, directories: list[dict[str, Any]]) -> None:
    """
    Persist the directory list to runtime_state.json.

    Args:
        path: Location of runtime_state.json.
        directories: List as returned and mutated by load_directories.
    """

    path.write_text(
        json.dumps({"directories": directories}, indent=2),
        encoding="utf-8",
    )
