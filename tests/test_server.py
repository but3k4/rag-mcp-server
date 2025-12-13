"""Unit tests for pure helpers in rag.server.

The MCP tool bodies themselves are not unit-tested here. They require a
running FastMCP instance. These tests cover the extractable helpers that
feed those tools.
"""

from __future__ import annotations

import signal
from typing import TYPE_CHECKING

import pytest

from rag.server import (
    _install_shutdown_handlers,
    _is_path_allowed,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestIsPathAllowed:
    """Tests for the allow-list check used by add_directory."""

    def test_empty_allow_list_permits_everything(self, tmp_path: Path) -> None:
        """An empty tuple means no restriction. Any path is allowed."""

        assert _is_path_allowed(tmp_path / "anywhere", ()) is True

    def test_path_under_base_is_allowed(self, tmp_path: Path) -> None:
        """A descendant of an allowed base is permitted."""

        base = tmp_path / "allowed"
        base.mkdir()
        sub = base / "nested" / "deep"
        sub.mkdir(parents=True)
        assert _is_path_allowed(sub.resolve(), (base.resolve(),)) is True

    def test_base_itself_is_allowed(self, tmp_path: Path) -> None:
        """The base directory itself satisfies the check."""

        base = tmp_path / "allowed"
        base.mkdir()
        assert _is_path_allowed(base.resolve(), (base.resolve(),)) is True

    def test_sibling_is_rejected(self, tmp_path: Path) -> None:
        """A sibling directory outside every base is rejected."""

        allowed = tmp_path / "allowed"
        allowed.mkdir()
        sibling = tmp_path / "other"
        sibling.mkdir()
        assert _is_path_allowed(sibling.resolve(), (allowed.resolve(),)) is False

    def test_prefix_trap_is_rejected(self, tmp_path: Path) -> None:
        """A path with a matching name prefix but different directory is rejected."""

        base = tmp_path / "foo"
        base.mkdir()
        trap = tmp_path / "foobar"
        trap.mkdir()
        assert _is_path_allowed(trap.resolve(), (base.resolve(),)) is False

    def test_multiple_bases_any_match_allows(self, tmp_path: Path) -> None:
        """Membership in any one of several allowed bases permits the path."""

        base_a = tmp_path / "a"
        base_b = tmp_path / "b"
        base_a.mkdir()
        base_b.mkdir()
        under_b = base_b / "nested"
        under_b.mkdir()
        bases = (base_a.resolve(), base_b.resolve())
        assert _is_path_allowed(under_b.resolve(), bases) is True


class TestShutdown:
    """Tests for signal handling used by the graceful-shutdown path."""

    def test_sigterm_handler_raises_keyboard_interrupt(self) -> None:
        """SIGTERM is rewired to raise KeyboardInterrupt, sharing the SIGINT path."""

        try:
            _install_shutdown_handlers()
            handler = signal.getsignal(signal.SIGTERM)
            assert callable(handler)
            with pytest.raises(KeyboardInterrupt):
                handler(signal.SIGTERM, None)
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
