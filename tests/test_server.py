"""
Unit tests for the lifecycle helpers in rag.server.

Tool bodies live in rag.tools and are tested in tests/test_tools.py.
This module covers the shutdown signal handler.
"""

from __future__ import annotations

import signal

import pytest

from rag.server import _install_shutdown_handlers


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
