"""Tests for the exception hierarchy in rag.errors."""

from __future__ import annotations

from rag.config import ConfigError
from rag.errors import IndexingError, RagError, StoreError
from rag.parsers import ParseError


class TestHierarchy:
    """Every expected RAG failure inherits from RagError for uniform catch."""

    def test_store_error_is_rag_error(self) -> None:
        """StoreError can be caught via the RagError base class."""

        assert issubclass(StoreError, RagError)

    def test_indexing_error_is_rag_error(self) -> None:
        """IndexingError can be caught via the RagError base class."""

        assert issubclass(IndexingError, RagError)

    def test_config_error_is_rag_error(self) -> None:
        """ConfigError is reparented under RagError without losing ValueError."""

        assert issubclass(ConfigError, RagError)
        assert issubclass(ConfigError, ValueError)

    def test_parse_error_is_rag_error(self) -> None:
        """ParseError is reparented under RagError."""

        assert issubclass(ParseError, RagError)

    def test_single_except_catches_every_category(self) -> None:
        """A caller can catch everything via RagError."""

        for cls in (StoreError, IndexingError, ConfigError, ParseError):
            try:
                raise cls("boom")
            except RagError:
                pass
            else:  # pragma: no cover (exercised only on regression)
                raise AssertionError(f"{cls.__name__} did not match RagError")
