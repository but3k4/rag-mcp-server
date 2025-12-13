"""
Typed exception hierarchy for the RAG server.

A small, flat hierarchy so callers can catch by category without needing to
know every concrete subclass. Concrete errors defined elsewhere (ConfigError
in rag.config, ParseError in rag.parsers) inherit from RagError so a single
except RagError handler at the server boundary can log any expected
failure without catching every Exception.
"""

from __future__ import annotations


class RagError(Exception):
    """Base class for all expected failures raised by rag.* modules."""


class StoreError(RagError):
    """
    Raised when the vector store or its metadata DB fails an operation.

    Wraps the underlying backend error (ChromaError, sqlite3.Error, etc.)
    via exception chaining. Callers that want to tolerate individual-op
    failures (e.g. the indexer's batch loop) should catch this type
    explicitly and decide whether to skip, retry, or abort.
    """


class IndexingError(RagError):
    """Raised when an indexing operation cannot proceed at all.

    Per-file failures during a batch run are counted in IndexSummary rather
    than raised. This type is reserved for failures that prevent the run
    from starting or completing at the batch level.
    """
