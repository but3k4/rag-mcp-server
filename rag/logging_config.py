"""
Unified structured logging for the RAG server.

Configures structlog as the primary API and bridges the stdlib logging
module through the same formatter so everything. Our code, Chroma,
sentence-transformers, watchdog. Ends up in one stream with one format.

Two output modes:
    - json: one JSON object per line. Intended for production.
    - text: human-readable with colours. Intended for local dev.

Use bound_context(**fields) to attach per-request fields (trace_id,
tool name) that will appear on every log line emitted within the
context-manager's scope.
"""

from __future__ import annotations

from contextlib import contextmanager
import logging
import sys
from typing import TYPE_CHECKING, Literal

import structlog

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import IO

LogFormat = Literal["json", "text"]


def configure_logging(
    level: str = "INFO",
    log_format: LogFormat = "json",
    stream: IO[str] | None = None,
) -> None:
    """
    Configure structlog + stdlib logging to emit to a single handler.

    Safe to call multiple times. The last call wins. Tests use this to
    redirect output into a buffer.

    Args:
        level: Logging level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_format: "json" for one object per line, "text" for console output.
        stream: Destination stream. Defaults to stderr.
    """

    stream = sys.stderr if stream is None else stream

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    shared_processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        timestamper,
        structlog.processors.StackInfoRenderer(),
    ]
    # JSONRenderer cannot serialise an exc_info tuple, so format it to a
    # string first. ConsoleRenderer formats exceptions itself with a richer
    # traceback view and emits a UserWarning if format_exc_info has already
    # pre-rendered them, so skip it in that mode.
    if log_format == "json":
        shared_processors.append(structlog.processors.format_exc_info)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )

    renderer: structlog.typing.Processor
    if log_format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=stream.isatty())

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(stream)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)


@contextmanager
def bound_context(**fields: object) -> Iterator[None]:
    """
    Bind fields to the logging context for the duration of the with block.

    Any log line emitted while this block is active (including from
    stdlib-logging code) will include the bound fields. Cleared on exit
    so concurrent requests do not leak context into each other.

    Args:
        **fields: Arbitrary JSON-serialisable values to attach to log lines.
    """

    tokens = structlog.contextvars.bind_contextvars(**fields)
    try:
        yield
    finally:
        structlog.contextvars.reset_contextvars(**tokens)
