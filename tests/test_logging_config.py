"""Tests for rag.logging_config: JSON shape, context propagation, stdlib bridge."""

from __future__ import annotations

import io
import json
import logging

import pytest
import structlog

from rag.logging_config import bound_context, configure_logging


@pytest.fixture(autouse=True)
def reset_logging() -> None:
    """Reset stdlib root handlers between tests to avoid bleed-through."""

    root = logging.getLogger()
    root.handlers = []
    root.setLevel(logging.NOTSET)
    structlog.contextvars.clear_contextvars()


def _last_json_line(buffer: io.StringIO) -> dict[str, object]:
    """Return the last non-empty line in buffer, parsed as JSON."""

    lines = [line for line in buffer.getvalue().splitlines() if line.strip()]
    record: dict[str, object] = json.loads(lines[-1])
    return record


class TestJsonOutput:
    """Tests for JSON-format output."""

    def test_basic_structlog_event_is_rendered_as_json(self) -> None:
        """A structlog.info call emits a single JSON object with standard fields."""

        expected_count = 3
        buf = io.StringIO()
        configure_logging(level="INFO", log_format="json", stream=buf)
        structlog.get_logger("rag.test").info("hello_event", count=expected_count)

        record = _last_json_line(buf)
        assert record["event"] == "hello_event"
        assert record["level"] == "info"
        assert record["logger"] == "rag.test"
        assert record["count"] == expected_count
        assert "timestamp" in record

    def test_stdlib_info_flows_through_the_same_formatter(self) -> None:
        """A stdlib logger.info ends up in the same JSON stream."""

        buf = io.StringIO()
        configure_logging(level="INFO", log_format="json", stream=buf)
        logging.getLogger("rag.test").info("stdlib message")

        record = _last_json_line(buf)
        assert record["event"] == "stdlib message"
        assert record["level"] == "info"
        assert record["logger"] == "rag.test"

    def test_exception_trace_is_included(self) -> None:
        """logger.exception renders the traceback into the record."""

        buf = io.StringIO()
        configure_logging(level="INFO", log_format="json", stream=buf)

        def _raise() -> None:
            raise ValueError("boom")

        try:
            _raise()
        except ValueError:
            logging.getLogger("rag.test").exception("caught")

        record = _last_json_line(buf)
        assert "ValueError" in str(record.get("exception", ""))


class TestContextPropagation:
    """Tests for bound_context and context-var inheritance."""

    def test_context_fields_appear_on_inner_logs(self) -> None:
        """Fields bound with bound_context decorate every log line inside."""

        buf = io.StringIO()
        configure_logging(level="INFO", log_format="json", stream=buf)

        with bound_context(trace_id="abc123", tool="search_docs"):
            structlog.get_logger().info("inner")

        record = _last_json_line(buf)
        assert record["trace_id"] == "abc123"
        assert record["tool"] == "search_docs"

    def test_context_is_cleared_on_exit(self) -> None:
        """Log lines emitted outside the with block do not see the fields."""

        buf = io.StringIO()
        configure_logging(level="INFO", log_format="json", stream=buf)

        with bound_context(trace_id="abc123"):
            pass
        structlog.get_logger().info("after")

        record = _last_json_line(buf)
        assert "trace_id" not in record

    def test_context_propagates_to_stdlib_loggers(self) -> None:
        """A stdlib logger.info inside bound_context also gets the fields."""

        buf = io.StringIO()
        configure_logging(level="INFO", log_format="json", stream=buf)

        with bound_context(trace_id="stdlib-aware"):
            logging.getLogger("rag.test").warning("stdlib inside context")

        record = _last_json_line(buf)
        assert record["trace_id"] == "stdlib-aware"
        assert record["level"] == "warning"


class TestTextMode:
    """Tests for human-readable console output."""

    def test_text_mode_does_not_emit_json(self) -> None:
        """Text mode produces something other than parseable JSON."""

        buf = io.StringIO()
        configure_logging(level="INFO", log_format="text", stream=buf)
        structlog.get_logger("rag.test").info("hello_event", count=3)

        output = buf.getvalue()
        assert "hello_event" in output
        # The first non-whitespace character is a timestamp digit, not '{'.
        assert not output.lstrip().startswith("{")
