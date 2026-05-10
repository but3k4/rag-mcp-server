"""Tests for rag.parsers section-aware chunking functions."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from rag.parsers import (
    SUPPORTED_EXTENSIONS,
    ParseError,
    PasswordProtectedError,
    chunk_file,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestChunkFileDispatch:
    """Tests for the dispatch logic in chunk_file."""

    def test_unsupported_extension_raises_parse_error(self, tmp_path: Path) -> None:
        """chunk_file raises ParseError for an unknown file extension."""

        f = tmp_path / "image.png"
        f.write_bytes(b"\x89PNG")
        with pytest.raises(ParseError, match="Unsupported file type"):
            chunk_file(f)

    def test_supported_extensions_set(self) -> None:
        """SUPPORTED_EXTENSIONS contains exactly the eight documented types."""

        expected = frozenset(
            {".txt", ".md", ".pdf", ".docx", ".pptx", ".csv", ".xlsx", ".xml"}
        )
        assert expected == SUPPORTED_EXTENSIONS

    def test_returns_list_of_dicts(self, tmp_path: Path) -> None:
        """chunk_file returns a list of dicts with required keys."""

        f = tmp_path / "note.txt"
        f.write_text("hello world", encoding="utf-8")
        result = chunk_file(f)
        assert isinstance(result, list)
        assert len(result) >= 1
        for chunk in result:
            assert "text" in chunk
            assert "section" in chunk
            assert "chunk_index" in chunk

    def test_chunk_indices_are_sequential(self, tmp_path: Path) -> None:
        """chunk_index values are contiguous starting from 0."""

        f = tmp_path / "note.txt"
        f.write_text("hello world", encoding="utf-8")
        result = chunk_file(f)
        assert [c["chunk_index"] for c in result] == list(range(len(result)))

    def test_custom_chunk_size_splits_text(self, tmp_path: Path) -> None:
        """A small chunk_size produces more chunks than the default."""

        f = tmp_path / "long.txt"
        # Paragraphs separated by blank lines give _split_text break points.
        f.write_text("\n\n".join(["word " * 10] * 20), encoding="utf-8")
        small = chunk_file(f, chunk_size=100, chunk_overlap=0)
        default = chunk_file(f)
        assert len(small) > len(default)


class TestChunkTxt:
    """Tests for plain-text and Markdown chunking."""

    def test_txt_contains_content(self, tmp_path: Path) -> None:
        """Content from a .txt file appears in chunk text."""

        f = tmp_path / "note.txt"
        f.write_text("hello world", encoding="utf-8")
        result = chunk_file(f)
        assert any("hello world" in c["text"] for c in result)

    def test_md_content_present(self, tmp_path: Path) -> None:
        """Markdown header and body text are present across chunks."""

        f = tmp_path / "note.md"
        f.write_text("# Heading\nBody text.", encoding="utf-8")
        result = chunk_file(f)
        all_text = " ".join(c["text"] for c in result)
        assert "Heading" in all_text
        assert "Body text" in all_text

    def test_unreadable_file_raises_parse_error(self, tmp_path: Path) -> None:
        """A file that does not exist raises ParseError."""

        f = tmp_path / "gone.txt"
        with pytest.raises(ParseError):
            chunk_file(f)

    def test_unreadable_markdown_raises_parse_error(self, tmp_path: Path) -> None:
        """A missing .md file raises ParseError with Cannot read wording."""

        f = tmp_path / "gone.md"
        with pytest.raises(ParseError, match="Cannot read"):
            chunk_file(f)


class TestChunkWithDocling:
    """
    Tests for the unified Docling-based parser (PDF, DOCX, PPTX, XLSX).

    Mocks the DocumentConverter at the module boundary so tests are fast
    and independent of the actual Docling models.
    """

    def _install_mock_converter(
        self, monkeypatch: pytest.MonkeyPatch, markdown: str
    ) -> MagicMock:
        """
        Override the cached converter with a mock that returns markdown.

        Setting an attribute on the _DoclingPipeline instance writes to
        instance __dict__ and masks the cached_property descriptor, so
        subsequent reads return the mock without ever building a real
        DocumentConverter. monkeypatch reverts this after the test.
        """

        import rag.parsers as parsers_mod  # noqa: PLC0415

        converter = MagicMock(name="DocumentConverter")
        document = MagicMock(name="DoclingDocument")
        document.export_to_markdown.return_value = markdown
        result = MagicMock(name="ConversionResult", document=document)
        converter.convert.return_value = result
        monkeypatch.setattr(parsers_mod._DOCLING_PIPELINE, "converter", converter)
        return converter

    @pytest.mark.parametrize("ext", [".pdf", ".docx", ".pptx", ".xlsx"])
    def test_dispatches_each_format_to_docling(
        self, ext: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """All four structured formats route through Docling."""

        converter = self._install_mock_converter(monkeypatch, "# Heading\n\nbody text")
        f = tmp_path / f"doc{ext}"
        f.write_bytes(b"placeholder")
        result = chunk_file(f)
        converter.convert.assert_called_once()
        assert any("body text" in c["text"] for c in result)

    def test_markdown_headings_become_sections(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ATX headers in Docling output are used as section labels."""

        markdown = (
            "# Introduction\n\n"
            "intro body content here\n\n"
            "## Details\n\n"
            "details body content here\n"
        )
        self._install_mock_converter(monkeypatch, markdown)
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"placeholder")
        result = chunk_file(f)
        sections = {c["section"] for c in result}
        assert "# Introduction" in sections
        assert "## Details" in sections

    def test_conversion_error_wrapped_as_parse_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A docling ConversionError is wrapped as ParseError."""

        from docling.exceptions import ConversionError  # noqa: PLC0415

        import rag.parsers as parsers_mod  # noqa: PLC0415

        converter = MagicMock(name="DocumentConverter")
        converter.convert.side_effect = ConversionError("simulated")
        monkeypatch.setattr(parsers_mod._DOCLING_PIPELINE, "converter", converter)
        f = tmp_path / "bad.pdf"
        f.write_bytes(b"placeholder")
        with pytest.raises(ParseError, match="Cannot parse"):
            chunk_file(f)

    def test_password_protected_pdf_raises_password_protected_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Password-protected PDFs surface as PasswordProtectedError.

        Pre-check is performed by _is_pdf_password_protected before the
        file is handed to Docling. Patching that helper lets us simulate
        a password-protected PDF without crafting a real one.
        """

        import rag.parsers as parsers_mod  # noqa: PLC0415

        monkeypatch.setattr(parsers_mod, "_is_pdf_password_protected", lambda _: True)
        f = tmp_path / "locked.pdf"
        f.write_bytes(b"placeholder")
        with pytest.raises(PasswordProtectedError, match="password-protected"):
            chunk_file(f)

    def test_pre_check_only_runs_for_pdf(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-PDF (e.g. .docx) should not invoke the password pre-check."""

        import rag.parsers as parsers_mod  # noqa: PLC0415

        calls: list[Path] = []

        def _spy(p: Path) -> bool:
            calls.append(p)
            return False

        monkeypatch.setattr(parsers_mod, "_is_pdf_password_protected", _spy)
        self._install_mock_converter(monkeypatch, "# H\n\nbody")
        f = tmp_path / "doc.docx"
        f.write_bytes(b"placeholder")
        chunk_file(f)
        assert calls == []

    def test_missing_docling_raises_parse_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If docling is not installed, parsing raises ParseError."""

        import rag.parsers as parsers_mod  # noqa: PLC0415

        f = tmp_path / "x.pdf"
        f.write_bytes(b"placeholder")
        monkeypatch.setitem(sys.modules, "docling.exceptions", None)
        # Replace the singleton so any cached converter from a prior test
        # is dropped; the missing-import check fires before the pipeline
        # is consulted, but a fresh instance keeps the test airtight.
        monkeypatch.setattr(
            parsers_mod, "_DOCLING_PIPELINE", parsers_mod._DoclingPipeline()
        )
        with pytest.raises(ParseError, match="docling is not installed"):
            chunk_file(f)


class TestChunkCsv:
    """Tests for CSV chunking."""

    def test_csv_rows_in_chunk_text(self, tmp_path: Path) -> None:
        """Each CSV row appears formatted in chunk text."""

        f = tmp_path / "data.csv"
        f.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
        result = chunk_file(f)
        all_text = " ".join(c["text"] for c in result)
        assert "a | b | c" in all_text
        assert "1 | 2 | 3" in all_text

    def test_empty_csv_returns_empty_list(self, tmp_path: Path) -> None:
        """An empty CSV file returns an empty list."""

        f = tmp_path / "empty.csv"
        f.write_text("", encoding="utf-8")
        result = chunk_file(f)
        assert result == []

    def test_unreadable_csv_raises_parse_error(self, tmp_path: Path) -> None:
        """A CSV that does not exist raises ParseError."""

        f = tmp_path / "missing.csv"
        with pytest.raises(ParseError, match="Cannot read CSV"):
            chunk_file(f)

    def test_csv_error_raises_parse_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A csv.Error during parsing is wrapped as ParseError."""

        import csv as _csv  # noqa: PLC0415

        f = tmp_path / "broken.csv"
        f.write_text("a,b\n", encoding="utf-8")

        def _boom(*_args: object, **_kwargs: object) -> object:
            """Monkeypatch stand-in that always raises csv.Error."""

            raise _csv.Error("simulated")

        monkeypatch.setattr(_csv, "reader", _boom)
        with pytest.raises(ParseError, match="Cannot parse CSV"):
            chunk_file(f)


class TestChunkXml:
    """Tests for XML chunking."""

    def test_xml_child_text_in_chunks(self, tmp_path: Path) -> None:
        """Text from child elements appears in chunk text."""

        f = tmp_path / "doc.xml"
        f.write_text(
            "<root><section>Hello XML world</section></root>",
            encoding="utf-8",
        )
        result = chunk_file(f)
        assert any("Hello XML world" in c["text"] for c in result)

    def test_xml_child_tag_used_as_section(self, tmp_path: Path) -> None:
        """The direct child tag name is used as the section label."""

        f = tmp_path / "doc.xml"
        f.write_text(
            "<root><chapter>Chapter text</chapter></root>",
            encoding="utf-8",
        )
        result = chunk_file(f)
        assert any(c["section"] == "chapter" for c in result)

    def test_xml_multiple_children_produce_sections(self, tmp_path: Path) -> None:
        """Each non-empty child element produces at least one chunk."""

        f = tmp_path / "doc.xml"
        f.write_text(
            "<root><a>Alpha</a><b>Beta</b></root>",
            encoding="utf-8",
        )
        result = chunk_file(f)
        sections = {c["section"] for c in result}
        assert "a" in sections
        assert "b" in sections

    def test_xml_flat_document_returns_chunks(self, tmp_path: Path) -> None:
        """A document with no child elements still returns chunks from root text."""

        f = tmp_path / "flat.xml"
        f.write_text("<root>Just some text here</root>", encoding="utf-8")
        result = chunk_file(f)
        assert len(result) >= 1
        assert any("Just some text here" in c["text"] for c in result)

    def test_invalid_xml_raises_parse_error(self, tmp_path: Path) -> None:
        """Malformed XML raises ParseError."""

        f = tmp_path / "bad.xml"
        f.write_text("<unclosed>", encoding="utf-8")
        with pytest.raises(ParseError, match="Cannot parse XML"):
            chunk_file(f)

    def test_xml_skips_comment_children(self, tmp_path: Path) -> None:
        """Comment and processing-instruction children are skipped."""

        f = tmp_path / "comments.xml"
        f.write_text(
            "<root><!-- skip me --><section>Real text</section></root>",
            encoding="utf-8",
        )
        result = chunk_file(f)
        assert any(c["section"] == "section" for c in result)

    def test_xml_skips_empty_children(self, tmp_path: Path) -> None:
        """Child elements with no text are skipped."""

        f = tmp_path / "empty_child.xml"
        f.write_text(
            "<root><empty></empty><full>Has text</full></root>",
            encoding="utf-8",
        )
        result = chunk_file(f)
        sections = {c["section"] for c in result}
        assert "empty" not in sections
        assert "full" in sections

    def test_missing_lxml_raises_parse_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If lxml is not installed, XML parsing raises ParseError."""

        f = tmp_path / "x.xml"
        f.write_text("<root/>", encoding="utf-8")
        monkeypatch.setitem(sys.modules, "lxml", None)
        monkeypatch.setitem(sys.modules, "lxml.etree", None)
        with pytest.raises(ParseError, match="lxml is not installed"):
            chunk_file(f)
