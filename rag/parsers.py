"""
Text extraction and section-aware chunking for supported file types.

Each parser returns a list of chunk dicts with three keys:
  - text: the chunk content (stripped)
  - section: section title (header, slide title, page label, or empty string)
  - chunk_index: sequential integer within the file
"""

from __future__ import annotations

import csv
from functools import cached_property
import logging
import re
from typing import TYPE_CHECKING, Any

from rag.errors import RagError

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

Chunk = dict[str, Any]

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".txt", ".md", ".pdf", ".docx", ".pptx", ".csv", ".xlsx", ".xml"}
)

# Identifier for the parser pipeline. Bump when the parsing or chunking
# behaviour changes in a way that invalidates previously embedded chunks.
# VectorStore reads this and forces a full reindex when it differs from
# the value recorded in the metadata DB.
PARSER_VERSION = "docling-v2-tesseract-ocr"

DEFAULT_CHUNK_SIZE = 1400  # characters
DEFAULT_CHUNK_OVERLAP = 150


class ParseError(RagError):
    """Raised when a file cannot be parsed into chunks."""


class PasswordProtectedError(ParseError):
    """
    Raised when a PDF cannot be parsed because it is password-protected.

    Distinct from generic ParseError so the indexer can log it cleanly
    (a one-line warning, no traceback) instead of treating it as an
    unexpected failure. The user has no way to fix this from the
    indexer's side — the file genuinely cannot be parsed without the
    password.
    """


def _is_pdf_password_protected(path: Path) -> bool:
    """
    Return True if a PDF file requires a password to open.

    Docling cannot be relied on to surface this: its document loader
    catches the underlying pypdfium2 PdfiumError, logs it, swallows it,
    and later raises a generic ConversionError("Input document ... is
    not valid") with no chained cause. Walking __cause__/__context__
    finds nothing. So we open the file directly with pypdfium2 first;
    the check is a header read, fast even on large PDFs.

    Best-effort: any failure to load pypdfium2 or unexpected exception
    returns False so Docling still gets to try (and surface its own
    error if appropriate).
    """

    try:
        import pypdfium2  # noqa: PLC0415
        from pypdfium2._helpers.misc import PdfiumError  # noqa: PLC0415
    except ImportError:
        return False

    try:
        doc = pypdfium2.PdfDocument(str(path))
        doc.close()
    except PdfiumError as exc:
        return "password" in str(exc).lower()
    except (OSError, ValueError):
        return False
    return False


def chunk_file(
    path: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """
    Parse a file and return section-aware text chunks.

    Args:
        path: Absolute path to the file to parse.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Characters of overlap between consecutive chunks.

    Returns:
        List of chunk dicts, each with text, section, and chunk_index keys.
        Returns an empty list for files with no extractable text.

    Raises:
        ParseError: If the file type is unsupported or extraction fails.
    """

    suffix = path.suffix.lower()
    dispatch = {
        ".txt": _chunk_text,
        ".md": _chunk_markdown,
        ".pdf": _chunk_with_docling,
        ".docx": _chunk_with_docling,
        ".pptx": _chunk_with_docling,
        ".xlsx": _chunk_with_docling,
        ".csv": _chunk_csv,
        ".xml": _chunk_xml,
    }

    fn = dispatch.get(suffix)
    if fn is None:
        raise ParseError(f"Unsupported file type: {suffix}")

    return fn(path, chunk_size, chunk_overlap)


def _make_chunk(text: str, section: str, index: int) -> Chunk:
    """Build a Chunk dict with stripped text, section title, and index."""

    return {"text": text.strip(), "section": section, "chunk_index": index}


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Split text into overlapping chunks at paragraph boundaries.

    Prefers splitting on double-newlines rather than mid-sentence. Each chunk
    is at most chunk_size characters. Consecutive chunks overlap by
    chunk_overlap characters to preserve context at boundaries.

    Args:
        text: The text to split.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Characters to repeat at the start of each new chunk.

    Returns:
        List of chunk strings. Returns an empty list for blank input.
    """

    if not text.strip():
        return []

    paragraphs = re.split(r"\n\n+", text)
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) > chunk_size and current:
            chunks.append(current.strip())
            current = (
                current[-chunk_overlap:] + "\n\n" + para if chunk_overlap else para
            )
        else:
            current = (current + "\n\n" + para).strip() if current else para

    if current.strip():
        chunks.append(current.strip())
    return chunks or [text.strip()]


def _chunk_text(path: Path, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """
    Read a plain text file and split it into overlapping unsectioned chunks.

    Args:
        path: Path to the .txt file.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Characters of overlap between consecutive chunks.

    Returns:
        List of chunk dicts with empty section titles.

    Raises:
        ParseError: If the file cannot be read.
    """

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ParseError(f"Cannot read {path}: {exc}") from exc
    return [
        _make_chunk(c, "", i)
        for i, c in enumerate(_split_text(text, chunk_size, chunk_overlap))
    ]


def _chunk_markdown(path: Path, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """
    Split a Markdown file on ATX headers (# through ####) into sections.

    Each header line becomes the section title for the text that follows it,
    up to the next header of depth 1–4. Sections longer than chunk_size are
    further split into overlapping sub-chunks.

    Args:
        path: Path to the .md file.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Characters of overlap between consecutive chunks.

    Returns:
        List of chunk dicts keyed by header section.

    Raises:
        ParseError: If the file cannot be read.
    """

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ParseError(f"Cannot read {path}: {exc}") from exc

    return _chunk_markdown_text(text, chunk_size, chunk_overlap)


def _silence_leptonica_stderr() -> None:
    """
    Mute libleptonica's C-level stderr output.

    Leptonica prints diagnostic messages (e.g. "Error in
    boxClipToRectangle: box outside rectangle") directly to stderr,
    bypassing Python logging. The messages are recoverable noise
    triggered by Docling's OSD pre-pass on tiny image residuals that
    layout-heron has already extracted text from. Calling
    setMsgSeverity(L_SEVERITY_NONE=6) silences them.

    Best-effort: if libleptonica is not loadable (different OS,
    different SONAME, missing symbol), silently leave stderr alone
    rather than fail the indexer.
    """

    import ctypes  # noqa: PLC0415
    import ctypes.util  # noqa: PLC0415

    lib_name = ctypes.util.find_library("lept")
    if lib_name is None:
        return

    try:
        lept = ctypes.CDLL(lib_name)
        lept.setMsgSeverity.restype = ctypes.c_int
        lept.setMsgSeverity.argtypes = [ctypes.c_int]
        l_severity_none = 6
        lept.setMsgSeverity(l_severity_none)
    except (OSError, AttributeError):
        return


class _DoclingPipeline:
    """
    Lazy holder for Docling's DocumentConverter.

    The converter is heavy: first access loads the layout and TableFormer
    models into memory (~360 MB on disk). cached_property defers
    construction until a PDF actually needs parsing and reuses the same
    converter for every subsequent call within the process.

    OCR is wired explicitly to Tesseract because Docling's "auto" OCR
    detector probes ocrmac / rapidocr / easyocr in that order and never
    falls through to Tesseract. Languages are pinned to English,
    Portuguese, Spanish (matching the tesseract-ocr-* apt packages baked
    into the Docker image). Add language codes here and the corresponding
    apt package in the Dockerfile to extend coverage.

    Tests can override the cached converter by assigning to the
    `converter` attribute (writes to __dict__ and masks the descriptor)
    or by replacing the module-level _DOCLING_PIPELINE singleton with a
    fresh instance.
    """

    @cached_property
    def converter(self) -> Any:  # noqa: ANN401
        """Build and cache the DocumentConverter on first access."""

        _silence_leptonica_stderr()
        from docling.datamodel.base_models import InputFormat  # noqa: PLC0415
        from docling.datamodel.pipeline_options import (  # noqa: PLC0415
            PdfPipelineOptions,
            TesseractOcrOptions,
        )
        from docling.document_converter import (  # noqa: PLC0415
            DocumentConverter,
            PdfFormatOption,
        )

        pdf_options = PdfPipelineOptions(
            ocr_options=TesseractOcrOptions(lang=["eng", "por", "spa"])
        )
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
            }
        )


_DOCLING_PIPELINE = _DoclingPipeline()


def _chunk_with_docling(path: Path, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """
    Convert a structured document to markdown via Docling and chunk it.

    Used for PDF, DOCX, PPTX, XLSX. Docling produces ATX-headed markdown
    that the existing markdown chunker splits into section-aware chunks,
    so layout-derived headings and tables flow through unchanged.

    Args:
        path: Path to the source file.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Characters of overlap between consecutive chunks.

    Returns:
        List of chunk dicts keyed by markdown header section.

    Raises:
        ParseError: If docling is not installed or conversion fails.
    """

    try:
        from docling.exceptions import ConversionError  # noqa: PLC0415
    except ImportError as exc:
        raise ParseError("docling is not installed") from exc

    if path.suffix.lower() == ".pdf" and _is_pdf_password_protected(path):
        raise PasswordProtectedError(f"PDF is password-protected: {path}")

    try:
        result = _DOCLING_PIPELINE.converter.convert(path)
        markdown = result.document.export_to_markdown()
    except (ConversionError, OSError, ValueError) as exc:
        raise ParseError(f"Cannot parse {path}: {exc}") from exc

    return _chunk_markdown_text(markdown, chunk_size, chunk_overlap)


def _chunk_markdown_text(text: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """
    Split markdown text on ATX headers (# through ####) into sections.

    Pulled out of _chunk_markdown so Docling-generated markdown can reuse
    the same section-aware chunking without round-tripping through disk.
    """

    sections = re.split(r"(?=^#{1,4} )", text, flags=re.MULTILINE)
    chunks: list[Chunk] = []

    for section in sections:
        if not section.strip():
            continue

        header_match = re.match(r"^(#{1,4} .+)", section)
        section_title = header_match.group(1).strip() if header_match else ""

        for sub in _split_text(section, chunk_size, chunk_overlap):
            chunks.append(_make_chunk(sub, section_title, len(chunks)))
    return chunks


def _chunk_csv(path: Path, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """
    Flatten a CSV file into pipe-delimited rows and split into chunks.

    Each row is joined with "|" between cells and rows are separated by
    newlines before being passed to the overlap-chunker. No header handling.

    Args:
        path: Path to the .csv file.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Characters of overlap between consecutive chunks.

    Returns:
        List of chunk dicts with empty section titles.

    Raises:
        ParseError: If the file cannot be read or parsed.
    """

    try:
        with path.open(encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.reader(fh)
            rows = [" | ".join(row) for row in reader]
    except OSError as exc:
        raise ParseError(f"Cannot read CSV {path}: {exc}") from exc
    except csv.Error as exc:
        raise ParseError(f"Cannot parse CSV {path}: {exc}") from exc

    text = "\n".join(rows)
    return [
        _make_chunk(c, "", i)
        for i, c in enumerate(_split_text(text, chunk_size, chunk_overlap))
    ]


def _chunk_xml(path: Path, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """
    Extract text from an XML file, using direct child tag names as sections.

    Each direct child of the root element is treated as a section. Text is
    extracted from the full subtree of each child and split into overlapping
    chunks. For flat documents with no children, all root text is returned as
    a single unsectioned chunk group.

    Args:
        path: Path to the XML file.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Characters of overlap between consecutive chunks.

    Returns:
        List of chunk dicts, one or more per non-blank section.

    Raises:
        ParseError: If lxml is not installed or the file cannot be parsed.
    """

    try:
        # Again, importing inside the function to avoid the dependency for
        # users who don't need XML support.
        from lxml import etree  # noqa: PLC0415
    except ImportError as exc:
        raise ParseError("lxml is not installed") from exc

    try:
        parser = etree.XMLParser(
            resolve_entities=False, no_network=True, load_dtd=False
        )
        tree = etree.parse(str(path), parser)
        root = tree.getroot()
        chunks: list[Chunk] = []

        for child in root:
            if not isinstance(child.tag, str):
                continue

            tag = etree.QName(child.tag).localname
            text = " ".join(child.itertext()).strip()

            if not text:
                continue

            for sub in _split_text(text, chunk_size, chunk_overlap):
                chunks.append(_make_chunk(sub, tag, len(chunks)))
        if not chunks:
            text = " ".join(root.itertext()).strip()
            chunks = [
                _make_chunk(sub, "", i)
                for i, sub in enumerate(_split_text(text, chunk_size, chunk_overlap))
            ]
    except (etree.LxmlError, OSError) as exc:
        raise ParseError(f"Cannot parse XML {path}: {exc}") from exc

    return chunks
