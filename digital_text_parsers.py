"""
Digital text parser engines for PDF first-pass extraction.

This module defines a reusable interface used to extract digital text from
PDF pages before vision-model enhancement. Parser engines can be selected
explicitly or resolved automatically. The rest of the pipeline depends only
on this interface, which makes it straightforward to add additional parser
packages later.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from PyPDF2 import PdfReader


class BaseDigitalTextParser(ABC):
    """Abstract base class for digital PDF page text parser engines."""

    @property
    @abstractmethod
    def parser_name(self) -> str:
        """Return short parser identifier."""

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Return True when parser dependencies are importable."""

    @abstractmethod
    def extract_pages(self, pdf_path: str) -> List[str]:
        """Extract text for each page in a PDF.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of extracted page text strings in page order.
        """


class PyPDF2DigitalTextParser(BaseDigitalTextParser):
    """Digital text parser engine implemented with PyPDF2."""

    @property
    def parser_name(self) -> str:
        """Return short parser identifier."""
        return "pypdf2"

    @classmethod
    def is_available(cls) -> bool:
        """Return True when parser dependencies are importable."""
        return True

    def extract_pages(self, pdf_path: str) -> List[str]:
        """Extract text for each page using PyPDF2.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of extracted page text strings in page order.
        """
        with open(pdf_path, "rb") as file:
            reader = PdfReader(file)
            return [page.extract_text() or "" for page in reader.pages]


class PyMuPDFDigitalTextParser(BaseDigitalTextParser):
    """Digital text parser engine implemented with PyMuPDF."""

    @property
    def parser_name(self) -> str:
        """Return short parser identifier."""
        return "pymupdf"

    @classmethod
    def is_available(cls) -> bool:
        """Return True when parser dependencies are importable."""
        try:
            import fitz  # pylint: disable=import-outside-toplevel,unused-import
        except ImportError:
            return False
        return True

    def extract_pages(self, pdf_path: str) -> List[str]:
        """Extract text for each page using PyMuPDF.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of extracted page text strings in page order.
        """
        import fitz  # pylint: disable=import-outside-toplevel

        page_texts: List[str] = []
        with fitz.open(pdf_path) as document:
            for page in document:
                page_texts.append(page.get_text("text") or "")
        return page_texts


@dataclass(frozen=True)
class DigitalTextParserSelection:
    """Resolved digital text parser details.

    Attributes:
        parser: Parser implementation instance.
        requested_parser: CLI/parser engine configuration requested by user.
        resolved_parser: Parser engine that is actually used at runtime.
    """

    parser: BaseDigitalTextParser
    requested_parser: str
    resolved_parser: str


def create_digital_text_parser(
    parser_name: str = "auto",
) -> DigitalTextParserSelection:
    """Create a digital text parser implementation.

    Args:
        parser_name: Requested parser name. Allowed values are `auto`, `pypdf2`,
            and `pymupdf`.

    Returns:
        A `DigitalTextParserSelection` with parser instance and resolution
        metadata.

    Raises:
        ValueError: If parser name is unsupported or an unavailable parser is
            explicitly requested.
    """
    normalized = parser_name.lower().strip()
    if normalized not in {"auto", "pypdf2", "pymupdf"}:
        raise ValueError(
            "Invalid digital text parser "
            f"'{parser_name}'. Valid options are auto, pypdf2, pymupdf."
        )

    if normalized == "auto":
        if PyMuPDFDigitalTextParser.is_available():
            parser: BaseDigitalTextParser = PyMuPDFDigitalTextParser()
        else:
            parser = PyPDF2DigitalTextParser()
        return DigitalTextParserSelection(
            parser=parser,
            requested_parser=normalized,
            resolved_parser=parser.parser_name,
        )

    if normalized == "pymupdf":
        if not PyMuPDFDigitalTextParser.is_available():
            raise ValueError(
                "PyMuPDF parser requested but dependency is not installed. "
                "Install with `pip install pymupdf` or `pip install '.[pymupdf]'`, "
                "or use `--digital-text-parser pypdf2`."
            )
        parser = PyMuPDFDigitalTextParser()
    else:
        parser = PyPDF2DigitalTextParser()

    return DigitalTextParserSelection(
        parser=parser,
        requested_parser=normalized,
        resolved_parser=parser.parser_name,
    )


def is_pymupdf_available() -> bool:
    """Return whether PyMuPDF can be imported on this system."""
    return PyMuPDFDigitalTextParser.is_available()


def get_available_digital_text_parsers() -> List[str]:
    """Return available digital text parser identifiers for this environment.

    Returns:
        Ordered list of available parser identifiers.
    """
    parsers = ["pypdf2"]
    if is_pymupdf_available():
        parsers.append("pymupdf")
    return parsers
