"""
Text parser backends for PDF first-pass extraction.

This module defines a reusable parser interface used to extract text from PDF
pages before vision-model enhancement. Backends can be selected explicitly or
resolved automatically. The rest of the pipeline depends only on the interface,
which makes it straightforward to add additional parser packages later.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from PyPDF2 import PdfReader


class BaseTextParser(ABC):
    """Abstract base class for PDF page-text extraction backends."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return short backend identifier."""

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Return True when backend dependencies are importable."""

    @abstractmethod
    def extract_pages(self, pdf_path: str) -> List[str]:
        """Extract text for each page in a PDF.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of extracted page text strings in page order.
        """


class PyPDF2TextParser(BaseTextParser):
    """PDF text parser backend implemented with PyPDF2."""

    @property
    def backend_name(self) -> str:
        """Return short backend identifier."""
        return "pypdf2"

    @classmethod
    def is_available(cls) -> bool:
        """Return True when backend dependencies are importable."""
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


class PyMuPDFTextParser(BaseTextParser):
    """PDF text parser backend implemented with PyMuPDF."""

    @property
    def backend_name(self) -> str:
        """Return short backend identifier."""
        return "pymupdf"

    @classmethod
    def is_available(cls) -> bool:
        """Return True when backend dependencies are importable."""
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
class TextParserSelection:
    """Resolved parser backend details.

    Attributes:
        parser: Parser implementation instance.
        requested_backend: CLI/backend configuration requested by user.
        resolved_backend: Backend that is actually used at runtime.
    """

    parser: BaseTextParser
    requested_backend: str
    resolved_backend: str


def create_text_parser(backend: str = "auto") -> TextParserSelection:
    """Create a parser backend implementation.

    Args:
        backend: Requested backend name. Allowed values are `auto`, `pypdf2`,
            and `pymupdf`.

    Returns:
        A `TextParserSelection` with parser instance and resolution metadata.

    Raises:
        ValueError: If backend is unsupported or an unavailable backend is
            explicitly requested.
    """
    normalized = backend.lower().strip()
    if normalized not in {"auto", "pypdf2", "pymupdf"}:
        raise ValueError(
            "Invalid text extractor backend "
            f"'{backend}'. Valid options are auto, pypdf2, pymupdf."
        )

    if normalized == "auto":
        if PyMuPDFTextParser.is_available():
            parser: BaseTextParser = PyMuPDFTextParser()
        else:
            parser = PyPDF2TextParser()
        return TextParserSelection(
            parser=parser,
            requested_backend=normalized,
            resolved_backend=parser.backend_name,
        )

    if normalized == "pymupdf":
        if not PyMuPDFTextParser.is_available():
            raise ValueError(
                "PyMuPDF backend requested but dependency is not installed. "
                "Install with `pip install pymupdf` or use "
                "`--text-extractor-backend pypdf2`."
            )
        parser = PyMuPDFTextParser()
    else:
        parser = PyPDF2TextParser()

    return TextParserSelection(
        parser=parser,
        requested_backend=normalized,
        resolved_backend=parser.backend_name,
    )


def is_pymupdf_available() -> bool:
    """Return whether PyMuPDF can be imported on this system."""
    return PyMuPDFTextParser.is_available()
