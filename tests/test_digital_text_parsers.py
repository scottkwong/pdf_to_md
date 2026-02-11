"""
Correctness tests for digital PDF text parser engines.

This suite validates direct package extraction with PyPDF2 and optional
PyMuPDF using deterministic fixture PDFs. It also verifies parser auto-
selection behavior for environments with or without PyMuPDF installed.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from digital_text_parsers import (
    create_digital_text_parser,
    get_available_digital_text_parsers,
    is_pymupdf_available,
)
from tests.create_test_pdf import ensure_default_fixture_pdfs


def _assert_fixture_markers(page_texts: list[str]) -> None:
    """Assert expected marker text exists in extracted page content.

    Args:
        page_texts: Extracted text list in page order.
    """
    assert page_texts
    for idx, page_text in enumerate(page_texts, start=1):
        assert f"Page {idx}" in page_text
        assert f"FIXTURE_PAGE_{idx}" in page_text


def test_pypdf2_extraction_on_generated_fixture() -> None:
    """Validate PyPDF2 extraction against generated benchmark fixture."""
    fixture_path = ensure_default_fixture_pdfs()[0]
    parser = create_digital_text_parser("pypdf2").parser
    pages = parser.extract_pages(fixture_path)
    assert len(pages) == 5
    _assert_fixture_markers(pages)


def test_auto_backend_resolution() -> None:
    """Validate parser resolution policy for auto selection."""
    selection = create_digital_text_parser("auto")
    if is_pymupdf_available():
        assert selection.resolved_parser == "pymupdf"
    else:
        assert selection.resolved_parser == "pypdf2"


def test_available_backends_reported() -> None:
    """Validate available parser listing includes expected entries."""
    available = get_available_digital_text_parsers()
    assert "pypdf2" in available
    if is_pymupdf_available():
        assert "pymupdf" in available


@pytest.mark.skipif(
    not is_pymupdf_available(),
    reason="PyMuPDF is not installed in this environment.",
)
def test_pymupdf_extraction_on_generated_fixture() -> None:
    """Validate PyMuPDF extraction when dependency is installed."""
    fixture_path = ensure_default_fixture_pdfs()[0]
    parser = create_digital_text_parser("pymupdf").parser
    pages = parser.extract_pages(fixture_path)
    assert len(pages) == 5
    _assert_fixture_markers(pages)


def run_all_tests() -> bool:
    """Run parser correctness tests and return pass/fail status."""
    print("Running digital text parser correctness tests...\n")
    try:
        test_pypdf2_extraction_on_generated_fixture()
        test_auto_backend_resolution()
        if is_pymupdf_available():
            test_pymupdf_extraction_on_generated_fixture()
            pymupdf_status = "PASSED"
        else:
            pymupdf_status = "SKIPPED (PyMuPDF not installed)"

        print("\n" + "=" * 50)
        print("Digital Text Parser Test Summary:")
        print("=" * 50)
        print("PyPDF2: PASSED")
        print(f"PyMuPDF: {pymupdf_status}")
        return True
    except Exception as exc:
        print(f"Digital text parser tests failed: {exc}")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
