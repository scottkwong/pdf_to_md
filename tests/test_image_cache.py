"""
Tests for VisionExtractor page-image caching.

Regression coverage for a crashed render leaving behind an empty cache
directory that later runs trusted, loading zero pages and failing a page-count
mismatch on every subsequent run until the directory was deleted by hand.
"""
import os
import sys
from unittest import mock

import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extractors import VisionExtractor  # noqa: E402


def _extractor() -> VisionExtractor:
    """Build a VisionExtractor with a stub provider; caching needs no network."""
    return VisionExtractor(provider=mock.Mock(), model_id="test-model")


def _pages(count: int) -> list:
    """Return `count` throwaway page images."""
    return [Image.new("RGB", (8, 8), color="white") for _ in range(count)]


def _make_pdf(tmp_path) -> str:
    """Write a placeholder PDF; every test here mocks the renderer."""
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    return str(pdf)


def test_empty_cache_directory_triggers_rerender(tmp_path) -> None:
    """An empty cache directory is a miss, not a zero-page cache hit."""
    pdf_path = _make_pdf(tmp_path)
    stale = tmp_path / "doc_images"
    stale.mkdir()

    with mock.patch(
        "extractors.convert_from_path", return_value=_pages(2)
    ) as convert:
        images = _extractor()._pdf_to_images_with_storage(pdf_path, str(tmp_path))

    assert convert.call_count == 1
    assert len(images) == 2
    assert len(list(stale.glob("*.png"))) == 2


def test_failed_render_leaves_no_cache_behind(tmp_path) -> None:
    """A render that raises must not leave a cache for the next run to trust."""
    pdf_path = _make_pdf(tmp_path)

    with mock.patch(
        "extractors.convert_from_path",
        side_effect=FileNotFoundError("pdfinfo"),
    ):
        with pytest.raises(FileNotFoundError):
            _extractor()._pdf_to_images_with_storage(pdf_path, str(tmp_path))

    assert not (tmp_path / "doc_images").exists()
    assert not (tmp_path / "doc_images.partial").exists()


def test_second_run_after_failure_succeeds(tmp_path) -> None:
    """The exact reported sequence: a failed run must not poison the retry."""
    pdf_path = _make_pdf(tmp_path)
    extractor = _extractor()

    with mock.patch(
        "extractors.convert_from_path",
        side_effect=FileNotFoundError("pdfinfo"),
    ):
        with pytest.raises(FileNotFoundError):
            extractor._pdf_to_images_with_storage(pdf_path, str(tmp_path))

    with mock.patch("extractors.convert_from_path", return_value=_pages(1)):
        images = extractor._pdf_to_images_with_storage(pdf_path, str(tmp_path))

    assert len(images) == 1


def test_populated_cache_is_reused(tmp_path) -> None:
    """A complete cache still short-circuits rendering."""
    pdf_path = _make_pdf(tmp_path)
    extractor = _extractor()

    with mock.patch("extractors.convert_from_path", return_value=_pages(3)):
        extractor._pdf_to_images_with_storage(pdf_path, str(tmp_path))

    with mock.patch("extractors.convert_from_path") as convert:
        images = extractor._pdf_to_images_with_storage(pdf_path, str(tmp_path))

    convert.assert_not_called()
    assert len(images) == 3


def test_cached_pages_load_in_page_order(tmp_path) -> None:
    """Page 10 must not sort ahead of page 2 when the cache is reloaded."""
    pdf_path = _make_pdf(tmp_path)
    extractor = _extractor()

    with mock.patch("extractors.convert_from_path", return_value=_pages(11)):
        extractor._pdf_to_images_with_storage(pdf_path, str(tmp_path))

    with mock.patch("extractors.convert_from_path"):
        images = extractor._pdf_to_images_with_storage(pdf_path, str(tmp_path))

    filenames = [os.path.basename(image.filename) for image in images]
    assert filenames == [f"doc_image_{i}.png" for i in range(11)]
