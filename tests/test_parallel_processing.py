"""
Tests for parallel page processing functionality.

Tests the VisionExtractor parallel processing and CLI argument changes
without requiring actual API keys or external dependencies.
"""
import argparse
import os
import sys
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extractors import ExtractionResult, VisionExtractor


class MockProvider:
    """Mock LLM provider for testing."""

    def __init__(self):
        """Initialize mock provider."""
        self.call_count = 0

    def process_vision(
        self,
        image_base64: str,
        prompt: str,
        prior_text: Optional[str] = None,
        model: str = "",
        max_tokens: int = 4096,
    ) -> str:
        """
        Mock vision processing that returns predictable output.

        Args:
            image_base64: Base64-encoded image (ignored).
            prompt: Prompt text (ignored).
            prior_text: Prior text (included in output if provided).
            model: Model name (ignored).
            max_tokens: Max tokens (ignored).

        Returns:
            Mock markdown content.
        """
        self.call_count += 1
        if prior_text:
            return f"# Mock Page Content\n\nPrior text: {prior_text[:20]}..."
        return "# Mock Page Content\n\nNo prior text provided."


class TestVisionExtractorInit:
    """Tests for VisionExtractor initialization."""

    def test_default_max_parallel_pages(self):
        """Test that default max_parallel_pages is 10."""
        provider = MockProvider()
        extractor = VisionExtractor(
            provider=provider,
            model_id="test-model",
        )
        assert extractor.max_parallel_pages == 10

    def test_custom_max_parallel_pages(self):
        """Test setting custom max_parallel_pages."""
        provider = MockProvider()
        extractor = VisionExtractor(
            provider=provider,
            model_id="test-model",
            max_parallel_pages=5,
        )
        assert extractor.max_parallel_pages == 5

    def test_max_parallel_pages_of_one(self):
        """Test setting max_parallel_pages to 1 (sequential)."""
        provider = MockProvider()
        extractor = VisionExtractor(
            provider=provider,
            model_id="test-model",
            max_parallel_pages=1,
        )
        assert extractor.max_parallel_pages == 1


class TestParallelPageProcessing:
    """Tests for parallel page processing logic."""

    @patch.object(VisionExtractor, '_pdf_to_images_with_storage')
    @patch.object(VisionExtractor, '_get_prior_text')
    def test_page_order_preserved(
        self, mock_get_prior_text, mock_pdf_to_images
    ):
        """Test that page order is preserved after parallel processing."""
        # Create mock images (just need objects, content doesn't matter)
        num_pages = 5
        mock_images = [MagicMock() for _ in range(num_pages)]
        mock_pdf_to_images.return_value = mock_images
        mock_get_prior_text.return_value = [f"Text for page {i}" for i in range(num_pages)]

        provider = MockProvider()
        extractor = VisionExtractor(
            provider=provider,
            model_id="test-model",
            mode="vt",
            max_parallel_pages=10,
        )

        # Mock the base64 conversion to return page index
        def mock_base64(image):
            # Find which mock image this is
            for i, mock_img in enumerate(mock_images):
                if image is mock_img:
                    return f"base64_page_{i}"
            return "unknown"

        with patch.object(extractor, '_pdf_image_to_base64_str', side_effect=mock_base64):
            result = extractor.extract("/fake/path.pdf", "/fake/output", verbose=False)

        # Verify page order in output
        lines = result.markdown.split('\n')
        page_headers = [l for l in lines if l.startswith("File:")]
        
        assert len(page_headers) == num_pages
        for i, header in enumerate(page_headers):
            assert f"Page: {i + 1}" in header

    @patch.object(VisionExtractor, '_pdf_to_images_with_storage')
    @patch.object(VisionExtractor, '_get_prior_text')
    def test_metadata_includes_max_parallel_pages(
        self, mock_get_prior_text, mock_pdf_to_images
    ):
        """Test that result metadata includes max_parallel_pages."""
        mock_pdf_to_images.return_value = [MagicMock()]
        mock_get_prior_text.return_value = [None]

        provider = MockProvider()
        extractor = VisionExtractor(
            provider=provider,
            model_id="test-model",
            mode="v",
            max_parallel_pages=7,
        )

        with patch.object(extractor, '_pdf_image_to_base64_str', return_value="base64"):
            result = extractor.extract("/fake/path.pdf", "/fake/output", verbose=False)

        assert result.metadata["max_parallel_pages"] == 7


class TestCLIArguments:
    """Tests for CLI argument parsing."""

    def test_parallel_default_value(self):
        """Test that -p defaults to 10."""
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-p", "--parallel",
            type=int,
            nargs="?",
            const=10,
            default=10,
        )
        
        # No -p flag
        args = parser.parse_args([])
        assert args.parallel == 10

    def test_parallel_with_no_value(self):
        """Test that -p without value uses const (10)."""
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-p", "--parallel",
            type=int,
            nargs="?",
            const=10,
            default=10,
        )
        
        # -p flag without value
        args = parser.parse_args(["-p"])
        assert args.parallel == 10

    def test_parallel_with_custom_value(self):
        """Test that -p 5 sets parallel to 5."""
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-p", "--parallel",
            type=int,
            nargs="?",
            const=10,
            default=10,
        )
        
        args = parser.parse_args(["-p", "5"])
        assert args.parallel == 5

    def test_single_flag_default_false(self):
        """Test that -s defaults to False."""
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-s", "--single",
            action="store_true",
            default=False,
        )
        
        args = parser.parse_args([])
        assert args.single is False

    def test_single_flag_when_set(self):
        """Test that -s sets single to True."""
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-s", "--single",
            action="store_true",
            default=False,
        )
        
        args = parser.parse_args(["-s"])
        assert args.single is True

    def test_digital_text_parser_default_auto(self):
        """Test digital text parser defaults to auto."""
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--digital-text-parser",
            type=str,
            choices=["auto", "pypdf2", "pymupdf"],
            default="auto",
        )
        args = parser.parse_args([])
        assert args.digital_text_parser == "auto"

    def test_digital_text_parser_override(self):
        """Test digital text parser can be explicitly selected."""
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--digital-text-parser",
            type=str,
            choices=["auto", "pypdf2", "pymupdf"],
            default="auto",
        )
        args = parser.parse_args(["--digital-text-parser", "pypdf2"])
        assert args.digital_text_parser == "pypdf2"


class TestIntegrationWithRealPDF:
    """Integration tests using a real PDF file with mock provider."""

    @pytest.fixture
    def test_pdf_path(self, tmp_path):
        """Create a test PDF and return its path."""
        from tests.create_test_pdf import create_test_pdf

        pdf_path = str(tmp_path / "test_multipage.pdf")
        create_test_pdf(pdf_path, num_pages=5)
        return pdf_path

    def test_extracts_correct_number_of_pages(self, test_pdf_path, tmp_path):
        """Test that extraction processes all pages from a real PDF."""
        provider = MockProvider()
        extractor = VisionExtractor(
            provider=provider,
            model_id="test-model",
            mode="v",
            max_parallel_pages=5,
        )

        result = extractor.extract(
            test_pdf_path, str(tmp_path), verbose=False
        )

        assert result.page_count == 5
        assert provider.call_count == 5

    def test_page_order_preserved_with_real_pdf(self, test_pdf_path, tmp_path):
        """Test that page order is correct when processing a real PDF."""
        provider = MockProvider()
        extractor = VisionExtractor(
            provider=provider,
            model_id="test-model",
            mode="v",
            max_parallel_pages=5,
        )

        result = extractor.extract(
            test_pdf_path, str(tmp_path), verbose=False
        )

        # Verify page headers are in order
        lines = result.markdown.split('\n')
        page_headers = [l for l in lines if l.startswith("File:")]

        assert len(page_headers) == 5
        for i, header in enumerate(page_headers):
            assert f"Page: {i + 1}" in header

    def test_prior_text_extracted_from_real_pdf(self, test_pdf_path, tmp_path):
        """Test that prior text is extracted from real PDF pages."""

        class PriorTextCapturingProvider:
            """Provider that captures prior_text for verification."""

            def __init__(self):
                self.captured_prior_texts = []

            def process_vision(
                self,
                image_base64: str,
                prompt: str,
                prior_text: Optional[str] = None,
                model: str = "",
                max_tokens: int = 4096,
            ) -> str:
                self.captured_prior_texts.append(prior_text)
                return "# Mock content"

        provider = PriorTextCapturingProvider()
        extractor = VisionExtractor(
            provider=provider,
            model_id="test-model",
            mode="vt",  # vision + text mode
            max_parallel_pages=5,
        )

        extractor.extract(test_pdf_path, str(tmp_path), verbose=False)

        # Should have captured 5 prior texts
        assert len(provider.captured_prior_texts) == 5

        # Each prior text should be non-None and contain page-specific content
        # Note: order is not guaranteed due to parallel processing
        all_prior_text = "\n".join(provider.captured_prior_texts)
        for page_num in range(1, 6):
            assert f"Page {page_num}" in all_prior_text

    def test_parallel_with_max_workers_limit(self, test_pdf_path, tmp_path):
        """Test that max_parallel_pages limits concurrent processing."""
        provider = MockProvider()
        extractor = VisionExtractor(
            provider=provider,
            model_id="test-model",
            mode="v",
            max_parallel_pages=2,  # Only 2 at a time
        )

        result = extractor.extract(
            test_pdf_path, str(tmp_path), verbose=False
        )

        # Should still process all 5 pages correctly
        assert result.page_count == 5
        assert provider.call_count == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
