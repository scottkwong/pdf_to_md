"""
Tests for local Ollama-OCR integration.

These tests mock out the network / Ollama-OCR package so they run without a
running Ollama server or the optional ``ollama-ocr`` dependency installed.
"""
import base64
import io
import os
import sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local_ocr import (  # noqa: E402
    DEFAULT_LOCAL_MODEL,
    DEFAULT_OLLAMA_GENERATE_URL,
    LocalOllamaProvider,
    _pull_url,
    _server_root,
    _tags_url,
)


def _sample_image_base64() -> str:
    """Return a tiny valid JPEG encoded as base64 for provider tests."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color="white").save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def test_url_derivation() -> None:
    """Server root, pull, and tags URLs derive from the generate URL."""
    assert _server_root(DEFAULT_OLLAMA_GENERATE_URL) == "http://localhost:11434"
    assert _pull_url(DEFAULT_OLLAMA_GENERATE_URL) == (
        "http://localhost:11434/api/pull"
    )
    assert _tags_url(DEFAULT_OLLAMA_GENERATE_URL) == (
        "http://localhost:11434/api/tags"
    )

    custom = "http://192.168.1.5:9999/api/generate"
    assert _server_root(custom) == "http://192.168.1.5:9999"
    assert _pull_url(custom) == "http://192.168.1.5:9999/api/pull"


def test_provider_uses_ocr_processor() -> None:
    """process_vision writes a temp image and delegates to OCRProcessor."""
    fake_processor = mock.Mock()
    fake_processor.process_image.return_value = "# Hello\n\nWorld"

    provider = LocalOllamaProvider(model_name=DEFAULT_LOCAL_MODEL)
    # Inject the mock processor to avoid importing the real ollama-ocr package.
    provider._processor = fake_processor

    result = provider.process_vision(
        image_base64=_sample_image_base64(),
        prompt="Convert to markdown",
        prior_text="prior digital text",
    )

    assert result.text == "# Hello\n\nWorld"
    # Local inference is free -> zero-cost usage.
    assert result.usage.input_tokens == 0
    assert result.usage.output_tokens == 0
    assert result.usage.cost_usd == 0.0

    # Verify the call passed a real (now-removed) temp file and folded prior text.
    fake_processor.process_image.assert_called_once()
    call_kwargs = fake_processor.process_image.call_args.kwargs
    assert call_kwargs["format_type"] == "markdown"
    assert "prior digital text" in call_kwargs["custom_prompt"]
    assert not os.path.exists(call_kwargs["image_path"])


def test_provider_missing_package_raises_importerror() -> None:
    """A missing ollama-ocr package produces a helpful ImportError."""
    provider = LocalOllamaProvider()
    with mock.patch.dict(sys.modules, {"ollama_ocr": None}):
        try:
            provider._get_processor()
        except ImportError as error:
            assert "ollama-ocr" in str(error)
        else:  # pragma: no cover
            raise AssertionError("Expected ImportError for missing package")


def run_all_tests() -> bool:
    """Run local OCR tests directly (used by run_tests.py-style runners)."""
    test_url_derivation()
    test_provider_uses_ocr_processor()
    test_provider_missing_package_raises_importerror()
    print("  ✓ local OCR provider tests passed")
    return True


if __name__ == "__main__":
    run_all_tests()
