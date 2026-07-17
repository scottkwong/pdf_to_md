"""
Tests for the local Ollama integration.

These tests mock the network, so they run without a running Ollama server.
"""
import base64
import io
import os
import sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local_ocr import (  # noqa: E402
    DEFAULT_LOCAL_MODEL,
    DEFAULT_NUM_CTX,
    DEFAULT_NUM_PREDICT,
    DEFAULT_OLLAMA_GENERATE_URL,
    LocalOllamaProvider,
    _pull_url,
    _server_root,
    _tags_url,
    strip_wrapping_code_fence,
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


def test_provider_posts_to_ollama() -> None:
    """process_vision posts the page to Ollama and returns its response."""
    fake_response = mock.Mock()
    fake_response.raise_for_status.return_value = None
    fake_response.json.return_value = {
        "response": "# Hello\n\nWorld",
        "prompt_eval_count": 1200,
        "eval_count": 340,
    }

    provider = LocalOllamaProvider(model_name=DEFAULT_LOCAL_MODEL)
    with mock.patch("requests.post", return_value=fake_response) as post:
        result = provider.process_vision(
            image_base64=_sample_image_base64(),
            prompt="Convert to markdown",
            prior_text="prior digital text",
            max_tokens=2048,
        )

    assert result.text == "# Hello\n\nWorld"
    # Token counts come from Ollama; local inference is still free.
    assert result.usage.input_tokens == 1200
    assert result.usage.output_tokens == 340
    assert result.usage.cost_usd == 0.0

    payload = post.call_args.kwargs["json"]
    assert payload["model"] == DEFAULT_LOCAL_MODEL
    assert payload["stream"] is False
    assert payload["images"] == [_sample_image_base64()]
    assert "prior digital text" in payload["prompt"]
    # Context is sized per page rather than left at the model's default.
    assert payload["options"]["num_ctx"] == DEFAULT_NUM_CTX
    # A reasoning model needs room to think *and* answer, so the provider's
    # budget wins over a smaller max_tokens from the caller.
    assert payload["options"]["num_predict"] == DEFAULT_NUM_PREDICT
    # An unbounded request would hang the whole run.
    assert post.call_args.kwargs["timeout"] > 0


def test_num_predict_floor_respects_larger_max_tokens() -> None:
    """A caller asking for more output tokens than the default still wins."""
    fake_response = mock.Mock()
    fake_response.raise_for_status.return_value = None
    fake_response.json.return_value = {"response": "ok"}

    provider = LocalOllamaProvider(num_predict=8192)
    with mock.patch("requests.post", return_value=fake_response) as post:
        provider.process_vision(
            image_base64=_sample_image_base64(),
            prompt="Convert",
            max_tokens=20000,
        )
    assert post.call_args.kwargs["json"]["options"]["num_predict"] == 20000


def test_provider_raises_on_http_error() -> None:
    """A failed HTTP call raises rather than yielding a bad page."""
    provider = LocalOllamaProvider()
    with mock.patch("requests.post", side_effect=OSError("connection refused")):
        try:
            provider.process_vision(
                image_base64=_sample_image_base64(), prompt="Convert"
            )
        except RuntimeError as error:
            assert "connection refused" in str(error)
        else:  # pragma: no cover
            raise AssertionError("Expected RuntimeError on transport failure")


def test_provider_raises_on_ollama_error_field() -> None:
    """An error reported in Ollama's JSON body raises."""
    fake_response = mock.Mock()
    fake_response.raise_for_status.return_value = None
    fake_response.json.return_value = {"error": "unknown model architecture"}

    provider = LocalOllamaProvider(model_name="broken-model")
    with mock.patch("requests.post", return_value=fake_response):
        try:
            provider.process_vision(
                image_base64=_sample_image_base64(), prompt="Convert"
            )
        except RuntimeError as error:
            assert "unknown model architecture" in str(error)
        else:  # pragma: no cover
            raise AssertionError("Expected RuntimeError for an error body")


def test_provider_raises_on_budget_exhaustion() -> None:
    """A reasoning model that thinks past its budget fails loudly.

    Ollama reports done_reason="length" with an empty response, which would
    otherwise be written into the document as a blank page.
    """
    fake_response = mock.Mock()
    fake_response.raise_for_status.return_value = None
    fake_response.json.return_value = {
        "response": "",
        "thinking": "let me work through this page...",
        "done_reason": "length",
    }

    provider = LocalOllamaProvider(model_name="qwen3-vl:8b")
    with mock.patch("requests.post", return_value=fake_response):
        try:
            provider.process_vision(
                image_base64=_sample_image_base64(), prompt="Convert"
            )
        except RuntimeError as error:
            # The message must name the flag that fixes it.
            assert "--local-num-predict" in str(error)
        else:  # pragma: no cover
            raise AssertionError("Expected RuntimeError on budget exhaustion")


def test_truncated_but_usable_page_is_kept() -> None:
    """A length-capped page that still returned text is not discarded."""
    fake_response = mock.Mock()
    fake_response.raise_for_status.return_value = None
    fake_response.json.return_value = {
        "response": "# Real content",
        "done_reason": "length",
    }

    provider = LocalOllamaProvider()
    with mock.patch("requests.post", return_value=fake_response):
        result = provider.process_vision(
            image_base64=_sample_image_base64(), prompt="Convert"
        )
    assert result.text == "# Real content"


def test_strip_wrapping_code_fence() -> None:
    """A whole-page ```markdown wrapper is removed; real code blocks are not.

    Local models frequently enclose an entire page of markdown in a fence,
    which would otherwise render the page as a literal code block.
    """
    assert strip_wrapping_code_fence("```markdown\n# Hi\n```") == "# Hi"
    assert strip_wrapping_code_fence("```\n# Hi\n```") == "# Hi"
    assert strip_wrapping_code_fence("# Hi") == "# Hi"

    # A fenced code block inside a page must survive untouched.
    page = "# Doc\n\n```python\nx = 1\n```\n\nEnd"
    assert strip_wrapping_code_fence(page) == page

    # An outer wrapper is removed while the inner block is preserved.
    wrapped = "```markdown\n# A\n\n```python\nx = 1\n```\n```"
    assert strip_wrapping_code_fence(wrapped) == "# A\n\n```python\nx = 1\n```"


def test_provider_strips_fence_from_result() -> None:
    """process_vision returns unwrapped markdown, not a fenced code block."""
    fake_response = mock.Mock()
    fake_response.raise_for_status.return_value = None
    fake_response.json.return_value = {"response": "```markdown\n# Page\n```"}

    provider = LocalOllamaProvider()
    with mock.patch("requests.post", return_value=fake_response):
        result = provider.process_vision(
            image_base64=_sample_image_base64(), prompt="Convert to markdown"
        )
    assert result.text == "# Page"


def run_all_tests() -> bool:
    """Run local OCR tests directly (used by run_tests.py-style runners)."""
    test_url_derivation()
    test_provider_posts_to_ollama()
    test_num_predict_floor_respects_larger_max_tokens()
    test_provider_raises_on_http_error()
    test_provider_raises_on_ollama_error_field()
    test_provider_raises_on_budget_exhaustion()
    test_truncated_but_usable_page_is_kept()
    test_strip_wrapping_code_fence()
    test_provider_strips_fence_from_result()
    print("  ✓ local OCR provider tests passed")
    return True


if __name__ == "__main__":
    run_all_tests()
