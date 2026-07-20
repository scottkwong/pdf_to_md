"""
Tests for the local vLLM integration (KDL-Frontier-Parser and friends).

These tests mock the OpenAI client and the network, so they run without a
running vLLM server.
"""
import base64
import io
import os
import sys
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm_ocr import (  # noqa: E402
    DEFAULT_MAX_TOKENS,
    DEFAULT_UNLIMITED_MODEL,
    DEFAULT_UNLIMITED_OCR_PROMPT,
    DEFAULT_VLLM_BASE_URL,
    DEFAULT_VLLM_MODEL,
    UnlimitedOcrProvider,
    VllmOpenAIProvider,
    _models_url,
    ensure_vllm_server,
    is_vllm_running,
    list_served_vllm_models,
    strip_grounding_tokens,
)


def _sample_image_base64() -> str:
    """Return a tiny valid JPEG encoded as base64 for provider tests."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color="white").save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _fake_completion(
    content: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    finish_reason: str = "stop",
):
    """Build a stub mimicking the OpenAI chat.completions response shape."""
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
    )
    return SimpleNamespace(choices=[choice], usage=usage)


def _provider_with_stub(response, capture: dict, provider=None):
    """Return a provider whose chat.completions.create returns ``response``.

    ``capture`` is populated with the kwargs the create() call receives so the
    test can assert on the request payload. Pass ``provider`` to stub a subclass
    (e.g. UnlimitedOcrProvider); defaults to a plain VllmOpenAIProvider.
    """
    if provider is None:
        provider = VllmOpenAIProvider(model_name=DEFAULT_VLLM_MODEL)

    def fake_create(**kwargs):
        capture.update(kwargs)
        return response

    provider.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=fake_create)
        )
    )
    return provider


def test_models_url_derivation() -> None:
    """The /models probe URL derives from the OpenAI base URL."""
    assert _models_url(DEFAULT_VLLM_BASE_URL) == (
        "http://localhost:8000/v1/models"
    )
    assert _models_url("http://host:8000/v1/") == "http://host:8000/v1/models"


def test_is_vllm_running_true_on_200() -> None:
    """is_vllm_running is True only when /models returns HTTP 200."""
    ok = mock.Mock(status_code=200)
    with mock.patch("requests.get", return_value=ok):
        assert is_vllm_running(DEFAULT_VLLM_BASE_URL) is True

    with mock.patch("requests.get", side_effect=OSError("refused")):
        assert is_vllm_running(DEFAULT_VLLM_BASE_URL) is False


def test_list_served_models() -> None:
    """Served model ids are read from the /models data list."""
    resp = mock.Mock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"data": [{"id": "kdl-frontier-parser-nano"}]}
    with mock.patch("requests.get", return_value=resp):
        assert list_served_vllm_models() == ["kdl-frontier-parser-nano"]


def test_ensure_server_raises_when_unreachable() -> None:
    """A missing server raises with a message that names `vllm serve`."""
    with mock.patch("vllm_ocr.is_vllm_running", return_value=False):
        try:
            ensure_vllm_server(DEFAULT_VLLM_MODEL, verbose=False)
        except RuntimeError as error:
            assert "vllm serve" in str(error)
        else:  # pragma: no cover
            raise AssertionError("Expected RuntimeError when server is down")


def test_ensure_server_raises_on_wrong_model() -> None:
    """A server that serves a different model name raises, listing what it has."""
    with mock.patch("vllm_ocr.is_vllm_running", return_value=True), mock.patch(
        "vllm_ocr.list_served_vllm_models", return_value=["some-other-model"]
    ):
        try:
            ensure_vllm_server("kdl-frontier-parser-nano", verbose=False)
        except RuntimeError as error:
            assert "some-other-model" in str(error)
            assert "--vllm-model" in str(error)
        else:  # pragma: no cover
            raise AssertionError("Expected RuntimeError on model-name mismatch")


def test_provider_calls_vllm_with_required_options() -> None:
    """process_vision sends the page and the model-card-required options."""
    capture: dict = {}
    response = _fake_completion(
        "# Hello\n\nWorld", prompt_tokens=1200, completion_tokens=340
    )
    provider = _provider_with_stub(response, capture)

    result = provider.process_vision(
        image_base64=_sample_image_base64(),
        prompt="Convert to markdown",
        prior_text="prior digital text",
        max_tokens=2048,
    )

    assert result.text == "# Hello\n\nWorld"
    # Token counts come from vLLM; local inference is still free.
    assert result.usage.input_tokens == 1200
    assert result.usage.output_tokens == 340
    assert result.usage.cost_usd == 0.0

    assert capture["model"] == DEFAULT_VLLM_MODEL
    # Greedy decoding per the model card.
    assert capture["temperature"] == 0.0
    # The provider's own budget wins over the smaller caller max_tokens.
    assert capture["max_tokens"] == DEFAULT_MAX_TOKENS
    # vLLM-specific options: keep special tokens, disable thinking.
    extra = capture["extra_body"]
    assert extra["skip_special_tokens"] is False
    assert extra["chat_template_kwargs"] == {"enable_thinking": False}

    # The image rides along as a base64 data URL, and prior text is in the prompt.
    content = capture["messages"][0]["content"]
    text_part = next(p for p in content if p["type"] == "text")
    image_part = next(p for p in content if p["type"] == "image_url")
    assert "prior digital text" in text_part["text"]
    assert image_part["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_num_predict_floor_respects_larger_max_tokens() -> None:
    """A caller asking for more output tokens than the default still wins."""
    capture: dict = {}
    provider = _provider_with_stub(_fake_completion("ok"), capture)
    provider.max_tokens = DEFAULT_MAX_TOKENS
    provider.process_vision(
        image_base64=_sample_image_base64(),
        prompt="Convert",
        max_tokens=20000,
    )
    assert capture["max_tokens"] == 20000


def test_provider_raises_on_transport_error() -> None:
    """A failed request raises rather than yielding a bad page."""
    provider = VllmOpenAIProvider()

    def boom(**_kwargs):
        raise OSError("connection refused")

    provider.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=boom))
    )
    try:
        provider.process_vision(
            image_base64=_sample_image_base64(), prompt="Convert"
        )
    except RuntimeError as error:
        assert "connection refused" in str(error)
    else:  # pragma: no cover
        raise AssertionError("Expected RuntimeError on transport failure")


def test_provider_raises_on_empty_length_truncation() -> None:
    """A length-capped answer with no text fails loudly, naming the flag."""
    capture: dict = {}
    response = _fake_completion("", finish_reason="length")
    provider = _provider_with_stub(response, capture)
    try:
        provider.process_vision(
            image_base64=_sample_image_base64(), prompt="Convert"
        )
    except RuntimeError as error:
        assert "--vllm-max-tokens" in str(error)
    else:  # pragma: no cover
        raise AssertionError("Expected RuntimeError on empty truncated page")


def test_truncated_but_usable_page_is_kept() -> None:
    """A length-capped page that still returned text is not discarded."""
    capture: dict = {}
    response = _fake_completion("# Real content", finish_reason="length")
    provider = _provider_with_stub(response, capture)
    result = provider.process_vision(
        image_base64=_sample_image_base64(), prompt="Convert"
    )
    assert result.text == "# Real content"


def test_provider_strips_wrapping_fence() -> None:
    """process_vision returns unwrapped markdown, not a fenced code block."""
    capture: dict = {}
    response = _fake_completion("```markdown\n# Page\n```")
    provider = _provider_with_stub(response, capture)
    result = provider.process_vision(
        image_base64=_sample_image_base64(), prompt="Convert to markdown"
    )
    assert result.text == "# Page"


def test_strip_grounding_tokens() -> None:
    """DeepSeek-OCR grounding markup is reduced to clean text.

    <|ref|> spans are unwrapped, <|det|> coordinate boxes are dropped, and
    stray grounding markers are removed.
    """
    raw = (
        "# Title\n"
        "<|ref|>Body paragraph.<|/ref|><|det|>[[10,20,30,40]]<|/det|>\n"
        "<|grounding|>Trailing line."
    )
    assert strip_grounding_tokens(raw) == (
        "# Title\nBody paragraph.\nTrailing line."
    )
    # Plain markdown with no grounding markup is unchanged (bar surrounding ws).
    assert strip_grounding_tokens("# Just markdown") == "# Just markdown"


def test_unlimited_provider_uses_recipe_prompt_and_strips_grounding() -> None:
    """Unlimited-OCR ignores the caller prompt and cleans grounded output."""
    capture: dict = {}
    grounded = (
        "<|ref|># Heading<|/ref|><|det|>[[1,2,3,4]]<|/det|>\n\nParagraph."
    )
    response = _fake_completion(
        grounded, prompt_tokens=900, completion_tokens=120
    )
    provider = _provider_with_stub(
        response, capture, provider=UnlimitedOcrProvider()
    )

    result = provider.process_vision(
        image_base64=_sample_image_base64(),
        prompt="Write a Markdown version of this page",  # ignored
        prior_text="prior digital text",  # ignored
        max_tokens=1024,
    )

    # Output is the recipe transcription with grounding markup removed.
    assert result.text == "# Heading\n\nParagraph."
    assert result.usage.cost_usd == 0.0

    # The fixed recipe prompt is sent verbatim (begins with <image>), not the
    # caller's prompt, and prior text is not appended.
    text_part = next(
        p for p in capture["messages"][0]["content"] if p["type"] == "text"
    )
    assert text_part["text"] == DEFAULT_UNLIMITED_OCR_PROMPT
    assert text_part["text"].startswith("<image>")
    assert "prior digital text" not in text_part["text"]

    # No chat template exists, so no enable_thinking kwarg is sent; special
    # tokens are kept so the grounding markup survives to be stripped.
    extra = capture["extra_body"]
    assert extra == {"skip_special_tokens": False}
    assert capture["model"] == DEFAULT_UNLIMITED_MODEL
    assert capture["temperature"] == 0.0


def test_unlimited_budget_error_names_its_flag() -> None:
    """An empty length-capped Unlimited page names --unlimited-max-tokens."""
    capture: dict = {}
    response = _fake_completion("", finish_reason="length")
    provider = _provider_with_stub(
        response, capture, provider=UnlimitedOcrProvider()
    )
    try:
        provider.process_vision(
            image_base64=_sample_image_base64(), prompt="ignored"
        )
    except RuntimeError as error:
        assert "--unlimited-max-tokens" in str(error)
    else:  # pragma: no cover
        raise AssertionError("Expected RuntimeError on empty truncated page")


def run_all_tests() -> bool:
    """Run vLLM OCR tests directly (used by run_tests.py-style runners)."""
    test_models_url_derivation()
    test_is_vllm_running_true_on_200()
    test_list_served_models()
    test_ensure_server_raises_when_unreachable()
    test_ensure_server_raises_on_wrong_model()
    test_provider_calls_vllm_with_required_options()
    test_num_predict_floor_respects_larger_max_tokens()
    test_provider_raises_on_transport_error()
    test_provider_raises_on_empty_length_truncation()
    test_truncated_but_usable_page_is_kept()
    test_provider_strips_wrapping_fence()
    test_strip_grounding_tokens()
    test_unlimited_provider_uses_recipe_prompt_and_strips_grounding()
    test_unlimited_budget_error_names_its_flag()
    print("  ✓ vLLM OCR provider tests passed")
    return True


if __name__ == "__main__":
    run_all_tests()
