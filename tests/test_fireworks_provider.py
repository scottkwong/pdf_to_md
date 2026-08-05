"""
Tests for the Fireworks AI provider integration.

Network is mocked, so these run without a FIREWORKS_API_KEY or any API call.
They cover provider construction, the vision request shape, availability
detection, model resolution/routing, and the models.json Fireworks entries.
"""
import os
import sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import llm_providers  # noqa: E402
from llm_providers import (  # noqa: E402
    FireworksProvider,
    get_available_providers,
    load_models_config,
    resolve_model,
)
from models_config import ModelKey, Provider  # noqa: E402


def _fake_openai_response(
    text: str,
    prompt_tokens: int,
    completion_tokens: int,
    finish_reason: str = "stop",
    reasoning_content=None,
):
    """Build a stub mimicking the OpenAI chat.completions response shape."""
    message = mock.Mock()
    message.content = text
    message.reasoning_content = reasoning_content
    choice = mock.Mock()
    choice.message = message
    choice.finish_reason = finish_reason
    usage = mock.Mock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    response = mock.Mock()
    response.choices = [choice]
    response.usage = usage
    return response


def test_provider_requires_key() -> None:
    """Constructing without a key (and none in env) raises ValueError."""
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("FIREWORKS_API_KEY", None)
        try:
            FireworksProvider()
        except ValueError as error:
            assert "FIREWORKS_API_KEY" in str(error)
        else:  # pragma: no cover
            raise AssertionError("Expected ValueError when key is absent")


def test_provider_uses_fireworks_base_url_and_vision_shape() -> None:
    """process_vision posts a data-URL image to the Fireworks base URL."""
    with mock.patch.object(llm_providers, "OpenAI") as fake_openai_cls:
        fake_client = fake_openai_cls.return_value
        fake_client.chat.completions.create.return_value = _fake_openai_response(
            "# Page\n\nhello", prompt_tokens=1200, completion_tokens=300
        )

        provider = FireworksProvider(api_key="test-key")

        # Client built against the Fireworks OpenAI-compatible endpoint.
        _, kwargs = fake_openai_cls.call_args
        assert kwargs["base_url"] == "https://api.fireworks.ai/inference/v1"
        assert kwargs["api_key"] == "test-key"

        result = provider.process_vision(
            image_base64="QUJD",
            prompt="Convert to markdown",
            prior_text="prior text",
            model="accounts/fireworks/models/qwen3p7-plus",
            max_tokens=4096,
        )

        assert result.text == "# Page\n\nhello"
        assert result.usage.input_tokens == 1200
        assert result.usage.output_tokens == 300

        call_kwargs = fake_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"].endswith("qwen3p7-plus")
        # Reasoning headroom: the request asks for more than the caller's 4096.
        assert call_kwargs["max_tokens"] == FireworksProvider.REASONING_TOKEN_HEADROOM
        content = call_kwargs["messages"][0]["content"]
        text_part = next(p for p in content if p["type"] == "text")
        image_part = next(p for p in content if p["type"] == "image_url")
        assert "prior text" in text_part["text"]
        assert image_part["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_provider_rejects_empty_model() -> None:
    """A blank model id is rejected (Fireworks has no implicit default)."""
    with mock.patch.object(llm_providers, "OpenAI"):
        provider = FireworksProvider(api_key="test-key")
        try:
            provider.process_vision(image_base64="QUJD", prompt="x", model="")
        except ValueError as error:
            assert "Fireworks" in str(error)
        else:  # pragma: no cover
            raise AssertionError("Expected ValueError for empty model")


def test_provider_raises_on_reasoning_truncation() -> None:
    """finish_reason=length with no reasoning_content means the thinking was
    truncated and `content` holds thinking text, not the answer — raise."""
    with mock.patch.object(llm_providers, "OpenAI") as fake_openai_cls:
        fake_client = fake_openai_cls.return_value
        fake_client.chat.completions.create.return_value = _fake_openai_response(
            "Thinking Process:\n\n1. Analyze...",
            prompt_tokens=2000,
            completion_tokens=16384,
            finish_reason="length",
            reasoning_content=None,
        )
        provider = FireworksProvider(api_key="test-key")
        try:
            provider.process_vision(
                image_base64="QUJD", prompt="x", model="m", max_tokens=4096
            )
        except ValueError as error:
            assert "truncated during reasoning" in str(error)
        else:  # pragma: no cover
            raise AssertionError("Expected ValueError on reasoning truncation")


def test_available_providers_includes_fireworks() -> None:
    """get_available_providers reports fireworks based on FIREWORKS_API_KEY."""
    with mock.patch.dict(os.environ, {"FIREWORKS_API_KEY": "k"}, clear=False):
        assert get_available_providers()[Provider.FIREWORKS] is True
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("FIREWORKS_API_KEY", None)
        assert get_available_providers()[Provider.FIREWORKS] is False


def test_resolve_model_routes_fireworks_to_fireworks_provider() -> None:
    """A fireworks model resolves to FireworksProvider with its direct id."""
    with mock.patch.object(llm_providers, "OpenAI"), mock.patch.dict(
        os.environ, {"FIREWORKS_API_KEY": "test-key"}, clear=False
    ), mock.patch.object(
        llm_providers,
        "get_available_providers",
        return_value={
            Provider.OPENROUTER.value: False,
            Provider.OPENAI.value: False,
            Provider.ANTHROPIC.value: False,
            Provider.GOOGLE.value: False,
            Provider.FIREWORKS.value: True,
        },
    ):
        model_id, provider = resolve_model(
            ModelKey.QWEN_3_7_PLUS, prefer_openrouter=False
        )
    assert isinstance(provider, FireworksProvider)
    assert model_id == "accounts/fireworks/models/qwen3p7-plus"


def test_models_json_has_fireworks_default() -> None:
    """models.json defines exactly one fireworks_default vision model."""
    config = load_models_config()
    fireworks = {
        name: cfg
        for name, cfg in config.items()
        if cfg.get("provider") == Provider.FIREWORKS
    }
    assert ModelKey.QWEN_3_7_PLUS in fireworks
    defaults = [n for n, c in fireworks.items() if c.get("fireworks_default")]
    assert defaults == [ModelKey.QWEN_3_7_PLUS]
    for cfg in fireworks.values():
        assert cfg.get("supports_vision") is True
        assert cfg["direct_id"].startswith("accounts/fireworks/models/")


def test_default_model_honors_env_override() -> None:
    """PDF_TO_MD_MODEL overrides the built-in default; helper finds fireworks default."""
    import pdf_to_md

    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("PDF_TO_MD_MODEL", None)
        assert pdf_to_md._default_model() == ModelKey.GPT_5_5
    with mock.patch.dict(
        os.environ, {"PDF_TO_MD_MODEL": ModelKey.QWEN_3_7_PLUS.value}, clear=False
    ):
        assert pdf_to_md._default_model() == ModelKey.QWEN_3_7_PLUS
    assert pdf_to_md._default_fireworks_model() == ModelKey.QWEN_3_7_PLUS


def run_all_tests() -> bool:
    """Run Fireworks provider tests directly (used by run_tests.py)."""
    test_provider_requires_key()
    test_provider_uses_fireworks_base_url_and_vision_shape()
    test_provider_rejects_empty_model()
    test_provider_raises_on_reasoning_truncation()
    test_available_providers_includes_fireworks()
    test_resolve_model_routes_fireworks_to_fireworks_provider()
    test_models_json_has_fireworks_default()
    test_default_model_honors_env_override()
    print("  ✓ Fireworks provider tests passed")
    return True


if __name__ == "__main__":
    run_all_tests()
