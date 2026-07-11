"""
Tests for the cross-provider model benchmark.

These validate spec construction and skip/precheck logic without invoking any
real provider APIs or Ollama server.
"""
import os
import sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_models import (  # noqa: E402
    BenchmarkModelSpec,
    _precheck_api_spec,
    _precheck_local_spec,
    _resolve_api_routing,
    default_benchmark_specs,
    parse_benchmark_specs,
)


def test_default_specs_cover_three_providers() -> None:
    """Default benchmark compares OpenAI, Anthropic, and Local top models."""
    specs = default_benchmark_specs(local_model="llava")
    kinds = [s.kind for s in specs]
    models = [s.model for s in specs]
    assert kinds == ["api", "api", "local"]
    assert "gpt-5.5" in models
    assert "claude-opus-4.6" in models
    assert "llava" in models


def test_parse_specs_handles_local_variants() -> None:
    """parse_benchmark_specs recognizes local, local:<model>, and api keys."""
    specs = parse_benchmark_specs(
        "gpt-5.5, local, local:granite3.2-vision",
        local_model="llama3.2-vision:11b",
    )
    assert specs[0].kind == "api" and specs[0].model == "gpt-5.5"
    assert specs[1].kind == "local" and specs[1].model == "llama3.2-vision:11b"
    assert specs[2].kind == "local" and specs[2].model == "granite3.2-vision"


def test_parse_specs_rejects_empty() -> None:
    """An empty override string raises ValueError."""
    try:
        parse_benchmark_specs("  , ,")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError for empty benchmark models")


def test_api_precheck_skips_unknown_model() -> None:
    """A model missing from models.json is skipped with a clear reason."""
    spec = BenchmarkModelSpec(label="x", kind="api", model="not-a-real-model")
    reason = _precheck_api_spec(spec)
    assert reason is not None
    assert "models.json" in reason


def test_api_precheck_skips_when_no_key() -> None:
    """A known model with no available key/OpenRouter is skipped."""
    spec = BenchmarkModelSpec(label="gpt", kind="api", model="gpt-5.5")
    with mock.patch(
        "llm_providers.get_available_providers",
        return_value={
            "openrouter": False,
            "openai": False,
            "anthropic": False,
            "google": False,
        },
    ):
        reason = _precheck_api_spec(spec)
    assert reason is not None
    assert "OPENAI_API_KEY" in reason


def test_api_routing_uses_openrouter_when_only_option() -> None:
    """With only an OpenRouter key, a cloud model runs routed via OpenRouter."""
    spec = BenchmarkModelSpec(
        label="gpt", kind="api", model="gpt-5.5", prefer_openrouter=False
    )
    with mock.patch(
        "llm_providers.get_available_providers",
        return_value={
            "openrouter": True,
            "openai": False,
            "anthropic": False,
            "google": False,
        },
    ):
        reason, prefer_openrouter = _resolve_api_routing(spec)
    assert reason is None
    assert prefer_openrouter is True


def test_api_routing_prefers_direct_when_key_present() -> None:
    """A direct provider key routes directly even if OpenRouter is present."""
    spec = BenchmarkModelSpec(
        label="gpt", kind="api", model="gpt-5.5", prefer_openrouter=False
    )
    with mock.patch(
        "llm_providers.get_available_providers",
        return_value={
            "openrouter": True,
            "openai": True,
            "anthropic": False,
            "google": False,
        },
    ):
        reason, prefer_openrouter = _resolve_api_routing(spec)
    assert reason is None
    assert prefer_openrouter is False


def test_local_precheck_skips_when_ollama_unavailable() -> None:
    """Local spec is skipped when ollama-ocr or the server is unavailable."""
    spec = BenchmarkModelSpec(label="local", kind="local", model="llava")
    with mock.patch(
        "benchmark_models.is_ollama_ocr_available", return_value=False
    ):
        reason = _precheck_local_spec(spec)
    assert reason is not None
    assert "ollama-ocr" in reason

    with mock.patch(
        "benchmark_models.is_ollama_ocr_available", return_value=True
    ), mock.patch(
        "benchmark_models.is_ollama_running", return_value=False
    ):
        reason = _precheck_local_spec(spec)
    assert reason is not None
    assert "Ollama server" in reason


def run_all_tests() -> bool:
    """Run benchmark spec tests directly."""
    test_default_specs_cover_three_providers()
    test_parse_specs_handles_local_variants()
    test_parse_specs_rejects_empty()
    test_api_precheck_skips_unknown_model()
    test_api_precheck_skips_when_no_key()
    test_api_routing_uses_openrouter_when_only_option()
    test_api_routing_prefers_direct_when_key_present()
    test_local_precheck_skips_when_ollama_unavailable()
    print("  ✓ model benchmark tests passed")
    return True


if __name__ == "__main__":
    run_all_tests()
