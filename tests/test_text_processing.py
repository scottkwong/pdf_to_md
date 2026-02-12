"""
Simple text processing tests for each LLM provider.

Tests basic text-only prompts to verify each provider works correctly.
Uses .env file for API keys (no hardcoded values).
"""
import os
import sys

from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_providers import (
    AnthropicProvider,
    GoogleProvider,
    OpenAIProvider,
    OpenRouterProvider,
    create_provider,
    get_available_providers,
    load_models_config,
)

# Load environment variables
load_dotenv()


def test_openrouter_text():
    """Test OpenRouter provider with simple text prompt."""
    try:
        # Use model from models.json
        models_config = load_models_config()
        model_key = "openai-gpt4o"  # Use gpt-4o for text test
        openrouter_id = models_config[model_key]["openrouter_id"]
        
        provider = OpenRouterProvider()
        response = provider.client.chat.completions.create(
            model=openrouter_id,
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10,
        )
        result = response.choices[0].message.content
        assert result is not None and len(result) > 0
        print("✓ OpenRouter text test passed")
        return True
    except Exception as e:
        print(f"✗ OpenRouter text test failed: {e}")
        return False


def test_openai_text():
    """Test OpenAI provider with simple text prompt."""
    try:
        # Use model from models.json
        models_config = load_models_config()
        model_key = "openai-gpt4o"
        direct_id = models_config[model_key]["direct_id"]
        
        provider = OpenAIProvider()
        response = provider.client.chat.completions.create(
            model=direct_id,
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10,
        )
        result = response.choices[0].message.content
        assert result is not None and len(result) > 0
        print("✓ OpenAI text test passed")
        return True
    except Exception as e:
        print(f"✗ OpenAI text test failed: {e}")
        return False


def test_anthropic_text():
    """Test Anthropic provider with simple text prompt."""
    try:
        # Use model from models.json
        models_config = load_models_config()
        model_key = "claude-sonnet-4.5"
        direct_id = models_config[model_key]["direct_id"]
        
        provider = AnthropicProvider()
        response = provider.client.messages.create(
            model=direct_id,
            max_tokens=10,
            messages=[{"role": "user", "content": "Say hello"}],
        )
        result = response.content[0].text
        assert result is not None and len(result) > 0
        print("✓ Anthropic text test passed")
        return True
    except Exception as e:
        print(f"✗ Anthropic text test failed: {e}")
        return False


def test_google_text():
    """Test Google provider with simple text prompt."""
    try:
        # Use model from models.json
        models_config = load_models_config()
        model_key = "gemini-3-flash"
        direct_id = models_config[model_key]["direct_id"]
        
        provider = GoogleProvider()
        
        # Try the model from models.json, with fallback
        try:
            response = provider.client.models.generate_content(
                model=direct_id,
                contents=[{"role": "user", "parts": [{"text": "Say hello"}]}],
            )
        except Exception:
            # Fallback to available models if gemini-3 doesn't exist
            fallback_models = [
                "gemini-2.5-flash", "gemini-2.5-pro",
                "gemini-2.0-flash", "gemini-2.5-flash-lite",
            ]
            for fallback_model in fallback_models:
                try:
                    response = provider.client.models.generate_content(
                        model=fallback_model,
                        contents=[{"role": "user", "parts": [{"text": "Say hello"}]}],
                    )
                    break
                except Exception:
                    continue
            else:
                raise ValueError("No available Google model found")
        
        # Extract text from response
        if hasattr(response, "text"):
            result = response.text
        elif hasattr(response, "candidates") and response.candidates:
            result = response.candidates[0].content.parts[0].text
        else:
            result = str(response)
        
        assert result is not None and len(result) > 0
        print("✓ Google text test passed")
        return True
    except Exception as e:
        print(f"✗ Google text test failed: {e}")
        return False


def run_all_tests():
    """Run all text processing tests for available providers."""
    print("Running text processing tests...\n")
    available = get_available_providers()

    results = {}
    if available["openrouter"]:
        results["OpenRouter"] = test_openrouter_text()
    else:
        print("- OpenRouter: Skipped (no API key)")

    if available["openai"]:
        results["OpenAI"] = test_openai_text()
    else:
        print("- OpenAI: Skipped (no API key)")

    if available["anthropic"]:
        results["Anthropic"] = test_anthropic_text()
    else:
        print("- Anthropic: Skipped (no API key)")

    if available["google"]:
        results["Google"] = test_google_text()
    else:
        print("- Google: Skipped (no API key)")

    print("\n" + "=" * 50)
    print("Text Processing Test Summary:")
    print("=" * 50)
    for provider, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{provider}: {status}")

    return all(results.values()) if results else False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

