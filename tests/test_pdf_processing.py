"""
PDF vision processing tests for each LLM provider.

Tests vision processing with a simple test PDF to verify each provider
works correctly with images. Uses .env file for API keys.
"""
import base64
import io
import os
import sys

from dotenv import load_dotenv
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_providers import (
    AnthropicProvider,
    GoogleProvider,
    OpenAIProvider,
    OpenRouterProvider,
    VisionResult,
    create_provider,
    get_available_providers,
)

# Load environment variables
load_dotenv()


def create_test_image_base64() -> str:
    """
    Create a simple test image with text and return as base64.

    Returns:
        Base64-encoded JPEG image string.
    """
    # Create a simple image with text
    img = Image.new("RGB", (400, 200), color="white")
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(img)
    # Try to use a default font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except Exception:
        font = ImageFont.load_default()

    draw.text((50, 50), "Test Document", fill="black", font=font)
    draw.text((50, 100), "This is a test page", fill="black", font=font)
    draw.text((50, 130), "Item 1 | Item 2 | Item 3", fill="black", font=font)
    draw.text((50, 160), "Value A | Value B | Value C", fill="black", font=font)

    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    byte_data = buffer.getvalue()
    return base64.b64encode(byte_data).decode("utf-8")


def test_openrouter_vision():
    """Test OpenRouter provider with vision processing."""
    try:
        provider, model_id = create_provider("gpt-5.2", prefer_openrouter=True)
        image_base64 = create_test_image_base64()
        prompt = (
            "Describe what you see in this image. Focus on any text or "
            "structured data."
        )

        result = provider.process_vision(
            image_base64=image_base64,
            prompt=prompt,
            model=model_id,
            max_tokens=200,
        )

        assert isinstance(result, VisionResult)
        assert result.text is not None and len(result.text) > 0
        assert "test" in result.text.lower() or "document" in result.text.lower()
        assert result.usage.input_tokens > 0
        print("✓ OpenRouter vision test passed")
        return True
    except Exception as e:
        print(f"✗ OpenRouter vision test failed: {e}")
        return False


def test_openai_vision():
    """Test OpenAI provider with vision processing."""
    try:
        provider, model_id = create_provider("gpt-5.2", prefer_openrouter=False)
        image_base64 = create_test_image_base64()
        prompt = (
            "Describe what you see in this image. Focus on any text or "
            "structured data."
        )

        result = provider.process_vision(
            image_base64=image_base64,
            prompt=prompt,
            model=model_id,
            max_tokens=200,
        )

        assert isinstance(result, VisionResult)
        assert result.text is not None and len(result.text) > 0
        assert "test" in result.text.lower() or "document" in result.text.lower()
        assert result.usage.input_tokens > 0
        print("✓ OpenAI vision test passed")
        return True
    except Exception as e:
        print(f"✗ OpenAI vision test failed: {e}")
        return False


def test_anthropic_vision():
    """Test Anthropic provider with vision processing."""
    try:
        provider, model_id = create_provider(
            "claude-sonnet-4.5", prefer_openrouter=False
        )
        image_base64 = create_test_image_base64()
        prompt = (
            "Describe what you see in this image. Focus on any text or "
            "structured data."
        )

        result = provider.process_vision(
            image_base64=image_base64,
            prompt=prompt,
            model=model_id,
            max_tokens=200,
        )

        assert isinstance(result, VisionResult)
        assert result.text is not None and len(result.text) > 0
        assert "test" in result.text.lower() or "document" in result.text.lower()
        assert result.usage.input_tokens > 0
        print("✓ Anthropic vision test passed")
        return True
    except Exception as e:
        print(f"✗ Anthropic vision test failed: {e}")
        return False


def test_google_vision():
    """Test Google provider with vision processing."""
    try:
        # Use gemini-3-flash which should fallback to available models
        # The provider will handle fallback if gemini-3-pro/flash don't exist
        provider, model_id = create_provider(
            "gemini-3-flash", prefer_openrouter=False
        )
        image_base64 = create_test_image_base64()
        prompt = (
            "Describe what you see in this image. Focus on any text or "
            "structured data."
        )

        result = provider.process_vision(
            image_base64=image_base64,
            prompt=prompt,
            model=model_id,
            max_tokens=200,
        )

        assert isinstance(result, VisionResult)
        assert result.text is not None and len(result.text) > 0
        assert "test" in result.text.lower() or "document" in result.text.lower()
        assert result.usage.input_tokens > 0
        print("✓ Google vision test passed")
        return True
    except Exception as e:
        print(f"✗ Google vision test failed: {e}")
        return False


def run_all_tests():
    """Run all vision processing tests for available providers."""
    print("Running PDF vision processing tests...\n")
    available = get_available_providers()

    results = {}

    # Test OpenRouter if available
    if available["openrouter"]:
        results["OpenRouter"] = test_openrouter_vision()
    else:
        print("- OpenRouter: Skipped (no API key)")

    # Test OpenAI if available
    if available["openai"]:
        results["OpenAI"] = test_openai_vision()
    else:
        print("- OpenAI: Skipped (no API key)")

    # Test Anthropic if available
    if available["anthropic"]:
        results["Anthropic"] = test_anthropic_vision()
    else:
        print("- Anthropic: Skipped (no API key)")

    # Test Google if available
    if available["google"]:
        results["Google"] = test_google_vision()
    else:
        print("- Google: Skipped (no API key)")

    print("\n" + "=" * 50)
    print("Vision Processing Test Summary:")
    print("=" * 50)
    for provider, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{provider}: {status}")

    return all(results.values()) if results else False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

