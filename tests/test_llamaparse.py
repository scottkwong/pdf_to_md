"""
LlamaParse extraction tests.

Tests LlamaParse integration if LLAMA_CLOUD_API_KEY is available.
Uses .env file for API keys.
"""
import os
import sys
import tempfile

from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()


def is_llamaparse_available() -> bool:
    """Check if LlamaParse API key is configured."""
    return bool(os.getenv("LLAMA_CLOUD_API_KEY"))


def test_llamaparse_initialization():
    """Test LlamaParseExtractor initialization with valid tier."""
    if not is_llamaparse_available():
        print("- LlamaParse initialization: Skipped (no API key)")
        return True

    try:
        from llamaparse_extractor import LlamaParseExtractor

        extractor = LlamaParseExtractor(tier="fast")
        assert extractor.tier == "fast"
        assert extractor.name == "LlamaParse (fast)"
        assert extractor.language == "en"
        print("✓ LlamaParse initialization test passed")
        return True
    except Exception as e:
        print(f"✗ LlamaParse initialization test failed: {e}")
        return False


def test_llamaparse_invalid_tier():
    """Test that invalid tier raises error."""
    try:
        from llamaparse_extractor import LlamaParseExtractor

        try:
            # This should raise ValueError even without API key
            # because tier validation happens before API key check
            LlamaParseExtractor(
                api_key="fake_key_for_test",  # Provide fake key to bypass env check
                tier="invalid_tier",
            )
            print("✗ LlamaParse invalid tier test failed (no exception raised)")
            return False
        except ValueError as e:
            assert "Invalid tier" in str(e)
            print("✓ LlamaParse invalid tier test passed")
            return True
    except ImportError:
        print("- LlamaParse import test: Skipped (package not installed)")
        return True
    except Exception as e:
        print(f"✗ LlamaParse invalid tier test failed: {e}")
        return False


def test_llamaparse_missing_api_key():
    """Test that missing API key raises appropriate error."""
    try:
        from llamaparse_extractor import LlamaParseExtractor

        # Temporarily unset the API key
        original_key = os.environ.pop("LLAMA_CLOUD_API_KEY", None)
        try:
            LlamaParseExtractor(tier="fast")
            print("✗ LlamaParse missing API key test failed (no exception)")
            return False
        except ValueError as e:
            assert "LLAMA_CLOUD_API_KEY" in str(e)
            print("✓ LlamaParse missing API key test passed")
            return True
        finally:
            # Restore the key if it existed
            if original_key:
                os.environ["LLAMA_CLOUD_API_KEY"] = original_key
    except ImportError:
        print("- LlamaParse import test: Skipped (package not installed)")
        return True
    except Exception as e:
        print(f"✗ LlamaParse missing API key test failed: {e}")
        return False


def test_llamaparse_all_tiers():
    """Test that all tiers can be initialized."""
    if not is_llamaparse_available():
        print("- LlamaParse all tiers: Skipped (no API key)")
        return True

    try:
        from llamaparse_extractor import LlamaParseExtractor

        tiers = ["fast", "cost_effective", "agentic", "agentic_plus"]
        for tier in tiers:
            extractor = LlamaParseExtractor(tier=tier)
            assert extractor.tier == tier
            assert tier in extractor.name

        print("✓ LlamaParse all tiers test passed")
        return True
    except Exception as e:
        print(f"✗ LlamaParse all tiers test failed: {e}")
        return False


def test_llamaparse_extraction():
    """Test actual PDF extraction with LlamaParse (optional - uses API credits)."""
    if not is_llamaparse_available():
        print("- LlamaParse extraction: Skipped (no API key)")
        return True

    try:
        from llamaparse_extractor import LlamaParseExtractor
        from tests.create_test_pdf import create_test_pdf

        # Create test PDF
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = os.path.join(tmpdir, "test.pdf")
            create_test_pdf(pdf_path)

            extractor = LlamaParseExtractor(tier="fast")
            result = extractor.extract(pdf_path, tmpdir, verbose=False)

            assert result.markdown is not None
            assert len(result.markdown) > 0
            assert result.page_count >= 1
            assert result.metadata["extractor"] == "llamaparse"
            assert result.metadata["tier"] == "fast"

            print("✓ LlamaParse extraction test passed")
            return True
    except ImportError as e:
        if "llama_parse" in str(e):
            print("- LlamaParse extraction: Skipped (llama-parse not installed)")
            return True
        raise
    except Exception as e:
        # Rate limits or other API issues shouldn't fail the test suite
        if "rate limit" in str(e).lower():
            print("- LlamaParse extraction: Skipped (rate limit)")
            return True
        print(f"✗ LlamaParse extraction test failed: {e}")
        return False


def run_all_tests():
    """Run all LlamaParse tests."""
    print("Running LlamaParse tests...\n")

    results = {}

    # Unit tests (no API calls)
    results["Initialization"] = test_llamaparse_initialization()
    results["Invalid Tier"] = test_llamaparse_invalid_tier()
    results["Missing API Key"] = test_llamaparse_missing_api_key()
    results["All Tiers"] = test_llamaparse_all_tiers()

    # Integration test (uses API - optional)
    # Uncomment to run extraction test (uses API credits)
    # results["Extraction"] = test_llamaparse_extraction()

    print("\n" + "=" * 50)
    print("LlamaParse Test Summary:")
    print("=" * 50)
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{test_name}: {status}")

    return all(results.values())


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
