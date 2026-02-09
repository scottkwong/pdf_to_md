#!/usr/bin/env python3
"""
Test runner for PDF to Markdown converter with LLM providers.

Loads .env file and runs all test suites. Reports which providers are
available and provides a summary of results.
"""
import os
import sys

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add tests directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_providers import get_available_providers
from tests.test_installation_setup import run_all_tests as run_setup_tests
from tests.test_text_extractor_backends import run_all_tests as run_parser_tests
from tests.test_pdf_processing import run_all_tests as run_vision_tests
from tests.test_text_processing import run_all_tests as run_text_tests
from tests.test_llamaparse import run_all_tests as run_llamaparse_tests


def main():
    """Run all test suites and provide summary."""
    print("=" * 60)
    print("PDF to Markdown - LLM Provider Test Suite")
    print("=" * 60)
    print()

    # Run setup and direct parser tests first (no API keys required)
    print("\n" + "=" * 60)
    print("INSTALLATION/SETUP TESTS")
    print("=" * 60)
    setup_success = run_setup_tests()

    print("\n" + "=" * 60)
    print("TEXT PARSER BACKEND TESTS")
    print("=" * 60)
    parser_success = run_parser_tests()

    # Check available providers (without validation for speed)
    available = get_available_providers(validate_keys=False)
    print("Available API Keys:")
    print("-" * 60)
    for provider, is_available in available.items():
        status = "✓ Available" if is_available else "✗ Not available"
        print(f"  {provider.title()}: {status}")
    
    # Validate OpenRouter key if available
    if available["openrouter"]:
        print("\nValidating OpenRouter API key...")
        from llm_providers import validate_openrouter_key
        api_key = os.getenv("OPENROUTER_API_KEY")
        is_valid = validate_openrouter_key(api_key)
        if is_valid:
            print("  ✓ OpenRouter API key validated (200 OK)")
        else:
            print("  ✗ OpenRouter API key validation failed")
            available["openrouter"] = False
    print()

    if not any(available.values()):
        print("ERROR: No API keys found in environment.")
        print("Please set at least one API key in your .env file.")
        sys.exit(1)

    # Run text tests
    print("\n" + "=" * 60)
    print("TEXT PROCESSING TESTS")
    print("=" * 60)
    text_success = run_text_tests()

    # Run vision tests
    print("\n" + "=" * 60)
    print("VISION PROCESSING TESTS")
    print("=" * 60)
    vision_success = run_vision_tests()

    # Run LlamaParse tests
    print("\n" + "=" * 60)
    print("LLAMAPARSE EXTRACTION TESTS")
    print("=" * 60)
    llamaparse_success = run_llamaparse_tests()

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Installation/Setup: {'PASSED' if setup_success else 'FAILED'}")
    print(f"Text Parser Backends: {'PASSED' if parser_success else 'FAILED'}")
    print(f"Text Processing: {'PASSED' if text_success else 'FAILED'}")
    print(f"Vision Processing: {'PASSED' if vision_success else 'FAILED'}")
    print(f"LlamaParse Extraction: {'PASSED' if llamaparse_success else 'FAILED'}")
    print()

    if (
        setup_success
        and parser_success
        and text_success
        and vision_success
        and llamaparse_success
    ):
        print("✓ All tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()

