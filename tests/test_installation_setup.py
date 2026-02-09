"""
Installation and setup verification tests.

This suite verifies required dependencies, optional PyMuPDF availability, and
system-level poppler utilities needed by image conversion workflow.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from text_parsers import is_pymupdf_available


REQUIRED_IMPORTS = [
    "PyPDF2",
    "pdf2image",
    "PIL",
    "dotenv",
    "tenacity",
    "tqdm",
]


def test_required_python_dependencies_importable() -> None:
    """Verify required Python dependencies are importable."""
    for module_name in REQUIRED_IMPORTS:
        importlib.import_module(module_name)


def test_optional_pymupdf_dependency_behavior() -> None:
    """Verify optional PyMuPDF import behavior is handled safely."""
    available = is_pymupdf_available()
    if available:
        importlib.import_module("fitz")
    else:
        try:
            importlib.import_module("fitz")
            raise AssertionError(
                "Expected fitz import to fail when PyMuPDF is unavailable."
            )
        except ImportError:
            pass


def test_poppler_binary_available() -> None:
    """Verify poppler `pdftoppm` binary is available in PATH."""
    if shutil.which("pdftoppm") is None:
        raise AssertionError(
            "pdftoppm not found. Install poppler (brew install poppler / "
            "apt-get install poppler-utils)."
        )

    result = subprocess.run(
        ["pdftoppm", "-v"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError("pdftoppm is installed but failed to run.")


def run_all_tests() -> bool:
    """Run setup tests and return pass/fail status."""
    print("Running installation/setup verification tests...\n")
    try:
        test_required_python_dependencies_importable()
        test_optional_pymupdf_dependency_behavior()
        test_poppler_binary_available()
        print("\n" + "=" * 50)
        print("Installation/Setup Test Summary:")
        print("=" * 50)
        print("Required dependencies: PASSED")
        print("Optional PyMuPDF behavior: PASSED")
        print("Poppler binary check: PASSED")
        return True
    except Exception as exc:
        print(f"Installation/setup tests failed: {exc}")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
