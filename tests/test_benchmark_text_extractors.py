"""
Tests for direct text extractor benchmark behavior.

This suite validates default fixture-based benchmarking and optional user PDF
path overrides without invoking any LLM provider APIs.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_text_extractors import benchmark_text_extractors
from tests.create_test_pdf import create_benchmark_fixture_pdf


def test_benchmark_uses_default_generated_fixtures() -> None:
    """Ensure benchmark defaults to generated fixture PDFs."""
    summary = benchmark_text_extractors(runs=1, pdf_path=None)
    assert summary.pdf_paths
    for path in summary.pdf_paths:
        assert os.path.isfile(path)
    assert "pypdf2" in summary.stats
    assert summary.stats["pypdf2"].runs == 1


def test_benchmark_uses_user_supplied_pdf(tmp_path) -> None:
    """Ensure benchmark input can be overridden with custom PDF path."""
    custom_pdf = str(tmp_path / "custom_benchmark.pdf")
    create_benchmark_fixture_pdf(
        output_path=custom_pdf,
        num_pages=3,
        title_prefix="Custom Benchmark",
    )
    summary = benchmark_text_extractors(runs=2, pdf_path=custom_pdf)
    assert summary.pdf_paths == [custom_pdf]
    assert summary.stats["pypdf2"].runs == 2
